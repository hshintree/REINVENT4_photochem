#!/usr/bin/env python
"""Headless DASA trial runner — TL + staged RL from the terminal.

Runs the DASA loop without Jupyter (no display() calls), reusing the enumerated
702-molecule corpus and the DASA scoring plugins. Use this to kick off a trial;
use dasa_complete.ipynb for the full interactive analysis (clustering, DFT, plots).

Usage (from the repo root, in the reinvent4 conda env):
    python notebooks/run_dasa_trial.py --quick        # fast smoke: TL + Stage 1
    python notebooks/run_dasa_trial.py                # TL + Stage 1 + Stage 2 (xTB, slow)
    python notebooks/run_dasa_trial.py --stage2       # include the slow xTB stage
    python notebooks/run_dasa_trial.py --device cuda:0

Stages:
  TL      — fine-tune reinvent.prior on the DASA corpus
  Stage 1 — DASAScaffold + AqueousSolubility + SA  (fast, cheap gate)
  Stage 2 — + XTBHomoLumo + DASASwitchability      (xTB, ~2-6 s/mol; opt-in)
"""
import os, sys, glob, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dasa_chem as dc  # noqa: E402
import pandas as pd  # noqa: E402

# Resolve how to invoke reinvent, as a command prefix (list). Prefer the console
# script from the current Python's env or PATH (local reinvent4 case); if it's not
# installed as a package (e.g. a clean Modal image where only the repo is on
# PYTHONPATH), fall back to `python -m reinvent` via reinvent/__main__.py.
_rv_bin = os.path.join(os.path.dirname(sys.executable), "reinvent")
if not os.path.isfile(_rv_bin):
    _rv_bin = shutil.which("reinvent")
REINVENT = [_rv_bin] if _rv_bin else [sys.executable, "-m", "reinvent"]


def sh(cmd, log, env, critical=True):
    print("▶", " ".join(cmd), "\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print(f"\n{'✓ done' if p.returncode == 0 else f'✗ exit {p.returncode}'}\n")
    if critical and p.returncode != 0:
        sys.exit(f"stage failed (exit {p.returncode}) — aborting so later stages "
                 "don't run on a broken state. Check the log above.")
    return p.returncode


def _guard_legacy_checkpoint(path: str) -> None:
    """REFUSE to resume from a checkpoint trained on the pre-2026-07-28 core.

    Until 2026-07-28 the corpus encoded a constitutional isomer of a DASA (hydroxyl
    on the carbon bonded to the acceptor rather than on C2). A generator trained on
    that corpus has learned the WRONG scaffold, so resuming from it silently
    reintroduces it through the back door -- every downstream gate would pass,
    because the gates were what we fixed, not the prior. This aborts rather than
    warns: a warning in a detached Modal run is a warning nobody reads.

    Detection is by sampling the checkpoint's own vocabulary-free output is not
    possible here, so we check the sibling summary CSV the stage wrote, which
    contains the molecules that checkpoint was producing.
    """
    import glob
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dasa_chem as _dc
    except Exception:
        print("WARNING: could not import dasa_chem to validate the checkpoint core.")
        return
    import csv
    d = os.path.dirname(os.path.abspath(path))
    csvs = sorted(glob.glob(os.path.join(d, "*_1.csv")))
    if not csvs:
        print(f"NOTE: no summary CSV beside {path}; cannot verify its core. "
              "If it predates 2026-07-28, do NOT resume from it.")
        return
    legacy = current = 0
    with open(csvs[0]) as fh:
        rows = list(csv.DictReader(fh))
    key = next((k for k in (rows[0] if rows else {}) if k.lower() in ("smiles", "smi")), None)
    for r in rows[:3000]:
        m = Chem.MolFromSmiles(r[key]) if key else None
        if m is None:
            continue
        legacy += _dc.is_legacy_core(m)
        current += _dc.is_dasa(m)
    if legacy > current:
        sys.exit(
            f"REFUSING to resume from {path}\n"
            f"  Its output ({os.path.basename(csvs[0])}) is {legacy} legacy-core vs "
            f"{current} corrected-core molecules.\n"
            f"  That generator was trained on the pre-2026-07-28 wrong DASA skeleton "
            f"(hydroxyl on the\n  carbon bonded to the acceptor instead of on C2). "
            f"Resuming would reintroduce it.\n"
            f"  Run a fresh TL + Stage 1 on the corrected corpus instead.")
    print(f"checkpoint core OK ({current} corrected / {legacy} legacy in "
          f"{os.path.basename(csvs[0])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true", help="short TL + Stage 1 only")
    ap.add_argument("--stage2", action="store_true",
                    help="run the slow xTB Stage 2 (DASATrap). OPT-IN and OFF by "
                         "default, including under --full: validated 2026-07-28 "
                         "against the corrected anchors, GFN2/ALPB-water "
                         "dG(closed-open) INVERTS the design objective -- it scores "
                         "the trapped 1st-gen alkyl 0.82 and the 2nd-gen aniline "
                         "escape architecture 0.21, i.e. it rewards exactly what we "
                         "are trying to avoid, and fights DASATrapEscape in the same "
                         "stage. dG in a continuum solvent cannot see the trap, which "
                         "is an electrostatic LOCK (reverse barrier) on the "
                         "zwitterion, not a free-energy sign. Use only for "
                         "experiments, never for a production campaign.")
    ap.add_argument("--stage3", action="store_true",
                    help="also run Stage 3 (ChemProp λ if trained, else structural fallback)")
    ap.add_argument("--full", action="store_true", help="TL + Stage 1 + 2 + 3")
    ap.add_argument("--stage1-checkpoint", default=None,
                    help="RESUME: skip TL + Stage 1, start Stage 2 from this .chkpt "
                         "(e.g. a prior run's trial_stage1/stage1.chkpt). Stage 1 gates "
                         "are unchanged, so reusing the checkpoint is valid and fast.")
    ap.add_argument("--stage2-steps", type=int, default=300,
                    help="RL steps for the slow xTB Stage 2 (lower = cheaper)")
    args = ap.parse_args()
    if args.full:
        # --full = TL + Stage 1 + Stage 3. Stage 2 (xTB DASATrap) is NOT included:
        # it inverts the objective (see --stage2 help). Pass --stage2 explicitly to
        # add it back for an experiment.
        args.stage3 = True
        args.quick = False
    # RESUME mode: reuse a prior Stage-1 checkpoint, run only the (improved) Stage 2/3.
    _resume = args.stage1_checkpoint
    if _resume:
        if not os.path.isfile(_resume):
            sys.exit(f"--stage1-checkpoint not found: {_resume}")
        _guard_legacy_checkpoint(_resume)
        args.stage2 = True
        args.quick = False

    out = os.path.join(ROOT, "outputs_dasa"); os.makedirs(out, exist_ok=True)
    plugin_dir = os.path.join(ROOT, "plugins")
    prior = os.path.join(ROOT, "reinvent.prior")
    data_dir = os.path.join(ROOT, "notebooks", "data")
    dataset_csv = os.path.join(data_dir, "dasa_dataset.csv")

    # PYTHONPATH: plugins dir (DASA components) + repo root. Repo root ensures the
    # repo's reinvent (whose scoring/importer.py tolerates optional components that
    # fail to import, e.g. comp_chemprop under this env's numpy/sklearn ABI break)
    # shadows any site-packages copy, so the scoring registry loads.
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(
               [plugin_dir, ROOT, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}

    # --- corpus: real extraction if present, else the AQUEOUS-focused library
    # (EWG-aniline / weak-acceptor / peripheral-solubiliser / tethered-amine
    # designs tuned for water-switchability -- the redesigned target region). ---
    if os.path.isfile(dataset_csv):
        corpus = dc.load_dasa_dataset(dataset_csv)["smiles_open"].tolist()
        print(f"corpus: {len(corpus)} DASAs from {dataset_csv}")
    else:
        corpus = [r["smiles_open"] for r in dc.enumerate_dasa_aqueous()]
        print(f"corpus: {len(corpus)} aqueous-focused DASAs (drop {dataset_csv} for real data)")

    # The reinvent.prior vocabulary has no stereo tokens (/ \) and no I/P/B/...,
    # so strip E/Z and DROP any molecule with an unsupported element before
    # feeding TL/inception (otherwise token validation crashes the run).
    from rdkit import Chem
    flat, dropped = [], 0
    for smi in corpus:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        if not dc.prior_supported(m):
            dropped += 1
            continue
        flat.append(Chem.MolToSmiles(m, isomericSmiles=False))
    if dropped:
        print(f"  dropped {dropped} corpus SMILES with prior-unsupported elements")
    tl_smi = os.path.join(out, "trial_corpus.smi")
    with open(tl_smi, "w") as f:
        f.write("\n".join(flat) + "\n")
    print(f"  wrote stereo-free corpus for the model: {len(flat)} SMILES")

    # decomposition-liability SMARTS (task-3) -> custom_alerts in the RL stages
    _decomp_smarts = ", ".join(f'"{s}"' for s in dc.DECOMPOSITION_SMARTS)

    tl_epochs = 15 if args.quick else 50
    s1_steps = 150 if args.quick else 500

    # --- Transfer learning ---
    tl_model = os.path.join(out, "TL_dasa_trial.model")
    tl_cfg = os.path.join(out, "trial_tl.toml")
    with open(tl_cfg, "w") as f:
        f.write(f'''run_type = "transfer_learning"
device = "{args.device}"
tb_logdir = "{out}/tb_tl_trial"
[parameters]
num_epochs = {tl_epochs}
save_every_n_epochs = 5
batch_size = 64
sample_batch_size = 1000
input_model_file = "{prior}"
output_model_file = "{tl_model}"
smiles_file = "{tl_smi}"
validation_smiles_file = "{tl_smi}"
standardize_smiles = true
randomize_smiles = true
''')
    if not _resume:                     # RESUME skips transfer learning
        sh(REINVENT + ["-d", args.device, "-l", f"{out}/trial_tl.log", tl_cfg], None, env)
    agent = tl_model if os.path.isfile(tl_model) else prior

    # --- Stage 1: structural + solubility gate ---
    s1_dir = os.path.join(out, "trial_stage1"); os.makedirs(s1_dir, exist_ok=True)
    s1_chkpt = _resume if _resume else os.path.join(s1_dir, "stage1.chkpt")
    s1_cfg = os.path.join(s1_dir, "stage1.toml")
    with open(s1_cfg, "w") as f:
        f.write(f'''run_type = "staged_learning"
device = "{args.device}"
tb_logdir = "{s1_dir}/tb"
[parameters]
prior_file = "{prior}"
agent_file = "{agent}"
summary_csv_prefix = "{s1_dir}/stage1"
batch_size = 128
use_checkpoint = false
[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
[[stage]]
max_score = 1.0
max_steps = {s1_steps}
chkpt_file = "{s1_chkpt}"
[stage.scoring]
type = "geometric_mean"
[[stage.scoring.component]]
[stage.scoring.component.DASAScaffold]
[[stage.scoring.component.DASAScaffold.endpoint]]
name = "DASA"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASAColor]
[[stage.scoring.component.DASAColor.endpoint]]
name = "Color"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASAIntegrity]
[[stage.scoring.component.DASAIntegrity.endpoint]]
name = "Integrity"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASATrapEscape]
[[stage.scoring.component.DASATrapEscape.endpoint]]
name = "TrapEscape"
weight = 0.6
[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "DecompAlerts"
weight = 1.0
params.smarts = [{_decomp_smarts}]
[[stage.scoring.component]]
[stage.scoring.component.AqueousSolubility]
[[stage.scoring.component.AqueousSolubility.endpoint]]
name = "Solubility"
weight = 0.5
params.logs_target = -2.0
params.logs_width = 1.5
params.logp_max = 1.0
params.logp_min = -2.5
[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name = "SA"
weight = 0.4
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low = 2.0
transform.k = 0.4
[diversity_filter]
type = "ScaffoldSimilarity"
# SIZING MATTERS AS MUCH AS THE AXIS. bucket_size 25 with minsimilarity 0.6 was a
# near-ban on the target class: ~22% of DASA scaffold PAIRS score >=0.6 atom-pair
# Tanimoto, so the whole family collapses into a handful of buckets, fills them in
# the first few hundred molecules, and every later DASA is zeroed. Measured effect:
# the DASA gate pass rate fell from 80.6% to 7.1% (Stage 1) and 98.6% to 2.1%
# (Stage 3) -- the generator was driven off the scaffold. minsimilarity 0.8 merges
# only near-duplicates, and bucket_size 400 is scaled to a 64k-molecule stage.
# IdenticalMurckoScaffold bucketed on EXACT Murcko scaffold, so the previous run
# reported healthy diversity while Butina found 275 of 300 shortlisted molecules in
# ONE cluster: swapping a heteroaryl ring changes the Murcko scaffold but not the
# chemistry. ScaffoldSimilarity buckets by atom-pair Tanimoto instead, which is the
# axis we actually mean by "diverse", so a family of look-alikes shares a bucket and
# gets penalised. This matters most once every band is satisfied: a satisfied
# constraint stops pushing, and diversity is then the only remaining objective.
minsimilarity = 0.8
bucket_size = 400
minscore = 0.4
[inception]
smiles_file = "{tl_smi}"
memory_size = 100
sample_size = 10
''')
    if not _resume:                     # RESUME skips Stage 1; reuse its checkpoint
        sh(REINVENT + ["-d", args.device, "-l", f"{s1_dir}/stage1.log", s1_cfg], None, env)
    else:
        print(f"RESUME: skipping TL + Stage 1; Stage 2 starts from {_resume}\n")

    # --- Stage 2 (opt-in, slow xTB) ---
    if args.stage2 and not args.quick:
        s2_dir = os.path.join(out, "trial_stage2"); os.makedirs(s2_dir, exist_ok=True)
        s2_agent = s1_chkpt if os.path.isfile(s1_chkpt) else agent
        s2_cfg = os.path.join(s2_dir, "stage2.toml")
        with open(s2_cfg, "w") as f:
            f.write(f'''run_type = "staged_learning"
device = "{args.device}"
tb_logdir = "{s2_dir}/tb"
[parameters]
prior_file = "{prior}"
agent_file = "{s2_agent}"
summary_csv_prefix = "{s2_dir}/stage2"
batch_size = 40
use_checkpoint = false
[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
[[stage]]
max_score = 1.0
max_steps = {args.stage2_steps}
chkpt_file = "{s2_dir}/stage2.chkpt"
[stage.scoring]
type = "geometric_mean"
[[stage.scoring.component]]
[stage.scoring.component.DASAScaffold]
[[stage.scoring.component.DASAScaffold.endpoint]]
name = "DASA"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASAColor]
[[stage.scoring.component.DASAColor.endpoint]]
name = "Color"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASAIntegrity]
[[stage.scoring.component.DASAIntegrity.endpoint]]
name = "Integrity"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASATrapEscape]
[[stage.scoring.component.DASATrapEscape.endpoint]]
name = "TrapEscape"
weight = 0.6
[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "DecompAlerts"
weight = 1.0
params.smarts = [{_decomp_smarts}]
[[stage.scoring.component]]
[stage.scoring.component.AqueousSolubility]
[[stage.scoring.component.AqueousSolubility.endpoint]]
name = "Solubility"
weight = 0.5
params.logs_target = -2.0
params.logs_width = 1.5
params.logp_max = 1.0
params.logp_min = -2.5
[[stage.scoring.component]]
[stage.scoring.component.DASATrap]
[[stage.scoring.component.DASATrap.endpoint]]
name = "AntiTrap"
weight = 0.6
params.dE_lo_kcal = [0.0]
params.dE_hi_kcal = [20.0]
params.dE_width_kcal = [3.0]
params.use_toluene = false
[diversity_filter]
type = "ScaffoldSimilarity"
# IdenticalMurckoScaffold bucketed on EXACT Murcko scaffold, so the previous run
# reported healthy diversity while Butina found 275 of 300 shortlisted molecules in
# ONE cluster: swapping a heteroaryl ring changes the Murcko scaffold but not the
# chemistry. ScaffoldSimilarity buckets by atom-pair Tanimoto instead, which is the
# axis we actually mean by "diverse", so a family of look-alikes shares a bucket and
# gets penalised. This matters most once every band is satisfied: a satisfied
# constraint stops pushing, and diversity is then the only remaining objective.
minsimilarity = 0.8
bucket_size = 400
minscore = 0.5
''')
        sh(REINVENT + ["-d", args.device, "-l", f"{s2_dir}/stage2.log", s2_cfg], None, env)

    # --- Stage 3 (opt-in): ChemProp λ surrogate if trained, else structural
    # fallback (DASAScaffold + AqueousSolubility + SA). ChemProp is skipped here
    # because it is env-blocked (numpy/sklearn ABI); train it via the notebook. ---
    if args.stage3 and not args.quick:
        s3_dir = os.path.join(out, "trial_stage3"); os.makedirs(s3_dir, exist_ok=True)
        s2_chkpt = os.path.join(out, "trial_stage2", "stage2.chkpt")
        s3_agent = s2_chkpt if os.path.isfile(s2_chkpt) else (
            s1_chkpt if os.path.isfile(s1_chkpt) else agent)
        s3_cfg = os.path.join(s3_dir, "stage3.toml")
        with open(s3_cfg, "w") as f:
            f.write(f'''run_type = "staged_learning"
device = "{args.device}"
tb_logdir = "{s3_dir}/tb"
[parameters]
prior_file = "{prior}"
agent_file = "{s3_agent}"
summary_csv_prefix = "{s3_dir}/stage3"
batch_size = 80
use_checkpoint = false
[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
[[stage]]
max_score = 1.0
max_steps = 400
chkpt_file = "{s3_dir}/stage3.chkpt"
[stage.scoring]
type = "geometric_mean"
[[stage.scoring.component]]
[stage.scoring.component.DASAScaffold]
[[stage.scoring.component.DASAScaffold.endpoint]]
name = "DASA"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASAColor]
[[stage.scoring.component.DASAColor.endpoint]]
name = "Color"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASAIntegrity]
[[stage.scoring.component.DASAIntegrity.endpoint]]
name = "Integrity"
weight = 1.0
[[stage.scoring.component]]
[stage.scoring.component.DASATrapEscape]
[[stage.scoring.component.DASATrapEscape.endpoint]]
name = "TrapEscape"
weight = 0.6
[[stage.scoring.component]]
[stage.scoring.component.AqueousSolubility]
[[stage.scoring.component.AqueousSolubility.endpoint]]
name = "Solubility"
weight = 0.5
params.logs_target = -2.0
params.logs_width = 1.5
params.logp_max = 1.0
params.logp_min = -2.5
[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name = "SA"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low = 2.0
transform.k = 0.4
[diversity_filter]
type = "ScaffoldSimilarity"
# IdenticalMurckoScaffold bucketed on EXACT Murcko scaffold, so the previous run
# reported healthy diversity while Butina found 275 of 300 shortlisted molecules in
# ONE cluster: swapping a heteroaryl ring changes the Murcko scaffold but not the
# chemistry. ScaffoldSimilarity buckets by atom-pair Tanimoto instead, which is the
# axis we actually mean by "diverse", so a family of look-alikes shares a bucket and
# gets penalised. This matters most once every band is satisfied: a satisfied
# constraint stops pushing, and diversity is then the only remaining objective.
minsimilarity = 0.8
bucket_size = 400
minscore = 0.5
''')
        sh(REINVENT + ["-d", args.device, "-l", f"{s3_dir}/stage3.log", s3_cfg], None, env)

    # --- report top candidates from the last stage that produced output ---
    if args.stage3 and not args.quick:
        last, prefix = os.path.join(out, "trial_stage3"), "stage3"
    elif args.stage2 and not args.quick:
        last, prefix = os.path.join(out, "trial_stage2"), "stage2"
    else:
        last, prefix = s1_dir, "stage1"
    csvs = sorted(glob.glob(os.path.join(last, f"{prefix}_*.csv")))
    if csvs:
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        sc = "Score" if "Score" in df.columns else df.columns[3]
        df = df.drop_duplicates("SMILES").sort_values(sc, ascending=False)
        keep = [c for c in ["SMILES", sc, "DASA", "Solubility", "xTB_Gap", "WaterSwitch"]
                if c in df.columns]
        n_dasa = int((df.get("DASA", pd.Series(dtype=float)) == 1.0).sum())
        print(f"\n=== {prefix}: {len(df)} unique molecules, {n_dasa} valid DASAs ===")
        print(df[keep].head(15).to_string(index=False))
        print(f"\nfull results: {last}")
    else:
        print("no summary CSV produced — check the logs in", last)


if __name__ == "__main__":
    main()
