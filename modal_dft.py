"""DFT verification on Modal — parallel TD-DFT solvatochromic-slope on cluster reps.

The DFT tier is embarrassingly parallel ACROSS molecules, so we run one container
per cluster representative and fan out. Each container does the multi-solvent
TD-DFT (CAM-B3LYP/ddCOSMO) for one DASA. Unlike the xTB/RL stage, PySCF wants
MANY OpenMP threads (no torch in this image), so OMP_NUM_THREADS = the container's
CPU count here.

Prereqs:  pip install modal ; modal token new

Run (after you have verified_candidates.csv from verify_dasa_outputs.py):
    modal run --detach modal_dft.py --csv outputs_dasa_full/verified_candidates.csv --n-reps 16
    modal run --detach modal_dft.py --csv <csv> --solvents "methanol,acetone,acetonitrile"
    modal run --detach modal_dft.py --csv <csv> --prepend "O=C(...)"   # DFT this probe FIRST
    modal run modal_dft.py --csv <csv> --n-reps 20 --dft-opt          # + DFT geometry opt (slower)

Solvents default to methanol,acetone,acetonitrile (comma-separated --solvents to change);
they must be keys in dft_verify.SOLVENTS. --prepend SMILES runs one probe molecule first,
bypassing the acceptor/AntiTrap filters and clustering.

Results (per-candidate lambda_max/solvent + solvatochromic slope) print and are
written to the 'dasa-outputs' volume as dft_results.json + downloaded locally.
NOTE: paid cloud compute on YOUR account; ~2-4 CPU-hours per representative.
"""
import os
import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-dft")

image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("rdkit=2024.03", "numpy<2", channels=["conda-forge"])
    .pip_install("pyscf==2.5.0", "geometric", "pandas", "scipy")
    .add_local_dir(
        REPO, "/repo",
        ignore=[".git", "__pycache__", ".ipynb_checkpoints",
                "outputs_dasa", "outputs_dasa_modal", "outputs", "outputs_rl2",
                "build", "*.egg-info", "FS_Ro5_10M.model", "mol2mol_scaffold.prior",
                "reinvent.prior"],
    )
)
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)

# CPU count per container -> PySCF OpenMP threads. Raise for bigger molecules.
_CPU = 8


@APP.function(image=image, cpu=2.0, timeout=60 * 10)
def cluster(smiles_list: list, cutoff: float, n_reps: int):
    """Butina clustering runs IN a container (has rdkit) so the local client
    doesn't need rdkit/pandas — it only needs `modal`."""
    import sys
    sys.path.insert(0, "/repo/notebooks")
    from dft_verify import cluster_representatives
    return cluster_representatives(smiles_list, cutoff, n_reps)


@APP.function(image=image, volumes={"/results": vol}, cpu=_CPU,
              timeout=60 * 60 * 6, retries=1)
def dft_one(smiles: str, solvents=None, functionals=None, min_visible_nm: float = 0.0,
            dft_opt: bool = False):
    import sys, json, hashlib
    sys.path.insert(0, "/repo/notebooks")
    os.environ["OMP_NUM_THREADS"] = str(int(_CPU))   # PySCF wants all cores here
    from dft_verify import run_candidate
    functionals = functionals or ["camb3lyp"]
    # one full characterisation per functional (CAM-B3LYP over-blue-shifts DASAs;
    # B3LYP tracks experiment better for these push-pull dyes -- report both).
    by_fn = {xc: run_candidate(smiles, solvents=solvents, quick=False, dft_opt=dft_opt,
                               xc=xc, min_visible_nm=min_visible_nm) for xc in functionals}
    result = {"smiles": smiles,
              "screen_lambda_nm": by_fn[functionals[0]].get("screen_lambda_nm"),
              "by_functional": by_fn}
    # persist each result to the volume so a client disconnect can't lose it
    # (unique filename per molecule -> no concurrent-write race).
    try:
        os.makedirs("/results/dft", exist_ok=True)
        h = hashlib.md5(smiles.encode()).hexdigest()[:10]
        with open(f"/results/dft/{h}.json", "w") as f:
            json.dump(result, f, indent=2)
        vol.commit()
    except Exception:
        pass
    return result


@APP.function(image=image, volumes={"/results": vol}, timeout=60 * 60 * 10)
def orchestrate(smiles_pool: list, n_reps: int, cutoff: float,
                quick: bool, dft_opt: bool, solvents=None, prepend: str = "",
                functionals=None, min_visible_nm: float = 0.0, no_cluster: bool = False):
    """Runs REMOTELY (detach-safe): (optionally cluster) -> fan out full DFT ->
    aggregate to the volume.

    `no_cluster` DFTs every molecule in `smiles_pool` in the given order (used when
    the pool is already the final hand-picked list, e.g. the 16 reps ordered by
    pre-screen lambda). `prepend`, if given, is a probe DFT'd FIRST. `min_visible_nm`
    <= 0 keeps the pre-screen as a ranking signal only (never rejects)."""
    import sys, json
    sys.path.insert(0, "/repo/notebooks")
    from dft_verify import cluster_representatives
    if no_cluster:
        reps = list(smiles_pool)
        print(f"no-cluster: DFT all {len(reps)} molecules in given order", flush=True)
    else:
        reps = cluster_representatives(smiles_pool, cutoff, n_reps)
    run_list = ([prepend] if prepend else []) + [r for r in reps if r != prepend]
    print(f"pool {len(smiles_pool)} -> {len(run_list)} molecules"
          + (" (+probe FIRST)" if prepend else "")
          + f" -> DFT {solvents} x {functionals} (min_visible={min_visible_nm})", flush=True)
    results = list(dft_one.starmap(
        [(r, solvents, functionals, min_visible_nm, dft_opt) for r in run_list]))
    with open("/results/dft_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    # rank by the first functional's reddest computed lambda_max (visible ~540-600 nm)
    fn0 = (functionals or ["camb3lyp"])[0]
    def lam_red(r):
        lb = (r.get("by_functional", {}).get(fn0, {}) or {}).get("lambda_by_solvent") or {}
        vals = [v for v in lb.values() if v]
        return max(vals) if vals else None
    ranked = sorted(((lam_red(r), r) for r in results),
                    key=lambda t: (t[0] is not None, t[0]), reverse=True)
    print(f"Ranked by {fn0} lambda_max (reddest first; visible target ~540-600 nm):", flush=True)
    for lam, r in ranked[:12]:
        scr = r.get("screen_lambda_nm")
        print(f"  lam_max={lam}  (pre-screen {scr})  {r['smiles'][:55]}", flush=True)
    return results


@APP.local_entrypoint()
def main(csv: str, n_reps: int = 16, pool: int = 150, cutoff: float = 0.4,
         quick: bool = False, dft_opt: bool = False,
         solvents: str = "toluene,acetone,methanol,acetonitrile", prepend: str = "",
         functionals: str = "camb3lyp,b3lyp", min_visible: float = 0.0,
         no_cluster: bool = False, per_arch: int = 3,
         min_antitrap: float = 0.7, min_gap: float = 0.5, min_solubility: float = 0.7):
    """Full TD-DFT on candidates.

    Two modes:
      * default: filter (acceptor/AntiTrap/Solubility) + acceptor-stratify + cluster
        the big verified CSV down to `n_reps` diverse reps, then DFT them.
      * --no-cluster: DFT EVERY row of `csv` in file order (used when the CSV is
        already the final hand-picked list, e.g. the 16 reps ordered by pre-screen
        lambda). No filtering, no clustering.

    `min_visible` <= 0 (default) keeps the gas-phase pre-screen as a RANKING signal
    only -- it never rejects (the pre-screen blue-shifts DASAs ~200 nm, so its
    absolute value is not a visibility verdict). `functionals` runs each molecule
    under every listed XC (default CAM-B3LYP + B3LYP) for a method-spread readout.
    """
    import csv as _csv
    with open(csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv} is empty")
    col = "SMILES" if "SMILES" in rows[0] else "smiles"
    solvent_list = [s.strip() for s in solvents.split(",") if s.strip()]
    functional_list = [s.strip() for s in functionals.split(",") if s.strip()]

    if no_cluster:
        picked = [r.get(col) for r in rows if r.get(col)]
        print(f"no-cluster: DFT all {len(picked)} CSV rows in order | "
              f"solvents={solvent_list} functionals={functional_list} min_visible={min_visible}")
        orchestrate.spawn(picked, n_reps, cutoff, quick, dft_opt, solvent_list, prepend,
                          functional_list, min_visible, True)
        print("dispatched DETACHED. watch: modal app logs <app-id> -f ; "
              "fetch: modal volume get --force dasa-outputs dft_results.json .")
        return

    def fget(r, k):
        try:
            return float(r.get(k, ""))
        except (TypeError, ValueError):
            return None

    # Stratify the DFT set BY DONOR ARCHITECTURE (the switching MECHANISM) first, then
    # by acceptor within each. A run can collapse onto one mechanism/acceptor; this
    # guarantees the validation set spans mechanisms -- 1st-gen dialkyl (trapped, but a
    # host-guest candidate), 2nd-gen aniline (neutral closed form), tethered -- and keeps
    # the top `per_arch` of EACH so meaningful within-mechanism variants are tested too.
    import itertools
    from collections import defaultdict
    by_arch = defaultdict(lambda: defaultdict(list))   # arch -> acceptor -> [smiles], rank order
    for r in rows:
        s = r.get(col)
        if not s or r.get("acceptor") == "other":
            continue
        at, sol = fget(r, "AntiTrap"), fget(r, "Solubility")
        if at is None or at < min_antitrap:
            continue
        if sol is not None and sol < min_solubility:
            continue
        by_arch[r.get("donor_arch", "other")][r.get("acceptor", "?")].append(s)

    if not by_arch:
        raise SystemExit(
            f"No candidate passed the filter (AntiTrap>={min_antitrap}, "
            f"sol>={min_solubility}, canonical acceptor). Loosen the thresholds.")
    picked, seen, summary = [], set(), []
    for arch, accs in by_arch.items():
        got, acc_buckets = [], [v for v in accs.values()]
        for group in itertools.zip_longest(*acc_buckets):   # interleave acceptors within arch
            for s in group:
                if s and s not in seen:
                    seen.add(s); got.append(s)
                if len(got) >= per_arch:
                    break
            if len(got) >= per_arch:
                break
        picked += got
        summary.append(f"{arch}={len(got)}")
    print(f"{len(rows)} candidates -> {len(picked)} pooled, top {per_arch} per donor "
          f"architecture: " + ", ".join(summary))

    picked = [s for s in picked if s != prepend]      # avoid DFT'ing the probe twice
    if prepend:
        print(f"probe prepended (DFT'd FIRST): {prepend}")
    print(f"solvents: {solvent_list}  functionals: {functional_list}  min_visible: {min_visible}")

    # spawn() so it survives disconnect; results land in the volume.
    orchestrate.spawn(picked, n_reps, cutoff, quick, dft_opt, solvent_list, prepend,
                      functional_list, min_visible, False)
    print("dispatched DETACHED. Results will appear in the 'dasa-outputs' volume:")
    print("  dft_results.json (aggregate) + dft/<hash>.json (per molecule, live)")
    print("  watch:  modal app logs <app-id> -f")
    print("  fetch:  modal volume get --force dasa-outputs dft_results.json .")
