# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (reinvent4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DASA Discovery — Complete Pipeline
#
# **Goal**: Use REINVENT4 to generate novel **Donor–Acceptor Stenhouse Adducts
# (DASAs)** that are **water-soluble AND water-switchable** — an open problem,
# since DASAs normally switch only in organic solvent and collapse irreversibly
# to the closed zwitterion in water.
#
# This is the DASA-adapted successor to `photoswitch_complete.py` (azobenzenes).
# The pipeline *shape* is unchanged (TL → staged RL → surrogate scoring →
# clustering → DFT validation); the **chemistry of the filters and scorers** is
# rebuilt for DASAs. See the project memory and `dasa_chem.py` for details.
#
# ## What changed from the azobenzene pipeline
#
# | Piece | Azobenzene version | DASA version |
# |-------|-------------------|--------------|
# | Scaffold gate | azo/hydrazone SMARTS | `DASAScaffold` (amino-triene + carbon-acid) |
# | Isomerisation | N=N E/Z flip (`XTBIsomerGap`) | open↔closed electrocyclization; scored via `DASASwitchability` |
# | λ target | ~450 nm | ~540–600 nm (redder, visible) |
# | Solubility | *(none)* | `AqueousSolubility` — new, core to the goal |
# | Switchability | *(none)* | `DASASwitchability` — xTB charge-separation window + differential solvation |
# | Data | `photoswitches.csv` (azo) | `data/dasa_dataset.csv` (literature extraction) |
#
# ## Staged RL design
#
# | Stage | Scoring components | Batch | Steps | Speed |
# |-------|--------------------|-------|-------|-------|
# | **1 — structural + solubility** | DASAScaffold, AqueousSolubility, SA | 128 | 500 | fast |
# | **2 — xTB electronic** | Stage 1 + XTBHomoLumo (λ) + DASASwitchability | 40 | 300 | ~2–6 s/mol |
# | **3 — ChemProp surrogate** | DASAScaffold, AqueousSolubility, ChemProp λ, SA | 80 | 400 | ~0.3 s/mol |

# %% [markdown]
# ## §1 — Setup & Imports

# %%
import os, sys, glob, shutil, subprocess, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit import DataStructs

# DASA chemistry library (this directory)
sys.path.insert(0, os.path.dirname(os.path.abspath("")) or ".")
sys.path.insert(0, os.path.join(os.getcwd(), "notebooks"))
try:
    import dasa_chem as dc
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath("dasa_chem.py")))
    import dasa_chem as dc
print("dasa_chem loaded — DASA SMARTS:", dc.DASA_OPEN_SMARTS)

# %% [markdown]
# ## §2 — Paths & Configuration

# %%
# Robust project-root detection (works from notebooks/ or repo root)
_here = os.getcwd()
PROJ_ROOT = _here if os.path.isdir(os.path.join(_here, "reinvent")) else os.path.dirname(_here)
DATA_DIR   = os.path.join(PROJ_ROOT, "notebooks", "data")
OUT_DIR    = os.path.join(PROJ_ROOT, "outputs_dasa"); os.makedirs(OUT_DIR, exist_ok=True)
PLUGIN_DIR = os.path.join(PROJ_ROOT, "plugins")

PRIOR_FILE = os.path.join(PROJ_ROOT, "reinvent.prior")
assert os.path.isfile(PRIOR_FILE), f"Prior not found: {PRIOR_FILE}"

# Literature-extracted dataset (drop your extraction here — see template):
DATASET_CSV   = os.path.join(DATA_DIR, "dasa_dataset.csv")
BOOTSTRAP_SMI = os.path.join(DATA_DIR, "dasa_bootstrap.smi")

print(f"Prior      : {PRIOR_FILE}")
print(f"Output     : {OUT_DIR}")
print(f"Dataset CSV: {DATASET_CSV}  ({'FOUND' if os.path.isfile(DATASET_CSV) else 'not yet — will bootstrap from enumeration'})")

# %%
DEVICE = "cpu"  # change to "cuda:0" on FarmShare GPU nodes

RUN_CHEMPROP = True
RUN_TL       = True
RUN_STAGE1   = True
RUN_STAGE2   = True
RUN_STAGE3   = True
RUN_DFT      = True

# DASA-tuned targets (calibrate against literature once data lands — see §5)
LAMBDA_TARGET_NM = 570.0     # open-form visible absorption target
# GFN2-xTB HOMO-LUMO gaps measured on enumerated DASAs cluster at ~1.68–2.03 eV.
# NOTE: the xTB ground-state gap is only a WEAK λ proxy for DASAs (it barely
# varies across donors/acceptors), and the azobenzene 2.5x gap→optical-gap
# correction does NOT transfer (it maps DASA gaps to ~250–295 nm, nonsense).
# So XTBHomoLumo here acts as a mild "reasonable conjugated chromophore" sanity
# term; real λ_max optimisation is deferred to the ChemProp surrogate (§5/§10)
# and TD-DFT validation (§14). Empirical DASA correction factor ≈ 1.2.
XTB_GAP_MIN_EV       = 1.50   # centred on the observed ~1.8 eV DASA range
XTB_GAP_MAX_EV       = 2.10
XTB_LAMBDA_CORRECTION = 1.2   # DASA-specific gap→λ factor (vs 2.5 for azobenzenes)


def run_reinvent(config_file, log_file, device=None):
    """Execute REINVENT and stream output live."""
    _device = device or DEVICE
    # PLUGIN_DIR = DASA scoring components; PROJ_ROOT ensures the repo's reinvent
    # (whose importer tolerates optional components that fail to import, e.g.
    # comp_chemprop under this env's numpy/sklearn ABI break) shadows any
    # site-packages copy so the scoring registry loads.
    _env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [PLUGIN_DIR, PROJ_ROOT, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
    }
    # prefer the reinvent console script; fall back to `python -m reinvent`
    # (works whenever the repo is importable, e.g. env not activated / headless)
    _rv = os.path.join(os.path.dirname(sys.executable), "reinvent")
    if not os.path.isfile(_rv):
        _rv = shutil.which("reinvent")
    _prefix = [_rv] if _rv else [sys.executable, "-m", "reinvent"]
    cmd = _prefix + ["-d", _device, "-l", log_file, config_file]
    print(f"▶ {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=_env)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        print(f"\n{'✓ Done' if proc.returncode == 0 else f'✗ Exit {proc.returncode}'}")
        return proc.returncode
    except FileNotFoundError:
        print("[ERROR] 'reinvent' not found — activate the reinvent4 env.")
        return -1


# %% [markdown]
# ## §3 — Load Data (literature extraction, else bootstrap enumeration)
#
# Drop your literature-extracted table at `data/dasa_dataset.csv`. Schema
# (`data/dasa_dataset_template.csv` shows it — only `smiles_open` is required):
#
# | column | meaning |
# |--------|---------|
# | `smiles_open` | **required** — open-form SMILES |
# | `smiles_closed` | optional measured closed form |
# | `donor_class` / `acceptor_class` | optional (auto-filled from SMARTS) |
# | `lambda_max_open_nm` | measured open-form λ_max |
# | `solvent` | measurement solvent |
# | `pct_open_equilibrium` | % open at dark equilibrium |
# | `solvatochromic_slope_nm` | charge-separation proxy (negative nm) |
# | `switches_in_water` | label: True / False / partial |
# | `source` | citation / DOI |
#
# Until that file exists, we bootstrap a transfer-learning corpus by
# **combinatorial enumeration** (donor × acceptor building blocks) so the loop
# is runnable end-to-end today.

# %%
# TL corpus: real literature extraction if present, else the enumerated library
# (18 donors x 3 backbones x 13 acceptors = ~700 DASAs, from Helmy/Hemmer blocks).
if os.path.isfile(DATASET_CSV):
    data = dc.load_dasa_dataset(DATASET_CSV)
    print(f"Loaded literature dataset: {len(data)} valid DASAs")
else:
    data = pd.DataFrame(dc.enumerate_dasa())
    print(f"No {os.path.basename(DATASET_CSV)} — using enumerated library: {len(data)} DASAs")
    print("  (drop your extraction at data/dasa_dataset.csv to switch to real data)")
corpus_smiles = data["smiles_open"].tolist()

# Labelled set for the ChemProp λ surrogate (§5) and switchability calibration.
# Always fold in the curated literature seed (real reported λ_max) if available.
LIT_SEED_CSV = os.path.join(DATA_DIR, "dasa_literature_seed.csv")
labelled_parts = []
if "lambda_max_open_nm" in data.columns:
    labelled_parts.append(data.dropna(subset=["lambda_max_open_nm"]))
if os.path.isfile(LIT_SEED_CSV):
    seed = dc.load_dasa_dataset(LIT_SEED_CSV)
    labelled_parts.append(seed.dropna(subset=["lambda_max_open_nm"]))
    print(f"  literature seed: {len(seed)} exemplars with reported λ_max")
labelled = (pd.concat(labelled_parts, ignore_index=True)
            .drop_duplicates(subset="smiles_open") if labelled_parts else pd.DataFrame())
print(f"  labelled (measured λ_max_open): {len(labelled)}")

from collections import Counter
print("  donor classes   :", dict(Counter(data.get("donor_class", []))))
print("  acceptor classes:", dict(Counter(data.get("acceptor_class", []))))
print("  backbones        :", dict(Counter(data.get("backbone", []))))

# %% [markdown]
# ## §4 — TL Corpus Preparation (scaffold split)

# %%
def scaffold_split(smiles_list, train_frac=0.8, seed=42):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    rng = np.random.default_rng(seed)
    buckets = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = "__generic__"
        buckets.setdefault(scaf, []).append(smi)
    groups = list(buckets.values())
    rng.shuffle(groups)
    n_target = int(sum(len(g) for g in groups) * train_frac)
    train, val, n = [], [], 0
    for g in groups:
        if n < n_target:
            train += g; n += len(g)
        else:
            val += g
    return train, val


def _flatten(smiles_list):
    # reinvent.prior has no stereo tokens (/ \), so strip E/Z for all model-facing
    # files (TL train/val + inception). DASA connectivity is preserved.
    out = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            out.append(Chem.MolToSmiles(m, isomericSmiles=False))
    return out


tl_train, tl_val = scaffold_split(corpus_smiles, 0.8)
TL_TRAIN_SMI = os.path.join(OUT_DIR, "dasa_tl_train.smi")
TL_VAL_SMI   = os.path.join(OUT_DIR, "dasa_tl_val.smi")
# guard: TL needs a non-trivial validation set; if tiny corpus, reuse train
if len(tl_val) < 3:
    tl_val = tl_train
with open(TL_TRAIN_SMI, "w") as f:
    f.write("\n".join(_flatten(tl_train)) + "\n")
with open(TL_VAL_SMI, "w") as f:
    f.write("\n".join(_flatten(tl_val)) + "\n")
print(f"TL train: {len(tl_train)}, TL val: {len(tl_val)} (stereo-stripped for the model vocab)")
print(f"  {TL_TRAIN_SMI}")

# %% [markdown]
# ## §5 — ChemProp λ_max Surrogate (only if enough labelled data)
#
# With a real literature dataset carrying `lambda_max_open_nm`, we train a
# ChemProp v1 GNN to predict open-form λ_max (used in Stage 3). This also the
# natural place to **calibrate** the xTB switchability descriptors: correlate
# `solvatochromic_slope_nm` / `switches_in_water` against the xTB dipole &
# differential-solvation values from `dasa_chem`/the plugin, then update the
# `DASASwitchability` target/width params accordingly.

# %%
CHEMPROP_LAMBDA_DIR = os.path.join(OUT_DIR, "chemprop_lambda_open")
_MIN_LABELS = 30
_cp_lam_ok = False

if RUN_CHEMPROP and len(labelled) >= _MIN_LABELS:
    os.makedirs(CHEMPROP_LAMBDA_DIR, exist_ok=True)
    lam = labelled[["smiles_open", "lambda_max_open_nm"]].rename(
        columns={"smiles_open": "smiles", "lambda_max_open_nm": "lambda_max"})
    lam_tr = lam.sample(frac=0.8, random_state=42)
    lam_va = lam.drop(lam_tr.index)
    tr_csv = os.path.join(CHEMPROP_LAMBDA_DIR, "train.csv")
    va_csv = os.path.join(CHEMPROP_LAMBDA_DIR, "val.csv")
    lam_tr.to_csv(tr_csv, index=False); lam_va.to_csv(va_csv, index=False)
    chemprop_bin = os.path.join(os.path.dirname(sys.executable), "chemprop_train")
    if os.path.isfile(chemprop_bin):
        cmd = [chemprop_bin, "--data_path", tr_csv, "--separate_val_path", va_csv,
               "--dataset_type", "regression", "--target_columns", "lambda_max",
               "--save_dir", CHEMPROP_LAMBDA_DIR, "--epochs", "80",
               "--hidden_size", "300", "--depth", "3", "--metric", "rmse"]
        subprocess.run(cmd, check=False)
        _cp_lam_ok = os.path.isfile(os.path.join(CHEMPROP_LAMBDA_DIR, "model_0", "model.pt"))
    print(f"ChemProp λ_max surrogate trained: {_cp_lam_ok}")
else:
    print(f"Skipping ChemProp: need ≥{_MIN_LABELS} labelled rows, have {len(labelled)}. "
          "Stage 3 will fall back to structural + solubility scoring.")

# %% [markdown]
# ## §6 — Verify DASA Scoring Plugins
#
# Unlike the azobenzene notebook (which wrote plugins at runtime), the DASA
# components are version-controlled files under
# `plugins/reinvent_plugins/components/`. We just confirm they're present.

# %%
PLUGIN_COMP = os.path.join(PLUGIN_DIR, "reinvent_plugins", "components")
_expected = ["comp_dasa_scaffold.py", "comp_aqueous_solubility.py",
             "comp_dasa_switchability.py", "dasa_common.py"]
for p in _expected:
    ok = os.path.isfile(os.path.join(PLUGIN_COMP, p))
    print(f"  [{'✓' if ok else '✗'}] {p}")
# XTBHomoLumo (reused for open-form λ) also lives there:
print(f"  [{'✓' if os.path.isfile(os.path.join(PLUGIN_COMP, 'comp_xtb_homo_lumo.py')) else '✗'}] comp_xtb_homo_lumo.py (reused for λ)")

# %% [markdown]
# ## §7 — Transfer Learning
#
# Fine-tune `reinvent.prior` on the DASA corpus to bias the generator toward
# amino-triene / carbon-acid fragments.

# %%
TL_OUT_DIR = os.path.join(OUT_DIR, "tl_run"); os.makedirs(TL_OUT_DIR, exist_ok=True)
TL_MODEL   = os.path.join(TL_OUT_DIR, "TL_dasa.model")
TB_TL_DIR  = os.path.join(TL_OUT_DIR, "tb_tl")

TL_CONFIG = f"""\
run_type = "transfer_learning"
device = "{DEVICE}"
tb_logdir = "{TB_TL_DIR}"

[parameters]
num_epochs            = 50
save_every_n_epochs   = 5
batch_size            = 64
sample_batch_size     = 1000
input_model_file      = "{PRIOR_FILE}"
output_model_file     = "{TL_MODEL}"
smiles_file           = "{TL_TRAIN_SMI}"
validation_smiles_file = "{TL_VAL_SMI}"
standardize_smiles    = true
randomize_smiles      = true
randomize_all_smiles  = false
internal_diversity    = true
"""
TL_CONFIG_FILE = os.path.join(TL_OUT_DIR, "tl_config.toml")
with open(TL_CONFIG_FILE, "w") as f:
    f.write(TL_CONFIG)
print(f"TL config: {TL_CONFIG_FILE}")

# %%
if RUN_TL:
    run_reinvent(TL_CONFIG_FILE, os.path.join(TL_OUT_DIR, "tl.log"))
else:
    print("Skipping TL (RUN_TL=False).")

# %%
chkpts = sorted(glob.glob(os.path.join(TL_OUT_DIR, "TL_dasa.model.*.chkpt")))
if chkpts:
    epochs = sorted(int(os.path.basename(c).split(".")[-2]) for c in chkpts)
    TL_EPOCH = 30 if 30 in epochs else epochs[len(epochs) // 2]
else:
    TL_EPOCH = 30
TL_BEST = os.path.join(TL_OUT_DIR, f"TL_dasa.model.{TL_EPOCH}.chkpt")
AGENT_FILE = TL_BEST if os.path.isfile(TL_BEST) else (TL_MODEL if os.path.isfile(TL_MODEL) else PRIOR_FILE)
print(f"Agent for RL: {AGENT_FILE}")

# %% [markdown]
# ## §8 — Stage 1: Structural Gate + Aqueous Solubility (fast)
#
# Geometric mean of: **DASAScaffold** (hard gate, wt 1.0), **AqueousSolubility**
# (wt 0.7 — the water-soluble half of the goal), **SA Score** (wt 0.4).

# %%
S1_DIR = os.path.join(OUT_DIR, "rl_stage1"); os.makedirs(S1_DIR, exist_ok=True)
S1_CHKPT = os.path.join(S1_DIR, "stage1.chkpt")

STAGE1_TOML = f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir = "{S1_DIR}/tb"
json_out_config = "{S1_DIR}/_stage1.json"

[parameters]
prior_file         = "{PRIOR_FILE}"
agent_file         = "{AGENT_FILE}"
summary_csv_prefix = "{S1_DIR}/stage1"
batch_size         = 128
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = 500
chkpt_file = "{S1_CHKPT}"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.DASAScaffold]
[[stage.scoring.component.DASAScaffold.endpoint]]
name   = "DASA"
weight = 1.0

[[stage.scoring.component]]
[stage.scoring.component.AqueousSolubility]
[[stage.scoring.component.AqueousSolubility.endpoint]]
name   = "Solubility"
weight = 0.7
params.logs_target = -2.0
params.logs_width  = 1.5
params.logp_max    = 3.0

[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.4
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 25
minscore    = 0.4

[inception]
smiles_file = "{TL_TRAIN_SMI}"
memory_size = 100
sample_size = 10
"""
s1_cfg = os.path.join(S1_DIR, "stage1.toml")
with open(s1_cfg, "w") as f:
    f.write(STAGE1_TOML)
print(f"Stage 1 config: {s1_cfg}")

# %%
if RUN_STAGE1:
    run_reinvent(s1_cfg, os.path.join(S1_DIR, "stage1.log"))
else:
    print("Skipping Stage 1")

# %% [markdown]
# ## §9 — Stage 2: xTB Electronic + Water-Switchability (slow)
#
# Stage 1 gates carried forward, plus:
# - **XTBHomoLumo** (wt 0.3, mild sanity term) — GFN2-xTB gap in 1.5–2.1 eV, the
#   observed DASA range. The xTB ground-state gap barely varies across DASAs, so
#   this only weeds out non-conjugated junk; real λ_max comes from ChemProp/TD-DFT.
# - **DASASwitchability** (wt 0.8, the real driver) — xTB charge-separation
#   window (dipole in ALPB water) + differential water/toluene solvation.
#
# Batch 40 (xTB ~2–6 s/mol). ~300 steps.

# %%
S2_DIR = os.path.join(OUT_DIR, "rl_stage2"); os.makedirs(S2_DIR, exist_ok=True)
S2_CHKPT = os.path.join(S2_DIR, "stage2.chkpt")
_s2_agent = S1_CHKPT if os.path.isfile(S1_CHKPT) else AGENT_FILE

STAGE2_TOML = f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir = "{S2_DIR}/tb"
json_out_config = "{S2_DIR}/_stage2.json"

[parameters]
prior_file         = "{PRIOR_FILE}"
agent_file         = "{_s2_agent}"
summary_csv_prefix = "{S2_DIR}/stage2"
batch_size         = 40
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = 300
chkpt_file = "{S2_CHKPT}"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.DASAScaffold]
[[stage.scoring.component.DASAScaffold.endpoint]]
name   = "DASA"
weight = 1.0

[[stage.scoring.component]]
[stage.scoring.component.AqueousSolubility]
[[stage.scoring.component.AqueousSolubility.endpoint]]
name   = "Solubility"
weight = 0.6
params.logs_target = -2.0
params.logs_width  = 1.5
params.logp_max    = 3.0

[[stage.scoring.component]]
[stage.scoring.component.XTBHomoLumo]
[[stage.scoring.component.XTBHomoLumo.endpoint]]
name   = "xTB_Gap"
weight = 0.3
params.gap_min_ev = {XTB_GAP_MIN_EV}
params.gap_max_ev = {XTB_GAP_MAX_EV}

[[stage.scoring.component]]
[stage.scoring.component.DASASwitchability]
[[stage.scoring.component.DASASwitchability.endpoint]]
name   = "WaterSwitch"
weight = 0.8
params.dipole_target_au      = 4.0
params.dipole_sigma_au       = 1.6
params.solv_diff_target_kcal = 0.0
params.solv_diff_sigma_kcal  = 6.0

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.5

[inception]
smiles_file = ""
memory_size = 100
sample_size = 10
"""
s2_cfg = os.path.join(S2_DIR, "stage2.toml")
with open(s2_cfg, "w") as f:
    f.write(STAGE2_TOML)
print(f"Stage 2 config: {s2_cfg}")

# %%
if RUN_STAGE2:
    run_reinvent(s2_cfg, os.path.join(S2_DIR, "stage2.log"))
else:
    print("Skipping Stage 2")

# %% [markdown]
# ## §10 — Stage 3: ChemProp λ Surrogate (medium; only if trained)

# %%
S3_DIR = os.path.join(OUT_DIR, "rl_stage3"); os.makedirs(S3_DIR, exist_ok=True)
S3_CHKPT = os.path.join(S3_DIR, "stage3.chkpt")
_s3_agent = S2_CHKPT if os.path.isfile(S2_CHKPT) else (
    S1_CHKPT if os.path.isfile(S1_CHKPT) else AGENT_FILE)

_CP_LAM = f"""
[[stage.scoring.component]]
[stage.scoring.component.ChemProp]
[[stage.scoring.component.ChemProp.endpoint]]
name   = "CP_Lambda"
weight = 0.7
params.checkpoint_dir      = ["{CHEMPROP_LAMBDA_DIR}"]
params.rdkit_2d_normalized = [false]
params.features            = [""]
params.target_column       = ["lambda_max"]
transform.type = "double_sigmoid"
transform.high = 640.0
transform.low  = 500.0
transform.coef_div  = 200.0
transform.coef_si   = 10.0
transform.coef_se   = 10.0
""" if _cp_lam_ok else ""

STAGE3_TOML = f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir = "{S3_DIR}/tb"
json_out_config = "{S3_DIR}/_stage3.json"

[parameters]
prior_file         = "{PRIOR_FILE}"
agent_file         = "{_s3_agent}"
summary_csv_prefix = "{S3_DIR}/stage3"
batch_size         = 80
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = 400
chkpt_file = "{S3_CHKPT}"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.DASAScaffold]
[[stage.scoring.component.DASAScaffold.endpoint]]
name   = "DASA"
weight = 1.0

[[stage.scoring.component]]
[stage.scoring.component.AqueousSolubility]
[[stage.scoring.component.AqueousSolubility.endpoint]]
name   = "Solubility"
weight = 0.6
params.logs_target = -2.0
params.logs_width  = 1.5
params.logp_max    = 3.0

[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4
""" + _CP_LAM + f"""
[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.5

[inception]
smiles_file = ""
memory_size = 100
sample_size = 10
"""
s3_cfg = os.path.join(S3_DIR, "stage3.toml")
with open(s3_cfg, "w") as f:
    f.write(STAGE3_TOML)
print(f"Stage 3 config: {s3_cfg}  (ChemProp λ active: {_cp_lam_ok})")

# %%
if RUN_STAGE3:
    run_reinvent(s3_cfg, os.path.join(S3_DIR, "stage3.log"))
else:
    print("Skipping Stage 3")

# %% [markdown]
# ## §11 — Results Collection & Ranking

# %%
def load_rl_csv(directory, prefix):
    csvs = sorted(glob.glob(os.path.join(directory, f"{prefix}_*.csv")))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


df_s1, df_s2, df_s3 = (load_rl_csv(S1_DIR, "stage1"),
                        load_rl_csv(S2_DIR, "stage2"),
                        load_rl_csv(S3_DIR, "stage3"))
for label, df in [("Stage 1", df_s1), ("Stage 2", df_s2), ("Stage 3", df_s3)]:
    if df.empty:
        print(f"{label}: no data")
    else:
        sc = "Score" if "Score" in df.columns else df.columns[3]
        print(f"{label}: {len(df)} rows, top score = {df[sc].max():.3f}")


def get_top(df, n=200, min_score=0.4):
    if df.empty:
        return df
    valid = df[df.get("SMILES_state", pd.Series([1] * len(df))) == 1].copy()
    valid = valid.drop_duplicates(subset=["SMILES"])
    sc = next((c for c in ("Score", "total_score") if c in valid.columns), None)
    if sc:
        valid = valid[valid[sc] >= min_score].sort_values(sc, ascending=False)
    return valid.head(n)


best_df = df_s3 if not df_s3.empty else (df_s2 if not df_s2.empty else df_s1)
top = get_top(best_df, 200, 0.4)
print(f"\nTop candidates: {len(top)}")
if not top.empty:
    cols = [c for c in ["SMILES", "Score", "DASA", "Solubility", "xTB_Gap", "WaterSwitch"]
            if c in top.columns]
    print(top[cols].head(20).to_string(index=False))

# %% [markdown]
# ## §12 — Clustering for Diversity (Butina on Morgan FP)

# %%
def tanimoto_cluster(smiles_list, cutoff=0.4):
    from rdkit.ML.Cluster import Butina
    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles_list]
    mols = [(s, m) for s, m in mols if m is not None]
    if not mols:
        return []
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for _, m in mols]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    return [{"representative": mols[c[0]][0], "cluster_size": len(c),
             "members": [mols[i][0] for i in c]} for c in clusters]


if not top.empty:
    clusters = tanimoto_cluster(top["SMILES"].tolist(), 0.4)
    representatives = [c["representative"] for c in clusters]
    print(f"{len(clusters)} clusters from {len(top)} molecules; "
          f"sizes {[c['cluster_size'] for c in clusters[:12]]}")
else:
    clusters, representatives = [], []
    print("No candidates to cluster.")

# %%
if representatives:
    mols = [Chem.MolFromSmiles(s) for s in representatives[:20]]
    mols = [m for m in mols if m is not None]
    if mols:
        display(Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 250),
                                     legends=[f"Cluster {i}" for i in range(len(mols))]))

# %% [markdown]
# ## §13 — xTB Screen on Cluster Representatives
#
# Fast per-representative readout of the same descriptors the RL used:
# open-form λ (from gap), dipole in water (charge separation), and the
# water-vs-toluene differential solvation. Uses `dasa_chem` helpers.

# %%
def xtb_screen(smiles):
    """Open-form descriptors via GFN2-xTB (water + toluene ALPB)."""
    import numpy as np
    sys.path.insert(0, PLUGIN_COMP)
    try:
        from dasa_common import embed_3d, xtb_properties  # plugin-local helpers
    except Exception:
        return None
    m = Chem.MolFromSmiles(smiles)
    m3d = embed_3d(m)
    if m3d is None:
        return None
    water = xtb_properties(m3d, "water")
    tol = xtb_properties(m3d, "toluene")
    if water is None or tol is None:
        return None
    e_w, dip_w, gap = water
    e_t, _, _ = tol
    lam = 1240.0 / (gap * XTB_LAMBDA_CORRECTION) if gap and gap > 0 else None
    return {"smiles": smiles, "gap_eV": gap, "lam_est_nm": lam,
            "dipole_water_au": dip_w, "solv_diff_kcal": (e_w - e_t) * 627.509}


try:
    from xtb.interface import Calculator  # noqa: F401
    _HAS_XTB = True
except ImportError:
    _HAS_XTB = False
    print("⚠ xtb-python not installed — skipping xTB screen.")

xtb_rows = []
if _HAS_XTB and representatives:
    n = min(40, len(representatives))
    print(f"{'#':>3} {'λ_est':>7} {'dipole(H2O)':>12} {'ΔsolvG(kcal)':>13}  SMILES")
    for i, smi in enumerate(representatives[:n]):
        r = xtb_screen(smi)
        if r:
            xtb_rows.append(r)
            print(f"{i+1:>3} {r['lam_est_nm'] or 0:>7.0f} {r['dipole_water_au']:>12.2f} "
                  f"{r['solv_diff_kcal']:>13.1f}  {smi[:44]}")
xtb_df = pd.DataFrame(xtb_rows)

# %%
# Rank representatives by the same criteria the reward encodes
if not xtb_df.empty:
    def _lam_s(x):
        return np.exp(-0.5 * ((x - LAMBDA_TARGET_NM) / 60) ** 2) if pd.notna(x) else 0
    def _dip_s(x):
        return np.exp(-0.5 * ((x - 4.0) / 1.6) ** 2)
    def _sd_s(x):
        return np.exp(-0.5 * ((x - 0.0) / 6.0) ** 2)
    xtb_df["combined"] = (xtb_df["lam_est_nm"].map(_lam_s)
                          * xtb_df["dipole_water_au"].map(_dip_s)
                          * xtb_df["solv_diff_kcal"].map(_sd_s)) ** (1 / 3)
    xtb_df = xtb_df.sort_values("combined", ascending=False)
    print(xtb_df[["smiles", "lam_est_nm", "dipole_water_au", "solv_diff_kcal", "combined"]]
          .head(15).to_string(index=False))
    dft_candidates = xtb_df["smiles"].head(6).tolist()
else:
    dft_candidates = representatives[:6] if representatives else []
print(f"\n{len(dft_candidates)} molecules selected for DFT")

# %% [markdown]
# ## §14 — TD-DFT on Open Forms (PySCF B3LYP/6-31G*)
#
# We characterise the **open** form (the photoactive, coloured species). A
# best-effort closed form (`dasa_chem.open_to_closed`) is attempted for
# reference only — supply measured closed forms via the dataset when possible.

# %%
def tddft_open(smiles, n_states=6, basis="6-31g*", xc="b3lyp"):
    try:
        from pyscf import gto, dft, tddft
    except ImportError:
        print("PySCF not installed."); return None
    sys.path.insert(0, PLUGIN_COMP)
    from dasa_common import embed_3d
    m3d = embed_3d(Chem.MolFromSmiles(smiles))
    if m3d is None:
        return None
    conf = m3d.GetConformer()
    atoms = [f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.6f} "
             f"{conf.GetAtomPosition(a.GetIdx()).y:.6f} "
             f"{conf.GetAtomPosition(a.GetIdx()).z:.6f}" for a in m3d.GetAtoms()]
    mol = gto.M(atom="\n".join(atoms), basis=basis,
                charge=Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)), spin=0, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc; mf.conv_tol = 1e-8; mf.kernel()
    td = tddft.TDA(mf); td.nstates = n_states; td.kernel()
    exc = []
    for i, (e_ev, f) in enumerate(zip(td.e * 27.2114, td.oscillator_strength())):
        lam = 1240.0 / e_ev if e_ev > 0 else None
        char = "π→π* (bright)" if f >= 0.1 else ("n→π* / mixed" if f >= 0.01 else "n→π* (dark)")
        exc.append({"state": i + 1, "energy_eV": e_ev, "lambda_nm": lam,
                    "osc_strength": f, "character": char})
    return {"smiles": smiles, "ground_energy_Ha": mf.e_tot, "excitations": exc}


dft_results = []
if RUN_DFT and dft_candidates:
    n = min(4, len(dft_candidates))
    print(f"TD-DFT (B3LYP/6-31G*) on {n} open forms...\n")
    for i, smi in enumerate(dft_candidates[:n]):
        print(f"[{i+1}/{n}] {smi[:60]}")
        r = tddft_open(smi)
        if r is None:
            print("  → failed\n"); continue
        dft_results.append(r)
        bright = [e for e in r["excitations"] if e["osc_strength"] > 0.05]
        b = bright[0] if bright else r["excitations"][0]
        print(f"  brightest: λ={b['lambda_nm']:.0f} nm  f={b['osc_strength']:.3f}  {b['character']}")
        closed = dc.open_to_closed(smi)
        print(f"  closed-form (best-effort): {closed or 'n/a'}\n")
else:
    print("DFT skipped.")

# %% [markdown]
# ## §15 — UV-Vis Visualisation

# %%
def simulate_uvvis(exc, lo=350, hi=750, sigma=15, n=1000):
    lams = np.linspace(lo, hi, n)
    total = np.zeros(n)
    for e in exc:
        if e["lambda_nm"]:
            total += e["osc_strength"] * np.exp(-0.5 * ((lams - e["lambda_nm"]) / sigma) ** 2)
    return lams, total


if dft_results:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(400, 700, alpha=0.08, color="gray", label="visible")
    cmap = plt.cm.tab10
    for i, r in enumerate(dft_results):
        lams, total = simulate_uvvis(r["excitations"])
        peak = total.max() or 1.0
        ax.plot(lams, total / peak, color=cmap(i), lw=1.8, label=f"mol {i+1}")
    ax.set_xlim(350, 750); ax.set_xlabel("λ (nm)"); ax.set_ylabel("Normalised absorption")
    ax.set_title("DASA open-form simulated UV-Vis (TD-DFT B3LYP/6-31G*)")
    ax.legend(fontsize=8); plt.tight_layout()
    out = os.path.join(OUT_DIR, "dasa_uvvis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
    print(f"Saved → {out}")
else:
    print("No DFT results to plot.")

# %% [markdown]
# ## §16 — FarmShare Cluster Instructions
#
# ```bash
# ssh <sunetid>@rice.stanford.edu
# module load miniconda3 && conda activate reinvent4
# export PYTHONPATH=$HOME/REINVENT4_photochem/plugins:$PYTHONPATH
# export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4
#
# reinvent -d cuda:0 -l outputs_dasa/rl_stage2/stage2.log \
#          outputs_dasa/rl_stage2/stage2.toml
# ```
#
# Stage 2 (xTB + switchability) is the expensive one — run it on a GPU node with
# `--partition=normal --gres=gpu:1 --time=06:00:00 --mem=16G --cpus-per-task=4`.
