# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Photoswitch Discovery with REINVENT4
#
# **End-to-end pipeline: data cleaning → transfer learning → staged RL with quantum-chemistry-informed scoring**
#
# ## Overview
#
# This notebook walks through:
# 1. **FarmShare connection & environment setup** (SSH, Micromamba, REINVENT install)
# 2. **Data cleaning** — filtering `photoswitches.csv` + `fulldata.lambda_train.xlsx` into "good switches" for TL
# 3. **ML surrogate models** — training ChemProp models on cleaned data to predict λ_max and thermal half-life
# 4. **Quantum-chemistry reward plugins** — custom REINVENT scoring components using xTB (GFN2-xTB) HOMO-LUMO gaps and structural heuristics
# 5. **Transfer learning** — fine-tuning the REINVENT prior on clean photoswitch SMILES
# 6. **Staged RL** — three-stage reward setup (structural → ML-predicted → xTB)
# 7. **FarmShare batch job scripts** — ready-to-submit Slurm scripts
#
# ## Dependencies
#
# | Package | Purpose | Install |
# |---------|---------|---------|
# | `rdkit` | SMILES parsing, 3D geometry (ETKDG), descriptors | `conda install -c conda-forge rdkit` |
# | `xtb-python` | GFN2-xTB semiempirical QM (geometry opt, HOMO-LUMO gap) | `conda install -c conda-forge xtb-python` |
# | `pyscf` | TD-DFT excitation energies (Stage 3, optional) | `pip install pyscf` |
# | `chemprop` | Graph-neural-network property predictors | `pip install chemprop==1.5.2` |
# | `mordred` | 1800+ 2D/3D molecular descriptors | `pip install mordred` |
# | `scikit-learn` | Fast surrogate models (RandomForest, etc.) | `pip install scikit-learn` |
# | `mols2grid` | Molecule grids in Jupyter | `pip install mols2grid` |
# | `reinvent` | REINVENT4 (installed separately, see §1) | from repo |
# | `tensorboard` | TensorBoard logging | `pip install tensorboard` |
# | `jupytext` | Convert .py ↔ .ipynb | `pip install jupytext` |
#
# ### Why these specific QM libraries?
# - **GFN2-xTB** (via `xtb-python`): ~1–3 s/molecule, predicts HOMO-LUMO gap (proxy for λ_max), ground-state geometry, partial charges.  Validated for organic chromophores including azo compounds.
# - **sTDA-xTB**: excited-state spectra via the simplified Tamm-Dancoff approximation seeded with xTB orbitals.  Available as the `stda` binary (see §11).
# - **pyscf TD-DFT**: reference-quality UV/Vis spectra (B3LYP/6-31G*) for top-ranked candidates only.  Not used in the RL loop.

# ## §1 — FarmShare Connection & Environment Setup
#
# > **Run these commands in your local terminal, not inside this notebook.**
#
# ### 1.1 SSH into FarmShare
#
# ```bash
# ssh YOUR_SUNETID@login.farmshare.stanford.edu
# # You will be prompted for your password + Duo two-factor auth.
# # Your first login creates the Slurm account needed for batch jobs.
# ```
#
# FarmShare uses a load-balanced login node (`login.farmshare.stanford.edu`).
# The ED25519 fingerprint you should see the first time is:
# `SHA256:bKb1Znir/1tOg+TMyALDYWeK0lclsulriDN8aOvWteU`
#
# ### 1.2 Create the project layout
#
# ```bash
# mkdir -p ~/projects/reinvent_photoswitch/{code,data,configs,logs,models/priors,outputs,scripts,plugins/reinvent_plugins/components}
# cd ~/projects/reinvent_photoswitch
# ```
#
# ### 1.3 Build the Python environment
#
# **On FarmShare** — Micromamba is available via `module load`:
# ```bash
# module load micromamba
# micromamba create -n reinvent4 python=3.10 -y
# micromamba activate reinvent4
# ```
#
# **Locally (macOS / Linux)** — use your existing conda/mamba install:
# ```bash
# conda create -n reinvent4 python=3.10 -y
# conda activate reinvent4
# # or: mamba create -n reinvent4 python=3.10 -y && mamba activate reinvent4
# ```
#
# **Install REINVENT4** (same on both platforms):
# ```bash
# git clone https://github.com/MolecularAI/REINVENT4.git code/REINVENT4
# cd code/REINVENT4
# python install.py cpu          # CPU-first; switch to 'gpu' once pipeline is stable
# reinvent --help                # verify install
# cd ~/projects/reinvent_photoswitch
# ```
#
# ### 1.4 Install extra dependencies
#
# **On FarmShare:**
# ```bash
# micromamba install -n reinvent4 -c conda-forge xtb-python rdkit -y
# pip install setuptools pyscf chemprop==1.5.2 mordred mols2grid jupytext tensorboard
# ```
#
# **Locally (macOS / Linux):**
# ```bash
# conda install -c conda-forge xtb-python rdkit -y
# pip install setuptools pyscf chemprop==1.5.2 mordred mols2grid jupytext tensorboard
# ```
#
# > **Note on `setuptools`**: conda updates sometimes remove `setuptools`, which
# > breaks both ChemProp (via `hyperopt → pkg_resources`) and REINVENT's plugin
# > importer.  Always include `pip install setuptools` in environment setup.
#
# > **Note on chemprop**: REINVENT4 uses the v1 ChemProp API.
# > The latest compatible release is `1.5.2` — do not install v2+.
#
# ### 1.5 Upload your data
#
# From your **local** machine (run in a separate terminal):
# ```bash
# scp notebooks/data/photoswitches.csv YOUR_SUNETID@login.farmshare.stanford.edu:~/projects/reinvent_photoswitch/data/
# scp notebooks/data/fulldata.lambda_train.xlsx YOUR_SUNETID@login.farmshare.stanford.edu:~/projects/reinvent_photoswitch/data/
# scp -r notebooks/photoswitch_discovery.ipynb YOUR_SUNETID@login.farmshare.stanford.edu:~/projects/reinvent_photoswitch/
# ```
#
# ### 1.6 Get a REINVENT prior model
#
# Download `FS_Ro5_10M.model` from the REINVENT4 Zenodo record and place it
# in the **project root** (same folder as `reinvent_plugins/`, `notebooks/`, etc.):
#
# ```bash
# # Local:
# cp ~/Downloads/FS_Ro5_10M.model /path/to/REINVENT4_photochem/
#
# # FarmShare:
# scp FS_Ro5_10M.model <sunetid>@rice.stanford.edu:~/projects/reinvent_photoswitch/
# ```
#
# The notebook's `PRIOR_FILE` variable (§9) points here by default.
#
# ### 1.7 Open JupyterLab (optional — for interactive work on FarmShare)
#
# ```bash
# # Start an interactive compute node first (do NOT run Jupyter on a login node)
# srun --partition=interactive --qos=interactive --pty bash
# module load micromamba
# micromamba run -n reinvent4 jupyter lab --no-browser --port=8888 &
#
# # On your local machine, tunnel the port:
# ssh -L 8888:localhost:8888 YOUR_SUNETID@login.farmshare.stanford.edu
# # Then open http://localhost:8888 in your browser.
# ```

# ## §2 — Imports & Project Paths

# +
import os
import sys
import shutil
import re
import subprocess
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, inchi, Draw,
    rdMolDescriptors, rdchem
)
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    import mols2grid
    HAS_MOLS2GRID = True
except ImportError:
    HAS_MOLS2GRID = False
    print("mols2grid not installed — molecule grids disabled")

# ── Critical: setuptools / pkg_resources ────────────────────────────────────
# conda updates sometimes drop setuptools.  This breaks ChemProp (hyperopt)
# and REINVENT's own plugin importer before a single cell runs.
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    # conda-forge setuptools ≥ 80 drops pkg_resources from the distribution.
    # Downgrade to 68.2.2, the last stable release that still bundles it.
    print("⚠  pkg_resources missing — installing setuptools==68.2.2 …")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall",
         "-q", "setuptools==68.2.2"],
        check=True,
    )
    import importlib
    importlib.invalidate_caches()
    try:
        import pkg_resources  # noqa: F401
        print("✓  setuptools 68.2.2 installed — please Restart Kernel and Re-run All Cells.")
        raise SystemExit(0)   # stop here so fresh imports load correctly
    except ModuleNotFoundError:
        raise RuntimeError(
            "\n\nsetuptools was installed but pkg_resources is still not importable.\n"
            "The pip install wrote to a different Python than this kernel.\n"
            "Fix (run in a terminal, then restart the kernel):\n"
            "  /Users/hakeemshindy/miniconda3/envs/reinvent4/bin/pip install "
            "--force-reinstall setuptools==68.2.2\n"
        )

# ── Paths (adjust PROJ_ROOT for FarmShare) ──────────────────────────────────
# Local (this repo):
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(""), ".."))
DATA_DIR  = os.path.join(PROJ_ROOT, "notebooks", "data")
OUT_DIR   = os.path.join(PROJ_ROOT, "outputs");  os.makedirs(OUT_DIR, exist_ok=True)
PLUGIN_DIR = os.path.join(PROJ_ROOT, "plugins"); os.makedirs(PLUGIN_DIR, exist_ok=True)
PLUGIN_COMP = os.path.join(PLUGIN_DIR, "reinvent_plugins", "components")
os.makedirs(PLUGIN_COMP, exist_ok=True)

# On FarmShare, change PROJ_ROOT to:
# PROJ_ROOT = os.path.expanduser("~/projects/reinvent_photoswitch")

print(f"Project root : {PROJ_ROOT}")
print(f"Data dir     : {DATA_DIR}")
print(f"Plugin dir   : {PLUGIN_DIR}")

# +
# ── Run-control flags ────────────────────────────────────────────────────────
# Set any flag to False to skip that stage.
# Useful for re-running from a checkpoint or for dry-runs.
#
# ⏱  Rough wall-clock estimates on Apple Silicon Mac (CPU only):
#   RUN_CHEMPROP  : ~10-30 min total (λ_max + t½ models)
#   RUN_TL        : ~30-90 min  (50 epochs, ~350 SMILES)
#   RUN_RL1       : ~15-30 min  (400 steps, structural scoring only)
#   RUN_RL2       : ~2-6 hours  (600 steps, + ChemProp batch scoring)
#   RUN_RL3       : ~8-24 hours (200 steps, + xTB per molecule — skip on Mac)
#
# On FarmShare with a GPU node, divide all times by ~5–10×.

RUN_CHEMPROP = True   # train λ_max and log(t½) surrogate models
RUN_TL       = True   # transfer learning
RUN_RL1      = True   # Stage 1 RL (structural filter)
RUN_RL2      = True   # Stage 2 RL (+ ChemProp property scores)
RUN_RL3      = False  # Stage 3 RL (+ xTB HOMO-LUMO; slow, needs xtb-python)

# "cpu" locally; "cuda:0" on FarmShare GPU node
DEVICE = "cpu"


def run_reinvent(config_file, log_file, device=None):
    """
    Run REINVENT and stream its output line-by-line to the notebook cell.
    Sets PYTHONPATH so custom plugins are found.
    Returns exit code (0 = success).

    KMP_DUPLICATE_LIB_OK=TRUE suppresses the macOS OpenMP crash (exit -6) that
    occurs when PyTorch, NumPy, and SciPy each bundle their own libomp.dylib.
    """
    _device = device or DEVICE
    _env = {
        **os.environ,
        "PYTHONPATH":             f"{PLUGIN_DIR}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "KMP_DUPLICATE_LIB_OK":  "TRUE",   # macOS OpenMP duplicate-library fix
        "OMP_NUM_THREADS":        "1",      # prevent runaway thread spawning on Mac
    }
    cmd = ["reinvent", "-d", _device, "-l", log_file, config_file]
    print(f"▶ {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=_env,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        status = "✓ Done" if proc.returncode == 0 else f"✗ Exit code {proc.returncode}"
        print(f"\n{status}")
        return proc.returncode
    except FileNotFoundError:
        print("[ERROR] 'reinvent' command not found — activate the reinvent4 conda env.")
        return -1
# -

# ## §3 — Load & Explore the Raw Data

# +
# ── photoswitches.csv  (Josch Helfferich dataset, ~405 molecules) ────────────
ps_raw = pd.read_csv(os.path.join(DATA_DIR, "photoswitches.csv"), index_col=0)

# Rename the most important columns for convenience
RENAME_PS = {
    "SMILES":                                     "smiles",
    "rate of thermal isomerisation from Z-E in s-1": "therm_rate_s",
    "Solvent used for thermal isomerisation rates":  "therm_solvent",
    "Z PhotoStationaryState":                     "PSS_Z",
    "E PhotoStationaryState":                     "PSS_E",
    "E isomer pi-pi* wavelength in nm":           "lam_E_pipi",
    "Extinction":                                 "ext_E_pipi",
    "E isomer n-pi* wavelength in nm":            "lam_E_npi",
    "Extinction coefficient in M-1 cm-1":         "ext_E_npi",
    "Z isomer pi-pi* wavelength in nm":           "lam_Z_pipi",
    "Z isomer n-pi* wavelength in nm":            "lam_Z_npi",
    "Wiberg index":                               "wiberg_idx",
    "E-Z irradiation wavelength in nm":           "irr_EtoZ",
    "Z-E irradiation wavelength":                 "irr_ZtoE",
    "Irradiation solvent":                        "irr_solvent",
}
ps = ps_raw.rename(columns=RENAME_PS)

# Compute t1/2 from first-order rate: t1/2 = ln(2)/k  (seconds)
ps["t12_s"] = np.log(2) / ps["therm_rate_s"]

print(f"photoswitches.csv  : {len(ps)} molecules, {ps.shape[1]} columns")
print(f"  λ_E_pipi range   : {ps['lam_E_pipi'].min():.0f}–{ps['lam_E_pipi'].max():.0f} nm  (n={ps['lam_E_pipi'].notna().sum()})")
print(f"  PSS_Z range      : {ps['PSS_Z'].min():.0f}–{ps['PSS_Z'].max():.0f} %      (n={ps['PSS_Z'].notna().sum()})")
print(f"  t1/2 range       : {ps['t12_s'].min():.1e}–{ps['t12_s'].max():.1e} s   (n={ps['t12_s'].notna().sum()})")

# +
# ── fulldata.lambda_train.xlsx  (enlarged training set, ~718 molecules) ────────
lt_raw = pd.read_excel(os.path.join(DATA_DIR, "fulldata.lambda_train.xlsx"))
lt = lt_raw.rename(columns={"SMILES": "smiles", "lambda": "lam_E_pipi", "t12": "t12_s"})
lt = lt[["smiles", "lam_E_pipi", "t12_s", "logt12", "solvent"]].copy()
lt["source"] = "lambda_train"

print(f"fulldata.lambda_train : {len(lt)} molecules")
print(f"  λ range  : {lt['lam_E_pipi'].min():.0f}–{lt['lam_E_pipi'].max():.0f} nm")

# +
# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(ps["lam_E_pipi"].dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(400, color="red", linestyle="--", label="400 nm")
axes[0].set_xlabel("E isomer π→π* λ_max (nm)"); axes[0].set_ylabel("Count")
axes[0].set_title("photoswitches.csv — λ_max (E)"); axes[0].legend()

axes[1].hist(np.log10(ps["t12_s"].dropna()), bins=30, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].axvline(np.log10(3600),    color="blue",  linestyle="--", label="1 h")
axes[1].axvline(np.log10(3.15e13), color="green", linestyle="--", label="1 Myr")
axes[1].set_xlabel("log₁₀(t₁/₂ / s)"); axes[1].set_ylabel("Count")
axes[1].set_title("photoswitches.csv — thermal t₁/₂"); axes[1].legend()

axes[2].hist(lt["lam_E_pipi"].dropna(), bins=30, color="#55A868", edgecolor="white", alpha=0.85)
axes[2].axvline(400, color="red", linestyle="--", label="400 nm")
axes[2].set_xlabel("λ_max (nm)"); axes[2].set_ylabel("Count")
axes[2].set_title("fulldata.lambda_train — λ_max"); axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "data_distributions.png"), dpi=150)
plt.show()
# -

# ## §4 — Data Cleaning: "Good Switches" Criteria
#
# We want molecules that are:
#
# | Criterion | Threshold | Rationale |
# |-----------|-----------|-----------|
# | Valid SMILES | — | parseable by RDKit |
# | Visible-light responsive | λ_max ≥ 380 nm | accessible without UV sources |
# | High PSS | max(PSS_E, PSS_Z) ≥ 65 % | meaningful E↔Z conversion |
# | Bistable half-life | 1 h ≤ t₁/₂ ≤ 100 yr | slow enough to be metastable |
# | No forbidden substructures | SMARTS below | remove reactive / unstable groups |
# | No metals or radicals | — | synthetic tractability |
# | Deduplication | by InChIKey | remove redundant structures |
#
# ### Photoswitch scaffold motifs (used for TL corpus — broader)
# We keep any molecule with N=N (azo), C=N (imine/hydrazone), or spiropyran motifs.
# For the stricter "good switch" set used to train scoring predictors, we also require the above property thresholds.

# +
# ── Forbidden SMARTS (reactive / unstable) ──────────────────────────────────
FORBIDDEN_SMARTS_LIST = [
    # generic large rings / problematic fragments
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    # peroxides, disulfides, charged carbon, strained bonds
    "[#8][#8]", "[#6;+]", "[#16][#16]", "C#C",
    # reactive heteroatom combos
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",        # hydrazines (allow aromatic N-N in azo)
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    # metals — any typical transition metal
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]",
    # radicals / unusual valences
    "[#6X5]", "[#7X4;!H0;v4]",
]

FORBIDDEN_TEMPLATES = []
for sma in FORBIDDEN_SMARTS_LIST:
    t = Chem.MolFromSmarts(sma)
    if t:
        FORBIDDEN_TEMPLATES.append((sma, t))

# ── Photoswitch scaffold SMARTS (at least one required) ─────────────────────
PHOTOSWITCH_SMARTS = {
    "azo":           Chem.MolFromSmarts("[#6]/N=N/[#6]"),
    "azo_cis":       Chem.MolFromSmarts("[#6]\\N=N\\[#6]"),
    "azo_any":       Chem.MolFromSmarts("[#6]N=N[#6]"),
    "azomethine":    Chem.MolFromSmarts("[#6]/C=N/[#6]"),   # imines / hydrazones
    "diarylethene":  Chem.MolFromSmarts("c1cc(cc1)-c2cc(-c3ccccc3)c(=O)[nH]2"),
    "spiropyran":    Chem.MolFromSmarts("C1(OC2=CC=CC=C2)=CC=CC=C1"),
}


def is_photoswitch_scaffold(mol):
    """Return True if molecule contains a recognised photoswitch motif."""
    for name, tmpl in PHOTOSWITCH_SMARTS.items():
        if tmpl and mol.HasSubstructMatch(tmpl):
            return True
    return False


def has_forbidden_substructure(mol):
    """Return True if any forbidden SMARTS matches."""
    for sma, tmpl in FORBIDDEN_TEMPLATES:
        if mol.HasSubstructMatch(tmpl):
            return True
    return False


def canonical_smiles(smiles):
    """Standardise, neutralise, and return canonical SMILES. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Standardise
        clean = rdMolStandardize.Cleanup(mol)
        # Remove salts — keep largest fragment
        remover = rdMolStandardize.LargestFragmentChooser()
        clean = remover.choose(clean)
        # Neutralise
        neutralizer = rdMolStandardize.Uncharger()
        clean = neutralizer.uncharge(clean)
        return Chem.MolToSmiles(clean)
    except Exception:
        return None


def inchikey(smiles):
    """Return InChIKey for deduplication."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi_str = inchi.MolToInchi(mol)
    if inchi_str is None:
        return None
    return inchi.InchiToInchiKey(inchi_str)

# +
# ── Clean photoswitches.csv ──────────────────────────────────────────────────

def clean_photoswitches(df, strict=False):
    """
    Filter the photoswitches dataframe.

    strict=False  → broad corpus for transfer learning (SMILES + scaffold check only)
    strict=True   → predictor-training set: must have a measured λ_max (any value);
                    PSS / t12 / Wiberg filters applied only when measurements exist.
                    λ_max threshold is intentionally absent here — the ChemProp model
                    must learn the full wavelength distribution; the RL reward handles
                    the visible-light preference at scoring time.
    """
    records = []
    for _, row in df.iterrows():
        smiles_raw = str(row.get("smiles", "")).strip()
        canon = canonical_smiles(smiles_raw)
        if canon is None:
            continue

        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            continue

        # Must be a photoswitch scaffold
        if not is_photoswitch_scaffold(mol):
            continue

        # No forbidden groups
        if has_forbidden_substructure(mol):
            continue

        if strict:
            lam = row.get("lam_E_pipi")
            t12 = row.get("t12_s")
            pss_z = row.get("PSS_Z")
            pss_e = row.get("PSS_E")
            wb    = row.get("wiberg_idx")

            # Must have a measured λ_max (any wavelength — no threshold)
            if pd.isna(lam):
                continue

            # PSS: only filter when both values are actually measured
            pss_z_known = pd.notna(pss_z)
            pss_e_known = pd.notna(pss_e)
            if pss_z_known or pss_e_known:
                pss_max = max(
                    pss_z if pss_z_known else 0,
                    pss_e if pss_e_known else 0,
                )
                if pss_max < 65:
                    continue

            # Bistability: only filter when t12 is measured
            if pd.notna(t12) and not (3_600 <= t12 <= 3.15e9):
                continue

            # Wiberg index sanity check (only when measured)
            if pd.notna(wb) and not (1.3 <= wb <= 2.1):
                continue

        t12_val = row.get("t12_s")
        logt12_val = np.log10(t12_val) if pd.notna(t12_val) and t12_val > 0 else None

        ik = inchikey(canon)
        records.append({
            "smiles":      canon,
            "inchikey":    ik,
            "lam_E_pipi": row.get("lam_E_pipi"),
            "lam_Z_pipi": row.get("lam_Z_pipi"),
            "lam_E_npi":  row.get("lam_E_npi"),
            "PSS_Z":      row.get("PSS_Z"),
            "PSS_E":      row.get("PSS_E"),
            "t12_s":      t12_val,
            "logt12":     logt12_val,
            "wiberg_idx": row.get("wiberg_idx"),
            "irr_EtoZ":   row.get("irr_EtoZ"),
            "source":     "photoswitches_csv",
        })

    out = pd.DataFrame(records).drop_duplicates(subset="inchikey")
    return out


ps_broad  = clean_photoswitches(ps, strict=False)  # TL corpus
ps_strict = clean_photoswitches(ps, strict=True)   # predictor training

print(f"photoswitches.csv  broad  (TL corpus)      : {len(ps_broad)}")
print(f"photoswitches.csv  strict (predictor train) : {len(ps_strict)}")

# +
# ── Clean fulldata.lambda_train.xlsx ────────────────────────────────────────

def clean_lambda_train(df, lam_min=300, t12_min=3600):
    records = []
    for _, row in df.iterrows():
        smiles_raw = str(row.get("smiles", "")).strip()
        canon = canonical_smiles(smiles_raw)
        if canon is None:
            continue

        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            continue

        if not is_photoswitch_scaffold(mol):
            continue

        if has_forbidden_substructure(mol):
            continue

        lam = row.get("lam_E_pipi")
        t12 = row.get("t12_s")

        if pd.isna(lam) or lam < lam_min:
            continue

        # t12 filter only when value is available
        if pd.notna(t12) and t12 < t12_min:
            continue

        ik = inchikey(canon)
        records.append({
            "smiles":      canon,
            "inchikey":    ik,
            "lam_E_pipi": lam,
            "t12_s":      t12,
            "logt12":     row.get("logt12"),
            "solvent":    row.get("solvent"),
            "source":     "lambda_train",
        })

    return pd.DataFrame(records).drop_duplicates(subset="inchikey")


lt_clean = clean_lambda_train(lt, lam_min=300, t12_min=3600)
print(f"lambda_train clean : {len(lt_clean)}")

# +
# ── Merge both datasets, deduplicate ────────────────────────────────────────
# Columns present in both: smiles, inchikey, lam_E_pipi, t12_s, source
SHARED = ["smiles", "inchikey", "lam_E_pipi", "t12_s", "logt12", "source"]

combined_strict = pd.concat(
    [ps_strict[SHARED],
     lt_clean[SHARED]],
    ignore_index=True
).drop_duplicates(subset="inchikey")

combined_broad = pd.concat(
    [ps_broad[["smiles", "inchikey", "source"]],
     lt_clean[["smiles", "inchikey", "source"]]],
    ignore_index=True
).drop_duplicates(subset="inchikey")

print(f"Strict set (TL train + predictor training) : {len(combined_strict)}")
print(f"Broad  set (TL corpus only)                : {len(combined_broad)}")
print(f"\nλ_max distribution in strict set:")
print(combined_strict["lam_E_pipi"].describe())

# +
# ── Scaffold-aware 80/20 train/val split ────────────────────────────────────
# Group by Murcko scaffold and put whole scaffold groups into train or val.
# This prevents data leakage between train and validation.

def scaffold_split(df, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    scaffolds = {}
    for idx, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = "__generic__"
        scaffolds.setdefault(scaf, []).append(idx)

    scaffold_groups = list(scaffolds.values())
    rng.shuffle(scaffold_groups)

    n_total = sum(len(g) for g in scaffold_groups)
    n_train_target = int(n_total * train_frac)

    train_idx, val_idx = [], []
    n_train_so_far = 0
    for group in scaffold_groups:
        if n_train_so_far < n_train_target:
            train_idx.extend(group)
            n_train_so_far += len(group)
        else:
            val_idx.extend(group)

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# Use combined_broad for TL (we only need SMILES)
tl_data = combined_broad[["smiles"]].dropna()
tl_train, tl_val = scaffold_split(tl_data, train_frac=0.8)

print(f"TL train : {len(tl_train)} molecules")
print(f"TL val   : {len(tl_val)} molecules")

# +
# ── Save TL files ────────────────────────────────────────────────────────────
TL_TRAIN_SMI = os.path.join(OUT_DIR, "photoswitch_tl_train.smi")
TL_VAL_SMI   = os.path.join(OUT_DIR, "photoswitch_tl_val.smi")
STRICT_CSV   = os.path.join(OUT_DIR, "photoswitch_strict.csv")

tl_train["smiles"].to_csv(TL_TRAIN_SMI, index=False, header=False)
tl_val["smiles"].to_csv(TL_VAL_SMI, index=False, header=False)
combined_strict.to_csv(STRICT_CSV, index=False)

print(f"Saved TL train  : {TL_TRAIN_SMI}")
print(f"Saved TL val    : {TL_VAL_SMI}")
print(f"Saved strict set: {STRICT_CSV}")
# -

# ## §5 — Quantum Chemistry Features with xTB (GFN2-xTB)
#
# **xTB** (extended tight-binding) from the Grimme group provides semiempirical QM calculations.
# GFN2-xTB is accurate enough for:
# - Geometry optimisation (~1–3 s/molecule on CPU)
# - HOMO and LUMO orbital energies (→ HOMO-LUMO gap as λ_max proxy)
# - Partial charges and bond orders (→ Wiberg bond index)
# - Total energy differences between conformers / isomers
#
# The HOMO-LUMO gap (optical gap) correlates with λ_max via:
#
# ```
# E_gap (eV) ≈ 0.75 × (HOMO-LUMO gap from GFN2-xTB)
# λ_max (nm) ≈ 1240 / E_gap
# ```
#
# The correction factor 0.75 is empirically calibrated for azo-type switches.
# For RL scoring we skip this conversion and directly score the gap.
#
# **In the RL loop (Stage 1 & 2)** we use pre-trained ChemProp surrogates (fast).
# **In Stage 3 / post-processing** we run actual xTB calculations on top candidates.

# +
def mol_to_xtb_inputs(smiles):
    """
    Prepare inputs for xTB from SMILES.
    Returns (atomic_numbers, coords_bohr) or (None, None) on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = Chem.AddHs(mol)

    # 3D embedding with ETKDGv3
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fallback to distance geometry
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Pre-optimize geometry with MMFF94
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass

    conf = mol.GetConformer()
    positions = conf.GetPositions()  # Å
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=int)

    ANGSTROM_TO_BOHR = 1.8897259886
    coords_bohr = positions * ANGSTROM_TO_BOHR

    return atomic_numbers, coords_bohr


def xtb_homo_lumo_gap(smiles, method="GFN2xTB"):
    """
    Compute HOMO-LUMO gap (eV) for a molecule using GFN2-xTB.

    Returns float (eV) or None on failure.

    The gap is derived from orbital eigenvalues and occupations rather than
    a dedicated API call (get_homo_lumo_gap is absent in xtb-python ≥ 22.x).
    Fermi-smearing artifacts are handled by using occ > 0.5 as the threshold.

    Requires: conda install -c conda-forge xtb-python
    """
    try:
        from xtb.interface import Calculator, Param

        PARAM_MAP = {
            "GFN0xTB": Param.GFN0xTB,
            "GFN1xTB": Param.GFN1xTB,
            "GFN2xTB": Param.GFN2xTB,
            "GFNFF":   Param.GFNFF,
        }
        param = PARAM_MAP.get(method, Param.GFN2xTB)

        atomic_numbers, coords_bohr = mol_to_xtb_inputs(smiles)
        if atomic_numbers is None:
            return None

        calc = Calculator(param, atomic_numbers, coords_bohr)
        calc.set_verbosity(0)
        res = calc.singlepoint()

        evals = res.get_orbital_eigenvalues()   # Hartree
        occs  = res.get_orbital_occupations()   # 0.0 or 2.0 (with small Fermi smearing)

        # Use 0.5 threshold to ignore Fermi-smearing residuals (~1e-12)
        occupied   = evals[occs > 0.5]
        unoccupied = evals[occs <= 0.5]

        if len(occupied) == 0 or len(unoccupied) == 0:
            return None

        homo_ha = occupied[-1]
        lumo_ha = unoccupied[0]
        gap_ev  = (lumo_ha - homo_ha) * 27.2114  # Hartree → eV
        return gap_ev if gap_ev > 0 else None

    except ImportError:
        print("xtb-python not installed. Run: conda install -c conda-forge xtb-python")
        return None
    except Exception as e:
        print(f"  [xTB error] {e}")
        return None


# Empirical calibration of GFN2-xTB HOMO-LUMO gap vs experimental λ_max
# (azobenzene 1.36 eV→320nm, 4-aminoazobenzene 1.34 eV→385nm,
#  methyl orange 1.25 eV→460nm). Average correction ~2.5.
# Note: GFN2-xTB gap is NOT a precise predictor of λ_max; it is useful
# for relative ranking and conjugation assessment in the RL reward.
_XTB_CORRECTION = 2.5


def xtb_gap_to_lambda_estimate(gap_eV, correction=_XTB_CORRECTION):
    """
    Rough estimate of λ_max (nm) from GFN2-xTB HOMO-LUMO gap.

    Uses an empirical correction factor calibrated on azo photoswitches
    (correction ≈ 2.5).  Accuracy is ±100 nm; treat as a ranking proxy
    rather than a quantitative prediction.
    Returns λ_max in nm, or None if gap ≤ 0.
    """
    optical_gap = gap_eV * correction
    if optical_gap <= 0:
        return None
    return 1240.0 / optical_gap

# +
# ── xTB smoke test ────────────────────────────────────────────────────────────
# If xtb-python is already installed but the cells below still show N/A it means
# the library was installed *after* the kernel started.
# Fix: Kernel ▸ Restart Kernel and Re-run All Cells (or just re-run from here).
#
# Install (once, in the reinvent4 env):
#   conda install -c conda-forge xtb-python
# ─────────────────────────────────────────────────────────────────────────────

DEMO_SMILES = {
    "azobenzene":        "c1ccc(/N=N/c2ccccc2)cc1",
    "4-aminoazobenzene": "Nc1ccc(/N=N/c2ccccc2)cc1",
    "methyl orange":     "CN(C)c1ccc(/N=N/c2ccc(cc2)S(=O)(=O)[O-])cc1",
}

print(f"{'Molecule':<25} {'Gap (eV)':>9} {'λ_est (nm)':>12} {'λ_exp (nm)':>12}")
print("-" * 62)
EXP_LAMBDA = {"azobenzene": 320, "4-aminoazobenzene": 385, "methyl orange": 460}
for name, smi in DEMO_SMILES.items():
    gap = xtb_homo_lumo_gap(smi)
    if gap is not None:
        lam = xtb_gap_to_lambda_estimate(gap)
        print(f"{name:<25} {gap:>9.3f} {lam:>12.0f} {EXP_LAMBDA[name]:>12}")
    else:
        print(f"{name:<25} {'N/A':>9} {'N/A':>12} {EXP_LAMBDA[name]:>12}")
        print("  → Install xtb-python to enable xTB calculations")

# +
# ── Batch xTB screening (Stage 3 post-processing) ────────────────────────────
def batch_xtb_screen(smiles_list, target_lam_min=400, target_lam_max=650,
                     n_jobs=1, verbose=True):
    """
    Screen a list of SMILES with GFN2-xTB.
    Returns a DataFrame with: smiles, gap_eV, lam_est_nm, score.

    score = Gaussian centred at midpoint of [target_lam_min, target_lam_max],
            width = half the range.
    """
    mid   = (target_lam_min + target_lam_max) / 2
    width = (target_lam_max - target_lam_min) / 2

    rows = []
    for i, smi in enumerate(smiles_list):
        if verbose and i % 20 == 0:
            print(f"  Processing {i}/{len(smiles_list)}...")
        gap = xtb_homo_lumo_gap(smi)
        if gap is None:
            rows.append({"smiles": smi, "gap_eV": np.nan, "lam_est_nm": np.nan, "xtb_score": 0.0})
            continue
        lam = xtb_gap_to_lambda_estimate(gap)
        if lam is None:
            score = 0.0
        else:
            score = float(np.exp(-0.5 * ((lam - mid) / width) ** 2))
        rows.append({"smiles": smi, "gap_eV": gap, "lam_est_nm": lam, "xtb_score": score})

    return pd.DataFrame(rows)


# Uncomment to run on the strict set (requires xtb-python):
# xtb_df = batch_xtb_screen(combined_strict["smiles"].dropna().tolist()[:50])
# print(xtb_df.sort_values("xtb_score", ascending=False).head(10))
# -

# ## §6 — TD-DFT with PySCF (Optional, Top Candidates Only)
#
# For the final shortlist of generated molecules, run TD-DFT/B3LYP to get:
# - S₀→S₁ and S₀→S₂ vertical excitation energies (→ λ_abs)
# - Oscillator strengths (→ extinction coefficient proxy)
# - Triplet energy T₁ (→ thermal isomerisation mechanism)
#
# This takes ~1–10 min per molecule; never run in the RL loop.

# +
def tddft_excitations(smiles, n_states=5, functional="b3lyp", basis="6-31g*"):
    """
    Run TD-DFT on a molecule using PySCF.

    Returns list of (excitation_energy_eV, oscillator_strength, lambda_nm)
    for the n_states lowest singlet excited states.

    Requires: pip install pyscf
    """
    try:
        from pyscf import gto, scf, tddft as pyscf_tddft
        import tempfile

        atomic_numbers, coords_bohr = mol_to_xtb_inputs(smiles)
        if atomic_numbers is None:
            return None

        mol_pyscf = Chem.MolFromSmiles(smiles)
        mol_pyscf = Chem.AddHs(mol_pyscf)
        AllChem.EmbedMolecule(mol_pyscf, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol_pyscf)

        conf = mol_pyscf.GetConformer()

        # Build PySCF atom string from RDKit conformer
        atom_lines = []
        for atom, pos in zip(mol_pyscf.GetAtoms(), conf.GetPositions()):
            sym = atom.GetSymbol()
            atom_lines.append(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")

        pyscf_mol = gto.Mole()
        pyscf_mol.atom = "; ".join(atom_lines)
        pyscf_mol.basis = basis
        pyscf_mol.verbose = 0
        pyscf_mol.build()

        mf = scf.RKS(pyscf_mol)
        mf.xc = functional
        mf.run()

        td = pyscf_tddft.TDDFT(mf)
        td.nstates = n_states
        td.run()

        results = []
        osc_strengths = td.oscillator_strength()
        for i, (e_au, f) in enumerate(zip(td.e, osc_strengths)):
            e_eV = e_au * 27.2114
            lam_nm = 1240.0 / e_eV
            results.append({"state": i + 1, "E_eV": e_eV, "lambda_nm": lam_nm, "osc_strength": f})

        return results

    except ImportError:
        print("pyscf not installed. Run: pip install pyscf")
        return None
    except Exception as e:
        print(f"TD-DFT failed: {e}")
        return None


# Example (requires pyscf):
# results = tddft_excitations("c1ccc(/N=N/c2ccccc2)cc1")
# if results:
#     for r in results:
#         print(f"S{r['state']}: {r['lambda_nm']:.0f} nm, f={r['osc_strength']:.4f}")
# -

# ## §7 — Train ChemProp Surrogate Models for RL Scoring
#
# We train two ChemProp (v1) models:
# - **λ_max model**: predicts π→π* absorption wavelength (nm)
# - **log(t₁/₂) model**: predicts log₁₀ of thermal half-life (s)
#
# These are used as scoring components in Stage 2 RL — fast enough for batched scoring.

# +
CHEMPROP_LAMBDA_DIR = os.path.join(OUT_DIR, "chemprop_lambda")
CHEMPROP_T12_DIR    = os.path.join(OUT_DIR, "chemprop_t12")
os.makedirs(CHEMPROP_LAMBDA_DIR, exist_ok=True)
os.makedirs(CHEMPROP_T12_DIR,    exist_ok=True)

# ── Prepare λ_max training file ──────────────────────────────────────────────
lam_data = combined_strict.dropna(subset=["lam_E_pipi"])[["smiles", "lam_E_pipi"]].copy()
lam_data = lam_data.rename(columns={"lam_E_pipi": "lambda_max"})
lam_data_train = lam_data.sample(frac=0.8, random_state=42)
lam_data_val   = lam_data.drop(lam_data_train.index)

LAM_TRAIN_CSV = os.path.join(CHEMPROP_LAMBDA_DIR, "train.csv")
LAM_VAL_CSV   = os.path.join(CHEMPROP_LAMBDA_DIR, "val.csv")
lam_data_train.to_csv(LAM_TRAIN_CSV, index=False)
lam_data_val.to_csv(LAM_VAL_CSV, index=False)

print(f"λ_max ChemProp  train={len(lam_data_train)}, val={len(lam_data_val)}")

# ── Prepare log(t1/2) training file ─────────────────────────────────────────
t12_data = combined_strict.dropna(subset=["t12_s"]).copy()
t12_data["logt12"] = np.log10(t12_data["t12_s"].clip(lower=1))
t12_data = t12_data[["smiles", "logt12"]]
t12_data_train = t12_data.sample(frac=0.8, random_state=42)
t12_data_val   = t12_data.drop(t12_data_train.index)

T12_TRAIN_CSV = os.path.join(CHEMPROP_T12_DIR, "train.csv")
T12_VAL_CSV   = os.path.join(CHEMPROP_T12_DIR, "val.csv")
t12_data_train.to_csv(T12_TRAIN_CSV, index=False)
t12_data_val.to_csv(T12_VAL_CSV, index=False)

print(f"log(t₁/₂) ChemProp  train={len(t12_data_train)}, val={len(t12_data_val)}")

# +
# ── Train ChemProp models ─────────────────────────────────────────────────────
# Requires: pip install chemprop==1.5.2

def train_chemprop(train_csv, val_csv, model_dir, target_col,
                   epochs=50, batch_size=50, hidden_size=300, depth=3):
    """
    Train a ChemProp (v1) regression model via subprocess.
    Uses the ``chemprop_train`` console-script entry point installed by
    chemprop 1.5.x (``python -m chemprop.train`` does not work because the
    ``train`` sub-package has no ``__main__.py``).
    """
    # Look for chemprop_train next to the running Python to avoid picking up
    # a stale copy from a different Python installation.
    env_bin = os.path.dirname(sys.executable)
    chemprop_bin = os.path.join(env_bin, "chemprop_train")
    if not os.path.isfile(chemprop_bin):
        chemprop_bin = shutil.which("chemprop_train")
    if chemprop_bin is None:
        print("[ERROR] chemprop_train not found. Install chemprop 1.5.2.")
        return
    print(f"  (using {chemprop_bin})")

    cmd = [
        chemprop_bin,
        "--data_path",         train_csv,
        "--separate_val_path", val_csv,
        "--dataset_type",      "regression",
        "--target_columns",    target_col,
        "--save_dir",          model_dir,
        "--epochs",            str(epochs),
        "--batch_size",        str(batch_size),
        "--hidden_size",       str(hidden_size),
        "--depth",             str(depth),
        "--metric",            "rmse",
    ]
    print(f"▶ Training ChemProp [{target_col}] → {model_dir}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    status = "✓ Done" if proc.returncode == 0 else f"✗ Exit code {proc.returncode}"
    print(f"\n{status}")
    return proc.returncode == 0


if RUN_CHEMPROP:
    train_chemprop(LAM_TRAIN_CSV, LAM_VAL_CSV, CHEMPROP_LAMBDA_DIR, "lambda_max", epochs=80)
    train_chemprop(T12_TRAIN_CSV, T12_VAL_CSV, CHEMPROP_T12_DIR,    "logt12",     epochs=80)
else:
    print("Skipping ChemProp training (RUN_CHEMPROP=False).")
# -

# ## §8 — Custom REINVENT Scoring Plugins
#
# REINVENT4 uses a plugin system: place files named `comp_*.py` in a directory
# reachable via `PYTHONPATH`. Each file contains a `Parameters` dataclass and a
# component class tagged with `@add_tag("__component")`.
#
# We write four plugins:
# 1. **comp_photoswitch_scaffold** — SMARTS-based structural filter (fast, Stage 1)
# 2. **comp_visible_abs_chemprop** — ChemProp λ_max surrogate (Stage 2)
# 3. **comp_half_life_chemprop** — ChemProp log(t₁/₂) surrogate (Stage 2)
# 4. **comp_xtb_homo_lumo** — GFN2-xTB HOMO-LUMO gap (Stage 3 / post-processing)
#
# After writing them, add the plugin directory to PYTHONPATH:
# ```bash
# export PYTHONPATH=/path/to/plugins:$PYTHONPATH
# ```

# +
# IMPORTANT: reinvent_plugins and reinvent_plugins/components must be
# *namespace packages* — they must NOT have __init__.py files.
# REINVENT's importer checks `components.__file__ is None` and raises
# RuntimeError("No valid component directories found") if it finds an __init__.py.
# We only ensure the directories exist; Python's namespace-package mechanism
# (PEP 420) takes care of the rest when PLUGIN_DIR is on PYTHONPATH.
for pkg_dir in [
    os.path.join(PLUGIN_DIR, "reinvent_plugins"),
    PLUGIN_COMP,
]:
    os.makedirs(pkg_dir, exist_ok=True)
    bad_init = os.path.join(pkg_dir, "__init__.py")
    if os.path.exists(bad_init):
        os.remove(bad_init)
        print(f"  Removed conflicting {bad_init}")

print(f"Plugin directory: {PLUGIN_COMP}")

# +
# ── Plugin 1: Photoswitch scaffold + forbidden alerts ────────────────────────
PLUGIN_SCAFFOLD = os.path.join(PLUGIN_COMP, "comp_photoswitch_scaffold.py")

with open(PLUGIN_SCAFFOLD, "w") as fh:
    fh.write('''"""Photoswitch scaffold filter + custom SMARTS alerts.

Scores 1 if molecule contains a recognised photoswitch motif (azo, etc.)
and does NOT match any forbidden SMARTS.  Otherwise scores 0.

[[stage.scoring.component]]
[stage.scoring.component.PhotoswitchScaffold]

[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name = "PS_Scaffold"
weight = 1.0
"""

__all__ = ["PhotoswitchScaffold"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


PHOTOSWITCH_SMARTS = [
    "[#6]/N=N/[#6]",     # E-azo
    "[#6]\\\\N=N\\\\[#6]",  # Z-azo
    "[#6]N=N[#6]",       # any azo
    "[#6]/C=N/[#7]",     # hydrazone
    "[#6]/N=C/[#6]",     # imine
]

FORBIDDEN_SMARTS = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]",
]


@add_tag("__parameters")
@dataclass
class Parameters:
    pass  # no configurable parameters — scaffold + forbidden SMARTS are hardcoded


@add_tag("__component", "filter")
class PhotoswitchScaffold:
    def __init__(self, params: Parameters):
        self.ps_templates = [Chem.MolFromSmarts(s) for s in PHOTOSWITCH_SMARTS]
        self.forbidden_templates = [
            t for t in [Chem.MolFromSmarts(s) for s in FORBIDDEN_SMARTS] if t
        ]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(0.0)
                continue

            # Must match a photoswitch motif
            if not any(mol.HasSubstructMatch(t) for t in self.ps_templates if t):
                scores.append(0.0)
                continue

            # Must not match forbidden
            if any(mol.HasSubstructMatch(t) for t in self.forbidden_templates):
                scores.append(0.0)
                continue

            scores.append(1.0)

        return ComponentResults([np.array(scores, dtype=float)])
''')
print(f"Written: {PLUGIN_SCAFFOLD}")

# +
# ── Plugin 2: Visible absorption score via ChemProp λ_max model ──────────────
PLUGIN_VIS_ABS = os.path.join(PLUGIN_COMP, "comp_visible_abs_chemprop.py")

with open(PLUGIN_VIS_ABS, "w") as fh:
    fh.write('''"""Visible-light absorption score using a ChemProp λ_max predictor.

Predicts π→π* λ_max (nm) and scores it with a trapezoidal function:
  score = 1 if λ_max in [vis_low, vis_high]
  score ramps linearly from 0 to 1 in the margins [vis_low-margin, vis_low]
  and from 1 to 0 in [vis_high, vis_high+margin].

[[stage.scoring.component]]
[stage.scoring.component.VisibleAbsChemProp]

[[stage.scoring.component.VisibleAbsChemProp.endpoint]]
name = "VisAbs"
weight = 0.8

params.checkpoint_dir = "/path/to/chemprop_lambda"
params.target_column  = "lambda_max"
params.vis_low        = 400.0
params.vis_high       = 650.0
params.margin         = 40.0
"""

__all__ = ["VisibleAbsChemProp"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles


@add_tag("__parameters")
@dataclass
class Parameters:
    checkpoint_dir: List[str]
    target_column:  List[str]
    vis_low:        List[float]
    vis_high:       List[float]
    margin:         List[float]


@add_tag("__component")
class VisibleAbsChemProp:
    def __init__(self, params: Parameters):
        import chemprop
        self.smiles_type = "rdkit_smiles"
        self.vis_low  = params.vis_low[0]
        self.vis_high = params.vis_high[0]
        self.margin   = params.margin[0]
        self.target_col = params.target_column[0]

        args_list = [
            "--checkpoint_dir", params.checkpoint_dir[0],
            "--test_path", "/dev/null",
            "--preds_path", "/dev/null",
        ]
        with suppress_output():
            cp_args  = chemprop.args.PredictArgs().parse_args(args_list)
            cp_model = chemprop.train.load_model(args=cp_args)
        self._chemprop = (cp_model, cp_args)
        target_cols = cp_model[-1]
        self._target_idx = target_cols.index(self.target_col)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        import chemprop
        batched = [[s] for s in smilies]
        with suppress_output():
            preds = chemprop.train.make_predictions(
                model_objects=self._chemprop[0],
                smiles=batched,
                args=self._chemprop[1],
                return_invalid_smiles=True,
            )
        raw = np.array(preds, dtype=object).flatten()
        scores = []
        lo, hi, mg = self.vis_low, self.vis_high, self.margin
        for v in raw:
            try:
                lam = float(v)
            except (TypeError, ValueError):
                scores.append(0.0)
                continue
            if lo <= lam <= hi:
                scores.append(1.0)
            elif lo - mg <= lam < lo:
                scores.append((lam - (lo - mg)) / mg)
            elif hi < lam <= hi + mg:
                scores.append(1.0 - (lam - hi) / mg)
            else:
                scores.append(0.0)
        return ComponentResults([np.array(scores, dtype=float)])
''')
print(f"Written: {PLUGIN_VIS_ABS}")

# +
# ── Plugin 3: Thermal half-life score via ChemProp log(t₁/₂) model ───────────
PLUGIN_T12 = os.path.join(PLUGIN_COMP, "comp_half_life_chemprop.py")

with open(PLUGIN_T12, "w") as fh:
    fh.write('''"""Thermal half-life score using a ChemProp log10(t1/2) predictor.

Target: bistable switches with t1/2 between t_min and t_max seconds.
Scores 1 inside the window, ramps to 0 outside.

[[stage.scoring.component]]
[stage.scoring.component.HalfLifeChemProp]

[[stage.scoring.component.HalfLifeChemProp.endpoint]]
name = "HalfLife"
weight = 0.6

params.checkpoint_dir = "/path/to/chemprop_t12"
params.target_column  = "logt12"
params.logt12_min     = 3.5     # log10(3162 s) ≈ 52 min
params.logt12_max     = 9.0     # log10(10^9 s) ≈ 31 yr
params.margin         = 0.5     # log10 units
"""

__all__ = ["HalfLifeChemProp"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles


@add_tag("__parameters")
@dataclass
class Parameters:
    checkpoint_dir: List[str]
    target_column:  List[str]
    logt12_min:     List[float]
    logt12_max:     List[float]
    margin:         List[float]


@add_tag("__component")
class HalfLifeChemProp:
    def __init__(self, params: Parameters):
        import chemprop
        self.smiles_type  = "rdkit_smiles"
        self.logt12_min   = params.logt12_min[0]
        self.logt12_max   = params.logt12_max[0]
        self.margin       = params.margin[0]
        self.target_col   = params.target_column[0]

        args_list = [
            "--checkpoint_dir", params.checkpoint_dir[0],
            "--test_path", "/dev/null",
            "--preds_path", "/dev/null",
        ]
        with suppress_output():
            cp_args  = chemprop.args.PredictArgs().parse_args(args_list)
            cp_model = chemprop.train.load_model(args=cp_args)
        self._chemprop = (cp_model, cp_args)
        target_cols = cp_model[-1]
        self._target_idx = target_cols.index(self.target_col)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        import chemprop
        batched = [[s] for s in smilies]
        with suppress_output():
            preds = chemprop.train.make_predictions(
                model_objects=self._chemprop[0],
                smiles=batched,
                args=self._chemprop[1],
                return_invalid_smiles=True,
            )
        raw = np.array(preds, dtype=object).flatten()
        scores = []
        lo, hi, mg = self.logt12_min, self.logt12_max, self.margin
        for v in raw:
            try:
                lt = float(v)
            except (TypeError, ValueError):
                scores.append(0.0)
                continue
            if lo <= lt <= hi:
                scores.append(1.0)
            elif lo - mg <= lt < lo:
                scores.append((lt - (lo - mg)) / mg)
            elif hi < lt <= hi + mg:
                scores.append(1.0 - (lt - hi) / mg)
            else:
                scores.append(0.0)
        return ComponentResults([np.array(scores, dtype=float)])
''')
print(f"Written: {PLUGIN_T12}")

# +
# ── Plugin 4: GFN2-xTB HOMO-LUMO gap (Stage 3 / post-processing) ─────────────
PLUGIN_XTB = os.path.join(PLUGIN_COMP, "comp_xtb_homo_lumo.py")

with open(PLUGIN_XTB, "w") as fh:
    fh.write('''"""GFN2-xTB HOMO-LUMO gap scorer for REINVENT.

Scores molecules based on their GFN2-xTB HOMO-LUMO gap (eV) as a proxy for
visible-light absorption.  Scoring is done directly in gap-space rather than
converting to λ_max, because the GFN2-xTB gap underestimates DFT values by
~2.5× and the correction is not constant across chemical space.

Calibration on azo photoswitches (xtb-python ≥ 22.x):
  azobenzene      gap=1.36 eV → λ_exp=320 nm
  4-NH2-azo       gap=1.34 eV → λ_exp=385 nm
  methyl orange   gap=1.25 eV → λ_exp=460 nm
Target range for visible absorbers (400–650 nm): gap_min=0.9, gap_max=1.4 eV.

CAUTION: ~1-3 s/molecule on CPU.  Use with small batches (batch_size=40).

[[stage.scoring.component]]
[stage.scoring.component.XTBHomoLumo]

[[stage.scoring.component.XTBHomoLumo.endpoint]]
name = "xTB_Gap"
weight = 0.7

params.gap_min_ev = [0.9]    # lower bound  (~650 nm visible)
params.gap_max_ev = [1.4]    # upper bound  (~400 nm visible)
"""

__all__ = ["XTBHomoLumo"]
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    gap_min_ev: List[float]
    gap_max_ev: List[float]


@add_tag("__component")
class XTBHomoLumo:
    def __init__(self, params: Parameters):
        self.gap_min = params.gap_min_ev[0]
        self.gap_max = params.gap_max_ev[0]
        self._mid    = (self.gap_min + self.gap_max) / 2.0
        self._width  = (self.gap_max - self.gap_min) / 2.0

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score_mol(mol) for mol in mols], dtype=float
        )])

    def _score_mol(self, mol):
        if mol is None:
            return 0.0
        try:
            from xtb.interface import Calculator, Param
        except ImportError:
            return 0.0

        try:
            mol3d = Chem.AddHs(mol)
            emb = AllChem.ETKDGv3()
            emb.randomSeed = 42
            if AllChem.EmbedMolecule(mol3d, emb) == -1:
                return 0.0
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)

            positions   = mol3d.GetConformer().GetPositions()
            atomic_nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
            coords_bohr = positions * 1.8897259886

            calc = Calculator(Param.GFN2xTB, atomic_nums, coords_bohr)
            calc.set_verbosity(0)
            res  = calc.singlepoint()

            evals = res.get_orbital_eigenvalues()   # Hartree
            occs  = res.get_orbital_occupations()   # 0.0 or 2.0 (± Fermi smearing)

            occupied   = evals[occs > 0.5]
            unoccupied = evals[occs <= 0.5]
            if len(occupied) == 0 or len(unoccupied) == 0:
                return 0.0

            gap_ev = (unoccupied[0] - occupied[-1]) * 27.2114
            if gap_ev <= 0:
                return 0.0

            # Gaussian centred on [gap_min, gap_max] midpoint
            score = float(np.exp(-0.5 * ((gap_ev - self._mid) / self._width) ** 2))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0
''')
print(f"Written: {PLUGIN_XTB}")
# -

# ## §9 — Transfer Learning Configuration & Run

# +
# ── Prior model file ─────────────────────────────────────────────────────────
# REINVENT4 ships without a bundled prior.  Download from Zenodo and place in
# the project root (or point PRIOR_FILE wherever you saved it).
#   Local:     FS_Ro5_10M.model   (30 MB, Zip/PyTorch)
#   FarmShare: ~/projects/reinvent_photoswitch/models/priors/FS_Ro5_10M.model
PRIOR_FILE  = os.path.join(PROJ_ROOT, "FS_Ro5_10M.model")

if not os.path.isfile(PRIOR_FILE):
    raise FileNotFoundError(
        f"Prior model not found at {PRIOR_FILE}\n"
        "Download it from Zenodo and place it in the project root, or\n"
        "edit PRIOR_FILE above to point to your copy."
    )
print(f"Using prior: {PRIOR_FILE}  ({os.path.getsize(PRIOR_FILE)/1e6:.1f} MB)")

TL_OUT_DIR  = os.path.join(OUT_DIR, "tl_run");  os.makedirs(TL_OUT_DIR, exist_ok=True)
TL_MODEL    = os.path.join(TL_OUT_DIR, "TL_photoswitch.model")
TB_TL_DIR   = os.path.join(TL_OUT_DIR, "tb_tl")

TL_CONFIG = f"""
run_type = "transfer_learning"
device = "{DEVICE}"
tb_logdir = "{TB_TL_DIR}"

[parameters]

num_epochs            = 50
save_every_n_epochs   = 2
batch_size            = 128
sample_batch_size     = 2000

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
print(f"TL config written: {TL_CONFIG_FILE}")
print(TL_CONFIG)

# +
# ── Run Transfer Learning ─────────────────────────────────────────────────────
# Order: TL must complete before any RL stage.
# Mac estimate: ~30-90 min (50 epochs, CPU).  FarmShare GPU: ~10-20 min.

TL_LOG = os.path.join(TL_OUT_DIR, "tl.log")

if RUN_TL:
    run_reinvent(TL_CONFIG_FILE, TL_LOG)
else:
    print("Skipping TL (RUN_TL=False).")

# +
# ── TL: inspect with TensorBoard & choose best checkpoint ────────────────────
# To view TensorBoard, run this in a terminal (NOT inside the notebook):
#   tensorboard --logdir <TB_TL_DIR printed below>
# Then open http://localhost:6006 in your browser.
# Pick the checkpoint where validation NLL is lowest AND valid-SMILES % is ≥ 95.

import glob
print(f"TensorBoard log dir: {TB_TL_DIR}")
chkpts = sorted(glob.glob(os.path.join(TL_OUT_DIR, "TL_photoswitch.model.*.chkpt")))
print(f"Found {len(chkpts)} TL checkpoints:")
for c in chkpts:
    print(f"  {os.path.basename(c)}")

# Auto-select the middle checkpoint as a sensible default;
# override TL_EPOCH below after inspecting TensorBoard.
if chkpts:
    # Default: pick epoch 30, or the latest available if fewer epochs ran
    available_epochs = sorted(
        int(os.path.basename(c).split(".")[-2]) for c in chkpts
    )
    TL_EPOCH = 30 if 30 in available_epochs else available_epochs[len(available_epochs) // 2]
else:
    TL_EPOCH = 30  # will fall back to PRIOR_FILE if file doesn't exist

TL_BEST_CHKPT = os.path.join(TL_OUT_DIR, f"TL_photoswitch.model.{TL_EPOCH}.chkpt")
if os.path.exists(TL_BEST_CHKPT):
    print(f"\nUsing TL checkpoint: {TL_BEST_CHKPT}")
else:
    print(f"\nCheckpoint not found — RL stages will use the prior directly.")
    print(f"(Run TL first, then re-execute this cell to pick a checkpoint.)")
# -

# ## §10 — Stage 1 RL: Structural Filter
#
# **Goal**: Drive the agent to generate valid photoswitch scaffolds, acceptable MW / cLogP,
# and no structural alerts.  Very fast — no ML models needed.
#
# Components:
# - `custom_alerts`: standard REINVENT SMARTS filter (returns 0/1)
# - `PhotoswitchScaffold`: our plugin (returns 0/1)
# - `QED`: drug-likeness proxy
# - `MolecularWeight`: keep within 150–550 Da via sigmoid transform

# +
RL1_OUT_DIR = os.path.join(OUT_DIR, "rl_stage1"); os.makedirs(RL1_OUT_DIR, exist_ok=True)
RL1_CHKPT   = os.path.join(RL1_OUT_DIR, "stage1.chkpt")
TB_RL1_DIR  = os.path.join(RL1_OUT_DIR, "tb_stage1")

PLUGIN_PYTHONPATH = PLUGIN_DIR  # parent of reinvent_plugins/

STAGE1_CONFIG = f"""
run_type = "staged_learning"
device = "{DEVICE}"
tb_logdir = "{TB_RL1_DIR}"
json_out_config = "{RL1_OUT_DIR}/_stage1.json"

[parameters]

prior_file          = "{PRIOR_FILE}"
agent_file          = "{TL_BEST_CHKPT if os.path.exists(TL_BEST_CHKPT) else PRIOR_FILE}"
summary_csv_prefix  = "{RL1_OUT_DIR}/stage1"
batch_size          = 100
use_checkpoint      = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = 400
chkpt_file = "{RL1_CHKPT}"

[stage.scoring]
type = "geometric_mean"

# ── 1. Structural alerts (score 0 on any match) ────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name   = "Alerts"
weight = 1.0

params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[*;r13]", "[*;r14]", "[*;r15]", "[*;r16]", "[*;r17]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "C#C",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]"
]

# ── 2. Photoswitch scaffold check (custom plugin) ──────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.PhotoswitchScaffold]

[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name   = "PS_Scaffold"
weight = 1.0

# ── 3. Drug-likeness (QED) ─────────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.5

# ── Diversity filter ───────────────────────────────────────────────────────
[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.6

# ── Inception / replay memory ──────────────────────────────────────────────
[inception]
smiles_file = ""
memory_size = 100
sample_size = 10
"""

RL1_CONFIG_FILE = os.path.join(RL1_OUT_DIR, "stage1.toml")
with open(RL1_CONFIG_FILE, "w") as f:
    f.write(STAGE1_CONFIG)
print(f"Stage 1 config written: {RL1_CONFIG_FILE}")

# +
# ── Run Stage 1 RL ────────────────────────────────────────────────────────────
# Mac estimate: ~15-30 min (400 steps, structural scoring only — very fast per step).
RL1_LOG = os.path.join(RL1_OUT_DIR, "stage1.log")

if RUN_RL1:
    run_reinvent(RL1_CONFIG_FILE, RL1_LOG)
else:
    print("Skipping Stage 1 RL (RUN_RL1=False).")
# -

# ## §11 — Stage 2 RL: Add ML-Predicted Photoswitch Properties
#
# Starting from the Stage 1 checkpoint, we add two ChemProp scoring components:
# - **VisibleAbsChemProp**: predicted λ_max must be 400–650 nm
# - **HalfLifeChemProp**: predicted log(t₁/₂) in the bistability window
#
# We also tighten the diversity filter and increase max_steps.

# +
RL2_OUT_DIR = os.path.join(OUT_DIR, "rl_stage2"); os.makedirs(RL2_OUT_DIR, exist_ok=True)
RL2_CHKPT   = os.path.join(RL2_OUT_DIR, "stage2.chkpt")
TB_RL2_DIR  = os.path.join(RL2_OUT_DIR, "tb_stage2")

# ChemProp v1 saves each fold as model_0/model.pt inside the checkpoint dir
_cp_lam_ready = os.path.exists(os.path.join(CHEMPROP_LAMBDA_DIR, "model_0", "model.pt"))
_cp_t12_ready = os.path.exists(os.path.join(CHEMPROP_T12_DIR,    "model_0", "model.pt"))

if not _cp_lam_ready:
    print(f"⚠ λ_max ChemProp model not found — VisAbs component will be omitted from Stage 2.")
    print(f"  Train it first: set RUN_CHEMPROP=True and re-run §7.")
if not _cp_t12_ready:
    print(f"⚠ t½ ChemProp model not found — HalfLife component will be omitted from Stage 2.")

# Conditional TOML fragments — only included when models exist
_STAGE2_VIS_ABS = f"""
# ── 3. Visible absorption (ChemProp λ_max surrogate) ──────────────────────
[[stage.scoring.component]]
[stage.scoring.component.VisibleAbsChemProp]

[[stage.scoring.component.VisibleAbsChemProp.endpoint]]
name   = "VisAbs"
weight = 0.8

params.checkpoint_dir = ["{CHEMPROP_LAMBDA_DIR}"]
params.target_column  = ["lambda_max"]
params.vis_low        = [400.0]
params.vis_high       = [650.0]
params.margin         = [40.0]
""" if _cp_lam_ready else "# VisAbs component skipped — ChemProp λ_max model not trained yet\n"

_STAGE2_HALFLIFE = f"""
# ── 4. Thermal half-life (ChemProp log(t1/2) surrogate) ───────────────────
[[stage.scoring.component]]
[stage.scoring.component.HalfLifeChemProp]

[[stage.scoring.component.HalfLifeChemProp.endpoint]]
name   = "HalfLife"
weight = 0.6

params.checkpoint_dir = ["{CHEMPROP_T12_DIR}"]
params.target_column  = ["logt12"]
params.logt12_min     = [3.5]
params.logt12_max     = [9.0]
params.margin         = [0.5]
""" if _cp_t12_ready else "# HalfLife component skipped — ChemProp t½ model not trained yet\n"

_s2_agent = (RL1_CHKPT if os.path.exists(RL1_CHKPT)
             else (TL_BEST_CHKPT if os.path.exists(TL_BEST_CHKPT)
                   else PRIOR_FILE))

STAGE2_CONFIG = f"""
run_type = "staged_learning"
device = "{DEVICE}"
tb_logdir = "{TB_RL2_DIR}"
json_out_config = "{RL2_OUT_DIR}/_stage2.json"

[parameters]

prior_file         = "{PRIOR_FILE}"
agent_file         = "{_s2_agent}"
summary_csv_prefix = "{RL2_OUT_DIR}/stage2"
batch_size         = 100
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = 600
chkpt_file = "{RL2_CHKPT}"

[stage.scoring]
type = "geometric_mean"

# ── 1. Structural alerts ───────────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name   = "Alerts"
weight = 1.0

params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]"
]

# ── 2. Photoswitch scaffold ────────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.PhotoswitchScaffold]

[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name   = "PS_Scaffold"
weight = 1.0
""" + _STAGE2_VIS_ABS + _STAGE2_HALFLIFE + f"""
# ── QED ───────────────────────────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.4

# ── Diversity filter ───────────────────────────────────────────────────────
[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.7

# ── Inception ─────────────────────────────────────────────────────────────
[inception]
smiles_file = ""
memory_size = 100
sample_size = 10
"""

RL2_CONFIG_FILE = os.path.join(RL2_OUT_DIR, "stage2.toml")
with open(RL2_CONFIG_FILE, "w") as f:
    f.write(STAGE2_CONFIG)
_active = []
if _cp_lam_ready: _active.append("VisAbs")
if _cp_t12_ready: _active.append("HalfLife")
_base = ["Alerts", "PS_Scaffold", "QED"]
print(f"Stage 2 config written: {RL2_CONFIG_FILE}")
print(f"  Active components: {', '.join(_base + _active)}")

# +
# ── Run Stage 2 RL ────────────────────────────────────────────────────────────
# Mac estimate: ~2-6 hours (600 steps; ChemProp batch scoring adds ~0.5-2 s/step).
# FarmShare GPU: ~30-60 min.
RL2_LOG = os.path.join(RL2_OUT_DIR, "stage2.log")

if RUN_RL2:
    run_reinvent(RL2_CONFIG_FILE, RL2_LOG)
else:
    print("Skipping Stage 2 RL (RUN_RL2=False).")
# -

# ## §12 — Stage 3 RL: xTB HOMO-LUMO Gap (Quantum Chemistry Reward)
#
# Stage 3 adds the `XTBHomoLumo` component.  Because xTB is ~1-3 s/molecule,
# reduce `batch_size` to 30–50 and `max_steps` to 200.
# Consider running Stage 3 only on FarmShare GPU/CPU nodes with proper time allocation.

# +
RL3_OUT_DIR = os.path.join(OUT_DIR, "rl_stage3"); os.makedirs(RL3_OUT_DIR, exist_ok=True)
RL3_CHKPT   = os.path.join(RL3_OUT_DIR, "stage3.chkpt")
TB_RL3_DIR  = os.path.join(RL3_OUT_DIR, "tb_stage3")

# Recheck model availability (cell order may differ)
_cp_lam_ready3 = os.path.exists(os.path.join(CHEMPROP_LAMBDA_DIR, "model_0", "model.pt"))
if not _cp_lam_ready3:
    print(f"⚠ λ_max ChemProp model not found — VisAbs component will be omitted from Stage 3.")

_STAGE3_VIS_ABS = f"""
[[stage.scoring.component]]
[stage.scoring.component.VisibleAbsChemProp]

[[stage.scoring.component.VisibleAbsChemProp.endpoint]]
name   = "VisAbs"
weight = 0.8
params.checkpoint_dir = ["{CHEMPROP_LAMBDA_DIR}"]
params.target_column  = ["lambda_max"]
params.vis_low        = [400.0]
params.vis_high       = [650.0]
params.margin         = [40.0]
""" if _cp_lam_ready3 else "# VisAbs component skipped — ChemProp λ_max model not trained yet\n"

_s3_agent = RL2_CHKPT if os.path.exists(RL2_CHKPT) else PRIOR_FILE

STAGE3_CONFIG = f"""
run_type = "staged_learning"
device = "{DEVICE}"
tb_logdir = "{TB_RL3_DIR}"
json_out_config = "{RL3_OUT_DIR}/_stage3.json"

[parameters]

prior_file         = "{PRIOR_FILE}"
agent_file         = "{_s3_agent}"
summary_csv_prefix = "{RL3_OUT_DIR}/stage3"
batch_size         = 40
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = 200
chkpt_file = "{RL3_CHKPT}"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name   = "Alerts"
weight = 1.0

params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]"
]

[[stage.scoring.component]]
[stage.scoring.component.PhotoswitchScaffold]

[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name   = "PS_Scaffold"
weight = 1.0
""" + _STAGE3_VIS_ABS + f"""
[[stage.scoring.component]]
[stage.scoring.component.XTBHomoLumo]

[[stage.scoring.component.XTBHomoLumo.endpoint]]
name   = "xTB_Gap"
weight = 0.7

params.gap_min_ev = [0.9]
params.gap_max_ev = [1.4]

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.7

[inception]
smiles_file = ""
memory_size = 100
sample_size = 10
"""

RL3_CONFIG_FILE = os.path.join(RL3_OUT_DIR, "stage3.toml")
with open(RL3_CONFIG_FILE, "w") as f:
    f.write(STAGE3_CONFIG)
_active3 = ["Alerts", "PS_Scaffold"]
if _cp_lam_ready3: _active3.append("VisAbs")
_active3.append("xTB_Gap")
print(f"Stage 3 config written: {RL3_CONFIG_FILE}")
print(f"  Active components: {', '.join(_active3)}")

# +
# ── Run Stage 3 RL ────────────────────────────────────────────────────────────
# Requires: conda install -c conda-forge xtb-python
# Mac estimate: ~8-24 hours (200 steps × ~2-3 s/mol xTB × batch 40).
# Recommended: run on FarmShare or skip and use Stage 2 outputs.
RL3_LOG = os.path.join(RL3_OUT_DIR, "stage3.log")

if RUN_RL3:
    try:
        from xtb.interface import Calculator  # noqa: F401
        run_reinvent(RL3_CONFIG_FILE, RL3_LOG)
    except ImportError:
        print("[SKIP] xtb-python not installed — Stage 3 requires it.")
        print("  Install: conda install -c conda-forge xtb-python")
        print("  Then set RUN_RL3 = True and re-run.")
else:
    print("Skipping Stage 3 RL (RUN_RL3=False — set True once xtb-python is installed).")
# -

# ## §13 — Results Analysis

# +
import glob

def load_rl_csv(stage_out_dir, stage_prefix):
    """Load and concatenate all CSV files from an RL stage run."""
    csvs = sorted(glob.glob(os.path.join(stage_out_dir, f"{stage_prefix}_*.csv")))
    if not csvs:
        print(f"No CSV files found in {stage_out_dir}")
        return pd.DataFrame()
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


df_s1 = load_rl_csv(RL1_OUT_DIR, "stage1")
df_s2 = load_rl_csv(RL2_OUT_DIR, "stage2")
df_s3 = load_rl_csv(RL3_OUT_DIR, "stage3")

for label, df in [("Stage 1", df_s1), ("Stage 2", df_s2), ("Stage 3", df_s3)]:
    if df.empty:
        print(f"{label}: no data yet")
    else:
        print(f"{label}: {len(df)} rows, valid SMILES: {(df.get('SMILES_state', pd.Series()) == 1).sum()}")

# +
# ── Post-process Stage 2 / 3 outputs ────────────────────────────────────────
def filter_good_candidates(df, min_total_score=0.6, min_qed=0.4):
    """Return unique, high-scoring molecules from an RL CSV.

    REINVENT writes the aggregate score as 'Score' (capital S).
    The QED column is named 'QED' when the QED component is active.
    """
    if df.empty:
        return df
    valid = df[df.get("SMILES_state", pd.Series([1]*len(df))) == 1].copy()
    valid = valid.drop_duplicates(subset=["SMILES"])

    # REINVENT names the aggregate column "Score" — fall back to first numeric col
    score_col = next((c for c in ("Score", "total_score") if c in valid.columns), None)
    qed_col   = "QED" if "QED" in valid.columns else None

    if score_col:
        valid = valid[valid[score_col] >= min_total_score]
    if qed_col:
        valid = valid[valid[qed_col] >= min_qed]

    sort_col = score_col or valid.columns[0]
    return valid.sort_values(sort_col, ascending=False)


good_s2 = filter_good_candidates(df_s2)
good_s3 = filter_good_candidates(df_s3)
print(f"Good Stage-2 candidates : {len(good_s2)}")
print(f"Good Stage-3 candidates : {len(good_s3)}")

if not good_s2.empty:
    print(good_s2.head(10))
# -

# ── Optional: run xTB on top Stage-2 candidates for final ranking ────────────
if not good_s2.empty:
    top50 = good_s2.head(50)["SMILES"].tolist()
    print("Running xTB on top-50 Stage-2 candidates...")
    xtb_results = batch_xtb_screen(top50, target_lam_min=400, target_lam_max=650)
    _score_col = "Score" if "Score" in good_s2.columns else "total_score"
    final = good_s2.head(50).merge(xtb_results, left_on="SMILES", right_on="smiles", how="left")
    display_cols = [c for c in [_score_col, "xtb_score", "lam_est_nm", "gap_eV"] if c in final.columns]
    print(final[["SMILES"] + display_cols].sort_values("xtb_score", ascending=False).to_string(index=False))

# +
# ── Optional: run TD-DFT on the very best candidates ────────────────────────
# results = tddft_excitations(good_s2.iloc[0]["SMILES"])
# if results:
#     for r in results:
#         print(f"S{r['state']}: λ={r['lambda_nm']:.0f} nm, f={r['osc_strength']:.4f}")
# -

# ── Display top candidates as a molecule grid ────────────────────────────────
if HAS_MOLS2GRID and not good_s2.empty:
    from reinvent.notebooks import create_mol_grid
    grid = create_mol_grid(good_s2.head(50))
    display(grid)

# ## §14 — FarmShare Batch Job Scripts
#
# Write ready-to-submit Slurm batch scripts for all runs.

# +
SCRIPTS_DIR = os.path.join(OUT_DIR, "scripts"); os.makedirs(SCRIPTS_DIR, exist_ok=True)

# Helper: substitute absolute paths with FarmShare paths
FS_HOME = "/home/users/YOUR_SUNETID/projects/reinvent_photoswitch"

def fs_path(local_path):
    """Replace PROJ_ROOT with FarmShare project root for batch scripts."""
    return local_path.replace(str(PROJ_ROOT), FS_HOME)


# ── TL batch script ──────────────────────────────────────────────────────────
TL_SLURM = f"""#!/bin/bash
#SBATCH --job-name=ps_tl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --output={FS_HOME}/logs/tl_%j.out
#SBATCH --error={FS_HOME}/logs/tl_%j.err

module load micromamba
export PYTHONPATH={FS_HOME}/plugins:$PYTHONPATH

cd {FS_HOME}/code/REINVENT4

micromamba run -n reinvent4 reinvent -d cpu \\
  -l {FS_HOME}/logs/tl.log \\
  {fs_path(TL_CONFIG_FILE)}
"""

with open(os.path.join(SCRIPTS_DIR, "run_tl.sh"), "w") as f:
    f.write(TL_SLURM)
os.chmod(os.path.join(SCRIPTS_DIR, "run_tl.sh"), 0o755)


# ── Stage 1 RL batch script ──────────────────────────────────────────────────
RL1_SLURM = f"""#!/bin/bash
#SBATCH --job-name=ps_rl1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --output={FS_HOME}/logs/rl1_%j.out
#SBATCH --error={FS_HOME}/logs/rl1_%j.err

module load micromamba
export PYTHONPATH={FS_HOME}/plugins:$PYTHONPATH

cd {FS_HOME}/code/REINVENT4

micromamba run -n reinvent4 reinvent -d cpu \\
  -l {FS_HOME}/logs/rl_stage1.log \\
  {fs_path(RL1_CONFIG_FILE)}
"""

with open(os.path.join(SCRIPTS_DIR, "run_rl_stage1.sh"), "w") as f:
    f.write(RL1_SLURM)
os.chmod(os.path.join(SCRIPTS_DIR, "run_rl_stage1.sh"), 0o755)


# ── Stage 2 RL batch script ──────────────────────────────────────────────────
RL2_SLURM = f"""#!/bin/bash
#SBATCH --job-name=ps_rl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --output={FS_HOME}/logs/rl2_%j.out
#SBATCH --error={FS_HOME}/logs/rl2_%j.err

module load micromamba
export PYTHONPATH={FS_HOME}/plugins:$PYTHONPATH

cd {FS_HOME}/code/REINVENT4

micromamba run -n reinvent4 reinvent -d cpu \\
  -l {FS_HOME}/logs/rl_stage2.log \\
  {fs_path(RL2_CONFIG_FILE)}
"""

with open(os.path.join(SCRIPTS_DIR, "run_rl_stage2.sh"), "w") as f:
    f.write(RL2_SLURM)
os.chmod(os.path.join(SCRIPTS_DIR, "run_rl_stage2.sh"), 0o755)


# ── Stage 3 RL batch script (CPU, longer walltime for xTB) ───────────────────
RL3_SLURM = f"""#!/bin/bash
#SBATCH --job-name=ps_rl3_xtb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output={FS_HOME}/logs/rl3_%j.out
#SBATCH --error={FS_HOME}/logs/rl3_%j.err

module load micromamba
export PYTHONPATH={FS_HOME}/plugins:$PYTHONPATH

# xTB uses OpenMP threading — match OMP_NUM_THREADS to cpus-per-task
export OMP_NUM_THREADS=16

cd {FS_HOME}/code/REINVENT4

micromamba run -n reinvent4 reinvent -d cpu \\
  -l {FS_HOME}/logs/rl_stage3.log \\
  {fs_path(RL3_CONFIG_FILE)}
"""

with open(os.path.join(SCRIPTS_DIR, "run_rl_stage3_xtb.sh"), "w") as f:
    f.write(RL3_SLURM)
os.chmod(os.path.join(SCRIPTS_DIR, "run_rl_stage3_xtb.sh"), 0o755)


print("Slurm scripts written:")
for fn in os.listdir(SCRIPTS_DIR):
    print(f"  {os.path.join(SCRIPTS_DIR, fn)}")
# -

# ## §15 — Submitting Jobs on FarmShare
#
# After uploading everything to FarmShare and editing `YOUR_SUNETID` in the scripts:
#
# ```bash
# # Interactive smoke-test (verify configs before committing to a long batch job)
# srun --partition=interactive --qos=interactive --pty bash
# module load micromamba
# export PYTHONPATH=~/projects/reinvent_photoswitch/plugins:$PYTHONPATH
# cd ~/projects/reinvent_photoswitch/code/REINVENT4
# micromamba run -n reinvent4 reinvent -d cpu \
#   -l /tmp/test_tl.log \
#   ~/projects/reinvent_photoswitch/outputs/tl_run/tl_config.toml
# # Let it run for 2-3 epochs, then Ctrl+C to verify it runs cleanly.
#
# # Submit batch jobs in order:
# sbatch ~/projects/reinvent_photoswitch/outputs/scripts/run_tl.sh
# # After TL completes, update TL_EPOCH in the Stage 1 config, then:
# sbatch ~/projects/reinvent_photoswitch/outputs/scripts/run_rl_stage1.sh
# sbatch ~/projects/reinvent_photoswitch/outputs/scripts/run_rl_stage2.sh
# sbatch ~/projects/reinvent_photoswitch/outputs/scripts/run_rl_stage3_xtb.sh
#
# # Monitor jobs:
# squeue -u YOUR_SUNETID
# tail -f ~/projects/reinvent_photoswitch/logs/tl.log
#
# # Switch to GPU (once CPU pipeline is verified):
# #   1. Change device = "cuda:0" in TOML configs
# #   2. Change python install.py cpu → python install.py gpu in environment setup
# #   3. Add to sbatch: #SBATCH --gres=gpu:1
# #      and consult sinfo / scontrol for the correct GPU partition on FarmShare
# ```
