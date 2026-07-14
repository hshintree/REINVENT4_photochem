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
# # Photoswitch Discovery — Complete Pipeline
#
# **Goal**: Generate novel visible-light photoswitches with good thermal stability
# and synthetic accessibility using REINVENT4, starting from raw data all the way
# through DFT validation.
#
# ## Pipeline overview
#
# | Step | Section | What happens |
# |------|---------|-------------|
# | 1 | §1–§2 | Setup, imports, paths |
# | 2 | §3–§4 | Load raw data, clean to "good switches", prepare TL + ChemProp files |
# | 3 | §5 | Train ChemProp surrogate models (λ_max, log t½) |
# | 4 | §6 | Write custom REINVENT scoring plugins to disk |
# | 5 | §7 | Transfer Learning — fine-tune the prior on photoswitch SMILES |
# | 6 | §8–§10 | Three-stage Reinforcement Learning with progressively expensive scoring |
# | 7 | §11 | Collect & rank RL results |
# | 8 | §12 | Cluster top candidates (Tanimoto / Butina) for chemical diversity |
# | 9 | §13 | Rough xTB screening on cluster representatives |
# | 10 | §14–§15 | TD-DFT analysis + Jablonski / UV-Vis plots on the best molecules |
# | 11 | §16 | FarmShare cluster instructions |
#
# ---
#
# ## Staged RL design
#
# | Stage | Scoring Components | Batch | Steps | Speed |
# |-------|--------------------|-------|-------|-------|
# | **1 — Structural gates** | Custom alerts, SA Score, QED | 128 | 500 | ~0.1 s/batch |
# | **2 — xTB electronic** | Stage 1 + xTB λ_max (hard cutoff 300 nm) + xTB ΔE(E/Z) | 40 | 300 | ~2–5 s/mol |
# | **3 — ChemProp surrogates** | Stage 1 + ChemProp λ_max + ChemProp t½ + SA Score | 80 | 400 | ~0.3 s/mol |

# %% [markdown]
# ---
# ## How REINVENT generates molecules — the generative mechanics
#
# ### The model
#
# We are running **REINVENT4** in **two sequential modes**:
#
# | Mode | What happens |
# |------|-------------|
# | **Transfer Learning (TL)** — §7 | Fine-tunes the prior on known photoswitch SMILES to bias the generator toward photoswitch-like molecules |
# | **Staged Reinforcement Learning (RL)** — §8–§10 | Uses the TL checkpoint as the starting agent and reshapes its probability distributions toward molecules that score highly on our photoswitch criteria |
#
# The underlying generative model is the **classic REINVENT RNN** ([Olivecrona et al. 2017](https://doi.org/10.1186/s13321-017-0235-x)):
# a 3-layer LSTM with 512 hidden units and 256-dimensional token embeddings,
# trained on drug-like molecules (the bundled `reinvent.prior`).
#
# ---
#
# ### The vocabulary
#
# The model works on a **SMILES token vocabulary** — not individual characters, but chemically meaningful tokens:
#
# ```
# Atoms:  C  c  N  n  O  o  S  s  F  Cl  Br  I  P
#         [nH] [N+] [N-] [O-] [n+] [o+] [s+] [S+] [P+] [CH] [CH-] [c-] [S] [O] [I+]
# Bonds:  =  #  -
# Ring:   1  2  3  4  5  6  7  8
# Branch: (  )
# Stereo: /  \  @  @@
# Special: ^  $  (begin-of-sequence, end-of-sequence)
# ```
#
# Maximum sequence length is **128 tokens**.
#
# ---
#
# ### How a new molecule is generated — step by step
#
# ```
# Step 0:  input_token  = ^ (begin token, ID=1)   ← FIXED, never random
#          hidden_state = None → PyTorch zeros     ← FIXED, always zero
#
# For each step t = 1 … 127:
#   logits[V]     = LSTM(input_token, hidden_state)     # raw scores over vocabulary
#   probs[V]      = softmax(logits)                     # probability distribution
#   next_token    = multinomial_sample(probs)           ← THE ONLY SOURCE OF RANDOMNESS
#   input_token   = next_token
#   if next_token == $ (EOS, ID=2): stop
#
# Decode token IDs → SMILES string
# ```
#
# **Key point**: the start token `^` is always the same — there is no random "seed".
# Every molecule in the batch begins from identical starting conditions.
# The only stochasticity is `torch.multinomial(softmax(logits))` at each step.
#
# ---
#
# ### What controls diversity (the knobs you can turn)
#
# | Parameter | Location in config | What it does | Default here |
# |-----------|-------------------|--------------|-------------|
# | `batch_size` | `[parameters]` | Molecules generated per RL step | 128 / 40 / 80 |
# | `sigma` (σ) | `[learning_strategy]` | Reward-signal strength. Higher σ → faster convergence, less diversity | **128** |
# | `max_steps` | `[[stage]]` | RL update steps per stage | 500 / 300 / 400 |
# | Diversity filter | `[diversity_filter]` | Penalizes identical Murcko scaffolds | IdenticalMurckoScaffold |
# | Inception memory | `[inception]` | Replays past high-scoring molecules | enabled |
#
# There is **no temperature parameter** — raw softmax probabilities are used.
# To add temperature scaling, divide logits by T before softmax (T<1 sharpens, T>1 flattens).
#
# ---
#
# ### How RL reshapes the distribution — the DAP algorithm
#
# At each RL step:
#
# 1. **Sample** a batch of SMILES from the agent
# 2. **Score** each SMILES with our multi-component scoring function
# 3. **Compute the DAP loss** (Directed Augmented Prior):
#
# $$\mathcal{L} = \bigl(\underbrace{\log P_\text{prior}(m)}_{\text{regularizer}} + \sigma \cdot \underbrace{S(m)}_{\text{score}} - \underbrace{\log P_\text{agent}(m)}_{\text{agent likelihood}}\bigr)^2$$
#
# The prior term acts as a **leash** — prevents the agent from collapsing to a
# single scaffold. σ=128 means a full-score improvement shifts the augmented
# log-likelihood by 128 nats. To generate more diverse molecules, **lower σ**
# (e.g. 32–64). To drill harder on high-scoring scaffolds, **raise σ** (200+).
#
# 4. **Backpropagate** and update the agent's LSTM weights (Adam optimizer).
#
# The prior is **frozen** — only the agent is updated.

# %% [markdown]
# ## §1 — Setup & Imports

# %%
import os, sys, shutil, subprocess, glob, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, inchi, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                    "-q", "setuptools==68.2.2"], check=True)
    import importlib; importlib.invalidate_caches()
    import pkg_resources  # noqa: F401

# %% [markdown]
# ## §2 — Paths & Configuration

# %%
PROJ_ROOT  = os.path.abspath(os.path.join(os.path.dirname(""), ".."))
DATA_DIR   = os.path.join(PROJ_ROOT, "notebooks", "data")
OUT_DIR    = os.path.join(PROJ_ROOT, "outputs"); os.makedirs(OUT_DIR, exist_ok=True)
PLUGIN_DIR = os.path.join(PROJ_ROOT, "plugins")
PLUGIN_COMP= os.path.join(PLUGIN_DIR, "reinvent_plugins", "components")

PRIOR_FILE = os.path.join(PROJ_ROOT, "reinvent.prior")
assert os.path.isfile(PRIOR_FILE), f"Prior not found: {PRIOR_FILE}"
print(f"Prior : {PRIOR_FILE}  ({os.path.getsize(PRIOR_FILE)/1e6:.1f} MB)")
print(f"Output: {OUT_DIR}")

# %%
DEVICE = "cpu"  # change to "cuda:0" on FarmShare GPU nodes

RUN_CHEMPROP = True
RUN_TL       = True
RUN_STAGE1   = True
RUN_STAGE2   = True
RUN_STAGE3   = True
RUN_DFT      = True

# %%
def run_reinvent(config_file, log_file, device=None):
    """Execute REINVENT and stream output live."""
    _device = device or DEVICE
    _env = {
        **os.environ,
        "PYTHONPATH":            f"{PLUGIN_DIR}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS":      "1",
    }
    cmd = ["reinvent", "-d", _device, "-l", log_file, config_file]
    print(f"▶ {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=_env)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        tag = "✓ Done" if proc.returncode == 0 else f"✗ Exit code {proc.returncode}"
        print(f"\n{tag}")
        return proc.returncode
    except FileNotFoundError:
        print("[ERROR] 'reinvent' not found — activate the reinvent4 env.")
        return -1


# %% [markdown]
# ## §3 — Load & Explore the Raw Data

# %%
ps_raw = pd.read_csv(os.path.join(DATA_DIR, "photoswitches.csv"), index_col=0)

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
ps["t12_s"] = np.log(2) / ps["therm_rate_s"]

print(f"photoswitches.csv  : {len(ps)} molecules, {ps.shape[1]} columns")
print(f"  λ_E_pipi range   : {ps['lam_E_pipi'].min():.0f}–{ps['lam_E_pipi'].max():.0f} nm  "
      f"(n={ps['lam_E_pipi'].notna().sum()})")
print(f"  t1/2 range       : {ps['t12_s'].min():.1e}–{ps['t12_s'].max():.1e} s   "
      f"(n={ps['t12_s'].notna().sum()})")

# %%
lt_raw = pd.read_excel(os.path.join(DATA_DIR, "fulldata.lambda_train.xlsx"))
lt = lt_raw.rename(columns={"SMILES": "smiles", "lambda": "lam_E_pipi", "t12": "t12_s"})
lt = lt[["smiles", "lam_E_pipi", "t12_s", "logt12", "solvent"]].copy()
lt["source"] = "lambda_train"
print(f"fulldata.lambda_train : {len(lt)} molecules")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(ps["lam_E_pipi"].dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(400, color="red", linestyle="--", label="400 nm")
axes[0].set_xlabel("E isomer π→π* λ_max (nm)"); axes[0].set_ylabel("Count")
axes[0].set_title("photoswitches.csv — λ_max (E)"); axes[0].legend()

axes[1].hist(np.log10(ps["t12_s"].dropna()), bins=30, color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].axvline(np.log10(3600), color="blue", linestyle="--", label="1 h")
axes[1].set_xlabel("log₁₀(t₁/₂ / s)"); axes[1].set_ylabel("Count")
axes[1].set_title("photoswitches.csv — thermal t₁/₂"); axes[1].legend()

axes[2].hist(lt["lam_E_pipi"].dropna(), bins=30, color="#55A868", edgecolor="white", alpha=0.85)
axes[2].axvline(400, color="red", linestyle="--", label="400 nm")
axes[2].set_xlabel("λ_max (nm)"); axes[2].set_ylabel("Count")
axes[2].set_title("fulldata.lambda_train — λ_max"); axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "data_distributions.png"), dpi=150)
plt.show()

# %% [markdown]
# ## §4 — Data Cleaning: "Good Switches"
#
# | Criterion | Threshold | Rationale |
# |-----------|-----------|-----------|
# | Valid SMILES | — | parseable by RDKit |
# | Visible-light responsive | λ_max ≥ 380 nm (strict only) | accessible without UV sources |
# | High PSS | max(PSS_E, PSS_Z) ≥ 65 % | meaningful E↔Z conversion |
# | Bistable half-life | 1 h ≤ t₁/₂ ≤ 100 yr | slow enough to be metastable |
# | No forbidden substructures | SMARTS below | remove reactive / unstable groups |
# | Deduplication | by InChIKey | remove redundant structures |

# %%
FORBIDDEN_SMARTS_LIST = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[#8][#8]", "[#6;+]", "[#16][#16]", "C#C",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]",
    "[#6X5]", "[#7X4;!H0;v4]",
]
FORBIDDEN_TEMPLATES = [(s, Chem.MolFromSmarts(s)) for s in FORBIDDEN_SMARTS_LIST
                       if Chem.MolFromSmarts(s) is not None]

PHOTOSWITCH_SMARTS_DICT = {
    "azo":           Chem.MolFromSmarts("[#6]/N=N/[#6]"),
    "azo_cis":       Chem.MolFromSmarts("[#6]\\N=N\\[#6]"),
    "azo_any":       Chem.MolFromSmarts("[#6]N=N[#6]"),
    "azomethine":    Chem.MolFromSmarts("[#6]/C=N/[#6]"),
    "diarylethene":  Chem.MolFromSmarts("c1cc(cc1)-c2cc(-c3ccccc3)c(=O)[nH]2"),
    "spiropyran":    Chem.MolFromSmarts("C1(OC2=CC=CC=C2)=CC=CC=C1"),
}


def is_photoswitch_scaffold(mol):
    for _, tmpl in PHOTOSWITCH_SMARTS_DICT.items():
        if tmpl and mol.HasSubstructMatch(tmpl):
            return True
    return False


def has_forbidden_substructure(mol):
    for _, tmpl in FORBIDDEN_TEMPLATES:
        if mol.HasSubstructMatch(tmpl):
            return True
    return False


def canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        clean = rdMolStandardize.Cleanup(mol)
        remover = rdMolStandardize.LargestFragmentChooser()
        clean = remover.choose(clean)
        neutralizer = rdMolStandardize.Uncharger()
        clean = neutralizer.uncharge(clean)
        return Chem.MolToSmiles(clean)
    except Exception:
        return None


def inchikey_from_smi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi_str = inchi.MolToInchi(mol)
    return inchi.InchiToInchiKey(inchi_str) if inchi_str else None


def clean_photoswitches(df, strict=False):
    records = []
    for _, row in df.iterrows():
        canon = canonical_smiles(str(row.get("smiles", "")).strip())
        if canon is None:
            continue
        mol = Chem.MolFromSmiles(canon)
        if mol is None or not is_photoswitch_scaffold(mol) or has_forbidden_substructure(mol):
            continue

        if strict:
            lam = row.get("lam_E_pipi")
            if pd.isna(lam):
                continue
            pss_z, pss_e = row.get("PSS_Z"), row.get("PSS_E")
            if (pd.notna(pss_z) or pd.notna(pss_e)):
                pss_max = max(pss_z if pd.notna(pss_z) else 0, pss_e if pd.notna(pss_e) else 0)
                if pss_max < 65:
                    continue
            t12 = row.get("t12_s")
            if pd.notna(t12) and not (3_600 <= t12 <= 3.15e9):
                continue
            wb = row.get("wiberg_idx")
            if pd.notna(wb) and not (1.3 <= wb <= 2.1):
                continue

        t12_val = row.get("t12_s")
        logt12 = np.log10(t12_val) if pd.notna(t12_val) and t12_val > 0 else None
        ik = inchikey_from_smi(canon)
        records.append({
            "smiles": canon, "inchikey": ik,
            "lam_E_pipi": row.get("lam_E_pipi"), "t12_s": t12_val, "logt12": logt12,
            "source": "photoswitches_csv",
        })
    return pd.DataFrame(records).drop_duplicates(subset="inchikey")


def clean_lambda_train(df, lam_min=300, t12_min=3600):
    records = []
    for _, row in df.iterrows():
        canon = canonical_smiles(str(row.get("smiles", "")).strip())
        if canon is None:
            continue
        mol = Chem.MolFromSmiles(canon)
        if mol is None or not is_photoswitch_scaffold(mol) or has_forbidden_substructure(mol):
            continue
        lam = row.get("lam_E_pipi")
        if pd.isna(lam) or lam < lam_min:
            continue
        t12 = row.get("t12_s")
        if pd.notna(t12) and t12 < t12_min:
            continue
        ik = inchikey_from_smi(canon)
        records.append({
            "smiles": canon, "inchikey": ik,
            "lam_E_pipi": lam, "t12_s": t12,
            "logt12": row.get("logt12"), "source": "lambda_train",
        })
    return pd.DataFrame(records).drop_duplicates(subset="inchikey")


ps_broad  = clean_photoswitches(ps, strict=False)
ps_strict = clean_photoswitches(ps, strict=True)
lt_clean  = clean_lambda_train(lt, lam_min=300, t12_min=3600)

print(f"photoswitches.csv  broad  (TL corpus)      : {len(ps_broad)}")
print(f"photoswitches.csv  strict (predictor train) : {len(ps_strict)}")
print(f"lambda_train clean : {len(lt_clean)}")

# %%
SHARED = ["smiles", "inchikey", "lam_E_pipi", "t12_s", "logt12", "source"]
combined_strict = pd.concat(
    [ps_strict[SHARED], lt_clean[SHARED]], ignore_index=True
).drop_duplicates(subset="inchikey")

combined_broad = pd.concat(
    [ps_broad[["smiles", "inchikey", "source"]],
     lt_clean[["smiles", "inchikey", "source"]]],
    ignore_index=True
).drop_duplicates(subset="inchikey")

print(f"Strict set (predictor training) : {len(combined_strict)}")
print(f"Broad  set (TL corpus)          : {len(combined_broad)}")

# %%
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
    n_so_far = 0
    for group in scaffold_groups:
        if n_so_far < n_train_target:
            train_idx.extend(group); n_so_far += len(group)
        else:
            val_idx.extend(group)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


tl_data = combined_broad[["smiles"]].dropna()
tl_train, tl_val = scaffold_split(tl_data, train_frac=0.8)

TL_TRAIN_SMI = os.path.join(OUT_DIR, "photoswitch_tl_train.smi")
TL_VAL_SMI   = os.path.join(OUT_DIR, "photoswitch_tl_val.smi")
STRICT_CSV   = os.path.join(OUT_DIR, "photoswitch_strict.csv")

tl_train["smiles"].to_csv(TL_TRAIN_SMI, index=False, header=False)
tl_val["smiles"].to_csv(TL_VAL_SMI, index=False, header=False)
combined_strict.to_csv(STRICT_CSV, index=False)

print(f"TL train: {len(tl_train)}, TL val: {len(tl_val)}")
print(f"Saved strict set: {STRICT_CSV}")

# %% [markdown]
# ## §5 — Train ChemProp Surrogate Models
#
# Two ChemProp (v1) GNN regression models:
# - **λ_max** — predicts π→π* absorption wavelength (nm)
# - **log(t½)** — predicts log₁₀ of thermal half-life (s)

# %%
CHEMPROP_LAMBDA_DIR = os.path.join(OUT_DIR, "chemprop_lambda")
CHEMPROP_T12_DIR    = os.path.join(OUT_DIR, "chemprop_t12")
os.makedirs(CHEMPROP_LAMBDA_DIR, exist_ok=True)
os.makedirs(CHEMPROP_T12_DIR, exist_ok=True)

lam_data = combined_strict.dropna(subset=["lam_E_pipi"])[["smiles", "lam_E_pipi"]].copy()
lam_data = lam_data.rename(columns={"lam_E_pipi": "lambda_max"})
lam_data_train = lam_data.sample(frac=0.8, random_state=42)
lam_data_val = lam_data.drop(lam_data_train.index)
LAM_TRAIN_CSV = os.path.join(CHEMPROP_LAMBDA_DIR, "train.csv")
LAM_VAL_CSV   = os.path.join(CHEMPROP_LAMBDA_DIR, "val.csv")
lam_data_train.to_csv(LAM_TRAIN_CSV, index=False)
lam_data_val.to_csv(LAM_VAL_CSV, index=False)
print(f"λ_max ChemProp  train={len(lam_data_train)}, val={len(lam_data_val)}")

t12_data = combined_strict.dropna(subset=["t12_s"]).copy()
t12_data["logt12"] = np.log10(t12_data["t12_s"].clip(lower=1))
t12_data = t12_data[["smiles", "logt12"]]
t12_data_train = t12_data.sample(frac=0.8, random_state=42)
t12_data_val = t12_data.drop(t12_data_train.index)
T12_TRAIN_CSV = os.path.join(CHEMPROP_T12_DIR, "train.csv")
T12_VAL_CSV   = os.path.join(CHEMPROP_T12_DIR, "val.csv")
t12_data_train.to_csv(T12_TRAIN_CSV, index=False)
t12_data_val.to_csv(T12_VAL_CSV, index=False)
print(f"log(t½) ChemProp  train={len(t12_data_train)}, val={len(t12_data_val)}")

# %%
def train_chemprop(train_csv, val_csv, model_dir, target_col,
                   epochs=50, batch_size=50, hidden_size=300, depth=3):
    env_bin = os.path.dirname(sys.executable)
    chemprop_bin = os.path.join(env_bin, "chemprop_train")
    if not os.path.isfile(chemprop_bin):
        chemprop_bin = shutil.which("chemprop_train")
    if chemprop_bin is None:
        print("[ERROR] chemprop_train not found. Install chemprop 1.5.2.")
        return
    cmd = [chemprop_bin, "--data_path", train_csv, "--separate_val_path", val_csv,
           "--dataset_type", "regression", "--target_columns", target_col,
           "--save_dir", model_dir, "--epochs", str(epochs),
           "--batch_size", str(batch_size), "--hidden_size", str(hidden_size),
           "--depth", str(depth), "--metric", "rmse"]
    print(f"▶ Training ChemProp [{target_col}] → {model_dir}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    print(f"\n{'✓ Done' if proc.returncode == 0 else f'✗ Exit code {proc.returncode}'}")


if RUN_CHEMPROP:
    train_chemprop(LAM_TRAIN_CSV, LAM_VAL_CSV, CHEMPROP_LAMBDA_DIR, "lambda_max", epochs=80)
    train_chemprop(T12_TRAIN_CSV, T12_VAL_CSV, CHEMPROP_T12_DIR, "logt12", epochs=80)
else:
    print("Skipping ChemProp training (RUN_CHEMPROP=False).")

# %% [markdown]
# ## §6 — Custom REINVENT Scoring Plugins
#
# Written to disk as namespace-package components.
# REINVENT's `importer.py` requires **no** `__init__.py` in the plugin directories.

# %%
os.makedirs(PLUGIN_COMP, exist_ok=True)
for d in [os.path.join(PLUGIN_DIR, "reinvent_plugins"), PLUGIN_COMP]:
    bad = os.path.join(d, "__init__.py")
    if os.path.exists(bad):
        os.remove(bad)

# ── Plugin: XTBLambdaFilter ──────────────────────────────────────────────────
with open(os.path.join(PLUGIN_COMP, "comp_xtb_lambda_filter.py"), "w") as f:
    f.write('''\
"""xTB-based lambda_max filter.
Gaussian centred on target absorption window, hard cutoff below threshold.
"""

__all__ = ["XTBLambdaFilter"]
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

_XTB_CORRECTION = 2.5

@add_tag("__parameters")
@dataclass
class Parameters:
    lambda_cutoff: List[float]
    lambda_target: List[float]
    lambda_sigma:  List[float]

@add_tag("__component")
class XTBLambdaFilter:
    def __init__(self, params: Parameters):
        self.cutoff = params.lambda_cutoff[0]
        self.target = params.lambda_target[0]
        self.sigma  = params.lambda_sigma[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score(mol) for mol in mols], dtype=float)])

    def _score(self, mol):
        if mol is None: return 0.0
        try:
            from xtb.interface import Calculator, Param
        except ImportError:
            return 0.5
        try:
            mol3d = Chem.AddHs(mol)
            emb = AllChem.ETKDGv3(); emb.randomSeed = 42
            if AllChem.EmbedMolecule(mol3d, emb) == -1: return 0.0
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
            pos  = mol3d.GetConformer().GetPositions()
            nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
            calc = Calculator(Param.GFN2xTB, nums, pos * 1.8897259886)
            calc.set_verbosity(0)
            res  = calc.singlepoint()
            evals, occs = res.get_orbital_eigenvalues(), res.get_orbital_occupations()
            occ, unocc = evals[occs > 0.5], evals[occs <= 0.5]
            if len(occ) == 0 or len(unocc) == 0: return 0.0
            gap = (unocc[0] - occ[-1]) * 27.2114
            if gap <= 0: return 0.0
            lam_est = 1240.0 / (gap * _XTB_CORRECTION)
            if lam_est < self.cutoff: return 0.0
            return float(np.clip(np.exp(-0.5 * ((lam_est - self.target)/self.sigma)**2), 0, 1))
        except Exception:
            return 0.0
''')

# ── Plugin: XTBIsomerGap ─────────────────────────────────────────────────────
_ISOMER_PLUGIN = '''"""xTB E/Z isomer energy gap scorer.
Rewards molecules with larger |dE(E-Z)|, indicating longer thermal half-life.
"""

__all__ = ["XTBIsomerGap"]
from typing import List
import re as _re
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
    de_min_kcal:    List[float]
    de_target_kcal: List[float]
    de_sigma_kcal:  List[float]

def _embed(mol):
    mol3d = Chem.AddHs(mol)
    emb = AllChem.ETKDGv3(); emb.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, emb) == -1: return None
    AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
    return mol3d

def _energy(mol3d):
    from xtb.interface import Calculator, Param
    pos  = mol3d.GetConformer().GetPositions()
    nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
    calc = Calculator(Param.GFN2xTB, nums, pos * 1.8897259886)
    calc.set_verbosity(0)
    return calc.singlepoint().get_energy()

def _flip_azo(smi):
    def _inv(m):
        return m.group(1) + "N=N" + ("/" if m.group(2) == chr(92) else chr(92))
    result = _re.sub(r"([/\\\\])N=N([/\\\\])", _inv, smi, count=1)
    if result == smi: return None
    if Chem.MolFromSmiles(result) is None: return None
    return result

@add_tag("__component")
class XTBIsomerGap:
    def __init__(self, params: Parameters):
        self.de_min = params.de_min_kcal[0]
        self.target = params.de_target_kcal[0]
        self.sigma  = params.de_sigma_kcal[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score(mol) for mol in mols], dtype=float)])

    def _score(self, mol):
        if mol is None: return 0.0
        try:
            from xtb.interface import Calculator, Param  # noqa: F401
        except ImportError:
            return 0.5
        try:
            smi = Chem.MolToSmiles(mol)
            e_mol = _embed(mol)
            if e_mol is None: return 0.0
            z_smi = _flip_azo(smi)
            if z_smi is None: return 0.3
            z_mol = Chem.MolFromSmiles(z_smi)
            if z_mol is None: return 0.0
            z_mol3d = _embed(z_mol)
            if z_mol3d is None: return 0.0
            dE = abs(_energy(z_mol3d) - _energy(e_mol)) * 627.509
            if dE < self.de_min: return 0.0
            return float(np.clip(
                np.exp(-0.5 * ((dE - self.target)/self.sigma)**2), 0, 1))
        except Exception:
            return 0.0
'''
with open(os.path.join(PLUGIN_COMP, "comp_xtb_isomer_gap.py"), "w") as f:
    f.write(_ISOMER_PLUGIN)

print("Plugins written:")
for p in sorted(glob.glob(os.path.join(PLUGIN_COMP, "comp_*.py"))):
    print(f"  {os.path.basename(p)}")

# %% [markdown]
# ## §7 — Transfer Learning
#
# Fine-tune `reinvent.prior` on photoswitch SMILES to bias the generator's token
# distributions toward azo / hydrazone / diarylethene fragments.

# %%
TL_OUT_DIR = os.path.join(OUT_DIR, "tl_run"); os.makedirs(TL_OUT_DIR, exist_ok=True)
TL_MODEL   = os.path.join(TL_OUT_DIR, "TL_photoswitch.model")
TB_TL_DIR  = os.path.join(TL_OUT_DIR, "tb_tl")

TL_CONFIG = f"""\
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
print(f"TL config: {TL_CONFIG_FILE}")

# %%
if RUN_TL:
    run_reinvent(TL_CONFIG_FILE, os.path.join(TL_OUT_DIR, "tl.log"))
else:
    print("Skipping TL (RUN_TL=False).")

# %%
chkpts = sorted(glob.glob(os.path.join(TL_OUT_DIR, "TL_photoswitch.model.*.chkpt")))
print(f"TensorBoard: {TB_TL_DIR}")
print(f"Found {len(chkpts)} TL checkpoints")

if chkpts:
    available_epochs = sorted(int(os.path.basename(c).split(".")[-2]) for c in chkpts)
    TL_EPOCH = 30 if 30 in available_epochs else available_epochs[len(available_epochs)//2]
else:
    TL_EPOCH = 30

TL_BEST_CHKPT = os.path.join(TL_OUT_DIR, f"TL_photoswitch.model.{TL_EPOCH}.chkpt")
AGENT_FILE = TL_BEST_CHKPT if os.path.isfile(TL_BEST_CHKPT) else (
    TL_MODEL if os.path.isfile(TL_MODEL) else PRIOR_FILE)
print(f"Agent for RL: {AGENT_FILE}")

# %% [markdown]
# ## §8 — Stage 1: Structural Gates (Fast)
#
# **Scoring**: geometric mean of:
# - **Custom Alerts** (wt 1.0) — SMARTS hard gate (macrocycles, peroxides, metals)
# - **SA Score** (wt 0.6) — reverse sigmoid: easy to synthesise → higher score
# - **QED** (wt 0.3) — gentle drug-likeness bias
#
# ~500 steps, ~20 min CPU / ~5 min GPU.

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
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name   = "Alerts"
weight = 1.0
params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]"
]

[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.6
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.3

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 25
minscore    = 0.5

[inception]
smiles_file = ""
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
# ## §9 — Stage 2: xTB Electronic Screening (Slow)
#
# **Scoring** (Stage 1 gates carried forward +):
# - **XTBLambdaFilter** (wt 0.8) — Gaussian at 450 nm, hard cutoff at 300 nm
# - **XTBIsomerGap** (wt 0.7) — rewards |ΔE(E−Z)| > 5 kcal/mol (Gaussian at 15)
#
# Batch size 40 (xTB ~2–5 s/mol). ~300 steps, ~3–6 h CPU.

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
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name   = "Alerts"
weight = 1.0
params.smarts = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]"
]

[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4

[[stage.scoring.component]]
[stage.scoring.component.XTBLambdaFilter]
[[stage.scoring.component.XTBLambdaFilter.endpoint]]
name   = "xTB_Lambda"
weight = 0.8
params.lambda_cutoff = [300.0]
params.lambda_target = [450.0]
params.lambda_sigma  = [100.0]

[[stage.scoring.component]]
[stage.scoring.component.XTBIsomerGap]
[[stage.scoring.component.XTBIsomerGap.endpoint]]
name   = "EZ_Gap"
weight = 0.7
params.de_min_kcal    = [5.0]
params.de_target_kcal = [15.0]
params.de_sigma_kcal  = [8.0]

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.6

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
# ## §10 — Stage 3: ChemProp Surrogates (Medium Speed)
#
# **Scoring** (Stage 1 structural gates +):
# - **ChemProp λ_max** (wt 0.7) — double-sigmoid 350–550 nm
# - **ChemProp t½** (wt 0.6) — sigmoid favouring log(t½) 3–8
# - **SA Score** (wt 0.5)
#
# Components included only if trained models exist.

# %%
_cp_lam_ok = os.path.isfile(os.path.join(CHEMPROP_LAMBDA_DIR, "model_0", "model.pt"))
_cp_t12_ok = os.path.isfile(os.path.join(CHEMPROP_T12_DIR, "model_0", "model.pt"))
print(f"ChemProp λ_max: {'✓' if _cp_lam_ok else '✗'}  |  ChemProp t½: {'✓' if _cp_t12_ok else '✗'}")

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
params.target_column       = ["lam_E_pipi"]
transform.type = "double_sigmoid"
transform.high = 550.0
transform.low  = 350.0
transform.coef_div  = 200.0
transform.coef_si   = 10.0
transform.coef_se   = 10.0
""" if _cp_lam_ok else ""

_CP_T12 = f"""
[[stage.scoring.component]]
[stage.scoring.component.ChemProp]
[[stage.scoring.component.ChemProp.endpoint]]
name   = "CP_HalfLife"
weight = 0.6
params.checkpoint_dir      = ["{CHEMPROP_T12_DIR}"]
params.rdkit_2d_normalized = [false]
params.features            = [""]
params.target_column       = ["logt12"]
transform.type = "sigmoid"
transform.high = 8.0
transform.low  = 3.0
transform.k    = 0.5
""" if _cp_t12_ok else ""

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
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4
""" + _CP_LAM + _CP_T12 + f"""
[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.7

[inception]
smiles_file = ""
memory_size = 100
sample_size = 10
"""

s3_cfg = os.path.join(S3_DIR, "stage3.toml")
with open(s3_cfg, "w") as f:
    f.write(STAGE3_TOML)
print(f"Stage 3 config: {s3_cfg}")

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

df_s1 = load_rl_csv(S1_DIR, "stage1")
df_s2 = load_rl_csv(S2_DIR, "stage2")
df_s3 = load_rl_csv(S3_DIR, "stage3")

for label, df in [("Stage 1", df_s1), ("Stage 2", df_s2), ("Stage 3", df_s3)]:
    if df.empty:
        print(f"{label}: no data")
    else:
        score_col = "Score" if "Score" in df.columns else df.columns[3]
        print(f"{label}: {len(df)} rows, top score = {df[score_col].max():.3f}")

# %%
def get_top_candidates(df, n=200, min_score=0.4):
    if df.empty:
        return df
    valid = df[df.get("SMILES_state", pd.Series([1]*len(df))) == 1].copy()
    valid = valid.drop_duplicates(subset=["SMILES"])
    score_col = next((c for c in ("Score", "total_score") if c in valid.columns), None)
    if score_col:
        valid = valid[valid[score_col] >= min_score].sort_values(score_col, ascending=False)
    return valid.head(n)

best_df = df_s3 if not df_s3.empty else (df_s2 if not df_s2.empty else df_s1)
top_candidates = get_top_candidates(best_df, n=200, min_score=0.4)
print(f"Top candidates: {len(top_candidates)}")

if not top_candidates.empty:
    disp_cols = [c for c in ["SMILES", "Score", "SA", "QED", "xTB_Lambda", "EZ_Gap"]
                 if c in top_candidates.columns]
    print(top_candidates[disp_cols].head(20).to_string(index=False))

# %% [markdown]
# ## §12 — Molecular Clustering for Diversity
#
# Before spending xTB and DFT budget, we cluster the top candidates by
# Tanimoto similarity on Morgan fingerprints (radius=2, 2048 bits) using
# **Butina clustering** (Taylor, 1995). This ensures we explore a diverse
# set of scaffolds rather than analysing many near-duplicates.
#
# **Parameters you can tune**:
# - `cutoff` (default 0.4): Tanimoto distance threshold. Lower = tighter clusters, more representatives.
# - `n_representatives`: max molecules to carry forward to xTB/DFT.

# %%
def tanimoto_cluster(smiles_list, cutoff=0.4):
    """Butina clustering on Morgan FP (r=2, 2048 bits).
    Returns list of (cluster_id, representative_smiles, cluster_size)."""
    mols = [(smi, Chem.MolFromSmiles(smi)) for smi in smiles_list]
    mols = [(smi, mol) for smi, mol in mols if mol is not None]
    if not mols:
        return []

    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
           for _, mol in mols]

    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    from rdkit.ML.Cluster import Butina
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    results = []
    for cid, cluster in enumerate(clusters):
        rep_idx = cluster[0]
        results.append({
            "cluster_id": cid,
            "representative": mols[rep_idx][0],
            "cluster_size": len(cluster),
            "members": [mols[idx][0] for idx in cluster],
        })
    return results


if not top_candidates.empty:
    all_smiles = top_candidates["SMILES"].tolist()
    clusters = tanimoto_cluster(all_smiles, cutoff=0.4)
    print(f"Found {len(clusters)} clusters from {len(all_smiles)} molecules")
    print(f"Cluster sizes: {[c['cluster_size'] for c in clusters[:15]]}{'...' if len(clusters)>15 else ''}")

    representatives = [c["representative"] for c in clusters]
    print(f"\nCluster representatives: {len(representatives)}")
else:
    clusters = []
    representatives = []
    print("No candidates to cluster.")

# %%
if representatives:
    N_SHOW = min(20, len(representatives))
    rep_mols = [Chem.MolFromSmiles(s) for s in representatives[:N_SHOW]]
    rep_mols = [m for m in rep_mols if m is not None]
    if rep_mols:
        img = Draw.MolsToGridImage(rep_mols, molsPerRow=5,
                                    subImgSize=(300, 250),
                                    legends=[f"Cluster {i}" for i in range(len(rep_mols))])
        display(img)

# %% [markdown]
# ## §13 — Rough xTB Screening on Cluster Representatives
#
# Before DFT (~min/mol), run fast xTB (~sec/mol) on each cluster
# representative to estimate λ_max and ΔE(E/Z). Only the best
# molecules proceed to the expensive TD-DFT step.

# %%
def _xtb_singlepoint(mol3d):
    from xtb.interface import Calculator, Param
    positions   = mol3d.GetConformer().GetPositions()
    atomic_nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
    coords_bohr = positions * 1.8897259886
    calc = Calculator(Param.GFN2xTB, atomic_nums, coords_bohr)
    calc.set_verbosity(0)
    res = calc.singlepoint()
    energy = res.get_energy()
    evals  = res.get_orbital_eigenvalues()
    occs   = res.get_orbital_occupations()
    occ    = evals[occs > 0.5]
    unocc  = evals[occs <= 0.5]
    if len(occ) == 0 or len(unocc) == 0:
        return energy, None
    gap_ev = (unocc[0] - occ[-1]) * 27.2114
    return energy, gap_ev


def _embed_and_optimise(mol):
    mol3d = Chem.AddHs(mol)
    emb = AllChem.ETKDGv3(); emb.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, emb) == -1:
        if AllChem.EmbedMolecule(mol3d, AllChem.ETKDG()) == -1:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=1000)
    except Exception:
        pass
    return mol3d


def _flip_azo_stereo(smiles):
    """Flip E/Z around the first N=N bond by inverting only the second slash."""
    def _invert_second(m):
        return m.group(1) + "N=N" + ("/" if m.group(2) == chr(92) else chr(92))
    result = re.sub(r"([/\\])N=N([/\\])", _invert_second, smiles, count=1)
    if result == smiles:
        return None
    if Chem.MolFromSmiles(result) is None:
        return None
    return result


def xtb_screen_molecule(smiles):
    """Return dict with gap_eV, lam_est_nm, dE_kcal, t12_category or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol3d = _embed_and_optimise(mol)
    if mol3d is None:
        return None
    try:
        energy_E, gap = _xtb_singlepoint(mol3d)
        if gap is None or gap <= 0:
            return None
        lam_est = 1240.0 / (gap * 2.5)

        dE_kcal = None
        t12_cat = "n/a"
        z_smi = _flip_azo_stereo(smiles)
        if z_smi:
            z_mol = Chem.MolFromSmiles(z_smi)
            if z_mol:
                z_mol3d = _embed_and_optimise(z_mol)
                if z_mol3d:
                    energy_Z, _ = _xtb_singlepoint(z_mol3d)
                    dE_kcal = (energy_Z - energy_E) * 627.509
                    if abs(dE_kcal) > 15: t12_cat = "hours-days"
                    elif abs(dE_kcal) > 5: t12_cat = "minutes"
                    else: t12_cat = "seconds"

        return {"smiles": smiles, "gap_eV": gap, "lam_est_nm": lam_est,
                "energy_Ha": energy_E, "dE_kcal": dE_kcal, "t12_cat": t12_cat}
    except Exception:
        return None

# %%
try:
    from xtb.interface import Calculator, Param  # noqa: F401
    _HAS_XTB = True
except ImportError:
    _HAS_XTB = False
    print("⚠ xtb-python not installed — xTB cells will be skipped.")

if _HAS_XTB and representatives:
    n_screen = min(50, len(representatives))
    print(f"Running xTB on {n_screen} cluster representatives...\n")
    print(f"{'#':>3} {'SMILES':<50} {'Gap(eV)':>8} {'λ_est':>7} {'ΔE(kcal)':>10} {'t½':>10}")
    print("-" * 95)

    xtb_results = []
    for i, smi in enumerate(representatives[:n_screen]):
        r = xtb_screen_molecule(smi)
        if r:
            xtb_results.append(r)
            de_str = f"{r['dE_kcal']:.1f}" if r['dE_kcal'] is not None else "n/a"
            print(f"{i+1:>3} {smi[:50]:<50} {r['gap_eV']:>7.3f} {r['lam_est_nm']:>7.0f} "
                  f"{de_str:>10} {r['t12_cat']:>10}")
        else:
            print(f"{i+1:>3} {smi[:50]:<50} {'FAILED':>8}")

    xtb_df = pd.DataFrame(xtb_results)
    print(f"\n✓ {len(xtb_results)}/{n_screen} molecules screened successfully")
elif not _HAS_XTB:
    xtb_df = pd.DataFrame()
    xtb_results = []
else:
    xtb_df = pd.DataFrame()
    xtb_results = []
    print("No cluster representatives to screen.")

# %%
if not xtb_df.empty:
    xtb_ranked = xtb_df.copy()
    xtb_ranked["lam_score"] = xtb_ranked["lam_est_nm"].apply(
        lambda x: np.exp(-0.5 * ((x - 450) / 100) ** 2) if pd.notna(x) else 0)
    xtb_ranked["de_score"] = xtb_ranked["dE_kcal"].apply(
        lambda x: np.exp(-0.5 * ((abs(x) - 15) / 8) ** 2) if pd.notna(x) else 0)
    xtb_ranked["combined"] = xtb_ranked["lam_score"] * 0.5 + xtb_ranked["de_score"] * 0.5
    xtb_ranked = xtb_ranked.sort_values("combined", ascending=False)

    print("Top xTB-ranked cluster representatives:")
    print(xtb_ranked[["smiles", "gap_eV", "lam_est_nm", "dE_kcal", "t12_cat", "combined"]
                     ].head(15).to_string(index=False))

    dft_candidates = xtb_ranked["smiles"].head(8).tolist()
else:
    dft_candidates = representatives[:8] if representatives else []

print(f"\n{len(dft_candidates)} molecules selected for DFT")

# %% [markdown]
# ## §14 — TD-DFT Analysis (PySCF B3LYP/6-31G\*)
#
# | Property | Method | What it tells us |
# |----------|--------|------------------|
# | S₀→Sₙ excitations | TD-DFT | Absorption wavelengths + oscillator strengths |
# | n→π\* vs π→π\* | TD-DFT | Dark vs bright states |
# | Ground state energy | DFT | E-isomer total energy |
# | Metastable state energy | DFT | Z-isomer total energy |
# | ΔE(Z−E) | DFT | Thermodynamic stability |
# | Approx. thermal barrier | BEP | Estimated Z→E barrier |

# %%
def tddft_analysis(smiles, n_states=6, basis="6-31g*", xc="b3lyp"):
    try:
        from pyscf import gto, dft, tddft
    except ImportError:
        print("PySCF not installed. Run: pip install pyscf")
        return None

    mol_rd = Chem.MolFromSmiles(smiles)
    if mol_rd is None:
        return None
    mol3d = _embed_and_optimise(mol_rd)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    atom_block = []
    for atom in mol3d.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_block.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

    pyscf_mol = gto.M(atom="\n".join(atom_block), basis=basis,
                       charge=Chem.GetFormalCharge(mol_rd), spin=0, verbose=0)
    mf = dft.RKS(pyscf_mol)
    mf.xc = xc
    mf.conv_tol = 1e-8
    mf.kernel()
    ground_energy = mf.e_tot

    td = tddft.TDA(mf)
    td.nstates = n_states
    td.kernel()

    excitations = []
    for i, (e_ev, strength) in enumerate(zip(td.e * 27.2114, td.oscillator_strength())):
        lam = 1240.0 / e_ev if e_ev > 0 else None
        if strength < 0.01:
            char = "n→π* (dark)"
        elif strength < 0.1:
            char = "n→π* / mixed"
        else:
            char = "π→π* (bright)"
        excitations.append({
            "state": i + 1, "energy_eV": e_ev, "lambda_nm": lam,
            "osc_strength": strength, "character": char,
        })

    return {"smiles": smiles, "ground_energy_Ha": ground_energy, "excitations": excitations}


def full_photoswitch_dft(smiles, **kwargs):
    e_result = tddft_analysis(smiles, **kwargs)
    if e_result is None:
        return None
    z_smi = _flip_azo_stereo(smiles)
    z_result = tddft_analysis(z_smi, **kwargs) if z_smi else None
    dE_kcal, barrier_est = None, None
    if z_result is not None:
        dE_Ha = z_result["ground_energy_Ha"] - e_result["ground_energy_Ha"]
        dE_kcal = dE_Ha * 627.509
        barrier_est = 0.5 * abs(dE_kcal) + 25.0
    return {"E_isomer": e_result, "Z_isomer": z_result,
            "dE_kcal_mol": dE_kcal, "barrier_est_kcal": barrier_est}


# %%
if RUN_DFT and dft_candidates:
    n_dft = min(5, len(dft_candidates))
    print(f"Running TD-DFT (B3LYP/6-31G*) on {n_dft} candidates...\n")
    dft_results = []

    for i, smi in enumerate(dft_candidates[:n_dft]):
        print(f"[{i+1}/{n_dft}] {smi[:60]}...")
        result = full_photoswitch_dft(smi)
        if result is None:
            print("  → DFT failed, skipping\n")
            continue
        dft_results.append(result)

        e = result["E_isomer"]
        print(f"  E-isomer ground state: {e['ground_energy_Ha']:.6f} Ha")
        print(f"  Excitations:")
        print(f"  {'State':>5} {'eV':>7} {'λ(nm)':>8} {'f':>8} Character")
        for ex in e["excitations"]:
            lam_str = f"{ex['lambda_nm']:.0f}" if ex["lambda_nm"] else "n/a"
            print(f"  S{ex['state']:>4} {ex['energy_eV']:>7.3f} {lam_str:>8} "
                  f"{ex['osc_strength']:>8.4f} {ex['character']}")

        if result["Z_isomer"]:
            print(f"\n  Z-isomer: {result['Z_isomer']['ground_energy_Ha']:.6f} Ha")
            print(f"  ΔE(Z−E) = {result['dE_kcal_mol']:.2f} kcal/mol")
            print(f"  Barrier est. (BEP): {result['barrier_est_kcal']:.1f} kcal/mol")
        print()

    if dft_results:
        print("=" * 80)
        print("DFT Summary")
        print("=" * 80)
        print(f"{'SMILES':<40} {'S1(nm)':>8} {'Bright(nm)':>11} {'ΔE(kcal)':>10} {'Barrier':>9}")
        print("-" * 82)
        for r in dft_results:
            smi = r["E_isomer"]["smiles"][:38]
            excs = r["E_isomer"]["excitations"]
            s1_nm = excs[0]["lambda_nm"] if excs else 0
            bright = [e for e in excs if e["osc_strength"] > 0.05]
            s_bright = bright[0]["lambda_nm"] if bright else 0
            de = r["dE_kcal_mol"] or 0
            bar = r["barrier_est_kcal"] or 0
            print(f"  {smi:<40} {s1_nm:>6.0f} {s_bright:>10.0f} {de:>10.2f} {bar:>9.1f}")
elif RUN_DFT:
    dft_results = []
    print("No candidates for DFT.")
else:
    dft_results = []
    print("DFT skipped (RUN_DFT=False)")

# %% [markdown]
# ## §15 — Energy Landscape Visualization
#
# Two panels per molecule:
# - **Top — Jablonski diagram**: S₀ at 0 eV, excited states as bars at TD-DFT energies.
#   Green = π→π* bright, Purple = n→π* dark, Orange = mixed.
# - **Bottom — Simulated UV-Vis**: Gaussian-broadened stick spectrum (σ = 15 nm).
#   Visible range (400–700 nm) shaded.

# %%
_raw_dft_fallback = [
    {"smiles":"c1ccc(-c2ccc(N=Nc3ccc(-c4ccc(-c5ccccc5)cc4)cc3)cc2)cc1",
     "ground_energy_Ha":-1265.886829,
     "excitations":[
        {"state":1,"energy_eV":3.033,"lambda_nm":409,"osc_strength":0.1524,"character":"π→π* (bright)"},
        {"state":2,"energy_eV":3.779,"lambda_nm":328,"osc_strength":0.2027,"character":"π→π* (bright)"},
        {"state":3,"energy_eV":3.913,"lambda_nm":317,"osc_strength":0.2570,"character":"π→π* (bright)"},
        {"state":4,"energy_eV":4.288,"lambda_nm":289,"osc_strength":0.0065,"character":"n→π* (dark)"},
        {"state":5,"energy_eV":4.296,"lambda_nm":289,"osc_strength":0.0009,"character":"n→π* (dark)"},
        {"state":6,"energy_eV":4.455,"lambda_nm":278,"osc_strength":0.0560,"character":"n→π* / mixed"},
    ]},
    {"smiles":"c1ccc(-c2ccc(N=Nc3ccc(-c4ccccc4)cc3)cc2)cc1",
     "ground_energy_Ha":-1034.835415,
     "excitations":[
        {"state":1,"energy_eV":3.039,"lambda_nm":408,"osc_strength":0.1239,"character":"π→π* (bright)"},
        {"state":2,"energy_eV":3.905,"lambda_nm":318,"osc_strength":0.0090,"character":"n→π* (dark)"},
        {"state":3,"energy_eV":3.946,"lambda_nm":314,"osc_strength":0.4067,"character":"π→π* (bright)"},
        {"state":4,"energy_eV":4.290,"lambda_nm":289,"osc_strength":0.0058,"character":"n→π* (dark)"},
        {"state":5,"energy_eV":4.301,"lambda_nm":288,"osc_strength":0.0027,"character":"n→π* (dark)"},
        {"state":6,"energy_eV":4.546,"lambda_nm":273,"osc_strength":0.0007,"character":"n→π* (dark)"},
    ]},
    {"smiles":"c1ccc(N=Nc2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1",
     "ground_energy_Ha":-1034.835474,
     "excitations":[
        {"state":1,"energy_eV":3.058,"lambda_nm":405,"osc_strength":0.0912,"character":"n→π* / mixed"},
        {"state":2,"energy_eV":3.795,"lambda_nm":327,"osc_strength":0.1549,"character":"π→π* (bright)"},
        {"state":3,"energy_eV":4.252,"lambda_nm":292,"osc_strength":0.1534,"character":"π→π* (bright)"},
        {"state":4,"energy_eV":4.300,"lambda_nm":288,"osc_strength":0.0094,"character":"n→π* (dark)"},
        {"state":5,"energy_eV":4.330,"lambda_nm":286,"osc_strength":0.0443,"character":"n→π* / mixed"},
        {"state":6,"energy_eV":4.436,"lambda_nm":280,"osc_strength":0.0886,"character":"n→π* / mixed"},
    ]},
]


def _e_isomer_data(r):
    if "E_isomer" in r:
        return r["E_isomer"]["smiles"], r["E_isomer"]["excitations"], r["dE_kcal_mol"], r["barrier_est_kcal"]
    return r["smiles"], r["excitations"], None, None


def _exc_color(character):
    if "π→π*" in character and "n→" not in character:
        return "#2ca02c"
    elif "n→π*" in character and "mixed" not in character:
        return "#9467bd"
    return "#ff7f0e"


def simulate_uvvis(excitations, lam_min=250, lam_max=700, sigma=15, n_pts=1000):
    lams = np.linspace(lam_min, lam_max, n_pts)
    total = np.zeros(n_pts); pipi = np.zeros(n_pts)
    npi = np.zeros(n_pts); mixed = np.zeros(n_pts)
    for ex in excitations:
        if not ex["lambda_nm"]:
            continue
        g = ex["osc_strength"] * np.exp(-0.5 * ((lams - ex["lambda_nm"]) / sigma) ** 2)
        total += g
        if "π→π*" in ex["character"] and "n→" not in ex["character"]:
            pipi += g
        elif "n→π*" in ex["character"] and "mixed" not in ex["character"]:
            npi += g
        else:
            mixed += g
    return lams, total, pipi, npi, mixed


def plot_jablonski(ax, excitations, dE_kcal, barrier_kcal, title):
    from matplotlib.patches import Patch
    ax.hlines(0, 0.15, 0.85, colors="#1f77b4", linewidth=3)
    ax.text(0.5, -0.12, "S₀", ha="center", va="top", fontsize=9, color="#1f77b4", fontweight="bold")
    x_positions = np.linspace(0.2, 0.8, len(excitations))
    for x, ex in zip(x_positions, excitations):
        col = _exc_color(ex["character"])
        lw = 1.5 + ex["osc_strength"] * 8
        ax.hlines(ex["energy_eV"], x-0.06, x+0.06, colors=col, linewidth=lw, alpha=0.9)
        ax.annotate("", xy=(x, ex["energy_eV"]-0.05), xytext=(x, 0.05),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.8))
        lam_str = f"{ex['lambda_nm']:.0f}" if ex["lambda_nm"] else "?"
        ax.text(x+0.07, ex["energy_eV"], f"S{ex['state']}\n{lam_str}nm\nf={ex['osc_strength']:.3f}",
                fontsize=5.5, va="center", color=col)
    if dE_kcal is not None:
        ev = 23.0609
        dE_eV = dE_kcal / ev
        ax.hlines(dE_eV, 1.05, 1.75, colors="#d62728", linewidth=2.5, linestyle="--")
        ax.text(1.4, dE_eV+0.05, f"Z  ΔE={dE_kcal:+.1f}\nkcal/mol",
                fontsize=7, color="#d62728", ha="center")
        if barrier_kcal is not None:
            bar_eV = barrier_kcal / ev
            ax.hlines(bar_eV, 0.8, 1.2, colors="#ff7f0e", linewidth=1.5, linestyle=":")
            ax.text(1.0, bar_eV+0.05, f"TS~{barrier_kcal:.0f}", fontsize=6.5, color="#ff7f0e", ha="center")
    ax.set_xlim(0.0, 1.9)
    ax.set_ylim(-0.5, max(e["energy_eV"] for e in excitations)+0.6)
    ax.set_ylabel("Excitation energy (eV)", fontsize=8); ax.set_xticks([])
    ax.set_title(title, fontsize=8, pad=4)
    ax.legend(handles=[Patch(color="#2ca02c", label="π→π* bright"),
                        Patch(color="#9467bd", label="n→π* dark"),
                        Patch(color="#ff7f0e", label="mixed")],
              fontsize=6, loc="upper left", framealpha=0.7)


def plot_uvvis(ax, excitations, title):
    lams, total, pipi, npi, mixed = simulate_uvvis(excitations)
    ax.axvspan(400, 700, alpha=0.07, color="gray", label="visible")
    ax.fill_between(lams, pipi, alpha=0.25, color="#2ca02c")
    ax.fill_between(lams, npi, alpha=0.25, color="#9467bd")
    ax.fill_between(lams, mixed, alpha=0.25, color="#ff7f0e")
    ax.plot(lams, total, "k-", lw=1.2, label="total")
    ax.plot(lams, pipi, color="#2ca02c", lw=1.0, label="π→π*")
    ax.plot(lams, npi, color="#9467bd", lw=1.0, label="n→π*")
    ax.plot(lams, mixed, color="#ff7f0e", lw=1.0, label="mixed")
    for ex in excitations:
        if ex["lambda_nm"]:
            ax.vlines(ex["lambda_nm"], 0, ex["osc_strength"],
                      colors=_exc_color(ex["character"]), lw=2.5, alpha=0.7)
            ax.text(ex["lambda_nm"], ex["osc_strength"]+0.005,
                    f"S{ex['state']}", fontsize=6, ha="center", color=_exc_color(ex["character"]))
    ax.set_xlim(250, 700)
    ax.set_xlabel("λ (nm)", fontsize=8)
    ax.set_ylabel("f (Gaussian-broadened)", fontsize=8)
    ax.set_title(title, fontsize=8, pad=4)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=7)


_use_live = 'dft_results' in dir() and dft_results  # type: ignore[name-defined]
_plot_data = dft_results if _use_live else [  # type: ignore[name-defined]
    {"E_isomer": d, "Z_isomer": None, "dE_kcal_mol": None, "barrier_est_kcal": None}
    for d in _raw_dft_fallback]
if not _use_live:
    print("⚠  Using hard-coded fallback DFT data.")

n_mol = len(_plot_data)
fig, axes = plt.subplots(2, n_mol, figsize=(5*n_mol, 9))
if n_mol == 1:
    axes = axes.reshape(2, 1)

for col, r in enumerate(_plot_data):
    smi, excs, dE, bar = _e_isomer_data(r)
    short = smi[:30] + ("…" if len(smi) > 30 else "")
    plot_jablonski(axes[0, col], excs, dE, bar, f"Jablonski — mol {col+1}\n{short}")
    plot_uvvis(axes[1, col], excs, f"UV-Vis — mol {col+1}\n{short}")

src = "live TD-DFT" if _use_live else "pre-computed TD-DFT"
plt.suptitle(f"B3LYP/6-31G* — {src} ({n_mol} candidates)", fontsize=12, y=1.01)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "dft_energy_landscapes.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out_path}")

# %%
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.axvspan(400, 700, alpha=0.08, color="gray", label="visible")
cmap = plt.cm.tab10  # type: ignore[attr-defined]
for i, r in enumerate(_plot_data):
    _, excs, _, _ = _e_isomer_data(r)
    lams, total, *_ = simulate_uvvis(excs)
    peak = total.max() if total.max() > 0 else 1.0
    ax2.plot(lams, total/peak, color=cmap(i), lw=1.8, label=f"mol {i+1}")
    brightest = max(excs, key=lambda e: e["osc_strength"])
    ax2.axvline(brightest["lambda_nm"], color=cmap(i), lw=0.8, linestyle="--", alpha=0.5)

ax2.set_xlim(250, 700)
ax2.set_xlabel("λ (nm)", fontsize=10)
ax2.set_ylabel("Normalised absorption", fontsize=10)
ax2.set_title("Simulated UV-Vis comparison (all candidates)", fontsize=11)
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "dft_uvvis_comparison.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out2}")

# %% [markdown]
# ## §16 — FarmShare Cluster Instructions
#
# ### 1. Connect
# ```bash
# ssh <sunetid>@rice.stanford.edu
# ```
#
# ### 2. Environment setup (once)
# ```bash
# module load miniconda3
# conda create -n reinvent4 python=3.10 -y
# conda activate reinvent4
#
# git clone https://github.com/MolecularAI/REINVENT4.git
# cd REINVENT4 && pip install -e . && cd ..
#
# conda install -c conda-forge xtb-python rdkit -y
# pip install setuptools==68.2.2 pyscf chemprop==1.5.2 jupytext tensorboard
#
# scp -r /path/to/REINVENT4_photochem/reinvent.prior rice:~/
# scp -r /path/to/REINVENT4_photochem/plugins rice:~/
# scp -r /path/to/REINVENT4_photochem/outputs rice:~/
# ```
#
# ### 3. Submit a batch job
# ```bash
# cat > run_stage2.sh << 'EOF'
# !/bin/bash
# #SBATCH --job-name=photoswitch_s2
# #SBATCH --partition=normal
# #SBATCH --gres=gpu:1
# #SBATCH --time=06:00:00
# #SBATCH --mem=16G
# #SBATCH --cpus-per-task=4
#
# module load miniconda3
# conda activate reinvent4
#
# export PYTHONPATH=$HOME/plugins:$PYTHONPATH
# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
#
# reinvent -d cuda:0 \
#   -l $HOME/outputs/rl_stage2/stage2.log \
#   $HOME/outputs/rl_stage2/stage2.toml
# EOF
#
# sbatch run_stage2.sh
# ```
#
# ### 4. Monitor
# ```bash
# squeue -u $USER
# tail -f ~/outputs/rl_stage2/stage2.log
# ```
#
# ### Expected runtimes (1× GPU)
#
# | Stage | Time |
# |-------|------|
# | TL (50 epochs) | ~10–20 min |
# | Stage 1 (structural) | ~20–40 min |
# | Stage 2 (xTB) | ~3–6 hours |
# | Stage 3 (ChemProp) | ~30–60 min |
# | DFT (5 mols) | ~30 min – 2 h |
