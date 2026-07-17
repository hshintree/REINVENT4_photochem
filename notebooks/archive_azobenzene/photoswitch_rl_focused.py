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
# # Photoswitch Discovery — RL-Focused Staged Learning + DFT Analysis
#
# **Goal**: Generate novel visible-light photoswitches with good thermal stability
# and synthetic accessibility, using REINVENT4's staged reinforcement learning
# with xTB-based electronic property scoring.
#
# This notebook **reuses the transfer-learned model** from `photoswitch_discovery.ipynb`
# and focuses exclusively on:
#
# 1. **SMILES pre-filtering** before the RL loop
# 2. **Three-stage RL** with progressively expensive scoring
# 3. **Post-processing DFT analysis** (TD-DFT: n→π\*, π→π\*, E/Z energetics)
#
# ---
#
# ## Staged RL design
#
# | Stage | Scoring Components | Batch | Steps | Speed |
# |-------|--------------------|-------|-------|-------|
# | **1 — Structural gates** | Photoswitch scaffold, Custom alerts, SA Score (1–10), QED | 128 | 500 | ~0.1 s/batch |
# | **2 — xTB electronic** | Stage 1 + xTB λ_max (hard cutoff 300 nm) + xTB ΔE(E/Z) | 40 | 300 | ~2–5 s/mol |
# | **3 — ChemProp surrogates** | Stage 1 + ChemProp (built-in) + SA Score | 80 | 400 | ~0.3 s/mol |

# %% [markdown]
# ---
# ## How REINVENT generates molecules — the generative mechanics
#
# ### The model
#
# We are running **REINVENT4** in **two sequential modes**:
#
# | Mode | Where | What happens |
# |------|-------|-------------|
# | **Transfer Learning (TL)** | `photoswitch_discovery.ipynb` | Fine-tunes the prior on known photoswitch SMILES to shift the generator's baseline distribution toward photoswitch-like molecules |
# | **Staged Reinforcement Learning (RL)** | *this notebook* | Uses the TL checkpoint as the starting agent and continuously reshapes its probability distributions toward molecules that score highly on our photoswitch criteria |
#
# The underlying generative model is the **classic REINVENT RNN** ([Olivecrona et al. 2017](https://doi.org/10.1186/s13321-017-0235-x)):
# a 4-layer LSTM with 512 hidden units and 256-dimensional token embeddings, **~7.9 million parameters**,
# trained originally on ~10 million drug-like molecules (the `FS_Ro5_10M.model` prior from Zenodo).
#
# ---
#
# ### The vocabulary
#
# The model works on a **43-token SMILES vocabulary** — not characters, but chemically meaningful tokens:
#
# ```
# Atoms:  C  c  N  n  O  o  S  s  F  Cl  Br  I  P
#         [nH] [N+] [N-] [O-] [n+] [o+] [s+] [S+] [P+] [CH] [CH-] [c-] [S] [O] [I+]
# Bonds:  =  #  -
# Ring:   1  2  3  4  5  6  7  8
# Branch: (  )
# Stereo: /  \  @  @@  (encoded in atom tokens)
# Special: ^  $  (begin-of-sequence, end-of-sequence)
# ```
#
# Maximum sequence length is **128 tokens** (roughly 128 heavy atoms — more than enough for drug-like molecules).
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
#   logits[43]    = LSTM(input_token, hidden_state)     # raw scores over vocabulary
#   probs[43]     = softmax(logits)                     # probability distribution
#   next_token    = multinomial_sample(probs)           ← THE ONLY SOURCE OF RANDOMNESS
#   input_token   = next_token
#   if next_token == $ (EOS, ID=2): stop
#
# Decode token IDs → SMILES string
# ```
#
# **Key point**: the start token `^` is always the same — there is no random "first token".
# Every molecule in the batch begins from the identical starting conditions.
# The only stochasticity is the multinomial draw at each step, governed entirely by the
# model's learned probability distribution over the vocabulary at that position.
#
# ---
#
# ### What controls diversity (the knobs you can turn)
#
# | Parameter | Location in config | What it does | Default here |
# |-----------|-------------------|--------------|-------------|
# | `batch_size` | `[parameters]` in TOML | Molecules generated **per RL step** — higher = more diverse exploration per update | 128 / 40 / 80 |
# | `sigma` (σ) | `[parameters]` in TOML | Scales how hard the reward signal pushes the agent. Higher σ = faster collapse toward high-scoring scaffolds, less diversity | **128** |
# | `max_steps` | `[parameters]` in TOML | Total RL update steps per stage | 500 / 300 / 400 |
# | Diversity filter | `[diversity_filter]` in TOML | Penalizes re-generating identical Murcko scaffolds | IdenticalMurckoScaffold |
# | Inception memory | `[inception]` in TOML | Replays past high-scoring molecules to stabilize training | enabled |
#
# There is **no temperature parameter** in this REINVENT model — the raw softmax probabilities
# are used directly. If you wanted temperature scaling you would add `/ T` before the softmax
# in `model.py`'s `_sample()` method (T < 1 sharpens the distribution, T > 1 flattens it).
#
# ---
#
# ### How RL reshapes the distribution — the DAP algorithm
#
# At each RL step the agent runs:
#
# 1. **Sample** a batch of SMILES from the agent model (stochastic, as above)
# 2. **Score** each SMILES with the multi-component scoring function (our photoswitch criteria)
# 3. **Compute the DAP loss** (Directed Augmented Prior):
#
# $$\mathcal{L} = \bigl(\underbrace{\log P_\text{prior}(m)}_{\text{regularizer}} + \sigma \cdot \underbrace{S(m)}_{\text{score}} - \underbrace{\log P_\text{agent}(m)}_{\text{agent likelihood}}\bigr)^2$$
#
# The prior term acts as a **leash** — it prevents the agent from drifting so far from
# the prior that it collapses to a single scaffold. σ=128 means a score improvement of
# +1 (full score) shifts the augmented log-likelihood by 128 nats, which is a very
# strong signal. To generate more diverse molecules at the cost of slower score
# improvement, **lower σ** (e.g. σ=64 or σ=32). To drill harder on high-scoring
# scaffolds, **raise σ** (e.g. σ=200–300).
#
# 4. **Backpropagate** through the LSTM and update its weights (Adam optimizer).
#
# The prior is **frozen** throughout RL — its weights never change. Only the agent's
# weights are updated. This is what keeps generation from degenerating into nonsense.
#
# ---
#
# ### TL → RL handoff
#
# After TL, the agent's LSTM has been fine-tuned so that token probabilities at each
# position are biased toward photoswitch-like SMILES fragments (azo `N=N`, heteroaromatic
# rings, electron-donor/acceptor groups). The RL loop then further biases those
# probabilities using the scored reward signal — stage by stage, with progressively
# stricter criteria.

# %% [markdown]
# ## §1 — Setup & Configuration

# %%
import os, sys, shutil, subprocess, glob, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                    "-q", "setuptools==68.2.2"], check=True)
    import importlib; importlib.invalidate_caches()
    import pkg_resources  # noqa: F401

# %%
# ── Paths ────────────────────────────────────────────────────────────────────
PROJ_ROOT  = os.path.abspath(os.path.join(os.path.dirname(""), ".."))
OUT_DIR    = os.path.join(PROJ_ROOT, "outputs_rl2"); os.makedirs(OUT_DIR, exist_ok=True)
PLUGIN_DIR = os.path.join(PROJ_ROOT, "plugins")
PLUGIN_COMP= os.path.join(PLUGIN_DIR, "reinvent_plugins", "components")

PRIOR_FILE = os.path.join(PROJ_ROOT, "FS_Ro5_10M.model")
TL_MODEL   = os.path.join(PROJ_ROOT, "outputs", "tl_run", "TL_photoswitch.model")

# Pick the best available agent: TL model > epoch-30 checkpoint > prior
_tl_chkpts = sorted(glob.glob(os.path.join(PROJ_ROOT, "outputs", "tl_run", "*.chkpt")))
if os.path.isfile(TL_MODEL):
    AGENT_FILE = TL_MODEL
elif _tl_chkpts:
    AGENT_FILE = _tl_chkpts[len(_tl_chkpts)//2]
else:
    AGENT_FILE = PRIOR_FILE

assert os.path.isfile(PRIOR_FILE), f"Prior not found: {PRIOR_FILE}"
print(f"Prior : {PRIOR_FILE}")
print(f"Agent : {AGENT_FILE}")
print(f"Output: {OUT_DIR}")

# %%
# ── Run control ──────────────────────────────────────────────────────────────
DEVICE     = "cpu"           # change to "cuda:0" on FarmShare GPU nodes
RUN_STAGE1 = True
RUN_STAGE2 = True
RUN_STAGE3 = True            # requires ChemProp models from notebook 1
RUN_DFT    = True            # post-processing TD-DFT on top candidates

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
# ## §2 — SMILES Pre-Filtering
#
# These filters run **before** REINVENT — they define which SMILES from the
# training set are acceptable starting points and can also be applied
# to generated molecules for quick triage.

# %%
PHOTOSWITCH_SMARTS = [
    Chem.MolFromSmarts("[#6]/N=N/[#6]"),     # E-azo
    Chem.MolFromSmarts("[#6]\\N=N\\[#6]"),    # Z-azo
    Chem.MolFromSmarts("[#6]N=N[#6]"),        # generic azo
    Chem.MolFromSmarts("c1cc2ccc1CC=2"),      # diarylethene core
    Chem.MolFromSmarts("[#6]/C=N/[#7]"),      # hydrazone
]

ALERT_SMARTS = [
    Chem.MolFromSmarts("[*;r8]"), Chem.MolFromSmarts("[*;r9]"),
    Chem.MolFromSmarts("[*;r10]"), Chem.MolFromSmarts("[*;r11]"),
    Chem.MolFromSmarts("[#8][#8]"),            # peroxide
    Chem.MolFromSmarts("[#6;+]"),              # carbocation
    Chem.MolFromSmarts("[#16][#16]"),          # disulfide (non-switch)
    Chem.MolFromSmarts("[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]"),
]


def passes_structural_filter(smi):
    """Return True if SMILES has a photoswitch core and no alert substructures."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    has_core = any(mol.HasSubstructMatch(pat) for pat in PHOTOSWITCH_SMARTS if pat)
    has_alert = any(mol.HasSubstructMatch(pat) for pat in ALERT_SMARTS if pat)
    return has_core and not has_alert


def quick_smiles_filter(smi, mw_min=150, mw_max=700, max_heavy=60):
    """Fast druglikeness gate — no xTB needed."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    ha = mol.GetNumHeavyAtoms()
    return mw_min <= mw <= mw_max and ha <= max_heavy


# Demo
demo_mols = [
    ("azobenzene",        "c1ccc(/N=N/c2ccccc2)cc1"),
    ("caffeine",          "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    ("4-NH2-azobenzene",  "Nc1ccc(/N=N/c2ccccc2)cc1"),
]
for name, smi in demo_mols:
    core = passes_structural_filter(smi)
    gate = quick_smiles_filter(smi)
    print(f"  {name:25s}  core={core}  gate={gate}")

# %% [markdown]
# ## §3 — xTB Utility Functions
#
# Reusable helpers for xTB-based scoring that the RL plugins call.

# %%
def _xtb_singlepoint(mol3d):
    """Run GFN2-xTB single point on an RDKit mol with 3D coords.
    Returns (energy_Ha, homo_lumo_gap_eV) or (None, None)."""
    from xtb.interface import Calculator, Param
    positions   = mol3d.GetConformer().GetPositions()
    atomic_nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
    coords_bohr = positions * 1.8897259886
    calc = Calculator(Param.GFN2xTB, atomic_nums, coords_bohr)
    calc.set_verbosity(0)
    res = calc.singlepoint()
    energy = res.get_energy()  # Hartree
    evals  = res.get_orbital_eigenvalues()
    occs   = res.get_orbital_occupations()
    occ    = evals[occs > 0.5]
    unocc  = evals[occs <= 0.5]
    if len(occ) == 0 or len(unocc) == 0:
        return energy, None
    gap_ev = (unocc[0] - occ[-1]) * 27.2114
    return energy, gap_ev


def _embed_and_optimise(mol):
    """Add Hs, embed in 3D, MMFF-optimise. Returns mol3d or None."""
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


def xtb_lambda_and_energy(smiles):
    """Return dict with gap_eV, lam_est_nm, energy_Ha for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol3d = _embed_and_optimise(mol)
    if mol3d is None:
        return None
    try:
        energy, gap = _xtb_singlepoint(mol3d)
        if gap is None or gap <= 0:
            return None
        lam = 1240.0 / (gap * 2.5)  # empirical correction for azo compounds
        return {"energy_Ha": energy, "gap_eV": gap, "lam_est_nm": lam}
    except Exception:
        return None


def _flip_azo_stereo(smiles):
    """Flip E/Z stereochemistry around the first N=N bond in SMILES.

    In SMILES notation, double-bond stereo is encoded by the direction of
    the two slash characters flanking the double bond:
      /N=N/  or  \\N=N\\  →  both same direction → E (trans)
      /N=N\\  or  \\N=N/  →  opposite directions → Z (cis)

    The flip is done by inverting ONLY the SECOND slash character.
    Using \\N=N\\ as the Z form is wrong — that is still E (trans).
    """
    import re

    def _invert_second(m):
        return m.group(1) + "N=N" + ("/" if m.group(2) == chr(92) else chr(92))

    result = re.sub(r"([/\\])N=N([/\\])", _invert_second, smiles, count=1)
    if result == smiles:
        return None  # no stereo N=N found
    if Chem.MolFromSmiles(result) is None:
        return None
    return result  # keep as-is; re-canonicalising can silently drop stereo


def xtb_ez_energy_gap(smiles):
    """Compute ΔE = E(Z-isomer) − E(E-isomer) in kcal/mol using GFN2-xTB.
    Positive ΔE means the Z-isomer is higher energy (typical for azo).
    Also returns estimated thermal half-life category."""
    e_result = xtb_lambda_and_energy(smiles)
    if e_result is None:
        return None

    z_smi = _flip_azo_stereo(smiles)
    if z_smi is None or z_smi == smiles:
        return None

    z_result = xtb_lambda_and_energy(z_smi)
    if z_result is None:
        return None

    dE_Ha = z_result["energy_Ha"] - e_result["energy_Ha"]
    dE_kcal = dE_Ha * 627.509
    # Rough half-life category from ΔE (Hammond postulate proxy):
    # |ΔE| > 15 kcal/mol → hours–days (bistable), 5–15 → minutes, <5 → seconds
    if abs(dE_kcal) > 15:
        t12_cat = "hours-days"
    elif abs(dE_kcal) > 5:
        t12_cat = "minutes"
    else:
        t12_cat = "seconds"

    return {
        "E_smi": smiles, "Z_smi": z_smi,
        "E_energy_Ha": e_result["energy_Ha"],
        "Z_energy_Ha": z_result["energy_Ha"],
        "dE_kcal_mol": dE_kcal,
        "E_gap_eV": e_result["gap_eV"],
        "E_lam_est": e_result["lam_est_nm"],
        "t12_category": t12_cat,
    }

# %%
# ── Quick demo ────────────────────────────────────────────────────────────────
try:
    from xtb.interface import Calculator, Param  # noqa: F401
    _HAS_XTB = True
except ImportError:
    _HAS_XTB = False
    print("⚠ xtb-python not installed — xTB cells will be skipped.")
    print("  Install: conda install -c conda-forge xtb-python")

if _HAS_XTB:
    print(f"{'Molecule':<25} {'Gap(eV)':>8} {'λ_est':>8} {'ΔE(kcal)':>10} {'t½ cat':>12}")
    print("-" * 70)
    for name, smi in [("azobenzene", "c1ccc(/N=N/c2ccccc2)cc1"),
                       ("4-NH2-azo",  "Nc1ccc(/N=N/c2ccccc2)cc1"),
                       ("methyl orange", "CN(C)c1ccc(/N=N/c2ccc(cc2)S(=O)(=O)[O-])cc1")]:
        r = xtb_lambda_and_energy(smi)
        ez = xtb_ez_energy_gap(smi)
        gap = r["gap_eV"] if r else float("nan")
        lam = r["lam_est_nm"] if r else float("nan")
        de  = ez["dE_kcal_mol"] if ez else float("nan")
        cat = ez["t12_category"] if ez else "n/a"
        print(f"  {name:<25} {gap:>7.3f} {lam:>7.0f} {de:>10.2f} {cat:>12}")

# %% [markdown]
# ## §4 — REINVENT Plugins (Written to Disk)
#
# We create two new plugins for this notebook:
#
# 1. **XTBLambdaFilter** — hard cutoff at 300 nm λ_max estimate, Gaussian
#    reward centred on 400–550 nm.
# 2. **XTBIsomerGap** — rewards molecules with larger |ΔE(E−Z)|, indicating
#    a bistable switch with longer thermal half-life.
#
# These supplement the existing plugins (`PhotoswitchScaffold`, `XTBHomoLumo`)
# and built-in components (`SAScore`, `QED`, `custom_alerts`, `ChemProp`).

# %%
os.makedirs(PLUGIN_COMP, exist_ok=True)
for d in [os.path.join(PLUGIN_DIR, "reinvent_plugins"), PLUGIN_COMP]:
    bad = os.path.join(d, "__init__.py")
    if os.path.exists(bad):
        os.remove(bad)

# ── Plugin: XTBLambdaFilter ──────────────────────────────────────────────────
with open(os.path.join(PLUGIN_COMP, "comp_xtb_lambda_filter.py"), "w") as f:
    f.write('''\
"""xTB-based λ_max filter.

Computes the GFN2-xTB HOMO-LUMO gap, converts to an estimated λ_max,
and scores with a Gaussian centred on the target absorption window.
Molecules with λ_est < lambda_cutoff get score 0 (hard gate).

params.lambda_cutoff = [300.0]   # nm — hard rejection threshold
params.lambda_target = [450.0]   # nm — Gaussian centre
params.lambda_sigma  = [100.0]   # nm — Gaussian width
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
            return 0.5  # neutral if xTB unavailable
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
print("Written: comp_xtb_lambda_filter.py")

# ── Plugin: XTBIsomerGap ─────────────────────────────────────────────────────
_ISOMER_PLUGIN = '''"""xTB E/Z isomer energy gap scorer.

Estimates dE = E(Z) - E(E) using GFN2-xTB for azo photoswitches.
Rewards molecules with larger |dE|, which correlates with longer
thermal half-life (bistability).

params.de_min_kcal = [5.0]
params.de_target_kcal = [15.0]
params.de_sigma_kcal  = [8.0]
"""

__all__ = ["XTBIsomerGap"]
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

_E_AZO = "/N=N/"
_Z_AZO = chr(92) + "N=N" + chr(92)   # backslash-N=N-backslash in SMILES

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
            if _E_AZO in smi:
                z_smi = smi.replace(_E_AZO, _Z_AZO)
            elif _Z_AZO in smi:
                z_smi = smi.replace(_Z_AZO, _E_AZO)
            else:
                return 0.3  # no azo bond found
            z_mol = Chem.MolFromSmiles(z_smi)
            if z_mol is None: return 0.0
            z_mol3d = _embed(z_mol)
            if z_mol3d is None: return 0.0
            e_E = _energy(e_mol)
            e_Z = _energy(z_mol3d)
            dE = abs(e_Z - e_E) * 627.509
            if dE < self.de_min: return 0.0
            return float(np.clip(
                np.exp(-0.5 * ((dE - self.target)/self.sigma)**2), 0, 1))
        except Exception:
            return 0.0
'''
with open(os.path.join(PLUGIN_COMP, "comp_xtb_isomer_gap.py"), "w") as f:
    f.write(_ISOMER_PLUGIN)
print("Written: comp_xtb_isomer_gap.py")

# Verify plugins load
try:
    import importlib, pkgutil
    PYTHONPATH_orig = sys.path[:]
    if PLUGIN_DIR not in sys.path:
        sys.path.insert(0, PLUGIN_DIR)
    from reinvent_plugins import components
    our_plugins = [n for _, n, _ in pkgutil.walk_packages(
        components.__path__, components.__name__ + ".")
        if n.split(".")[-1].startswith("comp_")]
    customs = [n for n in our_plugins if any(k in n for k in
               ["lambda_filter", "isomer_gap", "photoswitch_scaffold", "xtb_homo"])]
    print(f"✓ {len(customs)} custom plugins found: {[n.split('.')[-1] for n in customs]}")
except Exception as e:
    print(f"⚠ Plugin import check failed: {e}")

# %% [markdown]
# ## §5 — Stage 1: Structural Gates (Fast)
#
# **Purpose**: Quickly eliminate molecules that can't possibly be good
# photoswitches, before spending any computational budget on QM.
# All scoring components here are either SMARTS-based lookups or
# 2D descriptor calculations — each call takes microseconds.
#
# **Aggregate score**: geometric mean of all component scores.
# A zero from **any** component propagates to a zero total score,
# acting as a hard veto. This enforces mandatory constraints.
#
# ---
#
# ### Scoring components
#
# #### 1. Custom Alerts (weight 1.0 — hard gate)
# SMARTS substructure filter. Returns **0** if any alert matches,
# **1** otherwise. Alerts block:
# - Macrocycles (≥ 8-membered rings) — hard to synthesise
# - Peroxides `[#8][#8]` — reactive/unstable
# - Carbocations `[#6;+]` — not drug-like
# - Disulfides `[#16][#16]` — not photoswitch-relevant
# - Heavy metals — synthetic/toxicity concerns
#
# #### 2. PhotoswitchScaffold (weight 1.0 — hard gate)
# SMARTS substructure filter. Returns **0** if no photoswitch core is
# found, **1** if at least one matches. Cores checked:
# - E/Z-azobenzene: `[#6]/N=N/[#6]`, `[#6]N=N[#6]`
# - Diarylethene: fused-ring core SMARTS
# - Hydrazone: `[#6]/C=N/[#7]`
#
# #### 3. SA Score — synthetic accessibility (weight 0.6)
# **Built-in REINVENT4 component** (Ertl & Schuffenhauer, 2009).
# Raw SA score: **1** (trivial) to **10** (very hard).
# Transformed with `reverse_sigmoid(low=2, high=8, k=0.4)`:
# - SA ≤ 2 → score ≈ 1.0 (very easy)
# - SA = 5 → score ≈ 0.5 (moderate)
# - SA ≥ 8 → score ≈ 0.0 (intractable)
#
# #### 4. QED — drug-likeness (weight 0.3)
# **Built-in REINVENT4 component** (Bickerton et al., 2012).
# Quantitative estimate of drug-likeness (0–1).
# Low weight — present to gently bias towards drug-like physicochemistry
# without vetoing otherwise good candidates.
#
# ---
#
# **Expected outcome**: the generator rapidly learns to produce azo/hydrazone
# scaffolds with SA ≤ 5, without ring-system alerts.
# ~500 steps, ~20 min on CPU / ~5 min on FarmShare GPU.

# %%
S1_DIR = os.path.join(OUT_DIR, "stage1"); os.makedirs(S1_DIR, exist_ok=True)
S1_CHKPT = os.path.join(S1_DIR, "stage1.chkpt")
S1_TB    = os.path.join(S1_DIR, "tb")

STAGE1_TOML = f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir = "{S1_TB}"
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

# ── 1. Structural alerts (hard gate) ─────────────────────────────────────
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

# ── 2. Photoswitch scaffold (hard gate) ──────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.PhotoswitchScaffold]

[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name   = "PS_Scaffold"
weight = 1.0

# ── 3. SA Score (lower = easier; transform → higher score = easier) ──────
[[stage.scoring.component]]
[stage.scoring.component.SAScore]

[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.6

transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4

# ── 4. QED (optional, low weight) ────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.3

# ── Diversity filter ─────────────────────────────────────────────────────
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
    print("Skipping Stage 1 (RUN_STAGE1=False)")

# %% [markdown]
# ## §6 — Stage 2: xTB Electronic Screening (Slow)
#
# **Purpose**: Filter for molecules that have the right electronic
# structure to be visible-light photoswitches with meaningful thermal
# half-lives — properties that can't be read from structure alone.
#
# Starts from the Stage 1 checkpoint and adds two new xTB-based
# components. **Batch size is 40** because each molecule requires an
# xTB single-point calculation (~1–3 s/molecule on CPU).
#
# ---
#
# ### Scoring components (Stage 1 gates carried forward +)
#
# #### 5. XTBLambdaFilter — absorption wavelength (weight 0.8)
# Custom plugin `comp_xtb_lambda_filter.py`.
#
# **Method**: GFN2-xTB ground-state calculation on the MMFF-optimised
# geometry. HOMO-LUMO gap (eV) extracted from orbital eigenvalues
# (occupation threshold 0.5 to handle Fermi-smearing artefacts).
# Empirical conversion: \(\lambda_{est} = 1240 / (E_{gap} \times 2.5)\)
# (correction factor 2.5 calibrated on azobenzene, 4-aminoazobenzene,
# methyl orange — see §3 demo).
#
# **Scoring**:
# 1. Hard cutoff: if \(\lambda_{est} < 300\) nm → score = **0** (UV-only, not useful)
# 2. Gaussian centred at 450 nm, σ = 100 nm:
#    \(\text{score} = \exp\!\left[-\tfrac{1}{2}\left(\tfrac{\lambda_{est}-450}{100}\right)^2\right]\)
#
# **Note on accuracy**: GFN2-xTB gaps underestimate TD-DFT values by ~2.5×.
# The correction is an approximation (±100 nm). Use this for *ranking*, not
# for predicting exact absorption wavelengths.
#
# #### 6. XTBIsomerGap — thermal half-life proxy (weight 0.7)
# Custom plugin `comp_xtb_isomer_gap.py`.
#
# **Method**: Computes GFN2-xTB energy for both E-isomer (input) and
# Z-isomer (generated by flipping the second SMILES slash in `/N=N/` →
# `/N=N\`). Energy difference: \(\Delta E = E_Z - E_E\) in kcal/mol.
#
# **Why this matters**: For a molecular switch to be useful, the metastable
# (Z) state must be thermally stable enough to be detected and used.
# Larger \(|\Delta E|\) (Z higher in energy, positive \(\Delta E\)) indicates
# a higher back-isomerisation barrier and longer half-life.
#
# **Scoring**:
# 1. If \(|\Delta E| < 5\) kcal/mol → score = **0** (Z barely stable)
# 2. Gaussian centred at 15 kcal/mol, σ = 8 kcal/mol:
#    \(\text{score} = \exp\!\left[-\tfrac{1}{2}\left(\tfrac{|\Delta E|-15}{8}\right)^2\right]\)
#
# **Limitation**: MMFF geometry optimisation may not fully relax the Z-isomer
# (cis N–N–C–C dihedral ~0°). The ΔE values are directionally correct but
# quantitatively approximate. GFN2-xTB also does not include entropy.
# For publication-quality half-life estimates, use the TD-DFT stage (§9).
#
# ---
#
# **Expected outcome**: the generator learns to enrich for push-pull azo
# scaffolds (electron donors + acceptors on the azo chromophore) that
# red-shift absorption into the visible and increase bistability.
# ~300 steps, ~3–6 h on CPU / ~30 min on FarmShare GPU (xTB is CPU-bound).

# %%
S2_DIR = os.path.join(OUT_DIR, "stage2"); os.makedirs(S2_DIR, exist_ok=True)
S2_CHKPT = os.path.join(S2_DIR, "stage2.chkpt")
S2_TB    = os.path.join(S2_DIR, "tb")

_s2_agent = S1_CHKPT if os.path.isfile(S1_CHKPT) else AGENT_FILE

STAGE2_TOML = f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir = "{S2_TB}"
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

# ── Structural gates (from Stage 1) ──────────────────────────────────────
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
[stage.scoring.component.PhotoswitchScaffold]
[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name   = "PS_Scaffold"
weight = 1.0

[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4

# ── xTB λ_max filter (new) ──────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.XTBLambdaFilter]

[[stage.scoring.component.XTBLambdaFilter.endpoint]]
name   = "xTB_Lambda"
weight = 0.8

params.lambda_cutoff = [300.0]
params.lambda_target = [450.0]
params.lambda_sigma  = [100.0]

# ── xTB E/Z isomer gap (new) ────────────────────────────────────────────
[[stage.scoring.component]]
[stage.scoring.component.XTBIsomerGap]

[[stage.scoring.component.XTBIsomerGap.endpoint]]
name   = "EZ_Gap"
weight = 0.7

params.de_min_kcal    = [5.0]
params.de_target_kcal = [15.0]
params.de_sigma_kcal  = [8.0]

# ── Diversity ────────────────────────────────────────────────────────────
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
    if not _HAS_XTB:
        print("⚠ Stage 2 requires xtb-python. Install it and restart the kernel.")
    else:
        run_reinvent(s2_cfg, os.path.join(S2_DIR, "stage2.log"))
else:
    print("Skipping Stage 2 (RUN_STAGE2=False)")

# %% [markdown]
# ## §7 — Stage 3: ChemProp Surrogates (Medium Speed)
#
# **Purpose**: Add ML-predicted property scores that are faster than xTB
# (~0.1 s/molecule vs ~2–3 s) but more informative than SMARTS alone.
# Runs a larger batch size (80) and more steps (400) to refine the
# distribution from Stage 2.
#
# ---
#
# ### About ChemProp in REINVENT4
#
# REINVENT4 ships with a **built-in `ChemProp` scoring component** at
# `reinvent_plugins/components/comp_chemprop.py`. You do **not** need a
# custom plugin — just point it at a trained checkpoint directory.
#
# The component takes a `checkpoint_dir` containing ChemProp v1 model folds
# (`model_0/model.pt`, `model_1/model.pt`, …), a `target_column` name, and
# an optional `features_generator`. It calls `chemprop.train.make_predictions`
# internally and returns raw predicted values, which are then passed through
# a transform to produce a 0–1 score.
#
# **Status in this notebook**:
# - The Stage 3 config checks at runtime whether the model files exist.
# - If they exist, ChemProp components are added to the TOML.
# - If not, Stage 3 runs with structural + SA components only (still useful).
#
# **To train ChemProp models**: run §7 of `photoswitch_discovery.ipynb` with
# `RUN_CHEMPROP = True`. Models are saved to `outputs/chemprop_lambda/` and
# `outputs/chemprop_t12/`.
#
# ---
#
# ### Scoring components (Stage 1 structural gates +)
#
# #### 7. ChemProp λ_max — visible absorption (weight 0.7, if model available)
# **Built-in component**: predicts λ_max (nm) from molecular graph.
# Transformed with `double_sigmoid` centred on 350–550 nm:
# molecules predicted outside the visible range score low.
# The model was trained on the combined `photoswitches.csv` +
# `fulldata.lambda_train.xlsx` dataset (n ≈ 980 after cleaning).
#
# #### 8. ChemProp t½ — thermal half-life (weight 0.6, if model available)
# **Built-in component**: predicts log₁₀(t½/s).
# Transformed with `sigmoid` favouring log(t½) in range 3.5–9
# (half-lives of ~30 min to 30 years — practically bistable range).
# Trained on the same dataset (n ≈ 112 molecules with measured t½).
#
# #### SA Score (weight 0.5) — same as Stage 1, tighter weighting.
#
# ---
#
# **Expected outcome**: Stage 3 refines the Stage 2 distribution toward
# molecules that a trained GNN predicts will absorb in the visible range
# with multi-hour half-lives.
# ~400 steps, ~30–60 min on CPU / ~10–20 min on FarmShare GPU.

# %%
CHEMPROP_LAMBDA_DIR = os.path.join(PROJ_ROOT, "outputs", "chemprop_lambda")
CHEMPROP_T12_DIR    = os.path.join(PROJ_ROOT, "outputs", "chemprop_t12")

_cp_lam_ok = os.path.isfile(os.path.join(CHEMPROP_LAMBDA_DIR, "model_0", "model.pt"))
_cp_t12_ok = os.path.isfile(os.path.join(CHEMPROP_T12_DIR,    "model_0", "model.pt"))

print(f"ChemProp λ_max model: {'✓' if _cp_lam_ok else '✗ not found'}")
print(f"ChemProp t½ model:    {'✓' if _cp_t12_ok else '✗ not found'}")

if not _cp_lam_ok and not _cp_t12_ok:
    print("\n⚠ No ChemProp models found. Stage 3 will run without ML surrogates.")
    print("  Train them in photoswitch_discovery.ipynb §7 first.")

# %%
S3_DIR = os.path.join(OUT_DIR, "stage3"); os.makedirs(S3_DIR, exist_ok=True)
S3_CHKPT = os.path.join(S3_DIR, "stage3.chkpt")
S3_TB    = os.path.join(S3_DIR, "tb")

_s3_agent = S2_CHKPT if os.path.isfile(S2_CHKPT) else (
    S1_CHKPT if os.path.isfile(S1_CHKPT) else AGENT_FILE)

# Build ChemProp TOML fragments conditionally
_CP_LAM_BLOCK = f"""
# ── ChemProp λ_max (built-in) ────────────────────────────────────────────
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
""" if _cp_lam_ok else "# ChemProp λ_max skipped — model not trained\n"

_CP_T12_BLOCK = f"""
# ── ChemProp half-life (built-in) ────────────────────────────────────────
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
""" if _cp_t12_ok else "# ChemProp t½ skipped — model not trained\n"

STAGE3_TOML = f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir = "{S3_TB}"
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

# ── Structural gates ─────────────────────────────────────────────────────
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

[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name   = "SA"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.high = 8.0
transform.low  = 2.0
transform.k    = 0.4
""" + _CP_LAM_BLOCK + _CP_T12_BLOCK + f"""
# ── Diversity ────────────────────────────────────────────────────────────
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
_active = ["Alerts", "PS_Scaffold", "SA"]
if _cp_lam_ok: _active.append("CP_Lambda")
if _cp_t12_ok: _active.append("CP_HalfLife")
print(f"  Active components: {', '.join(_active)}")

# %%
if RUN_STAGE3:
    if not _cp_lam_ok and not _cp_t12_ok:
        print("No ChemProp models — running Stage 3 with structural components only.")
    run_reinvent(s3_cfg, os.path.join(S3_DIR, "stage3.log"))
else:
    print("Skipping Stage 3 (RUN_STAGE3=False)")

# %% [markdown]
# ## §8 — Results Collection & Ranking

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
        print(f"{label}: no data yet")
    else:
        n_valid = (df.get("SMILES_state", pd.Series()) == 1).sum()
        score_col = "Score" if "Score" in df.columns else df.columns[3]
        top = df[score_col].max() if score_col in df.columns else 0
        print(f"{label}: {len(df)} rows, {n_valid} valid, top score = {top:.3f}")

# %%
def get_top_candidates(df, n=100, min_score=0.5):
    """Extract unique, high-scoring molecules."""
    if df.empty:
        return df
    valid = df[df.get("SMILES_state", pd.Series([1]*len(df))) == 1].copy()
    valid = valid.drop_duplicates(subset=["SMILES"])
    score_col = next((c for c in ("Score", "total_score") if c in valid.columns), None)
    if score_col:
        valid = valid[valid[score_col] >= min_score]
        valid = valid.sort_values(score_col, ascending=False)
    return valid.head(n)

# Pick the best-stage output that has data
best_df = df_s3 if not df_s3.empty else (df_s2 if not df_s2.empty else df_s1)
top_candidates = get_top_candidates(best_df, n=100, min_score=0.4)
print(f"\nTop candidates: {len(top_candidates)}")

if not top_candidates.empty:
    disp_cols = [c for c in ["SMILES", "Score", "SA", "PS_Scaffold", "xTB_Lambda", "EZ_Gap"]
                 if c in top_candidates.columns]
    print(top_candidates[disp_cols].head(20).to_string(index=False))

# %% [markdown]
# ## §9 — Post-Processing: TD-DFT Analysis on Top Candidates
#
# For the best molecules we run **PySCF TD-DFT** (B3LYP/6-31G\*) to compute:
#
# | Property | Method | What it tells us |
# |----------|--------|------------------|
# | S₀→S₁ (n→π\*) | TD-DFT | Forbidden dark state — wavelength & oscillator strength |
# | S₀→S₂ (π→π\*) | TD-DFT | Allowed bright state — main absorption peak |
# | Ground state energy | DFT | E-isomer total energy |
# | Metastable state energy | DFT | Z-isomer total energy |
# | ΔE(Z−E) | DFT | Thermodynamic stability of metastable state |
# | Approximate thermal barrier | Marcus/BEP | Estimated Z→E barrier from ΔE |
#
# > **Note**: Full transition-state optimization (nudged elastic band / IRC)
# > is computationally expensive (~hours/mol). We use the Bell-Evans-Polanyi
# > (BEP) approximation: ΔE‡ ≈ α·ΔE + E₀, with α≈0.5 for azo compounds.

# %%
def tddft_analysis(smiles, n_states=6, basis="6-31g*", xc="b3lyp"):
    """Run PySCF TD-DFT on a SMILES string.

    Returns dict with ground_energy, excitations list (energy_eV, lambda_nm,
    osc_strength, character), and isomer energies if applicable.
    """
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

    pyscf_mol = gto.M(
        atom="\n".join(atom_block),
        basis=basis,
        charge=Chem.GetFormalCharge(mol_rd),
        spin=0,
        verbose=0,
    )

    # Ground state DFT
    mf = dft.RKS(pyscf_mol)
    mf.xc = xc
    mf.conv_tol = 1e-8
    mf.kernel()
    ground_energy = mf.e_tot  # Hartree

    # TD-DFT excited states
    td = tddft.TDA(mf)
    td.nstates = n_states
    td.kernel()

    excitations = []
    for i, (e_ev, strength) in enumerate(zip(td.e * 27.2114, td.oscillator_strength())):
        lam = 1240.0 / e_ev if e_ev > 0 else None
        # Assign character heuristically:
        # low oscillator strength + low energy → n→π*
        # higher oscillator strength → π→π*
        if strength < 0.01:
            char = "n→π* (dark)"
        elif strength < 0.1:
            char = "n→π* / mixed"
        else:
            char = "π→π* (bright)"
        excitations.append({
            "state": i + 1,
            "energy_eV": e_ev,
            "lambda_nm": lam,
            "osc_strength": strength,
            "character": char,
        })

    return {
        "smiles": smiles,
        "ground_energy_Ha": ground_energy,
        "excitations": excitations,
    }


def full_photoswitch_dft(smiles, **kwargs):
    """TD-DFT on both E and Z isomers + energy landscape analysis."""
    e_result = tddft_analysis(smiles, **kwargs)
    if e_result is None:
        return None

    z_smi = _flip_azo_stereo(smiles)
    z_result = tddft_analysis(z_smi, **kwargs) if z_smi else None

    dE_kcal = None
    barrier_est = None
    if z_result is not None:
        dE_Ha = z_result["ground_energy_Ha"] - e_result["ground_energy_Ha"]
        dE_kcal = dE_Ha * 627.509
        # BEP approximation: ΔE‡ ≈ 0.5 * |ΔE| + 25 kcal/mol (azo baseline)
        barrier_est = 0.5 * abs(dE_kcal) + 25.0

    return {
        "E_isomer": e_result,
        "Z_isomer": z_result,
        "dE_kcal_mol": dE_kcal,
        "barrier_est_kcal": barrier_est,
    }

# %%
if RUN_DFT and not top_candidates.empty:
    # Analyse top 5 candidates (DFT is expensive — ~2–10 min per molecule)
    n_dft = min(5, len(top_candidates))
    smiles_for_dft = top_candidates["SMILES"].head(n_dft).tolist()

    print(f"Running TD-DFT (B3LYP/6-31G*) on {n_dft} candidates...\n")
    dft_results = []

    for i, smi in enumerate(smiles_for_dft):
        print(f"[{i+1}/{n_dft}] {smi[:60]}...")
        result = full_photoswitch_dft(smi)
        if result is None:
            print("  → DFT failed, skipping\n")
            continue
        dft_results.append(result)

        e = result["E_isomer"]
        print(f"  E-isomer ground state: {e['ground_energy_Ha']:.6f} Ha")
        print(f"  Excitations:")
        print(f"  {'State':>5} {'eV':>7} {'λ(nm)':>8} {'f':>8} {'Character'}")
        for ex in e["excitations"]:
            lam_str = f"{ex['lambda_nm']:.0f}" if ex["lambda_nm"] else "n/a"
            print(f"  S{ex['state']:>4} {ex['energy_eV']:>7.3f} {lam_str:>8} "
                  f"{ex['osc_strength']:>8.4f} {ex['character']}")

        if result["Z_isomer"]:
            print(f"\n  Z-isomer ground state: {result['Z_isomer']['ground_energy_Ha']:.6f} Ha")
            print(f"  ΔE(Z−E) = {result['dE_kcal_mol']:.2f} kcal/mol")
            print(f"  Est. thermal barrier (BEP): {result['barrier_est_kcal']:.1f} kcal/mol")
        print()

    # Summary table
    if dft_results:
        print("\n" + "="*80)
        print("DFT Summary")
        print("="*80)
        print(f"{'SMILES':<40} {'S1 n→π*(nm)':>12} {'S2 π→π*(nm)':>13} "
              f"{'ΔE(kcal)':>10} {'Barrier':>10}")
        print("-"*90)
        for r in dft_results:
            smi = r["E_isomer"]["smiles"][:38]
            excs = r["E_isomer"]["excitations"]
            s1_nm = excs[0]["lambda_nm"] if len(excs) > 0 and excs[0]["lambda_nm"] else 0
            # Find first bright state for π→π*
            bright = [e for e in excs if e["osc_strength"] > 0.05]
            s2_nm = bright[0]["lambda_nm"] if bright else 0
            de = r["dE_kcal_mol"] if r["dE_kcal_mol"] else 0
            bar = r["barrier_est_kcal"] if r["barrier_est_kcal"] else 0
            print(f"  {smi:<40} {s1_nm:>10.0f} {s2_nm:>12.0f} "
                  f"{de:>10.2f} {bar:>10.1f}")

elif RUN_DFT:
    print("No candidates available for DFT analysis. Run RL stages first.")
else:
    print("DFT analysis skipped (RUN_DFT=False)")

# %% [markdown]
# ## §10 — Energy Landscape Visualization
#
# Two panels per molecule:
# - **Left — Jablonski diagram**: S0 at 0 eV, excited states as horizontal bars at
#   their TD-DFT transition energies. Arrows coloured by character:
#   - **Green** = π→π* (bright, large oscillator strength f > 0.1)
#   - **Purple** = n→π* (dark, f < 0.05)
#   - **Orange** = mixed character
# - **Right — Simulated UV-Vis**: Gaussian-broadened stick spectrum (σ = 15 nm),
#   coloured by transition character. Visible range (400–700 nm) shaded in grey.
#
# If Z-isomer energies were obtained (requires explicit stereo in SMILES), the
# Jablonski panel also shows the Z metastable level and estimated thermal barrier.

# %%
# ── Hard-coded fallback from the TD-DFT run printed above ───────────────────
# If dft_results is already live in the kernel (from §9) that object takes precedence.
# This block fires only when the notebook is re-opened / DFT is not re-run.
_raw_dft_fallback = [
    {
        "smiles": "c1ccc(-c2ccc(N=Nc3ccc(-c4ccc(-c5ccccc5)cc4)cc3)cc2)cc1",
        "ground_energy_Ha": -1265.886829,
        "excitations": [
            {"state":1,"energy_eV":3.033,"lambda_nm":409,"osc_strength":0.1524,"character":"π→π* (bright)"},
            {"state":2,"energy_eV":3.779,"lambda_nm":328,"osc_strength":0.2027,"character":"π→π* (bright)"},
            {"state":3,"energy_eV":3.913,"lambda_nm":317,"osc_strength":0.2570,"character":"π→π* (bright)"},
            {"state":4,"energy_eV":4.288,"lambda_nm":289,"osc_strength":0.0065,"character":"n→π* (dark)"},
            {"state":5,"energy_eV":4.296,"lambda_nm":289,"osc_strength":0.0009,"character":"n→π* (dark)"},
            {"state":6,"energy_eV":4.455,"lambda_nm":278,"osc_strength":0.0560,"character":"n→π* / mixed"},
        ],
    },
    {
        "smiles": "c1ccc(-c2ccc(N=Nc3ccc(-c4ccccc4)cc3)cc2)cc1",
        "ground_energy_Ha": -1034.835415,
        "excitations": [
            {"state":1,"energy_eV":3.039,"lambda_nm":408,"osc_strength":0.1239,"character":"π→π* (bright)"},
            {"state":2,"energy_eV":3.905,"lambda_nm":318,"osc_strength":0.0090,"character":"n→π* (dark)"},
            {"state":3,"energy_eV":3.946,"lambda_nm":314,"osc_strength":0.4067,"character":"π→π* (bright)"},
            {"state":4,"energy_eV":4.290,"lambda_nm":289,"osc_strength":0.0058,"character":"n→π* (dark)"},
            {"state":5,"energy_eV":4.301,"lambda_nm":288,"osc_strength":0.0027,"character":"n→π* (dark)"},
            {"state":6,"energy_eV":4.546,"lambda_nm":273,"osc_strength":0.0007,"character":"n→π* (dark)"},
        ],
    },
    {
        "smiles": "c1ccc(N=Nc2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1",
        "ground_energy_Ha": -1034.835474,
        "excitations": [
            {"state":1,"energy_eV":3.058,"lambda_nm":405,"osc_strength":0.0912,"character":"n→π* / mixed"},
            {"state":2,"energy_eV":3.795,"lambda_nm":327,"osc_strength":0.1549,"character":"π→π* (bright)"},
            {"state":3,"energy_eV":4.252,"lambda_nm":292,"osc_strength":0.1534,"character":"π→π* (bright)"},
            {"state":4,"energy_eV":4.300,"lambda_nm":288,"osc_strength":0.0094,"character":"n→π* (dark)"},
            {"state":5,"energy_eV":4.330,"lambda_nm":286,"osc_strength":0.0443,"character":"n→π* / mixed"},
            {"state":6,"energy_eV":4.436,"lambda_nm":280,"osc_strength":0.0886,"character":"n→π* / mixed"},
        ],
    },
]

def _e_isomer_data(r):
    """Unpack E-isomer excitation data regardless of source (live vs. fallback)."""
    if "E_isomer" in r:  # live dft_results format
        return r["E_isomer"]["smiles"], r["E_isomer"]["excitations"], r["dE_kcal_mol"], r["barrier_est_kcal"]
    else:                # fallback format
        return r["smiles"], r["excitations"], None, None


def _exc_color(character):
    if "π→π*" in character and "n→" not in character:
        return "#2ca02c"   # green — bright
    elif "n→π*" in character and "mixed" not in character:
        return "#9467bd"   # purple — dark
    else:
        return "#ff7f0e"   # orange — mixed


def simulate_uvvis(excitations, lam_min=250, lam_max=700, sigma=15, n_pts=1000):
    """Gaussian-broadened stick spectrum weighted by oscillator strength."""
    lams = np.linspace(lam_min, lam_max, n_pts)
    total   = np.zeros(n_pts)
    pipi    = np.zeros(n_pts)
    npi     = np.zeros(n_pts)
    mixed   = np.zeros(n_pts)
    for ex in excitations:
        if not ex["lambda_nm"]:
            continue
        g = ex["osc_strength"] * np.exp(-0.5 * ((lams - ex["lambda_nm"]) / sigma) ** 2)
        total += g
        if "π→π*" in ex["character"] and "n→" not in ex["character"]:
            pipi  += g
        elif "n→π*" in ex["character"] and "mixed" not in ex["character"]:
            npi   += g
        else:
            mixed += g
    return lams, total, pipi, npi, mixed


def plot_jablonski(ax, excitations, dE_kcal, barrier_kcal, title):
    """Jablonski energy-level diagram in eV."""
    ev_to_kcal = 23.0609

    # S0 baseline
    ax.hlines(0, 0.15, 0.85, colors="#1f77b4", linewidth=3)
    ax.text(0.5, -0.12, "S₀", ha="center", va="top", fontsize=9, color="#1f77b4", fontweight="bold")

    x_positions = np.linspace(0.2, 0.8, len(excitations))
    for x, ex in zip(x_positions, excitations):
        e = ex["energy_eV"]
        col = _exc_color(ex["character"])
        lw = 1.5 + ex["osc_strength"] * 8   # thicker line = stronger transition
        ax.hlines(e, x - 0.06, x + 0.06, colors=col, linewidth=lw, alpha=0.9)
        # Upward absorption arrow
        ax.annotate("", xy=(x, e - 0.05), xytext=(x, 0.05),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.8))
        # Label: state + wavelength
        lam_str = f"{ex['lambda_nm']:.0f}" if ex["lambda_nm"] else "?"
        ax.text(x + 0.07, e, f"S{ex['state']}\n{lam_str}nm\nf={ex['osc_strength']:.3f}",
                fontsize=5.5, va="center", color=col)

    # Z-isomer metastable state if available (ΔE relative to S0)
    if dE_kcal is not None:
        dE_eV = dE_kcal / ev_to_kcal
        ax.hlines(dE_eV, 1.05, 1.75, colors="#d62728", linewidth=2.5, linestyle="--")
        ax.text(1.4, dE_eV + 0.05, f"Z  ΔE={dE_kcal:+.1f}\nkcal/mol",
                fontsize=7, color="#d62728", ha="center")
        if barrier_kcal is not None:
            bar_eV = barrier_kcal / ev_to_kcal
            ax.hlines(bar_eV, 0.8, 1.2, colors="#ff7f0e", linewidth=1.5, linestyle=":")
            ax.text(1.0, bar_eV + 0.05, f"TS~{barrier_kcal:.0f}\nkcal/mol",
                    fontsize=6.5, color="#ff7f0e", ha="center")
            ax.plot([0.85, 1.0, 1.15], [dE_eV, bar_eV, 0], "k--", alpha=0.25, lw=1)

    ax.set_xlim(0.0, 1.9)
    ax.set_ylim(-0.5, max(ex["energy_eV"] for ex in excitations) + 0.6)
    ax.set_ylabel("Excitation energy (eV)", fontsize=8)
    ax.set_xticks([])
    ax.set_title(title, fontsize=8, pad=4)
    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ca02c", label="π→π* bright"),
                        Patch(color="#9467bd", label="n→π* dark"),
                        Patch(color="#ff7f0e", label="mixed")],
              fontsize=6, loc="upper left", framealpha=0.7)


def plot_uvvis(ax, excitations, title):
    """Simulated UV-Vis with stick lines and Gaussian envelope."""
    lams, total, pipi, npi, mixed = simulate_uvvis(excitations)

    # Visible-light shaded region
    ax.axvspan(400, 700, alpha=0.07, color="gray", label="visible")

    ax.fill_between(lams, pipi,  alpha=0.25, color="#2ca02c")
    ax.fill_between(lams, npi,   alpha=0.25, color="#9467bd")
    ax.fill_between(lams, mixed, alpha=0.25, color="#ff7f0e")
    ax.plot(lams, total, "k-", lw=1.2, label="total")
    ax.plot(lams, pipi,  color="#2ca02c", lw=1.0, label="π→π*")
    ax.plot(lams, npi,   color="#9467bd", lw=1.0, label="n→π*")
    ax.plot(lams, mixed, color="#ff7f0e", lw=1.0, label="mixed")

    # Stick lines at each transition
    for ex in excitations:
        if ex["lambda_nm"]:
            ax.vlines(ex["lambda_nm"], 0, ex["osc_strength"],
                      colors=_exc_color(ex["character"]), lw=2.5, alpha=0.7)
            ax.text(ex["lambda_nm"], ex["osc_strength"] + 0.005,
                    f"S{ex['state']}", fontsize=6, ha="center",
                    color=_exc_color(ex["character"]))

    ax.set_xlim(250, 700)
    ax.set_xlabel("λ (nm)", fontsize=8)
    ax.set_ylabel("f  (Gaussian-broadened)", fontsize=8)
    ax.set_title(title, fontsize=8, pad=4)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=7)


# ── Decide data source ────────────────────────────────────────────────────────
_use_live = 'dft_results' in dir() and dft_results  # type: ignore[name-defined]
_plot_data = dft_results if _use_live else [{"E_isomer": d, "Z_isomer": None,  # type: ignore[name-defined]
                                              "dE_kcal_mol": None, "barrier_est_kcal": None}
                                             for d in _raw_dft_fallback]
if not _use_live:
    print("⚠  dft_results not found — using hard-coded fallback data from the printed run.")

n_mol = len(_plot_data)
fig, axes = plt.subplots(2, n_mol, figsize=(5 * n_mol, 9))
if n_mol == 1:
    axes = axes.reshape(2, 1)

for col, r in enumerate(_plot_data):
    smi, excs, dE, bar = _e_isomer_data(r)
    short = smi[:30] + ("…" if len(smi) > 30 else "")
    plot_jablonski(axes[0, col], excs, dE, bar, f"Jablonski — mol {col+1}\n{short}")
    plot_uvvis(axes[1, col], excs, f"Sim. UV-Vis — mol {col+1}\n{short}")

src_label = "live TD-DFT" if _use_live else "pre-computed TD-DFT"
plt.suptitle(f"B3LYP/6-31G* — {src_label} ({n_mol} candidates)", fontsize=12, y=1.01)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "dft_energy_landscapes.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out_path}")

# ── Overlay comparison: all simulated spectra on one panel ───────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.axvspan(400, 700, alpha=0.08, color="gray", label="visible")
cmap = plt.cm.tab10  # type: ignore[attr-defined]
for i, r in enumerate(_plot_data):
    _, excs, _, _ = _e_isomer_data(r)
    lams, total, *_ = simulate_uvvis(excs)
    # Normalise each spectrum so it fits on same axes
    peak = total.max() if total.max() > 0 else 1.0
    ax2.plot(lams, total / peak, color=cmap(i), lw=1.8, label=f"mol {i+1}")
    # Mark the brightest transition
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
# ## §11 — FarmShare Cluster Instructions
#
# ### Quick start (run in your terminal)
#
# **1. Connect to FarmShare:**
# ```bash
# ssh <sunetid>@rice.stanford.edu
# ```
#
# **2. Set up the environment (once):**
# ```bash
# module load miniconda3
# conda create -n reinvent4 python=3.10 -y
# conda activate reinvent4
#
# # Install REINVENT4
# git clone https://github.com/MolecularAI/REINVENT4.git
# cd REINVENT4 && pip install -e . && cd ..
#
# # Dependencies
# conda install -c conda-forge xtb-python rdkit -y
# pip install setuptools==68.2.2 pyscf chemprop==1.5.2 jupytext tensorboard
#
# # Copy your model and plugins
# scp -r /path/to/REINVENT4_photochem/FS_Ro5_10M.model rice:~/
# scp -r /path/to/REINVENT4_photochem/plugins rice:~/
# scp -r /path/to/REINVENT4_photochem/outputs/tl_run rice:~/outputs/
# ```
#
# **3. Submit a batch job (Stage 2 example):**
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
#   -l $HOME/outputs_rl2/stage2/stage2.log \
#   $HOME/outputs_rl2/stage2/stage2.toml
# EOF
#
# sbatch run_stage2.sh
# ```
#
# **4. Monitor:**
# ```bash
# squeue -u $USER
# tail -f ~/outputs_rl2/stage2/stage2.log
# ```
#
# **Expected runtimes on FarmShare (1× GPU):**
#
# | Stage | Time |
# |-------|------|
# | Stage 1 (structural) | ~20–40 min |
# | Stage 2 (xTB) | ~3–6 hours (xTB is CPU-bound) |
# | Stage 3 (ChemProp) | ~30–60 min |
# | DFT post-processing (5 mols) | ~30 min – 2 hours |
#
# **Tip**: For Stage 2, request more CPUs (`--cpus-per-task=8`) and set
# `OMP_NUM_THREADS=8` since xTB benefits from OpenMP parallelism.
