# Photoswitch Discovery with REINVENT4

A REINVENT4-based computational pipeline for *de novo* design of visible-light-activated photoswitches using transfer learning and multi-stage reinforcement learning with quantum-chemistry-informed scoring.

---

## What we're trying to do

We want to generate novel photoswitch molecules — compounds that reversibly isomerise between two stable states (typically E and Z) when illuminated at specific wavelengths.  The ideal candidate:

- **Absorbs in the visible range** (400–650 nm), so it can be driven with simple LEDs rather than UV light
- **Has a long thermal half-life** (hours to years) so the metastable isomer persists without continuous irradiation
- **Shows high photostationary state conversion** (PSS ≥ 65 %) so most of the population actually switches
- **Is synthetically tractable** — no metals, peroxides, or otherwise reactive fragments

The prior REINVENT model was trained on general organic chemistry (ChEMBL).  Left alone it would generate drug-like molecules with no bias toward photoswitch chemistry.  Our workflow corrects this in two steps:

1. **Transfer learning** narrows the prior's distribution toward molecules that look like known photoswitches.
2. **Reinforcement learning** with a staged reward function then drives the agent to maximise the properties above.

---

## Datasets

| File | Source | Size | Used for |
|------|--------|------|---------|
| `data/photoswitches.csv` | [Josch Helfferich Photoswitch Dataset](https://github.com/Ryan-Rhys/The-Photoswitch-Dataset) | ~405 molecules | TL corpus + RL predictor training |
| `data/fulldata.lambda_train.xlsx` | Extended azo / heteroarene training set | ~718 molecules | TL corpus + λ_max predictor training |

Both datasets are merged, deduplicated by InChIKey, and split scaffold-aware 80/20 for transfer learning.

---

## Notebook: `photoswitch_discovery.ipynb`

The single notebook walks the full pipeline from raw data to batch-submitted RL jobs.

### §1 — FarmShare Setup
SSH instructions, Micromamba environment creation, REINVENT4 install, and data upload.  All commands are also shown for local macOS/Linux.

### §2–§3 — Imports & Data Loading
Standard imports (RDKit, pandas, numpy, matplotlib) and loading both raw datasets.

### §4 — Data Cleaning: "Good Switches"
Filters both datasets to keep only chemically plausible photoswitches:

- Valid, canonicalised SMILES (RDKit `MolStandardize`)
- Contains a recognised photoswitch motif (N=N azo, N=C hydrazone, C=N imine)
- λ_max ≥ 380 nm (near-visible or visible)
- PSS ≥ 65 % (for strict/predictor set)
- t₁/₂ between 1 hour and 100 years (bistability window)
- No forbidden SMARTS: peroxides, disulfides, charged carbon, metals, large rings
- Deduplicated by InChIKey

Outputs two files:
- `photoswitch_tl_train.smi` / `photoswitch_tl_val.smi` — SMILES for TL
- `photoswitch_strict.csv` — property-labelled set for training scoring models

### §5 — Quantum Chemistry: xTB (GFN2-xTB)
Introduces `xtb-python` for semiempirical quantum mechanics:
- **HOMO-LUMO gap** as a fast λ_max proxy (~1–3 s/molecule on CPU)
- Empirical correction: `λ_est ≈ 1240 / (gap_eV × 0.75)` validated for azo compounds
- `batch_xtb_screen()` for post-processing top RL outputs

### §6 — TD-DFT with PySCF (Optional)
For the final shortlist only.  `tddft_excitations()` runs B3LYP/6-31G* to get:
- Vertical S₀→Sₙ excitation energies and oscillator strengths
- Accurate λ_abs (not used in the RL loop — too slow)

### §7 — ChemProp Surrogate Models
Two ChemProp (v1, GNN) regression models trained on cleaned data:
- **λ_max model** — predicts π→π* absorption wavelength (nm)
- **log(t₁/₂) model** — predicts thermal half-life in log₁₀ seconds

These are fast enough (~ms per molecule in batch) to score every generated molecule during RL.

### §8 — Custom REINVENT Scoring Plugins
Four plugin files written to `plugins/reinvent_plugins/components/`:

| Plugin | Stage | Speed | Description |
|--------|-------|-------|-------------|
| `comp_photoswitch_scaffold.py` | 1 | <0.1 ms/mol | SMARTS scaffold filter + forbidden-group check |
| `comp_visible_abs_chemprop.py` | 2 | ~5 ms/mol | ChemProp λ_max → trapezoidal 400–650 nm score |
| `comp_half_life_chemprop.py` | 2 | ~5 ms/mol | ChemProp log(t₁/₂) → bistability window score |
| `comp_xtb_homo_lumo.py` | 3 | ~2 s/mol | GFN2-xTB HOMO-LUMO gap → Gaussian visible score |

### §9 — Transfer Learning
`run_type = "transfer_learning"`, 50 epochs, scaffold-aware train/val split.  TensorBoard output is inspected to choose the best checkpoint (lowest val loss, validity ≥ 95 %).

### §10–§12 — Three-Stage Reinforcement Learning

**Stage 1 — Structural filter (fast)**
Scaffold filter + custom alerts + QED.  No ML models.  Runs in seconds per step.  Goal: establish a high-quality base distribution of synthetically reasonable photoswitches.

**Stage 2 — ML-predicted properties (medium)**
Adds ChemProp λ_max and t₁/₂ predictors to the reward.  Geometric mean of all components.  Diversity filter on Murcko scaffolds prevents collapse.  Inception replay memory seeds exploration.

**Stage 3 — Quantum-chemistry reward (slow)**
Adds the xTB HOMO-LUMO gap scorer.  Batch size reduced to 40 (from 100) to keep step time manageable.  This is the stage where the model is genuinely guided by electronic structure.

### §13 — Results Analysis
Load RL CSVs, filter high-scoring unique molecules, optionally run xTB or TD-DFT on the top candidates, display in a molecule grid.

### §14–§15 — FarmShare Batch Scripts
Auto-generated Slurm `sbatch` scripts for each run stage.  Instructions for interactive smoke-test (`srun --partition=interactive`), GPU migration, and monitoring with `squeue`.

---

## Dependency Quick Reference

```bash
# Both local and FarmShare — install into reinvent4 conda env
conda install -c conda-forge xtb-python rdkit -y
pip install pyscf chemprop==1.6.1 mordred mols2grid jupytext tensorboard
```

> `chemprop==1.6.1` is the latest v1 release compatible with REINVENT4's scoring API.
> Do **not** install chemprop v2+ — the API is incompatible.

---

## Reward Design Rationale

All stages use a **geometric mean** aggregator.  A single zero-scoring component (e.g., a forbidden substructure alert) kills the total score, preventing the agent from finding loopholes.

The three-stage structure exists because:
- Running xTB on every batch step would take hours per epoch
- ML surrogates run in milliseconds and cover ~80% of the signal
- xTB adds physically grounded selectivity in the final stage for the most promising region of chemical space

The ChemProp surrogates are trained on experimental data from the same photoswitch distribution we're sampling from, making them well-calibrated for this chemical series rather than general drug-like molecules.

---

## File Layout After Running the Notebook

```
outputs/
├── photoswitch_tl_train.smi
├── photoswitch_tl_val.smi
├── photoswitch_strict.csv
├── chemprop_lambda/          ← trained λ_max ChemProp model
├── chemprop_t12/             ← trained log(t₁/₂) ChemProp model
├── tl_run/                   ← TL checkpoints + TensorBoard logs
├── rl_stage1/                ← Stage 1 checkpoints + CSV outputs
├── rl_stage2/                ← Stage 2 checkpoints + CSV outputs
├── rl_stage3/                ← Stage 3 (xTB) checkpoints + CSV outputs
└── scripts/                  ← Slurm batch scripts for FarmShare
plugins/
└── reinvent_plugins/
    └── components/
        ├── comp_photoswitch_scaffold.py
        ├── comp_visible_abs_chemprop.py
        ├── comp_half_life_chemprop.py
        └── comp_xtb_homo_lumo.py
```
