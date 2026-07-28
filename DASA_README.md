# Water-Switchable DASA Discovery — Current Pipeline

Goal: generate **Donor–Acceptor Stenhouse Adducts (DASAs)** that are, *simultaneously in water*:
1. **OPEN** (the colored triene form is the populated state — not trapped closed),
2. **VISIBLE** (open-form λmax ~540–600 nm),
3. **SOLUBLE**, and (aspirationally) **switchable**.

## The core problem (why this is hard)

Water is very polar, so it **stabilizes the closed cyclopentenone** form. The closed form is a
colorless deep-UV absorber. So a soluble DASA dropped in water tends to **collapse to the closed,
colorless state** ("dark switching") — useless. Pure-water reversible switching has **not been
achieved in the literature** (only cosolvent / cyclodextrin encapsulation). This project searches
for molecules that resist that collapse.

The objective therefore requires the SAME molecule to be **open-favored in water AND visible AND
soluble** — enforced by a `geometric_mean` score (conjunctive: a low score on any objective sinks
the whole molecule, so it can only reward the intersection).

## Results (deliverables)

Everything is in **`outputs_dasa_full/`**:
- **`final_candidates.csv`** + **`final_candidates_both_arch.png`** — the 8 final candidates,
  two architectures (tethered + aniline), all anti-trapped + soluble + visible-acceptor.
- `stage2_intersection_winners.png`, `stage2_top_structures.png` — structure grids.
- `dft_lambda_by_solvent.csv`, `switchability_dft_vs_xtb.txt`, `switchability_xtb.txt` — verification data.
- `verified_candidates.csv` — ranked shortlist from an earlier run.

## The pipeline

| Stage | What it enforces | Key components |
|-------|------------------|----------------|
| **TL** | learn a DASA-shaped prior from `enumerate_dasa_aqueous()` (160 mols, 65% aniline / 15% tethered) | transfer learning |
| **Stage 1** | structure + **color** (canonical acceptor + real amine donor) + **architecture** + banded solubility + SA | `DASAScaffold`, `DASAColor`, `DASA2ndGen`, `AqueousSolubility`, `SAScore`, custom_alerts |
| **Stage 2** | **anti-trap** — GFN2-xTB geometry-opt ΔE(closed−open) **in water**, banded to the switchable window; closed form = min(zwitterion, neutral-keto) | `DASATrap` |
| **Stage 3** | re-tighten structure/color/solubility (no architecture pressure) | `DASAScaffold`, `DASAColor`, `AqueousSolubility`, `SAScore` |
| **Verify** | color (TD-DFT λmax, +186 nm offset) + switchability (xTB ΔG, anchored) | `dft_verify.py` (LOCAL), `verify_dasa_outputs.py` |

## How to run

```bash
# Full run on Modal (detached; survives disconnect). Keep stages SHORT — spot preemption is
# aggressive (~hourly) and kills long single-container jobs.
modal run --detach modal_dasa.py --stage2-steps 12 --xtb-workers 16

# Resume from a saved Stage-1 checkpoint (skips TL+Stage1):
modal run --detach modal_dasa.py --resume --stage2-steps 12

# Verify candidates locally (Modal DFT is FUTILE for ~40-heavy molecules — preemption; use local):
python notebooks/verify_dasa_outputs.py --dir <outputs_dir> --save-top 500
```

## Tuning the aniline : tethered ratio  ← (the important knob)

The RL **mode-collapses to a single donor architecture** every run — the diversity filter only
prevents *scaffold* collapse, not *architecture* collapse. So:

- **You cannot get a mixed aniline+tethered population from one run.** Equal weights
  (`aniline = tethered = 1.0`) just flip which one it collapses to.
- **To bias toward ANILINE** (recommended — anilines are the thermodynamically water-escapable
  2nd-gen class; their neutral keto closed form is what `DASATrap` can reward): in
  `plugins/reinvent_plugins/components/comp_dasa_2ndgen.py` set
  `_SCORE = {"aniline": 1.0, "tethered": 0.6, "dialkyl": 0.3, "other": 0.2}` — the run collapses to aniline.
- **To bias toward TETHERED**: raise tethered above aniline. (Note: our tethered heads are
  tertiary → no neutral-keto form → `DASATrap` only sees their zwitterion; their real benefit is
  *kinetic*, which we don't yet model. Tethered is best assessed by DFT barrier / experiment.)
- **To get BOTH in the final pool in a chosen ratio**: run each architecture separately and combine
  the candidate CSVs in the proportion you want (e.g. two aniline runs + one tethered run for a
  ~2:1 aniline:tethered pool). This is how `final_candidates.csv` was built.

## Open scientific gaps (next steps)

1. **Color in water, not gas**: the λmax check should be run in an aqueous continuum (solvatochromism),
   and conditioned on the molecule actually being OPEN in water (i.e. only trust color for molecules
   that pass anti-trap).
2. **Switching kinetics**: `DASATrap` is thermodynamic (open-vs-closed ΔG) only. Whether a molecule
   *switches* needs the open→closed **barrier** (TS-DFT) — this is where tethered rigidification
   pays off and where a positive anchor (the JACS 2022 water-switcher) would calibrate "enough."
3. **Synthesis panel** spanning both architectures.

## Hard-won operational notes

- **Spot preemption** dominates: keep RL stages < ~45 min; use `retries=3`; long jobs die.
- **xTB-opt reliability**: `_OPT_CYCLES=20` + `_OPT_TIMEOUT=150` (bounds work + margin). Do **not**
  oversubscribe workers > CPU cores — it makes every opt time out → all-zero scores.
- **Modal DFT is futile** for large molecules (can't finish before preemption) — use local pyscf.
- **Stale volume data**: the `dasa-outputs` Modal volume persists across runs; a fresh run is
  identified by the `[DASATrap] … band[…]` log banner, not by CSV presence.
