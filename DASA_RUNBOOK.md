# DASA pipeline runbook

Ordered, end to end. Every step says what it writes and how to tell if it worked.

---

## 0. Pre-flight (always, ~3 s)

```bash
python notebooks/test_dasa_sentinels.py
```

60+ assertions against **measured** literature compounds. This is the check whose
absence let a non-DASA corpus survive for months. If anything fails, **stop** — do
not launch a campaign on a red suite.

---

## 1. Generate (Modal, detached, ~4-6 h)

```bash
modal volume rm -r dasa-outputs outputs_dasa      # clear stale state first
modal run --detach modal_dasa.py --xtb-workers 16
```

Runs TL → Stage 1 → Stage 3. Do **not** pass:
- `--resume` — the legacy-checkpoint guard aborts it (by design)
- `--stage2` — xTB is opt-in; see "xTB's role" below

Safe to close the laptop: the app is `detached`, and `detached_disconnected` is
just what that becomes once your client drops. Both keep running.

**Health check while it runs** — the number that matters is the Stage-1 DASA gate
pass rate. Above ~50% is healthy. Single digits means the diversity filter is
choking the target class (this happened at `bucket_size=25` / `minsimilarity=0.6`;
now 400 / 0.8).

---

## 2. Download (local, ~1 min)

The Modal entrypoint's auto-download only runs if your terminal is still attached,
so pull manually:

```bash
modal volume get --force dasa-outputs outputs_dasa outputs_dasa_modal
```

---

## 3. Verify + shortlist  →  `verified_candidates.csv`

```bash
python notebooks/verify_dasa_outputs.py --dir outputs_dasa_modal/outputs_dasa --save-top 300
```

Writes `outputs_dasa_modal/outputs_dasa/verified_candidates.csv` (ranked) and
`verified_top.png`.

Watch for: `SATURATED` warnings on any component (a saturated objective hands the
search to whatever is left — that is the mechanism behind every collapse so far),
and the `WARNING: no trap-escape column` line, which means the ranking is not a
water-switchability ranking at all.

---

## 4. Final candidates + DFT set  →  `final_candidates.csv` / `.png`

```bash
python notebooks/select_for_dft.py
```

Writes, into `outputs_dasa_modal/outputs_dasa/`:

| file | what it is |
|---|---|
| `final_candidates.csv` | **the canonical post-generation artifact** — every shortlisted molecule with donor axes, ΔpKa, acceptor + evidence tier, heavy-atom count after truncation, and `dft_affordable` |
| `final_candidates.png` | top 12, annotated with ΔpKa / acceptor / donor class / truncated size |
| `cluster_representatives.png` | Butina cluster reps — the real scaffold-diversity picture |
| `top_ranked.png` | top 12 by score |
| `dft_set.csv` / `.png` | the molecules step 5 will actually run |

**The affordability screen lives here**, not in the DFT job: molecules bigger than
`--max-heavy` (default 34) *after* chromophore truncation are excluded before any
container is spawned. A previous run put four untruncated molecules on Modal and
burned 5 h without one completed calculation.

---

## 5. DFT verification (Modal, detached)

```bash
modal run --detach modal_dft_v2.py
```

Reads `dft_set.csv` automatically and always prepends the two measured references.
One container per molecule; every result is written to
`/results/dft_v2/<hash>.json` and the volume is committed **immediately**, so a
preemption or an abort costs one molecule, not the run.

```bash
modal volume get --force dasa-outputs dft_v2 ./outputs_dasa_full/dft_v2
```

Partial pulls mid-flight are safe and real.

### What it reports

1. **λ verdict** — reference errors agreeing within 25 nm ⇒ a single systematic
   offset ⇒ calibrated ranking is defensible. Otherwise it says so and refuses to
   endorse ranking.
2. **Solvatochromic slope** — the literature's charge-separation measurement, and
   the non-arbitrary test for whether push-pull is alive. Near-zero or positive =
   the donor is not pushing. The working range is taken from the references, not
   asserted.

---

## Cost controls (why this terminates now)

| control | value | why |
|---|---|---|
| chromophore truncation | on by default | 38 heavy → 20-26, i.e. reference-sized |
| size screen | ≤34 heavy after truncation | never spawn a container that cannot finish |
| DFT geometry optimisation | **off** | 5 h reached step 12 of 300 on the *smallest* molecule; soft modes (Hessian eigenvalue 5.6e-4) make it hopeless. Geometry bias is systematic and calibrated out |
| basis | 6-31G* | same reasoning — the offset is calibrated away |
| TDA roots | 6 | the DASA band is the lowest bright H→L state |
| wall-clock budget | 2400 s/molecule | aborts and **returns partial results** with the stage recorded |
| container timeout | budget + 900 s | budget fires first, so we get data instead of a kill |
| `assert_convergence` | `False` | the default RAISES on cap, discarding hours of work |

**Accuracy comes from calibration, not brute force.** Every molecule runs an
identical protocol, so systematic error from basis and geometry is absorbed by the
fit. The residual σ gate is the test of whether that holds.

---

## xTB's role (it changed)

- **Not** `dG(closed − open)`. That observable inverted the objective and is disabled.
- **`E(zwitterion) − E(keto)`** — which closed tautomer wins. Correctly ordered:
  ChemSci-1 −4.0 (trapped), ChemSci-14 +0.9, indoline +11.3.
- Same quantity as the free in-loop `delta_pka`, at higher fidelity. Cheap one
  shapes the population in Stage 1; xTB discriminates within it in Stage 2, opt-in.

---

## Standing design rules

1. Hard requirement → **gate** (0/1, no gradient, can only exclude).
2. Has a window → **band** (interior optimum, cannot run away).
3. A band is **flat inside** — a satisfied constraint stops pushing. When all bands
   are satisfied, the only remaining objective is diversity.
4. Never let one objective be the sole gradient. That is the mechanism behind every
   collapse: pyrazolidinedione, acylated donors, floppy tails, azole drift.
5. Don't shrink the search space to fix a broken function — fix the function.
   Domain/precedent belongs in **verification**, not the RL gate.
