# Archived scripts

Superseded, kept for provenance. **Nothing live imports any of these** — verified by
reference scan before the move; the only mentions in live files are prose in
docstrings/comments.

Every one of these was written against the **pre-2026-07-28 DASA core**, which placed
the hydroxyl on the carbon bonded to the acceptor instead of on C2. That is a
constitutional isomer of a DASA, not a DASA, so any number they produce is about the
wrong molecule. Do not resurrect one without first re-checking it against
`dasa_chem.is_legacy_core` and `notebooks/test_dasa_sentinels.py`.

| archived | superseded by | why |
|---|---|---|
| `modal_dft.py` | `modal_dft_v2.py` | no truncation, no size screen, no wall-clock budget; hung for hours |
| `notebooks/dft_verify.py` | `notebooks/dft_verify_v2.py` | max-oscillator-strength state pick, no calibration ladder, no planarity check |
| `modal_dft_final.py` | `modal_dft_v2.py` | the run that produced the 229 nm anchor error (legacy core + xTB geometry) |
| `modal_switch.py` | `comp_dasa_trap.py` (Stage 2) | scored dG(closed-open), which INVERTS the objective; the right observable is E(zwitterion) - E(keto) |
| `modal_kinetics.py` | — | xTB barrier screen built on the legacy core; its "anchor-validated" result validated the wrong molecule |
| `notebooks/dasa_complete.py` | `run_dasa_trial.py` + `verify_dasa_outputs.py` + `select_for_dft.py` | old monolithic notebook pipeline, still wired to DASASwitchability |
| `comp_dasa_switchability.py` | `comp_dasa_trap.py` / `comp_dasa_trapescape.py` | single-point xTB on MMFF geometries; noisy enough to invert the truth |
| `comp_dasa_2ndgen.py` | `comp_dasa_trapescape.py` | 4-bin donor-architecture lookup; replaced by the continuous dpKa coordinate |
