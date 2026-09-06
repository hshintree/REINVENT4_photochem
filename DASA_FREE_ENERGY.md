# Gibbs energies of the open and closed states — literature scan and plan

Companion to [DASA.md](DASA.md). Scope: what the literature actually computes for
DASA open/closed thermodynamics, how accurate it is, and how to get a usable
free-energy signal into the staged RL loop.

Written 2026-09-04. Everything below is sourced; the last section lists what I
could *not* verify.

---

## 1. Nobody computes "ΔG(open)" and "ΔG(closed)". They compute a landscape.

Two independent groups have published ground-state DASA energy landscapes, and they
converged on the same protocol.

**Lerch, Feringa, Buma et al., *Angew. Chem. Int. Ed.* 2018, 57, 8063** ("Solvent
Effects on the Actinic Step"), open access at PMC6055754. Figure 1c is an
*"energy level diagram in kcal mol⁻¹ for 1 in selected solvents obtained at the
M06-2X/6-31+G(d)/SMD level of theory"*. Species: **A** (open triene) → **A′**
(photoisomer, C2–C3) → **A″** (after C3–C4 rotation) → **B** (zwitterion) or
**B′** (neutral). Numbers are in their Table S8.1.

Their conclusions, which are the physics we care about:

- In **polar protic** solvents (water, methanol) the cyclization product is the
  **zwitterion B**, it is *thermodynamically below A*, and the backward B→A″ barrier
  is high. Both together explain irreversible cyclization in water.
- In **aprotic** solvents the cyclization stops at the **neutral B′**, whose
  stability relative to A *decreases* with increasing polarity.
- Barriers for A′→A″ and A″→B/B′ are lowest in toluene — which is why toluene is
  the solvent where first-generation DASAs actually switch.

**Stricker, Peterson, … Read de Alaniz, *Chem* 2023, 9, 1994–2005** (open access).
Protocol: geometries **M06-2X/6-31+G(d,p)**, single points **M06-2X/def2-QZVP**,
**SMD**, Gaussian 16, with the dielectric scanned to model polarity. Chosen, in
their words, for *"inexpensive cost and reasonable performance compared with those
of previously benchmarked methods"* (their SI §4.1).

Their state labels differ and are more complete — **A, B, B′, C_enol, C_zwit,
C_keto** — and the mechanism they read off the landscape is exactly the trap:

> DASA-1 and DASA-2 have a less stabilized C_enol isomer and are therefore driven to
> other more favored ring-closed isomers, such as C_zwit or C_keto, which might have
> higher barriers for isomerization back to the open form.

That is the trap stated as a landscape feature, not as a sign of ΔG. It corroborates
the reading already in `comp_dasa_trap.py`.

Also from that paper, the polarity mechanism at bond level: raising polarity lowers
C3–C4 double-bond character (so the A→B rotation barrier *falls*), raises C2–C3
character (so the B→B′ barrier *rises*), and lengthens the C5···C1 distance (so
electrocyclization gets harder).

**Takeaway:** the field's default is **M06-2X + SMD**, minimum-to-minimum, no
conformer ensembles, no explicit solvent, no COSMO-RS. It is calibrated to support
narratives ("this barrier goes up with polarity"), not to rank congeners to
0.5 kcal/mol. Do not adopt it and assume you have inherited its accuracy.

---

## 2. "Open" and "closed" are each several species

A single ΔG between two SMILES is not the observable. What NMR measures at dark
equilibrium is an *ensemble against an ensemble*:

| manifold | species | notes |
|---|---|---|
| open | A, A′, A″ (Feringa) / A, B, B′ (Read de Alaniz) | rotamers about C2–C3 and C3–C4; only A is coloured |
| closed | **C_enol** | first-formed 4π product, before proton transfer |
| closed | **C_zwit** | ammonium + acceptor enolate — the water-locked state |
| closed | **C_keto** | neutral, proton on the acceptor carbon — the escape route |

plus conformers of each, and the closed forms are **4,5-disubstituted
cyclopentenones with two diastereomers** (Hemmer 2018 notes the ¹H NMR of the closed
form is a mixture of diastereomers).

So the quantity to compute is

&nbsp;&nbsp;&nbsp;&nbsp;`ΔG = −RT ln( Σ_closed e^(−G_i/RT) / Σ_open e^(−G_j/RT) )`

a Boltzmann-weighted ensemble free energy, not a difference of two optimised minima.

**Gap in our code:** `dasa_common._cyclize` builds **C_zwit** (`open_to_closed`) and
**C_keto** (`open_to_closed_keto`) only. **C_enol is missing**, and the Chem 2023
landscape makes C_enol's stability relative to A one of the two discriminators
between DASAs that switch and DASAs that trap. Worth building before any ΔG run.

---

## 3. The precision problem — quantify it before choosing a method

The experimental observable is the dark equilibrium by ¹H NMR. Converting
Hemmer et al. (*JACS* 2018, 140, 10425 — 2-methylindoline donor, eight carbon-acid
acceptors, CDCl₃) to free energies with `ΔG = −RT ln([closed]/[open])`,
RT = 0.592 kcal/mol at 298 K:

| % open (measured) | acceptor (Hemmer numbering) | ΔG°(open→closed), kcal/mol |
|---|---|---|
| ~5 % | indandione **6** (approx., read off Fig. 3b) | −1.75 |
| ~45 % | Meldrum **1**, barbituric **2** | −0.12 |
| ~75–85 % | pyrazolidinedione **3**, Me-isoxazolone **4**, hydroxypyridone **8** | +0.65 … +1.03 |
| >95 % | CF₃-isoxazolone **5**, CF₃-pyrazolone **7** | ≥ +1.75 |

**The entire third-generation acceptor library spans about 4.4 kcal/mol**, and the
design-relevant distinction — "45 % open" vs "80 % open" vs ">95 % open" — is
**0.8–0.9 kcal/mol per step**. A factor of ten in K_eq is 1.36 kcal/mol.

Against that:

- **GFN2-xTB / ALPB**: our own measurement (2026-07-28, `dasa-current-state`) put the
  ΔG(closed−open) magnitudes 4–7 kcal/mol away from the measured CHCl₃ equilibria.
  Signs were right; magnitudes were noise on this scale.
- **M06-2X/SMD** (the literature protocol): typical errors on solution-phase
  isomerization free energies are ~1–3 kcal/mol, i.e. **the same size as the entire
  measured range**.
- Implicit solvation is worst exactly where we need it: a **neutral ⇌ zwitterion**
  equilibrium **in water**. SMD does well on neutral tautomers (~0.4 kcal/mol on
  acetylacetone keto-enol) and much worse on zwitterions — the glycine
  neutral/zwitterion equilibrium is the standard cautionary case, where implicit
  models miss specific hydrogen bonding that decides the answer.

**Conclusion: absolute ΔG cannot rank DASAs.** Anything built on absolute computed
ΔG will reproduce the 2026-07-28 failure in a more expensive form. Two ways out,
both used below: relative (ΔΔG within a congeneric series, §6) and switching to a
better-conditioned observable (§4).

---

## 4. The kinetic observable is better conditioned than the thermodynamic one

New and directly on target: **Blandón-Cumbreras, … Pischel, *JACS* 2026, 148,
130–134**, "Reversible Photoswitching of Donor–Acceptor Stenhouse Adducts in Water"
(open access, PMC12814171). First-generation adamantyl DASAs, encapsulated by
cucurbit[7]- and [8]uril.

Measured dark-switching half-lives, and the Eyring barriers they imply
(ΔG‡ = RT·[ln(k_BT/h) − ln k], k = ln2/t½, 298 K):

| system | t½ | ΔG‡ (kcal/mol) |
|---|---|---|
| DASA 2 (Meldrum), free, 10 vol% THF/water | 29 min | 22.1 |
| DASA 1 (barbituric), free, 10 vol% THF/water | 38 min | 22.3 |
| DASA 1 · CB7, water | 12.4 h | 24.0 |
| DASA 1 · CB8, water | 45.1 h | 24.8 |
| DASA 2 · CB8, water | 112.2 h | 25.3 |
| DASA 2 · CB7, water | 586.3 h | 26.3 |

That is a **4 kcal/mol span measured to well under 0.1 kcal/mol**, versus the
4.4 kcal/mol span of the equilibrium measured to a few tenths. The barrier is the
better-determined quantity *and* it is the physically correct one for the trap —
which is the conclusion we had already reached from the wrong direction.

Two further things from that paper worth acting on:

1. **Steric bulk at the amino donor is a validated anti-trap handle**, independent of
   electronics: *"significant steric effects can overrule the electronic effects"*.
   We have no steric term at all; ΔpKa is a purely electronic coordinate.
2. **DASA.md §1 needs an update.** It says pure-water switching is reached "only via
   cosolvent or cyclodextrin encapsulation". Add cucurbit[n]uril (2026). The paper
   also quantifies the prior art we describe: the cyclodextrin route gave *~1 %*
   linear form at acidic pH, and the tethered second-generation route needs
   **60 vol% THF**. Pure-water switching of a *free* DASA remains unprecedented — the
   project premise holds, but the bar is now explicit.

---

## 5. Method ladder

| tier | method | cost / molecule | what it is honestly good for |
|---|---|---|---|
| 0 | `delta_pka` (have it) | free | shaping a population; no per-molecule resolution beyond substituent counting |
| 1 | GFN2-xTB //ALPB, opt | ~1–5 min | **tautomer ordering only** (C_zwit vs C_keto). Validated: −4.0 / +0.9 / +11.3 kcal/mol, correctly ordered. Not the equilibrium. |
| 2 | r²SCAN-3c (or M06-2X/6-31+G(d,p)→def2-QZVP) + CPCM/SMD, full Hessian → G | 1–6 h | ΔΔG within one architecture; matches the literature protocol so results are comparable |
| 3 | CREST → CENSO ensembles, ωB97M-V or DLPNO-CCSD(T)/CBS, COSMO-RS, mRRHO quasi-harmonic G | 1–3 days | genuine ensemble free energies; the honest number for a shortlist |
| 4 | explicit water: QM/MM FEP, or an ML/MM potential | days–weeks | the only tier that can defensibly do the **zwitterion in water** |

Notes:

- **Tier 2 is the workhorse.** r²SCAN-3c is the current best-practice composite
  (Bursch et al., *Angew. Chem. Int. Ed.* 2022, 61, e202205735, "Best-Practice DFT
  Protocols"). Running M06-2X/SMD *as well* on a subset buys direct comparability
  with Angew 2018 / Chem 2023 for roughly nothing.
- **Thermal corrections matter and are usually done wrong.** Use Grimme's
  quasi-harmonic mRRHO treatment of low-lying modes, not raw harmonic entropies —
  a ring closure kills several soft torsions, so this is a real, systematic term
  that hits exactly our reaction coordinate.
- **Conformers matter more than the functional here.** CENSO exists because
  conformational energies need ~0.1–0.2 kcal/mol accuracy to get Boltzmann
  populations right, which is the same tolerance our ΔG needs.
- **Tooling:** ORCA 6 (free for academics) is a much better fit than pyscf for tiers
  2–3 — r²SCAN-3c, CPCM/SMD, analytic Hessians, DLPNO-CCSD(T), and it pairs natively
  with CREST/CENSO/xtb. pyscf has no SMD and no composite methods, and its Hessians
  are expensive. Our Modal images already build micromamba environments, so
  `xtb + crest + censo + orca` is the same kind of image as `modal_dft_v2.py`.

---

## 6. Compute ΔΔG, not ΔG

Systematic error in solvation and in the functional is largely constant across a
congeneric series. So report

&nbsp;&nbsp;&nbsp;&nbsp;`ΔΔG(candidate) = ΔG(candidate) − ΔG(anchor of the same architecture)`

with anchors that have a measured equilibrium in the *same solvent*:

| architecture | anchor | measured | source |
|---|---|---|---|
| dialkyl donor | Me₂N / 1,3-diMe-barbituric | 86 % open, CDCl₃ | Mallo et al., *Chem. Sci.* 2018, 9, 8242 (cmpd 1) |
| aryl donor | aniline / barbituric | 57 % cyclic, CDCl₃ | same, cmpd 14 |
| fused-aryl donor | 2-Me-indoline / barbituric | ~45 % open, CDCl₃ | Hemmer, *JACS* 2018 (acceptor 2) |
| acceptor axis | indoline / CF₃-pyrazolone | >95 % open, CDCl₃ | Hemmer, *JACS* 2018 (acceptor 7) |

This is the same discipline `modal_dft_v2.py` already applies to colour (two measured
references through the identical pipeline, refuse to endorse ranking unless the
reference errors agree). Reuse that structure verbatim: **if the two anchors' errors
do not agree within ~1 kcal/mol, the run does not get to rank anything.**

---

## 7. Getting it into the staged loop

DFT free energies cannot run inside RL scoring — tier 2 is hours per molecule and the
loop scores tens of thousands. The way this is done is **offline compute → surrogate
in-loop → DFT at verification**:

**Offline (Modal, detached).** For 200–400 DASAs spanning donor architecture ×
acceptor × substituent, compute at tier 2:
`G(A)`, `G(C_enol)`, `G(C_zwit)`, `G(C_keto)`, and `ΔG‡` for the rate-determining
reverse step. Do it in **both water and chloroform** — chloroform is where the
calibration data lives, water is where the project lives, and the pair gives the
solvent-differential that is the actual design signal. Take ~30 of them to tier 3
for an error bar.

**Surrogate.** Train Δ-learning, not a cold model: target `ΔG_DFT − ΔG_xTB`, or
`ΔG_DFT − (linear model in delta_pka)`. Learning the *residual* on top of a physical
baseline needs far less data than learning ΔG outright, which matters at n≈300.
No published ML model predicts DASA thermodynamics — I searched; the field's ML
effort is all on λ_max. That is a contribution, not just a tool.

**In-loop, by stage.**

| stage | term | form |
|---|---|---|
| 1 | `delta_pka` | band, as now |
| 2 | surrogate **ΔΔG(A → closed ensemble)** in water | **band**, not maximised — too open-favoured and it will not close under light |
| 2 | surrogate **ΔG‡(closed → open)** in water | **band** ≈ 21–24 kcal/mol (t½ minutes to hours). This is the real anti-trap term. |
| 3 | — | unchanged |
| verify | tier-2 DFT on the shortlist, tier-3 + explicit water on finalists | with the two-anchor refusal rule of §6 |

Both are windows with failure at both ends, so both **band** under the standing
design rules — a reverse barrier that is maximised gives a switch that never comes
back, which is the trap wearing a different hat.

This is also the fix for the diagnosed root cause of the azole drift: the loop lost
its only high-resolution per-molecule gradient when Stage 2 was disabled, and
solubility became the sole gradient. A surrogate ΔG is a high-resolution gradient
that costs milliseconds.

---

## 8. The `chem-diagrams` lead

`Tonner-Zech-Group/chem-diagrams` (`pip install chemdiagrams`, MIT, Python ≥3.10,
matplotlib/numpy/scipy) is a **plotting library**. It computes nothing. `EnergyDiagram`
+ `draw_path(x_data, y_data, linetypes=…)`, `draw_difference_bar()`,
`add_numbers_auto()`, `set_xlabels()`, plus five diagram styles and connector styles.

It is nonetheless the right renderer for this work: the figure it draws *is* the
Angew 2018 Fig. 1c / Chem 2023 Fig. 3B format, and a per-candidate "switching
landscape card" (A / A′ / A″ / TS / C_enol / C_zwit / C_keto, two solvents overlaid
as two paths) is the natural artifact of the offline campaign — one glance says
whether a molecule is open-favoured, and whether it is barrier-locked. Feed it a JSON
of species energies per molecule; keep it strictly at the presentation layer, out of
`plugins/`.

---

## 9. Found in passing (not fixed — out of scope for a literature scan)

- `plugins/reinvent_plugins/components/comp_dasa_trap.py:176` — `dE_water` is used
  but never defined in `_score_smiles`. **NameError whenever `use_toluene=true`**;
  caught by the bare `except` upstream only by luck, so the toluene branch has never
  run.
- Same file: `eo_w = _energy_opt(mol, "water")` runs an extra geometry optimisation
  unconditionally and is then unused when `use_toluene=false` — roughly a quarter of
  Stage-2 xTB time, wasted, on the default path.
- `notebooks/dasa_complete.ipynb` still sets `params.dE_water_target_kcal` /
  `dE_water_width_kcal`; the current `Parameters` dataclass takes `dE_lo_kcal`,
  `dE_hi_kcal`, `dE_width_kcal`, `use_toluene`. That config cannot validate.

---

## 10. What I could not verify

- **Chem 2023 SI §4.1** — the actual functional benchmark for DASAs — is behind
  Cell's bot wall. That is the single most useful document for choosing tier 2 and
  it is worth pulling with Stanford access.
- Angew 2018 **Table S8.1** (the numeric landscape per solvent) — same, PMC serves it
  only as a download.
- A search summary asserted that "M06-2X/aug-cc-pVDZ/SMD reproduces the relative
  thermal stability of open and closed DASA isomers more accurately than other
  continuum models". **I could not find the primary source. Do not cite it.**
- **No measured open/closed equilibrium constant exists for any DASA in pure water** —
  dark switching goes to completion, so there is nothing to calibrate an aqueous ΔG
  against. Only t½ is measurable there. This is a hard limit on aqueous validation
  and an argument for making ΔG‡ the primary target.
- Whether **C_enol** is a genuine minimum for aryl (2nd-gen) donors, or only for the
  1st-generation compounds Chem 2023 studied.

---

## Sources

- Lerch, Szymański, Feringa et al., *Angew. Chem. Int. Ed.* **2018**, 57, 8063 — <https://pmc.ncbi.nlm.nih.gov/articles/PMC6055754/>
- Stricker, Peterson, … Read de Alaniz, *Chem* **2023**, 9, 1994–2005 — <https://www.cell.com/chem/fulltext/S2451-9294(23)00243-7>
- Hemmer, Page, … Read de Alaniz, *JACS* **2018**, 140, 10425 — <https://escholarship.org/uc/item/1vm8m3vt>
- Blandón-Cumbreras, … Pischel, *JACS* **2026**, 148, 130–134 — <https://pmc.ncbi.nlm.nih.gov/articles/PMC12814171/>
- Mallo, … Beves, *Chem. Sci.* **2018**, 9, 8242 — structure–function, X-ray of the zwitterionic closed form
- Sanchez, Raucci, Martínez, *JACS* **2021**, 143, 20015 — AIMS/BOMD switching dynamics — <https://www.osti.gov/servlets/purl/1832320>
- Berraud-Pache, … Sampedro, Izsák, *Chem. Sci.* **2021**, 12, 2916 — DLPNO-STEOM-CCSD colour protocol (not thermodynamics) — <https://pmc.ncbi.nlm.nih.gov/articles/PMC8179403/>
- Bursch, Mewes, Hansen, Grimme, *Angew. Chem. Int. Ed.* **2022**, 61, e202205735 — best-practice DFT protocols
- Grimme et al., CENSO/CREST — conformer ensembles and solution free energies
- `Tonner-Zech-Group/chem-diagrams` — <https://github.com/Tonner-Zech-Group/chem-diagrams>
