# What the literature says to try, and what to expect

Companion to [DASA.md](DASA.md), [papers.txt](papers.txt) and
[DASA_FREE_ENERGY.md](DASA_FREE_ENERGY.md). Written 2026-09-06.

**Target:** a free (unencapsulated) DASA that is open-favoured, switchable and
reversible in water. No such compound exists. The closest are Peterson's tethered
4-iodoaniline (switches in 60:40 THF:water) and Pischel's cucurbituril complexes
(host-guest, not a free molecule).

**Stated design criterion**, from the group that has done most of the work
(Peterson, Chem. Sci. 2023): derivatives must be *"stable, have a high equilibrium of
the open form, and have a low charge separation indicated by a less negative
solvatochromic slope."* Three requirements, and they are not independent.

---

## 1. The central conflict, measured

Charge separation raises the open-form dark equilibrium **and** worsens
polar-solvent switching. Same coordinate, opposite consequences.

| compound | slope (nm/E_T^N) | % open in MeOH | source |
|---|---|---|---|
| Peterson 3 (tethered aniline/Meldrum) | −56 | ~99 | Chem. Sci. 2023 |
| Peterson 4 (tethered aniline/Me-pyrazolone) | −32 | 70 | Chem. Sci. 2023 |
| Peterson 5 (tethered 4-I-aniline/Me-pyrazolone) | −20 | 24 | Chem. Sci. 2023 |

Compound 5 is the *worst* on equilibrium and the *only one that switches in
THF:water*. There is no setting of the electronic dial that gives both, which is why
every literature success uses a **second, structural** handle.

Generational ordering of charge separation (all measured):
2nd-gen indoline **−3 to −7** · tethered aniline **−20 to −56** · 1st-gen **−29** ·
3rd-gen CF3-pyrazolone **−54** · 1st-gen adamantyl **−65 to −79**.
(The last is 2.2–2.7× other 1st-gen values on a nominally identical scale — treat
Pischel's numbers as possibly not comparable until the solvent set is checked.)

---

## 2. What each paper actually establishes

**Helmy, *J. Org. Chem.* 2014, 79, 11316** — the original. 1st generation:
dialkylamine donors, Meldrum's/barbituric acceptors. Switches only in aromatic
solvents.

**Hemmer, *JACS* 2018, 140, 10425** — the acceptor axis. Eight carbon acids on a
fixed 2-methylindoline donor, dark equilibria 5–97% open in CDCl3. Electron-poor
acceptors (CF3-isoxazolone, CF3-pyrazolone) give >95% open. **This is the largest
single-variable dataset in the field** and the backbone of our seed.

**Mallo, *Chem. Sci.* 2018, 9, 8242** — structure–function; X-ray of the zwitterionic
closed form; first report that *minor steric modifications to the dialkylamine donor*
matter.

**Clerc, *Angew. Chem. Int. Ed.* 2021, 60, 10219** — HFIP promotes the furan
ring-opening (k 3 → 56 M⁻¹h⁻¹ at 1 vol%), unlocking **deactivated amine donors that
were previously unreactive**. Critically: *"the use of sterically hindered,
electron-poor amines enabled the dark equilibrium to be decoupled from closed-isomer
half-lives for the first time."* **This is the only demonstrated route to
orthogonality between the two properties we need.** Caveat: HFIP *inhibits* 1st-gen
alkylamine synthesis.

**Peterson, *Chem. Commun.* 2022, 58, 2303** — the triene axis. A methyl at C5 takes
the same donor/acceptor pair from **29% → 95% open** and from no recovery in 2000 s
to **t½ = 136 s**. DFT scan over positions 1/3/4/5: C5 destabilises the closed form
and grows with bulk (Me < Et < iPr); **C3 and C4 stabilise it** (wrong direction);
C1 does little. **But the free C5-methyl fully decomposes to amine + furan in
methanol within 10 min** (5% loss over 7 h in toluene).

**Peterson, *Chem. Sci.* 2023, 14, 13025** — the tether. Retains C5 substitution
while removing the decomposition. Compound 5 switches in 60:40 THF:water, 95%
recovery, t½ 240 s, ≥10 cycles. Also: **4-cyano and 4-nitro anilines do not form the
adduct** under standard conditions ("poor nucleophilicity", 3 days, no product) —
though Clerc's HFIP may reach them.

**Lerch/Feringa/Buma, *Angew.* 2018, 57, 8063** — in **protic** solvents the closed
product is the zwitterion, below the open form and behind a high back-barrier; in
**aprotic** solvents cyclization stops at the neutral keto form. Determines which
species even exists in which solvent.

**Blandón-Cumbreras/Pischel, *JACS* 2026, 148, 130** — steric shielding by
cucurbit[n]uril slows dark switching 20–1200× (t½ 29 min → 586 h).
*"Significant steric effects can overrule the electronic effects."*

**Nat. Commun. 2024** (amino DASAs) — hydroxy parents give the generational slope
ladder above. Amino DASAs are a **different chromophore**; do not pool.

---

## 3. Design axes: what to try, expected effect, evidence strength

| axis | expected effect | evidence | risk |
|---|---|---|---|
| **Electron-poor aryl donor** | ↓ charge separation → switches in polar solvents; ↓ dark equilibrium | strong (Peterson 5) | synthesis ceiling: 4-CN/4-NO2 don't form without HFIP |
| **Tether donor→C5** | ↑ equilibrium, keeps C5 substitution, removes decomposition | strong (Peterson 2023) | bespoke synthesis, not the standard 2-step route |
| **Bulky + electron-poor amine, HFIP** | **decouples equilibrium from half-life** | strong (Clerc 2021) | too much bulk ⇒ less closed form under irradiation, i.e. the switch stops switching |
| **Free C5 alkyl** | ↑ equilibrium, ↑ recovery rate | strong (Chem. Commun. 2022) | **decomposes in MeOH in 10 min — disqualifying for water** |
| **C3 / C4 substitution** | stabilises the CLOSED form | strong, and wrong direction | avoid; gate against |
| **Acceptor 3-position bulk (pyrazolone / isoxazolone)** | unknown | **none — untried** | see §4 |
| **Host encapsulation** | 20–1200× slower dark switching | strong (Pischel 2026) | not a free molecule |

---

## 4. Acceptor bulk — the untried axis

A carbon acid has its acidic carbon flanked by two carbonyls by definition, so on
**Meldrum's and barbituric there is nowhere to place bulk adjacent to the attachment
carbon** without destroying the acceptor. On **pyrazolones and isoxazolones** one
flanking position is a C=N carbon that already carries a substituent — methyl in
methylpyrazolone, CF3 in CF3-pyrazolone. That position is directly adjacent to the
bond that forms on cyclization.

So the accessible experiment is: hold the acceptor ring constant and walk that
3-position through **Me → iPr → tBu → cyclohexyl → adamantyl**. Those are large but
electronically much closer to methyl than CF3 is, which is the only way to vary
sterics without dragging electronics along.

**Two competing predictions, neither settled:**
* *destabilises the closed form more* — C1 goes sp3 and bonds to the acceptor's now-sp3
  acidic carbon, adjacent to C5 bearing the amine. A crowded 4,5-disubstituted
  cyclopentenone. This is the same logic that makes C5 substitution work.
* *destabilises the open form more* — the acceptor must stay coplanar with the triene
  for the push–pull chromophore; bulk twists it.

The existing data cannot separate these because the only pair that varies that
position (methyl vs CF3 pyrazolone) changes sterics **and** electronics together.

**Why this is worth trying rather than computing:** GFN2-xTB is disqualified — it got
the one measured steric case backwards (C5-methyl: predicted −1.39, measured +2.275).
DFT can do it (M06-2X/6-31+G(d,p)/SMD, the protocol Peterson used) but costs 5–12 h
per species on this hardware, ~60–120 h for a ladder. This is a case where synthesis
may be cheaper than computation.

---

## 5. What we can and cannot compute

| quantity | status |
|---|---|
| dark equilibrium ranking, one solvent | ΔE_xTB keto-only, **ρ = +0.748** (n=9, CDCl3) — our own validation, no literature precedent for the method |
| absolute ΔG | no — error exceeds the 4.4 kcal/mol range of the property |
| solvatochromic slope | **no** — no computed descriptor predicts it (all \|ρ\| ≤ 0.5, n=9) |
| steric effects | **no** — xTB gets the sign wrong; needs DFT |
| λmax | acceptor lookup table only |

**Consequence:** charge separation must be chosen **structurally** (donor class) using
the measured generational ladder in §1, not optimised by any computed score.

---

## 6. Recommended next experiments

1. **Acceptor 3-position ladder** on a pyrazolone, made and measured. The one design
   axis with no literature and a plausible mechanism. Cheaper to synthesise than to
   compute.
2. **Bulky electron-poor aryl amine + HFIP**, following Clerc — the only demonstrated
   route to decoupling equilibrium from half-life.
3. **Tethered scaffold as the base architecture**, following Peterson 2023, since it
   is the only free molecule shown to switch in a THF:water mixture.
4. Retrieve **Chem. Commun. 2022 ESI Fig. S15** (12 computed ΔΔG values for Me/Et/iPr
   at positions 1/3/4/5) — the only position-resolved ground truth, still unread.
