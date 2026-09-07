"""Rebuild notebooks/data/dasa_literature_seed.csv from the primary literature.

WHY THIS IS A SCRIPT AND NOT A HAND-EDITED CSV: the previous seed file was written
2026-07-13, encoded the pre-correction (LEGACY) connectivity in 10/10 rows, had
0/10 of its three measurement columns populated, and nothing in the repo read it.
Keeping the data in code means every structure is validated on every rebuild and
every number carries its citation to the page.

VALIDATION: every row must parse, match the corrected DASA open-form core, and NOT
match the legacy core. The script exits non-zero if any row fails.

STRUCTURE CONFIDENCE, per row:
  certain       - donor and acceptor are named unambiguously in the paper text
  reconstructed - inferred from prose or a figure; CHECK AGAINST THE PAPER before
                  fitting anything to it

MEASURED DATA NOT INCLUDED, because I could not pin the structure:
  * Peterson Chem. Sci. 2023 cmpd 1 - solvatochromic slope -7 nm, <1% open in
    MeOH, ~30% open in DCM/toluene. A 2nd-gen aryl amine; the paper cites ref 12
    for the slope. This is a valuable LOW-charge-separation point - worth adding
    once the structure is confirmed.
  * Stricker Chem 2023 DASA-3 - slope -60 nm, CF3-pyrazolone acceptor, donor not
    stated in the text I could read.
  * Hemmer JACS 2018 acceptor 8 (hydroxypyridone) - 680 nm, 75-85% open. The
    figure shows an N-butyl / methyl / nitrile pyridone; connectivity not certain.
  * Pischel JACS 2026 CB7 / CB8 host-guest complexes - t(1/2) 12.4 h, 45.1 h,
    112.2 h, 586.3 h. These are supramolecular complexes, not single molecules,
    so they cannot be represented as one SMILES. They are the best available
    STERIC calibration points; keep them in mind for that.
"""
from __future__ import annotations

import csv
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dasa_chem as dc

# --- fragments ------------------------------------------------------------
D_2ME_INDOLINE = "CC1Cc2ccccc2N1"
D_INDOLINE     = "C1Cc2ccccc2N1"
D_ISOINDOLINE  = "C1c2ccccc2CN1"
D_ME2N         = "CN(C)"
D_NME_ANILINE  = "CN(c1ccccc1)"

A_MELDRUM    = "=C1C(=O)OC(C)(C)OC1=O"
A_DIMEBARB   = "=C1C(=O)N(C)C(=O)N(C)C1=O"
A_DIMETHIOBARB = "=C1C(=O)N(C)C(=S)N(C)C1=O"
A_PYRAZDIONE = "=C1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"
A_MEISOX     = "=C1C(=O)ON=C1C"
A_CF3ISOX    = "=C1C(=O)ON=C1C(F)(F)F"
A_INDANDIONE = "=C1C(=O)c2ccccc2C1=O"
A_CF3PYRAZ   = "=C1C(=O)N(c2ccccc2)N=C1C(F)(F)F"
A_MEPYRAZ    = "=C1C(=O)N(c2ccccc2)N=C1C"


def d(donor: str, acceptor: str) -> str:
    """Literature (2Z,4E) open form, corrected core."""
    return f"{donor}/C=C/C=C(\\O)C{acceptor}"


# Tethered aniline donors (Peterson). RECONSTRUCTED from "the amine donor tethered
# to the triene via a three-carbon linkage, resulting in a 5-membered ring", which
# puts N-CH2CH2CH2-C5 in the ring and leaves C5=C4 exocyclic.
def tethered(aryl: str, acceptor: str) -> str:
    return f"C1(CCCN1{aryl})=CC=C(O)C{acceptor}"


AD = "C45CC6CC(CC(C6)C4)C5"          # 1-adamantyl


def adamantyl(acceptor: str) -> str:
    return f"N(/C=C/C=C(\\O)C{acceptor}){AD}"


# --- the data -------------------------------------------------------------
# columns: id, smiles, donor_class, acceptor_class, lambda_nm, lambda_solvent,
#          pct_open, equil_solvent, slope_nm, switches_in_water, t_half_s,
#          t_half_solvent, confidence, source, source_location
ROWS = [
    # ---- Hemmer, JACS 2018, 140, 10425. Donor = 2-methylindoline throughout.
    # Equilibria: Fig. 3b bar chart + text. "acceptors 1 and 2 (~45% triene at
    # equilibrium)"; "acceptors 3, 4, and 8 ... ~75-85%"; "5 and 7 were >95%";
    # "from 80 to 95%" for 4 -> 5. Acceptor 6 is the only one BELOW 1 and 2 and
    # is read off Fig. 3b, so its value is the least certain number in this file.
    # lambda: "peak absorption ranged from 585 to 680 nm in going from acceptor
    # 1 to 8"; acceptor 7 "lambda_max ~ 650 nm"; acceptor 8 "680 nm".
    ("H1", d(D_2ME_INDOLINE, A_MELDRUM),    "fused_aryl", "meldrum",
     585, "CDCl3", 45, "CDCl3", None, False, None, None, "certain",
     "Hemmer JACS 2018, 140, 10425", "Fig. 1b (acceptor 1); Fig. 3b + text"),
    ("H2", d(D_2ME_INDOLINE, A_DIMEBARB),   "fused_aryl", "dimethylbarbituric",
     615, "CH2Cl2", 45, "CDCl3", None, False, None, None, "certain",
     "Hemmer JACS 2018, 140, 10425",
     "Fig. 3b (45% open). lambda 615 nm is from Berraud-Pache Chem Sci 2021, 12, 2916, "
     "Table 1 D1-A2 -- the SAME COMPOUND measured in two papers."),
    ("H3", d(D_2ME_INDOLINE, A_PYRAZDIONE), "fused_aryl", "pyrazolidinedione",
     None, None, 80, "CDCl3", None, False, None, None, "reconstructed",
     "Hemmer JACS 2018, 140, 10425", "Fig. 1b acceptor 3 drawn as 1,2-diphenylpyrazolidinedione; 75-85% band"),
    ("H4", d(D_2ME_INDOLINE, A_MEISOX),     "fused_aryl", "methylisoxazolone",
     None, None, 80, "CDCl3", None, False, None, None, "certain",
     "Hemmer JACS 2018, 140, 10425", "text: 'from 80 to 95%' comparing 4 to 5"),
    ("H5", d(D_2ME_INDOLINE, A_CF3ISOX),    "fused_aryl", "CF3_isoxazolone",
     None, None, 95, "CDCl3", None, False, None, None, "certain",
     "Hemmer JACS 2018, 140, 10425", "text: 'from 80 to 95%'"),
    ("H6", d(D_2ME_INDOLINE, A_INDANDIONE), "fused_aryl", "indandione",
     None, None, 5, "CDCl3", None, False, None, None, "certain",
     "Hemmer JACS 2018, 140, 10425", "Fig. 3b - ONLY acceptor below 1/2; value read off the bar chart"),
    ("H7", d(D_2ME_INDOLINE, A_CF3PYRAZ),   "fused_aryl", "CF3_pyrazolone",
     650, "CDCl3", 97, "CDCl3", None, False, None, None, "certain",
     "Hemmer JACS 2018, 140, 10425", "text: 'lambda_max ~ 650 nm, >95% triene at equilibrium'"),

    # ---- Berraud-Pache, Santamaria-Aranda, de Souza, Bistoni, Neese, Sampedro,
    # Izsak, Chem. Sci. 2021, 12, 2916. NOT a Read de Alaniz paper - the repo
    # previously credited it to that group.
    ("C1", d(D_2ME_INDOLINE, A_DIMETHIOBARB), "fused_aryl", "thiobarbituric",
     647, "CH2Cl2", None, None, None, False, None, None, "reconstructed",
     "Berraud-Pache Chem Sci 2021, 12, 2916",
     "D1-A2d, 647 nm in DCM. A2d is the commercially-available thio-barbituric; "
     "which C=O became C=S is inferred"),

    # ---- Mallo, ... Beves, Chem. Sci. 2018, 9, 8242.
    ("M1",  d(D_ME2N, A_DIMEBARB),        "dialkyl_amine", "dimethylbarbituric",
     567, "CHCl3", 86, "CDCl3", None, False, None, None, "certain",
     "Mallo Chem Sci 2018, 9, 8242", "cmpd 1; 86% open. Also the modal_dft_v2 colour anchor."),
    ("M14", d(D_NME_ANILINE, A_DIMEBARB), "aniline", "dimethylbarbituric",
     588, "CHCl3", 43, "CDCl3", None, False, None, None, "reconstructed",
     "Mallo Chem Sci 2018, 9, 8242",
     "cmpd 14; 57% CYCLIC in the dark -> 43% open. X-ray of its closed form is the "
     "NEUTRAL keto. Exact aniline N-substituent inferred."),

    # ---- Peterson, Neris, Read de Alaniz, Chem. Sci. 2023, 14, 13025.
    # THE aqueous-compatibility paper. Slopes from Fig. S7 / Table S1, MeOH
    # equilibria from Table S2 / Fig. S10.
    ("P3", tethered("c1ccccc1", A_MELDRUM),        "tethered_aniline", "meldrum",
     560, "toluene", 99, "MeOH", -56, False, None, None, "reconstructed",
     "Peterson Chem Sci 2023, 14, 13025",
     "cmpd 3; slope -56 nm; 1H NMR shows only open form in toluene/DMSO/MeOH after 24 h; "
     "lambda from 'absorbance of A (560 nm)'. Tether geometry inferred from prose."),
    ("P4", tethered("c1ccccc1", A_MEPYRAZ),        "tethered_aniline", "methylpyrazolone",
     None, None, 70, "MeOH", -32, False, None, None, "reconstructed",
     "Peterson Chem Sci 2023, 14, 13025", "cmpd 4; slope -32 nm; 70% open in MeOH"),
    ("P5", tethered("c1ccc(I)cc1", A_MEPYRAZ),     "tethered_aniline", "methylpyrazolone",
     575, "THF:H2O 60:40", 24, "MeOH", -20, True, 240, "THF:H2O 60:40", "reconstructed",
     "Peterson Chem Sci 2023, 14, 13025",
     "cmpd 5; slope -20 nm; 24% open in MeOH; switches in 60:40 THF:water, 95% recovery "
     "t(1/2) 240 s, >=10 cycles. THE most water-compatible free DASA reported."),

    # ---- Peterson, Stricker, Read de Alaniz, Chem. Commun. 2022, 58, 2303.
    # "Improving the kinetics and dark equilibrium of DASA by triene backbone
    # design". Model system D1 = INDOLINE donor + Meldrum's acid acceptor, named
    # explicitly in the text. Table 1, toluene + 5% CH2Cl2 (the DCM is a solubility
    # aid; recorded here as toluene). This is the ONLY matched pair in the whole
    # literature that isolates a single triene substituent at a fixed donor and
    # acceptor, so it is the only clean STERIC anchor we have.
    ("D1H", d(D_INDOLINE, A_MELDRUM), "fused_aryl", "meldrum",
     590, "toluene", 29, "toluene", -7, False, None, None, "certain",
     "Peterson Chem Commun 2022, 58, 2303",
     "D1-H, Table 1: lambda 590 nm, 29% open, slope -7 nm, no recovery over 2000 s"),
    ("D1M", f"{D_INDOLINE}C(C)=CC=C(\\O)C{A_MELDRUM}", "fused_aryl", "meldrum",
     609, "toluene", 95, "toluene", -24, False, 136, "toluene", "certain",
     "Peterson Chem Commun 2022, 58, 2303",
     "D1-M, Table 1: 5-METHYL on the triene. lambda 609 nm, 95% open, slope -24 nm, "
     "t(1/2) 136 s, 99% recovery. Pairs with D1H -> measured ddG = +2.275 kcal/mol. "
     "CAVEAT: reverts to amine + furan in MeOH within 10 min (5% loss/7 h in toluene)"),

    # ---- Blandon-Cumbreras, ... Pischel, JACS 2026, 148, 130.
    # t(1/2) is DARK SWITCHING of the free dye (10 vol% THF/water), i.e. the
    # kinetic trap. Their CB7/CB8 complexes are in the module docstring.
    ("B1", adamantyl(A_DIMEBARB), "monoalkyl_amine", "dimethylbarbituric",
     None, None, None, None, -79, False, 2280, "THF:H2O 10:90", "reconstructed",
     "Blandon-Cumbreras JACS 2026, 148, 130",
     "DASA 1; slope -79 nm vs E_T(30); t(1/2) 38 min. 'Acceptor derived from "
     "barbituric acid' - N-methylation inferred."),
    ("B2", adamantyl(A_MELDRUM),  "monoalkyl_amine", "meldrum",
     None, None, None, None, -65, False, 1740, "THF:H2O 10:90", "certain",
     "Blandon-Cumbreras JACS 2026, 148, 130", "DASA 2; slope -65 nm; t(1/2) 29 min"),

    # ---- Nat. Commun. 2024 hydroxy parents. Recorded from project memory of that
    # paper, NOT re-read for this rebuild -> re-verify before fitting.
    # The paper's headline AMINO DASAs (NH replacing OH) are a DIFFERENT
    # chromophore and must not be mixed in.
    ("N9",  d(D_ISOINDOLINE, A_DIMEBARB), "cyclic_benzylic", "dimethylbarbituric",
     573, "CH2Cl2", None, None, -29, False, None, None, "reconstructed",
     "Nat Commun 2024, hydroxy parents",
     "cmpd 9 (1st-gen HYDROXY): lambda 573 nm; solvatochromic slope -29 nm/ETN, "
     "verified in the PMC full text 2026-09-06. Slope measured over 10 solvents "
     "(PhMe, Et2O, THF, EtOAc, CHCl3, CH2Cl2, acetone, DMSO, MeCN, MeOH) vs "
     "normalised Dimroth-Reichardt ETN. STRUCTURE still RE-VERIFY."),
    ("N10", d(D_INDOLINE, A_DIMEBARB),    "fused_aryl", "dimethylbarbituric",
     615, "CH2Cl2", None, None, -3, False, None, None, "reconstructed",
     "Nat Commun 2024, hydroxy parents",
     "cmpd 10 (2nd-gen HYDROXY): lambda 615 nm; slope -3 nm/ETN, verified in the PMC "
     "full text 2026-09-06 ('hydroxy counterpart 10 is more neutral'). Consistent "
     "with Peterson's indoline D1-H at -7 nm. Cross-checks H2 (615 nm)."),
    ("N11", d(D_INDOLINE, A_CF3PYRAZ),    "fused_aryl", "CF3_pyrazolone",
     646, "CH2Cl2", None, None, -54, False, None, None, "reconstructed",
     "Nat Commun 2024, hydroxy parents",
     "cmpd 11 (3rd-gen HYDROXY): lambda 646 nm; slope -54 nm/ETN, verified in the "
     "PMC full text 2026-09-06. Consistent with Peterson's D2 reference at -46 nm."),
]

COLUMNS = ["compound_id", "smiles_open", "smiles_closed", "donor_class",
           "acceptor_class", "lambda_max_open_nm", "lambda_solvent",
           "pct_open_equilibrium", "equilibrium_solvent", "solvatochromic_slope_nm",
           "switches_in_water", "half_life_s", "half_life_solvent",
           "structure_confidence", "source", "source_location"]


def main() -> int:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "dasa_literature_seed.csv")
    bad, records = [], []
    for r in ROWS:
        (cid, smi, dcls, acls, lam, lsol, pct, esol, slope, water,
         thalf, tsol, conf, src, loc) = r
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad.append((cid, "unparseable")); continue
        if not dc.is_dasa(m):
            bad.append((cid, "does not match the corrected DASA core")); continue
        if dc.is_legacy_core(m):
            bad.append((cid, "LEGACY core")); continue
        records.append({
            "compound_id": cid, "smiles_open": Chem.MolToSmiles(m),
            "smiles_closed": "",          # generate with dasa_chem.open_to_closed*
            "donor_class": dcls, "acceptor_class": acls,
            "lambda_max_open_nm": lam if lam is not None else "",
            "lambda_solvent": lsol or "",
            "pct_open_equilibrium": pct if pct is not None else "",
            "equilibrium_solvent": esol or "",
            "solvatochromic_slope_nm": slope if slope is not None else "",
            "switches_in_water": water,
            "half_life_s": thalf if thalf is not None else "",
            "half_life_solvent": tsol or "",
            "structure_confidence": conf, "source": src, "source_location": loc,
            "_formula": rdMolDescriptors.CalcMolFormula(m),
        })

    if bad:
        for cid, why in bad:
            print(f"  FAIL {cid}: {why}", file=sys.stderr)
        print(f"\n{len(bad)} row(s) failed validation; CSV NOT written.", file=sys.stderr)
        return 1

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print(f"{len(records)} rows, all corrected-core, 0 legacy -> {out}\n")
    print(f"{'id':5s} {'formula':16s} {'lam':>5s} {'%open':>6s} {'slope':>6s} "
          f"{'t1/2 s':>7s} {'conf':14s} donor/acceptor")
    for r in records:
        print(f"{r['compound_id']:5s} {r['_formula']:16s} "
              f"{str(r['lambda_max_open_nm']):>5s} {str(r['pct_open_equilibrium']):>6s} "
              f"{str(r['solvatochromic_slope_nm']):>6s} {str(r['half_life_s']):>7s} "
              f"{r['structure_confidence']:14s} {r['donor_class']}/{r['acceptor_class']}")
    n = lambda k: sum(1 for r in records if r[k] != "")
    print(f"\npopulated: lambda {n('lambda_max_open_nm')}, "
          f"%open {n('pct_open_equilibrium')}, "
          f"slope {n('solvatochromic_slope_nm')}, "
          f"t1/2 {n('half_life_s')}   (was 0/0/0)")
    print(f"certain structures: {sum(1 for r in records if r['structure_confidence']=='certain')}"
          f"/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
