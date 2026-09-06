"""The first test of a STERIC prediction against a STERIC measurement.

Ground truth: Peterson, Stricker & Read de Alaniz, Chem. Commun. 2022, 58, 2303.
Model system D1 = indoline donor + Meldrum's acid acceptor. Table 1 (toluene + 5%
DCM) gives a matched pair differing by ONE atom:

    D1-H  unsubstituted   29% open  ->  dG = -0.531 kcal/mol
    D1-M  5-methyl        95% open  ->  dG = +1.745 kcal/mol
    MEASURED ddG for one methyl at C5 = +2.275 kcal/mol

Their DFT (M06-2X/6-31+G(d,p)/SMD toluene, Fig. 2b) additionally predicts, for
substituents at the other triene positions:

    C5   ddG "increasingly positive ... as steric bulk is introduced", Me < Et < iPr
    C3   negative -- stabilises the closed form
    C4   negative -- stabilises the closed form
    C1   "little effect"

Chain numbering, from the Org. Synth. name of the parent
("...penta-2,4-dien-1-ylidene..."): C1 is the ylidene carbon bonded to the acceptor
ring, C2 carries the hydroxyl, C5 carries the amine. In this repo's core keys that
is C1=Ce, C2=Cd, C3=Cc, C4=Cb, C5=Ca.

Four independent things are being tested, and they fail independently:
  1. MAGNITUDE  does ddE(D1-M) reproduce +2.275 kcal/mol?
  2. ORDERING   does C5 go increasingly positive Me < Et < iPr?
  3. SIGN       are C3 and C4 negative?
  4. NULL       is C1 near zero?

Keto-only, toluene: the closed form in an aprotic solvent is the neutral keto
(Angew. 2018, 57, 8063), and the Chem. Commun. paper's own Fig. 2a labels its
closed isomer "keto form (C)".
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import cached_optimise

H2KCAL, RT, SOLV = 627.5094740631, 0.5925, "toluene"
D, A = "C1Cc2ccccc2N1", "=C1C(=O)OC(C)(C)OC1=O"      # indoline / Meldrum's acid

#          label                 C5        C4      C3      C1
LADDER = {
    "D1-H  parent":        f"{D}C=CC=C(O)C{A}",
    "D1-M  C5-methyl":     f"{D}C(C)=CC=C(O)C{A}",
    "D1-Et C5-ethyl":      f"{D}C(CC)=CC=C(O)C{A}",
    "D1-iPr C5-isopropyl": f"{D}C(C(C)C)=CC=C(O)C{A}",
    "C4-methyl":           f"{D}C=C(C)C=C(O)C{A}",
    "C3-methyl":           f"{D}C=CC(C)=C(O)C{A}",
    "C1-methyl":           f"{D}C=CC=C(O)C(C){A}",
}
MEASURED_DDG = {"D1-M  C5-methyl": +2.275}
PAPER_DFT_SIGN = {"D1-M  C5-methyl": "+", "D1-Et C5-ethyl": "+", "D1-iPr C5-isopropyl": "+",
                  "C4-methyl": "-", "C3-methyl": "-", "C1-methyl": "0"}


def dE(smi):
    """Boltzmann-weighted dE(open -> closed keto manifold), kcal/mol."""
    op = cached_optimise(smi, SOLV, "planar120")
    if op["status"] != "ok":
        return None, f"open form: {op.get('error')}", 0
    iso = dc.closed_stereoisomers(smi, form=dc.closed_forms_for_solvent(SOLV))
    es, bad = [], 0
    for d in iso:
        r = cached_optimise(d["smiles"], SOLV, "lowest20")
        if r["status"] == "ok":
            es.append(r["energy"] * H2KCAL)
        else:
            bad += 1
    if not es:
        return None, "no closed isomer converged", len(iso)
    e = np.array(es); emin = e.min()
    return emin - RT * np.log(np.exp(-(e - emin) / RT).sum()) - op["energy"] * H2KCAL, bad, len(iso)


def main():
    with ThreadPoolExecutor(max_workers=6) as ex:
        out = dict(zip(LADDER, ex.map(lambda s: dE(s), LADDER.values())))
    ref = out["D1-H  parent"][0]
    print(f"GFN2-xTB / ALPB toluene, keto manifold, indoline/Meldrum's\n")
    print(f"{'compound':22s} {'dE_closed':>10s} {'ddE vs H':>9s} {'paper':>6s} "
          f"{'measured':>9s} {'#iso':>5s}")
    rows = {}
    for k, (v, bad, n) in out.items():
        if v is None:
            print(f"{k:22s}  FAILED: {bad}"); continue
        dd = v - ref
        rows[k] = dd
        meas = f"{MEASURED_DDG[k]:+.2f}" if k in MEASURED_DDG else ""
        print(f"{k:22s} {v:10.2f} {dd:9.2f} {PAPER_DFT_SIGN.get(k,''):>6s} "
              f"{meas:>9s} {n:5d}" + (f"  ({bad} failed)" if bad else ""))

    print("\nTEST 1  MAGNITUDE")
    if "D1-M  C5-methyl" in rows:
        c = rows["D1-M  C5-methyl"]
        print(f"   ddE(C5-Me) = {c:+.2f}  vs measured {MEASURED_DDG['D1-M  C5-methyl']:+.2f}"
              f"   error {c-MEASURED_DDG['D1-M  C5-methyl']:+.2f} kcal/mol")
    print("\nTEST 2  ORDERING at C5 (paper: increasingly positive Me < Et < iPr)")
    lad = [rows.get(k) for k in ("D1-M  C5-methyl","D1-Et C5-ethyl","D1-iPr C5-isopropyl")]
    if all(x is not None for x in lad):
        print(f"   Me {lad[0]:+.2f}  Et {lad[1]:+.2f}  iPr {lad[2]:+.2f}   "
              f"monotone increasing: {lad[0] < lad[1] < lad[2]}")
        print(f"   all positive: {all(x > 0 for x in lad)}")
    print("\nTEST 3  SIGN at C3 / C4 (paper: negative, stabilises the closed form)")
    for k in ("C3-methyl", "C4-methyl"):
        if k in rows:
            print(f"   {k:12s} {rows[k]:+8.2f}   correct sign: {rows[k] < 0}")
    print("\nTEST 4  NULL at C1 (paper: little effect)")
    if "C1-methyl" in rows:
        print(f"   C1-methyl    {rows['C1-methyl']:+8.2f}   "
              f"|ddE| < 1 kcal/mol: {abs(rows['C1-methyl']) < 1.0}")

    with open(os.path.join(HERE, "data", "triene_ladder.json"), "w") as f:
        json.dump({k: {"smiles": LADDER[k], "dE": out[k][0],
                       "ddE_vs_parent": rows.get(k)} for k in LADDER}, f, indent=2,
                  default=float)
    print(f"\nwrote {os.path.join(HERE,'data','triene_ladder.json')}")


if __name__ == "__main__":
    main()
