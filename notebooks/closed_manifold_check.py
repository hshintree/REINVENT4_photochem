"""Re-run the C5-methyl ground truth with the closed manifold properly enumerated.

The previous ladder (notebooks/data/steric_groundtruth.json) embedded ONE arbitrary
diastereomer of each closed form and took min(zwitterion, keto). The configuration
it left unset is worth 5-25 kcal/mol, so the three donor families disagreed about
the sign of a C5 methyl: Me2N -1.05, indoline +0.60, anilino +6.06 kcal/mol.

Here every diastereomer of both tautomers is enumerated and optimised, and the
closed state is the Boltzmann sum over the whole manifold -- which is what a dark
equilibrium is actually measured against.

GROUND TRUTH: Peterson/Stricker/Read de Alaniz, a C5 methyl takes the same
donor/acceptor pair from ~30% to >96% open in DCM and toluene, i.e.
    ddG = -RT ln(4/96) + RT ln(70/30) = +2.39 kcal/mol
on an ANILINE donor. Sign and rough magnitude are the test; GFN2-xTB electronic
energies with no thermal correction cannot be expected to hit it exactly.
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from calibrate_proxies import xtb_optimise

H2KCAL = 627.5094740631
SOLV = "toluene"          # the solvent the ground truth was measured in
A = "=C1C(=O)N(C)C(=O)N(C)C1=O"

# Analogues of one parent, per donor family. C5 = Ca (bonded to N), C4 = Cb.
# Template {donor}C=CC=C(O)C{acceptor} lays out N-Ca=Cb-Cc=Cd(OH)-Ce=Cf, so
#   C5-Me -> {donor}C(C)=CC=C(O)C{acc}      C4-Me -> {donor}C=C(C)C=C(O)C{acc}
# accNEt swaps one acceptor N-methyl for N-ethyl: a REMOTE control six bonds from
# the ring. It "passed" in the old ladder (-0.18 to -0.32 kcal/mol) but only
# because it does not touch the closed-form stereocentres, so its arbitrary
# diastereomer cancelled between parent and analogue. Re-run here to check it is
# still well behaved once the manifold is handled properly.
A_ET = "=C1C(=O)N(CC)C(=O)N(C)C1=O"
DONORS = {"Me2N": "CN(C)", "indoline": "CC1Cc2ccccc2N1", "anilino": "CN(c1ccccc1)"}
FAMILIES = {
    fam: {
        "C5-H":   f"{d}C=CC=C(O)C{A}",
        "C5-Me":  f"{d}C(C)=CC=C(O)C{A}",
        "C4-Me":  f"{d}C=C(C)C=C(O)C{A}",
        "accNEt": f"{d}C=CC=C(O)C{A_ET}",
    }
    for fam, d in DONORS.items()
}
# vbur_site from notebooks/data/steric_groundtruth.json, for the comparison that
# matters: does a descriptor computed on the OPEN form separate C4 from C5?
VBUR_SITE = {
    ("Me2N", "C5-H"): 0.0664, ("Me2N", "C5-Me"): 0.1438,
    ("Me2N", "C4-Me"): 0.1525, ("Me2N", "accNEt"): 0.0662,
    ("indoline", "C5-H"): 0.0777, ("indoline", "C5-Me"): 0.1658,
    ("indoline", "C4-Me"): 0.1573, ("indoline", "accNEt"): 0.0775,
    ("anilino", "C5-H"): 0.0756, ("anilino", "C5-Me"): 0.1475,
    ("anilino", "C4-Me"): 0.1623, ("anilino", "accNEt"): 0.0769,
}
OLD_LADDER = {
    ("Me2N", "C5-Me"): -1.05, ("Me2N", "C4-Me"): -15.43, ("Me2N", "accNEt"): -0.32,
    ("indoline", "C5-Me"): 0.60, ("indoline", "C4-Me"): -11.33, ("indoline", "accNEt"): -0.22,
    ("anilino", "C5-Me"): 6.06, ("anilino", "C4-Me"): -9.92, ("anilino", "accNEt"): -0.18,
}
MEASURED_ANILINO_DDG = +2.39     # kcal/mol, from 30% -> >96% open


def embed(smi, n_confs=20, seed=42):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    if AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=p) == 0:
        return None
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=1000)
    except Exception:
        res = [(0, 0.0)] * m.GetNumConformers()
    best = int(np.argmin([e for _c, e in res]))
    single = Chem.Mol(m); single.RemoveAllConformers()
    c = Chem.Conformer(m.GetConformer(best)); c.SetId(0)
    single.AddConformer(c, assignId=True)
    return single


def energy(smi):
    m3d = embed(smi)
    if m3d is None:
        return None
    try:
        _xyz, e, _d, _n, _ok = xtb_optimise(m3d, SOLV)
        return e
    except Exception:
        return None


def analyse(label, smi):
    e_open = energy(smi)
    if e_open is None:
        return {"label": label, "error": "open form failed"}
    man = dc.closed_manifold(smi, energy, max_isomers=16)
    if man is None:
        return {"label": label, "error": "manifold failed"}
    e_open_kcal = e_open * H2KCAL
    per_form = {}
    for f in ("zwitterion", "keto"):
        sub = [s for s in man["isomers"] if s["form"] == f]
        if sub:
            per_form[f] = min(s["energy_hartree"] for s in sub) * H2KCAL - e_open_kcal
    return {"label": label, "smiles": smi,
            "n_isomers": man["n_isomers"],
            "dE_ensemble": man["ensemble_kcal"] - e_open_kcal,
            "dE_lowest": man["lowest_kcal"] - e_open_kcal,
            "spread_kcal": man["spread_kcal"],
            "per_form": per_form,
            "lowest_form": man["lowest"]["form"]}


def main():
    jobs = [(f"{fam} {tag}", smi)
            for fam, tags in FAMILIES.items() for tag, smi in tags.items()]
    with ThreadPoolExecutor(max_workers=6) as ex:
        recs = list(ex.map(lambda t: analyse(*t), jobs))
    by = {r["label"]: r for r in recs}

    print(f"GFN2-xTB / ALPB {SOLV}, every closed diastereomer of both tautomers\n")
    print(f"{'molecule':18s} {'#iso':>5s} {'dE_ens':>8s} {'spread':>8s} "
          f"{'zwit':>8s} {'keto':>8s}  lowest")
    for r in recs:
        if "error" in r:
            print(f"{r['label']:18s}  {r['error']}"); continue
        pf = r["per_form"]
        print(f"{r['label']:18s} {r['n_isomers']:5d} {r['dE_ensemble']:8.2f} "
              f"{r['spread_kcal']:8.2f} {pf.get('zwitterion', float('nan')):8.2f} "
              f"{pf.get('keto', float('nan')):8.2f}  {r['lowest_form']}")

    print("\nddE vs the C5-H parent  (positive = pushes the equilibrium OPEN)")
    print(f"{'donor':10s} {'analogue':9s} {'ddE_manifold':>13s} {'old ladder':>11s} "
          f"{'change':>8s} {'d vbur_site':>12s}")
    for fam, tags in FAMILIES.items():
        h = by.get(f"{fam} C5-H")
        if not h or "error" in h:
            continue
        for tag in ("C5-Me", "C4-Me", "accNEt"):
            m = by.get(f"{fam} {tag}")
            if not m or "error" in m:
                continue
            dd = m["dE_ensemble"] - h["dE_ensemble"]
            old = OLD_LADDER[(fam, tag)]
            dv = VBUR_SITE[(fam, tag)] - VBUR_SITE[(fam, "C5-H")]
            print(f"{fam:10s} {tag:9s} {dd:13.2f} {old:11.2f} {dd-old:8.2f} {dv:+12.4f}")

    print(f"\nmeasured anchor (aniline donor, C5-Me): ddG = {MEASURED_ANILINO_DDG:+.2f} "
          f"kcal/mol; there is NO measurement for any C4 analogue.")
    print("\nTHE CLAIM UNDER TEST: a C4 and a C5 methyl raise vbur_site by nearly the")
    print("same amount, so a descriptor computed on the open form cannot separate them.")
    print("Whether that MATTERS depends on the two ddE values actually being opposite.")

    out = os.path.join(HERE, "data", "closed_manifold_check.json")
    with open(out, "w") as f:
        json.dump(recs, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
