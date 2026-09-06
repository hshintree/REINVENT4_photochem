"""Is the DeltaE ground truth in validate_sterics_groundtruth.py trustworthy?

Two entries in that run came back with an OPEN-form conformer spread of 11.65 and
12.03 kcal/mol over just k=3 conformers, while most were under 0.3. A spread that
large in a set of conformers MMFF ranked as near-degenerate has one obvious cause
in a DASA: the enol O-H...O=C intramolecular hydrogen bond, which is worth roughly
that much and which the ETKDG+MMFF ranking does not reliably put first.

Why this matters more than it looks: I take the MINIMUM over conformers, so a large
spread is survivable -- the minimum is still the minimum. The DANGEROUS case is the
opposite, a molecule whose k conformers were ALL non-H-bonded, which comes back with
a SMALL spread and an open-form energy ~10 kcal/mol too high, and therefore a
DeltaE(open->closed) ~10 kcal/mol too negative. A small spread is not evidence of
convergence. That failure mode would manufacture exactly the result the run
reported: C4-Me stabilising the closed form by 10-15 kcal/mol, consistently, with
spreads of 0.00-2.31.

So: re-sample the three variants the argument rests on with much heavier sampling,
and report for every surviving conformer both its energy AND whether it is
H-bonded. If the C4-Me effect is real it survives; if it was a sampling artifact it
collapses.

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        ~/miniconda3/envs/reinvent4/bin/python notebooks/audit_open_conformers.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dasa_chem as dc
from calibrate_proxies import xtb_optimise
from dasa_descriptors import planar_ensemble
from validate_sterics_groundtruth import (DONORS, A_BARB, attach, _confs_of,
                                          H2KCAL, SOLVENT)

K = 8            # was 3
N_OPEN = 400     # was 120
N_CLOSED = 400   # was 200


def hbond(mol3d, xyz, idx) -> tuple[float, float]:
    """(shortest enol H...O=C distance, shortest enol O...O=C distance), Angstrom.

    The enol oxygen is idx["O"]; acceptor carbonyls are any other O double-bonded
    to carbon. < ~2.2 A on H...O is a formed intramolecular hydrogen bond.
    """
    o_enol = idx["O"]
    h = [n.GetIdx() for n in mol3d.GetAtomWithIdx(o_enol).GetNeighbors()
         if n.GetAtomicNum() == 1]
    carbonyls = [a.GetIdx() for a in mol3d.GetAtoms()
                 if a.GetAtomicNum() == 8 and a.GetIdx() != o_enol
                 and any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())]
    if not carbonyls:
        return float("nan"), float("nan")
    dOO = min(float(np.linalg.norm(xyz[o_enol] - xyz[c])) for c in carbonyls)
    dHO = (min(float(np.linalg.norm(xyz[h[0]] - xyz[c])) for c in carbonyls)
           if h else float("nan"))
    return dHO, dOO


def scan(name: str, smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi)
    idx = dc._core_idx(mol)
    ens = planar_ensemble(smi, n_confs=N_OPEN, k=K, seed=42)
    opens = []
    for m3d, twist, _mmff in ens:
        try:
            xyz, e, _d, _n, ok = xtb_optimise(m3d, SOLVENT)
        except Exception:
            continue
        dHO, dOO = hbond(m3d, xyz, idx)
        opens.append({"E": e, "twist": twist, "dHO": dHO, "dOO": dOO,
                      "d_CaCe": float(np.linalg.norm(xyz[idx["Ca"]] - xyz[idx["Ce"]])),
                      "conv": bool(ok)})
    if not opens:
        return {"name": name, "error": "no open conformers"}
    opens.sort(key=lambda r: r["E"])

    csmi = dc.open_to_closed_keto(smi)
    closed = []
    if csmi:
        for c in _confs_of(Chem.MolFromSmiles(csmi), N_CLOSED, K):
            try:
                _x, e, _d, _n, ok = xtb_optimise(c, SOLVENT)
            except Exception:
                continue
            closed.append(e)
    e_open = opens[0]["E"]
    e_keto = min(closed) if closed else None
    return {"name": name, "smiles": smi, "opens": opens,
            "n_closed": len(closed),
            "E_open": e_open, "E_keto": e_keto,
            "dE_keto": (e_keto - e_open) * H2KCAL if e_keto is not None else None,
            "spread_open": (opens[-1]["E"] - opens[0]["E"]) * H2KCAL,
            "spread_closed": (max(closed) - min(closed)) * H2KCAL if len(closed) > 1 else 0.0}


VARIANTS = [("C5-H", None, None), ("C5-Me", "Ca", "C"), ("C4-Me", "Cb", "C")]


def main():
    jobs = []
    for dname, d in DONORS.items():
        parent = f"{d}/C=C/C=C(\\O)C{A_BARB}"
        for vname, key, frag in VARIANTS:
            smi = parent if key is None else attach(parent, key, frag)
            if smi:
                jobs.append((f"{dname} {vname}", smi))

    print(f"re-sampling {len(jobs)} molecules at k={K} (was 3), "
          f"n_confs={N_OPEN} (was 120)\n")
    with ThreadPoolExecutor(max_workers=6) as ex:
        recs = list(ex.map(lambda t: scan(*t), jobs))
    recs = [r for r in recs if "error" not in r]

    print("A  OPEN-FORM CONFORMER LADDER.  dHO = enol H...O=C; < 2.2 A means the "
          "intramolecular\n   hydrogen bond is formed. rel = kcal/mol above that "
          "molecule's own best conformer.")
    for r in recs:
        print(f"\n   {r['name']}   (spread {r['spread_open']:.2f} kcal/mol over "
              f"{len(r['opens'])} conformers)")
        print(f"      {'rel':>7s} {'dHO':>6s} {'dOO':>6s} {'twist':>6s} "
              f"{'d_CaCe':>7s}  H-bond?")
        for c in r["opens"]:
            rel = (c["E"] - r["opens"][0]["E"]) * H2KCAL
            print(f"      {rel:7.2f} {c['dHO']:6.2f} {c['dOO']:6.2f} "
                  f"{c['twist']:6.1f} {c['d_CaCe']:7.3f}  "
                  f"{'YES' if c['dHO'] < 2.2 else 'no'}")

    print("\n\nB  DOES THE GROUND TRUTH SURVIVE BETTER SAMPLING?")
    print(f"   {'molecule':18s} {'dE_keto(k=8)':>13s} {'ddE vs C5-H':>12s} "
          f"{'best conf H-bonded?':>20s}")
    prev = json.load(open(os.path.join(HERE, "data", "steric_groundtruth.json")))
    prev = {p["name"]: p for p in prev}
    by = {r["name"]: r for r in recs}
    for dname in DONORS:
        base = by.get(f"{dname} C5-H")
        if not base:
            continue
        for vname, _k, _f in VARIANTS:
            r = by.get(f"{dname} {vname}")
            if not r or r["dE_keto"] is None:
                continue
            dd = r["dE_keto"] - base["dE_keto"]
            print(f"   {r['name']:18s} {r['dE_keto']:13.2f} {dd:12.2f} "
                  f"{('YES' if r['opens'][0]['dHO'] < 2.2 else 'NO'):>20s}")
    print("\n   side by side with the k=3 run:")
    print(f"   {'molecule':18s} {'ddE k=3':>9s} {'ddE k=8':>9s} {'change':>9s}")
    for dname in DONORS:
        b8, b3 = by.get(f"{dname} C5-H"), prev.get(f"{dname} C5-H")
        if not b8 or not b3:
            continue
        for vname, _k, _f in VARIANTS:
            r8, r3 = by.get(f"{dname} {vname}"), prev.get(f"{dname} {vname}")
            if not r8 or not r3 or r3.get("dE_keto") is None:
                continue
            d8 = r8["dE_keto"] - b8["dE_keto"]
            d3 = r3["dE_keto"] - b3["dE_keto"]
            print(f"   {r8['name']:18s} {d3:9.2f} {d8:9.2f} {d8-d3:+9.2f}")

    out = os.path.join(HERE, "data", "open_conformer_audit.json")
    with open(out, "w") as f:
        json.dump(recs, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
