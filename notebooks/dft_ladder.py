"""M06-2X single points on xTB geometries, for the D1-H / D1-M steric pair.

THE QUESTION. GFN2-xTB got the C5-methyl effect backwards: -1.39 kcal/mol against a
measured +2.275 (Chem. Commun. 2022, 58, 2303, Table 1: 29% -> 95% open). Two
candidate explanations, and they need separating:

    (a) LEVEL OF THEORY -- GFN2 is soft on short-range steric repulsion, and the
        M06-2X the paper used would get it right on the same geometry.
    (b) GEOMETRY -- the steric clash physically distorts the molecule and xTB does
        not reproduce that distortion, so no single point on an xTB geometry can
        recover it.

Single points on xTB geometries test (a) while holding (b) fixed. If the sign flips
to positive, it is the functional and a cheap single-point protocol is viable. If it
stays negative, geometry is the problem and nothing short of DFT optimisation helps
-- which the timing below says is not a laptop job.

MEASURED COST on this machine (8 threads, D1-H, 44 atoms):
    M06-2X/6-31G*  no density fitting   57.6 s/SCF cycle  -> geometry opt ~40 h
    ... with density fitting            17.0 s/cycle      -> ~12 h
    ... DF + grid level 1                7.4 s/cycle      -> ~5 h
A DFT geometry optimisation is therefore 50-120 h for these ten species. A single
point is ~7-15 min. That gap is the whole reason this script exists.

DEVIATION FROM THE PAPER: they used SMD; this pyscf build was compiled without the
SMD module, so C-PCM is used instead. Toluene is eps=2.4 (nearly vacuum) and the
quantity is a DIFFERENCE between two close neutral analogues, so the solvation model
should largely cancel -- but it is a deviation and the number is not strictly their
protocol.

Resumable and fail-safe: every single point is committed to the same SQLite cache the
xTB work uses, keyed by (method, recipe, solvent, canonical SMILES), so re-running
skips what is done and a kill costs one calculation.

    python notebooks/dft_ladder.py --status
    python notebooks/dft_ladder.py --budget 3600
    python notebooks/dft_ladder.py
"""
from __future__ import annotations
import argparse, os, signal, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import cache, mol_with_cached_geometry, DEFAULT_DB

H2KCAL, RT, SOLV = 627.5094740631, 0.5925, "toluene"
XTB_OPEN, XTB_CLOSED = "planar120", "lowest20"
BASIS, XC = "6-31+g**", "m06-2x"
METHOD = f"{XC}/{BASIS}//GFN2-xTB+CPCM"
_STOP = False

D, A = "C1Cc2ccccc2N1", "=C1C(=O)OC(C)(C)OC1=O"
PAIR = {"D1-H  parent":    f"{D}C=CC=C(O)C{A}",
        "D1-M  C5-methyl": f"{D}C(C)=CC=C(O)C{A}"}
MEASURED_DDG = +2.275


def _on_signal(_s, _f):
    global _STOP
    _STOP = True
    print("\n  interrupt: finishing the current single point, then stopping "
          "(re-run to resume)", flush=True)


def dft_single_point(smiles: str, geom_recipe: str, db: str = DEFAULT_DB):
    """M06-2X//xTB single point, cached. Returns dict with energy in Hartree."""
    c = cache(db)
    canon = Chem.CanonSmiles(smiles)
    key = c.key(canon, SOLV, f"sp-{geom_recipe}", METHOD)
    hit = c.get(key)
    if hit is not None:
        return hit
    geom = c.get(c.key(canon, SOLV, geom_recipe))
    if geom is None or geom["status"] != "ok":
        return {"status": "fail", "error": f"no xTB geometry ({geom_recipe})", "cached": False}
    t0 = time.time()
    try:
        from pyscf import gto, dft
        m = mol_with_cached_geometry(canon, geom)
        xyz = np.asarray(geom["xyz"])
        atoms = [(a.GetSymbol(), tuple(float(v) for v in xyz[a.GetIdx()]))
                 for a in m.GetAtoms()]
        mol = gto.M(atom=atoms, basis=BASIS, verbose=0)
        mf = dft.RKS(mol, xc=XC).density_fit()
        mf.grids.level = 3
        try:                                   # C-PCM; SMD is not in this build
            mf = mf.PCM()
            mf.with_solvent.method = "C-PCM"
            mf.with_solvent.eps = 2.374        # toluene
            solv = "C-PCM(toluene)"
        except Exception:
            solv = "gas phase"
        mf.conv_tol = 1e-8
        mf.max_cycle = 100
        e = mf.kernel()
        rec = {"status": "ok" if mf.converged else "fail",
               "energy": float(e), "dipole": None, "numbers": None, "xyz": None,
               "grad_calls": None, "converged": bool(mf.converged),
               "error": None if mf.converged else f"SCF not converged ({solv})"}
    except Exception as exc:
        rec = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    rec["seconds"] = time.time() - t0
    c.put(key, canon, SOLV, f"sp-{geom_recipe}", rec, method=METHOD)
    rec["cached"] = False
    return rec


def units():
    out = []
    for label, smi in PAIR.items():
        out.append((label, "open", smi, XTB_OPEN))
        for d in dc.closed_stereoisomers(smi, form=dc.closed_forms_for_solvent(SOLV)):
            out.append((label, f"closed:{d['form']}", d["smiles"], XTB_CLOSED))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--budget", type=float, default=0)
    a = ap.parse_args()
    signal.signal(signal.SIGINT, _on_signal); signal.signal(signal.SIGTERM, _on_signal)

    c = cache()
    us = units()
    pend = [u for u in us
            if c.get(c.key(Chem.CanonSmiles(u[2]), SOLV, f"sp-{u[3]}", METHOD)) is None]
    print(f"{METHOD}\n  {len(us)} single points, {len(pend)} pending")
    if a.status:
        return 0

    t0 = time.time()
    for i, (label, role, smi, recipe) in enumerate(pend, 1):
        if _STOP or (a.budget and time.time() - t0 > a.budget):
            print(f"  stopping with {len(pend)-i+1} left"); break
        r = dft_single_point(smi, recipe)
        el = time.time() - t0
        print(f"  [{i}/{len(pend)}] {label:18s} {role:18s} {r['status']:4s} "
              f"{r.get('seconds',0)/60:5.1f} min  (elapsed {el/60:.0f} min)"
              + (f"  {r.get('error')}" if r["status"] != "ok" else ""), flush=True)

    print("\n" + "=" * 68)
    res = {}
    for label, smi in PAIR.items():
        op = dft_single_point(smi, XTB_OPEN)
        if op["status"] != "ok":
            print(f"  {label}: open form incomplete"); continue
        es = []
        for d in dc.closed_stereoisomers(smi, form=dc.closed_forms_for_solvent(SOLV)):
            r = dft_single_point(d["smiles"], XTB_CLOSED)
            if r["status"] == "ok":
                es.append(r["energy"] * H2KCAL)
        if not es:
            print(f"  {label}: closed manifold incomplete"); continue
        e = np.array(es); emin = e.min()
        res[label] = emin - RT*np.log(np.exp(-(e-emin)/RT).sum()) - op["energy"]*H2KCAL
        print(f"  {label:18s} dE(open->closed) = {res[label]:+7.2f} kcal/mol "
              f"({len(es)} isomers)")
    if len(res) == 2:
        dd = res["D1-M  C5-methyl"] - res["D1-H  parent"]
        print(f"\n  ddE(C5-methyl) = {dd:+.2f}   measured {MEASURED_DDG:+.2f}   "
              f"error {dd-MEASURED_DDG:+.2f} kcal/mol")
        print(f"  GFN2-xTB gave -1.39 on the same geometries.")
        print(f"  VERDICT: sign is {'CORRECT (positive)' if dd > 0 else 'STILL WRONG (negative)'}"
              f" -> {'the functional was the problem; single points are viable' if dd > 0 else 'geometry is the problem; single points cannot fix it'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
