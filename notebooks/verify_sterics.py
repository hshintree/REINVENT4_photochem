"""Does %V_bur actually deserve to be here? Four tests it has to survive.

THE OBJECTION (correct): %V_bur was built by Cavallo for NHC ligands on METALS.
A metal-ligand bond is ~2.0-2.3 A and the metal's own radius is 2-3x nitrogen's,
so the standard 3.5 A sphere was calibrated against a geometry nothing like ours
(N-C ~1.35-1.47 A). A descriptor whose one free parameter was tuned for a
different system is guilty until proven useful.

  T1 RADIUS STABILITY  - if the compound RANKING flips when the sphere radius
     changes, the 3.5 A choice is doing the work and the descriptor is arbitrary.
  T2 COLLINEARITY      - if %V_bur just tracks how many atoms are near N, it is a
     fancy heavy-atom count and adds nothing over n_donor_heavy.
  T3 DISTORTION ENERGY - the descriptor that owes nothing to organometallic
     chemistry: restrain C1...C5 toward each other and measure what it COSTS in
     kcal/mol. That is the actual physical thing sterics does to this reaction.
     If %V_bur is a good steric proxy it should track this; if it does not, use
     this instead.
  T4 C5 SUBSTITUTION   - Peterson Chem. Sci. 2023 report that compound 2, carrying
     a substituent at the triene 5-position, is >96% open in DCM/toluene against
     ~30% for compound 1 with THE SAME DONOR AND ACCEPTOR. That is the one clean
     steric experiment in the DASA literature. We cannot reproduce the numbers
     (cmpd 1's structure is not pinned), but a descriptor that cannot even SEE a
     C5 methyl is disqualified.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from calibrate_proxies import xtb_optimise, bond_lengths, _calculator, BOHR
from dasa_descriptors import (planar_ensemble, steric_descriptors, buried_volume,
                              donor_side_atoms, _radii)

RADII = (2.5, 3.0, 3.5, 4.0, 5.0)
H2KCAL = 627.5094740631
D_TARGET = 3.0        # Angstrom; from ~4.8 A equilibrium toward the ~1.55 A bond


def restrained_energy(mol3d, xyz0_ang, idx, solvent, d_target=D_TARGET,
                      k=2.0, maxiter=300):
    """GFN2-xTB energy of the geometry in which C1...C5 is held at `d_target`,
    everything else relaxed. Returned relative to the free minimum, kcal/mol.

    A harmonic restraint is added to the objective and its gradient; k is stiff
    enough (2 Hartree/Bohr^2 ~ 4500 kcal/mol/A^2) that the constraint holds to
    ~0.01 A. The reported energy is the BARE xTB energy at that geometry, with the
    restraint term removed, so it is a real distortion energy.
    """
    a, b = idx["Ca"], idx["Ce"]          # C5 and C1
    pos = np.asarray(xyz0_ang) * BOHR
    nums = np.array([at.GetAtomicNum() for at in mol3d.GetAtoms()], dtype=int)
    calc = _calculator(nums, pos, solvent)
    d0 = d_target * BOHR
    state = {"res": None, "bare": None}

    def fun(x):
        p = x.reshape(-1, 3)
        calc.update(p)
        try:
            res = calc.singlepoint(state["res"]) if state["res"] else calc.singlepoint()
        except Exception:
            state["res"] = None
            res = calc.singlepoint()
        state["res"] = res
        e = res.get_energy()
        state["bare"] = e
        g = np.asarray(res.get_gradient()).copy()
        v = p[a] - p[b]
        d = np.linalg.norm(v)
        e += 0.5 * k * (d - d0) ** 2
        gd = k * (d - d0) * v / d
        g[a] += gd
        g[b] -= gd
        return e, g.flatten()

    out = minimize(fun, pos.flatten(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-10})
    p = out.x.reshape(-1, 3)
    d_final = float(np.linalg.norm(p[a] - p[b]) / BOHR)
    return state["bare"], d_final


def profile(name, smi, solvent="toluene"):
    mol = Chem.MolFromSmiles(smi)
    idx = dc._core_idx(mol) if mol else None
    if idx is None or not dc.is_dasa(mol):
        return {"name": name, "error": "not a DASA"}
    ens = planar_ensemble(smi, n_confs=120, k=1)
    if not ens:
        return {"name": name, "error": "geometry"}
    m3d = ens[0][0]
    xyz, e_free, dip, _n, _ok = xtb_optimise(m3d, solvent)
    rec = {"name": name, "smiles": smi, "mw": Descriptors.MolWt(mol),
           "heavy": mol.GetNumHeavyAtoms(), "dipole": dip,
           **bond_lengths(xyz, idx), **steric_descriptors(m3d, xyz, idx)}
    # T1: the same quantity at five sphere radii
    radii = _radii(m3d)
    dmask = donor_side_atoms(m3d, idx)
    for R in RADII:
        rec[f"vburN_{R}"] = buried_volume(xyz, radii, xyz[idx["N"]], R, dmask)
        rec[f"vburC5_{R}"] = buried_volume(xyz, radii, xyz[idx["Ca"]], R, dmask)
    # T3: what it costs to push C1 and C5 together
    try:
        e_con, d_fin = restrained_energy(m3d, xyz, idx, solvent)
        rec["E_distort_kcal"] = (e_con - e_free) * H2KCAL
        rec["d_final"] = d_fin
    except Exception as exc:
        rec["E_distort_kcal"] = None
        rec["d_final"] = None
        rec["distort_error"] = f"{type(exc).__name__}: {exc}"
    return rec


A_BARB = "=C1C(=O)N(C)C(=O)N(C)C1=O"
LADDER = [
    ("L1 dimethylamino",      f"CN(C)/C=C/C=C(\\O)C{A_BARB}"),
    ("L2 diethylamino",       f"CCN(CC)/C=C/C=C(\\O)C{A_BARB}"),
    ("L3 dipropylamino",      f"CCCN(CCC)/C=C/C=C(\\O)C{A_BARB}"),
    ("L4 diisopropylamino",   f"CC(C)N(C(C)C)/C=C/C=C(\\O)C{A_BARB}"),
    ("L5 dicyclohexylamino",  f"C1CCCCC1N(C1CCCCC1)/C=C/C=C(\\O)C{A_BARB}"),
    ("L6 di-tert-butylamino", f"CC(C)(C)N(C(C)(C)C)/C=C/C=C(\\O)C{A_BARB}"),
]
# T4: matched pairs, H vs METHYL at the triene 5-position (alpha to the donor).
# Written without stereo so both members of a pair are treated identically.
PAIRS = [
    ("P-Me2N   C5-H",   f"CN(C)C=CC=C(O)C{A_BARB}"),
    ("P-Me2N   C5-Me",  f"CN(C)C(C)=CC=C(O)C{A_BARB}"),
    ("P-indoline C5-H",  f"CC1Cc2ccccc2N1C=CC=C(O)C{A_BARB}"),
    ("P-indoline C5-Me", f"CC1Cc2ccccc2N1C(C)=CC=C(O)C{A_BARB}"),
    ("P-anilino C5-H",   f"CN(c1ccccc1)C=CC=C(O)C{A_BARB}"),
    ("P-anilino C5-Me",  f"CN(c1ccccc1)C(C)=CC=C(O)C{A_BARB}"),
]


def main():
    seed = list(csv.DictReader(open(os.path.join(HERE, "data",
                                                 "dasa_literature_seed.csv"))))
    jobs = ([(f"S-{r['compound_id']}", r["smiles_open"]) for r in seed]
            + LADDER + PAIRS)
    with ThreadPoolExecutor(max_workers=6) as ex:
        recs = list(ex.map(lambda t: profile(*t), jobs))
    bad = [r for r in recs if "error" in r]
    for r in bad:
        print(f"  !! {r['name']}: {r['error']}")
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)}/{len(jobs)} molecules OK\n")

    # ---------------- T1 ----------------
    print("T1  RADIUS STABILITY - does the compound ranking survive the sphere radius?")
    ref = np.array([r["vburN_3.5"] for r in recs])
    print(f"    {'radius':>7s} {'mean':>7s} {'range':>7s}   rank corr. vs 3.5 A")
    for R in RADII:
        v = np.array([r[f"vburN_{R}"] for r in recs])
        rho, _ = spearmanr(v, ref)
        print(f"    {R:7.1f} {v.mean():7.3f} {v.max()-v.min():7.3f}   rho={rho:+.3f}")
    print("    -> the ranking is what a scoring function uses; if rho stays ~1 the "
          "3.5 A\n       choice is cosmetic and the organometallic provenance does not matter.")

    # ---------------- T2 ----------------
    print("\nT2  COLLINEARITY - is %V_bur just an atom count in disguise?")
    for a, b in (("vbur_N", "n_donor_heavy"), ("vbur_N", "heavy"), ("vbur_N", "mw"),
                 ("sterimol_B1", "n_donor_heavy"), ("vbur_C5", "n_donor_heavy")):
        x = np.array([r[a] for r in recs]); y = np.array([r[b] for r in recs])
        rho, p = spearmanr(x, y)
        verdict = "REDUNDANT" if abs(rho) > 0.9 else ("partly redundant"
                                                      if abs(rho) > 0.7 else "adds shape info")
        print(f"    {a:14s} vs {b:16s} rho={rho:+.3f}  {verdict}")

    # ---------------- T3 ----------------
    print("\nT3  DISTORTION ENERGY - kcal/mol to hold C1...C5 at 3.0 A "
          "(no borrowed calibration)")
    ok = [r for r in recs if r.get("E_distort_kcal") is not None]
    lad = [r for r in ok if r["name"].startswith("L")]
    print(f"    {'molecule':22s} {'E_dist':>8s} {'d_final':>8s} {'vbur_N':>7s} "
          f"{'vbur_C5':>8s} {'B1':>6s} {'BLA':>8s}")
    for r in lad:
        print(f"    {r['name']:22s} {r['E_distort_kcal']:8.2f} {r['d_final']:8.3f} "
              f"{r['vbur_N']:7.3f} {r['vbur_C5']:8.3f} {r['sterimol_B1']:6.2f} "
              f"{r['BLA']:+8.4f}")
    print("\n    Does %V_bur predict the physical cost? (all molecules)")
    ed = [r["E_distort_kcal"] for r in ok]
    for k in ("vbur_N", "vbur_C5", "vbur_site", "sterimol_B1", "sterimol_B5",
              "n_donor_heavy", "BLA"):
        rho, p = spearmanr([r[k] for r in ok], ed)
        print(f"      {k:14s} vs E_distort   rho={rho:+.3f}  p={p:.3f}")
    print("\n    ...and within the ladder alone (electronics held ~fixed):")
    for k in ("vbur_N", "vbur_C5", "sterimol_B1", "n_donor_heavy"):
        rho, p = spearmanr([r[k] for r in lad], [r["E_distort_kcal"] for r in lad])
        print(f"      {k:14s} vs E_distort   rho={rho:+.3f}  p={p:.3f}  n={len(lad)}")

    # ---------------- T4 ----------------
    print("\nT4  C5 SUBSTITUTION - can the descriptors see a methyl at the "
          "5-position at all?")
    pr = {r["name"]: r for r in recs if r["name"].startswith("P-")}
    print(f"    {'pair':14s} {'delta vbur_C5':>14s} {'delta vbur_N':>13s} "
          f"{'delta B1':>9s} {'delta E_dist':>13s} {'delta BLA':>10s}")
    for base in ("P-Me2N  ", "P-indoline", "P-anilino"):
        h = pr.get(f"{base} C5-H".replace("  ", "   ")) or pr.get(f"{base} C5-H")
        m = pr.get(f"{base} C5-Me".replace("  ", "   ")) or pr.get(f"{base} C5-Me")
        if not h or not m:
            continue
        de = (m["E_distort_kcal"] - h["E_distort_kcal"]
              if None not in (m.get("E_distort_kcal"), h.get("E_distort_kcal")) else float("nan"))
        print(f"    {base.strip():14s} {m['vbur_C5']-h['vbur_C5']:+14.4f} "
              f"{m['vbur_N']-h['vbur_N']:+13.4f} "
              f"{m['sterimol_B1']-h['sterimol_B1']:+9.3f} {de:+13.2f} "
              f"{m['BLA']-h['BLA']:+10.4f}")

    out = os.path.join(HERE, "data", "steric_verification.json")
    with open(out, "w") as f:
        json.dump(recs, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
