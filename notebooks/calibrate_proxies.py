"""Calibrate two CHEAP proxies against MEASURED literature DASA data, locally.

The question this answers: can a GFN2-xTB geometry + dipole, which costs seconds
per molecule on a laptop, reproduce the ORDERING of two properties we currently
have no in-loop measurement for?

  PROXY 1  triene bond-length alternation (BLA)   ->  dark equilibrium (% open)
  PROXY 2  ground-state dipole moment             ->  solvatochromic slope

Both are literature-motivated, not invented here:

  * Strothmann, Amanpur, Nevesely, Hecht, Reuter, Margraf, Digital Discovery 2025,
    4, 3098 use "the length of the rotating bond ... predicted with the GFN2-xTB
    method" as a proxy for inverse thermostability of a spiropyran, precisely
    because "semiempirical methods like GFN2-xTB that work well for predicting
    molecular geometries are not sufficiently accurate for barriers". Same trick,
    different photoswitch, and they drove REINVENT with it.
  * Peterson, Neris, Read de Alaniz, Chem. Sci. 2023, 14, 13025: "Large charge
    separation contributes to lower A-B barriers and higher B-B' barriers due to
    increased single bond character around C2-C3 and increased double bond
    character between C3-C4." That is a statement about bond lengths.
  * Same paper states the aqueous design criterion outright: derivatives must be
    "stable, have a high equilibrium of the open form, and have a low charge
    separation indicated by a less negative solvatochromic slope".

DASA numbering (theirs) vs core keys (ours):
    C1 = Cf (acceptor ylidene)   C2 = Cd (enol C-OH)   C3 = Cc
    C4 = Cb                      C5 = Ca (bonded to N)

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      ~/miniconda3/envs/reinvent4/bin/python notebooks/calibrate_proxies.py
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dasa_chem as dc

BOHR = 1.8897261254578281          # Angstrom -> Bohr
WORKERS = int(os.environ.get("CAL_WORKERS", "6"))

# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------
# Literature (2Z,4E) open form. Same stereo pattern as the validated anchor in
# outputs_dasa_full/dft_v2 (ChemSci-1): {donor}/C=C/C=C(\O)C{=acceptor}
def dasa(donor: str, acceptor: str) -> str:
    return f"{donor}/C=C/C=C(\\O)C{acceptor}"


D_INDOLINE_2ME = "CC1Cc2ccccc2N1"      # Hemmer's donor, 2-methylindoline
D_ME2N         = "CN(C)"                # 1st-gen dimethylamino
D_PYRROLIDINE  = "C1CCCN1"              # 1st-gen cyclic
D_INDOLINE     = "C1Cc2ccccc2N1"        # 2nd-gen fused aryl
D_NMEANILINE   = "CN(c1ccccc1)"         # 2nd-gen aniline
D_CF3ANILINE   = "CN(c1ccc(C(F)(F)F)cc1)"   # 2nd-gen, EWG-weakened
D_ADAMANTYL    = "NC12CC3CC(CC(C3)C1)C2"    # Pischel JACS 2026, 1-adamantylamine

A_MELDRUM     = "=C1C(=O)OC(C)(C)OC1=O"
A_DIMEBARB    = "=C1C(=O)N(C)C(=O)N(C)C1=O"
A_PYRAZDIONE  = "=C1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"   # 1,2-diphenyl (Hemmer 3)
A_MEISOX      = "=C1C(=O)ON=C1C"
A_CF3ISOX     = "=C1C(=O)ON=C1C(F)(F)F"
A_INDANDIONE  = "=C1C(=O)c2ccccc2C1=O"
A_CF3PYRAZ    = "=C1C(=O)N(c2ccccc2)N=C1C(F)(F)F"

# --- PANEL 1: Hemmer JACS 2018, 140, 10425. One donor (2-methylindoline), eight
# carbon-acid acceptors, dark equilibrium by 1H NMR in CDCl3. Percentages: 1/2 and
# 4/5 are quoted in the text ("~45% triene", "from 80 to 95%"); 3 is the quoted
# 75-85% band; 6 and 7 are read off Fig. 3b, so treat the ORDER as the datum and
# the value as approximate. Rank correlation is therefore the honest statistic.
PANEL1 = [
    ("H6 indandione",        dasa(D_INDOLINE_2ME, A_INDANDIONE), 5.0),
    ("H1 Meldrum",           dasa(D_INDOLINE_2ME, A_MELDRUM),   45.0),
    ("H2 diMe-barbituric",   dasa(D_INDOLINE_2ME, A_DIMEBARB),  45.0),
    ("H3 pyrazolidinedione", dasa(D_INDOLINE_2ME, A_PYRAZDIONE), 80.0),
    ("H4 Me-isoxazolone",    dasa(D_INDOLINE_2ME, A_MEISOX),     80.0),
    ("H5 CF3-isoxazolone",   dasa(D_INDOLINE_2ME, A_CF3ISOX),    95.0),
    ("H7 CF3-pyrazolone",    dasa(D_INDOLINE_2ME, A_CF3PYRAZ),   97.0),
]

# --- PANEL 2: measured solvatochromic slopes (nm, vs E_T(30), reported practice
# for DASAs; more negative = more charge-separated) and, where measured, the dark
# equilibrium in METHANOL -- the polar-protic number this whole project is about.
#   Pischel, JACS 2026, 148, 130   : 1-adamantylamine donor, -79 / -65 nm
#   Peterson, Chem. Sci. 2023, 14, 13025 : aniline tethered to triene C5 by a
#     3-carbon linkage (5-ring). 3 -> -56 nm and only the open form by NMR in
#     MeOH; 4 (Me-pyrazolone) -> -32 nm, 70% open in MeOH; 5 (4-iodoaniline)
#     -> -20 nm, 24% open in MeOH, and the only DASA that switches in 60:40
#     THF:water.
# STRUCTURES: the adamantyl pair is unambiguous from the text. The three tethered
# compounds are RECONSTRUCTED from the prose ("amine donor tethered to the triene
# via a three-carbon linkage, resulting in a 5-membered ring") -- check them
# against Fig. 2a of that paper before trusting any fit.
PANEL2 = [
    ("Pischel-1 Ad/barbituric",  r"N(/C=C/C=C(\O)C=C1C(=O)N(C)C(=O)N(C)C1=O)C45CC6CC(CC(C6)C4)C5", -79.0, None,  "certain"),
    ("Pischel-2 Ad/Meldrum",     r"N(/C=C/C=C(\O)C=C1C(=O)OC(C)(C)OC1=O)C45CC6CC(CC(C6)C4)C5",      -65.0, None,  "certain"),
    ("Peterson-3 teth/Meldrum",  r"C1(CCCN1c1ccccc1)=CC=C(O)C=C1C(=O)OC(C)(C)OC1=O",                -56.0, 99.0,  "reconstructed"),
    ("Peterson-4 teth/Me-pyraz", r"C1(CCCN1c1ccccc1)=CC=C(O)C=C1C(=O)N(c2ccccc2)N=C1C",             -32.0, 70.0,  "reconstructed"),
    ("Peterson-5 teth-I/Me-pyr", r"C1(CCCN1c1ccc(I)cc1)=CC=C(O)C=C1C(=O)N(c2ccccc2)N=C1C",          -20.0, 24.0,  "reconstructed"),
]

# --- PANEL 3: donor ladder at a fixed acceptor. No compound-matched slopes, but
# the literature ordering is unambiguous: 1st-gen dialkyl is far more charge
# separated than 2nd-gen aniline, and EWGs on the aniline reduce it further
# (Peterson: 4-iodo took the slope from -32 to -20 nm). Directional check only.
PANEL3 = [
    ("D dimethylamino (1st)",  dasa(D_ME2N,        A_DIMEBARB)),
    ("D pyrrolidino (1st)",    dasa(D_PYRROLIDINE, A_DIMEBARB)),
    ("D indoline (2nd)",       dasa(D_INDOLINE,    A_DIMEBARB)),
    ("D N-Me-aniline (2nd)",   dasa(D_NMEANILINE,  A_DIMEBARB)),
    ("D 4-CF3-aniline (2nd)",  dasa(D_CF3ANILINE,  A_DIMEBARB)),
]


# ---------------------------------------------------------------------------
# GFN2-xTB: geometry optimisation (scipy L-BFGS on xtb gradients) + properties
# ---------------------------------------------------------------------------
def _calculator(nums, pos_bohr, solvent):
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED
    calc = Calculator(Param.GFN2xTB, nums, pos_bohr)
    calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_max_iterations(300)
    if solvent:
        from xtb.utils import get_solvent
        s = get_solvent(solvent)
        if s is None:
            # This used to fall through silently and run GAS PHASE. The seed CSV
            # stores "CDCl3" and "MeOH", both of which get_solvent returns None
            # for, so a solvent-matched campaign would have been gas phase
            # throughout with no warning. Fail loudly instead.
            raise ValueError(
                f"xtb does not recognise solvent {solvent!r}; refusing to run gas "
                f"phase by accident. Use e.g. 'chloroform', 'methanol', 'ch2cl2', "
                f"'water', 'toluene'.")
        calc.set_solvent(s)
    return calc


def xtb_optimise(mol3d, solvent, maxiter=400):
    """GFN2-xTB geometry optimisation in ALPB `solvent`.

    The local `xtb --opt` CLI is broken (Fortran format bug in optimizer.f90), so
    the optimisation is driven here: xtb-python supplies energy + gradient, scipy
    supplies L-BFGS. Coordinates are handled in Bohr throughout, which is what the
    xtb gradient is expressed in.
    """
    conf = mol3d.GetConformer()
    pos = conf.GetPositions() * BOHR
    nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
    calc = _calculator(nums, pos, solvent)
    state = {"res": None, "n": 0}

    def fun(x):
        p = x.reshape(-1, 3)
        calc.update(p)
        try:
            res = calc.singlepoint(state["res"]) if state["res"] else calc.singlepoint()
        except Exception:
            state["res"] = None
            res = calc.singlepoint()
        state["res"] = res
        state["n"] += 1
        return res.get_energy(), np.asarray(res.get_gradient()).flatten()

    out = minimize(fun, pos.flatten(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": 1e-5, "ftol": 1e-10})
    xyz_ang = out.x.reshape(-1, 3) / BOHR
    res = state["res"]
    dipole = float(np.linalg.norm(res.get_dipole())) * 2.541746  # a.u. -> Debye
    return xyz_ang, float(out.fun), dipole, state["n"], bool(out.success)


def bond_lengths(xyz, idx):
    """Triene bond lengths (Angstrom) from an optimised geometry.

    formal doubles: Ca=Cb (C5=C4), Cc=Cd (C3=C2), Ce=Cf (C1 ylidene)
    formal singles: Cb-Cc (C4-C3), Cd-Ce (C2-C1)
    """
    d = lambda a, b: float(np.linalg.norm(xyz[idx[a]] - xyz[idx[b]]))
    b = {
        "N-C5":  d("N", "Ca"),
        "C5=C4": d("Ca", "Cb"),
        "C4-C3": d("Cb", "Cc"),
        "C3=C2": d("Cc", "Cd"),
        "C2-C1": d("Cd", "Ce"),
        "C1=Cf": d("Ce", "Cf"),
    }
    doubles = [b["C5=C4"], b["C3=C2"], b["C1=Cf"]]
    singles = [b["C4-C3"], b["C2-C1"]]
    b["BLA"] = float(np.mean(singles) - np.mean(doubles))
    # Peterson's specific claim: charge separation lengthens C2-C3 and shortens
    # C3-C4. delta_mid goes toward zero (and can invert) as the ground state
    # becomes more zwitterionic / cyanine-like.
    b["d_mid"] = b["C3=C2"] - b["C4-C3"]
    return b


def profile(label, smi, solvents=("toluene", "water")):
    t0 = time.time()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"label": label, "error": "unparseable SMILES"}
    if not dc.is_dasa(mol):
        return {"label": label, "error": "does not match the DASA open-form core"}
    if dc.is_legacy_core(mol):
        return {"label": label, "error": "LEGACY core"}
    idx = dc._core_idx(mol)
    if idx is None:
        return {"label": label, "error": "no core index"}

    mol3d, twist = dc.planar_conformer(smi, n_confs=30, seed=42)
    if mol3d is None:
        return {"label": label, "error": "conformer search failed"}

    rec = {"label": label, "smiles": Chem.MolToSmiles(mol),
           "heavy": mol.GetNumHeavyAtoms(), "twist_deg": round(twist, 1)}
    for sv in solvents:
        try:
            xyz, e, dip, ncalls, ok = xtb_optimise(mol3d, sv)
        except Exception as exc:
            rec[sv] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        b = bond_lengths(xyz, idx)
        rec[sv] = {"E_hartree": round(e, 6), "dipole_D": round(dip, 3),
                   "grad_calls": ncalls, "converged": ok,
                   **{k: round(v, 4) for k, v in b.items()}}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
def run(panel, name):
    print(f"\n>>> {name}: {len(panel)} molecules, GFN2-xTB opt in toluene + water",
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        recs = list(ex.map(lambda t: profile(t[0], t[1]), panel))
    for r in recs:
        if "error" in r:
            print(f"    !! {r['label']}: {r['error']}", flush=True)
    return recs


def col(recs, solvent, key):
    return [r.get(solvent, {}).get(key) if "error" not in r else None for r in recs]


def rank_report(measured, predicted, mlabel, plabel, expect):
    ok = [(m, p) for m, p in zip(measured, predicted) if m is not None and p is not None]
    if len(ok) < 3:
        print(f"    {plabel:22s} -> too few points ({len(ok)})")
        return
    m, p = zip(*ok)
    rho, pv = spearmanr(m, p)
    verdict = "AGREES" if (rho > 0) == (expect > 0) else "WRONG SIGN"
    if abs(rho) < 0.5:
        verdict = "no relationship"
    print(f"    {plabel:22s} vs {mlabel:22s}  rho={rho:+.3f}  p={pv:.3f}   "
          f"n={len(ok)}  expect {'+' if expect > 0 else '-'}  -> {verdict}")


def main():
    all_out = {}

    # ---------------- PANEL 1 ----------------
    r1 = run(PANEL1, "PANEL 1  BLA vs dark equilibrium (Hemmer JACS 2018, CDCl3)")
    all_out["panel1"] = r1
    print(f"\n    {'compound':24s} {'%open':>6s} {'twist':>6s} "
          f"{'BLA_tol':>8s} {'dmid_tol':>9s} {'BLA_wat':>8s} {'dmid_wat':>9s} "
          f"{'mu_tol':>7s} {'mu_wat':>7s}")
    for r, (_, _, pct) in zip(r1, PANEL1):
        if "error" in r:
            continue
        t, w = r.get("toluene", {}), r.get("water", {})
        print(f"    {r['label']:24s} {pct:6.0f} {r['twist_deg']:6.1f} "
              f"{t.get('BLA', float('nan')):8.4f} {t.get('d_mid', float('nan')):9.4f} "
              f"{w.get('BLA', float('nan')):8.4f} {w.get('d_mid', float('nan')):9.4f} "
              f"{t.get('dipole_D', float('nan')):7.2f} {w.get('dipole_D', float('nan')):7.2f}")
    pct = [p for _, _, p in PANEL1]
    print("\n    Hypothesis: more open-favoured  <=>  LESS charge separation  <=>"
          "  LARGER BLA, MORE NEGATIVE d_mid, SMALLER dipole.")
    for sv in ("toluene", "water"):
        print(f"    [{sv}]")
        rank_report(pct, col(r1, sv, "BLA"),   "% open", "BLA",     +1)
        rank_report(pct, col(r1, sv, "d_mid"), "% open", "d_mid",   -1)
        rank_report(pct, col(r1, sv, "dipole_D"), "% open", "dipole", -1)

    # ---------------- PANEL 2 ----------------
    r2 = run([(a, b) for a, b, _, _, _ in PANEL2],
             "PANEL 2  dipole vs measured solvatochromic slope / MeOH equilibrium")
    all_out["panel2"] = r2
    print(f"\n    {'compound':28s} {'slope':>6s} {'%open':>6s} {'twist':>6s} "
          f"{'mu_tol':>7s} {'mu_wat':>7s} {'BLA_tol':>8s} {'dmid_tol':>9s}  source")
    for r, (_, _, slope, pct, conf) in zip(r2, PANEL2):
        if "error" in r:
            continue
        t, w = r.get("toluene", {}), r.get("water", {})
        print(f"    {r['label']:28s} {slope:6.0f} "
              f"{(pct if pct is not None else float('nan')):6.0f} {r['twist_deg']:6.1f} "
              f"{t.get('dipole_D', float('nan')):7.2f} {w.get('dipole_D', float('nan')):7.2f} "
              f"{t.get('BLA', float('nan')):8.4f} {t.get('d_mid', float('nan')):9.4f}  {conf}")
    slopes = [sl for _, _, sl, _, _ in PANEL2]
    meoh   = [pc for _, _, _, pc, _ in PANEL2]
    print("\n    Hypothesis: MORE charge separation <=> LARGER dipole <=> MORE NEGATIVE slope.")
    for sv in ("toluene", "water"):
        print(f"    [{sv}]")
        rank_report(slopes, col(r2, sv, "dipole_D"), "slope (nm)", "dipole", -1)
        rank_report(slopes, col(r2, sv, "BLA"),      "slope (nm)", "BLA",    +1)
        rank_report(slopes, col(r2, sv, "d_mid"),    "slope (nm)", "d_mid",  -1)
    print("\n    And against the MeOH dark equilibrium (n=3, tethered series only):")
    rank_report(meoh, col(r2, "water", "dipole_D"), "% open (MeOH)", "dipole", +1)

    # ---------------- PANEL 3 ----------------
    r3 = run(PANEL3, "PANEL 3  donor ladder at a fixed acceptor (directional)")
    all_out["panel3"] = r3
    print(f"\n    {'compound':24s} {'mu_tol':>7s} {'mu_wat':>7s} {'BLA_tol':>8s} "
          f"{'dmid_tol':>9s} {'twist':>6s}")
    for r in r3:
        if "error" in r:
            continue
        t, w = r.get("toluene", {}), r.get("water", {})
        print(f"    {r['label']:24s} {t.get('dipole_D', float('nan')):7.2f} "
              f"{w.get('dipole_D', float('nan')):7.2f} "
              f"{t.get('BLA', float('nan')):8.4f} {t.get('d_mid', float('nan')):9.4f} "
              f"{r['twist_deg']:6.1f}")
    print("\n    Expect: 1st-gen dialkyl dipole >> 2nd-gen aniline; 4-CF3 lowest.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "proxy_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
