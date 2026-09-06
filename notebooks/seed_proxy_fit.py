"""Fit the cheap xTB proxies -- electronic AND steric -- against the measured seed.

Reads notebooks/data/dasa_literature_seed.csv, builds a planar conformer ENSEMBLE
per molecule, optimises each member with GFN2-xTB, and reports:

  1. ensemble spread per descriptor  -> does Boltzmann averaging change anything?
  2. electronic proxies (BLA, dipole) vs measured slope and dark equilibrium
  3. STERIC descriptors vs the RESIDUALS of the electronic fit -- the right test,
     because sterics is only useful if it explains what electronics cannot
  4. a designed bulk ladder at fixed electronics (sensitivity check, no data)

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      ~/miniconda3/envs/reinvent4/bin/python notebooks/seed_proxy_fit.py
Env:  SPF_NCONFS (default 120), SPF_K (default 3), SPF_WORKERS (default 6)
"""
from __future__ import annotations
import csv, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr, linregress
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from calibrate_proxies import xtb_optimise, bond_lengths
from dasa_descriptors import planar_ensemble, boltzmann, steric_descriptors


LEGEND = """
DESCRIPTOR LEGEND  (electronic first, then steric; all lengths in Angstrom)

  BLA        bond-length alternation of the triene N-C5=C4-C3=C2(OH)-C1=acceptor:
             mean(formal SINGLE bonds) - mean(formal DOUBLE bonds).
             large positive = neutral polyene, electrons localised
             near zero / negative = alternation washed out or inverted, i.e. the
             ground state is charge-separated (cyanine-like)
  d_mid      d(C2=C3) - d(C3-C4). Peterson's specific claim, measured: charge
             separation lengthens C2-C3 and shortens C3-C4
  dipole     GFN2-xTB ground-state dipole moment, Debye
  d_C1C5     through-space distance between the two carbons that must bond during
             cyclization (~4.8 A in every untethered open form)
  twist      max deviation from planarity along donor-triene-acceptor, degrees
             (+/- column = spread across the conformer ensemble)

  vbur_*     PERCENT BURIED VOLUME, reported as a FRACTION 0-1, not a percent.
             Put a ball at a chosen centre; report what fraction of it is already
             filled by van der Waals volume. A crowding number.
      vbur_N     ball r=3.5 A at the donor NITROGEN, counting only the donor
                 substituents -> how hindered the nitrogen is
      vbur_C5    ball r=3.5 A at C5 (the triene carbon bonded to N), donor
                 substituents only -> does the donor reach OVER the carbon that
                 has to pyramidalise (sp2 -> sp3) when the ring closes?
      vbur_site  ball r=4.0 A at the C1...C5 MIDPOINT, counting everything that is
                 not the DASA backbone -> total crowding where the bond forms
  sterimol_* Verloop Sterimol parameters, axis pointing from N into the substituent
      L          reach along the axis
      B5         widest reach perpendicular to the axis
      B1         NARROWEST: the smallest maximum width over all rotations about the
                 axis. "How thin is it in its thinnest direction" - classically the
                 parameter that governs approach to a reaction site
  n_donor_heavy  heavy-atom count of the donor substituents (cheap topological control)
"""

SEED    = os.path.join(HERE, "data", "dasa_literature_seed.csv")
NCONFS  = int(os.environ.get("SPF_NCONFS", "120"))
KCONF   = int(os.environ.get("SPF_K", "3"))
WORKERS = int(os.environ.get("SPF_WORKERS", "6"))
ELEC    = ("BLA", "d_mid", "dipole")
STERIC  = ("vbur_N", "vbur_C5", "vbur_site", "sterimol_B1", "sterimol_B5",
           "sterimol_L", "d_C1C5", "n_donor_heavy")


def profile(smi, label=""):
    """Ensemble -> per-conformer xTB optimisation -> Boltzmann-averaged descriptors."""
    mol = Chem.MolFromSmiles(smi)
    idx = dc._core_idx(mol) if mol else None
    ens = planar_ensemble(smi, n_confs=NCONFS, k=KCONF)
    if idx is None or not ens:
        return {"label": label, "error": "geometry"}
    out = {"label": label, "n_conf": len(ens),
           "twist_deg": round(float(np.mean([t for _, t, _ in ens])), 1),
           "twist_sd": round(float(np.std([t for _, t, _ in ens])), 1)}
    for sv in ("toluene", "water"):
        per, energies = [], []
        for m3d, _tw, _e in ens:
            xyz, e, dip, _n, _ok = xtb_optimise(m3d, sv)
            d = bond_lengths(xyz, idx)
            d["dipole"] = dip
            d.update(steric_descriptors(m3d, xyz, idx))
            per.append(d); energies.append(e)
        for key in set().union(*(p.keys() for p in per)):
            mean, sd, rng = boltzmann([p[key] for p in per], energies)
            out[f"{sv}_{key}"] = mean
            out[f"{sv}_{key}_sd"] = sd
            out[f"{sv}_{key}_range"] = rng
            out[f"{sv}_{key}_lowest"] = per[int(np.argmin(energies))][key]
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def corr(x, y, xl, yl, expect=None):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a, b = map(np.array, zip(*pairs))
    if np.std(a) == 0:
        return None
    rho, p = spearmanr(a, b)
    lr = linregress(a, b)
    tag = ""
    if expect is not None:
        tag = "AGREES" if (rho > 0) == (expect > 0) else "WRONG SIGN"
        if abs(rho) < 0.5:
            tag = "none"
    print(f"    {xl:16s} vs {yl:22s} n={len(pairs):2d}  rho={rho:+.3f} p={p:.3f}  "
          f"R2={lr.rvalue**2:.3f}  {tag}")
    return lr


def main():
    rows = list(csv.DictReader(open(SEED)))
    print(LEGEND)
    print(f"n_confs={NCONFS}, k={KCONF} lowest-MMFF planar conformers, "
          f"Boltzmann-weighted on xTB energies\n")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        recs = list(ex.map(lambda r: {**r, **profile(r["smiles_open"], r["compound_id"])},
                           rows))
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)}/{len(rows)} ensembles OK\n")

    # ---- 1. does Boltzmann averaging matter? -------------------------------
    print("1) ENSEMBLE SPREAD  (weighted SD across the ensemble vs the spread "
          "across the whole series)")
    print(f"    {'descriptor':16s} {'series range':>13s} {'median ens SD':>14s} "
          f"{'worst ens SD':>13s}  {'SD/range':>9s}  verdict")
    for key in ELEC + STERIC:
        vals = np.array([r[f"toluene_{key}"] for r in recs])
        sds  = np.array([r[f"toluene_{key}_sd"] for r in recs])
        rng = float(vals.max() - vals.min())
        if rng == 0:
            continue
        frac = float(np.median(sds)) / rng
        verdict = ("single conformer OK" if frac < 0.10 else
                   "BOLTZMANN NEEDED" if frac > 0.25 else "borderline")
        print(f"    {key:16s} {rng:13.4f} {np.median(sds):14.4f} "
              f"{sds.max():13.4f}  {frac:9.1%}  {verdict}")

    # ---- 2. electronic proxies --------------------------------------------
    print("\n2) ELECTRONIC PROXIES")
    slope = [fnum(r["solvatochromic_slope_nm"]) for r in recs]
    print("   [solvatochromic slope]  more charge separation -> more negative slope")
    fit_bla = corr([r["toluene_BLA"] for r in recs], slope, "BLA(tol)", "slope (nm)", +1)
    corr([r["toluene_dipole"] for r in recs], slope, "dipole(tol)", "slope (nm)", -1)
    for solv in ("CDCl3", "MeOH"):
        sub = [r for r in recs if r["equilibrium_solvent"] == solv]
        if len(sub) < 3:
            continue
        pct = [fnum(r["pct_open_equilibrium"]) for r in sub]
        print(f"   [dark equilibrium, {solv}]  more charge separation -> more open")
        corr([r["toluene_BLA"] for r in sub], pct, "BLA(tol)", f"% open ({solv})", -1)
        corr([r["water_dipole"] for r in sub], pct, "dipole(wat)", f"% open ({solv})", +1)

    # ---- 3. sterics vs the residuals of the electronic fit -----------------
    print("\n3) STERICS  (the test that matters: do they explain what BLA cannot?)")
    print(f"    {'descriptor':16s} {'id':>4s} ..." )
    print(f"    {'compound':6s} {'vbur_N':>7s} {'vbur_C5':>8s} {'vbur_site':>10s} "
          f"{'B1':>6s} {'B5':>6s} {'L':>6s} {'twist':>6s} {'+/-':>5s}  measured")
    for r in recs:
        meas = []
        if r["solvatochromic_slope_nm"]:
            meas.append(f"slope {r['solvatochromic_slope_nm']}")
        if r["pct_open_equilibrium"]:
            meas.append(f"{r['pct_open_equilibrium']}% open/{r['equilibrium_solvent']}")
        if r["half_life_s"]:
            meas.append(f"t1/2 {r['half_life_s']}s")
        print(f"    {r['compound_id']:6s} {r['toluene_vbur_N']:7.3f} "
              f"{r['toluene_vbur_C5']:8.3f} {r['toluene_vbur_site']:10.3f} "
              f"{r['toluene_sterimol_B1']:6.2f} {r['toluene_sterimol_B5']:6.2f} "
              f"{r['toluene_sterimol_L']:6.2f} {r['twist_deg']:6.1f} "
              f"{r['twist_sd']:5.1f}  {'; '.join(meas)}")

    if fit_bla is not None:
        print("\n   [residuals of  slope ~ BLA ]  a steric term should pick these up")
        resid, keep = [], []
        for r in recs:
            s = fnum(r["solvatochromic_slope_nm"])
            if s is None:
                continue
            resid.append(s - (fit_bla.slope * r["toluene_BLA"] + fit_bla.intercept))
            keep.append(r)
        for key in STERIC:
            corr([r[f"toluene_{key}"] for r in keep], resid, key, "slope residual")

    print("\n   [dark equilibrium residuals, CDCl3]")
    sub = [r for r in recs if r["equilibrium_solvent"] == "CDCl3"]
    if len(sub) >= 5:
        pct = np.array([fnum(r["pct_open_equilibrium"]) for r in sub])
        bla = np.array([r["toluene_BLA"] for r in sub])
        lr = linregress(bla, pct)
        res = pct - (lr.slope * bla + lr.intercept)
        print(f"    (BLA alone explains R2={lr.rvalue**2:.3f} of % open here)")
        for key in STERIC:
            corr([r[f"toluene_{key}"] for r in sub], list(res), key, "%open residual")

    out = os.path.join(HERE, "data", "seed_proxy_fit.csv")
    keys = sorted({k for r in recs for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(recs)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
