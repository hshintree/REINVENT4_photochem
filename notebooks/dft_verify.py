#!/usr/bin/env python
"""DFT verification stage — turn xTB-screened DASA candidates into DFT-backed rankings.

This is the EXPENSIVE tier of the cascade (NOT in the RL loop). It takes a
candidate CSV (e.g. verify_dasa_outputs.py's verified_candidates.csv), clusters
to a diverse handful of representatives, and for each runs TD-DFT (CAM-B3LYP,
good for the charge-transfer states DASAs have) across a solvent-polarity series
with an implicit-solvent model (ddCOSMO). From that it computes:

  * open-form lambda_max per solvent (visible target ~540-600 nm),
  * the SOLVATOCHROMIC SLOPE = d(lambda_max)/d(solvent polarity) -- the literature's
    quantitative charge-separation metric (switchable sweet spot ~ -20 nm; a very
    large negative slope ~ -56 nm signals the over-charge-separated / trap regime),
  * a colourless-closed check (closed-form should not absorb in the visible).

This converts DASATrap's *directional* xTB signal into a *trustworthy* number.
The final A->B->B' barrier landscape (reversibility) is a further, mostly-manual
DFT step reserved for the top few here -- see the note at the bottom.

Geometry: MMFF conformer + TD-DFT single point (standard solvatochromism screen).
`--dft-opt` adds a DFT geometry optimisation (slower, more accurate).

Local (a few candidates):
    python notebooks/dft_verify.py --csv outputs_dasa_modal/outputs_dasa/verified_candidates.csv \
        --n-reps 6 --quick
For the full parallel run use modal_dft.py (one container per representative).
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc  # noqa: E402

# Solvent polarity series: name -> dielectric constant. Ordered low->high polarity.
SOLVENTS = {
    "toluene": 2.38, "acetone": 20.7, "methanol": 32.6, "acetonitrile": 37.5,
}


def onsager(eps: float) -> float:
    """Onsager reaction-field polarity factor f(eps) = (eps-1)/(2eps+1)."""
    return (eps - 1.0) / (2.0 * eps + 1.0)


def embed(smiles: str, dft_opt: bool = False, xc="camb3lyp", basis="6-31g*"):
    """Return (atom_list_str, charge) from an MMFF (or optional DFT-opt) geometry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
    except Exception:
        pass
    conf = m.GetConformer()
    atoms = "\n".join(
        f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.6f} "
        f"{conf.GetAtomPosition(a.GetIdx()).y:.6f} "
        f"{conf.GetAtomPosition(a.GetIdx()).z:.6f}" for a in m.GetAtoms())
    charge = Chem.GetFormalCharge(mol)
    if dft_opt:
        atoms = _dft_optimize(atoms, charge, xc, basis)
    return atoms, charge


def _dft_optimize(atoms, charge, xc, basis):
    # HARD-capped: floppy peptide/glycol-tailed DASAs never satisfy the
    # displacement criteria (soft modes), so DFT geom-opt would run to the
    # 300-cycle / timeout limit. 20 cycles is enough for small rigid molecules;
    # for the real (floppy) candidates use MMFF (default) + the lambda pre-screen.
    from pyscf import gto, dft
    from pyscf.geomopt.geometric_solver import optimize
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc
    mol_eq = optimize(mf, maxsteps=20)
    return "\n".join(f"{mol_eq.atom_symbol(i)} " +
                     " ".join(f"{c:.6f}" for c in mol_eq.atom_coord(i, unit="Angstrom"))
                     for i in range(mol_eq.natm))


def tddft_lambda(atoms, charge, eps, xc="camb3lyp", basis="6-31g*", nstates=8):
    """Lowest BRIGHT excitation (lambda_max nm, oscillator strength) in a solvent
    of dielectric eps, plus all excitations. TD-DFT (TDA) with ddCOSMO."""
    from pyscf import gto, dft, tddft
    from pyscf.solvent import ddCOSMO
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=0, verbose=0)
    mf = ddCOSMO(dft.RKS(mol)); mf.xc = xc
    mf.with_solvent.eps = eps
    mf.conv_tol = 1e-8
    mf.kernel()
    td = tddft.TDA(mf); td.nstates = nstates; td.kernel()
    exc = []
    for e_ev, fosc in zip(td.e * 27.2114, td.oscillator_strength()):
        if e_ev > 0:
            exc.append((1240.0 / e_ev, float(fosc)))
    bright = [(lam, f) for lam, f in exc if f > 0.05]
    lam_max = max(bright, key=lambda t: t[1])[0] if bright else (exc[0][0] if exc else None)
    return lam_max, exc


def run_candidate(smiles, solvents=None, quick=False, dft_opt=False,
                  xc="camb3lyp", basis="6-31g*", min_visible_nm=420.0):
    """Full DFT characterisation of one DASA: open-form lambda_max across solvents,
    solvatochromic slope, and a closed-form visible-absorption check.

    Has a CHEAP lambda_max PRE-SCREEN: computes the first-solvent lambda_max first
    and, if it's below `min_visible_nm` (a UV absorber, not a real visible DASA),
    returns immediately -- so we never spend the full multi-solvent + closed-form
    budget on a molecule that isn't even coloured. NOTE: geometry defaults to MMFF;
    DFT geometry opt (`dft_opt=True`) is capped and NOT recommended for the floppy
    peptide/glycol-tailed candidates (soft modes never converge)."""
    if not solvents:
        if quick:
            # minimum for a slope: the two polarity extremes of the available set
            _ordered = sorted(SOLVENTS, key=lambda s: SOLVENTS[s])
            solvents = [_ordered[0], _ordered[-1]]
        else:
            solvents = list(SOLVENTS)
    nstates = 5 if quick else 8
    geo = embed(smiles, dft_opt=dft_opt, xc=xc, basis=basis)
    if geo is None:
        return {"smiles": smiles, "error": "embed_failed"}
    atoms, charge = geo

    # --- cheap pre-screen: reject UV absorbers with a FAST small-basis gas-phase
    # TD-DFT (~seconds, vs minutes at 6-31G*). Only needs to tell ~300 nm (UV)
    # from ~550 nm (visible), so a crude basis is fine. ---
    try:
        lam_scr, _ = tddft_lambda(atoms, charge, 1.0, xc, "3-21g", 3)   # eps=1 ~ gas
    except Exception as e:
        return {"smiles": smiles, "error": f"prescreen_failed:{type(e).__name__}"}
    # min_visible_nm > 0 => HARD gate (skip full DFT for UV). min_visible_nm <= 0 =>
    # PRIORITISATION mode: never reject, keep screen_lambda_nm as a cheap ordering
    # signal and still run the full DFT. NOTE the gas/3-21G/CAM-B3LYP pre-screen
    # blue-shifts DASAs ~200 nm (a known-560 nm reference computes ~334 nm), so its
    # ABSOLUTE value is not a visibility verdict -- use it only for RELATIVE ranking.
    if min_visible_nm > 0 and (lam_scr is None or lam_scr < min_visible_nm):
        return {"smiles": smiles, "screen_lambda_nm": round(lam_scr, 1) if lam_scr else None,
                "rejected": "uv_not_visible",
                "note": f"pre-screen lambda_max {lam_scr:.0f} nm < {min_visible_nm:.0f} nm "
                        "-> not a visible photoswitch; full DFT skipped"}

    first = solvents[0]
    lam0, _ = tddft_lambda(atoms, charge, SOLVENTS[first], xc, basis, nstates)
    lam_by_solvent = {first: round(lam0, 1) if lam0 else None}
    for s in solvents[1:]:
        try:
            lam, _ = tddft_lambda(atoms, charge, SOLVENTS[s], xc, basis, nstates)
            lam_by_solvent[s] = round(lam, 1) if lam else None
        except Exception as e:
            lam_by_solvent[s] = None
            print(f"  [{smiles[:30]}] {s} TD-DFT failed: {type(e).__name__}")

    # solvatochromic slope: lambda_max vs Onsager polarity (nm per unit f(eps))
    pts = [(onsager(SOLVENTS[s]), lam_by_solvent[s]) for s in solvents
           if lam_by_solvent.get(s) is not None]
    slope = shift = None
    if len(pts) >= 2:
        xs, ys = zip(*pts)
        slope = round(float(np.polyfit(xs, ys, 1)[0]), 1)          # nm / f(eps)
        shift = round(ys[-1] - ys[0], 1)                            # polar - nonpolar (nm)

    # closed-form visible check (should be colourless)
    closed = dc.open_to_closed(smiles)
    closed_vis = None
    if closed is not None and not quick:
        cgeo = embed(closed)
        if cgeo:
            try:
                # reference the most-polar of the chosen solvents (not hard-coded
                # water) so custom solvent sets don't KeyError and stay consistent.
                _polar = max(solvents, key=lambda s: SOLVENTS[s])
                clam, cexc = tddft_lambda(cgeo[0], cgeo[1], SOLVENTS[_polar], xc, basis, nstates)
                closed_vis = any(lam > 450 and f > 0.05 for lam, f in cexc)
            except Exception:
                pass

    return {
        "smiles": smiles, "screen_lambda_nm": round(lam_scr, 1) if lam_scr else None,
        "lambda_by_solvent": lam_by_solvent,
        "solvatochromic_slope_nm_per_f": slope, "polar_minus_nonpolar_nm": shift,
        "closed_absorbs_visible": closed_vis,
    }


def cluster_representatives(smiles_list, cutoff=0.4, n_reps=None):
    """Diverse representatives of a RANK-ORDERED candidate list (best first).

    Two deliberate choices vs vanilla Butina: (1) the representative of each
    cluster is its BEST-RANKED member (lowest input index), not the fingerprint
    centroid -- so we DFT the strongest example of each scaffold family, which is
    the 'variant that works' you asked about; (2) clusters are returned in
    best-rank order and truncated to n_reps, so a small DFT budget goes to the
    best distinct scaffolds rather than near-duplicates. Passing an already
    top-N-filtered list (not all 58k) also makes the O(N^2) clustering tractable.
    """
    from rdkit.ML.Cluster import Butina
    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles_list]
    mols = [(s, m) for s, m in mols if m is not None]
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) for _, m in mols]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    rep_idx = sorted(min(c) for c in clusters)          # best-ranked member / cluster
    reps = [mols[i][0] for i in rep_idx]
    return reps[:n_reps] if n_reps else reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="candidate CSV (needs a SMILES column)")
    ap.add_argument("--smiles-col", default=None)
    ap.add_argument("--n-reps", type=int, default=8, help="max cluster representatives")
    ap.add_argument("--cutoff", type=float, default=0.4)
    ap.add_argument("--quick", action="store_true", help="2 solvents, fewer states")
    ap.add_argument("--dft-opt", action="store_true", help="DFT geometry optimisation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    col = args.smiles_col or ("SMILES" if "SMILES" in df.columns else "smiles")
    smis = df[col].dropna().astype(str).tolist()
    reps = cluster_representatives(smis, args.cutoff, args.n_reps)
    print(f"{len(smis)} candidates -> {len(reps)} cluster representatives for DFT")

    results = []
    for i, smi in enumerate(reps):
        print(f"[{i+1}/{len(reps)}] {smi[:55]}")
        r = run_candidate(smi, quick=args.quick, dft_opt=args.dft_opt)
        results.append(r)
        print(f"    lambda_max={r.get('lambda_by_solvent')}  slope={r.get('solvatochromic_slope_nm_per_f')}"
              f"  shift={r.get('polar_minus_nonpolar_nm')} nm  closed_vis={r.get('closed_absorbs_visible')}")

    out = args.out or os.path.splitext(args.csv)[0] + "_dft.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDFT results -> {out}")
    # rank by proximity to the switchable window (moderate negative shift, visible open)
    ranked = [r for r in results if r.get("polar_minus_nonpolar_nm") is not None]
    ranked.sort(key=lambda r: abs((r["polar_minus_nonpolar_nm"]) - (-20)))
    print("\nClosest to the switchable window (target shift ~ -20 nm):")
    for r in ranked[:5]:
        print(f"  shift={r['polar_minus_nonpolar_nm']:>6} nm  {r['smiles'][:55]}")


if __name__ == "__main__":
    main()
