"""Cheap Peterson-style A -> C energy diagram for ONE molecule.

RECIPE, and what each choice costs. Peterson (Chem. Commun. 2022, 58, 2303; Chem.
Sci. 2023, 14, 13025) used M06-2X/6-31+G(d,p)/SMD. We keep the FUNCTIONAL, because
that is where DASA-specific validation exists, and economise on the parts where the
error is smaller and more systematic:

    M06-2X          same as Peterson                        (keep)
    6-31G(d)        drop diffuse functions, 545 -> ~390 bf  (~2x cheaper)
    grid level 2    instead of 3                            (~1.3x)
    density fitting                                          (~3.4x)
    C-PCM(toluene)  this pyscf build has no SMD             (deviation)
    // GFN2-xTB     single points on xTB geometries, no DFT optimisation

Measured on this hardware: a full DFT geometry optimisation is 5-12 h PER SPECIES,
so optimisation is off the table locally. A single point at the recipe above is
~10-15 min for a 35-heavy-atom DASA.

HOW MANY CLOSED ISOMERS. The closed cyclopentenone has several diastereomers. We do
NOT compute them all: measured 2026-09-06 on D1-H, the four-isomer DFT spread was
only 1.50 kcal/mol and the Boltzmann sum sat 0.44 below the lowest, so a handful is
enough for a diagram. But we also measured that **xTB ranks these isomers
ANTI-correlated with DFT (rho = -0.60)**, so picking "the N lowest by xTB" is not
safe. Instead we drop only the isomers xTB places far above the low-lying cluster
(a gap that large survives any reranking) and then SPAN the remaining cluster.
The DFT spread across the chosen isomers is reported as the uncertainty on the
closed level -- it is an error bar, not a converged ensemble.
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
from xtb_cache import cache, mol_with_cached_geometry, cached_optimise, DEFAULT_DB

H2KCAL, RT, SOLV = 627.5094740631, 0.5925, "toluene"
BASIS, XC, GRID = "6-31g*", "m06-2x", 2
METHOD = f"{XC}/{BASIS}/grid{GRID}//GFN2-xTB+CPCM"
_STOP = False


def _sig(_s, _f):
    global _STOP
    _STOP = True
    print("\n  interrupt: finishing this single point then stopping (re-run to resume)",
          flush=True)


def sp(smiles, recipe):
    """M06-2X single point on the cached xTB geometry. Cached, resumable."""
    c = cache(DEFAULT_DB)
    canon = Chem.CanonSmiles(smiles)
    key = c.key(canon, SOLV, f"sp2-{recipe}", METHOD)
    hit = c.get(key)
    if hit is not None:
        return hit
    geom = c.get(c.key(canon, SOLV, recipe))
    if geom is None or geom["status"] != "ok":
        return {"status": "fail", "error": f"no xTB geometry ({recipe})"}
    t0 = time.time()
    try:
        from pyscf import gto, dft
        m = mol_with_cached_geometry(canon, geom)
        xyz = np.asarray(geom["xyz"])
        atoms = [(a.GetSymbol(), tuple(float(v) for v in xyz[a.GetIdx()]))
                 for a in m.GetAtoms()]
        mol = gto.M(atom=atoms, basis=BASIS, verbose=0)
        mf = dft.RKS(mol, xc=XC).density_fit()
        mf.grids.level = GRID
        try:
            mf = mf.PCM(); mf.with_solvent.method = "C-PCM"; mf.with_solvent.eps = 2.374
        except Exception:
            pass
        mf.conv_tol, mf.max_cycle = 1e-8, 100
        e = mf.kernel()
        rec = {"status": "ok" if mf.converged else "fail", "energy": float(e),
               "converged": bool(mf.converged), "nao": int(mol.nao),
               "error": None if mf.converged else "SCF not converged"}
    except Exception as exc:
        rec = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    rec["seconds"] = time.time() - t0
    c.put(key, canon, SOLV, f"sp2-{recipe}", rec, method=METHOD)
    return rec


def pick(smiles, n_closed, gap):
    """xTB-rank the keto diastereomers, drop those far above the low cluster, then
    SPAN what remains (xTB ordering within the cluster is not trustworthy)."""
    # The OPEN form needs an xTB geometry too. The first version only generated
    # geometries for the closed isomers, so a molecule whose open form had not been
    # prepped by hand failed its open-form single point instantly ("no xTB geometry")
    # -- and without the open form there is no dE at all, making the closed points
    # useless. Ensure it here.
    cached_optimise(smiles, SOLV, "planar120")
    iso = dc.closed_stereoisomers(smiles, form="keto")
    es, failed = [], []
    for d in iso:
        r = cached_optimise(d["smiles"], SOLV, "lowest20")
        if r["status"] == "ok":
            es.append((r["energy"], d["smiles"]))
        else:
            failed.append((d["smiles"], r.get("error")))
    if failed:
        # SILENT TRUNCATION was the bug here. A failed xTB geometry used to be
        # skipped with no counter and no message, which is worse than it sounds:
        # the cluster minimum `emin` is taken over the SURVIVORS, so if the true
        # lowest isomer failed, the 2.5 kcal/mol cutoff is measured from the wrong
        # zero AND the manifold is short a low-energy state, making dE too
        # positive. Failures are also CACHED, so a re-run reproduces the same
        # truncation forever without retrying. Same bug class as the one fixed in
        # campaign.py's report on 2026-09-06 -- worth grepping for elsewhere.
        print(f"  ** {len(failed)} of {len(iso)} closed isomers FAILED xTB — the "
              f"manifold is TRUNCATED and dE will be biased HIGH **")
        for sm, err in failed:
            print(f"     {str(err)[:90]}")
            print(f"     {sm}")
        print(f"     re-run with `campaign.py --retry-failed` to clear cached failures")
    es.sort()
    emin = es[0][0]
    cluster = [(e, s) for e, s in es if (e - emin) * H2KCAL <= gap]
    if len(cluster) <= n_closed:
        chosen = cluster
    else:
        idx = np.linspace(0, len(cluster) - 1, n_closed).round().astype(int)
        chosen = [cluster[i] for i in idx]
    return iso, es, cluster, chosen, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--label", default="molecule")
    ap.add_argument("--n-closed", type=int, default=3)
    ap.add_argument("--gap", type=float, default=2.5,
                    help="kcal/mol above the xTB minimum to still consider")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    smi = Chem.CanonSmiles(a.smiles)
    mol = Chem.MolFromSmiles(smi)
    if not dc.is_dasa(mol) or dc.is_legacy_core(mol):
        print("not a corrected-core DASA"); return 1
    iso, es, cluster, chosen, failed = pick(smi, a.n_closed, a.gap)
    print(f"{a.label}\n  {smi}")
    print(f"  {mol.GetNumHeavyAtoms()} heavy atoms | {METHOD}")
    print(f"  {len(iso)} keto diastereomers; {len(cluster)} within {a.gap} kcal/mol of the "
          f"xTB minimum; computing {len(chosen)} spanning that cluster + the open form")
    units = [("open", smi, "planar120")] + [(f"closed {i+1}", s, "lowest20")
                                            for i, (_e, s) in enumerate(chosen)]
    if a.status:
        c = cache()
        for role, s, rec in units:
            hit = c.get(c.key(Chem.CanonSmiles(s), SOLV, f"sp2-{rec}", METHOD))
            print(f"    {role:12s} {'done' if hit else 'pending'}")
        return 0

    t0 = time.time(); out = {}
    for role, s, rec in units:
        if _STOP:
            print("  stopped early"); break
        r = sp(s, rec)
        out[role] = r
        print(f"    {role:12s} {r['status']:5s} {r.get('seconds',0)/60:5.1f} min"
              f"  (elapsed {(time.time()-t0)/60:4.0f} min)"
              + (f"  {r.get('error')}" if r["status"] != "ok" else ""), flush=True)

    op = out.get("open")
    closed_all = [(k, r) for k, r in out.items() if k.startswith("closed")]
    cl = [r["energy"] for _k, r in closed_all if r["status"] == "ok"]
    dft_failed = [k for k, r in closed_all if r["status"] != "ok"]
    if dft_failed:
        # SECOND silent-drop site in this file. pick() dropping failed xTB
        # GEOMETRIES was fixed first; this one drops failed DFT SINGLE POINTS the
        # same way, and it is the more expensive failure -- 45 minutes of compute
        # vanishing from the Boltzmann sum with nothing in the diagram to say so.
        # Finding one instance of a pattern is not the same as fixing it.
        print(f"\n  ** {len(dft_failed)} DFT single point(s) FAILED: "
              f"{', '.join(dft_failed)} — the manifold below is truncated and dE is "
              f"biased HIGH. Do not compare this number to a complete one. **")
    if op and op["status"] == "ok" and cl:
        e = np.array(cl) * H2KCAL; emin = e.min()
        dE_low = emin - op["energy"] * H2KCAL
        dE_ens = emin - RT * np.log(np.exp(-(e - emin) / RT).sum()) - op["energy"] * H2KCAL
        print(f"\n  ENERGY DIAGRAM ({SOLV}, kcal/mol relative to the open form)")
        if failed:
            print(f"\n  ** DEGRADED: {len(failed)} of {len(iso)} closed isomers failed "
                  f"xTB; the manifold below is truncated and dE is biased HIGH **")
        print(f"    A  open                        0.00")
        print(f"    C  closed, lowest isomer   {dE_low:+7.2f}")
        print(f"    C  closed, Boltzmann       {dE_ens:+7.2f}   over {len(cl)} of "
              f"{len(closed_all)} attempted isomers"
              + ("  ** TRUNCATED **" if dft_failed else ""))
        print(f"       diastereomer spread      {e.max()-emin:6.2f}   <- uncertainty, not noise")
        print(f"\n    negative = closed favoured. Peterson's D1-H reference at the full")
        print(f"    6-31+G(d,p) recipe was -6.48 kcal/mol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
