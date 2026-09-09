"""Complete the closed manifolds for the acceptor-bulk ladder, then report ddE.

WHY. The three ladder molecules each have 4 UNIQUE closed states (8 enumerated
diastereomers = 4 enantiomeric pairs; enantiomers are degenerate). Only 2-3 were
computed, and asymmetrically between the two sides of the comparison -- 2 of 4 for
3-methyl against 3 of 4 for 3-pentyl. Adding a state can only lower a Boltzmann sum,
so an asymmetric manifold leaves the sign of the residual bias undetermined. That is
the first thing a referee would press on, and it costs ~4 single points to remove.

MIRROR-AWARE LOOKUP. Enantiomers are the same physical state, so a cached energy for
either member serves. Where both exist we take the LOWER: they should be exactly
degenerate, so any gap is conformer-search error and the lower geometry is the
better-converged one. This is also why nothing already computed is wasted when the
enumeration switches to the deduplicated set.
"""
from __future__ import annotations
import os, sys, signal, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import cache, cached_optimise
from dft_molecule import sp, METHOD, SOLV, H2KCAL, RT

MOLS = {
 "3-methyl":    "CC1=NN(C(=O)\\C1=C/C(O)=C/C=C/N(CCCC=C)c2ccccc2)c3ccccc3",
 "3-isopropyl": "CC(C)C1=NN(C(=O)\\C1=C/C(O)=C/C=C/N(CCCC=C)c2ccccc2)c3ccccc3",
 "3-pentyl":    "CCC(CC)C1=NN(C(=O)\\C1=C/C(O)=C/C=C/N(CCCC=C)c2ccccc2)c3ccccc3",
}
_STOP = False


def _sig(_s, _f):
    global _STOP
    _STOP = True
    print("\n  interrupt: finishing this point then stopping (re-run to resume)", flush=True)


def mirror(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(m)


def unique_states(smi):
    """One representative per enantiomeric pair, deterministic."""
    iso = [d["smiles"] for d in dc.closed_stereoisomers(smi, form="keto")]
    seen, keep = set(), []
    for s in sorted(iso):
        if s in seen:
            continue
        seen.add(s); seen.add(mirror(s))
        keep.append(s)
    return keep, len(iso)


def energy_of_state(s, compute=False):
    """DFT energy of a physical state: the lower of the structure and its mirror."""
    c = cache()
    vals = []
    for cand in (s, mirror(s)):
        q = c.get(c.key(Chem.CanonSmiles(cand), SOLV, "sp2-lowest20", METHOD))
        if q and q["status"] == "ok":
            vals.append(q["energy"])
    if vals:
        return min(vals), False
    if not compute:
        return None, True
    cached_optimise(s, SOLV, "lowest20")          # geometry first
    r = sp(s, "lowest20")
    return (r["energy"], True) if r["status"] == "ok" else (None, True)


def main():
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    c = cache()
    plan = {}
    print("PLAN\n")
    for lbl, smi in MOLS.items():
        can = Chem.CanonSmiles(smi)
        keep, nenum = unique_states(can)
        have = [s for s in keep if energy_of_state(s)[0] is not None]
        need = [s for s in keep if energy_of_state(s)[0] is None]
        plan[lbl] = (can, keep, need)
        print(f"  {lbl:12s} {nenum} enumerated -> {len(keep)} unique | "
              f"have {len(have)}, need {len(need)}")
    total = sum(len(v[2]) for v in plan.values())
    print(f"\n  {total} single point(s) to compute, ~30 min each\n")

    t0 = time.time()
    for lbl, (can, keep, need) in plan.items():
        for s in need:
            if _STOP:
                print("  stopped early"); break
            e, _ = energy_of_state(s, compute=True)
            print(f"  {lbl:12s} +1 state  {'ok' if e else 'FAILED'}  "
                  f"(elapsed {(time.time()-t0)/60:.0f} min)", flush=True)

    print("\n" + "=" * 68)
    print("COMPLETE-MANIFOLD RESULTS (toluene, kcal/mol, keto only)\n")
    res = {}
    for lbl, (can, keep, _need) in plan.items():
        op = c.get(c.key(can, SOLV, "sp2-planar120", METHOD))
        if not op or op["status"] != "ok":
            print(f"  {lbl:12s} open form missing"); continue
        es = [energy_of_state(s)[0] for s in keep]
        got = [e * H2KCAL for e in es if e is not None]
        if not got:
            continue
        a = np.array(got); m = a.min()
        dE = m - RT * np.log(np.exp(-(a - m) / RT).sum()) - op["energy"] * H2KCAL
        res[lbl] = dE
        tag = "COMPLETE" if len(got) == len(keep) else f"** {len(got)}/{len(keep)} — INCOMPLETE **"
        print(f"  {lbl:12s} dE = {dE:+7.2f}   over {len(got)} of {len(keep)} unique "
              f"states   spread {a.max()-m:5.2f}   {tag}")
    if "3-methyl" in res and "3-pentyl" in res:
        print(f"\n  ddE(pentyl - methyl)    = {res['3-pentyl']-res['3-methyl']:+.2f} kcal/mol")
    if "3-isopropyl" in res and "3-methyl" in res:
        print(f"  ddE(isopropyl - methyl) = {res['3-isopropyl']-res['3-methyl']:+.2f}")
        print(f"\n  monotonic Me < iPr < pentyl? "
              f"{res['3-methyl'] < res.get('3-isopropyl', 9e9) < res['3-pentyl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
