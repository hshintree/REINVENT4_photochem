"""Resumable local campaign driver: closed-manifold energies + descriptors.

Resumption is not a mode you invoke, it is the default. Every unit of work (one
xTB optimisation) is looked up in notebooks/data/xtb_cache.sqlite first, so
re-running the same command after any interruption simply skips what is done.

    python notebooks/campaign.py --status                 # what is left, and how long
    python notebooks/campaign.py --budget 1800            # chip away for 30 minutes
    python notebooks/campaign.py                          # run to completion
    python notebooks/campaign.py --report                 # assemble results, no new QM

Ctrl-C finishes the units already in flight and exits cleanly; nothing is lost
because each optimisation is committed the moment it completes.

--report recomputes every descriptor from CACHED GEOMETRIES, so inventing a new
descriptor costs seconds over the whole campaign instead of re-running it.
"""
from __future__ import annotations
import argparse, csv, json, os, signal, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import (cached_optimise, cache, DEFAULT_DB, mol_with_cached_geometry,
                       XTBCache)
from calibrate_proxies import bond_lengths
from dasa_descriptors import steric_descriptors

OPEN_RECIPE, CLOSED_RECIPE = "planar120", "lowest20"
OPEN_ONLY = False
H2KCAL, R_KCAL = 627.5094740631, 1.98720425e-3
_STOP = False


def _on_signal(_sig, _frm):
    global _STOP
    if _STOP:
        print("\n  second interrupt - exiting now", flush=True); sys.exit(130)
    _STOP = True
    print("\n  interrupt received: finishing units in flight, then stopping "
          "(nothing is lost; re-run to resume)", flush=True)


def units_for(smiles: str, solvent: str):
    """Every (smiles, solvent, recipe, role) this molecule needs.

    Only the tautomers that EXIST in this solvent: aprotic media stop at the
    neutral keto form (Angew. Chem. Int. Ed. 2018, 57, 8063). Including the
    zwitterion in an aprotic run is what wrecked the toluene campaign.
    """
    out = [(smiles, solvent, OPEN_RECIPE, "open")]
    if OPEN_ONLY:
        return out
    forms = dc.closed_forms_for_solvent(solvent)
    for d in dc.closed_stereoisomers(smiles, form=forms):
        out.append((d["smiles"], solvent, CLOSED_RECIPE, f"closed:{d['form']}"))
    return out


# The seed CSV records the solvent a measurement was made in, using chemists'
# names. xtb's get_solvent returns None for "CDCl3" and "MeOH" -- and the old
# code then ran GAS PHASE without saying so. Map explicitly, and refuse anything
# unmapped rather than guessing.
SOLVENT_MAP = {
    "CDCl3": "chloroform", "CHCl3": "chloroform", "chloroform": "chloroform",
    "MeOH": "methanol", "methanol": "methanol",
    "CH2Cl2": "ch2cl2", "DCM": "ch2cl2", "dichloromethane": "ch2cl2",
    "toluene": "toluene", "water": "water", "THF": "thf",
    "THF:H2O 60:40": "water", "THF:H2O 10:90": "water",   # approximated as water
}


def load_molecules(path: str | None, solvent: str, from_seed: bool = False):
    """Returns (label, smiles, solvent) triples.

    With from_seed, each molecule is run in the solvent its EQUILIBRIUM was
    measured in, and molecules without a measured equilibrium are dropped -- the
    point of that mode is a like-for-like comparison, and a molecule with nothing
    to compare against only costs compute.
    """
    if path and path.endswith(".smi"):
        out = []
        for i, ln in enumerate(open(path)):
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.split()
            out.append((parts[1] if len(parts) > 1 else f"m{i}", parts[0], solvent))
        return out
    seed = path or os.path.join(HERE, "data", "dasa_literature_seed.csv")
    rows = list(csv.DictReader(open(seed)))
    if not from_seed:
        return [(r["compound_id"], r["smiles_open"], solvent) for r in rows]
    out = []
    for r in rows:
        if not r["pct_open_equilibrium"]:
            continue
        raw = r["equilibrium_solvent"]
        if raw not in SOLVENT_MAP:
            raise ValueError(f"{r['compound_id']}: unmapped solvent {raw!r}")
        out.append((r["compound_id"], r["smiles_open"], SOLVENT_MAP[raw]))
    return out


def plan(mols):
    plan_, seen = [], set()
    for label, smi, solvent in mols:
        for u in units_for(smi, solvent):
            if u[:3] in seen:
                continue
            seen.add(u[:3])
            plan_.append((label, *u))
    return plan_


def cmd_status(plan_, db):
    c = cache(db)
    pend = [p for p in plan_ if c.get(c.key(Chem.CanonSmiles(p[1]), p[2], p[3])) is None]
    st = c.stats()
    done = len(plan_) - len(pend)
    avg = (st["seconds_saved"] / st["total"]) if st["total"] else 8.0
    print(f"  units total     {len(plan_)}")
    print(f"  cached          {done}  ({done/max(len(plan_),1):.0%})")
    print(f"  pending         {len(pend)}")
    print(f"  cache holds     {st['ok']} ok / {st['fail']} failed, "
          f"{st['seconds_saved']/3600:.2f} compute-hours banked")
    print(f"  mean unit       {avg:.1f} s")
    for w in (1, 6):
        print(f"  est. remaining  {len(pend)*avg/w/3600:6.2f} h at {w} worker(s)")
    return pend


def cmd_run(plan_, db, workers, budget, force=False):
    c = cache(db)
    pend = plan_ if force else [p for p in plan_ if c.get(c.key(Chem.CanonSmiles(p[1]), p[2], p[3])) is None]
    if not pend:
        print("  nothing pending - campaign complete"); return 0
    print(f"  {len(pend)} pending units, {workers} workers"
          + (f", budget {budget/60:.0f} min" if budget else ""))
    t0 = time.time()
    done = {"n": 0, "skipped": 0}

    def work(p):
        _lab, smi, solv, recipe, _role = p
        if _STOP or (budget and time.time() - t0 > budget):
            done["skipped"] += 1
            return
        r = cached_optimise(smi, solv, recipe, db=db, force=force)
        done["n"] += 1
        if done["n"] % 10 == 0 or done["n"] == 1:
            el = time.time() - t0
            rate = done["n"] / max(el, 1e-9)
            left = (len(pend) - done["n"] - done["skipped"]) / max(rate, 1e-9)
            print(f"    {done['n']}/{len(pend)}  {el/60:5.1f} min elapsed, "
                  f"~{left/60:5.1f} min left  [{r['status']}]", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, pend))
    el = time.time() - t0
    print(f"  computed {done['n']} units in {el/60:.1f} min; "
          f"{done['skipped']} left for the next run"
          if done["skipped"] else f"  computed {done['n']} units in {el/60:.1f} min")
    return done["skipped"]


def cmd_report(mols, db, out_path):
    """Assemble everything from cached geometries. No new quantum chemistry."""
    c, rows, T = cache(db), [], 298.15
    for label, smi, solvent in mols:
        oc = c.get(c.key(Chem.CanonSmiles(smi), solvent, OPEN_RECIPE))
        if oc is None or oc["status"] != "ok":
            rows.append({"label": label, "smiles": smi, "status": "open form pending"})
            continue
        row = {"label": label, "smiles": smi, "solvent": solvent, "status": "ok",
               "E_open_hartree": oc["energy"], "dipole": oc["dipole"]}
        m3d = mol_with_cached_geometry(smi, oc)
        idx = dc._core_idx(Chem.MolFromSmiles(smi))
        if m3d is not None and idx is not None:
            row.update(bond_lengths(oc["xyz"], idx))
            row.update(steric_descriptors(m3d, oc["xyz"], idx))
        if OPEN_ONLY:
            rows.append(row); continue
        es, missing, failed = [], 0, 0
        forms = dc.closed_forms_for_solvent(solvent)
        for d in dc.closed_stereoisomers(smi, form=forms):
            r = c.get(c.key(Chem.CanonSmiles(d["smiles"]), solvent, CLOSED_RECIPE))
            if r is None:
                missing += 1
            elif r["status"] == "ok":
                es.append((d["form"], r["energy"] * H2KCAL))
            else:
                failed += 1          # NEVER silently drop: a truncated manifold
                                     # is a degraded result, not a clean one
        row["n_closed_missing"] = missing
        row["n_closed_failed"] = failed
        row["closed_forms_used"] = forms
        if es:
            e = np.array([x[1] for x in es]); emin = float(e.min())
            z = float(np.exp(-(e - emin) / (R_KCAL * T)).sum())
            row["dE_ensemble"] = emin - R_KCAL * T * np.log(z) - oc["energy"] * H2KCAL
            row["dE_lowest"] = emin - oc["energy"] * H2KCAL
            row["closed_spread_kcal"] = float(e.max() - emin)
            row["n_closed"] = len(es)
            if failed:
                row["status"] = (f"DEGRADED ({failed} closed isomers failed; "
                                 f"ensemble is over {len(es)} of "
                                 f"{len(es) + failed + missing})")
            elif missing:
                row["status"] = f"partial ({missing} closed isomers pending)"
        else:
            row["status"] = "closed manifold pending"
        rows.append(row)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"  {len(ok)}/{len(rows)} molecules complete -> {out_path}")
    if ok and OPEN_ONLY:
        # --open-only has no closed manifold, so no dE_ensemble. The first version
        # indexed it unconditionally and crashed AFTER writing the results -- the
        # data was safe but the run reported failure, which is the worst kind of
        # cosmetic bug because it makes you distrust a good result.
        import statistics as _st
        bl = [r["BLA"] for r in ok if "BLA" in r]
        print(f"\n  open-form descriptors only. BLA over {len(bl)} molecules: "
              f"mean {_st.mean(bl):+.4f}, sd {_st.pstdev(bl):.4f}, "
              f"range {min(bl):+.4f} to {max(bl):+.4f}")
        print(f"  {'molecule':34s} {'BLA':>9s} {'d_mid':>9s} {'vbur_N':>7s}")
        for r in ok[:10]:
            print(f"  {r['label'][:34]:34s} {r.get('BLA', float('nan')):9.4f} "
                  f"{r.get('d_mid', float('nan')):9.4f} {r.get('vbur_N', float('nan')):7.3f}")
    elif ok:
        print(f"\n  {'molecule':10s} {'dE_ens':>8s} {'spread':>8s} {'#iso':>5s} "
              f"{'BLA':>9s} {'vbur_N':>7s}")
        for r in ok[:25]:
            print(f"  {r['label']:10s} {r['dE_ensemble']:8.2f} "
                  f"{r['closed_spread_kcal']:8.2f} {r['n_closed']:5d} "
                  f"{r.get('BLA', float('nan')):9.4f} {r.get('vbur_N', float('nan')):7.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="seed CSV or .smi (default: literature seed)")
    ap.add_argument("--solvent", default="toluene")
    ap.add_argument("--solvent-from-seed", action="store_true",
                    help="run each molecule in the solvent its equilibrium "
                         "was measured in; drops molecules with no measurement")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=0, help="seconds of wall clock, 0 = unlimited")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="delete cached failures so they are recomputed")
    ap.add_argument("--open-only", action="store_true",
                    help="compute only the open form (descriptor dataset); "
                         "1 unit per molecule instead of ~13")
    ap.add_argument("--force", action="store_true",
                    help="recompute and OVERWRITE every unit, ignoring the cache")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "campaign_results.json"))
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    global OPEN_ONLY
    OPEN_ONLY = a.open_only
    mols = load_molecules(a.input, a.solvent, a.solvent_from_seed)
    solv = sorted({m[2] for m in mols})
    print(f"campaign: {len(mols)} molecules, solvent(s)={solv}, db={a.db}")
    if a.retry_failed:
        print(f"  cleared {cache(a.db).clear_failures()} failed entries")
    plan_ = plan(mols)

    if a.status:
        cmd_status(plan_, a.db); return 0
    if a.report:
        cmd_report(mols, a.db, a.out); return 0

    cmd_status(plan_, a.db)
    print()
    left = cmd_run(plan_, a.db, a.workers, a.budget, a.force)
    print()
    cmd_report(mols, a.db, a.out)
    return 0 if not left else 2


if __name__ == "__main__":
    raise SystemExit(main())
