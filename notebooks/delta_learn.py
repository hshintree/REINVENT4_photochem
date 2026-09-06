"""How much model can 13 measured DASAs actually support?

THE TRAP THIS SCRIPT EXISTS TO AVOID. We have 14 measured dark equilibria; 13 have a
matching xTB two-state energy. Nine of them are one acceptor series at a fixed donor
(Hemmer JACS 2018). Fitting a flexible model to that and reporting leave-one-out CV
would measure INTERPOLATION WITHIN ONE SERIES and look excellent. It would tell us
nothing about whether the model helps on a molecule the RL generator invents.

So every model here is scored under LEAVE-ONE-PAPER-OUT: hold out an entire
publication, train on the rest, predict it cold. Because of how the literature is
distributed, that is simultaneously leave-one-solvent-out and
leave-one-architecture-out -- a hard test, and the one that matches how the model
would actually be used.

Models are ordered by number of free parameters, against a do-nothing baseline. The
question is not "which model is best" but "does ANY model beat predicting the
training mean, and does any beat using the xTB number raw". At n=13 the honest answer
may be no, and that is a result.

sklearn is unusable in this env (numpy 2.x ABI split), so the ridge and the GP are
implemented here in numpy/scipy. At this sample size they are a few lines each.
"""
from __future__ import annotations
import csv, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import cache, mol_with_cached_geometry
from calibrate_proxies import bond_lengths
from dasa_descriptors import steric_descriptors
from campaign import SOLVENT_MAP

H2KCAL, RT = 627.5094740631, 0.5925
# D1-H / D1-M were computed in toluene by triene_ladder.py, everything else in the
# solvent its equilibrium was measured in.
EXTRA_SOLVENT = {"D1H": "toluene", "D1M": "toluene"}


def dG(pct):
    q = min(max(float(pct), 1.0), 99.0)
    return -RT * np.log((100 - q) / q)


def assemble():
    c = cache()
    rows = []
    for r in csv.DictReader(open(os.path.join(HERE, "data", "dasa_literature_seed.csv"))):
        if not r["pct_open_equilibrium"]:
            continue
        cid = r["compound_id"]
        sv = EXTRA_SOLVENT.get(cid, SOLVENT_MAP.get(r["equilibrium_solvent"]))
        smi = r["smiles_open"]
        op = c.get(c.key(Chem.CanonSmiles(smi), sv, "planar120"))
        if op is None or op["status"] != "ok":
            continue
        forms = dc.closed_forms_for_solvent(sv)
        iso = dc.closed_stereoisomers(smi, form=forms)
        es, miss = [], 0
        for d in iso:
            q = c.get(c.key(Chem.CanonSmiles(d["smiles"]), sv, "lowest20"))
            if q and q["status"] == "ok":
                es.append(q["energy"] * H2KCAL)
            else:
                miss += 1
        if not es or miss:
            continue                       # never train on a truncated manifold
        e = np.array(es); emin = e.min()
        dE = emin - RT * np.log(np.exp(-(e - emin) / RT).sum()) - op["energy"] * H2KCAL
        m3d = mol_with_cached_geometry(smi, op)
        idx = dc._core_idx(Chem.MolFromSmiles(smi))
        f = {"dE_xtb": dE, "dipole": op["dipole"]}
        f.update({k: v for k, v in bond_lengths(op["xyz"], idx).items()
                  if k in ("BLA", "d_mid")})
        f.update({k: v for k, v in steric_descriptors(m3d, op["xyz"], idx).items()
                  if k in ("vbur_N", "vbur_C5", "vbur_site", "sterimol_B1")})
        try:
            f["delta_pka"] = dc.delta_pka(Chem.MolFromSmiles(smi))
        except Exception:
            f["delta_pka"] = np.nan
        rows.append({"id": cid, "paper": r["source"].split(",")[0],
                     "solvent": sv, "y": dG(r["pct_open_equilibrium"]), **f})
    return rows


# ---------------- models (numpy only) ----------------
def fit_ridge(X, y, alpha=1e-6):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    A = X1.T @ X1 + alpha * np.eye(X1.shape[1]); A[0, 0] -= alpha
    return np.linalg.solve(A, X1.T @ y)


def pred_ridge(w, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ w


def _k(A, B, ls, sf):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2 / ls ** 2).sum(-1)
    return sf ** 2 * np.exp(-0.5 * d2)


def fit_gp(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    ym = y.mean()

    def nll(t):
        ls, sf, sn = np.exp(t[:X.shape[1]]), np.exp(t[-2]), np.exp(t[-1])
        K = _k(Z, Z, ls, sf) + (sn ** 2 + 1e-8) * np.eye(len(Z))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e6
        a = np.linalg.solve(L.T, np.linalg.solve(L, y - ym))
        return 0.5 * (y - ym) @ a + np.log(np.diag(L)).sum()

    t0 = np.concatenate([np.zeros(X.shape[1]), [np.log(y.std() + 1e-6)], [np.log(0.3)]])
    r = minimize(nll, t0, method="L-BFGS-B")
    ls, sf, sn = np.exp(r.x[:X.shape[1]]), np.exp(r.x[-2]), np.exp(r.x[-1])
    K = _k(Z, Z, ls, sf) + (sn ** 2 + 1e-8) * np.eye(len(Z))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, y - ym))
    return dict(mu=mu, sd=sd, ym=ym, Z=Z, ls=ls, sf=sf, L=L, a=a)


def pred_gp(m, X):
    Zs = (X - m["mu"]) / m["sd"]
    Ks = _k(Zs, m["Z"], m["ls"], m["sf"])
    mean = m["ym"] + Ks @ m["a"]
    v = np.linalg.solve(m["L"], Ks.T)
    var = m["sf"] ** 2 - (v ** 2).sum(0)
    return mean, np.sqrt(np.maximum(var, 1e-12))


MODELS = {
    "M0 train mean":              (0, None),
    "M1 raw dE_xtb (no fit)":     (0, ["dE_xtb"]),
    "M2 linear(dE_xtb)":          (2, ["dE_xtb"]),
    "M3 linear(BLA)":             (2, ["BLA"]),
    "M4 linear(dE_xtb, BLA)":     (3, ["dE_xtb", "BLA"]),
    "M5 GP(dE_xtb, BLA)":         (4, ["dE_xtb", "BLA"]),
    "M6 GP(6 features)":          (8, ["dE_xtb", "BLA", "d_mid", "delta_pka",
                                       "vbur_N", "sterimol_B1"]),
}


def main():
    rows = assemble()
    papers = sorted({r["paper"] for r in rows})
    print(f"n = {len(rows)} compounds with BOTH a measured equilibrium and a complete "
          f"solvent-matched xTB manifold\n")
    print(f"{'id':5s} {'paper':28s} {'solv':11s} {'dG_meas':>8s} {'dE_xtb':>8s} {'BLA':>9s}")
    for r in rows:
        print(f"{r['id']:5s} {r['paper'][:28]:28s} {r['solvent']:11s} {r['y']:8.2f} "
              f"{r['dE_xtb']:8.2f} {r['BLA']:9.4f}")
    print(f"\nleave-one-paper-out groups ({len(papers)}): "
          + ", ".join(f"{p.split()[0]}({sum(1 for r in rows if r['paper']==p)})" for p in papers))

    y = np.array([r["y"] for r in rows])
    grp = np.array([r["paper"] for r in rows])
    print(f"\n{'model':26s} {'params':>7s} {'MAE':>7s} {'RMSE':>7s} {'rho':>7s}  vs M0")
    base = None
    for name, (npar, feats) in MODELS.items():
        preds = np.zeros(len(y))
        for p in papers:
            te = grp == p; tr = ~te
            if feats is None:
                preds[te] = y[tr].mean(); continue
            use = [f for f in feats
                   if not np.isnan(np.array([r[f] for r in rows], dtype=float)).any()]
            if not use:
                preds[te] = np.nan; continue
            X = np.array([[r[f] for f in use] for r in rows], dtype=float)
            if name.startswith("M1"):
                preds[te] = X[te, 0]
            elif name.startswith("M5") or name.startswith("M6"):
                m = fit_gp(X[tr], y[tr]); preds[te] = pred_gp(m, X[te])[0]
            else:
                w = fit_ridge(X[tr], y[tr]); preds[te] = pred_ridge(w, X[te])
        ok = ~np.isnan(preds)
        mae = np.abs(preds[ok] - y[ok]).mean()
        rmse = np.sqrt(((preds[ok] - y[ok]) ** 2).mean())
        rho = spearmanr(preds[ok], y[ok])[0]
        if base is None:
            base = mae
        tag = "baseline" if name.startswith("M0") else (
            f"{(base-mae)/base:+.0%} MAE" if mae == mae else "")
        print(f"{name:26s} {npar:7d} {mae:7.2f} {rmse:7.2f} {rho:+7.3f}  {tag}")

    # ---- the contrast that matters: naive CV vs grouped CV ----------------
    print("\n" + "=" * 74)
    print("WHY THE SPLIT MATTERS: same model, same data, two CV protocols")
    print("=" * 74)
    for name in ("M3 linear(BLA)", "M2 linear(dE_xtb)", "M5 GP(dE_xtb, BLA)"):
        feats = MODELS[name][1]
        X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
        # (a) naive leave-ONE-OUT: neighbours from the same series stay in training
        loo = np.zeros(len(y))
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            if name.startswith("M5"):
                loo[i] = pred_gp(fit_gp(X[tr], y[tr]), X[i:i+1])[0][0]
            else:
                loo[i] = pred_ridge(fit_ridge(X[tr], y[tr]), X[i:i+1])[0]
        # (b) honest leave-one-PAPER-out
        lopo = np.zeros(len(y))
        for pp in papers:
            te = grp == pp; tr = ~te
            if name.startswith("M5"):
                lopo[te] = pred_gp(fit_gp(X[tr], y[tr]), X[te])[0]
            else:
                lopo[te] = pred_ridge(fit_ridge(X[tr], y[tr]), X[te])
        print(f"  {name:22s} LOO MAE {np.abs(loo-y).mean():.2f}   "
              f"LOPO MAE {np.abs(lopo-y).mean():.2f}   "
              f"optimism {np.abs(lopo-y).mean()-np.abs(loo-y).mean():+.2f} kcal/mol")
    print("  LOO keeps same-series neighbours in training, so it measures interpolation")
    print("  inside the Hemmer acceptor scan. LOPO is the number to quote.")

    # ---- per-fold, because the groups are very unbalanced -----------------
    print("\nPER-FOLD (leave-one-paper-out), best model M3 linear(BLA):")
    feats = MODELS["M3 linear(BLA)"][1]
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    for pp in papers:
        te = grp == pp; tr = ~te
        w = fit_ridge(X[tr], y[tr]); pr = pred_ridge(w, X[te])
        ids = [rows[i]["id"] for i in np.where(te)[0]]
        print(f"  hold out {pp[:26]:26s} train n={tr.sum():2d} test n={te.sum():2d}  "
              f"MAE {np.abs(pr-y[te]).mean():5.2f}   {', '.join(ids)}")

    print(f"\nSpread of the target itself: sd(dG) = {y.std():.2f} kcal/mol, "
          f"range {y.min():.2f} to {y.max():.2f}")
    print("A model whose LOPO MAE is not clearly below sd(dG) has learned nothing "
          "transferable.")
    json.dump(rows, open(os.path.join(HERE, "data", "delta_learn_dataset.json"), "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
