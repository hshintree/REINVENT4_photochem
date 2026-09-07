"""Can a topology model reproduce an xTB-derived descriptor across DONOR chemistry?

The 224-molecule corpus answered "not across donor CLASSES" (R2 -0.349) but it only
had four classes, so holding one out removed up to 162 of 224 molecules. The
donor-diverse corpus has 31 individual donors across 8 classes, so both
leave-one-DONOR-out and leave-one-CLASS-out are now meaningful.

WHAT THIS DOES AND DOES NOT TELL US. It is a question about COMPRESSIBILITY -- can a
cheap model stand in for a 17 s calculation -- not about whether the descriptor means
anything physically. As of 2026-09-06 BLA has NO validated physical meaning: it does
not predict the measured solvatochromic slope (rho +0.18, n=9, and no other computed
descriptor does either). The learnability result transfers to whatever descriptor we
end up trusting; it is not an endorsement of BLA.
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import cache
from calibrate_proxies import bond_lengths


def load():
    c = cache(); rows = []
    for ln in open(os.path.join(HERE, "data", "donor_corpus.smi")):
        if not ln.strip() or ln.startswith("#"):
            continue
        smi, lab = ln.split()[0], ln.split()[1]
        donor, dclass, acc, bb, basis = lab.split("|")
        q = c.get(c.key(Chem.CanonSmiles(smi), "toluene", "planar120"))
        if q is None or q["status"] != "ok":
            continue
        idx = dc._core_idx(Chem.MolFromSmiles(smi))
        if idx is None:
            continue
        rows.append({"smiles": smi, "donor": donor, "dclass": dclass, "acceptor": acc,
                     "backbone": bb, "basis": basis,
                     **{k: v for k, v in bond_lengths(q["xyz"], idx).items()
                        if k in ("BLA", "d_mid")}})
    return rows


def feats(rows):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    X = []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        X.append(np.concatenate([
            np.array(gen.GetCountFingerprintAsNumPy(m), dtype=float),
            [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
             Descriptors.NumRotatableBonds(m), Descriptors.FractionCSP3(m),
             m.GetNumHeavyAtoms()]]))
    X = np.array(X)
    return X[:, X.std(0) > 0]


def ridge(X, y, lam=1.0):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, X1.T @ y)


def cvr(X, y, g, lam=1.0):
    pred = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        if tr.sum() < 5:
            pred[te] = y[tr].mean(); continue
        w = ridge(X[tr], y[tr], lam)
        pred[te] = np.hstack([np.ones((te.sum(), 1)), X[te]]) @ w
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.abs(pred - y).mean()


def main():
    rows = load(); y = np.array([r["BLA"] for r in rows]); X = feats(rows)
    from collections import Counter
    print(f"\nn = {len(rows)} of 240 donor-corpus molecules have a cached geometry")
    print(f"  donors {len(set(r['donor'] for r in rows))}, "
          f"classes {len(set(r['dclass'] for r in rows))}, "
          f"acceptors {len(set(r['acceptor'] for r in rows))}")
    print(f"  basis: {dict(Counter(r['basis'] for r in rows))}")
    print(f"  BLA mean {y.mean():+.4f} sd {y.std():.4f} "
          f"range {y.min():+.4f}..{y.max():+.4f}")
    rng = np.random.default_rng(0)
    tests = [("RANDOM 5-fold (the illusion)", rng.integers(0, 5, len(y))),
             ("leave-one-DONOR-out (31)", np.array([r["donor"] for r in rows])),
             ("leave-one-DONOR-CLASS-out (8)", np.array([r["dclass"] for r in rows])),
             ("leave-one-ACCEPTOR-out (4)", np.array([r["acceptor"] for r in rows])),
             ("leave-one-BACKBONE-out (2)", np.array([r["backbone"] for r in rows]))]
    print(f"\n{'split':34s} {'R2':>8s} {'MAE':>9s}  verdict")
    for name, g in tests:
        r2, mae = cvr(X, y, g)
        v = ("surrogate viable" if r2 > 0.9 else
             "usable" if r2 > 0.7 else "does NOT generalise")
        print(f"{name:34s} {r2:8.3f} {mae:9.5f}  {v}")
    print(f"\n  predicting the mean gives MAE {np.abs(y-y.mean()).mean():.5f}")
    json.dump(rows, open(os.path.join(HERE, "data", "donor_bla_dataset.json"), "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
