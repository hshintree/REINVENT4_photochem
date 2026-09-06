"""Is BLA predictable from TOPOLOGY alone? If yes, the in-loop surrogate is nearly
free and the large xTB campaign never needs to happen.

WHY THE SPLIT IS THE WHOLE EXPERIMENT. The corpus is COMBINATORIAL -- donors x
acceptors x backbones. A random train/test split puts the same donor on both sides,
so the model only has to memorise "this donor contributes X, this acceptor
contributes Y" and will score near-perfectly while learning nothing transferable.
That is precisely the situation the RL generator creates: it invents donors the
training set never contained.

So the honest tests are LEAVE-ONE-DONOR-OUT and LEAVE-ONE-ACCEPTOR-OUT. Random CV is
reported alongside only to show the size of the illusion.

sklearn is unusable in this env (numpy 2.x ABI), so ridge regression is closed-form
numpy here.
"""
from __future__ import annotations
import csv, json, os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc
from xtb_cache import cache, mol_with_cached_geometry
from calibrate_proxies import bond_lengths

SOLV = "toluene"


def load():
    c = cache()
    rows = []
    for ln in open(os.path.join(HERE, "data", "aqueous_corpus.smi")):
        if not ln.strip() or ln.startswith("#"):
            continue
        smi, label = ln.split()[0], ln.split()[1]
        r = c.get(c.key(Chem.CanonSmiles(smi), SOLV, "planar120"))
        if r is None or r["status"] != "ok":
            continue
        mol = Chem.MolFromSmiles(smi)
        idx = dc._core_idx(mol)
        if idx is None:
            continue
        b = bond_lengths(r["xyz"], idx)
        _, donor, acceptor, backbone = label.split("_", 3) + [""] * (4 - len(label.split("_", 3)))
        rows.append({"smiles": smi, "label": label, "donor": donor,
                     "acceptor": acceptor, "backbone": backbone,
                     "BLA": b["BLA"], "d_mid": b["d_mid"]})
    return rows


def featurise(rows):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    X = []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        fp = np.array(gen.GetCountFingerprintAsNumPy(m), dtype=float)
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m), Descriptors.FractionCSP3(m),
                m.GetNumHeavyAtoms()]
        X.append(np.concatenate([fp, desc]))
    X = np.array(X)
    keep = X.std(0) > 0                      # drop constant columns
    return X[:, keep]


def ridge_fit(X, y, lam=1.0):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, X1.T @ y)


def ridge_pred(w, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ w


def cv(X, y, groups, lam=1.0):
    pred = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if tr.sum() < 5:
            pred[te] = y[tr].mean() if tr.sum() else y.mean(); continue
        pred[te] = ridge_pred(ridge_fit(X[tr], y[tr], lam), X[te])
    ss = ((y - pred) ** 2).sum(); st = ((y - y.mean()) ** 2).sum()
    return 1 - ss / st, np.abs(pred - y).mean(), pred


def main():
    rows = load()
    y = np.array([r["BLA"] for r in rows])
    X = featurise(rows)
    print(f"n = {len(rows)} corpus molecules with a cached xTB geometry")
    print(f"features: {X.shape[1]} (Morgan counts r=2 1024 bit + 8 descriptors, "
          f"constant columns dropped)")
    print(f"BLA: mean {y.mean():+.4f}  sd {y.std():.4f}  range {y.min():+.4f} to {y.max():+.4f}")
    print(f"     (literature seed BLA range for scale: about 0.035 wide)\n")
    for name, col in (("donor", "donor"), ("acceptor", "acceptor"), ("backbone", "backbone")):
        u = sorted({r[col] for r in rows})
        print(f"  {name+'s':11s} {len(u):3d}  {', '.join(u[:6])}{' ...' if len(u) > 6 else ''}")

    rng = np.random.default_rng(0)
    random_groups = rng.integers(0, 5, len(y))
    tests = [("RANDOM 5-fold (the illusion)", random_groups),
             ("leave-one-DONOR-out", np.array([r["donor"] for r in rows])),
             ("leave-one-ACCEPTOR-out", np.array([r["acceptor"] for r in rows])),
             ("leave-one-BACKBONE-out", np.array([r["backbone"] for r in rows]))]
    print(f"\n{'split':32s} {'groups':>7s} {'R2':>8s} {'MAE':>9s}  verdict")
    for name, g in tests:
        r2, mae, _ = cv(X, y, g)
        verdict = ("surrogate is viable" if r2 > 0.9 else
                   "usable, needs more data" if r2 > 0.7 else
                   "does NOT generalise")
        print(f"{name:32s} {len(np.unique(g)):7d} {r2:8.3f} {mae:9.5f}  {verdict}")
    print(f"\n  MAE to beat: predicting the mean gives MAE = "
          f"{np.abs(y - y.mean()).mean():.5f}")
    json.dump(rows, open(os.path.join(HERE, "data", "corpus_bla_dataset.json"), "w"),
              indent=2, default=float)
    print(f"  wrote {os.path.join(HERE,'data','corpus_bla_dataset.json')}")


if __name__ == "__main__":
    main()
