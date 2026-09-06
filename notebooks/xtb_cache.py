"""Content-addressed, resumable cache for GFN2-xTB geometry optimisations.

WHY: a closed-manifold campaign is ~13 optimisations per molecule (one open form
plus every diastereomer of both tautomers) at ~8 s each. 300 molecules is ~4000
optimisations and ~7 h. On a laptop that run WILL be interrupted -- a lid close, a
kill, a reboot -- and re-running from zero is not acceptable.

THE UNIT OF WORK IS ONE OPTIMISATION, NOT ONE MOLECULE. Cache at that level and
resumption is not a feature you invoke, it is just what happens: the driver
enumerates every unit, looks each one up, and computes only the misses. Killing a
run costs at most one in-flight calculation per worker.

THE ACTUAL CLEVERNESS IS STORING THE GEOMETRY. Energies alone would still force a
full 7 h re-run every time we invent a descriptor. With coordinates in the cache,
BLA / %V_bur / Sterimol / d(C1..C5) / anything else we think of later recompute in
seconds over the whole campaign. Descriptor iteration becomes free; only new
CHEMISTRY costs time.

KEYING. The key is (canonical SMILES, solvent, method, recipe). Canonical, because
we already lost a day to the same molecule written two ways producing different
conformers -- see notebooks/data/seed_proxy_fit.csv and the n_confs=30 wobble. The
`recipe` names the embedding protocol, so changing it opens a new namespace instead
of silently mixing geometries from different protocols.

FAILURES ARE CACHED TOO, with status='fail', so a molecule whose SCF will not
converge is not retried on every restart. `--retry-failed` clears them.

SQLite, stdlib only, WAL mode: one file, atomic commits, safe under a thread pool.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "data", "xtb_cache.sqlite")
METHOD = "GFN2-xTB"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calc (
  key         TEXT PRIMARY KEY,
  smiles      TEXT NOT NULL,
  solvent     TEXT NOT NULL,
  method      TEXT NOT NULL,
  recipe      TEXT NOT NULL,
  status      TEXT NOT NULL,
  energy      REAL,
  dipole      REAL,
  numbers     TEXT,
  xyz         TEXT,
  grad_calls  INTEGER,
  converged   INTEGER,
  seconds     REAL,
  error       TEXT,
  created     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calc_smiles ON calc(smiles);
CREATE INDEX IF NOT EXISTS idx_calc_status ON calc(status);
"""


class XTBCache:
    """One SQLite file. Thread-safe. Every completed unit is committed immediately."""

    def __init__(self, path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @staticmethod
    def key(smiles: str, solvent: str, recipe: str, method: str = METHOD) -> str:
        return f"{method}|{recipe}|{solvent}|{smiles}"

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, energy, dipole, numbers, xyz, grad_calls, converged,"
                " seconds, error FROM calc WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        status, energy, dipole, nums, xyz, gc, conv, secs, err = row
        return {"status": status, "energy": energy, "dipole": dipole,
                "numbers": json.loads(nums) if nums else None,
                "xyz": np.array(json.loads(xyz)) if xyz else None,
                "grad_calls": gc, "converged": bool(conv) if conv is not None else None,
                "seconds": secs, "error": err, "cached": True}

    def put(self, key: str, smiles: str, solvent: str, recipe: str, rec: dict,
            method: str = METHOD) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calc (key, smiles, solvent, method, recipe,"
                " status, energy, dipole, numbers, xyz, grad_calls, converged,"
                " seconds, error, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, smiles, solvent, method, recipe, rec["status"],
                 rec.get("energy"), rec.get("dipole"),
                 json.dumps(rec["numbers"]) if rec.get("numbers") is not None else None,
                 json.dumps(np.asarray(rec["xyz"]).tolist()) if rec.get("xyz") is not None else None,
                 rec.get("grad_calls"), int(bool(rec.get("converged"))) if rec.get("converged") is not None else None,
                 rec.get("seconds"), rec.get("error"),
                 time.strftime("%Y-%m-%dT%H:%M:%S")))
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*), COALESCE(SUM(seconds),0) FROM calc GROUP BY status"
            ).fetchall()
        out = {"ok": 0, "fail": 0, "seconds_saved": 0.0}
        for status, n, secs in rows:
            out[status] = n
            out["seconds_saved"] += secs
        out["total"] = out["ok"] + out["fail"]
        return out

    def clear_failures(self) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM calc WHERE status='fail'").rowcount
            self._conn.commit()
        return n


# ---------------------------------------------------------------------------
# Embedding recipes. The recipe name is part of the cache key, so two protocols
# never contaminate each other.
# ---------------------------------------------------------------------------
def _lowest_mmff(smiles: str, n_confs: int, seed: int = 42):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    # EmbedMultipleConfs returns a LIST of conformer ids; `== 0` is always False,
    # so an empty embedding used to fall through to argmin([]) and raise.
    if len(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=p)) == 0:
        return None
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=1000)
    except Exception:
        res = [(0, 0.0)] * m.GetNumConformers()
    best = int(np.argmin([e for _c, e in res]))
    single = Chem.Mol(m); single.RemoveAllConformers()
    c = Chem.Conformer(m.GetConformer(best)); c.SetId(0)
    single.AddConformer(c, assignId=True)
    return single


def geometry_for(smiles: str, recipe: str):
    """Starting geometry for a recipe name."""
    if recipe.startswith("planar"):
        import dasa_chem as dc
        n = int(recipe.replace("planar", "") or 120)
        m3d, _tw = dc.planar_conformer(smiles, n_confs=n, seed=42)
        return m3d
    if recipe.startswith("lowest"):
        n = int(recipe.replace("lowest", "") or 20)
        return _lowest_mmff(smiles, n)
    raise ValueError(f"unknown recipe {recipe!r}")


_CACHE: Optional[XTBCache] = None


def cache(path: str = DEFAULT_DB) -> XTBCache:
    global _CACHE
    if _CACHE is None or _CACHE.path != path:
        _CACHE = XTBCache(path)
    return _CACHE


def cached_optimise(smiles: str, solvent: str, recipe: str = "lowest20",
                    db: str = DEFAULT_DB, force: bool = False) -> dict:
    """GFN2-xTB optimisation with resume-by-default caching.

    Returns {status, energy, dipole, xyz, numbers, ...}. `status` is "ok" or
    "fail"; a cached result carries cached=True. SMILES are canonicalised before
    keying, so the answer cannot depend on how the molecule was written.
    """
    from calibrate_proxies import xtb_optimise
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "fail", "error": "unparseable SMILES", "cached": False}
    canon = Chem.MolToSmiles(mol)
    c = cache(db)
    k = c.key(canon, solvent, recipe)
    if not force:
        hit = c.get(k)
        if hit is not None:
            return hit

    t0 = time.time()
    rec: dict
    try:
        m3d = geometry_for(canon, recipe)
        if m3d is None:
            rec = {"status": "fail", "error": "embedding failed"}
        else:
            xyz, e, dip, ncalls, ok = xtb_optimise(m3d, solvent)
            rec = {"status": "ok", "energy": float(e), "dipole": float(dip),
                   "numbers": [a.GetAtomicNum() for a in m3d.GetAtoms()],
                   "xyz": xyz, "grad_calls": int(ncalls), "converged": bool(ok)}
    except Exception as exc:
        rec = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    rec["seconds"] = time.time() - t0
    c.put(k, canon, solvent, recipe, rec)
    rec["cached"] = False
    if rec.get("xyz") is not None:
        rec["xyz"] = np.asarray(rec["xyz"])
    return rec


def is_cached(smiles: str, solvent: str, recipe: str = "lowest20",
              db: str = DEFAULT_DB) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return cache(db).get(cache(db).key(Chem.MolToSmiles(mol), solvent, recipe)) is not None


def mol_with_cached_geometry(smiles: str, rec: dict):
    """Rebuild an RDKit mol carrying a geometry that came out of the cache.

    `geometry_for` always starts from Chem.AddHs(MolFromSmiles(canonical)), so the
    atom order is reproducible from the canonical SMILES alone and the stored
    coordinates drop straight back on. This is what makes new descriptors free:
    no optimisation is repeated, only the geometry is re-read.
    """
    if rec.get("xyz") is None:
        return None
    m = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(smiles)))
    xyz = np.asarray(rec["xyz"])
    if m.GetNumAtoms() != len(xyz):
        return None
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    return m
