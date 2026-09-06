"""Ground-truth validation: does C5/triene crowding destabilise the CLOSED form,
and do our steric descriptors rank that effect?

WHY THIS EXISTS. `verify_sterics.py` ran a sensitivity test (do descriptors respond
to bulk?) and an orthogonality test (are they independent of BLA?). Neither is a
validation, because neither has an OUTCOME to predict. Here the outcome is
DeltaE(open -> closed): the thing "bulk at C5 destabilises the closed isomer"
actually asserts. A descriptor earns its place only if it ranks that.

TWO BUGS IN verify_sterics.py THAT THIS FILE EXISTS TO CORRECT
--------------------------------------------------------------
B1  T4 concluded "%V_bur cannot see a C5 methyl". It reported vbur_C5 and
    sterimol_B1, both of which are computed over `donor_side_atoms`, defined as
    "everything reachable from N WITHOUT ENTERING ANY CORE ATOM". Ca (=C5) is a
    core atom, so a substituent ON Ca is unreachable and is excluded from the mask
    by construction. Those two numbers CANNOT respond to a C5 methyl and their
    being ~0 says nothing about %V_bur. `vbur_site` uses the non-core mask, does
    contain the methyl, and moved +0.09/+0.14/+0.15 in the same saved JSON -- it
    was simply not printed. Verified directly: the methyl carbon has
    dmask=False, non_core=True.
B2  T4's pair SMILES carried no stereochemistry ("so both members are treated
    identically"). The DASA open form is (2Z,4E) and the E/Z pattern IS the
    cyclisation geometry. Here every analogue is built by GRAPH EDIT from the
    stereo-defined parent, so the double-bond configuration is inherited rather
    than re-guessed by ETKDG, and d(Ca..Ce) is reported as the check.

WHAT IS AND IS NOT GROUND TRUTH HERE
------------------------------------
This is a COMPUTED ground truth (GFN2-xTB/ALPB), not a measured one. That is a
deliberate choice, not an oversight: per DASA_FREE_ENERGY.md §10 the DASA
literature contains no matched pair isolating triene sterics with a measured
equilibrium, and per §3 our own xTB DeltaE magnitudes sat 4-7 kcal/mol off the
measured CHCl3 equilibria. So:
  * absolute DeltaE is NOT used for anything;
  * only WITHIN-SCAFFOLD DeltaDeltaE (substituted minus parent) is used, where the
    systematic error largely cancels (§6);
  * the claim tested is a SIGN and a RANKING, never a magnitude.

THE THREE TESTS
---------------
G1 SIGN         Does C5 bulk raise DeltaE(open->closed), i.e. destabilise closed?
                This is the premise itself. If it fails, the descriptor question is
                moot and the C5 band in the scoring function is backwards.
G2 POSITION     A methyl at C5 (Ca, an atom of the forming sigma bond) must matter
                MORE than a methyl at C4 (Cb, one bond away, not in the new bond),
                which must matter more than an ethyl on a remote acceptor N. Any
                descriptor that responds equally to all three is counting atoms,
                not sensing position. This is the test that separates a steric
                descriptor from molecular weight.
G3 RANKING      Spearman(descriptor, DeltaE) within each donor scaffold. The
                honest statistic, since only the ordering is defensible.

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
        ~/miniconda3/envs/reinvent4/bin/python notebooks/validate_sterics_groundtruth.py
      add `--preflight` to check structure construction without running xTB.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dasa_chem as dc
from calibrate_proxies import xtb_optimise, bond_lengths
from dasa_descriptors import (planar_ensemble, steric_descriptors, buried_volume,
                              core_atoms, _radii)

H2KCAL = 627.5094740631
SOLVENT = "toluene"          # aprotic: the closed form is the NEUTRAL keto
                             # (Lerch Angew 2018; see DASA_FREE_ENERGY.md §1)
N_CONF_OPEN = 120            # embedded, planar-filtered, then k kept
N_CONF_CLOSED = 200          # closed form has 2 stereocentres -> more sampling
K_CONF = 3                   # xTB-optimised per state; min taken, spread reported
SEED = 42
WORKERS = int(os.environ.get("VAL_WORKERS", "6"))


# ---------------------------------------------------------------------------
# Structure construction by graph edit (preserves the parent's (2Z,4E))
# ---------------------------------------------------------------------------
def attach(smiles: str, core_key: str, frag: str) -> str | None:
    """Attach `frag` to the core atom named `core_key`, replacing one implicit H.

    Graph edit rather than hand-written SMILES so the parent's double-bond
    stereochemistry is inherited. RDKit stores bond stereo against HEAVY
    neighbour atoms, so adding a heavy substituent to a stereo-bond terminus does
    not disturb the existing E/Z assignment; the assertion below checks that.
    """
    mol = Chem.MolFromSmiles(smiles)
    sub = Chem.MolFromSmiles(frag)
    if mol is None or sub is None:
        return None
    idx = dc._core_idx(mol)
    if idx is None:
        return None
    target = idx[core_key]
    stereo_before = {b.GetIdx(): str(b.GetStereo()) for b in mol.GetBonds()
                     if b.GetStereo() != Chem.BondStereo.STEREONONE}

    n_before = mol.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(mol, sub))
    combo.AddBond(target, n_before, Chem.BondType.SINGLE)
    out = combo.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    stereo_after = {b.GetIdx(): str(b.GetStereo()) for b in out.GetBonds()
                    if b.GetStereo() != Chem.BondStereo.STEREONONE}
    if stereo_after != stereo_before:      # the edit changed a configuration
        return None
    return Chem.MolToSmiles(out) if dc.is_dasa(out) else None


def swap_acceptor_nme_to_net(smiles: str) -> str | None:
    """N-methyl -> N-ethyl on the barbituric acceptor: the REMOTE control.

    Adds exactly one carbon, like a C5 methyl does, but ~6 bonds from the forming
    bond. Anything that moves for this is responding to size, not to position.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    patt = Chem.MolFromSmarts("[CH3]-[NX3]([#6]=O)[#6]=O")
    m = mol.GetSubstructMatch(patt)
    if not m:
        return None
    rw = Chem.RWMol(mol)
    c = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(m[0], c, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Chem.MolToSmiles(out) if dc.is_dasa(out) else None


A_BARB = "=C1C(=O)N(C)C(=O)N(C)C1=O"
DONORS = {
    "Me2N":     "CN(C)",
    "indoline": "CC1Cc2ccccc2N1",
    "anilino":  "CN(c1ccccc1)",
}
# C5 bulk ladder + the two positional controls. "" means the unsubstituted parent.
VARIANTS = [
    ("C5-H",   None,  None),
    ("C5-Me",  "Ca",  "C"),
    ("C5-Et",  "Ca",  "CC"),
    ("C5-iPr", "Ca",  "C(C)C"),       # attaches via atom 0 = the CENTRAL carbon
    ("C5-tBu", "Ca",  "C(C)(C)C"),
    ("C5-Ph",  "Ca",  "c1ccccc1"),
    ("C4-Me",  "Cb",  "C"),           # G2: one bond off the forming bond
    ("accNEt", "remote", None),       # G2: ~6 bonds away, same +1 carbon
]


def build_panel() -> list[tuple[str, str]]:
    panel = []
    for dname, d in DONORS.items():
        parent = f"{d}/C=C/C=C(\\O)C{A_BARB}"
        for vname, key, frag in VARIANTS:
            if key is None:
                smi = parent
            elif key == "remote":
                smi = swap_acceptor_nme_to_net(parent)
            else:
                smi = attach(parent, key, frag)
            if smi is None:
                print(f"  !! could not build {dname}/{vname}")
                continue
            panel.append((f"{dname} {vname}", smi))
    return panel


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------
def _confs_of(mol, n_confs: int, k: int):
    """k lowest-MMFF conformers of an arbitrary mol, as single-conformer copies."""
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = SEED
    p.pruneRmsThresh = 0.3
    if AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=p) == 0:
        return []
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    except Exception:
        res = [(0, 0.0)] * m.GetNumConformers()
    out = []
    for cid in range(m.GetNumConformers()):
        single = Chem.Mol(m)
        single.RemoveAllConformers()
        c = Chem.Conformer(m.GetConformer(cid))
        c.SetId(0)
        single.AddConformer(c, assignId=True)
        out.append((single, float(res[cid][1])))
    return [t[0] for t in sorted(out, key=lambda t: t[1])[:k]]


def dihedral(xyz, i, j, k, l) -> float:
    """Signed dihedral i-j-k-l in degrees, from raw coordinates.

    Used as the check on `attach`: RDKit's STEREOE/STEREOZ label depends on WHICH
    neighbours it picks as reference atoms, so the label can change meaning when a
    substituent is added without the configuration actually changing (and, worse,
    can stay the same while the configuration flips). The dihedral is unambiguous.
    """
    b0, b1, b2 = xyz[i] - xyz[j], xyz[k] - xyz[j], xyz[l] - xyz[k]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def best_energy(confs, solvent):
    """Lowest GFN2-xTB energy over a conformer list, plus the spread in kcal/mol."""
    es, geoms, conv = [], [], []
    for c in confs:
        try:
            xyz, e, _dip, _n, ok = xtb_optimise(c, solvent)
        except Exception:
            continue
        es.append(e)
        geoms.append((c, xyz))
        conv.append(bool(ok))
    if not es:
        return None, None, None, None
    j = int(np.argmin(es))
    spread = (max(es) - min(es)) * H2KCAL if len(es) > 1 else 0.0
    return es[j], geoms[j], float(spread), all(conv)


def profile(name: str, smi: str) -> dict:
    t0 = time.time()
    rec: dict = {"name": name, "smiles": smi}
    mol = Chem.MolFromSmiles(smi)
    idx = dc._core_idx(mol) if mol else None
    if idx is None or not dc.is_dasa(mol):
        return {**rec, "error": "not a DASA"}

    # ---- open form: planar ensemble (it is the chromophore) ----
    ens = planar_ensemble(smi, n_confs=N_CONF_OPEN, k=K_CONF, seed=SEED)
    if not ens:
        return {**rec, "error": "open geometry failed"}
    e_open, geom, sp_open, conv_open = best_energy([t[0] for t in ens], SOLVENT)
    if e_open is None:
        return {**rec, "error": "open xTB failed"}
    m3d, xyz = geom
    rec.update(twist_deg=round(ens[0][1], 1), E_open=e_open,
               spread_open=sp_open, conv_open=conv_open,
               # configuration check: ~180 deg = (4E), the cyclisable arrangement
               dih_N_Ca_Cb_Cc=dihedral(xyz, idx["N"], idx["Ca"], idx["Cb"], idx["Cc"]),
               dih_Cb_Cc_Cd_Ce=dihedral(xyz, idx["Cb"], idx["Cc"], idx["Cd"], idx["Ce"]),
               **bond_lengths(xyz, idx), **steric_descriptors(m3d, xyz, idx))

    # ---- the mask fix: %V_bur at C5 over the NON-CORE mask, which unlike the
    # donor-side mask contains substituents ON C5 (bug B1 above) ----
    radii = _radii(m3d)
    non_core = ~core_atoms(m3d, idx)
    for R in (3.0, 3.5, 4.0):
        rec[f"vbur_C5_all_{R}"] = buried_volume(xyz, radii, xyz[idx["Ca"]],
                                                R, non_core, seed=0)

    # ---- closed forms ----
    for label, fn in (("keto", dc.open_to_closed_keto), ("zwit", dc.open_to_closed)):
        csmi = fn(smi)
        if csmi is None:
            rec[f"E_{label}"] = None
            rec[f"err_{label}"] = "cyclisation failed"
            continue
        cmol = Chem.MolFromSmiles(csmi)
        confs = _confs_of(cmol, N_CONF_CLOSED, K_CONF) if cmol else []
        if not confs:
            rec[f"E_{label}"] = None
            rec[f"err_{label}"] = "closed geometry failed"
            continue
        e_c, _g, sp, conv = best_energy(confs, SOLVENT)
        rec[f"smiles_{label}"] = csmi
        rec[f"E_{label}"] = e_c
        rec[f"spread_{label}"] = sp
        rec[f"conv_{label}"] = conv
        rec[f"dE_{label}"] = (e_c - e_open) * H2KCAL if e_c is not None else None

    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
def main():
    panel = build_panel()
    print(f"panel: {len(panel)} molecules, GFN2-xTB/ALPB({SOLVENT}), "
          f"k={K_CONF} conformers per state\n")

    if "--preflight" in sys.argv:
        for name, smi in panel:
            mol = Chem.MolFromSmiles(smi)
            k = dc.open_to_closed_keto(smi)
            z = dc.open_to_closed(smi)
            print(f"  {name:18s} heavy={mol.GetNumHeavyAtoms():3d} "
                  f"keto={'ok' if k else 'FAIL':4s} zwit={'ok' if z else 'FAIL':4s}  {smi}")
        return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        recs = list(ex.map(lambda t: profile(*t), panel))
    for r in recs:
        if "error" in r:
            print(f"  !! {r['name']}: {r['error']}")
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)}/{len(panel)} molecules OK in {time.time()-t0:.0f} s\n")

    by = {r["name"]: r for r in recs}
    order = [v[0] for v in VARIANTS]

    # ---------------- G1 / G2 ----------------
    print("G1+G2  DeltaDeltaE(open->closed), kcal/mol, vs the C5-H parent of the "
          "SAME donor.\n       POSITIVE = the substituent destabilises CLOSED = "
          "pushes the equilibrium OPEN.")
    print(f"       (conformer spread column is the max-min over k={K_CONF}; it is "
          "the error bar)\n")
    rows = {}
    for d in DONORS:
        base = by.get(f"{d} C5-H")
        if not base:
            continue
        print(f"    donor = {d}")
        print(f"      {'variant':9s} {'dE_keto':>8s} {'ddE_keto':>9s} {'dE_zwit':>8s} "
              f"{'ddE_zwit':>9s} {'sprd_o':>7s} {'sprd_k':>7s} {'dih4E':>7s} "
              f"{'d(CaCe)':>8s} "
              f"{'vbur_site':>10s} {'vburC5all':>10s} {'vbur_C5':>8s} {'BLA':>8s}")
        for v in order:
            r = by.get(f"{d} {v}")
            if not r:
                continue
            f = lambda x: (f"{x:+8.2f}" if isinstance(x, (int, float)) else f"{'--':>8s}")
            ddk = (r["dE_keto"] - base["dE_keto"]
                   if None not in (r.get("dE_keto"), base.get("dE_keto")) else None)
            ddz = (r["dE_zwit"] - base["dE_zwit"]
                   if None not in (r.get("dE_zwit"), base.get("dE_zwit")) else None)
            rows.setdefault(d, []).append((v, ddk, ddz, r))
            print(f"      {v:9s} {f(r.get('dE_keto'))} {f(ddk):>9s} "
                  f"{f(r.get('dE_zwit'))} {f(ddz):>9s} "
                  f"{r.get('spread_open', 0):7.2f} {r.get('spread_keto', 0):7.2f} "
                  f"{abs(r['dih_N_Ca_Cb_Cc']):7.1f} "
                  f"{r['d_C1C5']:8.3f} {r['vbur_site']:10.4f} "
                  f"{r['vbur_C5_all_3.5']:10.4f} {r['vbur_C5']:8.4f} {r['BLA']:+8.4f}")
        print()

    print("    G2 verdict -- positional specificity of the OUTCOME "
          "(mean ddE over donors):")
    for v in order[1:]:
        vals = [x[1] for d in rows for x in rows[d] if x[0] == v and x[1] is not None]
        if vals:
            print(f"      {v:9s} mean ddE_keto = {np.mean(vals):+6.2f} kcal/mol   "
                  f"n={len(vals)}  ({', '.join(f'{x:+.2f}' for x in vals)})")

    # ---------------- G3 ----------------
    print("\nG3  RANKING - Spearman(descriptor, dE_keto) WITHIN each donor scaffold.")
    keys = ["vbur_site", "vbur_C5_all_3.0", "vbur_C5_all_3.5", "vbur_C5_all_4.0",
            "vbur_C5", "vbur_N", "sterimol_B1", "sterimol_B5", "n_donor_heavy",
            "heavy_all", "BLA", "d_C1C5"]
    for r in recs:
        r["heavy_all"] = Chem.MolFromSmiles(r["smiles"]).GetNumHeavyAtoms()
    print(f"    {'descriptor':18s} " + " ".join(f"{d:>12s}" for d in DONORS)
          + f" {'pooled':>12s}")
    pooled_x: dict = {k: [] for k in keys}
    pooled_y: list = []
    for d in DONORS:
        sub = [by[f"{d} {v}"] for v in order if f"{d} {v}" in by
               and by[f"{d} {v}"].get("dE_keto") is not None]
        for k in keys:
            pooled_x[k].extend([s[k] for s in sub])
        pooled_y.extend([s["dE_keto"] - by[f"{d} C5-H"]["dE_keto"] for s in sub])
    for k in keys:
        cells = []
        for d in DONORS:
            sub = [by[f"{d} {v}"] for v in order if f"{d} {v}" in by
                   and by[f"{d} {v}"].get("dE_keto") is not None]
            if len(sub) < 3:
                cells.append(f"{'n/a':>12s}")
                continue
            rho, _ = spearmanr([s[k] for s in sub], [s["dE_keto"] for s in sub])
            cells.append(f"{rho:>+12.3f}")
        rho_p, p_p = spearmanr(pooled_x[k], pooled_y)
        print(f"    {k:18s} " + " ".join(cells) + f" {rho_p:>+12.3f}")
    print(f"    (pooled column correlates against WITHIN-DONOR ddE, n={len(pooled_y)};"
          f" that is the\n     only pooling that is legitimate, since absolute dE "
          "carries the donor's electronics.)")

    out = os.path.join(HERE, "data", "steric_groundtruth.json")
    with open(out, "w") as f:
        json.dump(recs, f, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
