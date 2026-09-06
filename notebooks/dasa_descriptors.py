"""Conformer ensembles + steric descriptors for DASA open forms.

Two jobs:

1. ENSEMBLES. `planar_conformer` returns the single flattest conformer, which is
   what a TD-DFT calculation wants but is NOT necessarily what a solution-phase
   descriptor wants -- and it discards the conformational information that steric
   bulk actually carries. `planar_ensemble` returns the k lowest-MMFF conformers
   that are planar enough to be a chromophore, so a descriptor can be Boltzmann
   averaged and, more importantly, its ENSEMBLE SPREAD can be measured.

   Measured 2026-09-05: the same molecule written two ways gave twist 4.4 vs
   12.3 deg at n_confs=30, moving the dipole by 1.6 D (~40% of the range across
   the whole literature series) and d(C1..C5) by 0.5 A. BLA moved 0.002 A. So the
   spread is descriptor-specific and has to be measured per descriptor, not
   assumed.

2. STERICS. No new dependency: percent buried volume and Sterimol L/B1/B5 are
   plain geometry and are implemented here directly, so nothing has to be added to
   the (fragile) reinvent4 environment.

   Sterics is the axis that matters strategically. The electronic coordinate is a
   dead end on its own: charge separation RAISES the dark equilibrium and
   simultaneously WORSENS polar-solvent switching (measured, see
   notebooks/seed_proxy_fit.py). Both literature solutions to water are steric --
   Peterson's tether (Chem. Sci. 2023, 14, 13025) and Pischel's cucurbituril
   (JACS 2026, 148, 130, "significant steric effects can overrule the electronic
   effects"). Sterics is the only remaining orthogonal handle.

   Two DISTINCT steric roles, on different atoms, with different intent:
     * bulk at the donor N  -> raises the rotation barriers -> slows dark
       switching (good), but too much also kills forward photoswitching.
     * bulk at C5 / the triene -> destabilises the CLOSED isomer -> promotes
       reversion (good).
   They get separate descriptors and separate bands. One "bulk" number averages
   them out.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import dasa_chem as dc

_PT = Chem.GetPeriodicTable()
_KCAL = 627.5094740631


# ---------------------------------------------------------------------------
# Conformer ensembles
# ---------------------------------------------------------------------------
def planar_ensemble(smiles, n_confs: int = 120, k: int = 3,
                    max_twist_deg: float = 20.0, seed: int = 42):
    """The k lowest-MMFF conformers whose chain is planar to within max_twist_deg.

    Falls back to the k flattest if fewer than k are planar. Returns a list of
    (mol3d, twist_deg, mmff_energy) with one conformer each, ordered by MMFF
    energy. Empty list on failure.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return []
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.pruneRmsThresh = 0.3
    if len(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=p)) == 0:
        return []
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    except Exception:
        res = [(0, 0.0)] * m.GetNumConformers()

    cands = []
    for cid in range(m.GetNumConformers()):
        single = Chem.Mol(m)
        single.RemoveAllConformers()
        c = Chem.Conformer(m.GetConformer(cid))
        c.SetId(0)
        single.AddConformer(c, assignId=True)
        tw = dc.chain_planarity(single)
        if tw is None:
            continue
        cands.append((single, float(tw), float(res[cid][1])))
    if not cands:
        return []

    planar = [c for c in cands if c[1] <= max_twist_deg]
    pool = planar if len(planar) >= k else sorted(cands, key=lambda t: t[1])[:k]
    return sorted(pool, key=lambda t: t[2])[:k]


def boltzmann(values, energies_hartree, T: float = 298.15):
    """Boltzmann-weighted mean and (weighted) spread of a descriptor."""
    v = np.asarray([x for x in values], dtype=float)
    e = np.asarray(energies_hartree, dtype=float)
    rel = (e - e.min()) * _KCAL
    w = np.exp(-rel / (0.0019872041 * T))
    w /= w.sum()
    mean = float((w * v).sum())
    sd = float(np.sqrt((w * (v - mean) ** 2).sum()))
    return mean, sd, float(v.max() - v.min())


# ---------------------------------------------------------------------------
# Steric descriptors
# ---------------------------------------------------------------------------
def _radii(mol) -> np.ndarray:
    return np.array([_PT.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])


def donor_side_atoms(mol, idx) -> np.ndarray:
    """Boolean mask of the donor SUBSTITUENTS: everything reachable from N without
    entering any core atom (N itself excluded from the mask).

    BUG FIXED 2026-09-05: this previously blocked only the N->C5 *bond*. For a
    TETHERED donor the ring reconnects N to C5 the long way round, so the walk
    leaked through the tether into the triene, the acceptor and the entire rest of
    the molecule -- giving the three Peterson compounds vbur_C5 ~0.59 and B5 ~12 A
    against ~0.25 / ~5.7 for everything else, and manufacturing a spurious
    steric-vs-residual correlation. Blocking every core atom handles rings.
    """
    blocked = set(idx.values())
    n = idx["N"]
    seen, stack = {n}, [n]
    while stack:
        a = stack.pop()
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in blocked or j in seen:
                continue
            seen.add(j); stack.append(j)
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for a in seen:
        mask[a] = True
    mask[n] = False
    return mask


def core_atoms(mol, idx) -> np.ndarray:
    """The 8 core atoms plus their attached hydrogens."""
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for a in idx.values():
        mask[a] = True
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                mask[nb.GetIdx()] = True
    return mask


def buried_volume(xyz, radii, center, R: float, mask, n: int = 20000,
                  seed: int = 0) -> float:
    """Fraction of a sphere of radius R at `center` occupied by the vdW volume of
    the atoms selected by `mask`. The standard %V_bur of the organometallic
    literature, restricted to a chosen substructure so donor bulk and acceptor
    bulk can be told apart."""
    sel = np.where(mask)[0]
    if sel.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    pts = center + v * (R * rng.random(n) ** (1.0 / 3.0))[:, None]
    d = np.linalg.norm(pts[:, None, :] - xyz[None, sel, :], axis=2)
    return float((d < radii[sel][None, :]).any(axis=1).mean())


def sterimol(xyz, radii, origin_i: int, axis_from_i: int, mask,
             n_angles: int = 360):
    """Sterimol L / B1 / B5 for the substituents selected by `mask`, measured
    from `origin_i` along the (origin_i - axis_from_i) direction, i.e. pointing
    AWAY from the triene and into the substituent, which is the Sterimol convention.

    L  extent along the axis, B5 widest perpendicular extent, B1 narrowest
    (the minimum over rotations of the maximum perpendicular projection).
    """
    sel = np.where(mask)[0]
    if sel.size == 0:
        return 0.0, 0.0, 0.0
    o = xyz[origin_i]
    ax = o - xyz[axis_from_i]
    nrm = np.linalg.norm(ax)
    if nrm < 1e-6:
        return 0.0, 0.0, 0.0
    ax = ax / nrm
    rel = xyz[sel] - o
    r = radii[sel]

    L = float(np.max(rel @ ax + r))
    perp = rel - np.outer(rel @ ax, ax)
    B5 = float(np.max(np.linalg.norm(perp, axis=1) + r))

    # orthonormal basis in the perpendicular plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(ax @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ax, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    th = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dirs = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    widths = (perp @ dirs.T) + r[:, None]          # (natoms, n_angles)
    B1 = float(np.min(np.max(widths, axis=0)))
    return L, B1, B5


def steric_descriptors(mol3d, xyz, idx) -> dict:
    """All steric descriptors for one conformer geometry (Angstrom)."""
    radii = _radii(mol3d)
    dmask = donor_side_atoms(mol3d, idx)
    cmask = core_atoms(mol3d, idx)
    non_core = ~cmask

    n_pos, ca_pos, ce_pos = xyz[idx["N"]], xyz[idx["Ca"]], xyz[idx["Ce"]]
    site = 0.5 * (ca_pos + ce_pos)          # midpoint of the forming C5...C1 bond

    L, B1, B5 = sterimol(xyz, radii, idx["N"], idx["Ca"], dmask)
    return {
        # donor bulk, seen from the nitrogen
        "vbur_N": buried_volume(xyz, radii, n_pos, 3.5, dmask),
        # donor bulk crowding the carbon that must pyramidalise on cyclization
        "vbur_C5": buried_volume(xyz, radii, ca_pos, 3.5, dmask),
        # total non-core crowding at the bond the electrocyclization has to form
        "vbur_site": buried_volume(xyz, radii, site, 4.0, non_core),
        "sterimol_L": L, "sterimol_B1": B1, "sterimol_B5": B5,
        "d_C1C5": float(np.linalg.norm(ca_pos - ce_pos)),
        "n_donor_heavy": int(sum(1 for i in np.where(dmask)[0]
                                 if mol3d.GetAtomWithIdx(int(i)).GetAtomicNum() > 1)),
    }
