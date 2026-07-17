"""Self-contained DASA chemistry helpers for the scoring plugins.

Kept independent of notebooks/dasa_chem.py so the plugins import cleanly inside
the ``reinvent`` subprocess (which only has the plugins dir on PYTHONPATH). The
SMARTS here mirror the canonical library and are covered by the same validation.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# DASA open-form core: amino-triene enol whose enol carbon is double-bonded to a
# carbon bearing a carbonyl (the 1,3-dicarbonyl "carbon acid"). Triene carbons
# are [CX3] so substituted-backbone (methylated triene) variants also match.
# Mirrors notebooks/dasa_chem.py (validated against DASAs and decoys there).
DASA_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3]-[#6](-[OX2H1,OX1-])=[#6](~[#6]=O)"
_DASA_OPEN = Chem.MolFromSmarts(DASA_OPEN_SMARTS)

_BOHR = 1.8897259886  # Angstrom -> Bohr


def is_dasa(mol) -> bool:
    return mol is not None and mol.HasSubstructMatch(_DASA_OPEN)


def embed_3d(mol, seed: int = 42, max_iters: int = 500):
    """Add H, embed a single ETKDGv3 conformer and MMFF-optimise. None on fail."""
    if mol is None:
        return None
    try:
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(m, params) == -1:
            if AllChem.EmbedMolecule(m, AllChem.ETKDG()) == -1:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=max_iters)
        except Exception:
            pass
        return m
    except Exception:
        return None


def xtb_properties(mol3d, solvent: str | None = None):
    """Return (energy_Ha, dipole_norm_au, homo_lumo_gap_eV) from GFN2-xTB.

    ``solvent`` (e.g. "water", "toluene") switches on the ALPB implicit model.
    Returns None if xtb is unavailable or the calculation fails.
    """
    try:
        from xtb.interface import Calculator, Param
    except ImportError:
        return None
    try:
        pos = mol3d.GetConformer().GetPositions() * _BOHR
        nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
        calc = Calculator(Param.GFN2xTB, nums, pos)
        calc.set_verbosity(0)
        if solvent is not None:
            try:
                from xtb.utils import get_solvent
                calc.set_solvent(get_solvent(solvent))
            except Exception:
                pass
        res = calc.singlepoint()
        energy = res.get_energy()
        dip = float(np.linalg.norm(res.get_dipole()))
        evals = res.get_orbital_eigenvalues()
        occs = res.get_orbital_occupations()
        occ, unocc = evals[occs > 0.5], evals[occs <= 0.5]
        gap = (unocc[0] - occ[-1]) * 27.2114 if len(occ) and len(unocc) else None
        return energy, dip, gap
    except Exception:
        return None
