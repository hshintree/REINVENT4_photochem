"""xTB E/Z isomer energy gap scorer.
Rewards molecules with larger |dE(E-Z)|, indicating longer thermal half-life.
"""

__all__ = ["XTBIsomerGap"]
from typing import List
import re as _re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

@add_tag("__parameters")
@dataclass
class Parameters:
    de_min_kcal:    List[float]
    de_target_kcal: List[float]
    de_sigma_kcal:  List[float]

def _embed(mol):
    mol3d = Chem.AddHs(mol)
    emb = AllChem.ETKDGv3(); emb.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, emb) == -1: return None
    AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
    return mol3d

def _energy(mol3d):
    from xtb.interface import Calculator, Param
    pos  = mol3d.GetConformer().GetPositions()
    nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
    calc = Calculator(Param.GFN2xTB, nums, pos * 1.8897259886)
    calc.set_verbosity(0)
    return calc.singlepoint().get_energy()

def _flip_azo(smi):
    def _inv(m):
        return m.group(1) + "N=N" + ("/" if m.group(2) == chr(92) else chr(92))
    result = _re.sub(r"([/\\])N=N([/\\])", _inv, smi, count=1)
    if result == smi: return None
    if Chem.MolFromSmiles(result) is None: return None
    return result

@add_tag("__component")
class XTBIsomerGap:
    def __init__(self, params: Parameters):
        self.de_min = params.de_min_kcal[0]
        self.target = params.de_target_kcal[0]
        self.sigma  = params.de_sigma_kcal[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score(mol) for mol in mols], dtype=float)])

    def _score(self, mol):
        if mol is None: return 0.0
        try:
            from xtb.interface import Calculator, Param  # noqa: F401
        except ImportError:
            return 0.5
        try:
            smi = Chem.MolToSmiles(mol)
            e_mol = _embed(mol)
            if e_mol is None: return 0.0
            z_smi = _flip_azo(smi)
            if z_smi is None: return 0.3
            z_mol = Chem.MolFromSmiles(z_smi)
            if z_mol is None: return 0.0
            z_mol3d = _embed(z_mol)
            if z_mol3d is None: return 0.0
            dE = abs(_energy(z_mol3d) - _energy(e_mol)) * 627.509
            if dE < self.de_min: return 0.0
            return float(np.clip(
                np.exp(-0.5 * ((dE - self.target)/self.sigma)**2), 0, 1))
        except Exception:
            return 0.0
