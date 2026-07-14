"""xTB-based lambda_max filter.
Gaussian centred on target absorption window, hard cutoff below threshold.
"""

__all__ = ["XTBLambdaFilter"]
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

_XTB_CORRECTION = 2.5

@add_tag("__parameters")
@dataclass
class Parameters:
    lambda_cutoff: List[float]
    lambda_target: List[float]
    lambda_sigma:  List[float]

@add_tag("__component")
class XTBLambdaFilter:
    def __init__(self, params: Parameters):
        self.cutoff = params.lambda_cutoff[0]
        self.target = params.lambda_target[0]
        self.sigma  = params.lambda_sigma[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score(mol) for mol in mols], dtype=float)])

    def _score(self, mol):
        if mol is None: return 0.0
        try:
            from xtb.interface import Calculator, Param
        except ImportError:
            return 0.5
        try:
            mol3d = Chem.AddHs(mol)
            emb = AllChem.ETKDGv3(); emb.randomSeed = 42
            if AllChem.EmbedMolecule(mol3d, emb) == -1: return 0.0
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
            pos  = mol3d.GetConformer().GetPositions()
            nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
            calc = Calculator(Param.GFN2xTB, nums, pos * 1.8897259886)
            calc.set_verbosity(0)
            res  = calc.singlepoint()
            evals, occs = res.get_orbital_eigenvalues(), res.get_orbital_occupations()
            occ, unocc = evals[occs > 0.5], evals[occs <= 0.5]
            if len(occ) == 0 or len(unocc) == 0: return 0.0
            gap = (unocc[0] - occ[-1]) * 27.2114
            if gap <= 0: return 0.0
            lam_est = 1240.0 / (gap * _XTB_CORRECTION)
            if lam_est < self.cutoff: return 0.0
            return float(np.clip(np.exp(-0.5 * ((lam_est - self.target)/self.sigma)**2), 0, 1))
        except Exception:
            return 0.0
