"""GFN2-xTB HOMO-LUMO gap scorer for REINVENT.

Scores molecules based on their GFN2-xTB HOMO-LUMO gap (eV) as a proxy for
visible-light absorption.  Scoring is done directly in gap-space rather than
converting to λ_max, because the GFN2-xTB gap underestimates DFT values by
~2.5× and the correction is not constant across chemical space.

Calibration on azo photoswitches (xtb-python ≥ 22.x):
  azobenzene      gap=1.36 eV → λ_exp=320 nm
  4-NH2-azo       gap=1.34 eV → λ_exp=385 nm
  methyl orange   gap=1.25 eV → λ_exp=460 nm
Target range for visible absorbers (400–650 nm): gap_min=0.9, gap_max=1.4 eV.

CAUTION: ~1-3 s/molecule on CPU.  Use with small batches (batch_size=40).

[[stage.scoring.component]]
[stage.scoring.component.XTBHomoLumo]

[[stage.scoring.component.XTBHomoLumo.endpoint]]
name = "xTB_Gap"
weight = 0.7

params.gap_min_ev = [0.9]    # lower bound  (~650 nm visible)
params.gap_max_ev = [1.4]    # upper bound  (~400 nm visible)
"""

__all__ = ["XTBHomoLumo"]
from typing import List

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
    gap_min_ev: List[float]
    gap_max_ev: List[float]


@add_tag("__component")
class XTBHomoLumo:
    def __init__(self, params: Parameters):
        self.gap_min = params.gap_min_ev[0]
        self.gap_max = params.gap_max_ev[0]
        self._mid    = (self.gap_min + self.gap_max) / 2.0
        self._width  = (self.gap_max - self.gap_min) / 2.0

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score_mol(mol) for mol in mols], dtype=float
        )])

    def _score_mol(self, mol):
        if mol is None:
            return 0.0
        try:
            from xtb.interface import Calculator, Param
        except ImportError:
            return 0.0

        try:
            mol3d = Chem.AddHs(mol)
            emb = AllChem.ETKDGv3()
            emb.randomSeed = 42
            if AllChem.EmbedMolecule(mol3d, emb) == -1:
                return 0.0
            AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)

            positions   = mol3d.GetConformer().GetPositions()
            atomic_nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
            coords_bohr = positions * 1.8897259886

            calc = Calculator(Param.GFN2xTB, atomic_nums, coords_bohr)
            calc.set_verbosity(0)
            res  = calc.singlepoint()

            evals = res.get_orbital_eigenvalues()   # Hartree
            occs  = res.get_orbital_occupations()   # 0.0 or 2.0 (± Fermi smearing)

            occupied   = evals[occs > 0.5]
            unoccupied = evals[occs <= 0.5]
            if len(occupied) == 0 or len(unoccupied) == 0:
                return 0.0

            gap_ev = (unoccupied[0] - occupied[-1]) * 27.2114
            if gap_ev <= 0:
                return 0.0

            # Gaussian centred on [gap_min, gap_max] midpoint
            score = float(np.exp(-0.5 * ((gap_ev - self._mid) / self._width) ** 2))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.0
