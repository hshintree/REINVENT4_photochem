"""Aqueous-solubility / hydrophilicity scorer (cheap, RDKit-only).

Water solubility is one half of the DASA pivot goal (the other being water-
*switchability*, see DASASwitchability). This component rewards hydrophilic,
water-soluble molecules using a Delaney ESOL-style log(S) estimate blended with
Crippen logP. No 3D / quantum work, so it is cheap enough to run in every RL
stage as a soft bias toward water compatibility.

log(S) here is the Delaney (2004) linear ESOL model:
    logS = 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*AP
where AP is the aromatic proportion (aromatic atoms / heavy atoms). Higher logS
(and lower logP) -> higher score, via a sigmoid centred on the target window.

    [[stage.scoring.component]]
    [stage.scoring.component.AqueousSolubility]
    [[stage.scoring.component.AqueousSolubility.endpoint]]
    name = "Solubility"
    weight = 0.6
    params.logs_target = -2.0   # target log10(mol/L); -2 ~ 10 mM, quite soluble
    params.logs_width  = 1.5    # sigmoid softness in log units
    params.logp_max    = 3.0    # extra penalty above this clogP
"""

__all__ = ["AqueousSolubility"]
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


def _esol_logs(mol) -> float:
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    ap = (arom / heavy) if heavy else 0.0
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


@add_tag("__parameters")
@dataclass
class Parameters:
    logs_target: List[float]
    logs_width: List[float]
    logp_max: List[float]


@add_tag("__component")
class AqueousSolubility:
    def __init__(self, params: Parameters):
        self.logs_target = params.logs_target[0]
        self.logs_width = params.logs_width[0]
        self.logp_max = params.logp_max[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        return ComponentResults([np.array(
            [self._score(mol) for mol in mols], dtype=float)])

    def _score(self, mol):
        if mol is None:
            return 0.0
        try:
            logs = _esol_logs(mol)
            # rising sigmoid: soluble (logs >= target) -> ~1, insoluble -> ~0
            sol = 1.0 / (1.0 + np.exp(-(logs - self.logs_target) / self.logs_width))
            # gentle extra penalty for very lipophilic molecules
            clogp = Crippen.MolLogP(mol)
            lip = 1.0 / (1.0 + np.exp((clogp - self.logp_max) / 0.8))
            return float(np.clip(np.sqrt(sol * lip), 0.0, 1.0))
        except Exception:
            return 0.0
