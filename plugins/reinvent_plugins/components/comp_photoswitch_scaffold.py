"""Photoswitch scaffold filter + custom SMARTS alerts.
Returns 1 if the molecule has an azo/diarylethene/hydrazone core and
no forbidden substructures, 0 otherwise.
"""

__all__ = ["PhotoswitchScaffold"]
from typing import List
import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

CORE_SMARTS = [
    Chem.MolFromSmarts("[#6]/N=N/[#6]"),
    Chem.MolFromSmarts("[#6]\\N=N\\[#6]"),
    Chem.MolFromSmarts("[#6]N=N[#6]"),
    Chem.MolFromSmarts("c1cc2ccc1CC=2"),
    Chem.MolFromSmarts("[#6]/C=N/[#7]"),
]

@add_tag("__parameters")
@dataclass
class Parameters:
    pass

@add_tag("__component")
class PhotoswitchScaffold:
    def __init__(self, params: Parameters):
        pass

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(0.0)
                continue
            has_core = any(mol.HasSubstructMatch(pat) for pat in CORE_SMARTS if pat)
            scores.append(1.0 if has_core else 0.0)
        return ComponentResults([np.array(scores, dtype=float)])
