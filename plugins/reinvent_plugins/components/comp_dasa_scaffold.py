"""DASA scaffold gate.

Returns 1.0 if the molecule contains a DASA open-form core (amino-triene enol
attached to a 1,3-dicarbonyl carbon acid), else 0.0. Use as a hard structural
gate in a geometric-mean scoring function so non-DASA generations score zero --
this is the DASA replacement for the azobenzene PhotoswitchScaffold component.

    [[stage.scoring.component]]
    [stage.scoring.component.DASAScaffold]
    [[stage.scoring.component.DASAScaffold.endpoint]]
    name = "DASA"
    weight = 1.0
"""

__all__ = ["DASAScaffold"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from .dasa_common import is_dasa


@add_tag("__parameters")
@dataclass
class Parameters:
    pass


@add_tag("__component")
class DASAScaffold:
    def __init__(self, params: Parameters):
        pass

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = [1.0 if is_dasa(mol) else 0.0 for mol in mols]
        return ComponentResults([np.array(scores, dtype=float)])
