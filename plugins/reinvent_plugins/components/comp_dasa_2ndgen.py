"""DASA generation/architecture steering — bias the loop toward water-switchable classes.

The water-switchable class is the SECOND GENERATION: a weak aromatic (aniline) donor
gives reduced open-form charge separation and a NEUTRAL keto closed form that escapes
the water trap (Chem Soc Rev 2023 10.1039/D3CS00508A; Chem Eur J 2021 10.1002/chem.202005110;
water precedent JACS 2022 10.1021/jacs.2c04920). First-generation tertiary dialkylamine
donors give a zwitterionic closed form that is locked closed in water.

WHY THIS EXISTS: a prior run reused a Stage-1 checkpoint that had drifted to 96% tertiary
dialkylamine (1st-gen, trapped) -- the neutral-keto anti-trap reward then had almost nothing
to reward. This component pushes the population toward the architectures that CAN escape the
trap, so the anti-trap metric has 2nd-gen candidates to optimise. It does NOT ban 1st-gen
(they score 0.35, kept as host-guest/encapsulation candidates) -- it just prefers 2nd-gen.

    aniline (2nd-gen, weak aromatic -> neutral closed form) -> 1.0
    tethered (rigidified enamine, hydrolytically robust)    -> 0.85
    dialkyl (1st-gen, zwitterionic/trapped)                 -> 0.35
    other                                                    -> 0.20

    [[stage.scoring.component]]
    [stage.scoring.component.DASA2ndGen]
    [[stage.scoring.component.DASA2ndGen.endpoint]]
    name = "Gen2"
    weight = 0.6
"""

__all__ = ["DASA2ndGen"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

# donor-architecture classifier lives in the notebook chem lib; mirror it here so the
# plugin has no notebook dependency (identical logic to dasa_chem).
from .dasa_common import _DASA_OPEN

# aniline AND tethered are BOTH full-credit water-switchable architectures -- do not
# rank one over the other (aniline>tethered previously collapsed the population to 100%
# aniline and lost the tethered/rigidified designs). 1st-gen dialkyl kept low (host-guest
# candidate, not banned).
_SCORE = {"aniline": 1.0, "tethered": 1.0, "dialkyl": 0.35, "other": 0.20}
_TETHER = Chem.MolFromSmarts("[NX3;R]-[CX3;R]=[CX3]")


def _architecture(mol) -> str:
    if mol is None:
        return "other"
    if mol.HasSubstructMatch(_TETHER):
        return "tethered"
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if match:
        dN = mol.GetAtomWithIdx(match[0])
        nbrs = [nb for nb in dN.GetNeighbors() if nb.GetIdx() != match[1]]
        if any(nb.GetIsAromatic() for nb in nbrs):
            return "aniline"
        if nbrs and all(nb.GetSymbol() in ("C", "H") for nb in nbrs):
            return "dialkyl"
    return "other"


@add_tag("__parameters")
@dataclass
class Parameters:
    pass


@add_tag("__component")
class DASA2ndGen:
    def __init__(self, params: Parameters):
        pass

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = [_SCORE.get(_architecture(m), 0.20) for m in mols]
        return ComponentResults([np.array(scores, dtype=float)])
