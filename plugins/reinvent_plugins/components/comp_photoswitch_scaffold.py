"""Photoswitch scaffold filter + custom SMARTS alerts.

Scores 1 if molecule contains a recognised photoswitch motif (azo, etc.)
and does NOT match any forbidden SMARTS.  Otherwise scores 0.

[[stage.scoring.component]]
[stage.scoring.component.PhotoswitchScaffold]

[[stage.scoring.component.PhotoswitchScaffold.endpoint]]
name = "PS_Scaffold"
weight = 1.0
"""

__all__ = ["PhotoswitchScaffold"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag


PHOTOSWITCH_SMARTS = [
    "[#6]/N=N/[#6]",     # E-azo
    "[#6]\\N=N\\[#6]",  # Z-azo
    "[#6]N=N[#6]",       # any azo
    "[#6]/C=N/[#7]",     # hydrazone
    "[#6]/N=C/[#6]",     # imine
]

FORBIDDEN_SMARTS = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]",
]


@add_tag("__parameters")
@dataclass
class Parameters:
    pass  # no configurable parameters — scaffold + forbidden SMARTS are hardcoded


@add_tag("__component", "filter")
class PhotoswitchScaffold:
    def __init__(self, params: Parameters):
        self.ps_templates = [Chem.MolFromSmarts(s) for s in PHOTOSWITCH_SMARTS]
        self.forbidden_templates = [
            t for t in [Chem.MolFromSmarts(s) for s in FORBIDDEN_SMARTS] if t
        ]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(0.0)
                continue

            # Must match a photoswitch motif
            if not any(mol.HasSubstructMatch(t) for t in self.ps_templates if t):
                scores.append(0.0)
                continue

            # Must not match forbidden
            if any(mol.HasSubstructMatch(t) for t in self.forbidden_templates):
                scores.append(0.0)
                continue

            scores.append(1.0)

        return ComponentResults([np.array(scores, dtype=float)])
