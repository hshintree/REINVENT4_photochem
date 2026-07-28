"""DASA visible-colour gate.

Returns 1.0 if the molecule's acceptor is a canonical, known-VISIBLE carbon acid
(Meldrum's, barbituric, thiobarbituric, indandione, pyrazolone, isoxazolone,
pyrazolidinedione, oxindole -- all absorbing ~545-670 nm), else 0.0.

WHY THIS EXISTS: a DASA must be a *visible-light* photoswitch. The RL previously
maximised anti-trap (low charge separation) by inventing weak/unusual acceptors
that absorb in the UV (a 311 nm example from DFT). No cheap xTB property predicts
the optical lambda_max (ground-state gap and dipole both failed to separate UV
from visible), but the ACCEPTOR IDENTITY does -- it sets the colour (Helmy 2014).
Used as a hard gate in the geometric mean, this forces the generator to keep a
real chromophore while it optimises anti-trap + solubility, pushing it toward the
weak-but-visible acceptors (methyl-pyrazolone/isoxazolone) that the literature
uses for aqueous switching.

    [[stage.scoring.component]]
    [stage.scoring.component.DASAColor]
    [[stage.scoring.component.DASAColor.endpoint]]
    name = "Color"
    weight = 1.0
"""

__all__ = ["DASAColor"]
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from .dasa_common import has_canonical_acceptor, has_visible_donor


@add_tag("__parameters")
@dataclass
class Parameters:
    pass


@add_tag("__component")
class DASAColor:
    def __init__(self, params: Parameters):
        pass

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        # colour requires BOTH a known-visible acceptor AND a genuine amine donor.
        # The acceptor sets the hue; the donor must actually push (an acylated/N-O
        # donor collapses the chromophore to the UV -- TD-DFT-confirmed exploit).
        scores = [1.0 if (has_canonical_acceptor(m) and has_visible_donor(m)) else 0.0
                  for m in mols]
        return ComponentResults([np.array(scores, dtype=float)])
