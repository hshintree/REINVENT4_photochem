"""DASAIntegrity -- chromophore-integrity GATE (0/1, no gradient).

A DASA absorbs visible light only if the donor -> triene -> acceptor pi system is
CONTINUOUS and able to lie flat. Nothing in the pipeline ever checked this, and it
is how a candidate carrying a 75 degree twist through its triene reached TD-DFT and
came back "UV" -- a broken-conjugation artefact that was then misread as a method
failure.

This is a GATE, deliberately. Planarity/conjugation is a hard requirement, not
something to trade off: a molecule whose chromophore is severed is not a worse
DASA, it is not a DASA. Gates contribute no gradient, so they can only exclude --
they cannot be optimised toward and cannot become an exploit axis. (Anything with a
genuine window -- solubility, trap escape -- is a band instead.)

Scope: TOPOLOGICAL only, so it is free in-loop. It rejects conjugation breaks that
are intrinsic to the constitution (aromatic or sp3 chain atoms, adjacent
substituted chain carbons forcing an out-of-plane twist). CONFORMATIONAL twist is a
different failure and is asserted at verification time by
dasa_chem.chain_planarity(), where a conformer search can still recover a planar
geometry -- rejecting a molecule in-loop for a bad ETKDG conformer would be wrong.
"""
from __future__ import annotations

from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from .dasa_common import chromophore_integrity

__all__ = ["DASAIntegrity"]


@add_tag("__parameters")
@dataclass
class Parameters:
    # Max heavy substituents allowed on any single triene carbon. 1 permits the
    # methylated backbones used for barrier tuning; 0 would forbid them.
    max_chain_substituents: List[int] = None

    def __post_init__(self):
        if self.max_chain_substituents is None:
            self.max_chain_substituents = [1]


@add_tag("__component")
class DASAIntegrity:
    def __init__(self, params: Parameters):
        self.max_subs = int(params.max_chain_substituents[0])
        print(f"[DASAIntegrity] topological conjugation gate "
              f"(max {self.max_subs} substituent/chain carbon)", flush=True)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = [1.0 if chromophore_integrity(m, self.max_subs) else 0.0 for m in mols]
        return ComponentResults([np.array(scores, dtype=float)])
