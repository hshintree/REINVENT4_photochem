"""DASATrapEscape -- cheap, in-loop, literature-grounded water-trap escape score.

WHY THIS EXISTS
---------------
The water trap is a proton-transfer problem. On 4-pi ring closure the enol OH
transfers: to the amine (-> ZWITTERION: ammonium + acceptor enolate) or to the
acceptor carbon (-> NEUTRAL keto). The zwitterion is far more polar, so water
stabilises it and locks the switch closed. Which tautomer wins is set by AMINE
BASICITY -- and, through the charge separation of the push-pull system, by
ACCEPTOR PULL.

Peterson / Read de Alaniz (ionic-character study) measured exactly this: first-
and third-generation architectures show a higher zwitterionic resonance
contribution of the open form AND a zwitterionic closed form, while the
second-generation (aryl-amine donor) has a less charge-separated open form and a
NEUTRAL closed form.

THE COORDINATE: dpKa = pKa(amine conjugate acid) - pKa(carbon acid)
-------------------------------------------------------------------
Proton transfer is the mechanism, so the pKa DIFFERENCE is the physical variable.
It is a genuine PUSH-PULL descriptor -- it moves continuously with every
substituent on BOTH ends, giving per-molecule resolution instead of a class
lookup. Scored on the ASYMMETRIC band [0.0, 3.5], w_lo 2.5 / w_hi 0.8:

    ChemSci-14 aniline/barb     588 nm, SWITCHES     dpKa +1.37   0.986
    NatComm-10 indoline/barb    615 nm, SWITCHES     dpKa +0.89   0.943
    azole-donor candidate       UNKNOWN              dpKa -0.81   0.695
    4-CF3-anilino/barb          EWG aniline          dpKa -0.97   0.670
    aminotriazine/barb          donor far too weak   dpKa -4.51   0.235
    ChemSci-1  Me2N/barb        1st-gen, TRAPPED     dpKa +6.69   0.028
    Me2N/CF3-pyrazolone         3rd-gen, TRAPPED     dpKa +7.20   0.015

The two MEASURED switchers sit at the top; the two measured TRAPPED compounds are
firmly rejected; the unknown weak-donor region is DISFAVOURED BUT NOT BANNED,
pending the TD-DFT solvatochromic-slope measurement that will set the real floor.

NOTE isoindoline (573 nm) scores low too: its N is a benzylic ALKYL amine
(pKaH ~11), so it is 1st-gen-like and would be water-trapped. That is the correct
answer for a WATER-trap-escape term -- it is not a judgement on the molecule.

BANDED, NOT MINIMISED
---------------------
Both extremes break the molecule. Too much push-pull -> charge-separated ->
zwitterionic closed form -> water-locked. Too little -> the chromophore stops
absorbing in the visible (the UV failure the loop already found once by acylating
the donor). A band has its optimum in the INTERIOR, so the objective cannot run
away. Standing rule: hard requirements are gates (no gradient), anything with a
window is a band.

WHY CONTINUOUS RESOLUTION MATTERS: the previous version binned basicity into ~3
classes, so once the population became aryl it returned an IDENTICAL 0.836 for
every molecule. With trap escape flat, colour and integrity as gates, solubility
became the only gradient -- and solubility rewards N-rich heteroaromatics, so the
donors drifted to azoles/azines with no literature precedent. A saturated
objective does not merely stop helping; it hands the search to whatever is left.

Cost: pure RDKit, no 3D, no xTB. Runs in Stage 1 so push-pull shapes the
population from step 1 for free.
"""
from __future__ import annotations

from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from .dasa_common import is_dasa, delta_pka

__all__ = ["DASATrapEscape"]


def _band(x: float, lo: float, hi: float, w_lo: float, w_hi: float) -> float:
    """ASYMMETRIC double sigmoid, normalised so the plateau is 1.0.

    The two edges are deliberately NOT equally sharp, because we know one of them
    and not the other:

      HIGH side (large dpKa -> zwitterionic -> water-trapped): FIRM. We have
        measured compounds here -- ChemSci-1 at +6.69 and Me2N/CF3-pyrazolone at
        +7.20 are known to be trapped -- so a narrow width is justified.

      LOW side (small/negative dpKa -> donor too weak -> push-pull dies): GENTLE,
        a marginally decreasing tail rather than a cliff. We do NOT yet know how
        weak a donor can get before the chromophore stops absorbing in the visible.
        A sharp cutoff here would be inventing a threshold and would silently ban a
        donor class on no evidence. A slow decline disfavours weak donors without
        excluding them, and lets the TD-DFT solvatochromic-slope measurement set the
        real floor later.

    Normalising by the plateau keeps the scale comparable to the other components
    under the geometric mean -- an un-normalised asymmetric band peaks well below
    1.0 and would quietly under-weight this objective.
    """
    def _s(v):
        return 1.0 / (1.0 + np.exp(-v))
    raw = _s((x - lo) / w_lo) * _s(-(x - hi) / w_hi)
    mid = 0.5 * (lo + hi)
    peak = _s((mid - lo) / w_lo) * _s(-(mid - hi) / w_hi)
    return float(raw / peak) if peak > 1e-9 else 0.0


@add_tag("__parameters")
@dataclass
class Parameters:
    # Window on dpKa = pKa(amine conjugate acid) - pKa(carbon acid).
    # Centred on the MEASURED 2nd-generation compounds:
    #     NatComm-10 indoline/barbituric   (615 nm, switches)  dpKa +0.89
    #     ChemSci-14 4-MeO-aniline/barb    (588 nm, switches)  dpKa +1.37
    # and rejecting both failure directions:
    #     ChemSci-1 Me2N/barbituric  (1st-gen, water-trapped)  dpKa +6.69
    #     Me2N/CF3-pyrazolone        (3rd-gen, most trapped)   dpKa +7.20
    #     aminotriazine/barbituric   (donor too weak, no push) dpKa -4.51
    # The LOWER edge is provisional: we do not yet know how weak a donor can get
    # before the push-pull chromophore stops absorbing in the visible. It is set
    # just below the measured range so it penalises without banning, and should be
    # revised from the TD-DFT result on the azole-donor candidates.
    dpka_lo: List[float] = None
    dpka_hi: List[float] = None
    dpka_width_lo: List[float] = None     # GENTLE: unknown weak-donor cliff
    dpka_width_hi: List[float] = None     # FIRM: measured trapped compounds

    def __post_init__(self):
        if self.dpka_lo is None:
            self.dpka_lo = [0.0]
        if self.dpka_hi is None:
            self.dpka_hi = [3.5]
        if self.dpka_width_lo is None:
            self.dpka_width_lo = [2.5]
        if self.dpka_width_hi is None:
            self.dpka_width_hi = [0.8]


@add_tag("__component")
class DASATrapEscape:
    def __init__(self, params: Parameters):
        self.lo = float(params.dpka_lo[0])
        self.hi = float(params.dpka_hi[0])
        self.w_lo = float(params.dpka_width_lo[0])
        self.w_hi = float(params.dpka_width_hi[0])
        print(f"[DASATrapEscape] dpKa push-pull band[{self.lo},{self.hi}] "
              f"w_lo={self.w_lo} (gentle, unknown cliff) w_hi={self.w_hi} "
              f"(firm, measured) — continuous, no xTB", flush=True)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        out = []
        for mol in mols:
            if mol is None or not is_dasa(mol):
                out.append(0.0)
                continue
            out.append(_band(delta_pka(mol), self.lo, self.hi, self.w_lo, self.w_hi))
        return ComponentResults([np.array(out, dtype=float)])
