"""DASA water-switchability scorer (GFN2-xTB, implicit solvent).

The novel half of the DASA pivot. Water-switchable DASAs are governed by the
*charge separation* between the open and closed forms: too much charge
separation and the zwitterionic closed form becomes an irreversible thermodynamic
sink in water; too little and there is no stable coloured open form. The
Read de Alaniz "Tethered together" work (Chem. Sci. 2023) quantifies this
experimentally via the solvatochromic slope and computationally via the
isomerisation free-energy landscape in implicit solvent.

We use two robust, open-form descriptors as proxies (no fragile closed-form
generation required):

1. **Charge separation** -- the GFN2-xTB ground-state dipole of the open form in
   ALPB(water), scored with a *windowed* Gaussian. This deliberately penalises
   BOTH extremes: a near-zero dipole is not a real push-pull DASA, a very large
   dipole signals the irreversible-in-water regime.
2. **Differential solvation** -- E(ALPB water) - E(ALPB toluene). Strongly
   negative values mean water over-stabilises the charge-separated species (sink
   risk); we reward near-thermoneutral behaviour.

Final score = geometric mean of the two.

IMPORTANT: the default target/width values are physically-motivated placeholders.
They should be CALIBRATED against the literature dataset (map measured
`switches_in_water` / `solvatochromic_slope_nm` onto these xTB descriptors)
before trusting the absolute scores -- see notebooks/dasa_complete.py.

CAUTION: one embed + two xTB single points per molecule (~2-6 s/mol on CPU).
Use with small batches (batch_size ~40) in a late RL stage.

    [[stage.scoring.component]]
    [stage.scoring.component.DASASwitchability]
    [[stage.scoring.component.DASASwitchability.endpoint]]
    name = "WaterSwitch"
    weight = 0.8
    params.dipole_target_au    = 4.0    # windowed charge-separation target
    params.dipole_sigma_au     = 1.6
    params.solv_diff_target_kcal = 0.0  # want ~thermoneutral water-vs-toluene
    params.solv_diff_sigma_kcal  = 6.0
"""

__all__ = ["DASASwitchability"]
import os
from typing import List

import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from .dasa_common import is_dasa, embed_3d, xtb_properties

_HARTREE_KCAL = 627.509

# Per-molecule parallelism via THREADS (xtb releases the GIL). See comp_dasa_trap
# for the rationale -- the spawn process pool BrokenProcessPool'd inside reinvent.
_XTB_WORKERS = int(os.environ.get("DASA_XTB_WORKERS", "1"))


def _score_smiles(args):
    """Module-level (picklable) worker: SMILES + params -> switchability score."""
    smi, dip_target, dip_sigma, sd_target, sd_sigma = args
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None or not is_dasa(mol):
        return 0.0
    try:
        from xtb.interface import Calculator  # noqa: F401
    except ImportError:
        return 0.5
    mol3d = embed_3d(mol)
    if mol3d is None:
        return 0.0
    water = xtb_properties(mol3d, solvent="water")
    toluene = xtb_properties(mol3d, solvent="toluene")
    if water is None or toluene is None:
        return 0.0
    e_water, dip_water, _ = water
    e_tol, _, _ = toluene
    dip_score = np.exp(-0.5 * ((dip_water - dip_target) / dip_sigma) ** 2)
    solv_diff = (e_water - e_tol) * _HARTREE_KCAL
    sd_score = np.exp(-0.5 * ((solv_diff - sd_target) / sd_sigma) ** 2)
    return float(np.clip(np.sqrt(dip_score * sd_score), 0.0, 1.0))


@add_tag("__parameters")
@dataclass
class Parameters:
    dipole_target_au: List[float]
    dipole_sigma_au: List[float]
    solv_diff_target_kcal: List[float]
    solv_diff_sigma_kcal: List[float]


@add_tag("__component")
class DASASwitchability:
    def __init__(self, params: Parameters):
        self.dip_target = params.dipole_target_au[0]
        self.dip_sigma = params.dipole_sigma_au[0]
        self.sd_target = params.solv_diff_target_kcal[0]
        self.sd_sigma = params.solv_diff_sigma_kcal[0]

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        if _XTB_WORKERS > 1:
            from concurrent.futures import ThreadPoolExecutor
            args = [(
                Chem.MolToSmiles(m) if m is not None else "",
                self.dip_target, self.dip_sigma, self.sd_target, self.sd_sigma,
            ) for m in mols]
            with ThreadPoolExecutor(max_workers=_XTB_WORKERS) as ex:
                scores = list(ex.map(_score_smiles, args))
            return ComponentResults([np.array(scores, dtype=float)])
        return ComponentResults([np.array(
            [self._score(mol) for mol in mols], dtype=float)])

    def _score(self, mol):
        if mol is None or not is_dasa(mol):
            return 0.0
        try:
            from xtb.interface import Calculator  # noqa: F401
        except ImportError:
            # xtb absent: neutral, non-committal score so the run still proceeds
            return 0.5
        mol3d = embed_3d(mol)
        if mol3d is None:
            return 0.0
        water = xtb_properties(mol3d, solvent="water")
        toluene = xtb_properties(mol3d, solvent="toluene")
        if water is None or toluene is None:
            return 0.0
        e_water, dip_water, _ = water
        e_tol, _, _ = toluene

        # 1) windowed charge separation (dipole in water)
        dip_score = np.exp(-0.5 * ((dip_water - self.dip_target) / self.dip_sigma) ** 2)

        # 2) differential solvation (kcal/mol); near-thermoneutral rewarded
        solv_diff = (e_water - e_tol) * _HARTREE_KCAL
        sd_score = np.exp(-0.5 * ((solv_diff - self.sd_target) / self.sd_sigma) ** 2)

        return float(np.clip(np.sqrt(dip_score * sd_score), 0.0, 1.0))
