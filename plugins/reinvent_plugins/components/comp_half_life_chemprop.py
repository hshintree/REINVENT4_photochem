"""Thermal half-life score using a ChemProp log10(t1/2) predictor.

Target: bistable switches with t1/2 between t_min and t_max seconds.
Scores 1 inside the window, ramps to 0 outside.

[[stage.scoring.component]]
[stage.scoring.component.HalfLifeChemProp]

[[stage.scoring.component.HalfLifeChemProp.endpoint]]
name = "HalfLife"
weight = 0.6

params.checkpoint_dir = "/path/to/chemprop_t12"
params.target_column  = "logt12"
params.logt12_min     = 3.5     # log10(3162 s) ≈ 52 min
params.logt12_max     = 9.0     # log10(10^9 s) ≈ 31 yr
params.margin         = 0.5     # log10 units
"""

__all__ = ["HalfLifeChemProp"]
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles


@add_tag("__parameters")
@dataclass
class Parameters:
    checkpoint_dir: List[str]
    target_column:  List[str]
    logt12_min:     List[float]
    logt12_max:     List[float]
    margin:         List[float]


@add_tag("__component")
class HalfLifeChemProp:
    def __init__(self, params: Parameters):
        import chemprop
        self.smiles_type  = "rdkit_smiles"
        self.logt12_min   = params.logt12_min[0]
        self.logt12_max   = params.logt12_max[0]
        self.margin       = params.margin[0]
        self.target_col   = params.target_column[0]

        args_list = [
            "--checkpoint_dir", params.checkpoint_dir[0],
            "--test_path", "/dev/null",
            "--preds_path", "/dev/null",
        ]
        with suppress_output():
            cp_args  = chemprop.args.PredictArgs().parse_args(args_list)
            cp_model = chemprop.train.load_model(args=cp_args)
        self._chemprop = (cp_model, cp_args)
        target_cols = cp_model[-1]
        self._target_idx = target_cols.index(self.target_col)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        import chemprop
        batched = [[s] for s in smilies]
        with suppress_output():
            preds = chemprop.train.make_predictions(
                model_objects=self._chemprop[0],
                smiles=batched,
                args=self._chemprop[1],
                return_invalid_smiles=True,
            )
        raw = np.array(preds, dtype=object).flatten()
        scores = []
        lo, hi, mg = self.logt12_min, self.logt12_max, self.margin
        for v in raw:
            try:
                lt = float(v)
            except (TypeError, ValueError):
                scores.append(0.0)
                continue
            if lo <= lt <= hi:
                scores.append(1.0)
            elif lo - mg <= lt < lo:
                scores.append((lt - (lo - mg)) / mg)
            elif hi < lt <= hi + mg:
                scores.append(1.0 - (lt - hi) / mg)
            else:
                scores.append(0.0)
        return ComponentResults([np.array(scores, dtype=float)])
