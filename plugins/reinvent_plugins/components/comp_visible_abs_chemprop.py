"""Visible-light absorption score using a ChemProp λ_max predictor.

Predicts π→π* λ_max (nm) and scores it with a trapezoidal function:
  score = 1 if λ_max in [vis_low, vis_high]
  score ramps linearly from 0 to 1 in the margins [vis_low-margin, vis_low]
  and from 1 to 0 in [vis_high, vis_high+margin].

[[stage.scoring.component]]
[stage.scoring.component.VisibleAbsChemProp]

[[stage.scoring.component.VisibleAbsChemProp.endpoint]]
name = "VisAbs"
weight = 0.8

params.checkpoint_dir = "/path/to/chemprop_lambda"
params.target_column  = "lambda_max"
params.vis_low        = 400.0
params.vis_high       = 650.0
params.margin         = 40.0
"""

__all__ = ["VisibleAbsChemProp"]
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
    vis_low:        List[float]
    vis_high:       List[float]
    margin:         List[float]


@add_tag("__component")
class VisibleAbsChemProp:
    def __init__(self, params: Parameters):
        import chemprop
        self.smiles_type = "rdkit_smiles"
        self.vis_low  = params.vis_low[0]
        self.vis_high = params.vis_high[0]
        self.margin   = params.margin[0]
        self.target_col = params.target_column[0]

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
        lo, hi, mg = self.vis_low, self.vis_high, self.margin
        for v in raw:
            try:
                lam = float(v)
            except (TypeError, ValueError):
                scores.append(0.0)
                continue
            if lo <= lam <= hi:
                scores.append(1.0)
            elif lo - mg <= lam < lo:
                scores.append((lam - (lo - mg)) / mg)
            elif hi < lam <= hi + mg:
                scores.append(1.0 - (lam - hi) / mg)
            else:
                scores.append(0.0)
        return ComponentResults([np.array(scores, dtype=float)])
