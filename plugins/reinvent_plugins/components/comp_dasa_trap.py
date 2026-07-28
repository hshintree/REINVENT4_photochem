"""DASA anti-trapping score (GFN2-xTB, implicit solvent) — the real switchability metric.

Water-switchable DASAs fail primarily because the polar closed cyclopentenone
ZWITTERION is thermodynamically favoured in water: the open<->closed equilibrium
locks closed ("dark switching"). This component computes that equilibrium directly:

    dE_water = E(closed_zwitterion, ALPB water) - E(open, ALPB water)   [kcal/mol]

  * dE_water strongly negative  -> closed deeply favoured in water = TRAPPED (bad).
  * dE_water near zero / positive -> open form accessible in water = switchable.

GEOMETRY-OPTIMISED (the important upgrade): each form is GFN2-xTB `--opt`-relaxed
in its solvent before the energy is read. The previous version used SINGLE POINTS on
MMFF geometries, which was noisy enough to INVERT the truth -- its "anti-trap winners"
turned out (on optimised geometries) to be the MOST trapped. Validated against a
known anchor: first-gen DMA-Meldrum (experimentally >99% closed in water) optimises to
dE_water ~ -5 kcal/mol, exactly the trapped regime.

BANDED, not maximised: the score is a WINDOW on dE_water (double sigmoid), rewarding
the switchable regime (roughly [dE_lo, dE_hi], default [-2, +18] kcal/mol -- i.e.
clearly less trapped than the -5 first-gen anchor, without an unbounded "more positive
is always better" pressure that the RL would exploit by inventing unphysical closed
forms). The colour+donor gate (DASAColor) independently guarantees a real chromophore,
so anti-trap pressure can no longer cheat by killing the donor.

Water-only by default (`use_toluene=false`): 2 xTB opts/molecule (open+closed in water).
Set use_toluene=true to also reward a controlled water-vs-toluene response (4 opts/mol,
slower -- reserve for a short stage or the DFT verifier). Requires the `xtb` CLI on PATH
(conda-forge `xtb`); falls back to the old single-point energy if it is missing.

    [[stage.scoring.component]]
    [stage.scoring.component.DASATrap]
    [[stage.scoring.component.DASATrap.endpoint]]
    name = "AntiTrap"
    weight = 0.6
    params.dE_lo_kcal    = -2.0    # below this = trapped (anchor: first-gen ~ -5)
    params.dE_hi_kcal    = 18.0    # above this = suspiciously open-only (soft cap)
    params.dE_width_kcal = 4.0     # sigmoid shoulder width
    params.use_toluene   = false   # water-only (fast). true = + water-vs-toluene term
"""

__all__ = ["DASATrap"]
import os
import subprocess
import tempfile
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from .dasa_common import (is_dasa, open_to_closed, open_to_closed_neutral,
                          embed_3d, xtb_properties)

_H_KCAL = 627.509
_XTB_WORKERS = int(os.environ.get("DASA_XTB_WORKERS", "1"))
_MAX_HEAVY = 48          # opt on big floppy RL artifacts is what makes Stage 2 crawl
_OPT_TIMEOUT = int(os.environ.get("DASA_XTB_OPT_TIMEOUT", "150"))  # s/opt; roomy vs cycle-capped work, still bounded
_OPT_CYCLES = int(os.environ.get("DASA_XTB_OPT_CYCLES", "20"))      # cap geom-opt iterations (bounds runtime ~60s/opt)
_LOGGED = False

import shutil
_XTB_CLI = shutil.which("xtb")   # native optimiser; None -> single-point fallback


def _xyz_from_mol(mol):
    """MMFF-embedded RDKit mol -> (xyz string, total charge) or (None, None)."""
    m3d = embed_3d(mol)
    if m3d is None:
        return None, None
    conf = m3d.GetConformer()
    lines = [str(m3d.GetNumAtoms()), ""]
    for a in m3d.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines) + "\n", Chem.GetFormalCharge(mol)


def _energy_opt(mol, solvent):
    """GFN2-xTB geometry-OPTIMISED energy (Hartree) in ALPB `solvent`, via the xtb CLI.
    Falls back to a single-point on the MMFF geometry if the CLI is unavailable."""
    if _XTB_CLI is None:
        m3d = embed_3d(mol)
        if m3d is None:
            return None
        r = xtb_properties(m3d, solvent=solvent)
        return None if r is None else r[0]
    xyz, chg = _xyz_from_mol(mol)
    if xyz is None:
        return None
    d = tempfile.mkdtemp(prefix="dasatrap_")
    try:
        with open(f"{d}/m.xyz", "w") as f:
            f.write(xyz)
        cmd = [_XTB_CLI, "m.xyz", "--opt", "loose", "--gfn", "2",
               "--cycles", str(_OPT_CYCLES),
               "--alpb", solvent, "--chrg", str(chg), "--uhf", "0"]
        # fail-fast: a DASA opt that needs >90s is pathological (floppy tail / non-
        # convergence). Time it out -> return None -> the molecule scores 0, rather
        # than letting one stuck opt stall a whole 16-thread scoring batch.
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=_OPT_TIMEOUT,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        e = None
        for line in r.stdout.splitlines():
            if "TOTAL ENERGY" in line:
                try:
                    e = float(line.split()[-3])
                except Exception:
                    pass
        return e
    except Exception:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _band(x, lo, hi, w):
    """~1 inside [lo, hi] with soft sigmoid shoulders of scale w; ->0 well outside."""
    return (1.0 / (1.0 + np.exp(-(x - lo) / w))) * (1.0 / (1.0 + np.exp((x - hi) / w)))


def _score_smiles(args):
    """Module-level (picklable/threadable) worker: SMILES + params -> anti-trap score."""
    smi, lo, hi, w, use_tol = args
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None or not is_dasa(mol):
        return 0.0
    if mol.GetNumHeavyAtoms() > _MAX_HEAVY:
        return 0.0
    # The REAL closed form is the lower-energy of the zwitterion (1st/3rd-gen) and
    # the neutral keto tautomer (2nd-gen, only for N-H/aniline donors). Using min()
    # stops us from mis-scoring 2nd-gen water-switchers against the wrong (too-high,
    # over-trapped) zwitterion -- the fix demanded by the generation literature.
    closed_forms = [c for c in (open_to_closed(mol), open_to_closed_neutral(mol))
                    if c is not None]
    if not closed_forms:
        return 0.0
    eo_w = _energy_opt(mol, "water")
    if eo_w is None:
        return 0.0
    ec_opts = [_energy_opt(c, "water") for c in closed_forms]
    ec_opts = [e for e in ec_opts if e is not None]
    if not ec_opts:
        return 0.0
    ec_w = min(ec_opts)                      # lowest-energy closed tautomer = real closed state
    dE_water = (ec_w - eo_w) * _H_KCAL
    score = _band(dE_water, lo, hi, w)
    if use_tol:
        eo_t = _energy_opt(mol, "toluene")
        ect = [_energy_opt(c, "toluene") for c in closed_forms]
        ect = [e for e in ect if e is not None]
        ec_t = min(ect) if ect else None
        if None not in (eo_t, ec_t) and ec_t is not None:
            # a real switch gets MORE closed-favouring with polarity but shouldn't
            # collapse; reward a modest (not catastrophic) water-minus-toluene drop.
            dshift = dE_water - (ec_t - eo_t) * _H_KCAL
            score *= np.exp(-0.5 * ((dshift + 3.0) / 8.0) ** 2)
    return float(np.clip(score, 0.0, 1.0))


@add_tag("__parameters")
@dataclass
class Parameters:
    dE_lo_kcal: List[float]
    dE_hi_kcal: List[float]
    dE_width_kcal: List[float]
    use_toluene: List[bool]


@add_tag("__component")
class DASATrap:
    def __init__(self, params: Parameters):
        self.lo = params.dE_lo_kcal[0]
        self.hi = params.dE_hi_kcal[0]
        self.w = params.dE_width_kcal[0]
        self.use_tol = bool(params.use_toluene[0])

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        global _LOGGED
        args = [(Chem.MolToSmiles(m) if m is not None else "",
                 self.lo, self.hi, self.w, self.use_tol) for m in mols]
        if not _LOGGED:
            mode = "xtb --opt" if _XTB_CLI else "SINGLE-POINT fallback (no xtb CLI!)"
            print(f"[DASATrap] {mode}, water{'+toluene' if self.use_tol else '-only'}, "
                  f"band[{self.lo},{self.hi}] w{self.w}, {_XTB_WORKERS} workers", flush=True)
            _LOGGED = True
        if _XTB_WORKERS > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=_XTB_WORKERS) as ex:
                scores = list(ex.map(_score_smiles, args))
        else:
            scores = [_score_smiles(a) for a in args]
        return ComponentResults([np.array(scores, dtype=float)])
