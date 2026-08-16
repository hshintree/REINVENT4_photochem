"""DASATrap — high-resolution xTB tautomer-preference score (Stage 2 refinement).

ROLE, stated plainly, because this component has changed meaning:

  * IT NO LONGER SCORES dG(closed - open). That observable inverted the objective
    (it rewarded the trapped 1st-gen alkyl 0.82 over the 2nd-gen aniline 0.21) and
    the component was disabled on 2026-07-28 because of it.
  * IT NOW SCORES  ddE = E(zwitterion) - E(keto)  in ALPB water, i.e. WHICH CLOSED
    TAUTOMER WINS. That is the quantity carrying the trap physics: the zwitterion is
    the water-locked state, the neutral keto form is the escape route.

Why the change is a re-reading rather than a fix: the old energies were not wrong.
dG(closed-open) reproduces the measured CHCl3 dark equilibria correctly (ChemSci-1
open-favoured at 86% linear; ChemSci-14 closed-favoured at 57% cyclic). What was
wrong was the assumption that "escapes the water trap" means "more open-favoured".
It does not. A 2nd-generation DASA can sit majority-closed in the dark and still
switch perfectly well, because its closed form is NEUTRAL rather than an
electrostatically locked zwitterion. The trap is a lock, not a free-energy sign.

Validated (GFN2/ALPB water, geometry-optimised, kcal/mol):

    ChemSci-1  Me2N/barbituric   1st-gen, TRAPPED     ddE  -4.0   zwitterionic
    ChemSci-14 aniline/barb      2nd-gen, escapes     ddE  +0.9   neutral keto
    indoline/barbituric          2nd-gen, escapes     ddE +11.3   strongly neutral

Correctly ordered with NO sign flip -- it is a different subtraction of the same
energies.

RELATIONSHIP TO DASATrapEscape: they measure the SAME physical quantity at
different fidelity. `dasa_common.delta_pka` estimates the tautomer preference from
substituent pKa (free, per-molecule, runs in Stage 1); this computes it from GFN2
energies on optimised geometries (~4 xTB opts/molecule, Stage 2 only). Use the cheap
one to shape the population and this one to discriminate WITHIN it -- which is also
the answer to losing the only high-resolution gradient when this stage was disabled.

BANDED, not maximised. Large positive ddE means the zwitterion is far above the keto
form, which happens when the donor is very weakly basic -- and a donor that weak
stops pushing electron density through the triene, killing the visible chromophore.
Both ends of this coordinate fail, so it gets a window.

Requires the `xtb` CLI on PATH (conda-forge `xtb`); falls back to single-point
energies if missing. Water-only by default (`use_toluene=false`).
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
from .dasa_common import (is_dasa, open_to_closed, open_to_closed_keto,
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
    # THE OBSERVABLE: E(zwitterion) - E(keto), NOT E(closed) - E(open).
    #
    # This component previously scored dG(closed - open) in ALPB water and got the
    # objective exactly backwards -- it rewarded the trapped 1st-gen alkyl (0.82)
    # over the 2nd-gen aniline escape architecture (0.21). The energies were not
    # wrong; the QUESTION was. dG(closed-open) reproduces the measured CHCl3 dark
    # equilibria correctly (ChemSci-1 open-favoured, ChemSci-14 closed-favoured),
    # but "escapes the water trap" does NOT mean "more open-favoured": a 2nd-gen
    # DASA can sit majority-closed in the dark and still switch, because its closed
    # form is NEUTRAL rather than a water-locked zwitterion. The trap is an
    # electrostatic lock, not a free-energy sign.
    #
    # The tautomer competition is what carries the physics, and xTB gets it right
    # on all three validated compounds (kcal/mol, E_zwit - E_keto):
    #     ChemSci-1  Me2N/barbituric  (1st-gen, trapped)   -4.0   zwitterionic
    #     ChemSci-14 aniline/barb     (2nd-gen, escapes)   +0.9   neutral keto
    #     indoline/barbituric         (2nd-gen, escapes)  +11.3   strongly neutral
    # Correctly ordered with NO sign flip -- it is a different subtraction.
    # This is the high-resolution, per-molecule version of dasa_common.delta_pka.
    zwit, keto = open_to_closed(mol), open_to_closed_keto(mol)
    if zwit is None or keto is None:
        return 0.0
    e_z = _energy_opt(zwit, "water")
    e_k = _energy_opt(keto, "water")
    if e_z is None or e_k is None:
        return 0.0
    ddE = (e_z - e_k) * _H_KCAL          # >0 => keto favoured => escapes the trap
    score = _band(ddE, lo, hi, w)
    eo_w = _energy_opt(mol, "water")     # open form, for the optional solvent term
    ec_w = min(e_z, e_k)
    closed_forms = [zwit, keto]
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
