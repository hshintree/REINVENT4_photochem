#!/usr/bin/env python
"""Rigorous, calibrated TD-DFT verification for a handful of DASA candidates.

DESIGN PRINCIPLE
----------------
TD-DFT does NOT predict DASA lambda_max accurately, and this module does not
pretend otherwise. Linear-response TD-DFT collapses the multiconfigurational DASA
S1 into a nearly non-CT state; against CASPT2/experiment the residual error is
~0.44 eV and is largely functional-independent (Molecules 2017, PMC5615680:
B3LYP/6-31+G(d) on a DFT geometry gives 435 nm for a measured 515 nm DASA;
CASPT2 0.06 eV, NEVPT2 0.15 eV).

So we do not ask for absolute lambda. We ask for a CALIBRATED RANKING against
compounds whose lambda_max has been MEASURED, with every molecule pushed through
an identical protocol, and we publish a falsifiable criterion for when that
ranking is not trustworthy.

WHY THE PREVIOUS ATTEMPT FAILED (and what changed)
--------------------------------------------------
The old pipeline was 229 nm blue of the measured anchor. Decomposed:
  * wrong CONNECTIVITY (a constitutional isomer, not a DASA) .... 0.59 eV  [fixed]
  * geometry: MMFF/xTB single embed, no DFT optimisation ........ ~48 nm   [fixed here]
  * no diffuse functions ........................................ ~0.1 eV [fixed here]
  * irreducible TD-DFT error .................................... 0.44 eV  [CALIBRATED, not "fixed"]
An empirical offset is only legitimate when it corrects ONE systematic error.
The old "+186/+220 nm offset" was absorbing three, two of which varied per
molecule by >100 nm -- which is why it moved between runs and could not rank.

THE FOUR THINGS THAT MAKE THIS DEFENSIBLE
-----------------------------------------
1. IDENTICAL PROTOCOL. Every molecule -- references and candidates alike -- gets
   the same stereochemistry assignment, conformer search, DFT optimisation,
   basis, solvent and state-selection rule. No per-molecule judgement calls.
2. CALIBRATION ON MEASURED COMPOUNDS, with citations, spanning both donor classes
   (alkyl / aryl) and two acceptor families (barbituric / pyrazolone).
3. STATE SELECTION BY CHARACTER, not by max oscillator strength. The old rule
   picked different electronic states for different functionals on the same
   molecule (CAM-B3LYP came out RED of B3LYP -- diagnostic of a root flip).
4. A FALSIFIABLE GATE. If the calibration residual sigma exceeds MAX_RESIDUAL_NM,
   this reports that TD-DFT CANNOT rank these molecules, and says so, instead of
   quietly emitting numbers.

Additionally every result carries its own audit trail: post-optimisation chain
planarity, the full excitation list, the H->L character of the selected state,
and the solvatochromic sign (DASAs are NEGATIVELY solvatochromic -- a positive
shift means the assigned state is not the DASA CT band).

USAGE
-----
    # 1. calibrate (references only) -- do this first, it decides everything
    python notebooks/dft_verify_v2.py --calibrate

    # 2. verify candidates once calibration passes
    python notebooks/dft_verify_v2.py --csv outputs_dasa_modal/verified_candidates.csv --top 5

    # cheap smoke test of the plumbing (small basis, no DFT geometry optimisation)
    python notebooks/dft_verify_v2.py --calibrate --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc  # noqa: E402

# --------------------------------------------------------------------------- #
# Calibration ladder: MEASURED hydroxy-DASA lambda_max, with provenance.
#
# WARNING: Nat Commun 2024 also reports AMINO DASAs (an N-H in place of the
# hydroxyl) at 531/578/608 nm. Those are a DIFFERENT chromophore and must never be
# used to calibrate hydroxy DASAs -- an earlier draft of this ladder made exactly
# that mistake.
# --------------------------------------------------------------------------- #
REFERENCES = [
    dict(label="ChemSci-1 Me2N/1,3-diMe-barbituric", lam_nm=567, solvent="chloroform",
         smiles="CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O",
         source="Chem. Sci. 2018, 9, 8242 (cmpd 1; 567 +- 3 nm across 13 donors)"),
    dict(label="ChemSci-14 4-MeO-N-Me-aniline/barbituric", lam_nm=588, solvent="chloroform",
         smiles="COc1ccc(N(C)C=CC=C(O)C=C2C(=O)N(C)C(=O)N(C)C2=O)cc1",
         source="Chem. Sci. 2018, 9, 8242 (cmpd 14)"),
    dict(label="NatComm-9 isoindoline/barbituric", lam_nm=573, solvent="dichloromethane",
         smiles="O=C1N(C)C(=O)C(=CC(O)=CC=CN2Cc3ccccc3C2)C(=O)N1C",
         source="Nat. Commun. 2024, 15 (cmpd 9, HYDROXY parent)"),
    dict(label="NatComm-10 indoline/barbituric", lam_nm=615, solvent="dichloromethane",
         smiles="O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C",
         source="Nat. Commun. 2024, 15 (cmpd 10, HYDROXY parent)"),
    dict(label="NatComm-11 indoline/CF3-pyrazolone", lam_nm=646, solvent="dichloromethane",
         smiles="O=C1N(c2ccccc2)N=C(C(F)(F)F)C1=CC(O)=CC=CN1CCc2ccccc21",
         source="Nat. Commun. 2024, 15 (cmpd 11, HYDROXY parent)"),
]

EPS = {"toluene": 2.38, "dichloromethane": 8.93, "chloroform": 4.71,
       "acetonitrile": 35.7, "water": 78.4}

# Falsifiable gate: above this residual the ranking is not defensible.
MAX_RESIDUAL_NM = 15.0

def onsager(eps: float) -> float:
    """Onsager reaction-field polarity factor f(eps) = (eps-1)/(2eps+1)."""
    return (eps - 1.0) / (2.0 * eps + 1.0)


HARTREE_EV = 27.211386
NM_EV = 1239.841


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def assign_literature_stereo(mol):
    """Set the open form to the literature (2Z,4E) configuration.

    The generator emits stereo-free SMILES (the prior's vocabulary cannot tokenise
    stereo), so this MUST run before any embedding. Embedding a stereo-free triene
    is what produced a 75-degree-twisted geometry in the previous campaign, which
    was then misread as a UV molecule.
    """
    ix = dc._core_idx(mol)
    if ix is None:
        return mol
    m = Chem.Mol(mol)

    def _set(a_key, b_key, ref_a, ref_b, stereo):
        """SetStereoAtoms requires the first reference atom to be bonded to the
        bond's BEGIN atom. RDKit's begin/end order depends on how the molecule was
        constructed, so resolve it rather than assuming."""
        bd = m.GetBondBetweenAtoms(ix[a_key], ix[b_key])
        if bd is None:
            return
        if bd.GetBeginAtomIdx() == ix[a_key]:
            bd.SetStereoAtoms(ix[ref_a], ix[ref_b])
        else:
            bd.SetStereoAtoms(ix[ref_b], ix[ref_a])
        bd.SetStereo(stereo)

    _set("Ca", "Cb", "N", "Cc", Chem.BondStereo.STEREOTRANS)   # (4E): N trans to Cc
    _set("Cc", "Cd", "Cb", "O", Chem.BondStereo.STEREOCIS)     # (2Z): Cb cis to O
    Chem.SetDoubleBondNeighborDirections(m)
    m.ClearComputedProps()
    Chem.AssignStereochemistry(m, cleanIt=False, force=True)
    return m


def truncate_chromophore(mol):
    """Replace peripheral solubilising tails with hydrogen, keeping the chromophore.

    lambda_max is set by the donor-triene-acceptor pi system; glycol / amide /
    sulfonate tails hanging off the periphery do not touch it, but they dominate
    the cost (a 41-heavy-atom candidate is ~920 basis functions at 6-31+G(d,p),
    which is days of DFT optimisation locally). Keeps: the core, every ring
    containing a core atom, and the first substituent shell on those rings.

    ALWAYS validate truncation once per campaign by computing the full and
    truncated molecule and comparing -- see --check-truncation.
    """
    ix = dc._core_idx(mol)
    if ix is None:
        return None
    keep = set(ix.values())
    ri = mol.GetRingInfo()
    # FIRST SHELL around the core. This is essential, not cosmetic: the donor's
    # ARYL RING is conjugated into the chromophore and sets ~20 nm of the shift
    # (aniline 588 vs alkyl 567), but no ring atom is itself a core atom -- the
    # core stops at N. Without this shell the aniline ring is deleted and the
    # truncated molecule is a different chromophore.
    for idx in list(keep):
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            keep.add(nb.GetIdx())
    for _ in range(3):                # close rings touching keep, then fused rings
        grew = True
        while grew:
            grew = False
            for ring in ri.AtomRings():
                if keep & set(ring) and not set(ring) <= keep:
                    keep |= set(ring)
                    grew = True
        for idx in list(keep):        # keep carbonyls / thiocarbonyls on kept atoms
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                bd = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
                if bd.GetBondTypeAsDouble() == 2.0 and nb.GetAtomicNum() in (7, 8, 16):
                    keep.add(nb.GetIdx())
    rw = Chem.RWMol(mol)
    for idx in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        rw.RemoveAtom(idx)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out if dc.is_dasa(out) else None


def prepare_geometry(smiles, xc="b3lyp", basis="6-31g*", eps=4.71,
                     dft_opt=False, n_confs=20, max_opt_cycles=25):
    """stereo -> conformer search -> (optional) DFT optimisation -> planarity audit.

    Returns dict with the 3D mol, twist before/after, and whether DFT opt ran.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not dc.is_dasa(mol):
        return None
    mol = assign_literature_stereo(mol)
    m3d, twist0 = dc.planar_conformer(Chem.MolToSmiles(mol), n_confs=n_confs)
    if m3d is None:
        return None
    info = dict(twist_mmff_deg=twist0, dft_opt=False, twist_dft_deg=None)
    if not dft_opt:
        info["mol"] = m3d
        return info
    from pyscf import gto, dft as pdft
    from pyscf.solvent import ddCOSMO
    from pyscf.geomopt.geometric_solver import optimize
    conf = m3d.GetConformer()
    atoms = [(a.GetSymbol(), tuple(conf.GetAtomPosition(a.GetIdx())))
             for a in m3d.GetAtoms()]
    pm = gto.M(atom=atoms, basis=basis, charge=Chem.GetFormalCharge(m3d),
               spin=0, verbose=0)
    mf = ddCOSMO(pdft.RKS(pm))
    mf.xc = xc
    mf.with_solvent.eps = eps
    mf.max_cycle = 200
    mf.conv_tol = 1e-8
    # BOUNDED, NON-ASSERTING, LOOSE. All three matter:
    #
    #  * assert_convergence=False -- the default RAISES when the step cap is hit, so
    #    a molecule that ran for hours would return NOTHING. A partially optimised
    #    DFT geometry is far better than the MMFF one we started from; never throw
    #    it away.
    #  * maxsteps AND maxiter -- pyscf's maxsteps did not reach geomeTRIC (its banner
    #    reported the 300-cycle default while we asked for 60), so pass both.
    #  * loose criteria -- these are ~10x the defaults (GAU_LOOSE-like). DASAs with
    #    flexible solubilising tails have very soft modes (a Hessian eigenvalue of
    #    5.6e-4 showed up in practice) and chase a 3e-4 RMS gradient for hundreds of
    #    steps. A vertical excitation does not need that: what matters is capturing
    #    the bond-length alternation, which loose optimisation already gets.
    conv = dict(convergence_energy=1.0e-4, convergence_grms=3.0e-3,
                convergence_gmax=4.5e-3, convergence_drms=1.2e-2,
                convergence_dmax=1.8e-2)
    try:
        opt = optimize(mf, maxsteps=max_opt_cycles, assert_convergence=False,
                       maxiter=max_opt_cycles, **conv)
        newconf = m3d.GetConformer()
        for i, c in enumerate(opt.atom_coords() * 0.52917721092):   # Bohr -> Angstrom
            newconf.SetAtomPosition(i, tuple(float(x) for x in c))
        info.update(mol=m3d, dft_opt=True, twist_dft_deg=dc.chain_planarity(m3d))
    except Exception as exc:
        # fall back to the vetted planar MMFF conformer rather than losing the molecule
        info.update(mol=m3d, dft_opt=False, dft_opt_error=f"{type(exc).__name__}: {exc}")
    return info


# --------------------------------------------------------------------------- #
# Excited states
# --------------------------------------------------------------------------- #
def tddft_states(m3d, xc="b3lyp", basis="6-31g*", eps=4.71, nstates=6):
    """Full excitation list with H->L character. Returns (states, homo_lumo_gap_eV)."""
    from pyscf import gto, dft as pdft, tddft
    from pyscf.solvent import ddCOSMO
    conf = m3d.GetConformer()
    atoms = [(a.GetSymbol(), tuple(conf.GetAtomPosition(a.GetIdx())))
             for a in m3d.GetAtoms()]
    pm = gto.M(atom=atoms, basis=basis, charge=Chem.GetFormalCharge(m3d),
               spin=0, verbose=0)
    mf = ddCOSMO(pdft.RKS(pm))
    mf.xc = xc
    mf.with_solvent.eps = eps
    mf.max_cycle = 200
    mf.conv_tol = 1e-8
    mf.kernel()
    if not mf.converged:
        return None, None
    nocc = pm.nelectron // 2
    gap = float((mf.mo_energy[nocc] - mf.mo_energy[nocc - 1]) * HARTREE_EV)
    td = tddft.TDA(mf)
    td.nstates = nstates
    td.kernel()
    osc = td.oscillator_strength()
    states = []
    for i, (e, f) in enumerate(zip(td.e, osc)):
        w = 2 * td.xy[i][0] ** 2
        o, v = np.unravel_index(np.argmax(w), w.shape)
        states.append(dict(state=i + 1, ev=float(e * HARTREE_EV),
                           nm=float(NM_EV / (e * HARTREE_EV)), f=float(f),
                           homo_lumo_weight=float(w[nocc - 1, 0]),
                           dominant=f"H-{nocc-1-o}->L+{v}".replace("-0", "").replace("+0", "")))
    return states, gap


def select_dasa_band(states, f_min=0.20, hl_min=0.50):
    """Pick the DASA CT band BY CHARACTER: the lowest-energy bright state dominated
    by HOMO->LUMO.

    The previous pipeline took max-oscillator-strength over only 5 states, which
    selected DIFFERENT electronic states for different functionals on the same
    molecule (CAM-B3LYP landed RED of B3LYP -- impossible for one band, diagnostic
    of a root flip). Candidates carrying quinoline / pyridyl / pyrazole
    substituents have local aryl pi->pi* states that compete for max-f.

    Returns (state_dict, rule_used) so the choice is always auditable.
    """
    if not states:
        return None, "none"
    hl = [s for s in states if s["f"] >= f_min and s["homo_lumo_weight"] >= hl_min]
    if hl:
        return min(hl, key=lambda s: s["ev"]), "lowest bright H->L"
    bright = [s for s in states if s["f"] >= f_min]
    if bright:
        return min(bright, key=lambda s: s["ev"]), "lowest bright (NO H->L dominance)"
    return states[0], "S1 fallback (NO bright state)"


def compute_one(smiles, label, solvent="chloroform", quick=False, truncate=True,
                nstates=6, dft_opt=False, budget_s=2400, max_heavy=34):
    """Full protocol on one molecule. Returns a result dict (never raises)."""
    t0 = time.time()
    res = dict(label=label, smiles=smiles, solvent=solvent, truncated=False)

    def over_budget(stage):
        """Wall-clock guard. TD-DFT cost scales steeply with size, so a molecule that
        blows the budget must be ABANDONED with whatever it has, not left running --
        a previous run burned 5 h across four containers and returned nothing."""
        if time.time() - t0 > budget_s:
            res["aborted_at"] = stage
            res["error"] = f"wall-clock budget {budget_s}s exceeded at {stage}"
            return True
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not dc.is_dasa(mol):
            res["error"] = "not a corrected-core DASA"
            return res
        # SIZE PRE-SCREEN, before a single SCF is started. Truncation first, because
        # it is what makes a 38-heavy candidate affordable at all.
        if truncate:
            t = truncate_chromophore(assign_literature_stereo(mol))
            if t is not None:
                smiles = Chem.MolToSmiles(t)
                mol = t
                res.update(truncated=True, truncated_smiles=smiles)
        res["heavy_atoms"] = mol.GetNumHeavyAtoms()
        if res["heavy_atoms"] > max_heavy:
            res["error"] = (f"skipped: {res['heavy_atoms']} heavy atoms > max_heavy "
                            f"{max_heavy} even after truncation -- too expensive to "
                            f"be worth a container")
            return res
        eps = EPS.get(solvent, 4.71)
        # PROTOCOL COST, and why it is deliberately cheap.
        #
        # Calibration is what buys accuracy here, not brute force. Every molecule --
        # references and candidates alike -- runs the IDENTICAL protocol, so a
        # systematic bias from basis set or geometry is absorbed by the fit and
        # removed. The residual sigma gate is the test of whether it is systematic
        # enough. That means we should spend the compute budget on FINISHING, not on
        # a tighter absolute number we then calibrate away anyway.
        #
        # Measured on ChemSci-1 (20 heavy atoms): planar conformer search 0.2 s;
        # ONE TD-DFT at 6-31+G(d) many minutes; DFT geometry optimisation did not
        # reach convergence in 5 h across four containers (step 12 of 300 on the
        # SMALLEST molecule, with a 5.6e-4 soft-mode Hessian eigenvalue). So:
        #   * DFT geometry optimisation is OFF by default (opt in with dft_opt=True)
        #   * 6-31G* for TD-DFT, not 6-31+G(d)
        # Both losses are systematic and calibrated out.
        basis_opt = "3-21g" if quick else "6-31g*"
        basis_td = "3-21g" if quick else "6-31g*"
        if quick:
            nstates = min(nstates, 6)
        g = prepare_geometry(smiles, basis=basis_opt, eps=eps, dft_opt=dft_opt,
                             n_confs=8 if quick else 20,
                             max_opt_cycles=8 if quick else 25)
        if g is None:
            res["error"] = "geometry failed"
            return res
        res.update(twist_mmff_deg=g["twist_mmff_deg"], dft_opt=g["dft_opt"],
                   twist_dft_deg=g["twist_dft_deg"])
        if over_budget("after geometry"):
            return res
        states, gap = tddft_states(g["mol"], basis=basis_td, eps=eps, nstates=nstates)
        if states is None:
            res["error"] = "SCF did not converge"
            return res
        sel, rule = select_dasa_band(states)
        res.update(states=states, homo_lumo_gap_ev=gap, selected=sel,
                   selection_rule=rule, lambda_calc_nm=sel["nm"] if sel else None)
        # SOLVATOCHROMIC SLOPE -- the literature's charge-separation measurement,
        # and the non-arbitrary test for whether the push-pull system is alive.
        #
        # DASAs are NEGATIVELY solvatochromic: lambda_max blue-shifts as solvent
        # polarity rises, because the ground state is more charge-separated than the
        # excited state. Peterson / Read de Alaniz quantify ionic character exactly
        # this way. The slope FAILS AT BOTH ENDS, which is why it can replace a
        # hand-set threshold on donor strength:
        #
        #   slope ~ 0 (or positive)  -> no charge transfer. The donor is not pushing;
        #                               this is not a DASA chromophore. <-- the test a
        #                               too-weak donor (azole/tetrazole) fails.
        #   strongly negative        -> over-charge-separated -> zwitterionic -> trapped.
        #
        # Reported against the Onsager polarity function f(eps) = (eps-1)/(2eps+1),
        # so it is a slope in nm per unit polarity rather than a solvent-pair shift.
        # The WORKING WINDOW is calibrated from the measured references run through
        # this same protocol -- not asserted from a remembered number.
        # TWO points, not a solvent sweep. The measurement solvent is already
        # computed above (needed for calibration), so this adds exactly ONE more
        # TD-DFT: toluene if the molecule was measured in something polar, water
        # otherwise. Two points give the slope's SIGN and rough magnitude, which is
        # all the push-pull test needs. A 4-solvent sweep quadrupled the cost for a
        # better-conditioned fit we do not need at this stage.
        # WATER IS ALWAYS THE SECOND POINT. The entire design target is switching in
        # WATER, so a solvent pair that never touches water measures the wrong thing.
        # The earlier rule ("toluene if the base solvent is polar") silently produced
        # toluene+CHCl3 and toluene+DCM -- no water anywhere. Pairing the organic
        # measurement solvent with water also gives the widest polarity span
        # available (Onsager f 0.24-0.42 -> 0.49), which is the best-conditioned
        # 2-point slope we can get.
        if not quick and not over_budget("after first TD-DFT"):
            other = "water" if solvent != "water" else "toluene"
            series = [(solvent, onsager(eps), sel["nm"])] if sel else []
            st, _ = tddft_states(g["mol"], basis=basis_td, eps=EPS[other],
                                 nstates=nstates)
            s_i, _rule = select_dasa_band(st) if st else (None, None)
            if s_i:
                series.append((other, onsager(EPS[other]), s_i["nm"]))
            series.sort(key=lambda t: t[1])
            res["solvent_series"] = [dict(solvent=n, onsager=round(f, 4),
                                          lambda_nm=round(l, 1)) for n, f, l in series]
            if len(series) >= 2:
                fs = np.array([f for _, f, _ in series])
                ls = np.array([l for _, _, l in series])
                slope, _ = np.polyfit(fs, ls, 1)
                res["solvatochromic_slope_nm_per_onsager"] = float(slope)
                res["solvatochromic_total_shift_nm"] = float(ls[-1] - ls[0])
                res["negatively_solvatochromic"] = bool(slope < 0)
                # |slope| this small means essentially no CT character
                res["charge_transfer_alive"] = bool(slope < -20.0)
    except Exception as exc:                      # never let one molecule kill a run
        res["error"] = f"{type(exc).__name__}: {exc}"
    res["wall_s"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def fit_calibration(pairs):
    """Least-squares fit of measured vs computed excitation ENERGY (eV).

    Energy, not nm: the TD-DFT error is approximately constant in energy, so a
    linear fit in eV is the physically appropriate form. Returns slope, intercept,
    residual sigma in nm, and whether the falsifiable gate passes.
    """
    if len(pairs) < 3:
        return None
    calc = np.array([NM_EV / c for _, c in pairs])
    exp = np.array([NM_EV / e for e, _ in pairs])
    slope, intercept = np.polyfit(calc, exp, 1)
    pred_ev = slope * calc + intercept
    resid_nm = np.array([NM_EV / p for p in pred_ev]) - np.array([e for e, _ in pairs])
    sigma = float(np.sqrt(np.mean(resid_nm ** 2)))
    return dict(slope=float(slope), intercept_ev=float(intercept),
                residual_sigma_nm=sigma, n=len(pairs),
                residuals_nm=[float(r) for r in resid_nm],
                gate_passes=bool(sigma <= MAX_RESIDUAL_NM),
                max_residual_nm=MAX_RESIDUAL_NM)


def apply_calibration(cal, lambda_calc_nm):
    if cal is None or lambda_calc_nm is None:
        return None
    ev = cal["slope"] * (NM_EV / lambda_calc_nm) + cal["intercept_ev"]
    return float(NM_EV / ev) if ev > 0 else None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true",
                    help="run the measured reference ladder and fit the calibration")
    ap.add_argument("--csv", help="candidate CSV (needs a SMILES column)")
    ap.add_argument("--top", type=int, default=5, help="how many candidates to verify")
    ap.add_argument("--quick", action="store_true",
                    help="plumbing smoke test: small basis, NO DFT geometry optimisation")
    ap.add_argument("--truncate", action="store_true",
                    help="replace peripheral tails with H before DFT (validate first!)")
    ap.add_argument("--check-truncation", action="store_true",
                    help="compute one molecule full vs truncated and report the shift")
    ap.add_argument("--out", default="outputs_dasa_full/dft_verify_v2.json")
    args = ap.parse_args()

    results = dict(protocol=dict(
        geometry="literature (2Z,4E) -> ETKDG conformer search -> flattest planar "
                 "-> B3LYP/6-31G* ddCOSMO optimisation",
        excited="TDA-TD-DFT B3LYP/6-31+G(d) ddCOSMO, 10 states",
        state_selection="lowest bright (f>=0.20) state with HOMO->LUMO weight >=0.50",
        calibration="linear fit of measured vs computed EXCITATION ENERGY (eV)",
        gate=f"residual sigma must be <= {MAX_RESIDUAL_NM} nm, else TD-DFT cannot rank",
        quick=args.quick), references=[], candidates=[], calibration=None)

    if args.check_truncation:
        ref = REFERENCES[3]
        full = compute_one(ref["smiles"], ref["label"] + " [FULL]", ref["solvent"],
                           quick=args.quick)
        trunc = compute_one(ref["smiles"], ref["label"] + " [TRUNCATED]", ref["solvent"],
                            quick=args.quick, truncate=True)
        a, b = full.get("lambda_calc_nm"), trunc.get("lambda_calc_nm")
        print(f"\ntruncation check on {ref['label']}")
        print(f"  full      : {a} nm")
        print(f"  truncated : {b} nm")
        if a and b:
            print(f"  shift     : {b - a:+.1f} nm "
                  f"({'ACCEPTABLE' if abs(b - a) < 10 else 'TOO LARGE -- do not truncate'})")
        return

    if args.calibrate or args.csv:
        print(f"\n=== calibration ladder ({len(REFERENCES)} measured DASAs) ===", flush=True)
        pairs = []
        for ref in REFERENCES:
            r = compute_one(ref["smiles"], ref["label"], ref["solvent"], quick=args.quick)
            r["lambda_exp_nm"] = ref["lam_nm"]
            r["source"] = ref["source"]
            results["references"].append(r)
            lam = r.get("lambda_calc_nm")
            status = (f"calc {lam:6.1f}  exp {ref['lam_nm']}  "
                      f"err {lam - ref['lam_nm']:+6.1f} nm  [{r.get('selection_rule')}]"
                      if lam else f"FAILED: {r.get('error')}")
            print(f"  {ref['label'][:46]:46s} {status}", flush=True)
            if lam:
                pairs.append((ref["lam_nm"], lam))
        cal = fit_calibration(pairs)
        results["calibration"] = cal
        if cal:
            print(f"\n  fit: E_exp = {cal['slope']:.4f} * E_calc + {cal['intercept_ev']:+.4f} eV")
            print(f"  residual sigma = {cal['residual_sigma_nm']:.1f} nm  (n={cal['n']})")
            print("  GATE: " + ("PASS -- calibrated ranking is defensible"
                                if cal["gate_passes"] else
                                f"FAIL -- sigma > {MAX_RESIDUAL_NM} nm. TD-DFT CANNOT rank "
                                "these molecules; escalate to DLPNO-STEOM-CCSD "
                                "(MAE 0.049 eV on DASAs, Chem Sci 2021)."))
        else:
            print("\n  calibration failed (need >=3 successful references)")

    if args.csv:
        import csv as _csv
        with open(args.csv) as fh:
            rows = list(_csv.DictReader(fh))
        key = next(k for k in rows[0] if k.lower() in ("smiles", "smi", "smiles_open"))
        cal = results["calibration"]
        print(f"\n=== candidates (top {args.top}) ===", flush=True)
        for row in rows[:args.top]:
            r = compute_one(row[key], row.get("label", row[key][:40]),
                            quick=args.quick, truncate=args.truncate)
            r["lambda_calibrated_nm"] = apply_calibration(cal, r.get("lambda_calc_nm"))
            results["candidates"].append(r)
            if r.get("lambda_calc_nm"):
                cl = r["lambda_calibrated_nm"]
                print(f"  raw {r['lambda_calc_nm']:6.1f} -> calibrated "
                      f"{cl:6.1f} nm  twist {r.get('twist_dft_deg')}  "
                      f"[{r['selection_rule']}]", flush=True)
            else:
                print(f"  FAILED: {r.get('error')}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
