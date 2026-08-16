#!/usr/bin/env python
"""CASSCF + NEVPT2 excitation energies for DASAs — the CT-capable escalation.

WHY
---
TD-DFT failed on DASAs in our own hands, three independent ways:
  * reference lambda errors -131 and -173 nm, spread 41.6 nm (calibration gate FAILED)
  * solvatochromic slopes came out POSITIVE for all six molecules, including both
    measured references -- DASAs are NEGATIVELY solvatochromic, so the method is not
    describing the charge-transfer band at all
  * matching the published benchmark: TD-DFT gives the DASA S1 "almost null
    charge-transfer character" (PMC5615680)

That paper measured the alternatives on DASAs:
    TD-DFT   0.44 eV      NEVPT2   0.15 eV      CASPT2   0.06 eV
NEVPT2 is MULTIREFERENCE, so it can represent the CT configuration TD-DFT misses,
and unlike CASPT2 it is available in PySCF -- same environment we already have.

THE ACTIVE SPACE IS THE WHOLE BALLGAME
--------------------------------------
Everything here lives or dies on the active space, and a bad one produces
confident garbage with no automatic gate to catch it. So it is chosen by an
auditable rule, and the choice is REPORTED for inspection:

  1. Rotate the molecule so the conjugated chain's best-fit plane is the xy-plane.
     A pi active space is only well defined relative to that plane.
  2. Compute RHF, then measure each frontier orbital's PI CHARACTER as the fraction
     of its density on p_z AOs of the conjugated atoms (donor N, chain C, acceptor).
  3. Keep the frontier orbitals whose pi character exceeds a threshold, capped at
     ncas_max. Those are the pi/pi* orbitals of the push-pull system.
  4. Print every selected orbital with its energy and pi fraction, so the space can
     be defended or rejected on inspection rather than taken on trust.

Then state-averaged CASSCF over 2 roots, and NEVPT2 on each root. The excitation
energy is the NEVPT2 difference.

RUN THE REFERENCES FIRST. Two compounds with MEASURED lambda_max (567 and 615 nm)
go through the identical protocol. If NEVPT2 does not reproduce them within ~0.2 eV
AND give a negative solvatochromic direction, the active space or the method is
inadequate and no candidate number is worth computing.

    python notebooks/nevpt2_verify.py --references-only
    python notebooks/nevpt2_verify.py --references-only --dry-run   # active space only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugins"))
import dasa_chem as dc  # noqa: E402
import dft_verify_v2 as V  # noqa: E402

NM_EV = 1239.841

REFERENCES = [
    dict(label="REF ChemSci-1 Me2N/barbituric", lam_nm=567,
         smiles="CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O"),
    dict(label="REF NatComm-10 indoline/barbituric", lam_nm=615,
         smiles="O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C"),
]


def align_to_chain_plane(m3d, chain_idx):
    """Rotate so the conjugated chain's best-fit plane is the xy-plane.

    A pi orbital is only definable relative to the molecular plane, so this has to
    happen before any p_z analysis. Returns (coords, symbols).
    """
    conf = m3d.GetConformer()
    xyz = np.array(conf.GetPositions())
    sub = xyz[chain_idx]
    centroid = sub.mean(axis=0)
    # normal = smallest singular vector of the centred chain coordinates
    _u, _s, vt = np.linalg.svd(sub - centroid)
    normal = vt[2]
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    c = float(np.dot(normal, z))
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    out = (xyz - centroid) @ R.T
    syms = [a.GetSymbol() for a in m3d.GetAtoms()]
    return out, syms


def pi_character(mol_pyscf, mo_coeff, conj_atoms):
    """Fraction of each MO's Mulliken-like weight sitting on p_z AOs of the
    conjugated atoms. ~1 means a clean pi orbital, ~0 means sigma."""
    labels = mol_pyscf.ao_labels()
    pz = [i for i, lab in enumerate(labels)
          if "pz" in lab.replace(" ", "") and int(lab.split()[0]) in conj_atoms]
    if not pz:
        return np.zeros(mo_coeff.shape[1])
    s = mol_pyscf.intor("int1e_ovlp")
    w = np.einsum("ij,jk,ik->ik", mo_coeff.T, s, mo_coeff.T).T   # AO weights per MO
    tot = np.abs(w).sum(axis=0)
    tot[tot < 1e-12] = 1.0
    return np.abs(w[pz, :]).sum(axis=0) / tot


def select_pi_active_space(mf, mol_pyscf, conj_atoms, ncas_max=10, pi_min=0.60,
                           vir_window_eV=10.0):
    """Frontier orbitals with genuine pi character -> (ncas, nelecas, orb_indices).

    pi_min=0.60 is deliberately strict. At 0.25 the search ran out of high-pi
    virtuals in the frontier window and reached up to LUMO+11 at pi=0.39 -- a
    half-sigma orbital padding the space with cost but no CT physics. Better a
    smaller, clean pi space than a nominally larger dirty one.
    """
    nocc = mol_pyscf.nelectron // 2
    pc = pi_character(mol_pyscf, mf.mo_coeff, conj_atoms)
    e_eV = np.asarray(mf.mo_energy) * 27.2114
    # ENERGY WINDOW on the virtuals. pi character alone is not enough: the search
    # otherwise reached LUMO+36 at +18.1 eV with pi 0.84 -- a diffuse/Rydberg-like
    # orbital that happens to carry p_z weight. It is not a valence pi* orbital and
    # putting it in the active space buys cost and no CT physics.
    lumo_eV = e_eV[nocc]
    # FIXED SIZE, always ncas_max/2 occupied + ncas_max/2 virtual.
    #
    # A hard pi threshold made the space size vary BY MOLECULE -- ChemSci-1 came out
    # CAS(10,9) while NatComm-10 got CAS(10,10). Excitation energies from different
    # active spaces are not comparable, which destroys the one thing this protocol
    # depends on: that references and candidates are treated identically. So the
    # size is fixed and the orbitals are ranked BY pi character within the energy
    # window; pi_min becomes a reporting threshold, not a filter. The minimum pi
    # actually achieved is returned so a dirty space is visible rather than hidden.
    half = ncas_max // 2
    occ_pool = sorted(range(nocc), key=lambda i: -pc[i])
    vir_pool = sorted((i for i in range(nocc, mol_pyscf.nao)
                       if e_eV[i] - lumo_eV <= vir_window_eV), key=lambda i: -pc[i])
    chosen_occ = sorted(occ_pool[:half])
    chosen_vir = sorted(vir_pool[:half])
    idx = chosen_occ + chosen_vir
    return len(idx), 2 * len(chosen_occ), idx, pc


def run_one(smiles, label, lam_exp=None, basis="cc-pvdz", ncas_max=10,
            truncate=True, dry_run=False, budget_s=7200, eps=None,
            max_twist_deg=15.0, n_confs=200):
    t0 = time.time()
    res = dict(label=label, smiles=smiles, lambda_exp_nm=lam_exp, basis=basis)
    from pyscf import gto, scf, mcscf, mrpt

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not dc.is_dasa(mol):
        res["error"] = "not a corrected-core DASA"
        return res
    if truncate:
        t = V.truncate_chromophore(V.assign_literature_stereo(mol))
        if t is not None:
            mol, res["truncated"] = t, True
    m3d, twist = V.dc.planar_conformer(Chem.MolToSmiles(mol), n_confs=n_confs)
    if m3d is None:
        res["error"] = "no conformer"
        return res
    res.update(heavy=mol.GetNumHeavyAtoms(), twist_deg=round(twist, 1))
    # HARD STOP on a non-planar chromophore. This is not a nicety: on Modal the
    # ChemSci-1 reference came out at 40.5 deg twist (locally the same input gives
    # 4.5 deg -- the container ships rdkit 2024.03 vs 2026.03 here, and ETKDG
    # differs between them). A 40 deg twist mixes sigma into the pi manifold, which
    # dropped the active space's worst pi character to 0.22 and inflated the error
    # to 0.264 eV. The number was a geometry artefact, not a method failure -- so
    # refuse to compute rather than emit it.
    if twist > max_twist_deg:
        res["error"] = (f"chain twist {twist:.1f} deg > {max_twist_deg} deg: the "
                        f"chromophore is not planar, so a pi active space is not "
                        f"well defined. Refusing to compute (this would produce a "
                        f"geometry artefact, not a method result).")
        return res

    ix = dc._core_idx(mol)
    chain = [ix[k] for k in ("N", "Ca", "Cb", "Cc", "Cd", "O", "Ce", "Cf")]
    # conjugated set = the core + every atom in a ring touching it + attached C=O
    conj = set(chain)
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if conj & set(ring):
            conj |= set(ring)
    for i in list(conj):
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            b = mol.GetBondBetweenAtoms(i, nb.GetIdx())
            if b.GetBondTypeAsDouble() == 2.0:
                conj.add(nb.GetIdx())

    coords, syms = align_to_chain_plane(m3d, [c for c in chain if c < m3d.GetNumAtoms()])
    atom = [(s, tuple(c)) for s, c in zip(syms, coords)]
    pm = gto.M(atom=atom, basis=basis, charge=Chem.GetFormalCharge(mol),
               spin=0, verbose=0)
    res["nao"] = int(pm.nao)
    mf = scf.RHF(pm)
    if eps:
        from pyscf.solvent import ddCOSMO
        mf = ddCOSMO(mf)
        mf.with_solvent.eps = eps
    mf.max_cycle = 200
    mf.kernel()
    res["scf_converged"] = bool(mf.converged)
    if not mf.converged:
        res["error"] = "RHF did not converge"
        return res

    ncas, nelecas, idx, pc = select_pi_active_space(mf, pm, conj, ncas_max=ncas_max)
    nocc = pm.nelectron // 2
    res["active_space"] = dict(
        ncas=ncas, nelecas=nelecas,
        orbitals=[dict(mo=int(i), rel=("HOMO" if i == nocc - 1 else
                                       "LUMO" if i == nocc else
                                       f"HOMO-{nocc-1-i}" if i < nocc else
                                       f"LUMO+{i-nocc}"),
                       energy_eV=round(float(mf.mo_energy[i]) * 27.2114, 3),
                       pi_fraction=round(float(pc[i]), 3)) for i in idx])
    print(f"[{label}] {pm.nao} AOs, twist {res['twist_deg']} deg  -> "
          f"CAS({nelecas},{ncas})", flush=True)
    for o in res["active_space"]["orbitals"]:
        print(f"    {o['rel']:>8s}  MO {o['mo']:3d}  {o['energy_eV']:+8.3f} eV  "
              f"pi {o['pi_fraction']:.2f}", flush=True)
    res["active_space"]["min_pi_fraction"] = round(
        min(o["pi_fraction"] for o in res["active_space"]["orbitals"]), 3)
    print(f"    -> CAS({nelecas},{ncas}), worst pi character "
          f"{res['active_space']['min_pi_fraction']:.2f}", flush=True)
    if ncas < 4:
        res["error"] = f"pi active space too small (ncas={ncas}); check the alignment"
        return res
    if dry_run:
        res["wall_s"] = round(time.time() - t0, 1)
        return res

    # TWO-STEP, because PySCF forbids NEVPT2 on a state-averaged solver:
    #   "State-average FCI solver object cannot be used in NEVPT2 calculation.
    #    A separated multi-root CASCI calculation is required."
    # So: state-averaged CASSCF to relax the ORBITALS for both roots even-handedly
    # (using one root's orbitals would bias the excitation energy), then a separate
    # multi-root CASCI in those orbitals, and NEVPT2 on the CASCI roots.
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 50
    mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, [i + 1 for i in idx])
    mc.state_average_([0.5, 0.5])
    mc.kernel(mo)
    res["casscf_converged"] = bool(mc.converged)
    res["casscf_e"] = [float(x) for x in np.atleast_1d(mc.e_states)]
    res["casscf_dE_eV"] = round(float(
        np.atleast_1d(mc.e_states)[1] - np.atleast_1d(mc.e_states)[0]) * 27.2114, 4)
    print(f"    CASSCF done (converged={mc.converged}), "
          f"dE={res['casscf_dE_eV']} eV, {time.time()-t0:.0f}s", flush=True)

    if time.time() - t0 > budget_s:
        res["error"] = "budget exceeded before NEVPT2"
        return res

    mc_ci = mcscf.CASCI(mf, ncas, nelecas)
    mc_ci.fcisolver.nroots = 2
    mc_ci.kernel(mc.mo_coeff)
    res["casci_e"] = [float(x) for x in np.atleast_1d(mc_ci.e_tot)]
    e_nevpt = []
    for root in (0, 1):
        corr = float(mrpt.NEVPT(mc_ci, root=root).kernel())
        e_nevpt.append(float(np.atleast_1d(mc_ci.e_tot)[root]) + corr)
        print(f"    NEVPT2 root {root}: corr {corr:+.6f} Ha, "
              f"{time.time()-t0:.0f}s", flush=True)
    res["nevpt2_e"] = e_nevpt
    dE = (e_nevpt[1] - e_nevpt[0]) * 27.2114
    res.update(excitation_eV=round(dE, 4),
               lambda_nm=round(NM_EV / dE, 1) if dE > 0 else None)
    if lam_exp and res["lambda_nm"]:
        res["error_nm"] = round(res["lambda_nm"] - lam_exp, 1)
        res["error_eV"] = round(dE - NM_EV / lam_exp, 4)
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--references-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="RHF + active-space selection only; no CASSCF/NEVPT2")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--ncas-max", type=int, default=10)
    ap.add_argument("--top-candidate", default=None, help="extra SMILES to include")
    ap.add_argument("--out", default="outputs_dasa_full/nevpt2_results.json")
    args = ap.parse_args()

    jobs = [(r["smiles"], r["label"], r["lam_nm"]) for r in REFERENCES]
    if args.top_candidate and not args.references_only:
        jobs.append((args.top_candidate, "CAND top", None))

    results = []
    for smi, lab, exp in jobs:
        try:
            r = run_one(smi, lab, exp, basis=args.basis, ncas_max=args.ncas_max,
                        dry_run=args.dry_run)
        except Exception as exc:
            import traceback
            r = dict(label=lab, smiles=smi, error=traceback.format_exc()[-1200:])
        results.append(r)
        if r.get("lambda_nm"):
            print(f"[{lab}] lambda {r['lambda_nm']} nm  "
                  f"(exp {exp}) err {r.get('error_nm')} nm / {r.get('error_eV')} eV  "
                  f"{r.get('wall_s')}s", flush=True)
        elif r.get("error"):
            print(f"[{lab}] {r['error'][:200]}", flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    errs = [r["error_eV"] for r in results if r.get("error_eV") is not None]
    if len(errs) >= 2:
        print(f"\nreference errors (eV): {[round(e, 3) for e in errs]}")
        print(f"spread: {max(errs) - min(errs):.3f} eV")
        print("VERDICT: " + ("NEVPT2 reproduces the references -- proceed to candidates"
                            if max(abs(e) for e in errs) <= 0.25 else
                            "NEVPT2 does NOT reproduce the references at this active "
                            "space/basis. Do not compute candidates; revisit the "
                            "active space (or escalate to CASPT2/DLPNO-STEOM-CCSD)."))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
