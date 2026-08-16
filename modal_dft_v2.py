"""Calibrated TD-DFT verification on Modal — one container per molecule.

PURPOSE: decide whether the corrected DFT protocol is TRUSTWORTHY, by running two
MEASURED reference DASAs through exactly the same pipeline as the candidates and
seeing whether the calibration holds. This is a method test first, a candidate
ranking second.

Set (4 containers, parallel):
  * ChemSci-1   Me2N/1,3-diMe-barbituric ... 567 nm (CHCl3)   alkyl donor anchor
  * NatComm-10  indoline/barbituric ........ 615 nm (CH2Cl2)  FUSED-ARYL anchor,
        structurally closest to the candidates (both are rigid ring donors)
  * two candidates: the only rigid-donor molecules in the shortlist

Two references give a two-point local calibration instead of a single anchor, and
they bracket the donor axis (alkyl vs fused aryl).

PREEMPTION SAFETY: every calculation writes /results/dft_v2/<hash>.json and commits
the volume IMMEDIATELY, so a spot kill costs one molecule, not the run. No DFT
geometry optimisation is allowed to run unbounded (maxsteps capped, SCF max_cycle
capped, per-container timeout).

    modal run --detach modal_dft_v2.py
    modal volume get --force dasa-outputs dft_v2 ./outputs_dasa_full/dft_v2
"""
import os

import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-dft-v2")
image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("rdkit=2024.03", "numpy<2", channels=["conda-forge"])
    .pip_install("pyscf==2.5.0", "scipy", "geometric")
    .add_local_dir(REPO, "/repo",
                   ignore=[".git", "__pycache__", "outputs_dasa*", "outputs",
                           "outputs_rl2", "*.model", "*.prior", "*.egg-info", "build"])
)
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)

# label, SMILES, measured lambda_max (None = candidate), solvent
#
# THE COMPARISON: our best TETHERED candidate vs our best ANILINE candidate, both
# on the SAME barbituric acceptor, so the only variable is the DONOR ARCHITECTURE.
# Both were regenerated from the legacy corpus with dasa_chem.legacy_to_corrected()
# (the hydroxyl moves from the acceptor-bonded carbon to C2; donor, acceptor and
# every peripheral substituent are preserved, formula unchanged). Running the
# originals would have repeated the connectivity error.
#
# Note the aniline originally proposed (aniline-5) carried a PYRAZOLIDINEDIONE
# acceptor, which is uncharacterised -- a UV result would not have told us whether
# the donor or the acceptor was responsible. The one below has barbituric plus a
# dpKa of +0.59, sitting inside the measured 2nd-generation window.
#
# PREDICTION ON RECORD (dpKa): tethered-0 = +7.29, i.e. 1st-generation-like and
# water-TRAPPED, because its tethered N is a basic cyclic ALKYL amine. The aniline
# = +0.59, inside the switchable window. If DFT agrees, the tethered architecture
# as currently built is the wrong end of the trap axis despite its rigidity.
def load_molset(csv_path="outputs_dasa_modal/outputs_dasa/synthesis_shortlist.csv"):
    """References (always) + candidates read from dft_set.csv.

    The candidates are THE MOLECULES WE ACTUALLY RECOMMEND -- read from
    synthesis_shortlist.csv (notebooks/recommend_synthesis.py), not from a separate
    DFT-selection list. Verifying a different set than the one being recommended is
    how the two drift apart; keeping one source of truth means the DFT result speaks
    directly to the synthesis proposal.

    All of them have already passed the affordability screen (heavy-atom count after
    chromophore truncation), so nothing too expensive to finish reaches a container.
    """
    refs = [
        ("REF ChemSci-1 Me2N/barbituric",
         "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", 567, "chloroform"),
        ("REF NatComm-10 indoline/barbituric",
         "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C", 615, "dichloromethane"),
    ]
    out = list(refs)
    try:
        import csv as _csv
        with open(csv_path) as fh:
            for i, row in enumerate(_csv.DictReader(fh)):
                # toluene is the classic DASA solvent AND the widest polarity
                # contrast against water, so the 2-point solvatochromic slope is
                # best conditioned. References keep their MEASUREMENT solvent
                # because that is what the calibration compares against.
                out.append((f"REC-{row.get('rank', i+1)} "
                            f"{row.get('acceptor','')}/{row.get('donor','')}"[:34],
                            row["SMILES"], None, "toluene"))
    except Exception as exc:
        print(f"WARNING: could not read {csv_path} ({exc}); running references only.")
    return out


MOLSET = load_molset()



# Container timeout is deliberately LARGER than the in-function wall-clock budget
# so the budget fires first and returns a partial, committed result. A container
# killed by Modal's timeout returns nothing; a budget abort returns everything it
# managed, plus the stage it stopped at.
# MEASURED, not guessed: ChemSci-1 (18 heavy after truncation) took 2424 s for the
# full protocol with TWO TD-DFT calculations. A 2400 s budget would have aborted it
# at 99% complete and thrown away the second solvent. 4200 s gives real headroom for
# the larger candidates (24-26 heavy) without allowing an unbounded hang.
_BUDGET_S = 4200          # 70 min of real work per molecule
@APP.function(image=image, volumes={"/results": vol}, cpu=8.0,
              timeout=_BUDGET_S + 900, retries=1)
def dft_one(label: str, smiles: str, lam_exp, solvent: str):
    import hashlib
    import json
    import sys
    import traceback

    sys.path.insert(0, "/repo/notebooks")
    sys.path.insert(0, "/repo/plugins")
    import dft_verify_v2 as V

    h = hashlib.md5((label + smiles).encode()).hexdigest()[:10]
    os.makedirs("/results/dft_v2", exist_ok=True)
    out = {"label": label, "smiles": smiles, "lambda_exp_nm": lam_exp,
           "solvent": solvent, "log": []}

    def save(msg=None):
        if msg:
            out["log"].append(msg)
            print(f"[{label}] {msg}", flush=True)
        try:
            json.dump(out, open(f"/results/dft_v2/{h}.json", "w"), indent=2)
            vol.commit()
        except Exception:
            pass

    save("start")
    try:
        r = V.compute_one(smiles, label, solvent=solvent, quick=False,
                          truncate=True, dft_opt=False, budget_s=_BUDGET_S)
        out.update(r)
        lam = r.get("lambda_calc_nm")
        if lam and lam_exp:
            out["error_nm"] = lam - lam_exp
            save(f"calc {lam:.1f} nm vs exp {lam_exp} -> {lam - lam_exp:+.1f} nm")
        elif lam:
            save(f"calc {lam:.1f} nm (candidate, no measurement)")
        else:
            save(f"FAILED: {r.get('error')}")
        save(f"heavy={r.get('heavy_atoms')} truncated={r.get('truncated')} "
             f"twist={r.get('twist_mmff_deg')} rule={r.get('selection_rule')} "
             f"slope={r.get('solvatochromic_slope_nm_per_onsager')} "
             f"aborted_at={r.get('aborted_at')}")
    except Exception:
        out["error"] = traceback.format_exc()[-1500:]
        save("EXCEPTION")
    save("DONE")
    return out


@APP.function(image=image, volumes={"/results": vol}, timeout=60 * 60 * 6)
def orchestrate(molset: list):
    import json

    results = list(dft_one.starmap([tuple(m) for m in molset]))
    json.dump(results, open("/results/dft_v2_results.json", "w"), indent=2)
    vol.commit()

    print("\n=== DFT v2 SUMMARY ===", flush=True)
    refs = [r for r in results if r.get("lambda_exp_nm")]
    for r in results:
        lam = r.get("lambda_calc_nm")
        line = f"{r['label']:38s} calc={lam if lam else 'FAIL'}"
        if r.get("lambda_exp_nm"):
            line += f"  exp={r['lambda_exp_nm']}  err={r.get('error_nm')}"
        line += (f"  twist_dft={r.get('twist_dft_deg')}"
                 f"  rule={r.get('selection_rule')}")
        print(line, flush=True)

    # --- 1. is the LAMBDA error a single systematic offset? ------------------
    ok = [r for r in refs if r.get("lambda_calc_nm")]
    if len(ok) >= 2:
        errs = [r["lambda_calc_nm"] - r["lambda_exp_nm"] for r in ok]
        spread = max(errs) - min(errs)
        print(f"\nreference lambda errors (nm): {[round(x, 1) for x in errs]}", flush=True)
        print(f"spread across references: {spread:.1f} nm", flush=True)
        print("LAMBDA VERDICT: " + (
            "consistent offset -> the protocol behaves systematically; a calibrated "
            "ranking is defensible."
            if spread <= 25 else
            "INCONSISTENT -- not a single systematic offset, so a calibrated lambda "
            "cannot be trusted here. Do NOT rank candidates on it; escalate the "
            "method (DLPNO-STEOM-CCSD, MAE 0.049 eV on DASAs)."), flush=True)
    else:
        print("\nfewer than 2 references succeeded -- no calibration possible.", flush=True)

    # --- 2. IS THE PUSH-PULL SYSTEM ALIVE? -----------------------------------
    # The solvatochromic slope is the literature's charge-separation measurement,
    # and it replaces a hand-set threshold on donor strength. The working range is
    # taken from the MEASURED references run through this same protocol, so the
    # criterion is anchored rather than asserted.
    ref_slopes = [r.get("solvatochromic_slope_nm_per_onsager") for r in refs]
    ref_slopes = [s for s in ref_slopes if s is not None]
    print("\n--- solvatochromic slope (nm per Onsager unit; DASAs are NEGATIVE) ---",
          flush=True)
    for r in results:
        s = r.get("solvatochromic_slope_nm_per_onsager")
        tag = "REF " if r.get("lambda_exp_nm") else "CAND"
        print(f"  {tag} {r['label'][:36]:36s} slope="
              f"{('%+8.1f' % s) if s is not None else '    n/a'}"
              f"   total shift {r.get('solvatochromic_total_shift_nm', 'n/a')}", flush=True)
    if ref_slopes:
        weakest_ref = max(ref_slopes)          # least negative reference
        print(f"\n  reference slopes span {min(ref_slopes):+.1f} to {weakest_ref:+.1f}", flush=True)
        for r in results:
            if r.get("lambda_exp_nm"):
                continue
            s = r.get("solvatochromic_slope_nm_per_onsager")
            if s is None:
                continue
            if s > weakest_ref * 0.5:
                print(f"  *** {r['label']}: slope {s:+.1f} is far weaker than any "
                      f"measured DASA -> the donor is NOT pushing; treat as a failed "
                      f"chromophore, and raise DASATrapEscape's lower dpKa edge.",
                      flush=True)
            elif s > 0:
                print(f"  *** {r['label']}: POSITIVE slope -> not a DASA CT band at "
                      f"all (or the wrong state was selected).", flush=True)
            else:
                print(f"  OK  {r['label']}: slope {s:+.1f} is within the measured "
                      f"DASA regime -> push-pull is alive; the azole donor class is "
                      f"legitimate and should NOT be excluded.", flush=True)
    return results


@APP.local_entrypoint()
def main():
    orchestrate.spawn(MOLSET)
    print(f"dispatched DETACHED: {len(MOLSET)} molecules "
          "(2 measured references + 2 rigid-donor candidates), one container each.")
    print("  live: modal volume get --force dasa-outputs dft_v2 ./outputs_dasa_full/dft_v2")
