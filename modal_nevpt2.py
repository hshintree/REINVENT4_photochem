"""CASSCF + NEVPT2 on Modal — one container per molecule, detached-safe.

WHY MODAL: the last local attempt died when the laptop closed, after CASSCF had
already completed. Multireference work is hours per molecule, so it belongs on a
detached container that survives a lid close.

Set: the two MEASURED references first (567 and 615 nm) plus one candidate whose
chromophore is genuinely DISTINCT from both. Note that our top-ranked candidate
truncates to the SAME chromophore as NatComm-10, so computing it would just
re-verify the reference -- the informative one is REC-2 (benzoxazine donor,
isoxazolone acceptor), the only non-barbituric of the four.

Every result is committed to the volume the moment it exists, so a preemption or a
budget abort costs one molecule rather than the run.

    modal run --detach modal_nevpt2.py
    modal volume get --force dasa-outputs nevpt2 ./outputs_dasa_full/nevpt2
"""
import os

import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-nevpt2")
image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("rdkit=2024.03", "numpy<2", channels=["conda-forge"])
    .pip_install("pyscf==2.5.0", "scipy", "geometric")
    .add_local_dir(REPO, "/repo",
                   ignore=[".git", "__pycache__", "outputs_dasa*", "outputs",
                           "outputs_rl2", "*.model", "*.prior", "*.egg-info", "build"])
)
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)

_BUDGET_S = 12600          # 3.5 h of real work per molecule
_BASIS = "6-31g"           # the published 0.15 eV used cc-pVDZ; see the note below

# SOLVENTS. The previous run passed no eps at all, so every number -- including the
# 0.085 eV reference agreement -- was GAS PHASE. That is indefensible for a project
# whose target is behaviour in WATER. Each molecule now runs twice: its organic
# reference solvent (what the measured lambda_max was recorded in, so the calibration
# is like-for-like) and WATER (what we actually care about). The pair also gives the
# solvatochromic direction, which for a real DASA must be NEGATIVE.
_EPS = {"chloroform": 4.71, "dichloromethane": 8.93, "water": 78.4}
_ORGANIC = {"REF ChemSci-1 Me2N/barbituric": "chloroform",
            "REF NatComm-10 indoline/barbituric": "dichloromethane"}

MOLSET = [
    ("REF ChemSci-1 Me2N/barbituric",
     "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", 567),
    ("REF NatComm-10 indoline/barbituric",
     "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C", 615),
    ("CAND REC-2 benzoxazine/isoxazolone",
     "CC1=NOC(=O)C1=CC(O)=CC=CN1CCOc2ccc(C(=O)NCCS(N)(=O)=O)cc21", None),
]


@APP.function(image=image, volumes={"/results": vol}, cpu=8.0, memory=32768,
              timeout=_BUDGET_S + 1800, retries=1)
def nevpt2_one(label: str, smiles: str, lam_exp):
    import hashlib
    import json
    import sys
    import traceback

    sys.path.insert(0, "/repo/notebooks")
    sys.path.insert(0, "/repo/plugins")
    import nevpt2_verify as N

    h = hashlib.md5((label + smiles).encode()).hexdigest()[:10]
    os.makedirs("/results/nevpt2", exist_ok=True)
    out = {"label": label, "smiles": smiles, "lambda_exp_nm": lam_exp, "log": []}

    def save(msg=None):
        if msg:
            out["log"].append(msg)
            print(f"[{label}] {msg}", flush=True)
        try:
            # default=str so a stray numpy scalar cannot abort json.dump halfway and
            # leave a TRUNCATED file. That happened: every result JSON was cut off at
            # "nao": because pm.nao is a numpy int, and the bare except hid it. The
            # data survived only because it was also in the log strings.
            txt = json.dumps(out, indent=2, default=str)
            with open(f"/results/nevpt2/{h}.json", "w") as fh:
                fh.write(txt)
            vol.commit()
        except Exception as exc:
            print(f"[{label}] SAVE FAILED: {type(exc).__name__}: {exc}", flush=True)

    save("start")
    try:
        org = _ORGANIC.get(label, "dichloromethane")
        out["solvents"] = {}
        for solv in (org, "water"):
            r = N.run_one(smiles, label, lam_exp if solv == org else None,
                          basis=_BASIS, ncas_max=10, truncate=True,
                          budget_s=_BUDGET_S // 2, eps=_EPS[solv])
            out["solvents"][solv] = r
            if r.get("lambda_nm"):
                save(f"[{solv}] lambda {r['lambda_nm']} nm"
                     + (f" vs exp {lam_exp} -> {r.get('error_nm')} nm / "
                        f"{r.get('error_eV')} eV" if (lam_exp and solv == org) else ""))
            else:
                save(f"[{solv}] NO LAMBDA: {str(r.get('error'))[:250]}")
            a = r.get("active_space", {})
            save(f"[{solv}] CAS({a.get('nelecas')},{a.get('ncas')}) worst-pi "
                 f"{a.get('min_pi_fraction')} twist {r.get('twist_deg')} deg "
                 f"{r.get('wall_s')}s")
        lo = out["solvents"][org].get("lambda_nm")
        lw = out["solvents"]["water"].get("lambda_nm")
        if lo and lw:
            out["water_shift_nm"] = round(lw - lo, 1)
            out["negatively_solvatochromic"] = bool(lw < lo)
            save(f"organic->water shift {out['water_shift_nm']:+.1f} nm  "
                 f"negatively solvatochromic={out['negatively_solvatochromic']} "
                 f"(a real DASA MUST be negative)")
        # keep the organic-solvent result at top level for the calibration gate
        out.update({k: v for k, v in out["solvents"][org].items() if k != "log"})
    except Exception:
        out["error"] = traceback.format_exc()[-1500:]
        save("EXCEPTION")
    save("DONE")
    return out


@APP.function(image=image, volumes={"/results": vol}, timeout=60 * 60 * 12)
def orchestrate(molset: list):
    import json

    results = list(nevpt2_one.starmap([tuple(m) for m in molset]))
    json.dump(results, open("/results/nevpt2_results.json", "w"), indent=2)
    vol.commit()

    print("\n=== NEVPT2 SUMMARY ===", flush=True)
    for r in results:
        lam = r.get("lambda_nm")
        a = r.get("active_space", {})
        print(f"{r['label'][:40]:40s} lambda={lam if lam else 'FAIL':>8} "
              f"exp={r.get('lambda_exp_nm')} err={r.get('error_eV')} eV "
              f"CAS({a.get('nelecas')},{a.get('ncas')}) pi>={a.get('min_pi_fraction')}",
              flush=True)

    errs = [r["error_eV"] for r in results if r.get("error_eV") is not None]
    if len(errs) >= 2:
        worst = max(abs(e) for e in errs)
        print(f"\nreference errors (eV): {[round(e, 3) for e in errs]}", flush=True)
        print(f"spread {max(errs) - min(errs):.3f} eV, worst |err| {worst:.3f} eV",
              flush=True)
        print("VERDICT: " + (
            f"NEVPT2 reproduces both references to within {worst:.2f} eV -- "
            "comparable to the published 0.15 eV, so the candidate number is "
            "meaningful."
            if worst <= 0.25 else
            f"worst reference error {worst:.2f} eV EXCEEDS the 0.25 eV gate. The "
            "active space or basis is inadequate at this level; do not trust the "
            "candidate. Next step is cc-pVDZ (published basis) or CASPT2/"
            "DLPNO-STEOM-CCSD in ORCA, not more tuning here."), flush=True)
        print(f"\nNOTE: basis is {_BASIS}; the published 0.15 eV NEVPT2 benchmark "
              f"used cc-pVDZ. A residual error at this basis is partly basis, not "
              f"only method.", flush=True)
    else:
        print("\nfewer than 2 references produced a lambda -- no verdict possible.",
              flush=True)
    return results


@APP.local_entrypoint()
def main():
    orchestrate.spawn(MOLSET)
    print(f"dispatched DETACHED: {len(MOLSET)} molecules "
          "(2 measured references + 1 distinct candidate), one container each.")
    print("  pull: modal volume get --force dasa-outputs nevpt2 "
          "./outputs_dasa_full/nevpt2")
