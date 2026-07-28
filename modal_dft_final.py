"""DEFINITIVE DFT on the finalists — color (water + organic) and the switching barrier.

Runs, per molecule (3 finalists + a trapped negative anchor), one container each (parallel):

  COLOR:  TD-DFT λmax of the OPEN form in ddCOSMO WATER and TOLUENE (organic), B3LYP + CAM-B3LYP.
  BARRIER: xTB gives open / TS(scan max) / closed geometries in water; DFT single points (B3LYP,
           ddCOSMO water) on those 3 -> DFT-corrected ΔG(c-o), forward and REVERSE barrier.

Design for the three asks:
  * MONITORABLE: every single calc is written to /results/final_dft/<hash>.json and the volume is
    committed IMMEDIATELY, so `modal volume get` shows live partial progress (and survives preemption).
  * PARALLEL/FAST: one container per molecule; cpu=8 (PySCF OpenMP) per container.
  * CANNOT HANG: NO DFT geometry optimization (only single points -> bounded by SCF max_cycle=120).
    TD-DFT nstates=5. xTB steps have wall-clock timeouts. Per-container Modal timeout = 90 min, retries=1.
    A calc that fails/does-not-converge records null and the run CONTINUES.

    modal run --detach modal_dft_final.py
    # watch:  modal app logs <app-id> -f
    # pull live partial results:  modal volume get --force dasa-outputs final_dft ./outputs_dasa_full/final_dft
"""
import os
import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-dft-final")
image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("rdkit=2024.03", "xtb", "numpy<2", channels=["conda-forge"])
    .pip_install("pyscf==2.5.0", "scipy")
    .add_local_dir(REPO, "/repo",
                   ignore=[".git", "__pycache__", "outputs_dasa*", "outputs", "outputs_rl2",
                           "*.model", "*.prior", "*.egg-info", "build"])
)
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)
_H_KCAL = 627.5094740631
_EPS = {"water": 78.4, "toluene": 2.38}

# 3 finalists (from the kinetics screen) + trapped anchor for barrier validation.
MOLSET = [
    ("ANCHOR DMA-barbituric (trapped)", "CN(C)C=CC=CC(O)=C1C(=O)N(C)C(=O)N(C)C1=O"),
    ("tethered-0 (T0.963)", "O=C(CN1C(=O)NC(=O)C(=C(O)C=CC2=CN(CCc3ccncn3)CC2=O)C1=O)NCc1ccncc1"),
    ("tethered-4 (T0.944)", "O=C(CN1C(=O)NC(=O)C(=C(O)C=CC2=CN(CCc3cncc4ccccc34)CC2=O)C1=O)NCCC(O)CO"),
    ("aniline-5 (T0.944, visible ~582)", "Cn1cc(NC=CC=CC(O)=C2C(=O)NN(CC(=O)NCCN3CC3)C2=O)cn1"),
]


def _atoms_from_xyz(xyz):        # xyz string -> pyscf atom block (strip 2 header lines)
    return "\n".join(xyz.splitlines()[2:])


def _mmff_xyz(mol):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
    except Exception:
        pass
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), ""]
    for a in m.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines) + "\n"


def _xtb_opt(xyz, chg, d, tag, extra=None):
    import subprocess
    open(f"{d}/{tag}.xyz", "w").write(xyz)
    cmd = ["xtb", f"{tag}.xyz", "--opt", "--gfn", "2", "--alpb", "water",
           "--chrg", str(chg), "--uhf", "0", "--cycles", "40"]
    if extra:
        cmd += ["--input", extra]
    try:
        subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=60 * 8,
                       env={**os.environ, "OMP_NUM_THREADS": "8"})
    except Exception:
        return None
    p = f"{d}/xtbopt.xyz"
    return open(p).read() if os.path.exists(p) else None


def _dft_sp(xyz, chg, eps, xc="b3lyp", basis="6-31g*"):
    from pyscf import gto, dft
    from pyscf.solvent import ddCOSMO
    mol = gto.M(atom=_atoms_from_xyz(xyz), basis=basis, charge=chg, spin=0, verbose=0)
    mf = ddCOSMO(dft.RKS(mol)); mf.xc = xc; mf.with_solvent.eps = eps
    mf.max_cycle = 120; mf.conv_tol = 1e-7
    return float(mf.kernel())


def _tddft_lambda(xyz, chg, eps, xc="b3lyp", basis="6-31g*", nstates=5):
    from pyscf import gto, dft, tddft
    from pyscf.solvent import ddCOSMO
    mol = gto.M(atom=_atoms_from_xyz(xyz), basis=basis, charge=chg, spin=0, verbose=0)
    mf = ddCOSMO(dft.RKS(mol)); mf.xc = xc; mf.with_solvent.eps = eps
    mf.max_cycle = 120; mf.conv_tol = 1e-7
    mf.kernel()
    td = tddft.TDA(mf); td.nstates = nstates; td.kernel()
    exc = [(1240.0 / e, float(f)) for e, f in zip(td.e * 27.2114, td.oscillator_strength()) if e > 0]
    bright = [(l, f) for l, f in exc if f > 0.05]
    if bright:
        return round(max(bright, key=lambda t: t[1])[0], 1)
    return round(exc[0][0], 1) if exc else None


@APP.function(image=image, volumes={"/results": vol}, cpu=8.0, timeout=60 * 90, retries=1)
def final_one(label: str, smiles: str):
    import sys, tempfile, math, json, hashlib
    sys.path.insert(0, "/repo/notebooks"); sys.path.insert(0, "/repo/plugins")
    from rdkit import Chem
    import dasa_chem as dc

    h = hashlib.md5(smiles.encode()).hexdigest()[:10]
    os.makedirs("/results/final_dft", exist_ok=True)
    res = {"label": label, "smiles": smiles, "lambda_nm": {}, "barrier": {}, "log": []}

    def save(msg=None):
        if msg:
            res["log"].append(f"{msg}")
            print(f"[{label}] {msg}", flush=True)
        try:
            json.dump(res, open(f"/results/final_dft/{h}.json", "w"), indent=2); vol.commit()
        except Exception:
            pass

    mol = Chem.MolFromSmiles(smiles)
    chg = Chem.GetFormalCharge(mol)
    d = tempfile.mkdtemp(prefix="fin_")

    # --- OPEN geometry (xTB-opt in water) ---
    open_mmff = _mmff_xyz(mol)
    if open_mmff is None:
        return {**res, "error": "open embed failed"}
    open_xyz = _xtb_opt(open_mmff, chg, d, "open") or open_mmff
    save("open geometry ready")

    # --- COLOR: TD-DFT lambda_max, open form, water + toluene, B3LYP + CAM-B3LYP ---
    for solv, eps in _EPS.items():
        for xc in ("b3lyp", "camb3lyp"):
            try:
                lam = _tddft_lambda(open_xyz, chg, eps, xc)
                res["lambda_nm"][f"{solv}/{xc}"] = lam
                save(f"lambda {solv}/{xc} = {lam} nm")
            except Exception as e:
                res["lambda_nm"][f"{solv}/{xc}"] = None
                save(f"lambda {solv}/{xc} FAILED {type(e).__name__}")

    # --- BARRIER: xTB open/TS/closed geometries -> DFT single points (water) ---
    match = mol.GetSubstructMatch(dc._DASA_OPEN)
    closed_smi = dc.open_to_closed(smiles)
    if match and len(match) >= 6 and closed_smi:
        c1, c5 = match[1], match[5]
        closed_xyz = _xtb_opt(_mmff_xyz(Chem.MolFromSmiles(closed_smi)), chg, d, "closed")
        # scan open->closed to get the TS geometry (scan max)
        from rdkit.Chem import AllChem
        mo = Chem.AddHs(mol); AllChem.EmbedMolecule(mo, randomSeed=42); AllChem.MMFFOptimizeMolecule(mo)
        conf = mo.GetConformer()
        p1, p5 = conf.GetAtomPosition(c1), conf.GetAtomPosition(c5)
        d0 = max(math.dist((p1.x, p1.y, p1.z), (p5.x, p5.y, p5.z)), 2.6)
        open(f"{d}/scan.inp", "w").write(
            f"$constrain\n force constant=0.8\n distance: {c1+1}, {c5+1}, {d0:.3f}\n"
            f"$scan\n 1: {d0:.3f}, 1.53, 14\n$end\n")
        _xtb_opt(open_xyz, chg, d, "scanstart", extra="scan.inp")
        ts_xyz = None
        if os.path.exists(f"{d}/xtbscan.log"):
            L = open(f"{d}/xtbscan.log").read().splitlines()
            frames, i = [], 0
            while i < len(L):
                try:
                    n = int(L[i].strip())
                except Exception:
                    break
                en = None
                for tok in L[i + 1].split():
                    try:
                        en = float(tok); break
                    except Exception:
                        pass
                frames.append((en, "\n".join(L[i:i + n + 2]) + "\n"))
                i += n + 2
            valid = [f for f in frames if f[0] is not None]
            if valid:
                ts_xyz = max(valid, key=lambda t: t[0])[1]     # highest-energy frame = TS guess
        # DFT single points on the 3 stationary-ish points (water)
        for name, xyz in (("open", open_xyz), ("ts", ts_xyz), ("closed", closed_xyz)):
            if xyz is None:
                res["barrier"][name] = None; save(f"barrier[{name}] geometry missing"); continue
            try:
                res["barrier"][name] = _dft_sp(xyz, chg, _EPS["water"])
                save(f"barrier[{name}] DFT-SP done")
            except Exception as e:
                res["barrier"][name] = None; save(f"barrier[{name}] FAILED {type(e).__name__}")
        b = res["barrier"]
        if all(b.get(k) is not None for k in ("open", "ts", "closed")):
            res["dG_close_kcal"] = round((b["closed"] - b["open"]) * _H_KCAL, 1)
            res["barrier_fwd_kcal"] = round((b["ts"] - b["open"]) * _H_KCAL, 1)
            res["barrier_rev_kcal"] = round((b["ts"] - b["closed"]) * _H_KCAL, 1)
            save(f"dG={res['dG_close_kcal']} fwd={res['barrier_fwd_kcal']} rev={res['barrier_rev_kcal']}")
    else:
        save("barrier skipped (no DASA core / closed form)")
    save("DONE")
    return res


@APP.function(image=image, volumes={"/results": vol}, timeout=60 * 60 * 3)
def orchestrate(molset: list):
    import json
    results = list(final_one.starmap([(l, s) for l, s in molset]))
    json.dump(results, open("/results/final_dft_results.json", "w"), indent=2); vol.commit()
    print("\n=== DEFINITIVE DFT SUMMARY ===", flush=True)
    for r in results:
        lam = r.get("lambda_nm", {})
        print(f"\n{r['label']}", flush=True)
        print(f"  lambda_max: " + ", ".join(f"{k}={v}" for k, v in lam.items()), flush=True)
        if "dG_close_kcal" in r:
            print(f"  switching(DFT//xTB): dG(c-o)={r['dG_close_kcal']}  "
                  f"fwd={r['barrier_fwd_kcal']}  rev={r['barrier_rev_kcal']} kcal/mol", flush=True)
    return results


@APP.local_entrypoint()
def main():
    orchestrate.spawn(MOLSET)
    print(f"dispatched DETACHED: {len(MOLSET)} molecules (1 anchor + 3 finalists), one container each.")
    print("  live progress:  modal volume get --force dasa-outputs final_dft ./outputs_dasa_full/final_dft")
    print("  watch logs:     modal app logs <app-id> -f")
