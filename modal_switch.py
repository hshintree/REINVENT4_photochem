"""Switchability test on Modal -- the REAL water-switching metric (not color).

For each DASA we build the open triene and the closed cyclopentenone ZWITTERION
(dasa_common.open_to_closed), GFN2-xTB geometry-OPTIMISE both in water and in
toluene (ALPB implicit solvent, native `xtb --opt`), and take

    dE(solvent) = E(closed, opt) - E(open, opt)   [kcal/mol]

Interpretation (the water-trapping question):
  * dE_water strongly NEGATIVE  -> closed zwitterion deeply favoured in water = TRAPPED
                                   (dark, won't switch back -- the failure mode).
  * dE_water near 0 / positive   -> open form accessible in water = switchable-friendly.
  * dE_water - dE_toluene (the differential) shows how much water tips it toward closed.

This is a GROUND-STATE energy difference, so it is UNAFFECTED by the ~186 nm TD-DFT
lambda_max blue-shift (that was an excited-state artifact). It has its OWN error source
(zwitterion implicit-solvation), so we ANCHOR with a first-gen DASA that is experimentally
>99% closed (trapped) in water -- if the method nails that as deeply negative, the ranking
of the rest is trustworthy. `--dft` adds a B3LYP/6-31+G(d,p) ddCOSMO single point on the
xTB-optimised geometry for a refined dE (slower).

Run:
    modal run --detach modal_switch.py                # xTB-opt dE only (~30-60 min)
    modal run --detach modal_switch.py --dft          # + DFT single-point refinement
"""
import os
import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-switch")

image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("rdkit=2024.03", "xtb", "numpy<2", channels=["conda-forge"])
    .pip_install("pyscf==2.5.0", "scipy")
    .add_local_dir(
        REPO, "/repo",
        ignore=[".git", "__pycache__", ".ipynb_checkpoints", "outputs_dasa",
                "outputs_dasa_modal", "outputs_dasa_full", "outputs", "outputs_rl2",
                "build", "*.egg-info", "*.model", "*.prior"],
    )
)
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)
_H_KCAL = 627.5094740631

# The small evaluation set. Anchors first so we always calibrate.
MOLSET = [
    ("DMA-Meldrum ANCHOR (exp >99% closed/TRAPPED in water)", "CN(C)C=CC=CC(O)=C1C(=O)OC(C)(C)OC1=O"),
    ("DMA-barbituric reference",                              "CN(C)C=CC=CC(O)=C1C(=O)N(C)C(=O)N(C)C1=O"),
    ("your OXAZINE probe",                                    "O=C(CN1C=C(C=CC(O)=C2NCCC2=O)CCO1)NCCCO"),
    ("DIHYDROPYRROLE probe",                                  "O=C(CN1C=C(C=CC(O)=C2NCCC2=O)CC1)NCCCO"),
    ("barbituric candidate A",  "CN(C=CC=CC(O)=C1C(=O)N(C)C(=O)N(C)C1=O)C(=O)CCN1CCN(C(=O)CN2CCNCC2)CC1"),
    ("barbituric candidate B",  "CN1C(=O)C(=C(O)C=CC=CNC(=O)Cn2[nH]nnc2=S)C(=O)N(C)C1=O"),
    ("barbituric candidate C",  "CN1C(=O)C(=C(O)C=CC=CNC(=O)CNC(=O)CN2CCCC2=O)C(=O)NC1=S"),
]


def _mmff_xyz(mol):
    """RDKit MMFF starting geometry -> xyz string + total formal charge."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            return None, None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
    except Exception:
        pass
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), ""]
    for a in m.GetAtoms():
        pos = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines) + "\n", Chem.GetFormalCharge(mol)


def _xtb_opt(xyz, charge, solvent, tag):
    """Native `xtb --opt` in ALPB solvent. Returns (energy_Eh, optimized_xyz) or (None, None)."""
    import subprocess, tempfile, os
    d = tempfile.mkdtemp(prefix=f"xtb_{tag}_")
    with open(f"{d}/mol.xyz", "w") as f:
        f.write(xyz)
    cmd = ["xtb", "mol.xyz", "--opt", "tight", "--gfn", "2",
           "--alpb", solvent, "--chrg", str(charge), "--uhf", "0"]
    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=2400,
                           env={**os.environ, "OMP_NUM_THREADS": "4"})
    except subprocess.TimeoutExpired:
        return None, None
    e = None
    for line in r.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                e = float(line.split()[-3])
            except Exception:
                pass
    opt_xyz = None
    p = f"{d}/xtbopt.xyz"
    if os.path.exists(p):
        opt_xyz = open(p).read()
    return e, opt_xyz


def _dft_sp(xyz, charge, solvent_eps, xc="b3lyp", basis="6-31+g(d,p)"):
    """DFT single point (ddCOSMO) on a given geometry -> energy in Hartree."""
    from pyscf import gto, dft
    from pyscf.solvent import ddCOSMO
    atoms = "\n".join(xyz.splitlines()[2:])   # strip natoms + comment line
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=0, verbose=0)
    mf = ddCOSMO(dft.RKS(mol)); mf.xc = xc
    mf.with_solvent.eps = solvent_eps
    mf.conv_tol = 1e-8
    return mf.kernel()


@APP.function(image=image, volumes={"/results": vol}, cpu=4.0,
              timeout=60 * 60 * 4, retries=1)
def switch_one(label: str, smiles: str, do_dft: bool = False):
    import sys, json, hashlib
    sys.path.insert(0, "/repo/plugins")
    from rdkit import Chem
    from reinvent_plugins.components.dasa_common import open_to_closed
    SOLV = {"water": 78.4, "toluene": 2.38}

    m = Chem.MolFromSmiles(smiles)
    closed = open_to_closed(m) if m else None
    out = {"label": label, "smiles": smiles, "closed_ok": closed is not None}
    if m is None or closed is None:
        out["error"] = "open/closed build failed"
        return out

    forms = {"open": m, "closed": closed}
    xtb_E, dft_E = {}, {}
    for fname, fmol in forms.items():
        xyz0, chg = _mmff_xyz(fmol)
        if xyz0 is None:
            out["error"] = f"embed failed ({fname})"
            return out
        for sol in SOLV:
            e, oxyz = _xtb_opt(xyz0, chg, sol, f"{fname}_{sol}")
            xtb_E[(fname, sol)] = e
            if do_dft and oxyz is not None:
                try:
                    dft_E[(fname, sol)] = _dft_sp(oxyz, chg, SOLV[sol])
                except Exception as ex:
                    dft_E[(fname, sol)] = None
                    print(f"  [{label}] DFT SP failed {fname}/{sol}: {type(ex).__name__}", flush=True)

    def dE(store, sol):
        o, c = store.get(("open", sol)), store.get(("closed", sol))
        return round((c - o) * _H_KCAL, 1) if (o is not None and c is not None) else None

    out["xtb_dE_water_kcal"] = dE(xtb_E, "water")
    out["xtb_dE_toluene_kcal"] = dE(xtb_E, "toluene")
    if out["xtb_dE_water_kcal"] is not None and out["xtb_dE_toluene_kcal"] is not None:
        out["xtb_water_minus_toluene"] = round(out["xtb_dE_water_kcal"] - out["xtb_dE_toluene_kcal"], 1)
    if do_dft:
        out["dft_dE_water_kcal"] = dE(dft_E, "water")
        out["dft_dE_toluene_kcal"] = dE(dft_E, "toluene")

    try:
        os.makedirs("/results/switch", exist_ok=True)
        h = hashlib.md5(smiles.encode()).hexdigest()[:10]
        json.dump(out, open(f"/results/switch/{h}.json", "w"), indent=2)
        vol.commit()
    except Exception:
        pass
    print(f"[{label}] xtb dE_water={out.get('xtb_dE_water_kcal')} "
          f"dE_toluene={out.get('xtb_dE_toluene_kcal')} kcal/mol", flush=True)
    return out


@APP.function(image=image, volumes={"/results": vol}, timeout=60 * 60 * 8)
def orchestrate(molset: list, do_dft: bool):
    import json
    results = list(switch_one.starmap([(lbl, smi, do_dft) for lbl, smi in molset]))
    json.dump(results, open("/results/switch_results.json", "w"), indent=2)
    vol.commit()
    print("\n=== SWITCHABILITY (dE = E_closed - E_open; very negative in water = TRAPPED) ===",
          flush=True)
    print(f"{'molecule':50s} {'dE_water':>9s} {'dE_tol':>8s} {'w-tol':>7s}", flush=True)
    for r in results:
        print(f"{r['label'][:50]:50s} {str(r.get('xtb_dE_water_kcal')):>9s} "
              f"{str(r.get('xtb_dE_toluene_kcal')):>8s} {str(r.get('xtb_water_minus_toluene')):>7s}",
              flush=True)
    return results


@APP.local_entrypoint()
def main(dft: bool = False):
    orchestrate.spawn(MOLSET, dft)
    print(f"dispatched DETACHED (dft={dft}). {len(MOLSET)} molecules.")
    print("  results: dasa-outputs volume -> switch_results.json + switch/<hash>.json")
    print("  watch:   modal app logs <app-id> -f")
