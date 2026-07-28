"""Switching-KINETICS model on Modal — the missing piece.

Our anti-trap metric is THERMODYNAMIC only (ΔE open-vs-closed in water). Whether a DASA
*demonstrably switches* depends on the BARRIER connecting the open (colored) and closed
(colorless) forms. This does a GFN2-xTB relaxed scan along the electrocyclization coordinate
(the forming C1-C5 sigma bond) in ALPB water, from the open geometry (C1-C5 far) to the closed
ring (C1-C5 ~1.53 Å), and reads the energy profile:

    dE_water            = E(closed) - E(open)          [thermodynamics; >0 = open-favored]
    barrier_fwd (o->c)  = E(TS) - E(open)              [how easily light-driven closure completes]
    barrier_rev (c->o)  = E(TS) - E(closed)            [THERMAL REVERSION barrier -- the key one:
                                                        low enough => it reverts to open in water]

Interpretation for water-switching (open must be the resting colored state AND revert):
  * dE_water > 0 (open favored) AND a MODERATE reverse barrier (surmountable at RT, ~<25 kcal/mol)
    => sits open in water, can be switched, thermally reverts => SWITCHABLE.
  * reverse barrier very high => kinetically trapped closed even if open is thermodynamically favored.
  * forward barrier tiny => open form thermally unstable (spontaneously closes).

xTB barriers are approximate/directional -- refine top hits with DFT (TS opt) later. A first-gen
trap (DMA-barbituric) rides along as a NEGATIVE anchor.

    modal run --detach modal_kinetics.py --csv outputs_dasa_full/final_candidates.csv
"""
import os
import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-kinetics")
image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("rdkit=2024.03", "xtb", "numpy<2", channels=["conda-forge"])
    .add_local_dir(REPO, "/repo",
                   ignore=[".git", "__pycache__", "outputs_dasa*", "outputs", "outputs_rl2",
                           "*.model", "*.prior", "*.egg-info", "build"])
)
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)
_H_KCAL = 627.5094740631


def _mol_to_xyz(m):
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), ""]
    for a in m.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines) + "\n"


def _xtb_opt_E(xyz, chg, d, tag, extra=None):
    """Full xtb --opt in ALPB water. Returns (energy_Eh, optimized_xyz) or (None, None)."""
    import subprocess, os
    p = f"{d}/{tag}.xyz"
    open(p, "w").write(xyz)
    cmd = ["xtb", f"{tag}.xyz", "--opt", "--gfn", "2", "--alpb", "water",
           "--chrg", str(chg), "--uhf", "0"]
    if extra:
        cmd += ["--input", extra]
    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=60 * 12,
                           env={**os.environ, "OMP_NUM_THREADS": "4"})
    except Exception:
        return None, None
    e = None
    for line in r.stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                e = float(line.split()[-3])
            except Exception:
                pass
    opt = open(f"{d}/xtbopt.xyz").read() if os.path.exists(f"{d}/xtbopt.xyz") else None
    return e, opt


@APP.function(image=image, volumes={"/results": vol}, cpu=4.0, timeout=60 * 30, retries=1)
def kinetics_one(label: str, smiles: str):
    import sys, tempfile, math, json, hashlib, os
    sys.path.insert(0, "/repo/notebooks"); sys.path.insert(0, "/repo/plugins")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import dasa_chem as dc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"label": label, "error": "bad smiles"}
    match = mol.GetSubstructMatch(dc._DASA_OPEN)
    if not match or len(match) < 6:
        return {"label": label, "error": "no DASA core"}
    c1, c5 = match[1], match[5]
    chg = Chem.GetFormalCharge(mol)
    d = tempfile.mkdtemp(prefix="kin_")

    def embed(mm):
        h = Chem.AddHs(mm)
        if AllChem.EmbedMolecule(h, randomSeed=42) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(h, maxIters=1000)
        except Exception:
            pass
        return h

    # (1) fully relax the OPEN form
    mo = embed(mol)
    if mo is None:
        return {"label": label, "error": "open embed failed"}
    e_open, xyz_open = _xtb_opt_E(_mol_to_xyz(mo), chg, d, "open")
    if e_open is None:
        return {"label": label, "error": "open opt failed"}

    # (2) fully relax the CLOSED form(s): zwitterion (electrocyclization product) and,
    # if available, the neutral keto tautomer. dE uses the LOWER (real closed state).
    e_closed = None
    for gen, tag in ((dc.open_to_closed, "cz"), (dc.open_to_closed_neutral, "cn")):
        cs = gen(smiles)
        if not cs:
            continue
        mc = embed(Chem.MolFromSmiles(cs))
        if mc is None:
            continue
        ec, _ = _xtb_opt_E(_mol_to_xyz(mc), chg, d, tag)
        if ec is not None:
            e_closed = ec if e_closed is None else min(e_closed, ec)
    if e_closed is None:
        return {"label": label, "error": "closed opt failed"}
    dE = round((e_closed - e_open) * _H_KCAL, 1)      # >0 open-favored, <0 trapped

    # (3) barrier: relaxed scan along C1-C5 FROM the relaxed open geometry -> ring closure.
    #     TS = the scan maximum. Endpoints are the independently-relaxed open/closed above,
    #     so barriers are referenced to true minima (fixes the v1 clamped-endpoint flaw).
    conf = mo.GetConformer()
    p1, p5 = conf.GetAtomPosition(c1), conf.GetAtomPosition(c5)
    d0 = max(math.dist((p1.x, p1.y, p1.z), (p5.x, p5.y, p5.z)), 2.6)
    open(f"{d}/scan.inp", "w").write(
        f"$constrain\n  force constant=0.8\n  distance: {c1+1}, {c5+1}, {d0:.3f}\n"
        f"$scan\n  1: {d0:.3f}, 1.53, 18\n$end\n")
    _xtb_opt_E(xyz_open, chg, d, "scanstart", extra="scan.inp")
    prof = []
    if os.path.exists(f"{d}/xtbscan.log"):
        L = open(f"{d}/xtbscan.log").read().splitlines()
        i = 0
        while i < len(L):
            try:
                n = int(L[i].strip())
            except Exception:
                break
            for tok in L[i + 1].replace(",", " ").split():
                try:
                    prof.append(float(tok)); break
                except Exception:
                    continue
            i += n + 2
    e_ts = max(prof) if prof else None
    out = {"label": label, "smiles": smiles, "dE_water_kcal": dE}
    if e_ts is not None:
        out["barrier_fwd_kcal"] = round((e_ts - e_open) * _H_KCAL, 1)
        out["barrier_rev_kcal"] = round((e_ts - e_closed) * _H_KCAL, 1)
    try:
        os.makedirs("/results/kinetics", exist_ok=True)
        h = hashlib.md5(smiles.encode()).hexdigest()[:10]
        json.dump(out, open(f"/results/kinetics/{h}.json", "w"), indent=2)
        vol.commit()
    except Exception:
        pass
    print(f"[{label}] dE={dE} fwd={out.get('barrier_fwd_kcal')} rev={out.get('barrier_rev_kcal')}", flush=True)
    return out


@APP.function(image=image, volumes={"/results": vol}, timeout=60 * 60 * 2)
def orchestrate(molset: list):
    import json
    results = list(kinetics_one.starmap([(l, s) for l, s in molset]))
    json.dump(results, open("/results/kinetics_results.json", "w"), indent=2)
    vol.commit()
    ok = [r for r in results if "dE_water_kcal" in r]
    ok.sort(key=lambda r: r.get("barrier_rev_kcal", 1e9))   # low reverse barrier = reverts = switchable
    print("\n=== SWITCHING KINETICS (kcal/mol) — sorted by REVERSE barrier (low=reverts=switchable) ===", flush=True)
    print(f"{'molecule':30s} {'dE(c-o)':>8s} {'fwd(o->c)':>9s} {'rev(c->o)':>9s}", flush=True)
    for r in ok:
        print(f"{r['label'][:30]:30s} {str(r.get('dE_water_kcal')):>8} "
              f"{str(r.get('barrier_fwd_kcal')):>9} {str(r.get('barrier_rev_kcal')):>9}", flush=True)
    return results


@APP.local_entrypoint()
def main(csv: str = "outputs_dasa_full/final_candidates.csv"):
    import csv as _csv
    molset = [("NEG-ANCHOR DMA-barbituric (traps in water)",
               "CN(C)C=CC=CC(O)=C1C(=O)N(C)C(=O)N(C)C1=O")]
    for i, r in enumerate(_csv.DictReader(open(csv))):
        s = r.get("SMILES") or r.get("smiles")
        if s:
            molset.append((f"{r.get('architecture','cand')}-{i} (Total {r.get('Total','?')})", s))
    print(f"kinetics scan on {len(molset)} molecules (1 anchor + {len(molset)-1} candidates)")
    orchestrate.spawn(molset)
    print("dispatched DETACHED. results: dasa-outputs volume -> kinetics_results.json + kinetics/<hash>.json")
