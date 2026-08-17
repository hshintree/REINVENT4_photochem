#!/usr/bin/env python
"""Turn a candidate shortlist into PLATES for automated parallel synthesis.

Post-processing only. Reads final_candidates.csv, writes plate files. Imports the
existing modules read-only and changes nothing upstream.

WHY PLATES AND NOT MOLECULES
----------------------------
DASA synthesis is convergent in exactly the way a liquid handler wants
(Org. Synth. 2022, 99, 79):

    step 1  (BULK, one per acceptor)   furfural + carbon acid -> furfurylidene
    step 2  (PARALLEL, one per well)   furfurylidene + amine  -> DASA

So a plate is ONE bulk intermediate prep plus N pipetted amines, and the binding
constraint is not molecular complexity -- it is whether the amine is purchasable.
Ranking molecules individually ignores that a plate sharing an intermediate is
essentially free per extra well. We therefore rank PLATES.

THE PLATE IS A DESIGNED EXPERIMENT, NOT 96 LOOKALIKES
-----------------------------------------------------
The obvious failure mode is filling 96 wells with the same molecule wearing
different hats. Three things prevent it, and they follow from the physics:

  1. The ACCEPTOR IS FIXED per plate (that is what makes it cheap), and lambda_max
     is set by the acceptor -- 13 measured barbituric DASAs span Me/Me to Oct/Oct
     to pyrrolidine and all sit at 567 +- 3 nm. So colour is NOT the within-plate
     variable; there is nothing to learn by varying it here.

  2. The within-plate variable is dpKa -- the trap axis, the thing we cannot
     measure any other way and whose band is currently anchored on TWO literature
     compounds. Wells are STRATIFIED across the dpKa range so one plate maps that
     axis. The plate IS the dpKa calibration experiment.

  3. Amines must be structurally distinct from each other: Butina clustering on the
     AMINE (not the whole DASA, which trivially differs by the amine anyway), with
     a hard minimum pairwise Tanimoto. One representative per cluster.

ON-PLATE CONTROLS
-----------------
Every plate carries the measured literature compounds, made from catalogue amines
through the same chemistry in the same run. If the controls do not reproduce their
published lambda_max, the PLATE failed -- not the candidates. This is the same
logic as the DFT calibration ladder, except the controls here are real molecules
being measured rather than a method being calibrated.

    python notebooks/ht_plate_planner.py --wells 96 --plates 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugins"))
import dasa_chem as dc  # noqa: E402
from reinvent_plugins.components.dasa_common import (  # noqa: E402
    delta_pka, acceptor_evidence, chromophore_integrity)

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
    _HAS_SA = True
except Exception:
    _HAS_SA = False

FURFURAL = "O=Cc1ccco1"

# MEASURED controls. amine + carbon acid are both catalogue compounds, so these ride
# along on any plate whose acceptor matches, at the cost of one well each.
CONTROLS = [
    dict(name="CTRL ChemSci-1", amine="CNC", acid="CN1C(=O)CC(=O)N(C)C1=O",
         dasa="CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O",
         lam_nm=567, t_half_s=32, note="1st-gen alkyl; water-TRAPPED. dpKa +6.69"),
    dict(name="CTRL ChemSci-14", amine="CNc1ccc(OC)cc1", acid="CN1C(=O)CC(=O)N(C)C1=O",
         dasa="COc1ccc(N(C)C=CC=C(O)C=C2C(=O)N(C)C(=O)N(C)C2=O)cc1",
         lam_nm=588, t_half_s=265, note="2nd-gen aryl; switches. dpKa +1.37"),
    dict(name="CTRL NatComm-10", amine="C1Cc2ccccc2N1", acid="CN1C(=O)CC(=O)N(C)C1=O",
         dasa="O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C",
         lam_nm=615, t_half_s=None, note="fused aryl (indoline); switches. dpKa +0.89"),
    dict(name="CTRL-NEG pyrrolidine", amine="C1CCNC1", acid="CN1C(=O)CC(=O)N(C)C1=O",
         dasa="O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCCC2)C(=O)N1C",
         lam_nm=567, t_half_s=92,
         note="NEGATIVE control: Chem Sci 2018 cmpd 12 DECOMPOSES rather than "
              "reverting cleanly. If this well looks healthy, the assay is wrong."),
]


def amine_buyability(smi):
    """Heuristic 0..1 that an amine is orderable. NOT a catalogue lookup.

    Deliberately crude and deliberately flagged: a real order needs a Sigma/Enamine/
    Fluorochem search. This only stops the planner proposing a plate whose amines are
    themselves multi-step syntheses -- which would defeat the point of automating
    step 2.
    """
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi
    if m is None:
        return 0.0
    heavy = m.GetNumHeavyAtoms()
    sa = sascorer.calculateScore(m) if _HAS_SA else 3.0
    nring = rdMolDescriptors.CalcNumRings(m)
    stereo = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    s = 1.0
    s -= 0.04 * max(0, heavy - 12)        # small amines are catalogue items
    s -= 0.18 * max(0.0, sa - 2.5)        # synthetic accessibility
    s -= 0.10 * max(0, nring - 2)
    s -= 0.25 * stereo                    # chiral amines are pricier / often racemic
    return float(max(0.0, min(1.0, s)))


def furfurylidene(acid_smi):
    """The bulk step-1 intermediate: Knoevenagel of furfural with the carbon acid.

    Condenses the acid's CH2 with the furfural carbonyl. Returned for the ORDER LIST
    and the plate header; it is what gets made once and split across the wells.
    """
    acid = Chem.MolFromSmiles(acid_smi)
    fur = Chem.MolFromSmiles(FURFURAL)
    if acid is None or fur is None:
        return None
    patt = Chem.MolFromSmarts("[CX4H2](-[CX3]=O)-[CX3]=O")   # the active methylene
    hit = acid.GetSubstructMatch(patt)
    if not hit:
        return None
    combo = Chem.RWMol(Chem.CombineMols(acid, fur))
    off = acid.GetNumAtoms()
    ald_c = next((a.GetIdx() + off for a in fur.GetAtoms()
                  if a.GetSymbol() == "C" and any(
                      b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(a).GetSymbol() == "O"
                      for b in a.GetBonds())), None)
    if ald_c is None:
        return None
    ald_o = next(nb.GetIdx() for nb in combo.GetAtomWithIdx(ald_c).GetNeighbors()
                 if nb.GetSymbol() == "O")
    try:
        combo.RemoveAtom(ald_o)
        if ald_c > ald_o:
            ald_c -= 1
        combo.AddBond(hit[0], ald_c, Chem.BondType.DOUBLE)
        a = combo.GetAtomWithIdx(hit[0]); a.SetNumExplicitHs(0); a.SetNoImplicit(True)
        c = combo.GetAtomWithIdx(ald_c); c.SetNumExplicitHs(1); c.SetNoImplicit(True)
        out = combo.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return None


def fp(m):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048)


def stratified_diverse_pick(rows, n, min_sim_gap, n_strata):
    """Pick n amines that SPAN the dpKa axis and are structurally distinct.

    Stratify by dpKa, then round-robin the strata taking the best-scoring amine that
    is not too similar to anything already chosen. Round-robin rather than
    best-first is what stops the plate collapsing into whichever dpKa bin happens to
    contain the top scores -- that bin would be one experiment repeated n times.
    """
    if not rows:
        return []
    ds = np.array([r["delta_pka"] for r in rows])
    edges = np.linspace(ds.min(), ds.max() + 1e-9, n_strata + 1)
    strata = [[] for _ in range(n_strata)]
    for r in rows:
        k = min(int(np.searchsorted(edges, r["delta_pka"], side="right") - 1), n_strata - 1)
        strata[max(0, k)].append(r)
    for s in strata:
        s.sort(key=lambda r: -r["plate_score"])

    chosen, cursor = [], [0] * n_strata
    while len(chosen) < n and any(cursor[i] < len(strata[i]) for i in range(n_strata)):
        progressed = False
        for i in range(n_strata):
            if len(chosen) >= n:
                break
            while cursor[i] < len(strata[i]):
                cand = strata[i][cursor[i]]
                cursor[i] += 1
                if all(DataStructs.TanimotoSimilarity(cand["_afp"], c["_afp"]) < min_sim_gap
                       for c in chosen):
                    chosen.append(cand)
                    progressed = True
                    break
        if not progressed:
            break
    return chosen


def well_names(n, rows=8, cols=12):
    return [f"{chr(ord('A') + i // cols)}{i % cols + 1:02d}" for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs_dasa_modal/outputs_dasa/final_candidates.csv")
    ap.add_argument("--outdir", default="outputs_dasa_modal/plates")
    ap.add_argument("--wells", type=int, default=96)
    ap.add_argument("--plates", type=int, default=2)
    ap.add_argument("--min-sim-gap", type=float, default=0.55,
                    help="max pairwise Tanimoto between AMINES on one plate")
    ap.add_argument("--strata", type=int, default=6, help="dpKa bins to span")
    ap.add_argument("--min-buyability", type=float, default=0.35)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    rows = []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r["SMILES"])
        if m is None or not dc.is_dasa(m) or not chromophore_integrity(m):
            continue
        retro = dc.dasa_retrosynthesis(m)
        if not retro or not retro.get("amine"):
            continue          # tethered donors: N-C is endocyclic, no 2-step route
        am = Chem.MolFromSmiles(retro["amine"])
        if am is None:
            continue
        buy = amine_buyability(am)
        if buy < args.min_buyability:
            continue
        d = delta_pka(m)
        rows.append(dict(
            SMILES=r["SMILES"], amine=retro["amine"], acid=retro["carbon_acid"],
            acceptor=dc.classify_acceptor(m), acceptor_evidence=acceptor_evidence(m),
            delta_pka=round(d, 2), buyability=round(buy, 2),
            MW=round(Descriptors.MolWt(m), 1),
            donor=dc.classify_donor_architecture(m), rigid=bool(dc.is_rigid_donor(m)),
            plate_score=round(buy + 1.5 * (1 - abs(d - 1.13) / 3.0), 3),
            _afp=fp(am), _mol=m))
    print(f"{len(df)} candidates -> {len(rows)} with a real 2-step route and a "
          f"plausibly orderable amine")
    if not rows:
        sys.exit("nothing plateable")

    # group by ACCEPTOR: one bulk intermediate per group = one plate's step 1
    groups = {}
    for r in rows:
        groups.setdefault(r["acid"], []).append(r)
    ranked = sorted(groups.items(),
                    key=lambda kv: -(len(kv[1]) ** 0.5
                                     * np.mean([x["plate_score"] for x in kv[1]])))
    print(f"\n{len(ranked)} candidate plates (grouped by shared intermediate):")
    for acid, members in ranked[:6]:
        acc = members[0]["acceptor"]
        print(f"   {acc:16s} n={len(members):4d}  mean score "
              f"{np.mean([m['plate_score'] for m in members]):.2f}  acid {acid}")

    all_map, all_work, summaries = [], [], []
    for pi, (acid, members) in enumerate(ranked[:args.plates], 1):
        # CONTROLS GO ON EVERY PLATE, carrying THEIR OWN intermediate.
        # Matching controls to the plate's acid meant a plate whose acceptor differed
        # from the literature compounds got NO controls at all -- exactly the plate
        # you would least want to run blind. A control is only a control if it is the
        # literature molecule, so it needs the literature acid; that costs one extra
        # small bulk prep (4 wells) and makes every plate self-calibrating.
        ctrls = list(CONTROLS)
        n_cand = args.wells - len(ctrls)
        picked = stratified_diverse_pick(members, n_cand, args.min_sim_gap, args.strata)
        inter = furfurylidene(acid)
        wells = well_names(len(picked) + len(ctrls))
        acc = members[0]["acceptor"]
        plate_id = f"P{pi}_{acc}"

        entries = []
        for c in ctrls:
            entries.append(dict(role="control", name=c["name"], amine=c["amine"],
                                delta_pka=None,
                                expected_lambda_nm=c["lam_nm"],
                                expected_t_half_s=c["t_half_s"], note=c["note"],
                                dasa_smiles=c["dasa"],
                                _intermediate=furfurylidene(c["acid"])))
        for p in picked:
            entries.append(dict(role="candidate", name="", amine=p["amine"],
                                dasa_smiles=p["SMILES"], delta_pka=p["delta_pka"],
                                expected_lambda_nm=None, expected_t_half_s=None,
                                note=f"{p['donor']}{' rigid' if p['rigid'] else ''}",
                                _intermediate=inter))
        for w, e in zip(wells, entries):
            e = dict(e); wi = e.pop("_intermediate") or inter
            all_map.append(dict(plate=plate_id, well=w, intermediate=wi,
                                acceptor=acc, **e))
            all_work.append(dict(plate=plate_id, well=w,
                                 source_intermediate=wi, intermediate_uL=100,
                                 source_amine=e["amine"], amine_uL=100,
                                 solvent="THF", solvent_uL=800,
                                 temperature_C=25, time_min=60))

        ds = [p["delta_pka"] for p in picked]
        summaries.append(dict(plate=plate_id, acceptor=acc, intermediate=inter,
                              wells=len(entries), controls=len(ctrls),
                              candidates=len(picked),
                              dpka_min=min(ds) if ds else None,
                              dpka_max=max(ds) if ds else None,
                              dpka_span=round(max(ds) - min(ds), 2) if ds else None,
                              unique_amines=len({p["amine"] for p in picked})))
        print(f"\n=== {plate_id}: {len(picked)} candidates + {len(ctrls)} controls")
        print(f"    intermediate: {inter}")
        print(f"    dpKa span {min(ds):+.2f} .. {max(ds):+.2f}  "
              f"({max(ds)-min(ds):.2f} units across {args.strata} strata)")
        print(f"    unique amines: {len({p['amine'] for p in picked})}/{len(picked)}")
        print(f"    wells used {len(entries)}/{args.wells} -- the shortfall is "
              f"DELIBERATE: no amine within {args.min_sim_gap} Tanimoto of another "
              f"is added, so empty wells mean we ran out of genuinely distinct "
              f"chemistry, not that the plate is underpacked.")

    # STRUCTURE SHEET per plate -- every well drawn, for chemist review before
    # anything is ordered. Controls are drawn too (and labelled with their expected
    # lambda/t-half) so a reviewer can see the calibration set sitting on the plate.
    for s_ in summaries:
        pid = s_["plate"]
        wells_p = [e for e in all_map if e["plate"] == pid]
        mols, legends = [], []
        for e in wells_p:
            m = Chem.MolFromSmiles(e["dasa_smiles"]) if e.get("dasa_smiles") else None
            if m is None:
                continue
            mols.append(m)
            if e["role"] == "control":
                tail = (f"exp {e['expected_lambda_nm']:.0f} nm"
                        if pd.notna(e.get("expected_lambda_nm")) else "")
                th = e.get("expected_t_half_s")
                if pd.notna(th):
                    tail += f", t1/2 {float(th):.0f}s"
                legends.append(f"{e['well']}  {e['name']}\n{tail}")
            else:
                legends.append(f"{e['well']}  dpKa {e['delta_pka']:+.2f}\n{e['note']}")
        if not mols:
            continue
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(360, 300),
                                   legends=legends, useSVG=False, returnPNG=False)
        path = os.path.join(args.outdir, f"{pid}_structures.png")
        img.save(path) if hasattr(img, "save") else open(path, "wb").write(img)
        print(f"   {pid}_structures.png  ({len(mols)} molecules)")

    pm = pd.DataFrame(all_map)
    pm.to_csv(os.path.join(args.outdir, "plate_map.csv"), index=False)
    pd.DataFrame(all_work).to_csv(os.path.join(args.outdir, "worklist.csv"), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(args.outdir, "plate_summary.csv"), index=False)

    amines = pm[["amine"]].drop_duplicates().reset_index(drop=True)
    amines["buyability_heuristic"] = [amine_buyability(a) for a in amines["amine"]]
    amines["CHECK_CATALOGUE"] = "verify vs Sigma/Enamine/Fluorochem before ordering"
    amines.to_csv(os.path.join(args.outdir, "amine_order_list.csv"), index=False)

    inter_list = pd.DataFrame([dict(acceptor=s["acceptor"], intermediate=s["intermediate"],
                                    step="furfural + carbon acid, Knoevenagel",
                                    scale="bulk, one prep per plate")
                               for s in summaries]).drop_duplicates()
    inter_list.to_csv(os.path.join(args.outdir, "intermediates.csv"), index=False)

    # results template -- the shape the plate reader / LCMS data comes back in
    tmpl = pm[["plate", "well", "role", "name", "amine", "dasa_smiles",
               "delta_pka", "expected_lambda_nm", "expected_t_half_s"]].copy()
    for col in ("obs_formed_lcms", "obs_lambda_max_nm", "obs_abs_max",
                "obs_bleach_percent", "obs_t_half_s", "obs_decomposed", "notes"):
        tmpl[col] = ""
    tmpl.to_csv(os.path.join(args.outdir, "results_template.csv"), index=False)

    print(f"\nwrote -> {args.outdir}/")
    for f in ("plate_map.csv", "worklist.csv", "plate_summary.csv",
              "amine_order_list.csv", "intermediates.csv", "results_template.csv"):
        print(f"   {f}")
    print(f"\n{len(amines)} unique amines to source across {len(summaries)} plates")


if __name__ == "__main__":
    main()
