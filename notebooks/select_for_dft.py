#!/usr/bin/env python
"""Pick a defensible DFT set from the shortlist, and draw it.

THE SELECTION CRITERION THAT MATTERS
------------------------------------
Our TD-DFT calibration ladder is built entirely from CARBOCYCLIC aryl-amine and
alkyl-amine donors on barbituric / pyrazolone acceptors (Chem Sci 2018 cmpds 1
and 14; Nat Commun 2024 cmpds 9, 10, 11). A calibration is only valid over the
chemical domain it was fitted on.

The RL run drove donors to the LOW-BASICITY EDGE -- aminotriazoles,
aminopyrimidines, aminotetrazoles. Those maximise the trap-escape term, but no
measured DASA in our ladder has an azole/azine donor, so a calibrated lambda for
them is an EXTRAPOLATION, not a measurement-anchored estimate. Worse, the open
question about them is precisely whether such a weak donor still pushes enough
density to absorb in the visible -- the same class of risk as pyrazolidinedione.

So the DFT set is chosen to be mostly INSIDE the calibration domain, plus a
deliberate boundary probe:

  * in-domain  : donor N on a carbocyclic aromatic ring (like every reference)
  * rigidity   : donor N in a ring (the indoline motif, a measured 615 nm DASA)
  * probe      : the best azole/azine-donor candidate, flagged, to test whether
                 the low-basicity edge keeps its colour

Usage:
    python notebooks/select_for_dft.py --csv outputs_dasa_modal/outputs_dasa/verified_candidates.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugins"))
import dasa_chem as dc  # noqa: E402
from reinvent_plugins.components.dasa_common import (  # noqa: E402
    zwitterion_character, acceptor_evidence, delta_pka)
import dft_verify_v2 as V  # noqa: E402

_CARBOARYL_N = Chem.MolFromSmarts("[NX3]-c1ccccc1")   # donor N on a benzene ring


def dft_cost(mol):
    """Heavy-atom count AFTER chromophore truncation -- the real DFT cost driver.

    TD-DFT cost scales steeply with size, so this must be screened BEFORE a
    container is spawned. A previous run put four untruncated molecules on Modal
    and burned 5 hours without a single completed calculation. Truncation cuts the
    38-heavy candidates to 20-26, i.e. reference-sized.
    """
    t = V.truncate_chromophore(V.assign_literature_stereo(mol))
    return (t.GetNumHeavyAtoms() if t is not None else mol.GetNumHeavyAtoms()), t is not None


def domain_flags(mol):
    """Where does this molecule sit relative to the calibration domain?"""
    return dict(
        carbocyclic_aryl_donor=bool(mol.HasSubstructMatch(_CARBOARYL_N)),
        rigid_donor=bool(dc.is_rigid_donor(mol)),
        basicity_class=dc.classify_donor_architecture(mol),
        acceptor=dc.classify_acceptor(mol),
        acceptor_evidence=acceptor_evidence(mol),
        delta_pka=round(delta_pka(mol), 2),
        zwitterion=round(zwitterion_character(mol), 3),
        heavy=mol.GetNumHeavyAtoms(),
        MW=round(Descriptors.MolWt(mol), 1),
    )


def in_domain_score(mol, row):
    """Higher = better anchored to the measured ladder AND better by our own score."""
    f = domain_flags(mol)
    s = 0.0
    s += 2.0 if f["carbocyclic_aryl_donor"] else 0.0   # matches every reference
    s += 1.5 if f["rigid_donor"] else 0.0              # indoline motif, 615 nm
    s += 1.0 if f["acceptor_evidence"] == "confirmed" else 0.0
    s += 0.8 * float(row.get("TrapEscape") or 0)
    s += 0.5 * float(row.get("Solubility") or 0)
    s -= 0.02 * max(0, f["heavy"] - 30)                # DFT cost + synthesisability
    return s


def cluster_reps(mols, cutoff=0.5):
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in sims)
    return Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)


def draw(mols, legends, path, per_row=4, size=(430, 330)):
    img = Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=size,
                               legends=legends, useSVG=False, returnPNG=False)
    if hasattr(img, "save"):                 # PIL image (RDKit >=2023 in a script)
        img.save(path)
    else:                                    # bytes / IPython PNG payload
        data = img.data if hasattr(img, "data") else img
        with open(path, "wb") as fh:
            fh.write(data)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs_dasa_modal/outputs_dasa/verified_candidates.csv")
    ap.add_argument("--outdir", default="outputs_dasa_modal/outputs_dasa")
    ap.add_argument("--n-dft", type=int, default=5)
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--max-heavy", type=int, default=34,
                    help="skip molecules bigger than this AFTER truncation -- the "
                         "DFT affordability screen")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    smi_col = next(c for c in df.columns if c.upper() == "SMILES")
    df["_mol"] = df[smi_col].map(Chem.MolFromSmiles)
    df = df[df["_mol"].notna()].reset_index(drop=True)
    print(f"shortlist: {len(df)} molecules")

    # ---- 1. top-ranked grid -------------------------------------------------
    top = df.head(12)
    draw(list(top["_mol"]),
         [f"#{i+1}  TrapEsc {r.get('TrapEscape', float('nan')):.2f}  "
          f"MW {r.get('MW', 0):.0f}\n{dc.classify_donor_architecture(r['_mol'])}"
          f" / {dc.classify_acceptor(r['_mol'])}"
          for i, r in top.iterrows()],
         os.path.join(args.outdir, "top_ranked.png"))

    # ---- 2. cluster representatives (scaffold diversity) --------------------
    clusters = cluster_reps(list(df["_mol"]), args.cutoff)
    print(f"Butina @ {args.cutoff}: {len(clusters)} clusters")
    reps, legends = [], []
    for ci, cl in enumerate(clusters[:12]):
        best = df.iloc[list(cl)].iloc[0]          # csv is already rank-ordered
        reps.append(best["_mol"])
        legends.append(f"cluster {ci+1} (n={len(cl)})  MW {best.get('MW', 0):.0f}\n"
                       f"{dc.classify_donor(best['_mol'])} / "
                       f"{dc.classify_acceptor(best['_mol'])}")
    draw(reps, legends, os.path.join(args.outdir, "cluster_representatives.png"))

    # ---- 3. FINAL CANDIDATES (the canonical post-generation artifact) -------
    recs = []
    for _, r in df.iterrows():
        m = r["_mol"]
        heavy_t, ok_t = dft_cost(m)
        f = domain_flags(m)
        recs.append(dict(SMILES=r[smi_col], **f, heavy_truncated=heavy_t,
                         truncatable=ok_t,
                         dft_affordable=bool(heavy_t <= args.max_heavy),
                         TrapEscape=r.get("TrapEscape"),
                         Solubility=r.get("Solubility"), SA=r.get("SA")))
    fc = pd.DataFrame(recs)
    fc_path = os.path.join(args.outdir, "final_candidates.csv")
    fc.to_csv(fc_path, index=False)
    print(f"\n  wrote {fc_path}  ({len(fc)} rows)")
    print(f"  DFT-affordable after truncation (<= {args.max_heavy} heavy): "
          f"{int(fc['dft_affordable'].sum())} / {len(fc)}")
    top12 = df.head(12)
    draw(list(top12["_mol"]),
         [f"#{i+1}  dpKa {delta_pka(r['_mol']):+.2f}  {dc.classify_acceptor(r['_mol'])}\n"
          f"{dc.classify_donor_architecture(r['_mol'])}"
          f"{' rigid' if dc.is_rigid_donor(r['_mol']) else ''}  "
          f"{dft_cost(r['_mol'])[0]}heavy"
          for i, r in top12.iterrows()],
         os.path.join(args.outdir, "final_candidates.png"))

    # ---- 4. the DFT set -----------------------------------------------------
    df["_domain"] = [in_domain_score(m, r) for m, (_, r) in zip(df["_mol"], df.iterrows())]
    df["_carboaryl"] = [m.HasSubstructMatch(_CARBOARYL_N) for m in df["_mol"]]
    df["_rigid"] = [dc.is_rigid_donor(m) for m in df["_mol"]]

    df["_afford"] = [dft_cost(m)[0] <= args.max_heavy for m in df["_mol"]]
    n_drop = int((~df["_afford"]).sum())
    if n_drop:
        print(f"  excluded {n_drop} molecules as too expensive for DFT "
              f"(> {args.max_heavy} heavy after truncation)")
    df = df[df["_afford"]].reset_index(drop=True)
    in_dom = df[df["_carboaryl"]].sort_values("_domain", ascending=False)
    out_dom = df[~df["_carboaryl"]].sort_values("_domain", ascending=False)
    print(f"\nin calibration domain (carbocyclic aryl donor): {len(in_dom)} / {len(df)}")
    print(f"rigid donors in shortlist                     : {int(df['_rigid'].sum())}")

    picks = []
    for _, r in in_dom.iterrows():
        if len(picks) >= args.n_dft - 1:
            break
        picks.append(("in-domain (carbocyclic aryl donor)", r))
    # Prefer rigid donors next: the indoline motif is the one structural feature
    # of the measured 615 nm reference that any of these still share.
    used = {id(p[1]) for p in picks}
    for _, r in out_dom[out_dom["_rigid"]].iterrows():
        if len(picks) >= args.n_dft:
            break
        if id(r) not in used:
            picks.append(("OUT-OF-DOMAIN, rigid donor", r))
    for _, r in out_dom.iterrows():
        if len(picks) >= args.n_dft:
            break
        picks.append(("OUT-OF-DOMAIN (azole/azine donor, no ladder precedent)", r))
    if not in_dom.shape[0]:
        print("\n  *** WARNING: NO candidate has a carbocyclic aryl-amine donor. Every")
        print("      measured DASA in the calibration ladder does. The whole shortlist")
        print("      sits outside the domain the calibration was fitted on, so a")
        print("      calibrated lambda for these is an EXTRAPOLATION. Treat the DFT")
        print("      result as a test of whether these are DASAs at all, not as a rank.")

    print(f"\n=== DFT set ({len(picks)}) ===")
    rows = []
    for tag, r in picks:
        f = domain_flags(r["_mol"])
        print(f"  [{tag}]")
        print(f"    {r[smi_col]}")
        print(f"    heavy {f['heavy']}  MW {f['MW']}  zwitterion {f['zwitterion']}  "
              f"rigid {f['rigid_donor']}  acceptor {f['acceptor']} ({f['acceptor_evidence']})")
        rows.append(dict(label=tag, SMILES=r[smi_col], **f))
    out_csv = os.path.join(args.outdir, "dft_set.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")
    draw([p[1]["_mol"] for p in picks],
         [f"{p[0][:26]}\nheavy {p[1]['_mol'].GetNumHeavyAtoms()}" for p in picks],
         os.path.join(args.outdir, "dft_set.png"), per_row=3)


if __name__ == "__main__":
    main()
