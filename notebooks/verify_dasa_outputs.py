#!/usr/bin/env python
"""Verify DASA RL outputs — quality control on what the pipeline actually produced.

The RL summary CSVs contain every generated molecule, but a high Score does NOT
mean a good DASA: the loose scaffold gate + placeholder switchability window let
RL reward-hack (aziridine/primary-amine "donors", saturated WaterSwitch, etc.).
This script separates credible candidates from artifacts and tells you whether a
run is trustworthy.

What it does:
  1. Per-stage summary + component-score distributions, with SATURATION warnings
     (a component whose mean~1 / std~0 is no longer discriminating = reward-hacking).
  2. A strict chemical-quality filter (beyond the RL gate): real secondary/tertiary
     amine donor, sane ring sizes, MW/SA window, intact carbon-acid.
  3. Novelty vs the training corpus, and Butina clustering for scaffold diversity.
  4. Writes a ranked shortlist CSV + a structure-grid PNG of the top credible hits.

Usage (reinvent4 env):
  python notebooks/verify_dasa_outputs.py                      # auto-find outputs
  python notebooks/verify_dasa_outputs.py --dir outputs_dasa_modal/outputs_dasa
  python notebooks/verify_dasa_outputs.py --xtb-check 10       # re-score top 10 w/ xTB
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from rdkit import DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dasa_chem as dc  # noqa: E402

# SA score from RDKit contrib (optional)
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
    _HAS_SA = True
except Exception:
    _HAS_SA = False

_DASA_OPEN = Chem.MolFromSmarts(dc.DASA_OPEN_SMARTS)


# --------------------------------------------------------------------------
def donor_is_real_amine(mol) -> bool:
    """The triene-terminal N must be a genuine secondary/tertiary amine donor:
    >=2 heavy neighbours and not in a 3-membered ring (rejects primary enamines
    like NC=C and strained aziridine 'donors' the RL likes to exploit)."""
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match:
        return False
    n = mol.GetAtomWithIdx(match[0])
    if n.GetDegree() < 2:                       # primary enamine (NH2-C=C)
        return False
    if n.IsInRingSize(3):                       # aziridine-type
        return False
    return True


def quality(smiles, mw_range=(200, 600), sa_max=6.0):
    """Return (ok, reason, props) for a strict credible-DASA check."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False, "invalid", {}
    if not dc.is_dasa(mol):
        return False, "not_dasa", {}
    if dc.has_forbidden(mol):
        return False, "forbidden_group", {}
    if not donor_is_real_amine(mol):
        return False, "bad_donor", {}
    # colour gate: reject non-canonical ("other") acceptors -- those are the
    # weak/unusual carbon acids the RL used to cheat anti-trap, and they absorb
    # in the UV, not the visible (see DASAColor / the 311 nm DFT result).
    if dc.classify_acceptor(mol) == "other":
        return False, "uv_acceptor", {}
    mw = Descriptors.MolWt(mol)
    if not (mw_range[0] <= mw <= mw_range[1]):
        return False, f"mw_{mw:.0f}", {}
    # reject 3/4-membered carbocycles/heterocycles anywhere (strain artifacts)
    ri = mol.GetRingInfo()
    if any(len(r) < 5 for r in ri.AtomRings()):
        return False, "small_ring", {}
    sa = sascorer.calculateScore(mol) if _HAS_SA else float("nan")
    if _HAS_SA and sa > sa_max:
        return False, f"sa_{sa:.1f}", {}
    props = {
        "MW": round(mw, 1), "SA": round(sa, 2) if _HAS_SA else None,
        "logP": round(Descriptors.MolLogP(mol), 2),
        "donor": dc.classify_donor(mol), "acceptor": dc.classify_acceptor(mol),
        "donor_arch": dc.classify_donor_architecture(mol),   # mechanism, for DFT stratification
        "canon": Chem.MolToSmiles(mol),
    }
    return True, "ok", props


# --------------------------------------------------------------------------
def load_stage(base, s):
    fs = sorted(glob.glob(os.path.join(base, f"trial_stage{s}", f"stage{s}_*.csv")))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def summarize_stage(df, label):
    if df.empty:
        print(f"\n{label}: no data"); return
    n = len(df)
    valid = df[df.get("SMILES_state", pd.Series([1] * n)) == 1]
    uniq = valid.drop_duplicates("SMILES")
    dasa = int((uniq.get("DASA", pd.Series(dtype=float)) == 1.0).sum())
    sc = "Score" if "Score" in df.columns else None
    print(f"\n{label}: {n} scored, {len(valid)} valid, {len(uniq)} unique, "
          f"{dasa} DASA-gate pass"
          + (f", top Score {uniq[sc].max():.3f}" if sc else ""))
    # component distributions + saturation warning
    comps = [c for c in ["DASA", "Solubility", "xTB_Gap", "AntiTrap",
                          "WaterSwitch", "DecompAlerts", "SA"]
             if c in uniq.columns]
    for c in comps:
        v = pd.to_numeric(uniq[c], errors="coerce").dropna()
        if v.empty:
            continue
        flag = ""
        if c not in ("DASA",) and (v.mean() > 0.9 and v.std() < 0.06):
            flag = "  <-- SATURATED (not discriminating; recalibrate this component)"
        print(f"    {c:12s} mean={v.mean():.3f} std={v.std():.3f} "
              f"[{v.min():.2f}, {v.max():.2f}]{flag}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="outputs dir (auto-detected if omitted)")
    ap.add_argument("--top", type=int, default=25, help="shortlist size to print")
    ap.add_argument("--save-top", type=int, default=500,
                    help="rows to write to verified_candidates.csv (0 = all); "
                         "keeps the file small — DFT only ever uses the best few hundred")
    ap.add_argument("--xtb-check", type=int, default=0, help="re-score N top hits with xTB")
    args = ap.parse_args()

    # locate outputs
    root = os.path.dirname(HERE)
    cands = [args.dir] if args.dir else [
        os.path.join(root, "outputs_dasa_modal", "outputs_dasa"),
        os.path.join(root, "outputs_dasa"),
    ]
    base = next((d for d in cands if d and os.path.isdir(d)), None)
    if base is None:
        sys.exit(f"no outputs dir found (looked in {cands})")
    print(f"verifying outputs in: {base}")
    if not _HAS_SA:
        print("(note: RDKit SA scorer unavailable — skipping SA filter)")

    stages = {s: load_stage(base, s) for s in (1, 2, 3)}
    for s in (1, 2, 3):
        summarize_stage(stages[s], f"Stage {s}")

    # Build the credible-candidate pool. Prefer Stage 2 (has WaterSwitch), then
    # merge Stage 3 (more chemistry-optimised) so we keep each molecule's best row.
    frames = []
    for s in (2, 3, 1):
        d = stages[s]
        if not d.empty:
            d = d.copy(); d["__stage"] = s
            frames.append(d)
    if not frames:
        sys.exit("no stage data to verify")
    allmol = pd.concat(frames, ignore_index=True)
    allmol = allmol[allmol.get("SMILES_state", 1) == 1].dropna(subset=["SMILES"])

    print(f"\n=== applying strict quality filter to {allmol['SMILES'].nunique()} "
          "unique generated molecules ===")
    rows, reasons = [], {}
    for smi, grp in allmol.groupby("SMILES"):
        ok, why, props = quality(smi)
        if not ok:
            reasons[why.split("_")[0]] = reasons.get(why.split("_")[0], 0) + 1
            continue
        best = grp.sort_values("Score", ascending=False).iloc[0]
        def col(name):
            return round(float(best[name]), 3) if name in best and pd.notna(best[name]) else np.nan
        rec = {"SMILES": props["canon"], **props,
               "Score": round(float(best.get("Score", 0)), 3),
               "Solubility": col("Solubility"),
               "AntiTrap": col("AntiTrap"),        # the calibrated anti-trap metric
               "WaterSwitch": col("WaterSwitch"),  # legacy (older runs)
               "xTB_Gap": col("xTB_Gap"),
               "from_stage": int(best["__stage"])}
        rows.append(rec)
    credible = pd.DataFrame(rows).drop_duplicates("SMILES")
    print(f"  credible DASAs: {len(credible)}")
    print("  rejected (by reason):", dict(sorted(reasons.items(), key=lambda x: -x[1])))

    if credible.empty:
        print("\nNo credible candidates survived — the run is dominated by artifacts. "
              "Recalibrate scoring (esp. any SATURATED component) and re-run.")
        return

    # Rank for the water-soluble/switchable goal: the calibrated AntiTrap metric
    # is the primary signal (falls back to legacy WaterSwitch, then neutral),
    # balanced with solubility and a sane open-form gap.
    def obj(r):
        sol = r["Solubility"] if pd.notna(r["Solubility"]) else 0
        gap = r["xTB_Gap"] if pd.notna(r["xTB_Gap"]) else 0.5
        anti = r["AntiTrap"] if pd.notna(r["AntiTrap"]) else (
            r["WaterSwitch"] if pd.notna(r["WaterSwitch"]) else 0.5)
        return 0.45 * anti + 0.35 * sol + 0.2 * gap
    credible["water_obj"] = credible.apply(obj, axis=1)
    credible = credible.sort_values("water_obj", ascending=False).reset_index(drop=True)

    # Novelty vs training corpus
    corpus_file = os.path.join(base, "trial_corpus.smi")
    novel_frac = None
    if os.path.isfile(corpus_file):
        corpus = set()
        for line in open(corpus_file):
            m = Chem.MolFromSmiles(line.strip())
            if m:
                corpus.add(Chem.MolToSmiles(m, isomericSmiles=False))
        cred_flat = credible["SMILES"].map(
            lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False))
        novel = ~cred_flat.isin(corpus)
        novel_frac = novel.mean()
        credible["novel"] = novel.values

    # Diversity: Butina clustering of the shortlist
    top = credible.head(max(args.top, 20))
    mols = [Chem.MolFromSmiles(s) for s in top["SMILES"]]
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    from rdkit.ML.Cluster import Butina
    nclust = len(Butina.ClusterData(dists, len(fps), 0.4, isDistData=True)) if fps else 0

    print(f"\n=== VERDICT ===")
    print(f"  credible DASAs: {len(credible)} / {allmol['SMILES'].nunique()} unique generated"
          f" ({100*len(credible)/max(allmol['SMILES'].nunique(),1):.1f}%)")
    if novel_frac is not None:
        print(f"  novel vs training corpus (top {len(top)}): {100*novel_frac:.0f}%")
    print(f"  scaffold clusters in top {len(top)}: {nclust}")
    print(f"\n  Top {args.top} credible candidates for water-soluble/switchable DASA:")
    show = [c for c in ["SMILES", "donor", "donor_arch", "acceptor", "MW", "SA", "logP",
                        "Solubility", "AntiTrap", "xTB_Gap", "from_stage"]
            if c in credible.columns]
    print(credible[show].head(args.top).to_string(index=False))

    # Write shortlist + structure grid (capped so the CSV stays small)
    out_csv = os.path.join(base, "verified_candidates.csv")
    to_save = credible if args.save_top <= 0 else credible.head(args.save_top)
    to_save.to_csv(out_csv, index=False)
    print(f"\n  ranked shortlist -> {out_csv}  ({len(to_save)} of {len(credible)} rows)")
    try:
        grid = Draw.MolsToGridImage(
            mols[:20], molsPerRow=5, subImgSize=(280, 220),
            legends=[f"{r.donor[:10]}/{r.acceptor[:8]}\nSol {r.Solubility}"
                     for _, r in top.head(20).iterrows()])
        png = os.path.join(base, "verified_top.png")
        grid.save(png)
        print(f"  structure grid   -> {png}")
    except Exception as e:
        print(f"  (structure grid skipped: {e})")

    # Optional xTB re-check of the very top hits (confirm switchability isn't a fluke)
    if args.xtb_check > 0:
        print(f"\n=== xTB re-check of top {args.xtb_check} (independent of RL scores) ===")
        sys.path.insert(0, os.path.join(root, "plugins", "reinvent_plugins", "components"))
        try:
            from dasa_common import embed_3d, xtb_properties
        except Exception as e:
            print(f"  xtb helpers unavailable: {e}"); return
        for _, r in credible.head(args.xtb_check).iterrows():
            m3d = embed_3d(Chem.MolFromSmiles(r["SMILES"]))
            if m3d is None:
                print(f"  {r['SMILES'][:45]:45s} embed failed"); continue
            w = xtb_properties(m3d, "water"); t = xtb_properties(m3d, "toluene")
            if not w or not t:
                print(f"  {r['SMILES'][:45]:45s} xtb failed"); continue
            dip = w[1]; dsolv = (w[0] - t[0]) * 627.509
            print(f"  {r['SMILES'][:45]:45s} dipole(H2O)={dip:.2f}au  ΔsolvG={dsolv:+.1f}kcal")


if __name__ == "__main__":
    main()
