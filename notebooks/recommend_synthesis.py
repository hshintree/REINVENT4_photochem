#!/usr/bin/env python
"""Pick N molecules to synthesise and write the argument for each.

WHY THIS IS NOT A DFT SCRIPT
----------------------------
The strongest case for a DASA candidate is NOT a computed lambda_max. TD-DFT has a
~0.44 eV irreducible error on DASAs even when done perfectly (PMC5615680: B3LYP on
a proper DFT geometry gives 435 nm for a measured 515 nm compound; CASPT2 0.06 eV).
Anything we compute is worse evidence than the literature we can transfer from.

What we CAN say with high confidence, from measurements:

  COLOUR.  lambda_max is set by the ACCEPTOR, essentially independent of the donor.
    Chem Sci 2018 measured 13 barbituric DASAs spanning donors from Me/Me to
    Oct/Oct to pyrrolidine to tetrahydroisoquinoline: ALL 567 +- 3 nm. An aryl
    donor moves it to 588 (cmpd 14); indoline to 615 (Nat Commun 2024 cmpd 10).
    So a barbituric DASA with an intact aryl-amine donor is ~570-615 nm by direct
    analogy -- a tighter statement than TD-DFT can make.

  TRAP ESCAPE.  Set by which closed tautomer wins, i.e. by amine basicity vs carbon
    acid acidity (Peterson / Read de Alaniz ionic-character study). Our dpKa
    coordinate places each candidate against MEASURED switchers (indoline +0.89,
    4-MeO-aniline +1.37) and MEASURED trapped compounds (Me2N/barbituric +6.69,
    Me2N/CF3-pyrazolone +7.20).

  SYNTHESIS.  Every DASA is two steps from furfural (Org. Synth. 2022, 99, 79):
    Knoevenagel with the carbon acid, then ring-opening with the amine. So the
    retrosynthesis is fixed, and we can name both precursors.

A recommendation built from those three is defensible to a synthetic chemist in a
way "our model says 560 nm" is not. DFT, when it lands, is corroboration -- not the
foundation.

    python notebooks/recommend_synthesis.py --n 4
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugins"))
import dasa_chem as dc  # noqa: E402
import dft_verify_v2 as V  # noqa: E402
from reinvent_plugins.components.dasa_common import (  # noqa: E402
    delta_pka, acceptor_evidence, chromophore_integrity)

# MEASURED anchors, for the analogy argument.
ANCHORS = [
    ("Chem Sci 2018 cmpd 1  Me2N / 1,3-diMe-barbituric", 567, "CHCl3", 6.69,
     "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", "water-TRAPPED (1st-gen)"),
    ("Chem Sci 2018 cmpd 14 4-MeO-N-Me-aniline / barbituric", 588, "CHCl3", 1.37,
     "COc1ccc(N(C)C=CC=C(O)C=C2C(=O)N(C)C(=O)N(C)C2=O)cc1", "switches (2nd-gen)"),
    ("Nat Commun 2024 cmpd 10 indoline / barbituric", 615, "CH2Cl2", 0.89,
     "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C", "switches (2nd-gen)"),
]
# lambda_max by acceptor, from measurement (nm), for the transfer argument
ACCEPTOR_LAMBDA = {"barbituric": (567, "Chem Sci 2018, 13 compounds, 567 +- 3 nm"),
                   "thiobarbituric": (600, "Helmy JOC 2014 (approximate)"),
                   "meldrum": (550, "1st-gen archetype, Org. Synth. 2022"),
                   "pyrazolone": (646, "Nat Commun 2024 cmpd 11, with indoline"),
                   "isoxazolone": (630, "3rd-gen acceptor, approximate")}
ARYL_DONOR_SHIFT = 21          # 567 -> 588 measured, alkyl -> aryl donor


def fp(m):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048)


def nearest_anchor(mol):
    f = fp(mol)
    best = max(ANCHORS, key=lambda a: DataStructs.TanimotoSimilarity(
        f, fp(Chem.MolFromSmiles(a[4]))))
    return best, DataStructs.TanimotoSimilarity(f, fp(Chem.MolFromSmiles(best[4])))


def predicted_lambda(mol):
    """lambda by literature transfer, NOT by calculation."""
    acc = dc.classify_acceptor(mol)
    if acc not in ACCEPTOR_LAMBDA:
        return None, None, f"acceptor '{acc}' has no measured DASA lambda"
    base, src = ACCEPTOR_LAMBDA[acc]
    aryl = dc.classify_donor_architecture(mol) == "aniline"
    lam = base + (ARYL_DONOR_SHIFT if aryl else 0)
    note = (f"{acc} acceptor -> {base} nm [{src}]"
            + (f"; aryl donor +{ARYL_DONOR_SHIFT} nm (567->588 measured)" if aryl else ""))
    return lam, acc, note


def trap_verdict(d):
    if d > 4.0:
        return "TRAPPED-leaning", "close to the measured 1st-gen trapped regime (+6.7)"
    if -0.5 <= d <= 2.5:
        return "SWITCHABLE window", ("brackets the two measured switchers "
                                     "(indoline +0.89, 4-MeO-aniline +1.37)")
    if d < -0.5:
        return "donor may be too weak", ("below every measured DASA; push-pull may be "
                                         "too weak to absorb in the visible")
    return "borderline", "between the measured switchable and trapped regimes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs_dasa_modal/outputs_dasa/final_candidates.csv")
    ap.add_argument("--outdir", default="outputs_dasa_modal/outputs_dasa")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-sim-gap", type=float, default=0.55,
                    help="max Tanimoto between chosen molecules (diversity)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["_mol"] = df["SMILES"].map(Chem.MolFromSmiles)
    df = df[df["_mol"].notna()].reset_index(drop=True)

    # HARD requirements for anything we would ask a chemist to make.
    keep = []
    for _, r in df.iterrows():
        m = r["_mol"]
        d = delta_pka(m)
        retro = dc.dasa_retrosynthesis(m)
        ok = (dc.is_dasa(m) and not dc.is_legacy_core(m)
              and chromophore_integrity(m)
              and acceptor_evidence(m) == "confirmed"      # measured lambda exists
              and dc.classify_acceptor(m) in ACCEPTOR_LAMBDA
              and -0.5 <= d <= 2.5                          # measured switchable window
              and retro is not None and retro.get('amine')  # 2-step route exists
              and r.get("SA", 9) is not None and float(r.get("SA", 9)) <= 4.5
              and 300 <= float(r["MW"]) <= 600)
        if ok:
            keep.append((r, m, d, retro))
    print(f"{len(df)} candidates -> {len(keep)} pass every hard requirement")
    if not keep:
        sys.exit("nothing passes; loosen the filters or regenerate")

    # rank: prefer literature-like donors, rigidity, simplicity
    def score(t):
        r, m, d, _ = t
        s = 0.0
        s += 2.0 if bool(r["carbocyclic_aryl_donor"]) else 0.0
        s += 1.0 if bool(r["rigid_donor"]) else 0.0
        s += 1.5 * (1.0 - abs(d - 1.13) / 2.0)      # centre of the measured window
        s -= 0.004 * float(r["MW"])
        s -= 0.15 * float(r.get("SA", 3))
        return s

    keep.sort(key=score, reverse=True)

    # DIVERSITY IS CHECKED ON THE TRUNCATED CHROMOPHORE, not the whole molecule.
    # Full-molecule Tanimoto let two picks through that differed only in peripheral
    # solubilising groups; they collapsed to the SAME structure once truncated for
    # DFT and returned identical numbers. If two candidates share a chromophore they
    # are one experiment, not two, so a chemist should not be asked to make both.
    def core_fp(m):
        t = V.truncate_chromophore(V.assign_literature_stereo(m))
        return fp(t if t is not None else m)

    chosen = []
    for t in keep:
        if len(chosen) >= args.n:
            break
        f, cf = fp(t[1]), core_fp(t[1])
        if all(DataStructs.TanimotoSimilarity(f, fp(c[1])) < args.min_sim_gap
               and DataStructs.TanimotoSimilarity(cf, core_fp(c[1])) < 0.85
               for c in chosen):
            chosen.append(t)

    lines = ["# Synthesis recommendation", "",
             f"{len(chosen)} molecules, selected from {len(df)} shortlisted "
             f"({len(keep)} passed every hard requirement).", "",
             "## The basis for these claims", "",
             "* **Colour** is transferred from measurement, not computed. lambda_max is",
             "  set by the acceptor and is nearly donor-independent: 13 barbituric DASAs",
             "  spanning Me/Me to Oct/Oct to pyrrolidine all sit at 567 +- 3 nm",
             "  (Chem. Sci. 2018, 9, 8242). An aryl donor adds ~21 nm (588, cmpd 14);",
             "  indoline reaches 615 (Nat. Commun. 2024, cmpd 10).",
             "* **Trap escape** is placed on the dpKa coordinate against measured",
             "  compounds: switchers at +0.89 and +1.37, trapped at +6.69 and +7.20.",
             "* **Synthesis** is the fixed DASA route (Org. Synth. 2022, 99, 79):",
             "  furfural + carbon acid (Knoevenagel), then amine (Stenhouse ring-open).",
             "", "---", ""]

    for i, (r, m, d, retro) in enumerate(chosen, 1):
        lam, acc, note = predicted_lambda(m)
        verdict, why = trap_verdict(d)
        (alab, alam, asolv, adpka, _asmi, abehav), sim = nearest_anchor(m)
        lines += [
            f"## {i}. {Chem.MolToSmiles(m)}", "",
            f"| | |", "|---|---|",
            f"| **Make it from** | `{retro['amine']}` (amine) + `{retro['carbon_acid']}` (carbon acid) |",
            f"| **Route** | {retro['route']} — 2 steps |",
            f"| **Expected lambda_max** | **~{lam} nm** — {note} |",
            f"| **Trap escape** | dpKa {d:+.2f} → {verdict}; {why} |",
            f"| **Closest measured DASA** | {alab} ({alam} nm in {asolv}, dpKa {adpka:+.2f}, {abehav}); Tanimoto {sim:.2f} |",
            f"| **Donor** | {r['basicity_class']}{', rigid (ring-locked)' if r['rigid_donor'] else ''}"
            f"{', carbocyclic aryl' if r['carbocyclic_aryl_donor'] else ''} |",
            f"| **MW / SA / heavy** | {r['MW']:.0f} / {float(r.get('SA', 0)):.2f} / {int(r['heavy'])} |",
            "",
            f"**Why this one:** the acceptor is {acc}, which has a measured DASA "
            f"lambda_max, so the colour claim is transfer rather than extrapolation. "
            f"The donor puts it at dpKa {d:+.2f}, inside the window bracketed by the "
            f"two measured switchers, and far from the +6.7 measured trapped regime. "
            f"Both precursors are simple fragments, so it is two steps from furfural.",
            "",
        ]

    lines += ["---", "", "## Honest limitations", "",
              "* dpKa is an ESTIMATE (Hammett + literature carbon-acid pKa). Its",
              "  aza-heteroaryl corrections are the least certain values in it.",
              "* Colour transfer assumes the chromophore is intact. That is checked",
              "  structurally (donor not acylated, conjugation continuous, planar",
              "  conformer available) but not spectroscopically.",
              "* No candidate has a measured lambda_max. TD-DFT would corroborate but",
              "  cannot beat the literature transfer: its irreducible error on DASAs is",
              "  ~0.44 eV, wider than the 567-615 nm range we are transferring within.",
              "* Pure-water switching remains unprecedented; these target the window,",
              "  they do not guarantee it.", ""]

    md = os.path.join(args.outdir, "SYNTHESIS_RECOMMENDATION.md")
    with open(md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {md}")

    rows = [dict(rank=i, SMILES=Chem.MolToSmiles(m), amine=t[3]["amine"],
                 carbon_acid=t[3]["carbon_acid"], delta_pka=round(t[2], 2),
                 predicted_lambda_nm=predicted_lambda(m)[0],
                 acceptor=dc.classify_acceptor(m),
                 donor=t[0]["basicity_class"], rigid=bool(t[0]["rigid_donor"]),
                 MW=float(t[0]["MW"]), SA=float(t[0].get("SA", 0)))
            for i, t in enumerate(chosen, 1) for m in [t[1]]]
    csv = os.path.join(args.outdir, "synthesis_shortlist.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")

    img = Draw.MolsToGridImage(
        [t[1] for t in chosen], molsPerRow=2, subImgSize=(520, 400),
        legends=[f"#{i}  ~{predicted_lambda(t[1])[0]} nm  dpKa {t[2]:+.2f}\n"
                 f"{dc.classify_acceptor(t[1])} / {t[0]['basicity_class']}"
                 for i, t in enumerate(chosen, 1)], useSVG=False, returnPNG=False)
    png = os.path.join(args.outdir, "synthesis_shortlist.png")
    img.save(png) if hasattr(img, "save") else open(png, "wb").write(img)
    print(f"wrote {png}")
    for i, (r, m, d, retro) in enumerate(chosen, 1):
        print(f"  {i}. lam~{predicted_lambda(m)[0]} nm  dpKa {d:+.2f}  "
              f"MW {r['MW']:.0f}  <- {retro['amine']} + {retro['carbon_acid']}")


if __name__ == "__main__":
    main()
