"""Donor-diverse corpus for training a BLA surrogate, every donor cited.

WHY THIS EXISTS. The 224-molecule aqueous corpus spans only FOUR donor classes, and
a ridge model on it gave R2 0.88-0.92 held out by acceptor or backbone but **-0.349
held out by donor class** -- worse than predicting the mean. The surrogate does not
extrapolate to donor chemistry it has never seen, which is exactly the azole-drift
failure mode. The fix is donor breadth, not more molecules.

LITERATURE BASIS is recorded per donor, in four honest tiers:
  measured      this exact donor appears in a DASA with published data
  analogous     a close homologue of a measured donor (e.g. dibutyl vs diethyl)
  extrapolation chemically reasonable, no DASA precedent found
  KNOWN-FAILURE published evidence that the adduct does NOT form -- excluded

KNOWN-FAILURE is not hypothetical. Peterson, Neris & Read de Alaniz, Chem. Sci. 2023,
14, 13025: "we did not observe any formation of the 4-cyano or 4-nitro aniline
analogues after 3 days due to their poor nucleophilicity." Both of those donors are
currently in dasa_chem.AQUEOUS_DONOR_FRAGMENTS (lines ~460 and ~463) and would be
enumerated into any campaign built from it. They are excluded here and flagged.

ACCEPTORS are restricted to the four that dasa_common.acceptor_evidence() rates
'confirmed' (Meldrum, 1,3-dimethylbarbituric, N-phenyl-methylpyrazolone,
methylisoxazolone). Indandione is 'contraindicated' (Chem Soc Rev 2023 calls it a
non-photochromic DASA with a neutral ground state); thiobarbituric, pyrazolidinedione
and oxindole are 'uncharacterised'. A surrogate trained on unverified acceptors would
be learning chemistry nobody has shown works.

BACKBONES: unsubstituted, and the C5-methyl that Peterson/Stricker/Read de Alaniz,
Chem. Commun. 2022, 58, 2303 showed takes the same donor/acceptor pair from 29% to
95% open. C3- and C4-methyl are deliberately EXCLUDED: the same paper's DFT scan
found substituents there stabilise the closed form, which is the wrong direction.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import dasa_chem as dc

# name -> (SMILES prefix ending on the donor N, class, basis, citation)
DONORS = {
 # ---- 1st generation, dialkyl / cyclic aliphatic -------------------------
 "dimethylamino":      ("CN(C)",            "dialkyl", "analogous",
                        "Helmy JOC 2014 79 11316 (diethyl/morpholine homologue)"),
 "diethylamino":       ("CCN(CC)",          "dialkyl", "measured",
                        "Helmy JOC 2014 79 11316; Org. Synth. 2022 99 79"),
 "dipropylamino":      ("CCCN(CCC)",        "dialkyl", "analogous", "Helmy JOC 2014 homologue"),
 "dibutylamino":       ("CCCCN(CCCC)",      "dialkyl", "analogous", "Helmy JOC 2014 homologue"),
 "pyrrolidino":        ("C1CCCN1",          "cyclic_alkyl", "measured",
                        "Mallo Chem Sci 2018 9 8242 cmpd 12 (switches poorly)"),
 "piperidino":         ("C1CCCCN1",         "cyclic_alkyl", "measured",
                        "Mallo Chem Sci 2018 9 8242 cmpd 13"),
 "morpholino":         ("C1COCCN1",         "cyclic_alkyl", "measured", "Helmy JOC 2014 79 11316"),
 "azepano":            ("C1CCCCCN1",        "cyclic_alkyl", "analogous", "Helmy JOC 2014 homologue"),
 "adamantylamino":     ("N(C45CC6CC(CC(C6)C4)C5)", "monoalkyl", "measured",
                        "Blandon-Cumbreras JACS 2026 148 130 (slope -79/-65 nm)"),
 # ---- hydrophilic 1st-gen handles ---------------------------------------
 "diethanolamino":     ("OCCN(CCO)",        "dialkyl_polar", "extrapolation", "no DASA precedent found"),
 "N_Me_hydroxyethyl":  ("CN(CCO)",          "dialkyl_polar", "extrapolation", "no DASA precedent found"),
 "N_Me_PEG2":          ("CN(CCOCCO)",       "dialkyl_polar", "extrapolation", "no DASA precedent found"),
 # ---- 2nd generation: fused aryl (the best-characterised class) ----------
 "indoline":           ("C1Cc2ccccc2N1",    "fused_aryl", "measured",
                        "Peterson Chem Commun 2022 58 2303 (D1); Nat Commun 2024 cmpd 10"),
 "2-methylindoline":   ("CC1Cc2ccccc2N1",   "fused_aryl", "measured",
                        "Hemmer JACS 2018 140 10425; Berraud-Pache Chem Sci 2021 12 2916 (D1)"),
 "isoindoline":        ("C1c2ccccc2CN1",    "cyclic_benzylic", "measured",
                        "Nat Commun 2024 cmpd 9 (573 nm)"),
 "tetrahydroquinolino":("C1CCc2ccccc2N1",   "fused_aryl", "measured", "Nat Commun 2024"),
 "5-F-indoline":       ("C1Cc2cc(F)ccc2N1", "fused_aryl", "extrapolation", "EWG variant, no DASA precedent"),
 "5-CF3-indoline":     ("C1Cc2cc(C(F)(F)F)ccc2N1", "fused_aryl", "extrapolation", "EWG variant"),
 "5-CN-indoline":      ("C1Cc2cc(C#N)ccc2N1", "fused_aryl", "extrapolation", "EWG variant"),
 # ---- 2nd generation: anilines ------------------------------------------
 "N_methylanilino":    ("CN(c1ccccc1)",     "aniline", "measured",
                        "Mallo Chem Sci 2018 9 8242 cmpd 14 (57% cyclic, keto by X-ray)"),
 "anilino":            ("N(c1ccccc1)",      "aniline", "measured",
                        "Peterson Chem Sci 2023 14 13025 (tethered variants)"),
 "4-methoxyanilino":   ("N(c1ccc(OC)cc1)",  "aniline", "measured",
                        "Mallo Chem Sci 2018 (4-MeO-N-Me-aniline, 588 nm)"),
 "4-bromoanilino":     ("N(c1ccc(Br)cc1)",  "aniline", "analogous",
                        "halogen analogue of Peterson Chem Sci 2023 cmpd 5 (4-iodo)"),
 "4-chloroanilino":    ("N(c1ccc(Cl)cc1)",  "aniline", "analogous", "halogen analogue of cmpd 5"),
 "4-iodoanilino":      ("N(c1ccc(I)cc1)",   "aniline", "measured",
                        "Peterson Chem Sci 2023 cmpd 5: slope -20 nm, switches in 60:40 THF:water"),
 "4-CF3-anilino":      ("N(c1ccc(C(F)(F)F)cc1)", "aniline", "extrapolation", "EWG aniline"),
 "3-CF3-anilino":      ("N(c1cccc(C(F)(F)F)c1)", "aniline", "extrapolation", "EWG aniline"),
 "4-acetylanilino":    ("N(c1ccc(C(C)=O)cc1)", "aniline", "extrapolation", "EWG aniline"),
 "4-sulfamoylanilino": ("N(c1ccc(S(N)(=O)=O)cc1)", "aniline", "extrapolation",
                        "EWG + water solubility handle"),
 # ---- electron-poor heteroaryl amines ------------------------------------
 "3-aminopyridyl":     ("N(c1cccnc1)",      "heteroaryl", "extrapolation", "no DASA precedent"),
 "4-aminopyridyl":     ("N(c1ccncc1)",      "heteroaryl", "extrapolation", "no DASA precedent"),
}

# Documented NOT to form the adduct -- kept here so nobody re-adds them.
EXCLUDED = {
 "4-cyanoanilino":  ("N(c1ccc(C#N)cc1)",
    "Peterson Chem Sci 2023 14 13025: 'we did not observe any formation of the "
    "4-cyano or 4-nitro aniline analogues after 3 days due to their poor "
    "nucleophilicity'. Currently present in dasa_chem.AQUEOUS_DONOR_FRAGMENTS."),
 "4-nitroanilino":  ("N(c1ccc([N+](=O)[O-])cc1)", "same citation as 4-cyanoanilino"),
}

# acceptor_evidence() == 'confirmed' only
ACCEPTORS = {
 "meldrum":            "=C1C(=O)OC(C)(C)OC1=O",
 "dimethylbarbituric": "=C1C(=O)N(C)C(=O)N(C)C1=O",
 "methylpyrazolone":   "=C1C(=O)N(c2ccccc2)N=C1C",
 "methylisoxazolone":  "=C1C(=O)ON=C1C",
}
# unsubstituted, and the C5 methyl shown to raise the equilibrium 29% -> 95%
BACKBONES = {"triene": "C=CC=C(O)C", "C5_methyl": "C(C)=CC=C(O)C"}


def build():
    out, seen = [], set()
    for dn, (dfrag, dcls, basis, cite) in DONORS.items():
        for an, afrag in ACCEPTORS.items():
            for bn, bfrag in BACKBONES.items():
                smi = f"{dfrag}{bfrag}{afrag}"
                m = Chem.MolFromSmiles(smi)
                if m is None or not dc.is_dasa(m) or dc.is_legacy_core(m):
                    continue
                if dc.has_forbidden(m) or not dc.prior_supported(m):
                    continue
                canon = Chem.MolToSmiles(m)
                if canon in seen:
                    continue
                seen.add(canon)
                out.append({"smiles": canon, "donor": dn, "donor_class": dcls,
                            "basis": basis, "citation": cite,
                            "acceptor": an, "backbone": bn})
    return out


def main():
    rows = build()
    from collections import Counter
    print(f"{len(rows)} molecules from {len(DONORS)} donors x {len(ACCEPTORS)} "
          f"confirmed acceptors x {len(BACKBONES)} backbones")
    print(f"  donor classes: {dict(Counter(r['donor_class'] for r in rows))}")
    print(f"  basis tiers:   {dict(Counter(r['basis'] for r in rows))}")
    print(f"  EXCLUDED donors (documented not to form): {list(EXCLUDED)}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "donor_corpus.smi")
    with open(path, "w") as f:
        f.write("# Donor-diverse DASA corpus for BLA surrogate training.\n")
        f.write("# Built by notebooks/donor_corpus.py -- every donor carries a\n")
        f.write("# literature basis tier and citation; see that file.\n")
        f.write("# Acceptors restricted to acceptor_evidence()=='confirmed'.\n")
        for r in rows:
            f.write(f"{r['smiles']}\t{r['donor']}|{r['donor_class']}|{r['acceptor']}|"
                    f"{r['backbone']}|{r['basis']}\n")
    print(f"wrote {path}")
    import json
    json.dump(rows, open(path.replace(".smi", "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
