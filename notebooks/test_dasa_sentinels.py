#!/usr/bin/env python
"""Sentinel regression suite — run BEFORE launching any DASA campaign, and after
any edit to the core chemistry or the scoring components.

This is NOT a pipeline stage: it runs outside the generator loop, takes seconds,
and costs nothing in-loop. Its job is to fail loudly when the chemistry or the
scorer drifts away from molecules that have been MEASURED.

Why it exists: for months the corpus encoded a constitutional isomer of a DASA
(hydroxyl on the carbon bonded to the acceptor instead of on C2). Every internal
check passed, because every check compared our molecules to each other. The one
quantity ever compared against a real measurement — colour — was 229 nm off, and
that was blamed on DFT. A single assertion that a literature compound parses as a
DASA would have caught it immediately.

    python notebooks/test_dasa_sentinels.py          # or: pytest notebooks/test_dasa_sentinels.py
"""
from __future__ import annotations
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugins"))

import dasa_chem as dc  # noqa: E402

# ---------------------------------------------------------------------------
# Measured literature compounds. lambda_max are for HYDROXY DASAs (our chemistry).
# NOTE: Nat Commun 2024 also reports AMINO DASAs (NH in place of OH) at 531/578/608
# -- those are a DIFFERENT chromophore and must NOT be used to calibrate us.
# ---------------------------------------------------------------------------
MEASURED = [
    # (label, SMILES, lambda_max nm, solvent, basicity class, rigid?)
    ("OrgSyn-2  Et2N / Meldrum",
     "CCN(CC)C=CC=C(O)C=C1C(=O)OC(C)(C)OC1=O", None, "CDCl3", "dialkyl", False),
    ("ChemSci-1  Me2N / 1,3-diMe-barbituric",
     "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", 567, "CHCl3", "dialkyl", False),
    ("ChemSci-12 pyrrolidine / 1,3-diMe-barbituric",
     "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCCC2)C(=O)N1C", 567, "CHCl3", "dialkyl", True),
    ("ChemSci-14 4-MeO-N-Me-aniline / 1,3-diMe-barbituric",
     "COc1ccc(N(C)C=CC=C(O)C=C2C(=O)N(C)C(=O)N(C)C2=O)cc1", 588, "CHCl3", "aniline", False),
    ("NatComm-9  isoindoline / 1,3-diMe-barbituric",
     "O=C1N(C)C(=O)C(=CC(O)=CC=CN2Cc3ccccc3C2)C(=O)N1C", 573, "CH2Cl2", "dialkyl", True),
    ("NatComm-10 indoline / 1,3-diMe-barbituric",
     "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C", 615, "CH2Cl2", "aniline", True),
]

# Things that must NEVER be accepted as DASAs.
NEGATIVES = [
    ("legacy wrong core (pre-2026-07-28 corpus)",
     "CN(C)C=CC=CC(O)=C1C(=O)N(C)C(=O)N(C)C1=O"),
    ("azobenzene", "c1ccc(/N=N/c2ccccc2)cc1"),
    ("plain aminotriene, no acceptor", "CN(C)C=CC=CC=C"),
    ("enaminone", "CN(C)C=CC(=O)C"),
]

# Donor-integrity exploits that previously slipped past the colour gate.
DONOR_EXPLOITS = [
    ("acylated donor N (enaminone, UV)",
     "O=C(C)N(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O"),
    ("N-O donor (1,2-oxazine / alkoxyamine)",
     "CON(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O"),
]

_failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _failures.append(msg)


def test_measured_compounds_are_recognised():
    """Every MEASURED literature DASA must parse as a DASA. This is the assertion
    whose absence let a non-DASA corpus survive for months."""
    print("\n[1] measured literature DASAs are recognised as DASAs")
    for label, smi, *_ in MEASURED:
        mol = Chem.MolFromSmiles(smi)
        check(mol is not None and dc.is_dasa(mol), f"{label} parses as a DASA")
        check(mol is not None and not dc.is_legacy_core(mol), f"{label} is not legacy core")


def test_negatives_rejected():
    print("\n[2] decoys and the legacy skeleton are rejected")
    for label, smi in NEGATIVES:
        mol = Chem.MolFromSmiles(smi)
        check(not dc.is_dasa(mol), f"rejected: {label}")
    check(dc.is_legacy_core(Chem.MolFromSmiles(NEGATIVES[0][1])),
          "legacy skeleton is positively FLAGGED (not merely unmatched)")


def test_closed_forms():
    """Both closed tautomers must be true isomers and 4,5-disubstituted
    cyclopentenones, with the zwitterion an AMMONIUM (not an iminium)."""
    print("\n[3] closed forms match the literature cyclopentenone")
    iminium = Chem.MolFromSmarts("[NX3+]=[CX3]")
    ammonium = Chem.MolFromSmarts("[NX4+;H1]")
    for label, smi, *_ in MEASURED:
        z = dc.open_to_closed(smi)
        k = dc.open_to_closed_keto(smi)
        check(z is not None and k is not None, f"{label}: both tautomers build")
        if z is None:
            continue
        zm = Chem.MolFromSmiles(z)
        check(dc.is_cyclopentenone_closed(zm), f"{label}: 4,5-disubstituted cyclopentenone")
        check(zm.HasSubstructMatch(ammonium) and not zm.HasSubstructMatch(iminium),
              f"{label}: zwitterion is an AMMONIUM, not an iminium")


def test_donor_axes():
    """Basicity and rigidity are independent axes; indoline must read as BOTH."""
    print("\n[4] donor axes (basicity x rigidity) are reported independently")
    for label, smi, _lam, _solv, basicity, rigid in MEASURED:
        mol = Chem.MolFromSmiles(smi)
        ax = dc.donor_axes(mol)
        check(ax["basicity"] == basicity,
              f"{label}: basicity={ax['basicity']} (expected {basicity})")
        check(bool(ax["rigid"]) == rigid,
              f"{label}: rigid={ax['rigid']} (expected {rigid})")


def test_colour_gate():
    """Colour is a GATE. It must pass measured DASAs, reject crippled donors, and
    exclude acceptors whose visibility is not literature-backed."""
    print("\n[5] colour gate: passes measured DASAs, rejects donor exploits")
    from reinvent_plugins.components.dasa_common import (
        has_canonical_acceptor, has_visible_donor)
    for label, smi, *_ in MEASURED:
        mol = Chem.MolFromSmiles(smi)
        check(has_canonical_acceptor(mol) and has_visible_donor(mol),
              f"{label}: passes the colour gate")
    for label, smi in DONOR_EXPLOITS:
        mol = Chem.MolFromSmiles(smi)
        check(not has_visible_donor(mol), f"donor exploit rejected: {label}")
    # pyrazolidinedione: the acceptor a prior run collapsed onto, visibility uncited
    pzd = Chem.MolFromSmiles("CN(C)C=CC=C(O)C=C1C(=O)NNC1=O")
    check(not has_canonical_acceptor(pzd),
          "unverified acceptor (pyrazolidinedione) excluded from the gate by default")
    check(has_canonical_acceptor(pzd, require_verified=False),
          "...but still reachable explicitly via require_verified=False")


def test_corpus_is_clean():
    print("\n[6] the enumerated corpus carries the corrected core")
    rows = dc.enumerate_dasa_aqueous()
    mols = [Chem.MolFromSmiles(r["smiles_open"]) for r in rows]
    check(len(rows) > 0, f"corpus is non-empty ({len(rows)} molecules)")
    check(all(m is not None and dc.is_dasa(m) for m in mols),
          "every corpus molecule matches the corrected core")
    check(not any(m is not None and dc.is_legacy_core(m) for m in mols),
          "no corpus molecule carries the legacy core")
    check(not any(r["decomp_liable"] for r in rows), "no decomposition-liable molecules")
    rigid_aniline = sum(1 for m in mols
                        if m is not None and dc.is_rigid_donor(m)
                        and dc.classify_donor_architecture(m) == "aniline")
    check(rigid_aniline > 0,
          f"corpus contains the rigid+weakly-basic corner (indoline-type): {rigid_aniline}")


def test_trap_escape_reproduces_generation_ordering():
    """The structural trap term must reproduce the MEASURED ionic-character
    ordering (Peterson / Read de Alaniz): first- AND third-generation architectures
    are strongly zwitterionic (water-trapped); second-generation is not."""
    print("\n[7] trap escape reproduces the literature generation ordering")
    # tests the LIVE coordinate (delta_pka), which is what DASATrapEscape scores
    from reinvent_plugins.components.dasa_common import (
        delta_pka as dpka, acceptor_pka)
    first = dpka(Chem.MolFromSmiles("CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O"))
    third = dpka(Chem.MolFromSmiles(
        "CN(C)C=CC=C(O)C=C1C(=O)N(c2ccccc2)N=C1C(F)(F)F"))
    second = dpka(Chem.MolFromSmiles(
        "COc1ccc(N(C)C=CC=C(O)C=C2C(=O)N(C)C(=O)N(C)C2=O)cc1"))
    indoline = dpka(Chem.MolFromSmiles(
        "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C"))
    weak = dpka(Chem.MolFromSmiles(
        "CN1C(=O)C(=CC(O)=CC=CNc2nc(N)nc(N)n2)C(=O)N(C)C1=O"))
    check(first > second, f"1st-gen more charge-separated than 2nd ({first:+.2f} > {second:+.2f})")
    check(third > second, f"3rd-gen more charge-separated than 2nd ({third:+.2f} > {second:+.2f})")
    check(third > first, f"3rd-gen most charge-separated ({third:+.2f} > {first:+.2f})")
    check(weak < second,
          f"aminotriazine donor far weaker push than aniline ({weak:+.2f} < {second:+.2f})")
    check(abs(indoline - second) < 2.0,
          f"the two MEASURED 2nd-gen switchers sit close together "
          f"(indoline {indoline:+.2f}, aniline {second:+.2f})")
    # resolution: the previous 3-bin version returned an identical value for every
    # aryl donor, which flattened the objective and handed the search to solubility
    aryls = [dpka(Chem.MolFromSmiles(s)) for s in (
        "COc1ccc(N(C)C=CC=C(O)C=C2C(=O)N(C)C(=O)N(C)C2=O)cc1",
        "CN1C(=O)C(=CC(O)=CC=CNc2ccc(C(F)(F)F)cc2)C(=O)N(C)C1=O",
        "CN1C(=O)C(=CC(O)=CC=CNc2ccc([N+](=O)[O-])cc2)C(=O)N(C)C1=O",
        "O=C1N(C)C(=O)C(=CC(O)=CC=CN2CCc3ccccc32)C(=O)N1C")]
    check(max(aryls) - min(aryls) > 1.5,
          f"CONTINUOUS resolution across aryl donors (spread {max(aryls)-min(aryls):.2f} pKa units)")
    # the CF3 EWG must be attributed to the ring it is actually on
    check(abs(acceptor_pka(Chem.MolFromSmiles(
        "CN1C(=O)C(=CC(O)=CC=CNc2ccc(C(F)(F)F)cc2)C(=O)N(C)C1=O")) - 4.01) < 0.01,
        "CF3 on the DONOR ring does not alter the acceptor pKa")


def test_integrity_gate():
    """Conjugation must be continuous. This is the check whose absence let a
    75-degree-twisted candidate reach TD-DFT and be misread as a UV molecule."""
    print("\n[8] chromophore-integrity gate")
    from reinvent_plugins.components.dasa_common import chromophore_integrity as ci
    for label, smi, expected in [
        ("clean barbituric DASA",
         "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", True),
        ("alpha-methyl backbone (allowed)",
         "CN(C)C(C)=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", True),
        ("adjacent substituted chain carbons -> forced twist",
         "CN(C)C(C)=C(C)C=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", False),
        ("bulky aryl mid-chain with neighbour",
         "CN(C)C(c1ccccc1)=C(C)C=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", False),
    ]:
        check(ci(Chem.MolFromSmiles(smi)) is expected, f"{label} -> {expected}")
    for label, smi, *_ in MEASURED:
        check(ci(Chem.MolFromSmiles(smi)), f"{label}: measured DASA passes integrity")


def test_acceptor_evidence_tiers():
    """'no citation' and 'known not to work' must never be conflated."""
    print("\n[9] acceptor evidence tiers")
    from reinvent_plugins.components.dasa_common import acceptor_evidence as ev
    for label, smi, tier in [
        ("barbituric", "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O", "confirmed"),
        ("indandione (non-photochromic)",
         "CN(C)C=CC=C(O)C=C1C(=O)c2ccccc2C1=O", "contraindicated"),
        ("pyrazolidinedione", "CN(C)C=CC=C(O)C=C1C(=O)NNC1=O", "uncharacterised"),
    ]:
        check(ev(Chem.MolFromSmiles(smi)) == tier, f"{label} -> {tier}")


def test_planar_conformer_search():
    """A single seed-42 embed produced a 75-degree twist. Conformer search must
    find a planar geometry for molecules that have one."""
    print("\n[10] planar conformer search (verification-stage geometry)")
    for label, smi, *_ in MEASURED[:4]:
        _m3, tw = dc.planar_conformer(smi, n_confs=20)
        check(tw is not None and tw <= 15.0,
              f"{label}: planar conformer found (twist {tw:.1f} deg)"
              if tw is not None else f"{label}: no conformer")


def test_closed_stereochemistry_is_enumerated():
    """The closed cyclopentenone has sp3 ring centres that `_cyclize` does not set.
    Measured 2026-09-06: the configuration nobody pins is worth 5-25 kcal/mol,
    against a 4.4 kcal/mol total range for DASA dark equilibria. Any closed-form
    energy taken on ONE arbitrary embedding is a draw, not a measurement, so
    `closed_stereoisomers` must enumerate them and must do so deterministically."""
    print("\n[11] closed-form stereochemistry is enumerated deterministically")
    A = "=C1C(=O)N(C)C(=O)N(C)C1=O"
    anilino = f"CN(c1ccccc1)C=CC=C(O)C{A}"

    iso = dc.closed_stereoisomers(anilino)
    check(len(iso) > 1, f"more than one closed diastereomer found ({len(iso)})")
    forms = {d["form"] for d in iso}
    check(forms == {"zwitterion", "keto"}, f"both tautomers enumerated ({forms})")
    check(all(dc.is_cyclopentenone_closed(Chem.MolFromSmiles(d["smiles"])) for d in iso),
          "every enumerated isomer is still a 4,5-disubstituted cyclopentenone")

    # Deterministic: the answer must not depend on how the input was written.
    # NOTE 2026-09-06: closed_stereoisomers is lru_cached, so calling it twice in
    # one process tests the MEMO, not the enumeration. Clear the cache between
    # calls or this assertion is vacuous -- it was, until a P5 unit came up short
    # and sent me looking. (Enumeration did turn out to be deterministic across
    # processes; the test was still not testing it.)
    dc._closed_stereoisomers_cached.cache_clear()
    again = dc.closed_stereoisomers(Chem.MolToSmiles(Chem.MolFromSmiles(anilino)))
    check([d["smiles"] for d in iso] == [d["smiles"] for d in again],
          "enumeration is independent of the input SMILES writing (memo cleared)")
    dc._closed_stereoisomers_cached.cache_clear()
    third = dc.closed_stereoisomers(anilino)
    check([d["smiles"] for d in iso] == [d["smiles"] for d in third],
          "enumeration is reproducible with a cold memo (tryEmbedding is not random)")

    # a symmetric dialkyl donor has no ammonium stereocentre; an aniline does
    sym = dc.closed_stereoisomers(r"CN(C)/C=C/C=C(\O)C=C1C(=O)N(C)C(=O)N(C)C1=O",
                                  form="zwitterion")
    asym = [d for d in iso if d["form"] == "zwitterion"]
    check(sym and sym[0]["n_centres"] == 2,
          f"Me2N zwitterion has 2 ring stereocentres (got {sym[0]['n_centres'] if sym else None})")
    check(asym and asym[0]["n_centres"] == 3,
          f"aniline zwitterion adds the ammonium N (got {asym[0]['n_centres'] if asym else None})")

    # the manifold helper must report the spread, which is the diagnostic that
    # tells you whether a closed-form energy means anything
    fake = {d["smiles"]: -100.0 + 0.001 * i for i, d in enumerate(iso)}
    man = dc.closed_manifold(anilino, lambda s: fake.get(s))
    check(man is not None and man["n_isomers"] == len(iso),
          "closed_manifold scores every enumerated isomer")
    check(man is not None and man["spread_kcal"] > 0,
          "closed_manifold reports a non-zero spread when isomers differ")
    check(man is not None and man["ensemble_kcal"] <= man["lowest_kcal"] + 1e-9,
          "ensemble energy is at or below the single lowest isomer")


def test_enol_closed_form():
    """C_enol must be a true isomer of the open form for EVERY substitution pattern,
    not just the unsubstituted parent. The first implementation hard-coded hydrogen
    counts and silently returned None for any C5-substituted DASA -- i.e. exactly the
    compounds the Chem. Commun. 2022 substituent study is about."""
    print("\n[12] C_enol closed form (derived connectivity - see _cyclize_enol)")
    A = "=C1C(=O)OC(C)(C)OC1=O"
    D = "C1Cc2ccccc2N1"
    cases = {
        "D1-H unsubstituted": f"{D}C=CC=C(O)C{A}",
        "D1-M 5-methyl":      f"{D}C(C)=CC=C(O)C{A}",
        "5-isopropyl":        f"{D}C(C(C)C)=CC=C(O)C{A}",
        "4-methyl":           f"{D}C=C(C)C=C(O)C{A}",
        "3-methyl":           f"{D}C=CC(C)=C(O)C{A}",
    }
    for label, smi in cases.items():
        o = Chem.MolFromSmiles(smi)
        e = dc.open_to_closed_enol(smi)
        em = Chem.MolFromSmiles(e) if e else None
        check(em is not None, f"{label}: C_enol builds")
        if em is None:
            continue
        check(rdMolDescriptors.CalcMolFormula(em) == rdMolDescriptors.CalcMolFormula(o),
              f"{label}: C_enol is a true constitutional isomer")
        check(dc.is_enol_closed(em), f"{label}: carries a cyclopentene enol C=C-OH")

    # all three closed forms are isomers of one another (one proton, three homes)
    smi = cases["D1-H unsubstituted"]
    forms = [dc.open_to_closed_enol(smi), dc.open_to_closed(smi), dc.open_to_closed_keto(smi)]
    check(all(f is not None for f in forms), "enol / zwitterion / keto all build")
    fs = {rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(f)) for f in forms if f}
    check(len(fs) == 1, f"all three closed forms share one formula ({fs})")
    check(len({Chem.CanonSmiles(f) for f in forms if f}) == 3,
          "and they are three DISTINCT structures")


def test_enantiomers_are_deduplicated():
    """A closed DASA has no external chiral reference, so a stereoisomer and its
    mirror image are the same physical species and are exactly degenerate.
    EnumerateStereoisomers emits both. Left in, they double-count every state in a
    Boltzmann sum by -RT ln 2 and burn half the DFT budget on symmetry-fixed
    duplicates. Measured 2026-09-09: two structures that are provably one physical
    state came out 0.69 kcal/mol apart at M06-2X, which is the conformer-search
    noise floor of the acceptor-bulk experiment.
    """
    print("\n[13] enantiomeric closed isomers are collapsed")
    smi = ("CC1=NN(C(=O)\\C1=C/C(O)=C/C=C/N(CCCC=C)c2ccccc2)c3ccccc3")  # 3-methyl ladder
    raw = dc.closed_stereoisomers(smi, form="keto", dedupe_enantiomers=False)
    ded = dc.closed_stereoisomers(smi, form="keto")
    check(len(raw) == 8, f"C5-substituted keto enumerates 8 raw isomers (got {len(raw)})")
    check(len(ded) == 4, f"collapses to 4 physical states (got {len(ded)})")

    kept = {d["smiles"] for d in ded}
    dropped = {d["smiles"] for d in raw} - kept
    recorded = {d["mirror"] for d in ded if d.get("mirror")}
    check(dropped == recorded,
          "every dropped structure is recorded as a kept one's mirror (lossless)")
    check(not any(dc._mirror_image(s) in kept - {s} for s in kept),
          "no enantiomeric pair survives the collapse")
    check(all(dc._mirror_image(dc._mirror_image(d["smiles"])) == d["smiles"] for d in raw),
          "_mirror_image is an involution")

    # it must NOT collapse genuine diastereomers, and must not touch E/Z
    ez = "C/C=C/C"
    check(dc._mirror_image(Chem.CanonSmiles(ez)) == Chem.CanonSmiles(ez),
          "double-bond (E/Z) stereo is left alone")
    achiral = "CCO"
    check(dc._mirror_image(achiral) == Chem.CanonSmiles(achiral),
          "an achiral molecule is its own mirror")

    # deterministic: same representative every time, cold memo
    dc._closed_stereoisomers_cached.cache_clear()
    again = {d["smiles"] for d in dc.closed_stereoisomers(smi, form="keto")}
    check(again == kept, "the same representative is chosen on a cold memo")


def main() -> int:
    for fn in (test_measured_compounds_are_recognised, test_negatives_rejected,
               test_closed_forms, test_donor_axes, test_colour_gate,
               test_corpus_is_clean, test_trap_escape_reproduces_generation_ordering,
               test_integrity_gate, test_acceptor_evidence_tiers,
               test_planar_conformer_search,
               test_closed_stereochemistry_is_enumerated,
               test_enol_closed_form,
               test_enantiomers_are_deduplicated):
        fn()
    print("\n" + "=" * 70)
    if _failures:
        print(f"{len(_failures)} SENTINEL FAILURE(S) — do not launch a campaign:")
        for f in _failures:
            print("   - " + f)
        return 1
    print("ALL SENTINELS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
