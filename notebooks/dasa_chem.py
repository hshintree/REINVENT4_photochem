"""DASA cheminformatics library.

Canonical helpers for the DASA (Donor-Acceptor Stenhouse Adduct) pivot:
scaffold detection, donor/acceptor classification, a combinatorial enumerator
for bootstrapping a transfer-learning corpus, a best-effort open->closed
electrocyclization, and a dataset loader for the literature-extraction schema.

Chemistry background
--------------------
A DASA is a negative photoswitch built from three pieces joined into a linear,
push-pull *open* form:

    [secondary-amine DONOR] - CH=CH-CH=C(OH) - CH= [1,3-dicarbonyl carbon-acid ACCEPTOR]

Note the hydroxyl is on C2 of the pentadienylidene chain (the former furan
oxygen), with a methine bridging to the acceptor -- NOT on the carbon bonded to
the acceptor. See the SMARTS section below; getting this wrong produces a
constitutional isomer that is not a DASA.

Under visible light the coloured open (linear triene) form undergoes a 4-pi
conrotatory electrocyclization to a colourless *closed* 4,5-disubstituted
cyclopentenone, with concomitant proton transfer from the enol: to N gives the
zwitterion (ammonium + acceptor enolate), to the acceptor carbon gives the
neutral keto form. Amine basicity decides which dominates. In
water the charge-separated open form is usually destabilised and collapses
irreversibly to the closed zwitterion -- making water-switchable DASAs an open
design problem. See notebooks/dasa_complete.py for the full pipeline and the
project memory for the literature grounding (Read de Alaniz "Tethered together"
Chem. Sci. 2023, etc.).

This module is import-safe (only rdkit/numpy) so it can be reused by the
notebook and by tests. The scoring plugins keep their own self-contained copy
of the SMARTS in plugins/reinvent_plugins/components/dasa_common.py so they load
inside the reinvent subprocess without this directory on PYTHONPATH.
"""
from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# ---------------------------------------------------------------------------
# SMARTS -- the DASA open-form core
# ---------------------------------------------------------------------------
# The DASA open form, per Org. Synth. 2022, 99, 79 (a CHECKED procedure, title
# compound "5-((2Z,4E)-5-(diethylamino)-2-hydroxypenta-2,4-dien-1-ylidene)-2,2-
# dimethyl-1,3-dioxane-4,6-dione") and Chem Soc Rev 2023 (D3CS00508A):
#
#     R2N-Ca(H)=Cb(H)-Cc(H)=Cd(OH)-Ce(H)=Cf(acceptor)
#
# The hydroxyl sits on Cd -- the "C2 position" -- which is the former FURAN OXYGEN's
# carbon, NOT the carbon bonded to the acceptor. Mechanism: the amine opens
# furfurylidene at furan-C5 and the furan O stays on former furan-C2, adjacent to
# the Knoevenagel methine. Literature open-form configuration is (2Z,4E).
#
# WARNING (2026-07-28): this SMARTS previously placed the enol carbon DIRECTLY on
# the acceptor (LEGACY_OPEN_SMARTS below). That is a constitutional isomer of a
# DASA, not a DASA. It computed 229 nm blue of the measured anchor (ChemSci 2018
# cmpd 1, 567 nm) and invalidated every corpus, candidate and score derived from it.
# Triene carbons stay [CX3] (sp2, H optional) so substituted backbones still match.
#   match indices:  0:N  1:Ca  2:Cb  3:Cc  4:Cd(enol C)  5:O  6:Ce  7:Cf(acceptor C)
DASA_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3](-[OX2H1,OX1-])-[CX3]=[#6](~[#6]=O)"
# stricter: enol carbon flanked by a genuine 1,3-dicarbonyl (two C=O)
DASA_CLASSIC_SMARTS = (
    "[NX3]-[CH]=[CH]-[CH]=[CH](-[OX2H1,OX1-])-[CH]=[#6]([#6]=[OX1])[#6]=[OX1]"
)
# pre-2026-07-28 (incorrect) skeleton -- retained ONLY to detect stale structures
LEGACY_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3]-[#6](-[OX2H1,OX1-])=[#6](~[#6]=O)"

_DASA_OPEN = Chem.MolFromSmarts(DASA_OPEN_SMARTS)
_DASA_CLASSIC = Chem.MolFromSmarts(DASA_CLASSIC_SMARTS)
_LEGACY_OPEN = Chem.MolFromSmarts(LEGACY_OPEN_SMARTS)


def is_legacy_core(mol) -> bool:
    """True for the pre-2026-07-28 wrong-connectivity skeleton. Use to reject
    stale corpora, checkpoints and candidate CSVs generated before the fix."""
    return (mol is not None and mol.HasSubstructMatch(_LEGACY_OPEN)
            and not mol.HasSubstructMatch(_DASA_OPEN))

# Donor sub-patterns (matched on the amino nitrogen environment)
_DONOR_PATTERNS: Dict[str, Chem.Mol] = {
    "morpholine": Chem.MolFromSmarts("N1CCOCC1"),
    "piperazine": Chem.MolFromSmarts("N1CCNCC1"),
    "hydroxy_amine": Chem.MolFromSmarts("[NX3]CC[OX2H]"),
    "aniline": Chem.MolFromSmarts("[NX3]c"),
    "dialkyl_amine": Chem.MolFromSmarts("[NX3]([CX4])[CX4]"),
}
# Acceptor sub-patterns (the carbon-acid ring, as it appears in the adduct:
# the attachment carbon carries the exocyclic C=C to the triene). Order matters:
# thiobarbituric is checked before barbituric so the C=S variant is not shadowed.
_ACCEPTOR_PATTERNS: Dict[str, Chem.Mol] = {
    "meldrum": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)O[#6](C)(C)O[#6]1=O"),
    "thiobarbituric": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)[#7][#6](=S)[#7][#6]1=O"),
    "barbituric": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)[#7][#6](=O)[#7][#6]1=O"),
    "indandione": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)c2ccccc2[#6]1=O"),
    "pyrazolone": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)[#7][#7]=[#6]1"),
    "isoxazolone": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)O[#7]=[#6]1"),
    "pyrazolidinedione": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)[#7][#7][#6]1=O"),
    "oxindole": Chem.MolFromSmarts("[#6]1(=[#6])[#6](=O)[#7]c2ccccc21"),
}

# Structural liabilities to gate out (reactive / unstable / metal / macrocycle)
FORBIDDEN_SMARTS = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]",
    "[#8][#8]", "[#16][#16]", "C#C", "[#6X5]",
    "[Fe,Co,Ni,Cu,Zn,Ru,Rh,Pd,Ag,Os,Ir,Pt,Au]",
]
_FORBIDDEN = [Chem.MolFromSmarts(s) for s in FORBIDDEN_SMARTS]
_FORBIDDEN = [p for p in _FORBIDDEN if p is not None]

# Task-3: hydrolysis / decomposition liabilities that matter in water. DASAs
# already revert to amine+furan in protic solvent; these motifs make it worse
# (fast aqueous hydrolysis), so we filter generated molecules carrying them.
# NOTE: the amino-triene/enol of the DASA core itself is intrinsically the labile
# part; the mitigation (tethered/cyclic amine) is handled in scoring, not here.
DECOMPOSITION_SMARTS = [
    "[NX3][CX4]([OX2H1])[#6]",          # hemiaminal
    "[NX3][CX4]([NX3])[#6]",            # aminal
    "[OX2][CX4]([OX2])[#6]",            # acetal / ketal
    "[CX3](=O)[OX2][CX3]=O",            # anhydride
    "[CX3](=O)[F,Cl,Br,I]",            # acyl halide
    "[CX3H1,CX3]=[NX2][#6;!$([#6]=[#6])]",  # non-conjugated imine (Schiff base)
    "[CX4]([OX2H])[OX2H]",              # gem-diol
    "[NX3][OX2H1]",                     # N-O (hydroxylamine)
    "[CX3](=O)[CX3](=O)[CX3](=O)",      # 1,2,3-tricarbonyl (very electrophilic)
]
_DECOMP = [Chem.MolFromSmarts(s) for s in DECOMPOSITION_SMARTS]
_DECOMP = [p for p in _DECOMP if p is not None]

# Tethered/cyclic amine donor: the N sits in a ring that also contains a triene
# carbon (or a small ring fused to it) -> rigidified, hydrolytically protected,
# and raises the A->B barrier. This is the Read de Alaniz aqueous-design motif.
_TETHER = Chem.MolFromSmarts("[NX3;R]-[CX3;R]=[CX3]")


# ---------------------------------------------------------------------------
# Detection / classification
# ---------------------------------------------------------------------------
def is_dasa(mol: Chem.Mol, classic_only: bool = False) -> bool:
    """True if the molecule contains a DASA open-form core."""
    if mol is None:
        return False
    patt = _DASA_CLASSIC if classic_only else _DASA_OPEN
    return mol.HasSubstructMatch(patt)


def has_forbidden(mol: Chem.Mol) -> bool:
    if mol is None:
        return True
    return any(mol.HasSubstructMatch(p) for p in _FORBIDDEN)


# Elements the bundled reinvent.prior can tokenise (ChEMBL drug-like vocab).
# I, P, B, Si, Se, etc. are NOT supported -> molecules containing them crash
# TL/inception token validation, so they must never enter the corpus.
PRIOR_SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "S", "F", "Cl", "Br"}


def prior_supported(mol: Chem.Mol) -> bool:
    """True if every atom is in the reinvent.prior vocabulary (no I/P/B/...)."""
    if mol is None:
        return False
    return all(a.GetSymbol() in PRIOR_SUPPORTED_ELEMENTS for a in mol.GetAtoms())


def has_decomposition_liability(mol: Chem.Mol) -> bool:
    """True if the molecule carries a readily water-hydrolysed motif (beyond the
    intrinsic DASA core) -- likely to decompose during switching in water."""
    if mol is None:
        return True
    return any(mol.HasSubstructMatch(p) for p in _DECOMP)


def is_tethered_amine(mol: Chem.Mol) -> bool:
    """True if the donor amine is tethered/cyclised onto the triene (the aqueous-
    stabilising motif that rigidifies the open form and resists hydrolysis)."""
    return mol is not None and mol.HasSubstructMatch(_TETHER)


def has_visible_donor(mol: Chem.Mol) -> bool:
    """True iff the amino-triene DONOR nitrogen is a genuine amine/enamine donor
    (bonds only C/H besides the triene, none acylated). An acylated donor (N-C=O
    enaminone) or an N-O/N-N donor cripples the push-pull chromophore -> UV even
    with a canonical acceptor -- TD-DFT-confirmed (barbituric candidates with
    acylated donors computed ~300-360 nm). Mirrors dasa_common.has_visible_donor;
    the DASAColor gate now requires this AND a canonical acceptor."""
    if mol is None:
        return False
    for match in mol.GetSubstructMatches(_DASA_OPEN):
        dN, triene_c = match[0], match[1]
        ok = True
        for nb in mol.GetAtomWithIdx(dN).GetNeighbors():
            if nb.GetIdx() == triene_c:
                continue
            if nb.GetAtomicNum() not in (1, 6):
                ok = False
                break
            for nb2 in nb.GetNeighbors():
                b = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                if b.GetBondTypeAsDouble() == 2.0 and nb2.GetAtomicNum() in (7, 8, 16):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


def classify_donor(mol: Chem.Mol) -> str:
    for name, patt in _DONOR_PATTERNS.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return "other"


def classify_donor_architecture(mol: Chem.Mol) -> str:
    """The mechanistic donor class that governs water-trapping (for verification
    stratification): 'tethered' (rigidified enamine ring), 'aniline' (2nd-gen weak
    aromatic-amine -> neutral closed form, less trapped), 'dialkyl' (1st-gen -> zwitterion
    -> trapped, but a host-guest candidate), or 'other'. We DFT the top few of EACH so
    the validation set spans switching *mechanisms*, not just one. See dasa_common."""
    if mol is None:
        return "other"
    if is_tethered_amine(mol):
        return "tethered"
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if match:
        dN = mol.GetAtomWithIdx(match[0])
        nbrs = [nb for nb in dN.GetNeighbors() if nb.GetIdx() != match[1]]
        if any(nb.GetIsAromatic() for nb in nbrs):
            return "aniline"
        if nbrs and all(nb.GetSymbol() in ("C", "H") for nb in nbrs):
            return "dialkyl"
    return "other"


def chain_planarity(mol3d: Chem.Mol) -> Optional[float]:
    """Max deviation from planarity (degrees) along the donor-triene-acceptor chain
    of a 3D conformer, or None if there is no core / no conformer.

    VERIFICATION-STAGE check, deliberately NOT in the RL loop. Conformational twist
    is a property of the embedded geometry, not of the constitution, so a bad ETKDG
    conformer must not disqualify a molecule -- it must be fixed by conformer
    search. Use this to (a) pick the flattest conformer before any TD-DFT, and
    (b) assert planarity afterwards.

    This is the check whose absence let a candidate enter TD-DFT with a 75 degree
    twist (and a second at 48 degrees), which blue-shifted it to 317 nm and was
    then misattributed to the method. Measured on the old finalists: anchor 4.4,
    tethered-0 12.8, aniline-5 22.6, tethered-4 75.0 degrees.
    """
    from rdkit.Chem import rdMolTransforms as rmt
    if mol3d is None or mol3d.GetNumConformers() == 0:
        return None
    ix = _core_idx(mol3d)
    if ix is None:
        return None
    conf = mol3d.GetConformer()
    quads = [(ix["N"], ix["Ca"], ix["Cb"], ix["Cc"]),
             (ix["Ca"], ix["Cb"], ix["Cc"], ix["Cd"]),
             (ix["Cb"], ix["Cc"], ix["Cd"], ix["Ce"]),
             (ix["Cc"], ix["Cd"], ix["Ce"], ix["Cf"])]
    worst = 0.0
    for q in quads:
        t = rmt.GetDihedralDeg(conf, *q)
        worst = max(worst, min(abs(t), abs(180.0 - abs(t))))
    return float(worst)


def planar_conformer(smiles: str, n_confs: int = 20, seed: int = 42,
                     max_twist_deg: float = 15.0):
    """Embed n_confs conformers, MMFF-minimise, and return the LOWEST-ENERGY one
    whose chain is planar to within max_twist_deg -- or the flattest available.

    Returns (mol3d, twist_deg) or (None, None). This is the geometry that should
    feed DFT: a single seed-42 embed is what produced the 75-degree candidate.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return None, None
    m = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.pruneRmsThresh = 0.3
    if AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=p) == 0:
        return None, None
    try:
        energies = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    except Exception:
        energies = [(0, 0.0)] * m.GetNumConformers()
    cands = []
    for cid in range(m.GetNumConformers()):
        single = Chem.Mol(m, confId=cid)
        tw = chain_planarity(single)
        if tw is None:
            continue
        e = energies[cid][1] if cid < len(energies) else 0.0
        cands.append((tw, e, cid))
    if not cands:
        return None, None
    planar = [c for c in cands if c[0] <= max_twist_deg]
    tw, _e, cid = min(planar, key=lambda c: c[1]) if planar else min(cands)
    return Chem.Mol(m, confId=cid), tw


def is_rigid_donor(mol: Chem.Mol) -> bool:
    """True if the donor nitrogen sits in a ring (fused or tethered).

    RIGIDITY is a SEPARATE axis from basicity and must not be folded into
    classify_donor_architecture(), which reports basicity class only. The two axes
    are what actually matter and they are independent:

      * basicity (aryl vs alkyl N) -> decides zwitterion vs neutral keto closed
        form, i.e. whether the molecule escapes the water trap;
      * rigidity (N locked in a ring) -> raises the A->B barrier, resists
        hydrolysis, and keeps the chromophore planar.

    Indoline is BOTH (aryl N, fused 5-ring) -- the corner where the benefits stack,
    and a measured 615 nm DASA (Nat Commun 2024 cmpd 10). Reporting it only as
    'aniline' would hide the rigidity from verification stratification. Treating
    'aniline' and 'tethered' as disjoint families was an enumeration artefact.
    """
    if mol is None:
        return False
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match:
        return False
    return mol.GetAtomWithIdx(match[0]).IsInRing()


def donor_axes(mol: Chem.Mol) -> Dict[str, object]:
    """Both donor axes at once: {'basicity': <class>, 'rigid': bool}."""
    return {"basicity": classify_donor_architecture(mol), "rigid": is_rigid_donor(mol)}


def classify_acceptor(mol: Chem.Mol) -> str:
    for name, patt in _ACCEPTOR_PATTERNS.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return "other"


def canonical(smiles: str) -> Optional[str]:
    """Cleanup + largest-fragment canonical SMILES (keeps charges/stereo)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Combinatorial enumeration (bootstrap corpus / TL priming)
# ---------------------------------------------------------------------------
# Building blocks curated from the Read de Alaniz group theses/papers
# (Helmy JOC 2014 & thesis; Hemmer JACS 2018 "carbon acid design" & Chem. Sci.
# 2021 redesign; Robust Red-Absorbing DASAs 2024). Three combinatorial axes:
# DONOR (secondary amine) x BACKBONE (triene bridge) x ACCEPTOR (carbon acid).

# Donor fragments: each is a valid SMILES prefix ending on the amino N that bonds
# to the triene. Grouped by generation / purpose.
DONOR_FRAGMENTS = {
    # 1st-gen acyclic aliphatic
    "dimethylamino": "CN(C)",
    "diethylamino": "CCN(CC)",
    "dipropylamino": "CCCN(CCC)",
    "dibutylamino": "CCCCN(CCCC)",
    "dioctylamino": "CCCCCCCCN(CCCCCCCC)",   # hydrophobic (phase-transfer studies)
    # 1st-gen cyclic aliphatic
    "pyrrolidino": "C1CCCN1",
    "piperidino": "C1CCCCN1",
    "azepano": "C1CCCCCN1",
    "morpholino": "C1COCCN1",
    "thiomorpholino": "C1CSCCN1",
    "N_methylpiperazino": "CN1CCN(CC1)",      # extra basic N (ionisable)
    # functionalised / hydrophilic (water-solubility handles)
    "diethanolamino": "OCCN(CCO)",
    "N_methyl_hydroxyethyl": "CN(CCO)",
    "azidopropyl_methylamino": "CN(CCCN=[N+]=[N-])",
    # 2nd/3rd-gen aryl secondary amines (red-shifting donors)
    "N_methylanilino": "CN(c1ccccc1)",
    "N_ethylanilino": "CCN(c1ccccc1)",
    "indolino": "C1Cc2ccccc2N1",
    "tetrahydroquinolino": "C1CCc2ccccc2N1",
}

# Backbone/bridge variants written to sit between {donor} and {=acceptor}:
# {donor}{backbone}{=acceptor}. The unsubstituted triene is the 1st-gen
# (furfural) bridge; methylated trienes model substituted-backbone derivatives
# (cf. aqueous-compatibility work that tunes charge separation via the bridge).
BACKBONE_FRAGMENTS = {
    "triene": "C=CC=C(O)C",                 # 1st-gen unsubstituted (furfural)
    "alpha_methyl_triene": "C(C)=CC=C(O)C",      # methyl alpha to donor
    "central_methyl_triene": "C=CC(C)=C(O)C",     # methyl on central carbon
}

# Acceptor fragments written as "=<ring>" completing C(O)=<acceptor>, with the
# approximate published open-form lambda_max (nm) they impart (weak donor
# dependence; aryl donors red-shift further). Used to seed est_lambda.
ACCEPTOR_FRAGMENTS = {
    "meldrum": "=C1C(=O)OC(C)(C)OC1=O",
    "dimethylbarbituric": "=C1C(=O)N(C)C(=O)N(C)C1=O",
    "barbituric": "=C1C(=O)NC(=O)NC1=O",
    "diethylbarbituric": "=C1C(=O)N(CC)C(=O)N(CC)C1=O",
    "thiobarbituric": "=C1C(=O)NC(=S)NC1=O",
    "dimethylthiobarbituric": "=C1C(=O)N(C)C(=S)N(C)C1=O",
    "indandione": "=C1C(=O)c2ccccc2C1=O",
    "N_phenyl_methylpyrazolone": "=C1C(=O)N(c2ccccc2)N=C1C",
    "N_phenyl_CF3_pyrazolone": "=C1C(=O)N(c2ccccc2)N=C1C(F)(F)F",
    "methylisoxazolone": "=C1C(=O)ON=C1C",
    "CF3_isoxazolone": "=C1C(=O)ON=C1C(F)(F)F",
    "pyrazolidinedione": "=C1C(=O)NNC1=O",
    "oxindole": "=C1C(=O)Nc2ccccc21",
}

# Approximate published open-form lambda_max per acceptor (nm), aliphatic donor.
ACCEPTOR_LAMBDA_NM = {
    "meldrum": 545, "dimethylbarbituric": 570, "barbituric": 570,
    "diethylbarbituric": 570, "thiobarbituric": 600, "dimethylthiobarbituric": 600,
    "indandione": 600, "N_phenyl_methylpyrazolone": 590,
    "N_phenyl_CF3_pyrazolone": 669, "methylisoxazolone": 626,
    "CF3_isoxazolone": 649, "pyrazolidinedione": 610, "oxindole": 600,
}
# Aryl (2nd/3rd-gen) donors red-shift the open form by roughly this many nm.
_ARYL_DONOR_REDSHIFT_NM = 40
_ARYL_DONOR_CLASSES = {"aniline"}

# ---------------------------------------------------------------------------
# Task-4: AQUEOUS-FOCUSED building blocks (water-switchable design)
# ---------------------------------------------------------------------------
# Strategy from the literature: (a) REDUCE charge separation via EWG-substituted
# aniline donors + WEAK acceptors (methyl-pyrazolone/isoxazolone), so water does
# not lock the closed zwitterion; (b) DECOUPLE solubility by putting hydrophilic
# handles on the PERIPHERY (remote glycol/carboxyl/ammonium off the push-pull
# axis) rather than strengthening the donor; (c) prefer cyclic/tethered amines
# for hydrolytic stability. Backbones: unsubstituted + central-methyl only
# (alpha-methyl breaks the iminium closure).
# DONOR MIX IS DELIBERATELY 2ND-GEN-HEAVY. The water-switchable class is 2nd-gen
# (WEAK AROMATIC / aniline donor -> reduced charge separation -> NEUTRAL keto closed
# form that escapes the water trap; Chem Soc Rev 2023, Chem Eur J 2021). A prior run
# reused a Stage-1 checkpoint that had drifted to 96% tertiary dialkylamine (1st-gen,
# zwitterionic, trapped) -- so we now SEED the corpus with many aniline/heteroaryl
# (secondary N-H, weak aromatic) donors and keep only a few tertiary alkyls for
# diversity / host-guest 1st-gen candidates. The DASA2ndGen component reinforces this.
AQUEOUS_DONOR_FRAGMENTS = {
    # --- 2nd-gen: EWG anilines (secondary N-H, weak aromatic -> less charge sep) ---
    # No 4-IODO: the reinvent.prior vocab lacks an 'I' token; use Br/Cl/CF3/etc.
    "4-bromoanilino": "N(c1ccc(Br)cc1)",
    "4-chloroanilino": "N(c1ccc(Cl)cc1)",
    "4-CF3-anilino": "N(c1ccc(C(F)(F)F)cc1)",
    "4-cyanoanilino": "N(c1ccc(C#N)cc1)",
    "3-CF3-anilino": "N(c1cccc(C(F)(F)F)c1)",
    "3,4-dichloroanilino": "N(c1ccc(Cl)c(Cl)c1)",
    "4-nitroanilino": "N(c1ccc([N+](=O)[O-])cc1)",
    "4-methoxycarbonylanilino": "N(c1ccc(C(=O)OC)cc1)",
    "4-acetylanilino": "N(c1ccc(C(C)=O)cc1)",
    "4-sulfamoylanilino": "N(c1ccc(S(N)(=O)=O)cc1)",     # H-bonding + water solubility
    # --- FUSED ARYL AMINES: weak basicity AND ring-rigidity in one donor ---
    # These are the corner where the "aniline" and "tethered" benefits STACK: the N
    # is aryl-attached (weakly basic -> neutral keto closed form -> escapes the water
    # trap) AND locked in a fused ring (rigidified open form, higher A->B barrier,
    # hydrolytically robust). They are also the best-characterised 2nd-gen donors in
    # the literature -- indoline/1,3-dimethylbarbituric is a measured 615 nm hydroxy
    # DASA (Nat Commun 2024, cmpd 10). Treating aniline and tethered as disjoint
    # families was an artefact of enumerating them from separate fragment sets.
    "indoline": "C1Cc2ccccc2N1",
    "tetrahydroquinoline": "C1CCc2ccccc2N1",
    "5-F-indoline": "C1Cc2cc(F)ccc2N1",
    "5-CF3-indoline": "C1Cc2cc(C(F)(F)F)ccc2N1",
    "5-CN-indoline": "C1Cc2cc(C#N)ccc2N1",
    "indoline-5-sulfonamide": "C1Cc2cc(S(N)(=O)=O)ccc2N1",   # + water solubility
    # cyclic BENZYLIC amines: rigid but still basic alkyl N (isoindoline = 573 nm,
    # Nat Commun 2024 cmpd 9). Kept as the rigidity-without-weak-basicity control.
    "isoindoline": "C1c2ccccc2CN1",
    "tetrahydroisoquinoline": "C1Cc2ccccc2CN1",
    # --- 2nd-gen: electron-poor HETEROARYL amines (even weaker donors) ---
    "3-aminopyridyl": "N(c1cccnc1)",
    "4-aminopyridyl": "N(c1ccncc1)",
    "5-aminopyrimidyl": "N(c1cncnc1)",
    # --- 1st-gen tertiary alkyls: kept for DIVERSITY / host-guest candidates only ---
    "morpholino": "C1COCCN1",
    "N_carboxyethyl_methyl": "CN(CCC(=O)O)",       # ionisable carboxylate tail
    "N_PEG2_methyl": "CN(CCOCCO)",                  # short glycol tail
    "N_sulfoethyl_methyl": "CN(CCS(=O)(=O)O)",      # sulfonate (strongly hydrophilic)
}
# Weak acceptors keep charge separation in the switchable window (~-20 nm slope).
AQUEOUS_ACCEPTOR_FRAGMENTS = {
    "N_phenyl_methylpyrazolone": "=C1C(=O)N(c2ccccc2)N=C1C",
    "N_methyl_methylpyrazolone": "=C1C(=O)N(C)N=C1C",
    "methylisoxazolone": "=C1C(=O)ON=C1C",
    "dimethylbarbituric": "=C1C(=O)N(C)C(=O)N(C)C1=O",   # moderate reference
}
AQUEOUS_BACKBONE_FRAGMENTS = {
    "triene": "C=CC=C(O)C",
    "central_methyl_triene": "C=CC(C)=C(O)C",
}


# TETHERED-amine heads: the donor N is locked into a 2,3-dihydropyrrole (5-ring)
# or tetrahydropyridine (6-ring) so the enamine C1=C2 is endocyclic -- the Read
# de Alaniz aqueous-stabilising motif (rigidifies the open form, raises the A->B
# barrier, resists hydrolysis). Each head is a full donor+triene up to C(O); the
# acceptor "=C1..." fragment is appended directly. The N-substituent carries the
# PERIPHERAL solubiliser, so rigidity and hydrophilicity are combined.
TETHERED_HEAD_FRAGMENTS = {
    "tether5_methyl": "CN1CCC(=C1)C=C(O)C",
    "tether6_methyl": "CN1CCCC(=C1)C=C(O)C",
    "tether5_PEG": "OCCOCCN1CCC(=C1)C=C(O)C",          # glycol tail
    "tether5_carboxyethyl": "OC(=O)CCN1CCC(=C1)C=C(O)C",  # ionisable
    "tether5_sulfoethyl": "OS(=O)(=O)CCN1CCC(=C1)C=C(O)C",  # sulfonate
    "tether5_hydroxyethyl": "OCCN1CCC(=C1)C=C(O)C",
}


def enumerate_tethered_dasa(
    heads: Optional[Dict[str, str]] = None,
    acceptors: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """head x acceptor -> validated TETHERED-amine open-form DASAs."""
    heads = heads if heads is not None else TETHERED_HEAD_FRAGMENTS
    acceptors = acceptors if acceptors is not None else AQUEOUS_ACCEPTOR_FRAGMENTS
    seen, out = set(), []
    for hname, head in heads.items():
        for aname, acc in acceptors.items():
            mol = Chem.MolFromSmiles(head + acc)
            if mol is None or not is_dasa(mol) or has_forbidden(mol):
                continue
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            out.append({
                "smiles_open": can, "donor_class": "tethered",
                "acceptor_class": classify_acceptor(mol),
                "backbone": hname, "est_lambda_nm": ACCEPTOR_LAMBDA_NM.get(aname),
            })
    return out


def enumerate_dasa_aqueous(include_tethered: bool = True, **kwargs) -> List[dict]:
    """Enumerate the AQUEOUS-focused DASA library (task-4): EWG-aniline / weak-
    acceptor / peripheral-solubiliser combos, plus (by default) the tethered-amine
    scaffolds. Annotates ``tethered`` and ``decomp_liable`` on every row."""
    kwargs.setdefault("donors", AQUEOUS_DONOR_FRAGMENTS)
    kwargs.setdefault("acceptors", AQUEOUS_ACCEPTOR_FRAGMENTS)
    kwargs.setdefault("backbones", AQUEOUS_BACKBONE_FRAGMENTS)
    rows = enumerate_dasa(**kwargs)
    if include_tethered:
        rows = rows + enumerate_tethered_dasa(acceptors=kwargs["acceptors"])
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles_open"])
        r["tethered"] = is_tethered_amine(m)
        r["decomp_liable"] = has_decomposition_liability(m)
    return rows


def _open_form_smiles(donor: str, backbone: str, acceptor: str) -> str:
    # DONOR  <backbone triene ...C(O)>  =ACCEPTOR
    return f"{donor}{backbone}{acceptor}"


def _acceptor_name(fragment: str) -> Optional[str]:
    for name, frag in ACCEPTOR_FRAGMENTS.items():
        if frag == fragment:
            return name
    return None


def enumerate_dasa(
    donors: Optional[Dict[str, str]] = None,
    acceptors: Optional[Dict[str, str]] = None,
    backbones: Optional[Dict[str, str]] = None,
    require_classic: bool = False,
) -> List[dict]:
    """Cartesian product donor x backbone x acceptor -> validated open-form DASAs.

    Each argument is a {name: fragment} dict (defaults to the full curated
    libraries). Returns dicts with: smiles_open, donor_class, acceptor_class,
    backbone, est_lambda_nm. Invalid combinations (bad valence, failed SMARTS
    gate, forbidden group) are silently dropped; duplicates are removed.
    """
    donors = donors if donors is not None else DONOR_FRAGMENTS
    acceptors = acceptors if acceptors is not None else ACCEPTOR_FRAGMENTS
    backbones = backbones if backbones is not None else BACKBONE_FRAGMENTS
    seen, out = set(), []
    for d_frag in donors.values():
        for bb in backbones.values():
            for a_name, a_frag in acceptors.items():
                smi = _open_form_smiles(d_frag, bb, a_frag)
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                if not is_dasa(mol, classic_only=require_classic) or has_forbidden(mol):
                    continue
                can = Chem.MolToSmiles(mol)
                if can in seen:
                    continue
                seen.add(can)
                dclass = classify_donor(mol)
                lam = ACCEPTOR_LAMBDA_NM.get(a_name)
                if lam is not None and dclass in _ARYL_DONOR_CLASSES:
                    lam += _ARYL_DONOR_REDSHIFT_NM
                out.append({
                    "smiles_open": can,
                    "donor_class": dclass,
                    "acceptor_class": classify_acceptor(mol),
                    "backbone": [k for k, v in backbones.items() if v == bb][0],
                    "est_lambda_nm": lam,
                })
    return out


# ---------------------------------------------------------------------------
# Best-effort open -> closed electrocyclization (for DFT/inspection ONLY)
# ---------------------------------------------------------------------------
def _core_idx(mol):
    """Map the DASA core match to named atoms, or None.
    0:N 1:Ca 2:Cb 3:Cc 4:Cd(enol C) 5:O 6:Ce 7:Cf(acceptor C)"""
    m = mol.GetSubstructMatch(_DASA_OPEN) if mol is not None else None
    if not m or len(m) < 8:
        return None
    return dict(zip(("N", "Ca", "Cb", "Cc", "Cd", "O", "Ce", "Cf"), m))


def _cyclize(mol, proton_to: str):
    """Open triene -> closed 4,5-disubstituted CYCLOPENTENONE (RDKit Mol or None).

    Thermally allowed conrotatory 4-pi electrocyclization forms the Ca-Ce sigma
    bond (Chem Soc Rev 2023, D3CS00508A). The enol Cd-OH becomes the ring KETONE
    and its proton transfers:
      proton_to='N'  -> zwitterion (b) : AMMONIUM (R3N+-H) + acceptor enolate (Cf-)
      proton_to='Cf' -> neutral keto (b'): neutral amine + acceptor C-H

    Both are reachable for ANY amine -- the proton comes from the enol, not from N.
    Which dominates is set by AMINE BASICITY: basic alkyl amines take it
    (zwitterion -> polar -> water-trapped), weakly basic anilines do not (keto ->
    escapes). Chem Sci 2018 X-ray: 1b/2b/9b (alkyl) are zwitterionic enolates with
    a PROTONATED amine; 14b' (aniline) is the keto form. The product is asserted to
    be a true constitutional isomer of the open form.
    """
    ix = _core_idx(mol)
    if ix is None:
        return None
    f_open = rdMolDescriptors.CalcMolFormula(mol)
    rw = Chem.RWMol(mol)
    B = Chem.BondType

    def sb(a, b, t):
        bd = rw.GetBondBetweenAtoms(ix[a], ix[b])
        if bd is not None:
            bd.SetBondType(t)

    try:
        if rw.GetBondBetweenAtoms(ix["Ca"], ix["Ce"]) is None:
            rw.AddBond(ix["Ca"], ix["Ce"], B.SINGLE)     # the new sigma bond
        sb("Ca", "Cb", B.SINGLE)      # Ca -> sp3, bears the amine    (ring C4)
        sb("Cb", "Cc", B.DOUBLE)      # retained alkene -> cyclopent-2-enone
        sb("Cc", "Cd", B.SINGLE)
        sb("Cd", "O", B.DOUBLE)       # enol -> ring ketone
        sb("Cd", "Ce", B.SINGLE)
        sb("Ce", "Cf", B.SINGLE)      # Ce -> sp3, bears the acceptor (ring C5)

        a_o = rw.GetAtomWithIdx(ix["O"])
        a_o.SetNumExplicitHs(0); a_o.SetNoImplicit(False)
        for k in ("Ca", "Cb", "Cc", "Cd", "Ce"):
            a = rw.GetAtomWithIdx(ix[k]); a.SetNumExplicitHs(0); a.SetNoImplicit(False)

        if proton_to == "N":
            n = rw.GetAtomWithIdx(ix["N"])
            n.SetFormalCharge(+1); n.SetNumExplicitHs(n.GetTotalNumHs() + 1)
            n.SetNoImplicit(True)
            cf = rw.GetAtomWithIdx(ix["Cf"])
            cf.SetFormalCharge(-1); cf.SetNumExplicitHs(0); cf.SetNoImplicit(True)
        else:
            cf = rw.GetAtomWithIdx(ix["Cf"])
            cf.SetFormalCharge(0); cf.SetNumExplicitHs(cf.GetTotalNumHs() + 1)
            cf.SetNoImplicit(True)

        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        if rdMolDescriptors.CalcMolFormula(m2) != f_open:    # must be an isomer
            return None
        return m2
    except Exception:
        return None


def open_to_closed(smiles: str) -> Optional[str]:
    """Closed form (b): ammonium + acceptor enolate -- the polar, water-trapped
    state favoured by BASIC (alkyl) amines. Returns SMILES or None."""
    m = _cyclize(Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles, "N")
    return Chem.MolToSmiles(m) if m is not None else None


def open_to_closed_keto(smiles: str) -> Optional[str]:
    """Closed form (b'): neutral ketone, proton on the acceptor carbon -- the
    water-trap escape route favoured by WEAKLY basic (aniline) donors."""
    m = _cyclize(Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles, "Cf")
    return Chem.MolToSmiles(m) if m is not None else None


def is_cyclopentenone_closed(mol) -> bool:
    """Structural check that a closed form is a 4,5-disubstituted cyclopentenone."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return False
    return mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]1(=O)[#6]=[#6][CX4][CX4]1"))


# ---------------------------------------------------------------------------
# Dataset schema + loader (literature-extraction path)
# ---------------------------------------------------------------------------
DATASET_COLUMNS = [
    "smiles_open",           # required: open-form SMILES
    "smiles_closed",         # optional: measured/known closed form
    "donor_class",           # optional (auto-filled if blank)
    "acceptor_class",        # optional (auto-filled if blank)
    "lambda_max_open_nm",    # measured open-form absorption
    "solvent",               # solvent of the measurement
    "pct_open_equilibrium",  # % open at dark equilibrium (0-100)
    "solvatochromic_slope_nm",  # charge-separation proxy (negative nm)
    "switches_in_water",     # label: True/False/partial
    "source",                # citation / DOI
]


def load_dasa_dataset(csv_path: str):
    """Load and clean a literature DASA table (see DATASET_COLUMNS).

    Only ``smiles_open`` is required. Missing donor/acceptor classes are filled
    from SMARTS. Rows with unparseable SMILES or non-DASA cores are dropped and
    the frame is de-duplicated by canonical open-form SMILES.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "smiles_open" not in df.columns:
        raise ValueError(
            f"{csv_path} must have a 'smiles_open' column. "
            f"Expected schema: {DATASET_COLUMNS}"
        )
    records = []
    for _, row in df.iterrows():
        can = canonical(str(row["smiles_open"]).strip())
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if not is_dasa(mol) or has_forbidden(mol):
            continue
        rec = row.to_dict()
        rec["smiles_open"] = can
        if pd.isna(rec.get("donor_class")) or not str(rec.get("donor_class", "")).strip():
            rec["donor_class"] = classify_donor(mol)
        if pd.isna(rec.get("acceptor_class")) or not str(rec.get("acceptor_class", "")).strip():
            rec["acceptor_class"] = classify_acceptor(mol)
        records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.drop_duplicates(subset="smiles_open").reset_index(drop=True)
    return out


def legacy_to_corrected(smiles):
    """Convert a pre-2026-07-28 (legacy-core) DASA to the correct connectivity.

    The two skeletons differ in exactly ONE thing -- which chain carbon bears the
    hydroxyl. Bond orders are identical in both:

        legacy     N-Ca=Cb-Cc=Cd-Ce(OH)=Cf     enol carbon bonded to the acceptor
        corrected  N-Ca=Cb-Cc=Cd(OH)-Ce=Cf     hydroxyl on C2 (the furan oxygen)

    So the conversion is just: move the O from Ce to Cd. Everything else -- donor,
    acceptor, every peripheral substituent -- is preserved, which makes it safe to
    carry an old candidate forward for comparison instead of discarding it.

    Returns corrected SMILES, or None if the input is not a legacy DASA.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return None
    m = mol.GetSubstructMatch(_LEGACY_OPEN)
    if not m or len(m) < 8:
        return None
    cd, ce, o = m[4], m[5], m[6]
    f_in = rdMolDescriptors.CalcMolFormula(mol)
    rw = Chem.RWMol(mol)
    try:
        rw.RemoveBond(ce, o)
        rw.AddBond(cd, o, Chem.BondType.SINGLE)
        for i in (cd, ce, o):
            a = rw.GetAtomWithIdx(i)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        if rdMolDescriptors.CalcMolFormula(out) != f_in:   # must stay an isomer
            return None
        return Chem.MolToSmiles(out) if is_dasa(out) else None
    except Exception:
        return None


def dasa_retrosynthesis(smiles):
    """Split a DASA into its two commercial precursors.

    DASAs are made by ONE short route (Org. Synth. 2022, 99, 79):

        1. furfural + carbon acid  --Knoevenagel-->  furfurylidene-carbon acid
        2. + secondary amine       --Stenhouse-->    DASA

    So the retrosynthesis is fixed and trivial: every DASA is two steps from
    furfural plus (a) the donor amine and (b) the carbon acid. Reporting those two
    fragments is the single most useful thing to hand a synthetic chemist -- it
    turns "here is a generated structure" into "make it from these two catalogue
    compounds in two steps", which is defensible in a way a computed lambda is not.

    Returns {'amine', 'carbon_acid', 'route'} as SMILES, or None.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    ix = _core_idx(mol) if mol is not None else None
    if ix is None:
        return None

    def _fragment(bond_a, bond_b, keep_atom, add_hs):
        rw = Chem.RWMol(mol)
        if rw.GetBondBetweenAtoms(bond_a, bond_b) is None:
            return None
        rw.RemoveBond(bond_a, bond_b)
        a = rw.GetAtomWithIdx(keep_atom)
        a.SetNumExplicitHs(a.GetTotalNumHs() + add_hs)
        a.SetNoImplicit(True)
        try:
            frags = Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=True)
        except Exception:
            return None
        for f in frags:
            if any(at.HasProp("_keep") for at in f.GetAtoms()):
                return Chem.MolToSmiles(f)
        # fall back: pick the fragment that still contains the kept atom's symbol set
        return None

    # tag the atoms we want to keep so the right fragment can be identified
    m2 = Chem.Mol(mol)
    m2.GetAtomWithIdx(ix["N"]).SetProp("_keep", "amine")
    rw = Chem.RWMol(m2)
    rw.RemoveBond(ix["N"], ix["Ca"])
    n = rw.GetAtomWithIdx(ix["N"])
    n.SetNumExplicitHs(n.GetTotalNumHs() + 1)
    n.SetNoImplicit(True)
    amine = None
    try:
        for f in Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=True):
            if any(a.HasProp("_keep") for a in f.GetAtoms()):
                for a in f.GetAtoms():
                    a.ClearProp("_keep")
                amine = Chem.MolToSmiles(f)
    except Exception:
        amine = None

    m3 = Chem.Mol(mol)
    m3.GetAtomWithIdx(ix["Cf"]).SetProp("_keep", "acid")
    rw = Chem.RWMol(m3)
    rw.RemoveBond(ix["Ce"], ix["Cf"])
    cf = rw.GetAtomWithIdx(ix["Cf"])
    cf.SetNumExplicitHs(cf.GetTotalNumHs() + 2)      # exocyclic C= becomes CH2
    cf.SetNoImplicit(True)
    acid = None
    try:
        for f in Chem.GetMolFrags(rw.GetMol(), asMols=True, sanitizeFrags=True):
            if any(a.HasProp("_keep") for a in f.GetAtoms()):
                for a in f.GetAtoms():
                    a.ClearProp("_keep")
                acid = Chem.MolToSmiles(f)
    except Exception:
        acid = None

    if amine is None or acid is None:
        return None
    # SANITY: the standard route only applies if BOTH bonds are exocyclic, i.e. the
    # cuts actually disconnected something. For a TETHERED donor the N-Ca bond is
    # inside a ring, so removing it leaves the molecule whole and the "amine"
    # fragment comes back containing the entire chromophore. That is not a
    # precursor -- it is a failed retrosynthesis, and a tethered DASA genuinely is
    # NOT two steps from furfural + a simple amine. Surface it instead of emitting
    # a nonsense fragment.
    a_mol, c_mol = Chem.MolFromSmiles(amine), Chem.MolFromSmiles(acid)
    if a_mol is None or c_mol is None:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    if (a_mol.GetNumHeavyAtoms() >= n_heavy - 2
            or c_mol.GetNumHeavyAtoms() >= n_heavy - 2
            or a_mol.HasSubstructMatch(_DASA_OPEN)
            or c_mol.HasSubstructMatch(_DASA_OPEN)):
        return {"amine": None, "carbon_acid": None,
                "route": "NOT accessible by the standard 2-step Stenhouse route "
                         "(the donor N-C bond is endocyclic -- tethered donor); "
                         "needs a bespoke synthesis"}
    return {"amine": amine, "carbon_acid": acid,
            "route": "furfural + carbon acid (Knoevenagel), then amine (Stenhouse)"}
