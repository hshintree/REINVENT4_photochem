"""DASA cheminformatics library.

Canonical helpers for the DASA (Donor-Acceptor Stenhouse Adduct) pivot:
scaffold detection, donor/acceptor classification, a combinatorial enumerator
for bootstrapping a transfer-learning corpus, a best-effort open->closed
electrocyclization, and a dataset loader for the literature-extraction schema.

Chemistry background
--------------------
A DASA is a negative photoswitch built from three pieces joined into a linear,
push-pull *open* form:

    [secondary-amine DONOR] - CH=CH-CH=CH - C(OH)= [1,3-dicarbonyl carbon-acid ACCEPTOR]

Under visible light the coloured open (linear triene) form undergoes a 4-pi
electrocyclization to a colourless *closed* cyclopentenone (zwitterionic). In
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
# amino-triene enol whose enol carbon is double-bonded to a carbon bearing a
# carbonyl (the 1,3-dicarbonyl "carbon acid"). Triene carbons are [CX3] (sp2, H
# optional) so methyl/substituted-backbone variants match too. Validated to match
# Meldrum's, barbituric, thiobarbituric, indandione, pyrazolone, isoxazolone,
# oxindole and pyrazolidinedione acceptors and substituted trienes, while
# rejecting azobenzenes, plain/retinal trienes, curcumin, aminocinnamaldehydes,
# short enaminones and typical drug-like decoys.
DASA_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3]-[#6](-[OX2H1,OX1-])=[#6](~[#6]=O)"
# stricter: enol carbon flanked by a genuine 1,3-dicarbonyl (two C=O)
DASA_CLASSIC_SMARTS = (
    "[NX3]-[CH]=[CH]-[CH]=[CH]-[#6]([OX2H1,OX1-])=[#6]([#6]=[OX1])[#6]=[OX1]"
)

_DASA_OPEN = Chem.MolFromSmarts(DASA_OPEN_SMARTS)
_DASA_CLASSIC = Chem.MolFromSmarts(DASA_CLASSIC_SMARTS)

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
    "triene": "/C=C/C=C/C(O)",              # 1st-gen unsubstituted (furfural)
    "alpha_methyl_triene": "/C(C)=C/C=C/C(O)",   # methyl alpha to donor
    "central_methyl_triene": "/C=C/C(C)=C/C(O)",  # methyl on central carbon
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
    "triene": "/C=C/C=C/C(O)",
    "central_methyl_triene": "/C=C/C(C)=C/C(O)",
}


# TETHERED-amine heads: the donor N is locked into a 2,3-dihydropyrrole (5-ring)
# or tetrahydropyridine (6-ring) so the enamine C1=C2 is endocyclic -- the Read
# de Alaniz aqueous-stabilising motif (rigidifies the open form, raises the A->B
# barrier, resists hydrolysis). Each head is a full donor+triene up to C(O); the
# acceptor "=C1..." fragment is appended directly. The N-substituent carries the
# PERIPHERAL solubiliser, so rigidity and hydrophilicity are combined.
TETHERED_HEAD_FRAGMENTS = {
    "tether5_methyl": "CN1CCC(=C1)/C=C/C(O)",
    "tether6_methyl": "CN1CCCC(=C1)/C=C/C(O)",
    "tether5_PEG": "OCCOCCN1CCC(=C1)/C=C/C(O)",          # glycol tail
    "tether5_carboxyethyl": "OC(=O)CCN1CCC(=C1)/C=C/C(O)",  # ionisable
    "tether5_sulfoethyl": "OS(=O)(=O)CCN1CCC(=C1)/C=C/C(O)",  # sulfonate
    "tether5_hydroxyethyl": "OCCN1CCC(=C1)/C=C/C(O)",
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
def open_to_closed_neutral(smiles: str) -> Optional[str]:
    """Open triene -> NEUTRAL keto closed form (2nd-generation closed state).

    Only reachable when the donor N bears a proton (secondary amine / aniline): the
    N-H shifts to the acceptor carbon giving a neutral imine (N=C1) + sp3 C6-H. A
    tertiary dialkylamine (1st-gen) has no such proton -> returns None -> zwitterion
    is the only closed form -> trapped. Mirrors dasa_common.open_to_closed_neutral;
    encodes the 1st-vs-2nd-generation water behaviour (Chem Soc Rev 10.1039/D3CS00508A)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not is_dasa(mol):
        return None
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match or len(match) < 8:
        return None
    n, c1, c2, c3, c4, c5, c6 = (match[i] for i in (0, 1, 2, 3, 4, 5, 7))
    if mol.GetAtomWithIdx(n).GetTotalNumHs() < 1:
        return None
    f_open = rdMolDescriptors.CalcMolFormula(mol)
    rw = Chem.RWMol(mol)
    B = Chem.BondType

    def setbond(a, b, t):
        bd = rw.GetBondBetweenAtoms(a, b)
        if bd is not None:
            bd.SetBondType(t)

    try:
        if rw.GetBondBetweenAtoms(c1, c5) is None:
            rw.AddBond(c1, c5, B.SINGLE)
        setbond(c1, c2, B.SINGLE); setbond(c2, c3, B.DOUBLE); setbond(c3, c4, B.SINGLE)
        setbond(c4, c5, B.SINGLE); setbond(c5, c6, B.SINGLE); setbond(n, c1, B.DOUBLE)
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        if any(a.GetFormalCharge() != 0 for a in m2.GetAtoms()):
            return None
        if rdMolDescriptors.CalcMolFormula(m2) != f_open:
            return None
        return Chem.MolToSmiles(m2)
    except Exception:
        return None


def open_to_closed(smiles: str) -> Optional[str]:
    """4-pi conrotatory electrocyclization: open triene -> closed cyclopentenone.

    Models the DASA ring-closure as the literature-accepted metastable
    **zwitterion**: a new C1-C5 sigma bond forms the cyclopentene ring, the amine
    becomes an iminium (N+=C1), and the acceptor carbon becomes an enolate/
    carbanion (C6-) delocalised into the 1,3-dicarbonyl. C5 is the sp3 quaternary
    carbon retaining the hydroxyl. The product is a CONSTITUTIONAL ISOMER of the
    open form (electrocyclization conserves all atoms), which is asserted below.

    Validated across Meldrum's, barbituric, indandione, pyrazolone, isoxazolone
    acceptors and substituted backbones. Returns None on failure. NOTE: the
    zwitterion is the canonical, most-polar depiction and is ideal for capturing
    the water-stabilised closed state; the exact neutral<->zwitterion protomer is
    refined at the DFT stage.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not is_dasa(mol):
        return None
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match or len(match) < 8:
        return None
    # DASA_OPEN_SMARTS order: 0:N 1:C1 2:C2 3:C3 4:C4 5:C5(enol) 6:O 7:C6(acceptor)
    n, c1, c2, c3, c4, c5, c6 = (match[i] for i in (0, 1, 2, 3, 4, 5, 7))
    f_open = rdMolDescriptors.CalcMolFormula(mol)
    rw = Chem.RWMol(mol)
    B = Chem.BondType

    def setbond(a, b, t):
        bd = rw.GetBondBetweenAtoms(a, b)
        if bd is not None:
            bd.SetBondType(t)

    try:
        if rw.GetBondBetweenAtoms(c1, c5) is None:
            rw.AddBond(c1, c5, B.SINGLE)            # close the cyclopentene ring
        setbond(c1, c2, B.SINGLE)                   # conrotatory bond-order shift
        setbond(c2, c3, B.DOUBLE)
        setbond(c3, c4, B.SINGLE)
        setbond(c4, c5, B.SINGLE)
        setbond(c5, c6, B.SINGLE)                   # acceptor -> enolate/carbanion
        setbond(n, c1, B.DOUBLE)                     # amine -> iminium
        rw.GetAtomWithIdx(n).SetFormalCharge(+1)
        rw.GetAtomWithIdx(c6).SetFormalCharge(-1)
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        if rdMolDescriptors.CalcMolFormula(m2) != f_open:   # must be an isomer
            return None
        return Chem.MolToSmiles(m2)
    except Exception:
        return None


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
