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
from rdkit.Chem import AllChem
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


def classify_donor(mol: Chem.Mol) -> str:
    for name, patt in _DONOR_PATTERNS.items():
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
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
def open_to_closed(smiles: str) -> Optional[str]:
    """Approximate 4-pi electrocyclization: form the C-C bond that closes the
    amino-triene into a cyclopentene and tautomerise the enol to a ketone.

    This is deliberately best-effort -- it returns None on any failure rather
    than emitting a chemically wrong structure. DASA scoring does NOT depend on
    it; it exists so the DFT/validation step can render a candidate closed form
    for a chemist to inspect. Do not treat its output as authoritative.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not is_dasa(mol):
        return None
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match or len(match) < 8:
        return None
    # match atom order follows DASA_OPEN_SMARTS:
    #   0:N 1:CH 2:CH 3:CH 4:CH 5:C(enol) 6:O 7:C(acceptor carbon)
    c1, c5, o_idx, c_acc = match[1], match[5], match[6], match[7]
    rw = Chem.RWMol(mol)
    try:
        # new sigma bond between C1 (alpha to N) and C5 (former enol carbon)
        if rw.GetBondBetweenAtoms(c1, c5) is None:
            rw.AddBond(c1, c5, Chem.BondType.SINGLE)
        # the exocyclic C5=C(acceptor) collapses to a single bond (enolate forms)
        b_acc = rw.GetBondBetweenAtoms(c5, c_acc)
        if b_acc is not None:
            b_acc.SetBondType(Chem.BondType.SINGLE)
        # enol -> ketone: C5-OH becomes C5=O
        b_o = rw.GetBondBetweenAtoms(c5, o_idx)
        if b_o is not None:
            b_o.SetBondType(Chem.BondType.DOUBLE)
        o = rw.GetAtomWithIdx(o_idx)
        o.SetNoImplicit(True)
        o.SetNumExplicitHs(0)
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
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
