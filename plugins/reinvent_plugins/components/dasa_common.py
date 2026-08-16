"""Self-contained DASA chemistry helpers for the scoring plugins.

Kept independent of notebooks/dasa_chem.py so the plugins import cleanly inside
the ``reinvent`` subprocess (which only has the plugins dir on PYTHONPATH). The
SMARTS here mirror the canonical library and are covered by the same validation.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# DASA open-form core, per Org. Synth. 2022, 99, 79 (checked procedure) and
# Chem Soc Rev 2023 (D3CS00508A):
#
#     R2N-Ca(H)=Cb(H)-Cc(H)=Cd(OH)-Ce(H)=Cf(acceptor)
#
# The hydroxyl sits on Cd -- the "C2 position", which is the former FURAN OXYGEN's
# carbon -- NOT on the carbon bonded to the acceptor. (Mechanism: the amine opens
# furfurylidene at furan-C5; the furan O stays on former furan-C2, which is adjacent
# to the Knoevenagel methine.) Literature open-form configuration is (2Z,4E).
#
# NOTE: until 2026-07-28 this SMARTS placed the enol carbon DIRECTLY on the acceptor
# (see LEGACY_OPEN_SMARTS). That is a constitutional isomer of a real DASA, not a
# DASA -- it computed 229 nm blue of the measured anchor and invalidated the corpus
# built from it. Triene carbons stay [CX3] so substituted backbones still match.
#   match indices:  0:N  1:Ca  2:Cb  3:Cc  4:Cd(enol C)  5:O  6:Ce  7:Cf(acceptor C)
DASA_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3](-[OX2H1,OX1-])-[CX3]=[#6](~[#6]=O)"
_DASA_OPEN = Chem.MolFromSmarts(DASA_OPEN_SMARTS)

# The pre-2026-07-28 (incorrect) skeleton, retained ONLY to detect and reject
# legacy structures -- corpora, checkpoints and CSVs generated before the fix.
LEGACY_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3]-[#6](-[OX2H1,OX1-])=[#6](~[#6]=O)"
_LEGACY_OPEN = Chem.MolFromSmarts(LEGACY_OPEN_SMARTS)

# Canonical carbon-acid acceptors with KNOWN visible absorption (545-670 nm;
# Helmy JOC 2014 / Hemmer). A DASA built on one of these is a real visible dye;
# "other" acceptors the generator invents (weak diesters etc.) tend to absorb in
# the UV. This is the reliable CHEAP colour proxy -- xTB gap/dipole are not (both
# failed to separate a 311 nm UV molecule from visible DASAs). Mirrors dasa_chem.
# Acceptor evidence tiers. The point of the tiering is that "we have no citation"
# and "it does not work" are DIFFERENT statements, and an acceptor we simply have
# not characterised should not be silently discarded -- it should be queued for
# verification. Only CONFIRMED members gate the RL loop.
#
# CONFIRMED -- used as DASA acceptors in the literature with measured open-form
# lambda_max for a real HYDROXY DASA:
_ACCEPTOR_SMARTS_CONFIRMED = [
    "[#6]1(=[#6])[#6](=O)O[#6](C)(C)O[#6]1=O",       # Meldrum      1st-gen archetype (Org. Synth. 2022, 99, 79)
    "[#6]1(=[#6])[#6](=O)[#7][#6](=O)[#7][#6]1=O",    # barbituric   567 +- 3 over 13 donors (Chem Sci 2018)
    "[#6]1(=[#6])[#6](=O)[#7][#7]=[#6]1",             # pyrazolone   646 w/ indoline (Nat Commun 2024 cmpd 11)
    "[#6]1(=[#6])[#6](=O)O[#7]=[#6]1",                # isoxazolone  named 3rd-gen acceptor (Chem Soc Rev 2023)
]
# CONTRAINDICATED -- the literature says these do NOT work. Chem Soc Rev 2023
# describes "a non-photochromic DASA derivative based on indandione" with a neutral
# ground state, i.e. it is coloured but does not SWITCH. Excluded on switching
# grounds, not colour grounds. (This one was previously on our accepted list on the
# strength of an attribution that was never actually checked.)
_ACCEPTOR_SMARTS_CONTRAINDICATED = [
    "[#6]1(=[#6])[#6](=O)c2ccccc2[#6]1=O",            # indandione -- non-photochromic
]
# UNCHARACTERISED -- no evidence either way. Plausible carbon acids, but we have
# found no DASA built on them. Kept OUT of the RL gate (an uncited acceptor is not a
# colour guarantee, and pyrazolidinedione is precisely the one a prior run collapsed
# onto -- it is the weakest acceptor, so it maximised anti-trap while passing a gate
# that only checked list membership). They are NOT discarded: dasa_chem queues them
# for DFT verification, and any that verifies gets promoted to CONFIRMED.
_ACCEPTOR_SMARTS_UNCHARACTERISED = [
    "[#6]1(=[#6])[#6](=O)[#7][#6](=S)[#7][#6]1=O",    # thiobarbituric
    "[#6]1(=[#6])[#6](=O)[#7][#7][#6]1=O",            # pyrazolidinedione
    "[#6]1(=[#6])[#6](=O)[#7]c2ccccc21",              # oxindole
]
_ACCEPTOR_SMARTS = (_ACCEPTOR_SMARTS_CONFIRMED + _ACCEPTOR_SMARTS_CONTRAINDICATED
                    + _ACCEPTOR_SMARTS_UNCHARACTERISED)
_ACCEPTORS_CONFIRMED = [p for p in map(Chem.MolFromSmarts, _ACCEPTOR_SMARTS_CONFIRMED) if p]
_ACCEPTORS_CANDIDATE = _ACCEPTORS_CONFIRMED + [
    p for p in map(Chem.MolFromSmarts, _ACCEPTOR_SMARTS_UNCHARACTERISED) if p]
_ACCEPTORS_ALL = [p for p in map(Chem.MolFromSmarts, _ACCEPTOR_SMARTS) if p]
_ACCEPTORS = _ACCEPTORS_ALL          # back-compat alias

# Acceptor "pull" strength, which together with donor basicity sets how ZWITTERIONIC
# the open and closed forms are (Peterson/Read de Alaniz, ionic-character study):
# 1st- and 3rd-generation architectures give a higher zwitterionic resonance
# contribution and a ZWITTERIONIC closed form; the 2nd-generation (aryl-amine donor)
# gives a less charge-separated open form and a NEUTRAL closed form. Stronger
# acceptor -> more charge separation -> more water-trapped.
_ACCEPTOR_PULL = {           # 0 = mild, 1 = strong
    "meldrum": 0.45, "barbituric": 0.50, "thiobarbituric": 0.60,
    "pyrazolone": 0.85, "isoxazolone": 0.90,     # 3rd-gen, strongly pulling
    "pyrazolidinedione": 0.35, "oxindole": 0.40, "indandione": 0.70, "other": 0.5,
}

_BOHR = 1.8897259886  # Angstrom -> Bohr


def is_dasa(mol) -> bool:
    return mol is not None and mol.HasSubstructMatch(_DASA_OPEN)


def is_legacy_core(mol) -> bool:
    """True for the pre-2026-07-28 wrong-connectivity skeleton (enol carbon bonded
    straight to the acceptor). Use to reject stale corpora / checkpoints / CSVs."""
    return (mol is not None and mol.HasSubstructMatch(_LEGACY_OPEN)
            and not mol.HasSubstructMatch(_DASA_OPEN))


def _core_idx(mol):
    """Map the DASA core match to named atoms, or None."""
    m = mol.GetSubstructMatch(_DASA_OPEN) if mol is not None else None
    if not m or len(m) < 8:
        return None
    return dict(zip(("N", "Ca", "Cb", "Cc", "Cd", "O", "Ce", "Cf"), m))


def has_canonical_acceptor(mol, require_verified: bool = True) -> bool:
    """True if the acceptor is a known-visible carbon acid (the colour proxy).

    Colour is deliberately a GATE, not a graded reward. lambda_max can be pushed
    red almost without bound (longer conjugation / stronger acceptor), and both of
    those trade directly against anti-trap and against MW/solubility -- so grading
    it would add a runaway axis rather than close one. A gate has no gradient: it
    can only exclude. Rigour therefore has to live in the MEMBERSHIP list, which is
    why unverified acceptors are excluded by default.
    """
    if mol is None:
        return False
    pats = _ACCEPTORS_CONFIRMED if require_verified else _ACCEPTORS_CANDIDATE
    return any(mol.HasSubstructMatch(p) for p in pats)


def acceptor_evidence(mol) -> str:
    """'confirmed' | 'uncharacterised' | 'contraindicated' | 'none'.

    Separates "no citation" from "known not to work" so the two are never conflated.
    Uncharacterised acceptors are excluded from the RL gate but queued for DFT
    verification rather than discarded.
    """
    if mol is None:
        return "none"
    for tier, pats in (("confirmed", _ACCEPTOR_SMARTS_CONFIRMED),
                       ("contraindicated", _ACCEPTOR_SMARTS_CONTRAINDICATED),
                       ("uncharacterised", _ACCEPTOR_SMARTS_UNCHARACTERISED)):
        for s in pats:
            p = Chem.MolFromSmarts(s)
            if p is not None and mol.HasSubstructMatch(p):
                return tier
    return "none"


def has_visible_donor(mol) -> bool:
    """True iff the amino-triene DONOR nitrogen is a genuine amine/enamine donor.

    A DASA is coloured only if the head nitrogen strongly PUSHES electron density
    through the triene into the acceptor. If that N is instead acylated (N-C=O, an
    'enaminone'), or bonded to O/N (1,2-oxazine, hydroxylamine, hydrazine), its lone
    pair is tied up and the push-pull chromophore collapses to the UV -- even with a
    canonical acceptor. Confirmed by TD-DFT: barbituric candidates whose donor N was
    acylated computed ~300-360 nm (UV), while genuine amine-donor DASAs are visible.
    The acceptor-only colour gate missed this, so the RL exploited it (acylating the
    donor wins anti-trap + solubility while killing the colour). Requires the donor N
    to bond only C/H besides the triene, and none of those C to be a carbonyl.
    """
    if mol is None:
        return False
    for match in mol.GetSubstructMatches(_DASA_OPEN):
        dN, triene_c = match[0], match[1]
        ok = True
        for nb in mol.GetAtomWithIdx(dN).GetNeighbors():
            if nb.GetIdx() == triene_c:
                continue
            if nb.GetAtomicNum() not in (1, 6):        # N-O / N-N -> weak donor
                ok = False
                break
            for nb2 in nb.GetNeighbors():              # acyl carbon on the donor?
                b = mol.GetBondBetweenAtoms(nb.GetIdx(), nb2.GetIdx())
                if b.GetBondTypeAsDouble() == 2.0 and nb2.GetAtomicNum() in (7, 8, 16):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


def _cyclize(mol, proton_to: str):
    """Open triene -> closed 4,5-disubstituted CYCLOPENTENONE.

    Thermally allowed conrotatory 4pi-electrocyclization forms the Ca-Ce sigma bond
    (Chem Soc Rev 2023, D3CS00508A). The enol Cd-OH becomes the ring KETONE and its
    proton transfers:
      proton_to='N'  -> zwitterion (b) : AMMONIUM (R3N+-H) + acceptor enolate (Cf-)
      proton_to='Cf' -> neutral keto (b'): neutral amine + acceptor C-H

    Which tautomer dominates is set by AMINE BASICITY, not by whether N already
    carries a proton -- the proton comes from the enol, so both are open to any
    amine. Basic alkyl amines take it (zwitterion -> polar -> water-TRAPPED); weakly
    basic anilines do not (keto -> escapes). Chem Sci 2018 X-ray: 1b/2b/9b (alkyl)
    are zwitterionic enolates with a PROTONATED amine; 14b' (aniline) is the keto form.

    Returns a Mol (verified to be a true constitutional isomer) or None.
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
            rw.AddBond(ix["Ca"], ix["Ce"], B.SINGLE)      # the new sigma bond
        sb("Ca", "Cb", B.SINGLE)     # Ca -> sp3, bears the amine   (ring C4)
        sb("Cb", "Cc", B.DOUBLE)     # retained alkene -> cyclopent-2-enone
        sb("Cc", "Cd", B.SINGLE)
        sb("Cd", "O", B.DOUBLE)      # enol -> ring ketone (O loses its proton)
        sb("Cd", "Ce", B.SINGLE)
        sb("Ce", "Cf", B.SINGLE)     # Ce -> sp3, bears the acceptor (ring C5)

        a_o = rw.GetAtomWithIdx(ix["O"])
        a_o.SetNumExplicitHs(0)
        a_o.SetNoImplicit(False)
        for k in ("Ca", "Cb", "Cc", "Cd", "Ce"):
            a = rw.GetAtomWithIdx(ix[k])
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)

        if proton_to == "N":
            n = rw.GetAtomWithIdx(ix["N"])
            n.SetFormalCharge(+1)
            n.SetNumExplicitHs(n.GetTotalNumHs() + 1)
            n.SetNoImplicit(True)
            cf = rw.GetAtomWithIdx(ix["Cf"])
            cf.SetFormalCharge(-1)
            cf.SetNumExplicitHs(0)
            cf.SetNoImplicit(True)
        else:
            cf = rw.GetAtomWithIdx(ix["Cf"])
            cf.SetFormalCharge(0)
            cf.SetNumExplicitHs(cf.GetTotalNumHs() + 1)
            cf.SetNoImplicit(True)

        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        if rdMolDescriptors.CalcMolFormula(m2) != f_open:   # must be a true isomer
            return None
        return m2
    except Exception:
        return None


def open_to_closed(mol):
    """Closed form (b): ammonium + acceptor enolate. Favoured by BASIC (alkyl) amines
    -> the polar, water-trapped state."""
    return _cyclize(mol, "N")


def open_to_closed_keto(mol):
    """Closed form (b'): neutral ketone, proton on the acceptor carbon. Favoured by
    WEAKLY basic (aniline) donors -> the water-trap escape route."""
    return _cyclize(mol, "Cf")


def is_cyclopentenone_closed(mol) -> bool:
    """Structural check that a closed form is a 4,5-disubstituted cyclopentenone."""
    if mol is None:
        return False
    return mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]1(=O)[#6]=[#6][CX4][CX4]1"))



# --- push-pull strength via proton-transfer thermodynamics --------------------
# The open<->closed proton transfer is what decides the trap, so the physically
# right variable is the pKa DIFFERENCE between the two ends:
#
#     dpKa = pKa(R3NH+)  -  pKa(carbon acid)
#
# Large positive -> the amine wins the proton -> ZWITTERION -> water-locked.
# Near zero / negative -> neutral keto closed form -> reversible.
#
# This is a genuine PUSH-PULL descriptor: it moves continuously with every
# substituent on BOTH ends, so unlike a class lookup it has per-molecule
# resolution. It is BANDED, never maximised, because both extremes break the
# molecule: too much push-pull -> zwitterionic trap; too little -> the chromophore
# stops absorbing in the visible.
#
# PROVENANCE: alkyl/aniline pKaH and the Hammett slope are standard; carbon-acid
# pKa are literature. The AZA-HETEROARYL corrections are the least certain values
# here -- and they are exactly the population a run drifted into -- so they are
# marked and should be revised once TD-DFT says where the colour cliff is.
_PKAH_BASE = {          # conjugate-acid pKa of the donor amine
    "dialkyl": 10.7,    # Me2NH 10.7, Et2NH 10.9
    "pyrrolidine": 11.3, "piperidine": 11.1, "morpholine": 8.4, "piperazine": 9.8,
    "benzylic_cyclic": 9.5,   # isoindoline / THIQ: N on two sp3 carbons
    "aniline": 4.6,           # anilinium 4.60
    "indoline": 4.9,          # fused aryl amine
}
_HAMMETT_SLOPE = 2.89   # pKa(anilinium) = 4.60 - 2.89 * sum(sigma)
_SIGMA = [  # (SMARTS on the donor ring, sigma_para/meta)
    ("c-[NX3+](=O)[O-]", 0.78), ("c-[CX2]#N", 0.66), ("c-S(=O)(=O)N", 0.57),
    ("c-C(F)(F)F", 0.54), ("c-[CX3](=O)[OX2]", 0.45), ("c-[CX3](=O)[CX4]", 0.50),
    ("c-[Cl,Br]", 0.23), ("c-F", 0.06), ("c-[OX2][CX4]", -0.27), ("c-[CX4;H3]", -0.17),
]
_AZA_PER_N = 1.7        # ESTIMATE: each additional ring N on the donor ring
_ACID_PKA = {           # carbon-acid pKa (water)
    "meldrum": 4.97, "barbituric": 4.01, "thiobarbituric": 2.3,
    "pyrazolone": 5.5, "isoxazolone": 4.5,          # estimates
    "indandione": 4.2, "pyrazolidinedione": 4.5, "oxindole": 8.0, "other": 4.7,
}


def donor_pkah(mol) -> float:
    """Estimated conjugate-acid pKa of the DASA donor nitrogen."""
    ix = _core_idx(mol)
    if ix is None:
        return 7.0
    dN = mol.GetAtomWithIdx(ix["N"])
    nbrs = [nb for nb in dN.GetNeighbors() if nb.GetIdx() != ix["Ca"]]
    aromatic = [nb for nb in nbrs if nb.GetIsAromatic()]
    if not aromatic:
        if dN.IsInRing():
            has_o = any(a.GetSymbol() == "O" for a in mol.GetAtoms()
                        if a.IsInRing() and a.GetIdx() != ix["N"]
                        and mol.GetRingInfo().AreAtomsInSameRing(a.GetIdx(), ix["N"]))
            return _PKAH_BASE["morpholine"] if has_o else _PKAH_BASE["pyrrolidine"]
        return _PKAH_BASE["dialkyl"]
    # aryl donor: start from anilinium, apply Hammett, then aza corrections
    pka = _PKAH_BASE["indoline"] if dN.IsInRing() else _PKAH_BASE["aniline"]
    sigma = 0.0
    for smarts, s in _SIGMA:
        p = Chem.MolFromSmarts(smarts)
        if p is not None and mol.HasSubstructMatch(p):
            sigma += s
    pka -= _HAMMETT_SLOPE * sigma
    ring = next((r for r in mol.GetRingInfo().AtomRings()
                 if aromatic[0].GetIdx() in r), ())
    n_aza = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "N"
                and mol.GetAtomWithIdx(i).GetIsAromatic())
    pka -= _AZA_PER_N * n_aza
    return float(pka)


def acceptor_pka(mol) -> float:
    """Estimated pKa of the carbon acid (the 'pull' end)."""
    pairs = [("meldrum", _ACCEPTOR_SMARTS_CONFIRMED[0]),
             ("barbituric", _ACCEPTOR_SMARTS_CONFIRMED[1]),
             ("pyrazolone", _ACCEPTOR_SMARTS_CONFIRMED[2]),
             ("isoxazolone", _ACCEPTOR_SMARTS_CONFIRMED[3]),
             ("thiobarbituric", _ACCEPTOR_SMARTS_UNCHARACTERISED[0]),
             ("pyrazolidinedione", _ACCEPTOR_SMARTS_UNCHARACTERISED[1]),
             ("oxindole", _ACCEPTOR_SMARTS_UNCHARACTERISED[2]),
             ("indandione", _ACCEPTOR_SMARTS_CONTRAINDICATED[0])]
    pka, acc_atoms = _ACID_PKA["other"], ()
    for name, s in pairs:
        p = Chem.MolFromSmarts(s)
        if p is not None and mol is not None and mol.HasSubstructMatch(p):
            pka = _ACID_PKA[name]
            acc_atoms = mol.GetSubstructMatch(p)
            break
    # EWG on the ACCEPTOR ring only. Scoping this matters: an unscoped CF3 search
    # also matches a CF3 on the DONOR ring, which would wrongly make an
    # EWG-aniline look like a far stronger acceptor and corrupt dpKa.
    if acc_atoms and mol is not None:
        acc = set(acc_atoms)
        for idx in acc_atoms:
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetIdx() in acc:
                    continue
                fl = sum(1 for x in nb.GetNeighbors() if x.GetSymbol() == "F")
                if nb.GetSymbol() == "C" and fl >= 3:
                    pka -= 2.0
                    break
            else:
                continue
            break
    return float(pka)


def delta_pka(mol) -> float:
    """dpKa = pKa(amine conjugate acid) - pKa(carbon acid).

    IMPLIED SOLVENT: WATER. Every pKa in the tables above is an AQUEOUS value
    (anilinium 4.60, Me2NH2+ 10.7, barbituric acid 4.01, Meldrum's 4.97), so this
    coordinate asks "in water, does the amine take the enol proton?" -- which is
    exactly the target, since the design goal is switching in WATER.

    HONEST CAVEAT about what the band is anchored on. The two compounds pinning the
    switchable end (indoline +0.89, 4-MeO-aniline +1.37) were measured switching in
    CHCl3 / CH2Cl2, NOT in water -- pure-water DASA switching is still unprecedented.
    The upper (trapped) end IS a water statement: 1st-generation dialkylamine DASAs
    are the classic water/methanol "dark switching" failures.

    So the coordinate is anchored by STRUCTURE rather than by water kinetics, and
    that anchor is solid: Chem Sci 2018 X-ray shows 1b/2b/9b (alkyl donors) are
    ZWITTERIONIC closed forms while 14b' (aniline) is the NEUTRAL keto form. That
    tautomer split is the physics, it is what dpKa estimates, and it does not depend
    on which solvent the switching was demonstrated in. What dpKa does NOT do is
    promise a given molecule will switch in pure water -- nothing has yet.

    Reference points:
        Me2N + barbituric   (1st-gen, water-trapped)   ~ +6.7
        aniline + barbituric (2nd-gen, switches)       ~  0.0
        indoline + barbituric (2nd-gen, 615 nm)        ~ +0.2
        Me2N + CF3-pyrazolone (3rd-gen, most trapped)  ~ +7.2
    """
    if mol is None:
        return 0.0
    return donor_pkah(mol) - acceptor_pka(mol)


def _donor_basicity(mol) -> float:
    """Rough conjugate-acid basicity of the donor N, normalised to 0..1 (1 = most
    basic). Basicity decides whether the amine takes the enol proton on ring
    closure: basic alkyl amines do (-> zwitterionic closed form -> polar -> WATER
    TRAPPED), weakly basic aryl amines do not (-> neutral keto -> escapes)."""
    ix = _core_idx(mol)
    if ix is None:
        return 0.5
    dN = mol.GetAtomWithIdx(ix["N"])
    nbrs = [nb for nb in dN.GetNeighbors() if nb.GetIdx() != ix["Ca"]]
    aromatic = [nb for nb in nbrs if nb.GetIsAromatic()]
    if not aromatic:
        return 0.90                       # dialkyl / cyclic alkyl amine (pKaH ~10-11)
    # aniline-type (pKaH ~4-5); electron-poor rings are weaker bases still
    base = 0.30
    ewg = Chem.MolFromSmarts(
        "c-[$([NX3+](=O)[O-]),$([CX2]#N),$(C(F)(F)F),$(S(=O)=O),$([CX3]=O)]")
    if ewg is not None and mol.HasSubstructMatch(ewg):
        base -= 0.10
    if any(nb.GetSymbol() == "N" for r in aromatic
           for nb in r.GetOwningMol().GetAtomWithIdx(r.GetIdx()).GetNeighbors()
           if nb.GetIsAromatic()):
        base -= 0.05                      # electron-poor heteroaryl
    return max(0.0, base)


def zwitterion_character(mol) -> float:
    """Estimated zwitterionic character, 0..1 (higher = more charge-separated =
    more water-TRAPPED).

    Literature basis (Peterson / Read de Alaniz ionic-character study): first- and
    third-generation architectures show a HIGHER zwitterionic resonance
    contribution of the open form AND a zwitterionic closed form, while the
    second-generation (aryl-amine donor) has a LESS charge-separated open form and
    a NEUTRAL closed form. Generation is set by BOTH ends -- donor basicity
    (1st/3rd = basic alkyl, 2nd = weak aryl) and acceptor pull (3rd-gen pyrazolone/
    isoxazolone pull hardest) -- so this is a product of the two ends, not either
    alone. That coupling is also why it cannot be gamed from one end only.
    """
    from math import sqrt
    pairs = [("meldrum", _ACCEPTOR_SMARTS_CONFIRMED[0]),
             ("barbituric", _ACCEPTOR_SMARTS_CONFIRMED[1]),
             ("pyrazolone", _ACCEPTOR_SMARTS_CONFIRMED[2]),
             ("isoxazolone", _ACCEPTOR_SMARTS_CONFIRMED[3]),
             ("thiobarbituric", _ACCEPTOR_SMARTS_UNCHARACTERISED[0]),
             ("pyrazolidinedione", _ACCEPTOR_SMARTS_UNCHARACTERISED[1]),
             ("oxindole", _ACCEPTOR_SMARTS_UNCHARACTERISED[2]),
             ("indandione", _ACCEPTOR_SMARTS_CONTRAINDICATED[0])]
    acc = "other"
    for name, s in pairs:
        p = Chem.MolFromSmarts(s)
        if p is not None and mol is not None and mol.HasSubstructMatch(p):
            acc = name
            break
    return float(sqrt(max(_donor_basicity(mol), 1e-6) * _ACCEPTOR_PULL.get(acc, 0.5)))


def chromophore_integrity(mol, max_chain_substituents: int = 1) -> bool:
    """True if the donor-triene-acceptor pi system is continuous and unhindered.

    A DASA is coloured only if that pi system is CONTINUOUS and can lie flat.
    Nothing in the old pipeline checked this, which is how a candidate with a 75 deg
    twist through its triene reached DFT. Deliberately TOPOLOGICAL so it costs
    nothing in-loop: it catches conjugation breaks intrinsic to the constitution.
    Conformational twist is a different question and is asserted at verification
    time (dasa_chem.chain_planarity), where a conformer search can still fix it.

    Rejects: aromatic chain atoms (cross-conjugated into a ring instead of through
    the triene), non-sp2 chain atoms (conjugation broken outright), and
    over-substituted chains (adjacent substituted carbons force the triene out of
    plane -- the classic steric route to a colourless DASA).
    """
    ix = _core_idx(mol)
    if ix is None:
        return False
    chain = [ix[k] for k in ("Ca", "Cb", "Cc", "Cd", "Ce")]
    subs = []
    for i in chain:
        a = mol.GetAtomWithIdx(i)
        if a.GetIsAromatic():
            return False
        if a.GetHybridization() != Chem.HybridizationType.SP2:
            return False
        extra = [nb.GetIdx() for nb in a.GetNeighbors()
                 if nb.GetAtomicNum() > 1 and nb.GetIdx() not in chain
                 and nb.GetIdx() not in (ix["N"], ix["O"], ix["Cf"])]
        if len(extra) > max_chain_substituents:
            return False
        subs.append(len(extra))
    return not any(x and y for x, y in zip(subs, subs[1:]))


def embed_3d(mol, seed: int = 42, max_iters: int = 200):
    """Add H, embed one conformer and MMFF-optimise -- BOUNDED so no molecule can
    stall an RL batch. A single capped ETKDG attempt (no slow random-coords
    fallback); returns None fast on failure. Some strained zwitterions will fail
    to embed and score 0, which is the right trade for a fast, hang-proof loop."""
    if mol is None:
        return None
    try:
        m = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        p.maxIterations = 200          # hard cap: fail fast instead of hanging
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=max_iters)
        except Exception:
            pass
        return m
    except Exception:
        return None


def xtb_properties(mol3d, solvent: str | None = None):
    """Return (energy_Ha, dipole_norm_au, homo_lumo_gap_eV) from GFN2-xTB.

    ``solvent`` (e.g. "water", "toluene") switches on the ALPB implicit model.
    Returns None if xtb is unavailable or the calculation fails.
    """
    try:
        from xtb.interface import Calculator, Param
    except ImportError:
        return None
    try:
        pos = mol3d.GetConformer().GetPositions() * _BOHR
        nums = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=int)
        calc = Calculator(Param.GFN2xTB, nums, pos)
        calc.set_verbosity(0)
        calc.set_max_iterations(250)   # bound SCF: non-converging zwitterions fail fast
        if solvent is not None:
            try:
                from xtb.utils import get_solvent
                calc.set_solvent(get_solvent(solvent))
            except Exception:
                pass
        res = calc.singlepoint()
        energy = res.get_energy()
        dip = float(np.linalg.norm(res.get_dipole()))
        evals = res.get_orbital_eigenvalues()
        occs = res.get_orbital_occupations()
        occ, unocc = evals[occs > 0.5], evals[occs <= 0.5]
        gap = (unocc[0] - occ[-1]) * 27.2114 if len(occ) and len(unocc) else None
        return energy, dip, gap
    except Exception:
        return None
