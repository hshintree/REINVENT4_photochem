"""Self-contained DASA chemistry helpers for the scoring plugins.

Kept independent of notebooks/dasa_chem.py so the plugins import cleanly inside
the ``reinvent`` subprocess (which only has the plugins dir on PYTHONPATH). The
SMARTS here mirror the canonical library and are covered by the same validation.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# DASA open-form core: amino-triene enol whose enol carbon is double-bonded to a
# carbon bearing a carbonyl (the 1,3-dicarbonyl "carbon acid"). Triene carbons
# are [CX3] so substituted-backbone (methylated triene) variants also match.
# Mirrors notebooks/dasa_chem.py (validated against DASAs and decoys there).
DASA_OPEN_SMARTS = "[NX3]-[CX3]=[CX3]-[CX3]=[CX3]-[#6](-[OX2H1,OX1-])=[#6](~[#6]=O)"
_DASA_OPEN = Chem.MolFromSmarts(DASA_OPEN_SMARTS)

# Canonical carbon-acid acceptors with KNOWN visible absorption (545-670 nm;
# Helmy JOC 2014 / Hemmer). A DASA built on one of these is a real visible dye;
# "other" acceptors the generator invents (weak diesters etc.) tend to absorb in
# the UV. This is the reliable CHEAP colour proxy -- xTB gap/dipole are not (both
# failed to separate a 311 nm UV molecule from visible DASAs). Mirrors dasa_chem.
_ACCEPTOR_SMARTS = [
    "[#6]1(=[#6])[#6](=O)O[#6](C)(C)O[#6]1=O",          # Meldrum
    "[#6]1(=[#6])[#6](=O)[#7][#6](=S)[#7][#6]1=O",       # thiobarbituric
    "[#6]1(=[#6])[#6](=O)[#7][#6](=O)[#7][#6]1=O",       # barbituric
    "[#6]1(=[#6])[#6](=O)c2ccccc2[#6]1=O",               # indandione
    "[#6]1(=[#6])[#6](=O)[#7][#7]=[#6]1",                # pyrazolone
    "[#6]1(=[#6])[#6](=O)O[#7]=[#6]1",                   # isoxazolone
    "[#6]1(=[#6])[#6](=O)[#7][#7][#6]1=O",               # pyrazolidinedione
    "[#6]1(=[#6])[#6](=O)[#7]c2ccccc21",                 # oxindole
]
_ACCEPTORS = [Chem.MolFromSmarts(s) for s in _ACCEPTOR_SMARTS]
_ACCEPTORS = [p for p in _ACCEPTORS if p is not None]

_BOHR = 1.8897259886  # Angstrom -> Bohr


def is_dasa(mol) -> bool:
    return mol is not None and mol.HasSubstructMatch(_DASA_OPEN)


def has_canonical_acceptor(mol) -> bool:
    """True if the acceptor is a known-visible carbon acid (colour proxy)."""
    return mol is not None and any(mol.HasSubstructMatch(p) for p in _ACCEPTORS)


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


def open_to_closed_neutral(mol):
    """Open triene -> NEUTRAL keto closed form (the 2nd-generation closed state).

    Literature (Chem Soc Rev 2023, 10.1039/D3CS00508A; Chem Eur J 2021,
    10.1002/chem.202005110): 1st/3rd-gen DASAs have a ZWITTERIONIC closed form
    (water-trapped), while 2nd-gen (weak aromatic-amine/aniline donors) have a
    NEUTRAL keto closed form that escapes the water trap. The neutral tautomer is
    only reachable when the donor N bears a proton (secondary amine / aniline): that
    N-H shifts to the acceptor carbon C6, giving a neutral imine (N=C1) + sp3 C6-H +
    a neutral 1,3-dicarbonyl. A TERTIARY dialkylamine (1st-gen) has no such proton,
    so this returns None -> the zwitterion is the only closed form -> trapped, exactly
    matching the experimental 1st- vs 2nd-generation water behaviour. Returns a Mol or
    None. The switchability metric uses the LOWER-energy of {zwitterion, neutral}."""
    if mol is None:
        return None
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match or len(match) < 8:
        return None
    n, c1, c2, c3, c4, c5, c6 = (match[i] for i in (0, 1, 2, 3, 4, 5, 7))
    if mol.GetAtomWithIdx(n).GetTotalNumHs() < 1:      # tertiary donor: no mobile proton
        return None
    f_open = rdMolDescriptors.CalcMolFormula(mol)
    rw = Chem.RWMol(mol)
    B = Chem.BondType

    def sb(a, b, t):
        bd = rw.GetBondBetweenAtoms(a, b)
        if bd is not None:
            bd.SetBondType(t)

    try:
        if rw.GetBondBetweenAtoms(c1, c5) is None:
            rw.AddBond(c1, c5, B.SINGLE)
        sb(c1, c2, B.SINGLE); sb(c2, c3, B.DOUBLE); sb(c3, c4, B.SINGLE)
        sb(c4, c5, B.SINGLE); sb(c5, c6, B.SINGLE); sb(n, c1, B.DOUBLE)  # NO formal charges
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        if any(a.GetFormalCharge() != 0 for a in m2.GetAtoms()):    # must be neutral
            return None
        if rdMolDescriptors.CalcMolFormula(m2) != f_open:           # must be an isomer
            return None
        return m2
    except Exception:
        return None


def open_to_closed(mol):
    """Open triene -> closed cyclopentenone ZWITTERION (iminium + acceptor
    enolate), a validated constitutional isomer. Mirrors notebooks/dasa_chem.py.
    Returns an RDKit Mol or None. See that file for the mechanistic rationale."""
    if mol is None:
        return None
    match = mol.GetSubstructMatch(_DASA_OPEN)
    if not match or len(match) < 8:
        return None
    n, c1, c2, c3, c4, c5, c6 = (match[i] for i in (0, 1, 2, 3, 4, 5, 7))
    f_open = rdMolDescriptors.CalcMolFormula(mol)
    rw = Chem.RWMol(mol)
    B = Chem.BondType

    def sb(a, b, t):
        bd = rw.GetBondBetweenAtoms(a, b)
        if bd is not None:
            bd.SetBondType(t)

    try:
        if rw.GetBondBetweenAtoms(c1, c5) is None:
            rw.AddBond(c1, c5, B.SINGLE)
        sb(c1, c2, B.SINGLE); sb(c2, c3, B.DOUBLE); sb(c3, c4, B.SINGLE)
        sb(c4, c5, B.SINGLE); sb(c5, c6, B.SINGLE); sb(n, c1, B.DOUBLE)
        rw.GetAtomWithIdx(n).SetFormalCharge(+1)
        rw.GetAtomWithIdx(c6).SetFormalCharge(-1)
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        if rdMolDescriptors.CalcMolFormula(m2) != f_open:
            return None
        return m2
    except Exception:
        return None


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
