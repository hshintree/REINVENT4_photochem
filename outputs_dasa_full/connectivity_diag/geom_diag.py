"""Reproduce modal_dft_final.py's geometry pipeline and measure whether the DASA
conjugated chain comes out planar. A twisted chain would blue-shift TD-DFT hard,
so check the input before blaming the method."""
import os, sys, tempfile, subprocess
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms as rmt

SCRATCH = os.path.dirname(os.path.abspath(__file__))
MOLSET = [
    ("anchor", "CN(C)C=CC=CC(O)=C1C(=O)N(C)C(=O)N(C)C1=O"),
    ("tethered0", "O=C(CN1C(=O)NC(=O)C(=C(O)C=CC2=CN(CCc3ccncn3)CC2=O)C1=O)NCc1ccncc1"),
    ("tethered4", "O=C(CN1C(=O)NC(=O)C(=C(O)C=CC2=CN(CCc3cncc4ccccc34)CC2=O)C1=O)NCCC(O)CO"),
    ("aniline5", "Cn1cc(NC=CC=CC(O)=C2C(=O)NN(CC(=O)NCCN3CC3)C2=O)cn1"),
]

# donor N - C1 = C2 - C3 = C4 - C5(OH) = C6(acceptor); loose enough for tethered heads
DASA_OPEN = Chem.MolFromSmarts("[N;!$(N=*)]-[#6]=[#6]-[#6]=[#6]-[#6](-[OX2H1])=[#6]")


def mmff_mol(mol, seed=42):
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
    except Exception:
        pass
    return m


def to_xyz(m):
    conf = m.GetConformer()
    lines = [str(m.GetNumAtoms()), ""]
    for a in m.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines) + "\n"


def xtb_opt(xyz, chg, d, tag):
    open(f"{d}/{tag}.xyz", "w").write(xyz)
    cmd = ["xtb", f"{tag}.xyz", "--opt", "--gfn", "2", "--alpb", "water",
           "--chrg", str(chg), "--uhf", "0", "--cycles", "40"]
    try:
        subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=1800,
                       env={**os.environ, "OMP_NUM_THREADS": "4"})
    except Exception as e:
        print("   xtb failed", e); return None
    p = f"{d}/xtbopt.xyz"
    return open(p).read() if os.path.exists(p) else None


def load_conf(m_with_h, xyz):
    m = Chem.Mol(m_with_h)
    conf = m.GetConformer()
    for i, ln in enumerate(xyz.splitlines()[2:]):
        s = ln.split()
        conf.SetAtomPosition(i, (float(s[1]), float(s[2]), float(s[3])))
    return m


def report(m, match):
    conf = m.GetConformer()
    N, c1, c2, c3, c4, c5, O, c6 = match
    chain = [N, c1, c2, c3, c4, c5, c6]
    names = ["N-C1=C2-C3", "C1=C2-C3=C4", "C2-C3=C4-C5", "C3=C4-C5=C6"]
    quads = [(N, c1, c2, c3), (c1, c2, c3, c4), (c2, c3, c4, c5), (c3, c4, c5, c6)]
    out, worst = [], 0.0
    for nm, q in zip(names, quads):
        t = rmt.GetDihedralDeg(conf, *q)
        dev = min(abs(t), abs(180 - abs(t)))
        worst = max(worst, dev)
        out.append(f"{nm}={t:7.1f}(off{dev:4.1f})")
    bl = [rmt.GetBondLength(conf, chain[i], chain[i + 1]) for i in range(6)]
    print("   torsions: " + " ".join(out))
    print("   N-C1=C2-C3=C4-C5=C6 lengths: " + " ".join(f"{b:.3f}" for b in bl)
          + f"   MAX OFF-PLANE {worst:.1f} deg")
    return worst


for tag, smi in MOLSET:
    print(f"\n=== {tag}", flush=True)
    mol = Chem.MolFromSmiles(smi)
    match = mol.GetSubstructMatch(DASA_OPEN)
    if not match:
        print("   NO DASA-open match!"); continue
    stereo = [str(b.GetStereo()) for b in mol.GetBonds()
              if b.GetBondType() == Chem.BondType.DOUBLE
              and b.GetBeginAtomIdx() in match and b.GetEndAtomIdx() in match]
    print(f"   declared C=C stereo along the triene: {stereo}")
    mh = mmff_mol(mol)
    if mh is None:
        print("   embed failed"); continue
    print("  [MMFF]")
    report(mh, match)
    d = tempfile.mkdtemp()
    xyz = xtb_opt(to_xyz(mh), Chem.GetFormalCharge(mol), d, "open")
    if xyz:
        print("  [xTB/GFN2 ALPB-water, 40 cycles  <-- the geometry DFT actually saw]")
        report(load_conf(mh, xyz), match)
        open(f"{SCRATCH}/{tag}_xtb.xyz", "w").write(xyz)
