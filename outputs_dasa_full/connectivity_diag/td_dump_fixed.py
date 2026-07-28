"""Reproduce modal_dft_final.py's anchor lambda (water/b3lyp = 338.0 nm) locally, but
dump the FULL excitation list with orbital character instead of only the max-f state.

Decides between the competing explanations:
  (a) the bright CT band IS there near ~550 nm but max-oscillator-strength picks a bluer state
  (b) TD-DFT genuinely puts everything at ~340 nm  -> geometry or method problem
"""
import sys, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft, tddft
from pyscf.solvent import ddCOSMO

SMI = "CN(C)C=CC=C(O)C=C1C(=O)N(C)C(=O)N(C)C1=O"   # CORRECTED connectivity: 2-hydroxy...1-ylidene
BASIS = sys.argv[1] if len(sys.argv) > 1 else "6-31g*"
XC = sys.argv[2] if len(sys.argv) > 2 else "b3lyp"
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 78.4     # <=1 means gas phase
NST = 12


def mmff_atoms(smi, seed=42):
    mol = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(mol)
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    conf = m.GetConformer()
    return "\n".join(f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.6f} "
                     f"{conf.GetAtomPosition(a.GetIdx()).y:.6f} "
                     f"{conf.GetAtomPosition(a.GetIdx()).z:.6f}" for a in m.GetAtoms())


atoms = mmff_atoms(SMI)
mol = gto.M(atom=atoms, basis=BASIS, charge=0, spin=0, verbose=0)
print(f"molecule: {mol.natm} atoms, {mol.nao} basis fns ({BASIS}), xc={XC}, eps={EPS}", flush=True)

mf = dft.RKS(mol)
if EPS > 1.0:
    mf = ddCOSMO(mf)
    mf.with_solvent.eps = EPS
mf.xc = XC
mf.max_cycle = 200
mf.conv_tol = 1e-8
e = mf.kernel()
print(f"SCF converged={mf.converged}  E={e:.6f}", flush=True)

nocc = mol.nelectron // 2
mo_e = mf.mo_energy
print(f"HOMO={mo_e[nocc-1]*27.2114:.3f} eV  LUMO={mo_e[nocc]*27.2114:.3f} eV  "
      f"gap={(mo_e[nocc]-mo_e[nocc-1])*27.2114:.3f} eV", flush=True)

td = tddft.TDA(mf)
td.nstates = NST
td.kernel()
osc = td.oscillator_strength()

print(f"\n{'state':>5} {'eV':>7} {'nm':>8} {'f':>8}   dominant transitions (weight)", flush=True)
for i, (ee, f) in enumerate(zip(td.e, osc)):
    ev = ee * 27.2114
    nm = 1240.0 / ev
    x = td.xy[i][0]
    w = 2 * x ** 2
    idx = np.dstack(np.unravel_index(np.argsort(-w.ravel()), w.shape))[0][:3]
    desc = ", ".join(
        f"H{'' if o == nocc-1 else f'-{nocc-1-o}'}->L{'' if v == 0 else f'+{v}'} {w[o,v]:.2f}"
        for o, v in idx if w[o, v] > 0.08)
    star = "  <== max-f" if f == max(osc) else ""
    print(f"{i+1:>5} {ev:7.3f} {nm:8.1f} {f:8.4f}   {desc}{star}", flush=True)

bright = [(1240.0 / (e_ * 27.2114), f) for e_, f in zip(td.e, osc) if f > 0.05]
print(f"\nmax-f pick (what the pipeline reported): "
      f"{max(bright, key=lambda t: t[1])[0]:.1f} nm" if bright else "\nno bright state", flush=True)
print(f"lowest bright (f>0.05): {bright[0][0]:.1f} nm" if bright else "", flush=True)
print(f"S1: {1240.0/(td.e[0]*27.2114):.1f} nm  f={osc[0]:.4f}", flush=True)
