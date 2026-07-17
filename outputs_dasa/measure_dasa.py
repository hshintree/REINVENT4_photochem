import sys, pandas as pd
sys.path.insert(0, "notebooks")
import dasa_chem as dc
from rdkit import Chem
df = pd.read_csv("outputs_dasa/tl_samples.csv")
col = "SMILES" if "SMILES" in df.columns else df.columns[0]
smis = df[col].dropna().tolist()
n_valid = sum(1 for s in smis if Chem.MolFromSmiles(str(s)) is not None)
n_dasa = sum(1 for s in smis if dc.is_dasa(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)))
print(f"sampled {len(smis)} | valid {n_valid} | DASA-gate pass {n_dasa} ({100*n_dasa/max(len(smis),1):.1f}%)")
# show a few DASA hits
hits = [s for s in smis if Chem.MolFromSmiles(str(s)) and dc.is_dasa(Chem.MolFromSmiles(str(s)))][:6]
for h in hits: print("  DASA:", h)
