"""GNPS 数据质量检查"""
import json, os
from collections import Counter

print("Loading GNPS_clean.json (1186MB)...")
with open("data/GNPS_clean.json") as f:
    data = json.load(f)
print(f"Total spectra: {len(data)}")

# Check fields
s = data[0]
print(f"Fields: {list(s.keys())}")

# SMILES validity
from rdkit import Chem
valid_smi = 0; invalid_smi = 0
for s in data:
    smi = s.get("SMILES","").strip()
    if smi and Chem.MolFromSmiles(smi): valid_smi += 1
    else: invalid_smi += 1
print(f"\nSMILES: valid={valid_smi}, invalid={invalid_smi} ({invalid_smi/len(data)*100:.1f}%)")

# InChIKey
ik_counts = Counter(s.get("INCHIKEY","") for s in data)
valid_ik = sum(1 for ik in ik_counts if ik)
print(f"InChIKeys: valid={valid_ik}, unique={len(ik_counts)}")
multi_ik = sum(1 for ik,c in ik_counts.items() if c >= 2)
print(f"Multi-spectrum IKs (>=2 spectra): {multi_ik}")

# Peaks
peak_counts = [len(s.get("peaks",[])) for s in data]
print(f"\nPeaks: min={min(peak_counts)}, mean={sum(peak_counts)/len(peak_counts):.0f}, max={max(peak_counts)}")

# Ion mode
ion_modes = Counter(s.get("IONMODE","").strip().upper() for s in data)
print(f"Ion modes: {ion_modes.most_common(5)}")

# Overall assessment
print(f"\n=== QUALITY ASSESSMENT ===")
print(f"Total: {len(data)}")
print(f"SMILES valid: {valid_smi} ({valid_smi/len(data)*100:.1f}%)")
print(f"InChIKey present: {valid_ik}")
print(f"Multi-spectrum molecules: {multi_ik}")
print(f"Suitable for MIL training: {'YES' if multi_ik >= 100 and valid_smi > 100000 else 'NEEDS MORE'}")
