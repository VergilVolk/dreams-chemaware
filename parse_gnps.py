"""解析 GNPS MGF → JSON（SMILES+InChIKey+peaks）"""
from rdkit.Chem.inchi import MolFromInchi, InchiToInchiKey
import json, os
from tqdm import tqdm

spectra = []; cur = {}; peaks = []
print('Parsing GNPS_ALL_GNPS.mgf (2.8GB)...')
with open('data/GNPS_ALL_GNPS.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line:
            if cur and peaks: cur['peaks'] = peaks; spectra.append(cur)
            cur = {}; peaks = []; continue
        if '=' in line and line[0].isalpha():
            k, v = line.split('=', 1)
            if k in ('SMILES', 'INCHI', 'PEPMASS', 'IONMODE', 'CHARGE', 'MSLEVEL'): cur[k] = v
        elif line and (line[0].isdigit() or line[0] == '-'):
            p = line.split()
            if len(p) >= 2:
                try: mz, i = float(p[0]), float(p[1])
                except: continue
                if mz > 0 and i > 0: peaks.append((mz, i))
    if cur and peaks: cur['peaks'] = peaks; spectra.append(cur)

print(f'Total raw: {len(spectra)}')

good = []
for s in spectra:
    if not s.get('SMILES', '').strip(): continue
    if not s.get('INCHI', '').strip(): continue
    if len(s.get('peaks', [])) < 3: continue
    try:
        s['INCHIKEY'] = InchiToInchiKey(s['INCHI'])
        good.append(s)
    except: pass

print(f'Valid (SMILES+INCHI+>=3peaks): {len(good)}')

with open('data/GNPS_clean.json', 'w') as f:
    json.dump(good[:500000], f)

size = os.path.getsize('data/GNPS_clean.json') / 1e6
print(f'Saved {min(500000, len(good))} to GNPS_clean.json ({size:.0f}MB)')
