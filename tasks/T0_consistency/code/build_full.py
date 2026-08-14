"""
T0 全量构造 v3 — 扫描模式，内存安全

策略: 第一遍扫描建索引 → 选子集 → 第二遍加载 → 配对
"""
import json, os, re
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

ALL_MGF = []
for f in os.listdir('data'):
    if f.endswith('.mgf') and f not in ('massbank_50.mgf','_eval_specs.mgf'):
        ALL_MGF.append(f'data/{f}')

print(f'Files: {len(ALL_MGF)}')

# Pass 1: scan InChIKeys + count spectra per IK
ik_counts = Counter()
ik_sources = defaultdict(set)
total_specs = 0

for fp in ALL_MGF:
    print(f'Scanning {os.path.basename(fp)[:50]}...')
    cur_ik = None; has_smi = False
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if 'BEGIN IONS' in line: cur_ik = None; has_smi = False
            elif 'END IONS' in line:
                if cur_ik and has_smi:
                    ik_counts[cur_ik] += 1
                    total_specs += 1
            elif 'INCHIKEY=' in line and not cur_ik:
                cur_ik = line.split('=', 1)[1].strip()[:27]
            elif 'INCHI=' in line and not cur_ik:
                try:
                    from rdkit.Chem.inchi import InchiToInchiKey
                    cur_ik = InchiToInchiKey(line.split('=', 1)[1].strip())
                except: pass
            elif 'SMILES=' in line: has_smi = True

print(f'\nTotal: {total_specs} spectra, {len(ik_counts)} InChIKeys')

# Distribution
counts = list(ik_counts.values())
multi_ik = [(ik, c) for ik, c in ik_counts.items() if c >= 2]
print(f'Multi-spectrum IKs: {len(multi_ik)}')
print(f'Spectrum counts: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}, median={np.median(counts):.0f}')
for cutoff in [2, 3, 5, 10, 20, 50, 100]:
    n = sum(1 for c in counts if c >= cutoff)
    print(f'  >= {cutoff:3d}: {n:6d}')

# Build balanced pairs: 1 positive pair per multi-IK, up to 50K
rng = np.random.RandomState(42)
n_pairs = min(50000, len(multi_ik))
selected_iks = set()
for ik, c in sorted(multi_ik, key=lambda x: -x[1])[:n_pairs]:
    selected_iks.add(ik)

print(f'\nSelected {len(selected_iks)} IKs for pair construction')
print(f'Target: 1 pos pair per IK = {len(selected_iks)} positive pairs')

# Negative: random diff-IK, 10K total
all_iks = list(ik_counts.keys())
neg_pairs_count = min(10000, len(all_iks) * 2)
neg_iks = set()
for _ in range(neg_pairs_count * 2):
    a, b = rng.choice(all_iks, 2, replace=False)
    if a != b: neg_iks.add(a); neg_iks.add(b)
    if len(neg_iks) >= min(20000, len(all_iks)): break

needed_iks = selected_iks | neg_iks
print(f'Need to load spectra for {len(needed_iks)} IKs ({len(selected_iks)} pos + {len(neg_iks)} neg)')

# Pass 2: load only needed spectra
ik_to_spectra = defaultdict(list)
loaded = 0
for fp in ALL_MGF:
    if loaded >= len(needed_iks) * 3: break  # enough
    cur = {}; peaks = []; cur_ik = None
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur and peaks and cur_ik in needed_iks:
                    cur['peaks'] = peaks
                    ik_to_spectra[cur_ik].append(cur)
                    loaded += 1
                cur = {}; peaks = []; cur_ik = None; continue
            if '=' in line and line[0].isalpha():
                k, v = line.split('=', 1)
                if k == 'SMILES': cur['SMILES'] = v
                elif k == 'INCHIKEY': cur_ik = v[:27]; cur['INCHIKEY'] = v[:27]
                elif k == 'PEPMASS': cur['PEPMASS'] = v
                elif k == 'IONMODE': cur['IONMODE'] = v
            elif line and (line[0].isdigit() or line[0] == '-'):
                p = line.split()
                if len(p) >= 2:
                    try: mz, i = float(p[0]), float(p[1])
                    except: continue
                    if mz > 0 and i > 0: peaks.append((mz, i))
    if loaded % 10000 == 0:
        print(f'  Loaded {loaded} spectra...')

print(f'Loaded {loaded} spectra for {len(ik_to_spectra)} IKs')

# Build pairs
pos_pairs = []; neg_pairs = []
pos_iks_found = set()
for ik in selected_iks:
    specs = ik_to_spectra.get(ik, [])
    if len(specs) < 2: continue
    a, b = rng.choice(len(specs), 2, replace=False)
    pos_pairs.append({'ik': ik, 'smiles_a': specs[a].get('SMILES', ''), 'smiles_b': specs[b].get('SMILES', '')})
    pos_iks_found.add(ik)

neg_ik_list = list(ik_to_spectra.keys())
for _ in range(10000):
    ika, ikb = rng.choice(neg_ik_list, 2, replace=False)
    if ika == ikb: continue
    neg_pairs.append({'ik_a': ika, 'ik_b': ikb})

print(f'\n=== T0 FULL RESULTS ===')
print(f'  Total spectra scanned: {total_specs}')
print(f'  Unique InChIKeys: {len(ik_counts)}')
print(f'  Multi-spectrum IKs: {len(multi_ik)}')
print(f'  Positive pairs: {len(pos_pairs)} (from {len(pos_iks_found)} IKs)')
print(f'  Negative pairs: {len(neg_pairs)}')

with open('tasks/T0_consistency/test_cases/full_pairs.json', 'w') as f:
    json.dump({'positive': pos_pairs, 'negative': neg_pairs}, f)
print(f'\nSaved: tasks/T0_consistency/test_cases/full_pairs.json')
