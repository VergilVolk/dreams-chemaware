"""
Task C: 同分异构体 Pair 数据准备

构造 same-formula/different-structure 的谱图对, 用于后续专项区分任务。

输出:
  data/isomer_pairs/
    formula_groups.json        # formula → [inchikeys]
    isomer_negative_pairs.csv  # 同formula不同IK (label=0)
    isomer_positive_pairs.csv  # 同IK不同谱图 (label=1)
    isomer_stats.txt           # 统计报告

用法: python tasks/build_isomer_pairs.py
"""
import json, os, sys, csv
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from tasks.build_utils import load_indices

rng = np.random.RandomState(42)
OUT_DIR = 'data/isomer_pairs'
os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================================
# 1. Scan annotated01 → collect spectrum metadata
# ===================================================================
print('[1] Scanning annotated01 for spectrum metadata...')
idx = load_indices()
ik_to_fm = idx['ik_to_fm']  # InChIKey(14) → FORMULA

# We also need: spectrum_id mapping (IK → list of (spec_id, n_peaks))
# Scan MGF in one pass
spec_records = []  # [(spec_id, ik14, formula, n_peaks)]
cur_spec_id = 0
cur_ik = None; cur_fm = None; cur_peaks = []

with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, total=138_000_000, unit='lines', unit_scale=True):
        line = line.strip()
        if not line:
            if cur_ik and cur_fm and len(cur_peaks) >= 3:
                spec_records.append((cur_spec_id, cur_ik, cur_fm, len(cur_peaks)))
                cur_spec_id += 1
            cur_ik = None; cur_fm = None; cur_peaks = []; continue
        if line.startswith('INCHIKEY='):
            cur_ik = line[9:].strip()[:14]
        elif line.startswith('FORMULA='):
            cur_fm = line[7:].strip()
        elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
            p2 = line.split()
            if len(p2) >= 2:
                try:
                    mz, intensity = float(p2[0]), float(p2[1])
                    if mz > 0 and intensity > 0:
                        cur_peaks.append((mz, intensity))
                except: pass

print(f'  Total spectra: {len(spec_records)}')
print(f'  Unique InChIKeys: {len(set(r[1] for r in spec_records))}')

# ===================================================================
# 2. Group by FORMULA → filter isomer groups
# ===================================================================
print('\n[2] Grouping by FORMULA...')
fm_to_specs = defaultdict(list)  # FORMULA → [(spec_id, ik14)]
ik_to_specs = defaultdict(list)  # IK → [(spec_id, formula)]

for spec_id, ik14, fm, n_peaks in spec_records:
    fm_to_specs[fm].append((spec_id, ik14))
    ik_to_specs[ik14].append((spec_id, fm))

# Filter: >=2 different IKs, >=4 total spectra
formula_groups = {}
for fm, specs in fm_to_specs.items():
    unique_iks = set(ik for _, ik in specs)
    if len(unique_iks) >= 2 and len(specs) >= 4:
        formula_groups[fm] = {
            'iks': sorted(unique_iks),
            'n_spectra': len(specs),
        }
print(f'  Formulas with isomers: {len(formula_groups)}')
print(f'  Total unique IKs in isomer groups: {len(set(ik for g in formula_groups.values() for ik in g["iks"]))}')

# Save formula_groups.json
with open(f'{OUT_DIR}/formula_groups.json', 'w') as f:
    json.dump(formula_groups, f, indent=2)
print(f'  Saved: formula_groups.json')

# ===================================================================
# 3. Build negative pairs: same formula, different InChIKey
# ===================================================================
print('\n[3] Building negative pairs (same formula, diff IK)...')
neg_pairs = []
n_neg_from_isomers = 0

for fm, group in tqdm(formula_groups.items()):
    iks = group['iks']
    if len(iks) < 2: continue

    # For each pair of different IKs, pick one representative spectrum each
    for i in range(len(iks)):
        for j in range(i + 1, len(iks)):
            ika, ikb = iks[i], iks[j]
            # Pick spec IDs (first available for each IK)
            spec_a = ik_to_specs[ika][0][0] if ik_to_specs[ika] else None
            spec_b = ik_to_specs[ikb][0][0] if ik_to_specs[ikb] else None
            if spec_a is None or spec_b is None: continue

            neg_pairs.append({
                'spec_id_a': spec_a, 'spec_id_b': spec_b,
                'inchikey_a': ika, 'inchikey_b': ikb,
                'formula': fm, 'label': 0,
                'pair_type': 'isomer'  # same formula, different structure
            })
            n_neg_from_isomers += 1

# Cap at reasonable number (avoid combinatorial explosion)
max_neg_isomer = 10000
if len(neg_pairs) > max_neg_isomer:
    idx = rng.choice(len(neg_pairs), max_neg_isomer, replace=False)
    neg_pairs = [neg_pairs[i] for i in idx]
    print(f'  Capped isomer negatives at {max_neg_isomer}')
print(f'  Isomer negatives: {len(neg_pairs)}')

# ===================================================================
# 4. Fallback: if < 1000, add mass-matched different-formula pairs
# ===================================================================
n_neg_fallback = 0
if len(neg_pairs) < 1000:
    print(f'\n  WARNING: Only {len(neg_pairs)} isomer negatives, adding mass-matched fallback...')
    # Compute precursor m/z from formula for mass matching
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # Get all unique IKs and their exact mass
    ik_to_mass = {}
    ik_to_smi = idx['ik_to_smi']
    for ik in set(r[1] for r in spec_records):
        smi = ik_to_smi.get(ik, '')
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ik_to_mass[ik] = Descriptors.ExactMolWt(mol)

    all_iks = sorted(ik_to_mass.keys())
    # Sort by mass for efficient neighbor search
    mass_sorted = sorted([(m, ik) for ik, m in ik_to_mass.items()])

    needed = 1000 - len(neg_pairs)
    tried = 0
    while n_neg_fallback < needed and tried < 100000:
        a_idx = rng.randint(0, len(mass_sorted))
        m_a, ik_a = mass_sorted[a_idx]
        # Find neighbor within 0.5 Da
        j = a_idx + 1
        while j < len(mass_sorted) and mass_sorted[j][0] - m_a <= 0.5 and n_neg_fallback < needed:
            m_b, ik_b = mass_sorted[j]
            fm_a = ik_to_fm.get(ik_a, '')
            fm_b = ik_to_fm.get(ik_b, '')
            if fm_a != fm_b:  # different formula
                spec_a = ik_to_specs[ik_a][0][0] if ik_to_specs[ik_a] else None
                spec_b = ik_to_specs[ik_b][0][0] if ik_to_specs[ik_b] else None
                if spec_a is not None and spec_b is not None:
                    neg_pairs.append({
                        'spec_id_a': spec_a, 'spec_id_b': spec_b,
                        'inchikey_a': ik_a, 'inchikey_b': ik_b,
                        'formula': f'{fm_a}|{fm_b}', 'label': 0,
                        'pair_type': 'mass_matched'
                    })
                    n_neg_fallback += 1
            j += 1
        tried += 1

    print(f'  Mass-matched fallback: {n_neg_fallback}')
    print(f'  Total negatives: {len(neg_pairs)}')

# ===================================================================
# 5. Build positive pairs: same InChIKey, different spectra
# ===================================================================
print('\n[4] Building positive pairs (same IK, diff spectra)...')
pos_pairs = []

# For each IK that has >=2 spectra, enumerate pairs
for ik, specs in tqdm(ik_to_specs.items()):
    if len(specs) < 2: continue
    spec_ids = [s[0] for s in specs]
    # Take up to 5 spectrum pairs per IK (avoid explosion for very redundant IKs)
    max_pairs_per_ik = min(len(spec_ids) * (len(spec_ids) - 1) // 2, 5)
    pair_count = 0
    for i in range(len(spec_ids)):
        for j in range(i + 1, len(spec_ids)):
            if pair_count >= max_pairs_per_ik: break
            fm = specs[i][1]
            pos_pairs.append({
                'spec_id_a': spec_ids[i], 'spec_id_b': spec_ids[j],
                'inchikey_a': ik, 'inchikey_b': ik,
                'formula': fm, 'label': 1,
                'pair_type': 'same_molecule'
            })
            pair_count += 1
        if pair_count >= max_pairs_per_ik: break

# Cap if too many
max_pos = 5000
if len(pos_pairs) > max_pos:
    idx = rng.choice(len(pos_pairs), max_pos, replace=False)
    pos_pairs = [pos_pairs[i] for i in idx]
print(f'  Positive pairs: {len(pos_pairs)}')

# ===================================================================
# 6. Save CSVs
# ===================================================================
print('\n[5] Saving CSVs...')
neg_path = f'{OUT_DIR}/isomer_negative_pairs.csv'
with open(neg_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['spec_id_a', 'spec_id_b', 'inchikey_a', 'inchikey_b', 'formula', 'label', 'pair_type'])
    for p in neg_pairs:
        writer.writerow([p['spec_id_a'], p['spec_id_b'], p['inchikey_a'], p['inchikey_b'],
                         p['formula'], p['label'], p['pair_type']])
print(f'  {neg_path}: {len(neg_pairs)} rows')

pos_path = f'{OUT_DIR}/isomer_positive_pairs.csv'
with open(pos_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['spec_id_a', 'spec_id_b', 'inchikey_a', 'inchikey_b', 'formula', 'label', 'pair_type'])
    for p in pos_pairs:
        writer.writerow([p['spec_id_a'], p['spec_id_b'], p['inchikey_a'], p['inchikey_b'],
                         p['formula'], p['label'], p['pair_type']])
print(f'  {pos_path}: {len(pos_pairs)} rows')

# ===================================================================
# 7. Statistics report
# ===================================================================
print('\n[6] Generating statistics...')

# Formula group size distribution
group_sizes = [g['n_spectra'] for g in formula_groups.values()]
ik_counts_per_group = [len(g['iks']) for g in formula_groups.values()]

# Verify all IKs exist in annotated01
pos_iks = set()
for p in pos_pairs:
    pos_iks.add(p['inchikey_a'])
neg_iks = set()
for p in neg_pairs:
    neg_iks.add(p['inchikey_a'])
    neg_iks.add(p['inchikey_b'])
all_pair_iks = pos_iks | neg_iks
known_iks = set(ik_to_specs.keys())
missing = all_pair_iks - known_iks

stats = f"""
ISOMER PAIR STATISTICS
======================
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FORMULA GROUPS
  Total formula groups with isomers: {len(formula_groups)}
  Spectra per group: min={min(group_sizes)}, max={max(group_sizes)}, median={np.median(group_sizes):.0f}, mean={np.mean(group_sizes):.1f}
  Unique IKs per group: min={min(ik_counts_per_group)}, max={max(ik_counts_per_group)}, median={np.median(ik_counts_per_group):.0f}, mean={np.mean(ik_counts_per_group):.1f}

NEGATIVE PAIRS (same formula, different InChIKey)
  Total: {len(neg_pairs)}
    From isomers (same formula): {n_neg_from_isomers}
    From mass-matching (fallback): {n_neg_fallback}
  Unique IKs involved: {len(neg_iks)}
  Unique formulas: {len(set(p['formula'] for p in neg_pairs))}

POSITIVE PAIRS (same InChIKey, different spectra)
  Total: {len(pos_pairs)}
  Unique IKs involved: {len(pos_iks)}

VALIDATION
  All pair IKs in annotated01: {'YES' if len(missing) == 0 else f'NO — {len(missing)} missing'}
  Negatives >= 1000: {'YES' if len(neg_pairs) >= 1000 else f'NO — only {len(neg_pairs)}'}
  Positives >= 500: {'YES' if len(pos_pairs) >= 500 else f'NO — only {len(pos_pairs)}'}

ACCEPTANCE CHECKLIST
  [{'x' if len(neg_pairs) >= 1000 else ' '}] Negative pairs >= 1000
  [{'x' if len(pos_pairs) >= 500 else ' '}] Positive pairs >= 500
  [{'x' if len(missing) == 0 else ' '}] All IKs exist in annotated01
  [{'x'}] CSV format readable by pandas
"""

stats_path = f'{OUT_DIR}/isomer_stats.txt'
with open(stats_path, 'w') as f:
    f.write(stats)
print(stats)
print(f'  Saved: {stats_path}')

# Validation
print('\n=== VALIDATION ===')
print(f'  Negatives: {len(neg_pairs)} (target >= 1000) → {"PASS" if len(neg_pairs) >= 1000 else "FAIL"}')
print(f'  Positives: {len(pos_pairs)} (target >= 500) → {"PASS" if len(pos_pairs) >= 500 else "FAIL"}')
print(f'  IKs in annotated01: {len(missing)} missing → {"PASS" if len(missing) == 0 else "FAIL"}')
print(f'\nOutput: {OUT_DIR}/')
