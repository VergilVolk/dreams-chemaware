"""
T1 v2 — Hard Negative Fix: MCES [3,5] as negative_hard

Changes from v1:
  - negative_hard: MCES [3,5] (was excluded, was the real hard negatives!)
  - negative_easy: MCES [6,10] (was negative_hard, but too easy for DreaMS)
  - margin: 0.2 → 0.4 (since MCES [3,5] separation is smaller)
  - Output: tasks/T1_v2_hard/

Usage:
  python tasks/T1_near_isomers/code/build_test_cases_v2.py
"""
import json, os, sys, argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from tasks.build_utils import load_indices

rng = np.random.RandomState(42)
OUT_DIR = 'tasks/T1_v2_hard'
os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================================
# 1. Load existing MCES results
# ===================================================================
print('[1] Loading existing T1 MCES data...')
# Load old pairs for MCES data
old_pairs_path = 'tasks/T1_near_isomers/test_cases/pairs.json'
if os.path.exists(old_pairs_path):
    with open(old_pairs_path) as f:
        old_pairs = json.load(f)
    all_mces = old_pairs['positive'] + old_pairs['negative_hard'] + old_pairs['negative_easy']
    # Also load the MCES [3,5] excluded pairs from stats
    with open('tasks/T1_near_isomers/test_cases/stats.json') as f:
        old_stats = json.load(f)
    print(f'  Loaded {len(all_mces)} pairs from v1 pairs.json')
    print(f'  v1 excluded: {old_stats["excluded_mces_3_5"]} pairs with MCES [3,5]')
else:
    print(f'  ERROR: {old_pairs_path} not found')
    sys.exit(1)

# But wait — the excluded MCES [3,5] pairs are NOT in pairs.json!
# They were filtered out during build. We need to REBUILD from scratch.
# Let's do a full rebuild.

print('\n[1b] Rebuilding MCES from annotated01 (need excluded pairs)...')
idx = load_indices()
ik_to_smi = idx['ik_to_smi']
ik_to_fm = idx['ik_to_fm']
fm_to_iks = idx['fm_to_iks']

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from myopic_mces import MCES as compute_mces_raw

MAX_BONDS_FOR_MCES = 50

def compute_mces(smi_a, smi_b, nb_a=None, nb_b=None):
    if nb_a is not None and nb_b is not None:
        if nb_a > MAX_BONDS_FOR_MCES or nb_b > MAX_BONDS_FOR_MCES:
            return None
    try:
        return compute_mces_raw(smi_a, smi_b)[1]
    except Exception:
        return None

# Use same formula groups as v1
fm_items = list(fm_to_iks.items())
max_groups = min(len(fm_items), 2000)
if max_groups < len(fm_items):
    indices = rng.choice(len(fm_items), max_groups, replace=False)
    fm_items = [fm_items[i] for i in sorted(indices)]
print(f'  Processing {len(fm_items)} formula groups (max_pairs_per_group=20)')

all_mces_v2 = []
for fm, iks in tqdm(fm_items, desc='MCES v2'):
    mols = {}; fps = {}
    for ik in iks:
        smi = ik_to_smi.get(ik, '')
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mols[ik] = mol
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        fps[ik] = fp

    if len(mols) < 2: continue

    ik_list = list(mols.keys())
    candidates = []
    for i in range(len(ik_list)):
        for j in range(i + 1, len(ik_list)):
            ika, ikb = ik_list[i], ik_list[j]
            if ika in fps and ikb in fps:
                tan = DataStructs.TanimotoSimilarity(fps[ika], fps[ikb])
                if tan < 0.2: continue  # same v1 filter
                candidates.append((ika, ikb, tan))

    if len(candidates) > 20:
        idx_cand = rng.choice(len(candidates), 20, replace=False)
        candidates = [candidates[i] for i in idx_cand]

    for ika, ikb, tan in candidates:
        smi_a = ik_to_smi[ika]; smi_b = ik_to_smi[ikb]
        nb_a = mols[ika].GetNumBonds(); nb_b = mols[ikb].GetNumBonds()
        mces_raw = compute_mces(smi_a, smi_b, nb_a, nb_b)
        if mces_raw is None: continue
        max_bonds = max(nb_a, nb_b)
        mces_norm = mces_raw / max_bonds if max_bonds > 0 else 0
        all_mces_v2.append({
            'mces_raw': mces_raw, 'mces_norm': round(mces_norm, 4),
            'tanimoto': round(tan, 4), 'n_bonds_a': nb_a, 'n_bonds_b': nb_b,
            'ik_a': ika, 'ik_b': ikb, 'smi_a': smi_a[:80], 'smi_b': smi_b[:80], 'fm': fm,
        })

print(f'\n  MCES v2 computed: {len(all_mces_v2)}')

# ===================================================================
# 2. Assign new labels
# ===================================================================
print('\n[2] Assigning v2 labels...')
pos_pairs = [x for x in all_mces_v2 if 0 <= x['mces_raw'] <= 2]
neg_hard = [x for x in all_mces_v2 if 3 <= x['mces_raw'] <= 5]  # THE FIX
neg_easy = [x for x in all_mces_v2 if 6 <= x['mces_raw'] <= 10]

print(f'  Positive (MCES 0-2):      {len(pos_pairs)}')
print(f'  Neg hard (MCES 3-5):     {len(neg_hard)}  ← WAS EXCLUDED, NOW HARD')
print(f'  Neg easy (MCES 6-10):    {len(neg_easy)}   ← WAS HARD, NOW EASY')

# Tanimoto stats
tan_pos = [p['tanimoto'] for p in pos_pairs if p['tanimoto'] >= 0]
tan_hard = [p['tanimoto'] for p in neg_hard if p['tanimoto'] >= 0]
tan_easy = [p['tanimoto'] for p in neg_easy if p['tanimoto'] >= 0]
print(f'\n  Pos Tanimoto:     mean={np.mean(tan_pos):.4f} std={np.std(tan_pos):.4f}')
print(f'  Neg hard Tanimoto: mean={np.mean(tan_hard):.4f} std={np.std(tan_hard):.4f}')
print(f'  Neg easy Tanimoto: mean={np.mean(tan_easy):.4f} std={np.std(tan_easy):.4f}')

# ===================================================================
# 3. Build triplets
# ===================================================================
print('\n[3] Building MCES [3,5] hard negative triplets...')

# Get same-molecule pairs for anchor-positive
# Scan annotated01 for IKs with >=2 spectra
ik_to_spectra = defaultdict(list)
cur_ik = None; cur_peaks = []
with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, desc='Scan MGF', total=138_000_000, unit='lines', unit_scale=True):
        line = line.strip()
        if not line:
            if cur_ik and len(cur_peaks) >= 3:
                ik_to_spectra[cur_ik].append(cur_peaks[:])
            cur_ik = None; cur_peaks = []; continue
        if line.startswith('INCHIKEY='): cur_ik = line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
            p2 = line.split()
            if len(p2) >= 2:
                try:
                    mz, intensity = float(p2[0]), float(p2[1])
                    if mz > 0 and intensity > 0: cur_peaks.append((mz, intensity))
                except: pass

multi_iks = {ik: specs for ik, specs in ik_to_spectra.items() if len(specs) >= 2}
print(f'  {len(multi_iks)} IKs with >=2 spectra')

# Build triplets: anchor(ik), pos(same ik diff spec), neg(diff ik, MCES 3-5)
# For each MCES [3,5] pair, create a triplet:
#   anchor: random spectrum of ik_a
#   pos:    different spectrum of ik_a
#   neg:    spectrum of ik_b
triplets = []
for p in neg_hard:
    ik_a = p['ik_a'][:14]
    ik_b = p['ik_b'][:14]
    if ik_a not in multi_iks or len(multi_iks[ik_a]) < 2: continue
    if ik_b not in ik_to_spectra or len(ik_to_spectra[ik_b]) < 1: continue

    # Pick first 2 spectra of anchor as anchor+pos
    triplets.append({
        'anchor_ik': ik_a,
        'pos_ik': ik_a,       # same molecule, different spectrum
        'neg_ik': ik_b,       # different molecule, MCES [3,5]
        'neg_mces_raw': p['mces_raw'],
        'neg_mces_norm': p['mces_norm'],
        'neg_tanimoto': p['tanimoto'],
    })

print(f'  Built {len(triplets)} triplets')

# Also include MCES=0 anchor-positive pairs (stereoisomers)
for p in pos_pairs:
    if p['mces_raw'] > 0: continue  # only stereoisomers (MCES=0)
    ik_a = p['ik_a'][:14]
    ik_b = p['ik_b'][:14]
    if ik_a not in multi_iks or len(multi_iks[ik_a]) < 2: continue
    if ik_b not in ik_to_spectra or len(ik_to_spectra[ik_b]) < 1: continue
    triplets.append({
        'anchor_ik': ik_a,
        'pos_ik': ik_a,       # same IK, diff spectrum (stereochemistry doesn't change IK!)
        'neg_ik': ik_b,       # stereoisomer as negative
        'neg_mces_raw': p['mces_raw'],
        'neg_mces_norm': p['mces_norm'],
        'neg_tanimoto': p['tanimoto'],
    })

print(f'  After adding MCES=0 triplets: {len(triplets)}')

# ===================================================================
# 4. Train/val split by formula
# ===================================================================
print('\n[4] Train/val split...')
# Collect formulas used in triplets
fm_to_triplets = defaultdict(list)
for t in triplets:
    # Get formula from ik_to_fm
    ank = t['anchor_ik']
    fm = ik_to_fm.get(ank, '')
    if fm: fm_to_triplets[fm].append(t)

all_fms = sorted(fm_to_triplets.keys())
rng.shuffle(all_fms)
n_train_fm = int(len(all_fms) * 0.9)
train_fms = set(all_fms[:n_train_fm])
val_fms = set(all_fms[n_train_fm:])

train_trips = []; val_trips = []
for fm, ts in fm_to_triplets.items():
    if fm in train_fms:
        train_trips.extend(ts)
    else:
        val_trips.extend(ts)

print(f'  Train: {len(train_trips)} triplets ({len(train_fms)} formulas)')
print(f'  Val:   {len(val_trips)} triplets ({len(val_fms)} formulas)')

# ===================================================================
# 5. Save
# ===================================================================
print('\n[5] Saving...')
pairs_output = {
    'positive': pos_pairs,
    'negative_hard': neg_hard,
    'negative_easy': neg_easy,
}
stats_output = {
    'total_mces_computed': len(all_mces_v2),
    'positive_pairs': len(pos_pairs),
    'negative_hard': len(neg_hard),
    'negative_easy': len(neg_easy),
    'triplets_train': len(train_trips),
    'triplets_val': len(val_trips),
    'positive_tanimoto': {'mean': float(np.mean(tan_pos)), 'std': float(np.std(tan_pos))} if tan_pos else {},
    'neg_hard_tanimoto': {'mean': float(np.mean(tan_hard)), 'std': float(np.std(tan_hard))} if tan_hard else {},
    'neg_easy_tanimoto': {'mean': float(np.mean(tan_easy)), 'std': float(np.std(tan_easy))} if tan_easy else {},
    'recommended_margin': 0.4,
}

for fn, data in [('pairs_v2.json', pairs_output), ('stats_v2.json', stats_output),
                  ('triplets_train_v2.json', train_trips), ('triplets_val_v2.json', val_trips)]:
    path = f'{OUT_DIR}/{fn}'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  {fn}: {os.path.getsize(path)/1e6:.1f}MB')

print(f'\n=== T1 V2 DONE ===')
print(f'  Pos: {len(pos_pairs)}  Neg(hard): {len(neg_hard)}  Neg(easy): {len(neg_easy)}')
print(f'  Triplets: {len(train_trips)} train + {len(val_trips)} val')
print(f'  Margin: 0.4 (recommended for MCES [3,5])')
