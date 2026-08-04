"""
T1 v2 — MCES [3,5] Hard Negative Fix

Minimal change from v1 build_test_cases.py (well-tested pipeline).
Only changes:
  1. neg_hard: MCES [6,10] → [3,5] (true hard negatives for DreaMS)
  2. neg_easy: from T3 → MCES [6,10] (too easy, moved down)
  3. Build triplets from MCES pairs (anchor+pos MCES[0,2], neg MCES[3,5])
  4. Output to tasks/T1_v2_hard/

Same as v1:
  - Morgan Tanimoto > 0.2 pre-filter
  - max 2000 formula groups, 20 pairs/group
  - myopic-mces computation with MAX_BONDS_FOR_MCES=50
  - 9:1 train/val split by formula

Usage:
  python tasks/T1_near_isomers/code/build_test_cases_v2.py
"""
import json, os, sys, argparse
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from tasks.build_utils import load_indices, compute_morgan_fp
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from myopic_mces import MCES as compute_mces_raw

p = argparse.ArgumentParser()
p.add_argument('--validate', action='store_true',
               help='Quick validation run on 100 formula groups')
p.add_argument('--n_groups', type=int, default=100)
p.add_argument('--max_pairs_per_group', type=int, default=20)
p.add_argument('--max_groups', type=int, default=0,
               help='Max formula groups (0=2000)')
args = p.parse_args()

rng = np.random.RandomState(42)
OUT_DIR = 'tasks/T1_v2_hard'
os.makedirs(OUT_DIR, exist_ok=True)

MAX_BONDS_FOR_MCES = 50


def compute_mces(smi_a, smi_b, nb_a=None, nb_b=None):
    if nb_a is not None and nb_b is not None:
        if nb_a > MAX_BONDS_FOR_MCES or nb_b > MAX_BONDS_FOR_MCES:
            return None
    try:
        return compute_mces_raw(smi_a, smi_b)[1]
    except Exception:
        return None


def main():
    print('[1] Loading indices...')
    idx = load_indices()
    ik_to_smi = idx['ik_to_smi']
    ik_to_fm = idx['ik_to_fm']
    fm_to_iks = idx['fm_to_iks']
    print(f'  Multi-IK formulas: {len(fm_to_iks)}')

    # Scope
    fm_items = list(fm_to_iks.items())
    if args.validate:
        if args.n_groups < len(fm_items):
            indices = rng.choice(len(fm_items), args.n_groups, replace=False)
            fm_items = [fm_items[i] for i in sorted(indices)]
        print(f'  [VALIDATE] {len(fm_items)} groups')
    else:
        n_total = len(fm_items)
        max_g = args.max_groups if args.max_groups > 0 else min(n_total, 2000)
        if max_g < n_total:
            indices = rng.choice(n_total, max_g, replace=False)
            fm_items = [fm_items[i] for i in sorted(indices)]
        print(f'  [FULL] {len(fm_items)}/{n_total} groups')

    # MCES computation (same as v1)
    print('\n[2] Computing MCES on formula groups...')
    all_mces = []
    for fm, iks in tqdm(fm_items, desc='Formula groups'):
        mols = {}; fps = {}
        for ik in iks:
            smi = ik_to_smi.get(ik, '')
            if not smi: continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            mols[ik] = mol
            fp = compute_morgan_fp(smi)
            if fp is not None: fps[ik] = fp

        if len(mols) < 2: continue

        ik_list = list(mols.keys())
        candidates = []
        for i in range(len(ik_list)):
            for j in range(i + 1, len(ik_list)):
                ika, ikb = ik_list[i], ik_list[j]
                if ika in fps and ikb in fps:
                    tan = DataStructs.TanimotoSimilarity(fps[ika], fps[ikb])
                    if tan < 0.2: continue
                    candidates.append((ika, ikb, tan))
                else:
                    candidates.append((ika, ikb, -1))

        if len(candidates) > args.max_pairs_per_group:
            idx_cand = rng.choice(len(candidates), args.max_pairs_per_group, replace=False)
            candidates = [candidates[i] for i in idx_cand]

        for ika, ikb, tan in candidates:
            smi_a = ik_to_smi[ika]; smi_b = ik_to_smi[ikb]
            nb_a = mols[ika].GetNumBonds(); nb_b = mols[ikb].GetNumBonds()
            mces_raw = compute_mces(smi_a, smi_b, nb_a, nb_b)
            if mces_raw is None: continue
            max_bonds = max(nb_a, nb_b)
            mces_norm = mces_raw / max_bonds if max_bonds > 0 else 0
            all_mces.append({
                'mces_raw': mces_raw, 'mces_norm': round(mces_norm, 4),
                'tanimoto': round(tan, 4), 'n_bonds_a': nb_a, 'n_bonds_b': nb_b,
                'ik_a': ika, 'ik_b': ikb, 'smi_a': smi_a[:80], 'smi_b': smi_b[:80], 'fm': fm,
            })

    print(f'\n  MCES computed: {len(all_mces)}')
    if len(all_mces) == 0:
        print('ERROR: No MCES values!'); return

    mces_raw = [x['mces_raw'] for x in all_mces]
    print(f'  Raw MCES: min={min(mces_raw)} max={max(mces_raw)} mean={np.mean(mces_raw):.1f} median={np.median(mces_raw):.0f}')

    # ===================================================================
    # V2 CHANGE: reassign categories
    # ===================================================================
    print(f'\n[3] V2 category assignment:')
    pos_pairs = [x for x in all_mces if 0 <= x['mces_raw'] <= 2]
    neg_hard = [x for x in all_mces if 3 <= x['mces_raw'] <= 5]   # ← THE FIX
    neg_easy = [x for x in all_mces if 6 <= x['mces_raw'] <= 10]   # ← was neg_hard

    print(f'  Positive (MCES 0-2):      {len(pos_pairs)}')
    print(f'  Neg hard (MCES 3-5):     {len(neg_hard)}  ← WAS EXCLUDED, NOW HARD')
    print(f'  Neg easy (MCES 6-10):    {len(neg_easy)}   ← WAS HARD, NOW EASY')

    tan_pos = [p['tanimoto'] for p in pos_pairs if p['tanimoto'] >= 0]
    tan_hard = [p['tanimoto'] for p in neg_hard if p['tanimoto'] >= 0]
    if tan_pos: print(f'  Pos Tanimoto: mean={np.mean(tan_pos):.4f} std={np.std(tan_pos):.4f}')
    if tan_hard: print(f'  Neg hard Tanimoto: mean={np.mean(tan_hard):.4f} std={np.std(tan_hard):.4f}')

    # ===================================================================
    # V2 CHANGE: Build triplets from MCES pairs
    # ===================================================================
    print(f'\n[4] Building triplets...')
    # Anchor+positive: MCES [0,2] pairs (different IKs, near-isomers)
    # Negative: MCES [3,5] pairs (different IKs, true hard negatives)
    # For each MCES [3,5] pair, pick a random MCES [0,2] pair sharing one IK as anchor+pos

    # Index pos_pairs by IK
    ik_to_pos = defaultdict(list)
    for p in pos_pairs:
        ik_to_pos[p['ik_a']].append(p)
        ik_to_pos[p['ik_b']].append(p)

    triplets = []
    for p in neg_hard:
        ik_a = p['ik_a']; ik_b = p['ik_b']
        # Find a positive pair that shares ik_a
        pos_candidates = ik_to_pos.get(ik_a, [])
        if not pos_candidates:
            pos_candidates = ik_to_pos.get(ik_b, [])
        if not pos_candidates: continue

        pos = pos_candidates[rng.randint(0, len(pos_candidates))]
        # Determine which is the shared anchor
        if pos['ik_a'] == ik_a or pos['ik_b'] == ik_a:
            anchor_ik = ik_a
            pos_ik = pos['ik_b'] if pos['ik_a'] == ik_a else pos['ik_a']
        else:
            anchor_ik = ik_b
            pos_ik = pos['ik_b'] if pos['ik_a'] == ik_b else pos['ik_a']

        triplets.append({
            'anchor_ik': anchor_ik,
            'pos_ik': pos_ik,
            'neg_ik': ik_b if anchor_ik == ik_a else ik_a,
            'pos_mces_raw': pos['mces_raw'],
            'neg_mces_raw': p['mces_raw'],
            'neg_tanimoto': p['tanimoto'],
        })

    print(f'  Built {len(triplets)} triplets')

    # Train/val split by formula
    fm_to_triplets = defaultdict(list)
    for t in triplets:
        ank = t['anchor_ik']
        fm = ik_to_fm.get(ank, '')
        if fm: fm_to_triplets[fm].append(t)

    all_fms = sorted(fm_to_triplets.keys())
    rng.shuffle(all_fms)
    n_train_fm = int(len(all_fms) * 0.9)
    train_fms = set(all_fms[:n_train_fm])
    val_fms = set(all_fms[n_train_fm:])

    train_trips = [t for fm in all_fms if fm in train_fms for t in fm_to_triplets[fm]]
    val_trips = [t for fm in all_fms if fm in val_fms for t in fm_to_triplets[fm]]
    print(f'  Train: {len(train_trips)} ({len(train_fms)} formulas)')
    print(f'  Val:   {len(val_trips)} ({len(val_fms)} formulas)')

    # ===================================================================
    # 5. Save
    # ===================================================================
    print(f'\n[5] Saving to {OUT_DIR}/...')
    pairs_output = {
        'positive': pos_pairs,
        'negative_hard': neg_hard,
        'negative_easy': neg_easy,
    }
    stats_output = {
        'total_mces_computed': len(all_mces),
        'positive_pairs': len(pos_pairs),
        'negative_hard': len(neg_hard),
        'negative_easy': len(neg_easy),
        'triplets_train': len(train_trips),
        'triplets_val': len(val_trips),
        'positive_tanimoto': {'mean': float(np.mean(tan_pos)), 'std': float(np.std(tan_pos))} if tan_pos else {},
        'neg_hard_tanimoto': {'mean': float(np.mean(tan_hard)), 'std': float(np.std(tan_hard))} if tan_hard else {},
        'recommended_margin': 0.4,
    }

    for fn, data in [('pairs_v2.json', pairs_output), ('stats_v2.json', stats_output),
                      ('triplets_train_v2.json', train_trips), ('triplets_val_v2.json', val_trips)]:
        path = f'{OUT_DIR}/{fn}'
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'  {fn}: {os.path.getsize(path)/1e6:.1f}MB')

    print(f'\n=== T1 V2 DONE ===')
    print(f'  Pos={len(pos_pairs)}  Neg(hard,[3,5])={len(neg_hard)}  Neg(easy,[6,10])={len(neg_easy)}')
    print(f'  Triplets: {len(train_trips)} train + {len(val_trips)} val')
    print(f'  Recommended margin: 0.4')


if __name__ == '__main__':
    main()
