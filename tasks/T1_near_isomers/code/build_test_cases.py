"""
T1 v5 — 近同分异构体 (Near-Isomers)
数据驱动阈值，基于 2119 对同分子式 MCES 分布核准

构造:
  候选生成: 同 FORMULA + 不同 IK + Morgan Tanimoto > 0.2
  正样本:  raw MCES [0, 2]   (近同分异构体: 几乎相同骨架, 1-2键差异)
  负样本(hard): raw MCES [6, 10]  (明确不同异构体: 同分子式下骨架有实质差异)
  负样本(easy): 随机不同 FORMULA

用法:
  python tasks/T1_near_isomers/code/build_test_cases.py --validate
  python tasks/T1_near_isomers/code/build_test_cases.py
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
               help='只在 100 个 formula group 上运行，打印 MCES 分布')
p.add_argument('--n_groups', type=int, default=100,
               help='validate 模式下的 formula group 数量')
p.add_argument('--max_pairs_per_group', type=int, default=20,
               help='每个 formula group 的最大 MCES 配对数量')
p.add_argument('--max_groups', type=int, default=0,
               help='最大处理 group 数量 (0=全部)')
args = p.parse_args()

rng = np.random.RandomState(42)
OUT_DIR = 'tasks/T1_near_isomers/test_cases'
os.makedirs(OUT_DIR, exist_ok=True)


MAX_BONDS_FOR_MCES = 50  # Skip molecules > 50 bonds (MILP too slow)


def compute_mces(smi_a, smi_b, nb_a=None, nb_b=None):
    """Safe MCES wrapper with bond-count guard"""
    # Quick bond-count check before expensive MCES
    if nb_a is not None and nb_b is not None:
        if nb_a > MAX_BONDS_FOR_MCES or nb_b > MAX_BONDS_FOR_MCES:
            return None  # Skip large molecules
    try:
        result = compute_mces_raw(smi_a, smi_b)
        return result[1]  # raw MCES
    except Exception:
        return None


def main():
    print('[1] Loading indices...')
    idx = load_indices()
    ik_to_smi = idx['ik_to_smi']
    ik_to_fm = idx['ik_to_fm']
    fm_to_iks = idx['fm_to_iks']

    print(f'  Multi-IK formulas: {len(fm_to_iks)}')

    fm_sizes = [len(v) for v in fm_to_iks.values()]
    print(f'  Group sizes: max={max(fm_sizes)}, mean={np.mean(fm_sizes):.1f}, median={np.median(fm_sizes):.0f}')

    # Determine scope
    fm_items = list(fm_to_iks.items())
    if args.validate:
        if args.n_groups < len(fm_items):
            indices = rng.choice(len(fm_items), args.n_groups, replace=False)
            fm_items = [fm_items[i] for i in sorted(indices)]
        print(f'  [VALIDATE] Using {len(fm_items)} formula groups')
    else:
        # Full mode: cap groups to avoid excessive runtime
        n_total = len(fm_items)
        max_g = args.max_groups if args.max_groups > 0 else min(n_total, 2000)
        if max_g < n_total:
            indices = rng.choice(n_total, max_g, replace=False)
            fm_items = [fm_items[i] for i in sorted(indices)]
        print(f'  [FULL] Using {len(fm_items)}/{n_total} formula groups '
              f'(max_pairs_per_group={args.max_pairs_per_group})')

    # Compute MCES on formula groups
    print('\n[2] Computing MCES on formula groups...')
    all_mces = []

    for fm, iks in tqdm(fm_items, desc='Formula groups'):
        # Validate SMILES
        mols = {}
        fps = {}
        for ik in iks:
            smi = ik_to_smi.get(ik, '')
            if not smi: continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            mols[ik] = mol
            fp = compute_morgan_fp(smi)
            if fp is not None:
                fps[ik] = fp

        if len(mols) < 2: continue

        # Pre-filter: Morgan Tanimoto > 0.2
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

        # Cap candidates per group to avoid explosion
        if len(candidates) > args.max_pairs_per_group:
            idx_cand = rng.choice(len(candidates), args.max_pairs_per_group, replace=False)
            candidates = [candidates[i] for i in idx_cand]

        for ika, ikb, tan in candidates:
            smi_a = ik_to_smi[ika]
            smi_b = ik_to_smi[ikb]
            nb_a = mols[ika].GetNumBonds()
            nb_b = mols[ikb].GetNumBonds()

            mces_raw = compute_mces(smi_a, smi_b, nb_a, nb_b)
            if mces_raw is None: continue
            max_bonds = max(nb_a, nb_b)
            mces_norm = mces_raw / max_bonds if max_bonds > 0 else 0

            all_mces.append({
                'mces_raw': mces_raw,
                'mces_norm': round(mces_norm, 4),
                'tanimoto': round(tan, 4),
                'n_bonds_a': nb_a,
                'n_bonds_b': nb_b,
                'ik_a': ika,
                'ik_b': ikb,
                'smi_a': smi_a[:80],
                'smi_b': smi_b[:80],
                'fm': fm,
            })

    print(f'\n  MCES computed: {len(all_mces)}')

    if len(all_mces) == 0:
        print('ERROR: No MCES values computed!')
        return

    # Distribution analysis
    mces_raw = [x['mces_raw'] for x in all_mces]
    mces_norm = [x['mces_norm'] for x in all_mces]
    bonds = [max(x['n_bonds_a'], x['n_bonds_b']) for x in all_mces]

    print(f'\n[3] MCES Distribution:')
    print(f'  Raw MCES:  min={min(mces_raw)}, max={max(mces_raw)}, '
          f'mean={np.mean(mces_raw):.1f}, median={np.median(mces_raw):.0f}')
    print(f'  Norm MCES: min={min(mces_norm):.3f}, max={max(mces_norm):.3f}, '
          f'mean={np.mean(mces_norm):.3f}, median={np.median(mces_norm):.3f}')

    print(f'\n  Raw MCES bins:')
    for lo, hi in [(0, 0), (1, 2), (3, 4), (5, 6), (7, 10), (11, 20), (21, 100)]:
        n = sum(1 for m in mces_raw if lo <= m <= hi)
        if n > 0:
            print(f'    [{lo:3d}, {hi:3d}]: {n:6d} ({n / len(mces_raw) * 100:.1f}%)')

    # Stratified by molecule size
    print(f'\n  Stratified by molecule size:')
    for bond_lo, bond_hi, label in [(5, 20, 'small (5-20)'), (21, 40, 'med (21-40)'), (41, 200, 'large (41-200)')]:
        idx_s = [i for i, b in enumerate(bonds) if bond_lo <= b <= bond_hi]
        if len(idx_s) < 10: continue
        nr = [mces_raw[i] for i in idx_s]
        print(f'    {label}: n={len(idx_s)}')
        for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 10)]:
            n = sum(1 for m in nr if lo <= m <= hi)
            if n > 0:
                nm = [mces_norm[i] for i, j in enumerate(idx_s) if lo <= mces_raw[j] <= hi]
                print(f'      raw [{lo},{hi}]: {n:5d} ({n / len(idx_s) * 100:.0f}%)  norm_mean={np.mean(nm):.3f}')

    if args.validate:
        print(f'\n[VALIDATE] Review distribution above.')
        print(f'  Approved thresholds: pos=MCES [0,2], neg_hard=MCES [6,10]')
        print(f'  MCES [3,5] reserved for T2 (analogs/homologs)')
        print(f'  Then run without --validate to build full pairs.')
        return

    # === BUILD PAIRS ===
    print(f'\n[4] Building T1 pairs...')

    # Positive: raw MCES [0, 2] — near-isomers
    pos_pairs = [x for x in all_mces if 0 <= x['mces_raw'] <= 2]
    # Negative hard: raw MCES [6, 10] — clearly different within same formula
    neg_hard = [x for x in all_mces if 6 <= x['mces_raw'] <= 10]

    print(f'  Positive (MCES 0-2):      {len(pos_pairs)}')
    print(f'  Neg hard (MCES 6-10):     {len(neg_hard)}')
    print(f'  Excluded (MCES 3-5):      {sum(1 for x in all_mces if 3 <= x["mces_raw"] <= 5)} '
          f'(reserved for T2)')

    # Stratified norm MCES for positive pairs
    print(f'\n  Positive pairs by molecule size:')
    for bond_lo, bond_hi, label in [(5, 20, 'small (5-20)'), (21, 40, 'med (21-40)'), (41, 200, 'large (41-200)')]:
        sub = [p for p in pos_pairs if bond_lo <= max(p['n_bonds_a'], p['n_bonds_b']) <= bond_hi]
        if len(sub) > 0:
            norms = [p['mces_norm'] for p in sub]
            print(f'    {label}: n={len(sub)}  norm_mean={np.mean(norms):.3f}  '
                  f'norm_min={np.min(norms):.3f}  norm_max={np.max(norms):.3f}')

    # Add easy negatives: reuse T3 (different formula)
    t3_path = 'tasks/T3_unrelated/test_cases/pairs.json'
    neg_easy = []
    if os.path.exists(t3_path):
        with open(t3_path) as f:
            t3_data = json.load(f)
        t3_pairs = t3_data.get('negative', [])
        n_easy = min(len(neg_hard), len(t3_pairs))  # balance with hard negs
        neg_easy = t3_pairs[:n_easy]
        print(f'\n  Neg easy (from T3):       {len(neg_easy)} (reused)')
    else:
        print(f'\n  Neg easy: 0 (T3 not found at {t3_path})')

    # Tanimoto sample
    tan_pos = [p['tanimoto'] for p in pos_pairs[:500] if p['tanimoto'] >= 0]
    tan_neg = [p['tanimoto'] for p in neg_hard[:500] if p['tanimoto'] >= 0]

    if tan_pos:
        print(f'\n  Pos Tanimoto: mean={np.mean(tan_pos):.3f} std={np.std(tan_pos):.3f}')
    if tan_neg:
        print(f'  Neg Tanimoto: mean={np.mean(tan_neg):.3f} std={np.std(tan_neg):.3f}')

    # Save
    output = {
        'positive': pos_pairs,
        'negative_hard': neg_hard,
        'negative_easy': neg_easy,
    }
    stats = {
        'total_mces_computed': len(all_mces),
        'positive_pairs': len(pos_pairs),
        'negative_hard': len(neg_hard),
        'negative_easy': len(neg_easy),
        'excluded_mces_3_5': sum(1 for x in all_mces if 3 <= x['mces_raw'] <= 5),
        'mces_raw_stats': {
            'min': int(min(mces_raw)), 'max': int(max(mces_raw)),
            'mean': float(np.mean(mces_raw)), 'median': float(np.median(mces_raw)),
        },
        'positive_tanimoto': {
            'mean': float(np.mean(tan_pos)) if tan_pos else None,
            'std': float(np.std(tan_pos)) if tan_pos else None,
        },
        'negative_tanimoto': {
            'mean': float(np.mean(tan_neg)) if tan_neg else None,
            'std': float(np.std(tan_neg)) if tan_neg else None,
        },
    }

    for fn, data in [('pairs.json', output), ('stats.json', stats)]:
        path = f'{OUT_DIR}/{fn}'
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'  {fn}: {os.path.getsize(path) / 1e6:.1f}MB')

    print(f'\n=== T1 DONE ===')
    print(f'  Pos: {len(pos_pairs)}  Neg: {len(neg_hard)} hard + {len(neg_easy)} easy')


if __name__ == '__main__':
    main()
