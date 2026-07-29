"""
T3 v3 — 不相关分子基线 (sanity check)
修正: 使用 annotated01.mgf 统一数据源

构造: 随机不同 InChIKey + 不同 FORMULA → 纯负样本
      如 IK 不在 ik_to_fm 中 → 显式跳过
      检查 IK 总数是否足够 → 不足则减少目标对数

用法: python tasks/T3_unrelated/code/build_test_cases.py
"""
import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')
from tasks.build_utils import load_indices

rng = np.random.RandomState(42)
OUT_DIR = 'tasks/T3_unrelated/test_cases'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print('[1] Loading indices...')
    idx = load_indices()
    ik_to_fm = idx['ik_to_fm']
    ik_to_smi = idx['ik_to_smi']

    # Only use IKs that have FORMULA
    iks_with_fm = [ik for ik in ik_to_fm if ik in ik_to_smi]
    print(f'  IKs with FORMULA + SMILES: {len(iks_with_fm)}')

    if len(iks_with_fm) < 100:
        print('ERROR: Not enough IKs with FORMULA!')
        return

    # Determine target: 10K pairs or 10% of max possible
    max_possible = len(iks_with_fm) * (len(iks_with_fm) - 1) // 2
    target = min(10000, max_possible // 10)
    print(f'  Target: {target} pairs (max possible: {max_possible})')

    # Build negative pairs: different IK, different FORMULA
    pairs = []
    seen = set()
    attempts = 0
    max_attempts = target * 50  # generous limit

    all_iks = iks_with_fm

    while len(pairs) < target and attempts < max_attempts:
        ika, ikb = rng.choice(all_iks, 2, replace=False)

        # Must have different formula
        fm_a = ik_to_fm.get(ika, '')
        fm_b = ik_to_fm.get(ikb, '')
        if not fm_a or not fm_b:
            attempts += 1
            continue
        if fm_a == fm_b:
            attempts += 1
            continue

        # Canonical ordering
        pk = (ika, ikb) if ika < ikb else (ikb, ika)
        if pk in seen:
            attempts += 1
            continue

        seen.add(pk)
        pairs.append({
            'ik_a': ika,
            'ik_b': ikb,
            'fm_a': fm_a,
            'fm_b': fm_b,
        })
        attempts += 1

    print(f'  Built: {len(pairs)} pairs (attempts={attempts})')

    # Formula diversity stats
    formulas = set()
    for p in pairs:
        formulas.add(p['fm_a'])
        formulas.add(p['fm_b'])
    print(f'  Unique formulas: {len(formulas)}')

    # Save
    output = {
        'negative': pairs,
        'n_pairs': len(pairs),
        'n_iks_with_fm': len(iks_with_fm),
        'n_attempts': attempts,
    }

    path = f'{OUT_DIR}/pairs.json'
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {path} ({os.path.getsize(path)/1e6:.1f}MB)')

    # Stats
    stats = {
        'total_pairs': len(pairs),
        'unique_inchikeys_used': len(set([p['ik_a'] for p in pairs] + [p['ik_b'] for p in pairs])),
        'unique_formulas': len(formulas),
    }
    with open(f'{OUT_DIR}/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'=== T3 DONE ===')
    print(f'  {len(pairs)} negative pairs, {stats["unique_inchikeys_used"]} IKs, {len(formulas)} formulas')


if __name__ == '__main__':
    main()
