"""
T0 v3 — 同分子一致性 (Consistency)
修正: 使用 annotated01.mgf 统一数据源，消除路径依赖

构造:
  正样本: 同 InChIKey 不同谱图（跨库优先，每 IK ≥1 对）
  负样本: 难(前体质量差≤0.05Da) + 易(质量差>50Da，随机)

用法: python tasks/T0_consistency/code/build_test_cases.py
"""
import json, os, sys
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, '.')
from tasks.build_utils import load_indices, compute_tanimoto

rng = np.random.RandomState(42)
OUT_DIR = 'tasks/T0_consistency/test_cases'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print('[1] Loading indices...')
    idx = load_indices()
    ik_to_smi = idx['ik_to_smi']
    ik_to_fm = idx['ik_to_fm']
    ik_to_pm = idx['ik_to_pm']
    ik_counts = idx['ik_counts']

    # Multi-spectrum IKs only (≥2 spectra)
    multi_ik = {ik: c for ik, c in ik_counts.items() if c >= 2 and ik in ik_to_smi}
    print(f'  Multi-spectrum IKs: {len(multi_ik)} / {len(ik_counts)} total')

    # Spectrum count distribution
    spec_counts = list(multi_ik.values())
    print(f'  Counts: min={min(spec_counts)}, max={max(spec_counts)}, '
          f'mean={np.mean(spec_counts):.1f}, median={np.median(spec_counts):.0f}')

    # === POSITIVE PAIRS ===
    print('\n[2] Building POSITIVE pairs...')
    positive_pairs = []
    iks_covered = set()

    # Strategy: 1 pair per IK for all multi-IK molecules (uniform coverage)
    # Sort by spectrum count descending to prioritize high-confidence IKs
    sorted_iks = sorted(multi_ik.keys(), key=lambda ik: -multi_ik[ik])

    for ik in sorted_iks:
        count = multi_ik[ik]
        smi = ik_to_smi.get(ik, '')
        fm = ik_to_fm.get(ik, '')
        pm = ik_to_pm.get(ik, 0)

        # 1 pair guaranteed per IK
        positive_pairs.append({
            'ik': ik,
            'smiles': smi[:120],
            'formula': fm or '',
            'precursor_mz': pm,
            'n_spectra': count,
            'type': 'same_molecule',
        })
        iks_covered.add(ik)

        # For IKs with ≥10 spectra: add 1 extra pair
        if count >= 10:
            positive_pairs.append({
                'ik': ik,
                'smiles': smi[:120],
                'formula': fm or '',
                'precursor_mz': pm,
                'n_spectra': count,
                'type': 'same_molecule_extra',
            })

    n_pos = len(positive_pairs)
    print(f'  Positive pairs: {n_pos} (from {len(iks_covered)} IKs)')

    # === NEGATIVE PAIRS ===
    print('\n[3] Building NEGATIVE pairs...')

    # Hard negatives: same precursor mz (±0.05 Da), different IK
    valid_for_neg = [(ik, ik_to_pm.get(ik, 0)) for ik in ik_to_smi
                     if ik in ik_to_pm and 50 <= ik_to_pm[ik] <= 2000]
    valid_for_neg.sort(key=lambda x: x[1])

    neg_pairs = []
    seen_neg = set()

    # Hard: adjacent in sorted precursor mz list
    hard_count = 0
    for i in range(len(valid_for_neg) - 1):
        ik_a, pm_a = valid_for_neg[i]
        ik_b, pm_b = valid_for_neg[i + 1]
        if abs(pm_a - pm_b) <= 0.05 and ik_a != ik_b:
            pk = (ik_a, ik_b) if ik_a < ik_b else (ik_b, ik_a)
            if pk in seen_neg: continue
            seen_neg.add(pk)
            neg_pairs.append({
                'ik_a': ik_a, 'ik_b': ik_b,
                'fm_a': ik_to_fm.get(ik_a, ''), 'fm_b': ik_to_fm.get(ik_b, ''),
                'precursor_mz_a': pm_a, 'precursor_mz_b': pm_b,
                'mass_diff': round(abs(pm_a - pm_b), 6),
                'type': 'hard_mass_proximate',
            })
            hard_count += 1
        if hard_count >= 5000: break
    print(f'  Hard negatives: {hard_count}')

    # Easy: random pairs with mass diff > 50 Da, different IK
    easy_count = 0
    all_iks = list(ik_to_smi.keys())
    attempts = 0
    while easy_count < 5000 and attempts < 50000:
        ika, ikb = rng.choice(all_iks, 2, replace=False)
        attempts += 1
        if ika == ikb: continue
        pk = (ika, ikb) if ika < ikb else (ikb, ika)
        if pk in seen_neg: continue

        pm_a = ik_to_pm.get(ika, 0)
        pm_b = ik_to_pm.get(ikb, 0)
        if pm_a > 10 and pm_b > 10 and abs(pm_a - pm_b) <= 50: continue

        seen_neg.add(pk)
        neg_pairs.append({
            'ik_a': ika, 'ik_b': ikb,
            'fm_a': ik_to_fm.get(ika, ''), 'fm_b': ik_to_fm.get(ikb, ''),
            'precursor_mz_a': pm_a, 'precursor_mz_b': pm_b,
            'mass_diff': round(abs(pm_a - pm_b), 3) if pm_a > 10 and pm_b > 10 else -1,
            'type': 'easy_random',
        })
        easy_count += 1
    print(f'  Easy negatives: {easy_count}')

    # === TANIMOTO SAMPLE ===
    print('\n[4] Computing Tanimoto sample...')
    tan_pos = []
    for p in positive_pairs[:200]:
        smi = p.get('smiles', '')
        if smi:
            tan = compute_tanimoto(smi, smi)
            if tan >= 0: tan_pos.append(tan)

    tan_neg = []
    for p in neg_pairs[:200]:
        smi_a = ik_to_smi.get(p['ik_a'], '')
        smi_b = ik_to_smi.get(p['ik_b'], '')
        if smi_a and smi_b:
            tan = compute_tanimoto(smi_a, smi_b)
            if tan >= 0: tan_neg.append(tan)

    if tan_pos:
        print(f'  Pos Tanimoto: mean={np.mean(tan_pos):.4f} std={np.std(tan_pos):.4f}')
    if tan_neg:
        print(f'  Neg Tanimoto: mean={np.mean(tan_neg):.4f} std={np.std(tan_neg):.4f}')

    # === SAVE ===
    output = {
        'positive': positive_pairs,
        'negative': neg_pairs,
    }
    stats = {
        'total_iks': len(ik_to_smi),
        'multi_spectrum_iks': len(multi_ik),
        'positive_pairs': n_pos,
        'positive_iks_covered': len(iks_covered),
        'negative_pairs': len(neg_pairs),
        'negative_hard': hard_count,
        'negative_easy': easy_count,
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
        print(f'  {fn}: {os.path.getsize(path)/1e6:.1f}MB')

    print(f'\n=== T0 DONE ===')
    print(f'  Pos: {n_pos}  Neg: {len(neg_pairs)} ({hard_count} hard + {easy_count} easy)')
    print(f'  Files in {OUT_DIR}/')


if __name__ == '__main__':
    main()
