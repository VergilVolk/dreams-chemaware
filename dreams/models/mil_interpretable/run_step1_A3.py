"""
run_step1_A3.py — MIL Step 1 A3: 修正数据构造 [仅数据验证, 不训练]

数据源: MassBank_NIST.msp + MoNA Pos + MoNA Neg (三MSP文件)
构造: 30% 正样本(同InChIKey) + 30% 硬负样本(同分异构体) + 40% 易负样本(随机)
FORMULA: 从SMILES通过RDKit计算
验证: Tanimoto分布 + bag大小共线性 + 分子级隔离

用法:
  python -m dreams.models.mil_interpretable.run_step1_A3
"""

import torch, numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm
import json

torch.manual_seed(42); np.random.seed(42)

from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto

from rdkit import Chem
from rdkit.Chem import AllChem


def main():
    out_dir = Path('outputs') / f'mil_A3_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    sep = '=' * 60
    print(sep)
    print(f'  A3 DATA CONSTRUCTION ONLY (no training)')
    print(sep)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')

    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    MSP_FILES = ['data/MassBank_NIST.msp',
                 'data/MoNA-export-LC-MS-MS_Spectra.msp',
                 'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']

    # ===== 1. Parse =====
    print('\n[1] Parsing 3 MSP files...')
    spectra = []
    for fp in MSP_FILES:
        name = fp.replace('\\', '/').split('/')[-1]
        s = parse_msp(fp, max_spectra=20000)
        print(f'   {name}: {len(s)} spectra')
        spectra.extend(s)
    print(f'   Total raw: {len(spectra)}')

    # ===== 2. Filter + compute FORMULA from SMILES =====
    print('\n[2] Filtering + computing FORMULA from SMILES...')
    valid = []
    for s in spectra:
        smi = s.get('SMILES', '').strip()
        ik = s.get('InChIKey', '').strip()
        if not smi or not ik or len(smi) < 2:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fm = Chem.rdMolDescriptors.CalcMolFormula(mol)
        s['_formula'] = fm
        valid.append(s)
    print(f'   Valid (SMILES+InChIKey+Formula): {len(valid)}')

    # ===== 3. Match vectors =====
    print('\n[3] Computing rule match vectors...')
    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None:
            match_vecs[i] = vec
    vidx = [i for i in range(len(valid)) if i in match_vecs]
    print(f'   {len(vidx)} spectra with rule vectors')

    # ===== 4. Group =====
    print('\n[4] Grouping by InChIKey and Formula...')
    ik_to_idx = defaultdict(list)
    fm_to_idx = defaultdict(list)
    for i in vidx:
        ik_to_idx[valid[i]['InChIKey']].append(i)
        fm = valid[i].get('_formula', '')
        if fm:
            fm_to_idx[fm].append(i)
    multi_ik = {k: v for k, v in ik_to_idx.items() if len(v) >= 2}
    multi_fm = {k: v for k, v in fm_to_idx.items() if len(v) >= 2}
    print(f'   Multi-spectrum InChIKeys: {len(multi_ik)}')
    print(f'   Multi-spectrum Formulas:  {len(multi_fm)}')

    # ===== 5. Build pairs =====
    print('\n[5] Building balanced pairs...')
    rng = np.random.RandomState(42)
    N_TARGET = 3000
    pairs, labels, pair_types = [], [], []

    # --- Positive: same InChIKey (30%, 900) ---
    n_pos = N_TARGET * 30 // 100
    ik_list = list(multi_ik.keys()); rng.shuffle(ik_list)
    for ik in ik_list:
        idxs = multi_ik[ik]
        a, b = rng.choice(idxs, 2, replace=False)
        pairs.append((a, b)); labels.append(1.0); pair_types.append('pos')
        if len(pairs) >= n_pos: break
    print(f'   Positive (same mol, T~1.0): {len(pairs)}')

    # --- Hard negative: same Formula, different InChIKey (30%, 900) ---
    n_iso = N_TARGET * 30 // 100
    fm_list = list(multi_fm.keys()); rng.shuffle(fm_list)
    iso_found = 0
    for fm in fm_list:
        idxs = multi_fm[fm]
        if len(idxs) < 2: continue
        seen = set()
        for _ in range(min(50, len(idxs) * 3)):
            a, b = rng.choice(idxs, 2, replace=False)
            if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
            pk = (min(a, b), max(a, b))
            if pk in seen: continue
            seen.add(pk)
            tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
            if 0.3 <= tan <= 0.9:
                pairs.append((a, b)); labels.append(tan); pair_types.append('hard_neg')
                iso_found += 1
            if iso_found >= n_iso: break
        if iso_found >= n_iso: break
    print(f'   Hard negative (isomers, T=0.3~0.9): {iso_found}, mean T={np.mean(labels[-iso_found:]):.4f}')

    # --- Easy negative: random, mass diff > 1Da, Tanimoto < 0.2 (40%, 1200) ---
    n_easy = N_TARGET - len(pairs)
    for _ in range(n_easy * 5):
        a, b = rng.choice(vidx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        pm_a = float(valid[a].get('PrecursorMZ', 0) or 0)
        pm_b = float(valid[b].get('PrecursorMZ', 0) or 0)
        if abs(pm_a - pm_b) <= 1.0: continue
        tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
        if 0 <= tan < 0.2:
            pairs.append((a, b)); labels.append(tan); pair_types.append('easy_neg')
        if len(pairs) >= N_TARGET: break
    easy_n = len(pairs) - n_pos - iso_found
    print(f'   Easy negative (random, T<0.2): {easy_n}, mean T={np.mean(labels[n_pos+iso_found:]):.4f}')

    labels = np.array(labels, dtype=np.float32)
    pair_types = np.array(pair_types)

    # ===== 6. DATA VALIDATION =====
    print(f'\n{sep}')
    print(f'  DATA VALIDATION')
    print(f'{sep}')

    # V1: Tanimoto distribution
    print(f'\n  [V1] Tanimoto distribution:')
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((labels >= lo) & (labels < hi)).sum()
        bar = '#' * max(1, n * 60 // len(labels))
        print(f'    [{lo:.1f},{hi:.1f}): {n:5d} ({n/len(labels)*100:5.1f}%) {bar}')
    print(f'    mean={labels.mean():.4f}  median={np.median(labels):.4f}  std={labels.std():.4f}')

    # V1 detail: by pair type
    print(f'\n  [V1] Tanimoto by pair type:')
    for pt in ['pos', 'hard_neg', 'easy_neg']:
        mask = pair_types == pt
        if mask.sum() > 0:
            subset = labels[mask]
            print(f'    {pt:12s}: n={mask.sum():4d}  mean={subset.mean():.4f}  '
                  f'median={np.median(subset):.4f}  range=[{subset.min():.4f},{subset.max():.4f}]')

    # V2: Bag size collinearity
    print(f'\n  [V2] Bag-size vs Tanimoto collinearity:')
    bag_sizes = []
    for a, b in pairs:
        va, vb = match_vecs[a], match_vecs[b]
        bag_sizes.append(((va * vb) > 0).sum().item())
    r_bag, p_bag = pearsonr(bag_sizes, labels)
    print(f'    Pearson r(bag_size, Tanimoto) = {r_bag:.4f} (p={p_bag:.2e})')
    # Also check per-type
    for pt in ['pos', 'hard_neg', 'easy_neg']:
        mask = pair_types == pt
        if mask.sum() > 10:
            bs_sub = np.array(bag_sizes)[mask]
            tan_sub = labels[mask]
            r_pt, _ = pearsonr(bs_sub, tan_sub)
            print(f'    {pt:12s}: r(bag_size, T) = {r_pt:.4f} (n={mask.sum()})')

    if r_bag > 0.4:
        print(f'    WARNING: bag size dominates (r={r_bag:.3f} > 0.4)')
    else:
        print(f'    PASS: bag size effect resolved (r={r_bag:.3f} < 0.4)')

    # V3: Molecule-level isolation (simulate 1 fold to check)
    print(f'\n  [V3] Molecule-level isolation check:')
    pair_mols = []
    for a, b in pairs:
        ms = set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols))
    rng.shuffle(ams)
    # Check first fold
    mpf = len(ams) // 10
    vm = set(ams[0:mpf])
    tm = set(ams[mpf:])
    tr = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
    va = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
    tr_m = set().union(*[pair_mols[p] for p in tr]) if tr else set()
    va_m = set().union(*[pair_mols[p] for p in va]) if va else set()
    overlap = tr_m & va_m
    if len(overlap) == 0:
        print(f'    PASS: no molecule overlap between train/val')
    else:
        print(f'    FAIL: {len(overlap)} overlapping molecules!')

    # ===== 7. Save data =====
    print(f'\n[7] Saving data...')
    data = {
        'pairs': pairs, 'labels': labels.tolist(), 'pair_types': pair_types.tolist(),
        'bag_sizes': [int(b) for b in bag_sizes],
        'n_valid': len(valid), 'n_spectra': len(spectra),
        'multi_ik': len(multi_ik), 'multi_fm': len(multi_fm),
        'r_bag_tan': float(r_bag),
    }
    with open(out_dir / 'data_report.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f'\n{sep}')
    print(f'  DATA REPORT')
    print(f'{sep}')
    print(f'  Raw spectra:          {len(spectra)}')
    print(f'  Valid (SMILES+IK+FM): {len(valid)}')
    print(f'  With rule vectors:    {len(vidx)}')
    print(f'  Multi-IK molecules:   {len(multi_ik)}')
    print(f'  Multi-FM groups:      {len(multi_fm)}')
    print(f'  Total pairs:          {len(pairs)}')
    print(f'    Positive:           {n_pos}')
    print(f'    Hard neg (isomers): {iso_found}')
    print(f'    Easy neg (random):  {easy_n}')
    print(f'  Tanimoto: mean={labels.mean():.4f} median={np.median(labels):.4f} std={labels.std():.4f}')
    print(f'  Bag-Tanimoto r:      {r_bag:.4f}')
    print(f'  Output: {out_dir}')
    print(f'  >>> Data ready. NO training run. Review report first.')


if __name__ == '__main__':
    main()
