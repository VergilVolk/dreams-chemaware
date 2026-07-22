"""
build_balanced_data.py — 全谱系 Tanimoto 训练数据构造

策略（替代旧的"质量差≤0.05Da"困难样本逻辑）：
  1. 同分子对：同一 InChIKey 的不同谱图 → Tanimoto ≈ 1.0
  2. 同分异构体对：相同分子式、不同 InChIKey → Tanimoto 0.3-0.9
  3. 随机不同分子对 → Tanimoto 0.0-0.3
  （同系物/同类骨架对 → P2，暂不实现）

输出：pickle 文件，包含 pairs、labels (Tanimoto)、match_vecs 缓存

用法：
  python -m dreams.models.mil_interpretable.build_balanced_data \
      --dataset_path ./data/MassSpecGym_MurckoHist_split.hdf5 \
      --n_pairs 3000 --output_dir ./mil_data
"""

import torch
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str,
                   default='data/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--n_pairs', type=int, default=3000)
    p.add_argument('--output_dir', type=str, default='./mil_data')
    return p.parse_args()


def compute_tanimoto(smi_a, smi_b):
    """计算两个 SMILES 的 Tanimoto 相似度。"""
    try:
        ma = Chem.MolFromSmiles(str(smi_a).strip())
        mb = Chem.MolFromSmiles(str(smi_b).strip())
        if ma is None or mb is None:
            return -1.0
        fpa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fpb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fpa, fpb)
    except Exception:
        return -1.0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    print('=' * 60)
    print('Building BALANCED training data (full Tanimoto spectrum)')
    print('=' * 60)

    # ---- Load ----
    print('\n[1] Loading data...')
    msdata = du.MSData.load(args.dataset_path)
    engine = ChemicalRuleEngine(tolerance=0.02)
    n_total = len(msdata)
    print(f'   {n_total} spectra, {len(engine.rules)} rules')

    # ---- Read metadata ----
    print('\n[2] Reading metadata...')
    n_read = min(30000, n_total)  # Read 30K to have enough diversity
    inchikeys, smiles_list, formulas = [], [], []
    prec_mzs = []

    for i in tqdm(range(n_read), desc='   Metadata'):
        try:
            ik = msdata.get_values('INCHIKEY', i)
            if isinstance(ik, bytes): ik = ik.decode('utf-8')
            inchikeys.append(str(ik).strip())

            smi = msdata.get_values('smiles', i)
            if isinstance(smi, bytes): smi = smi.decode('utf-8')
            smiles_list.append(str(smi).strip())

            fm = msdata.get_values('FORMULA', i)
            if isinstance(fm, bytes): fm = fm.decode('utf-8')
            formulas.append(str(fm).strip())

            pm = msdata.get_values('precursor_mz', i)
            prec_mzs.append(float(pm) if pm else 0.0)
        except Exception:
            inchikeys.append('')
            smiles_list.append('')
            formulas.append('')
            prec_mzs.append(0.0)

    print(f'   Read {len(inchikeys)} records')

    # ---- Group by InChIKey (same molecule) and formula (isomers) ----
    print('\n[3] Grouping by InChIKey and formula...')
    ik_to_indices = defaultdict(list)
    formula_to_indices = defaultdict(list)
    for i, (ik, fm) in enumerate(zip(inchikeys, formulas)):
        if ik:
            ik_to_indices[ik].append(i)
        if fm:
            formula_to_indices[fm].append(i)

    multi_ik = {k: v for k, v in ik_to_indices.items() if len(v) >= 2}
    multi_formula = {k: v for k, v in formula_to_indices.items() if len(v) >= 2}
    print(f'   Multi-spectrum InChIKeys: {len(multi_ik)}')
    print(f'   Multi-spectrum formulas: {len(multi_formula)}')

    # ---- Pre-compute match vectors (for speed) ----
    print('\n[4] Computing match vectors...')
    spec_preproc = du.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(), n_highest_peaks=128)
    match_vecs_cache = {}

    for i in tqdm(range(n_read), desc='   Match vecs'):
        try:
            spec = torch.as_tensor(msdata.get_spectra(i), dtype=torch.float32)
            spec_pp = spec_preproc(spec.numpy(), high_form=False)
            spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)
            mz = spec_t[:, 0].unsqueeze(0)
            pad = mz[:, 0] == 0
            mz_diffs = torch.abs(mz.unsqueeze(-1) - mz.unsqueeze(-2))
            vec = engine.get_rule_match_vectors(
                mz_diffs, mz_values=mz,
                precursor_mz=mz[:, 0].unsqueeze(0),
                padding_mask=pad, categories=['NL', 'CF', 'ISO', 'HR'])
            match_vecs_cache[i] = vec.squeeze(0)
        except Exception:
            match_vecs_cache[i] = torch.zeros(len(engine.rules))

    # ---- Construct pairs ----
    print(f'\n[5] Constructing pairs (target: {args.n_pairs} total)...')
    pairs = []   # [(idx_A, idx_B), ...]
    labels = []  # [tanimoto, ...]

    # --- Type 1: Same molecule (25%) ---
    n_same = args.n_pairs // 4
    ik_list = list(multi_ik.keys())
    rng.shuffle(ik_list)
    print(f'   Type 1: Same molecule (target {n_same})...')
    for ik in ik_list:
        idxs = multi_ik[ik]
        if len(idxs) >= 2:
            a, b = rng.choice(idxs, 2, replace=False)
            tan = compute_tanimoto(smiles_list[a], smiles_list[b])
            if tan > 0.9:
                pairs.append((a, b))
                labels.append(tan)
        if len(pairs) >= n_same:
            break
    print(f'     Got {len(pairs)} same-molecule pairs')

    # --- Type 2: Isomers (P0 — same FORMULA, different INCHIKEY) ---
    n_isomer = args.n_pairs // 3  # 1/3 of total
    fm_list = list(multi_formula.keys())
    rng.shuffle(fm_list)
    print(f'   Type 2: Isomers (target {n_isomer})...')
    isomer_count = 0
    for fm in fm_list:
        idxs = multi_formula[fm]
        if len(idxs) < 2:
            continue
        # Build ALL distinct-InChIKey pairs within this formula group
        # (limit to avoid combinatorial explosion)
        seen_pairs = set()
        for _ in range(min(20, len(idxs) * 2)):
            a, b = rng.choice(idxs, 2, replace=False)
            ika, ikb = inchikeys[a], inchikeys[b]
            if ika == ikb or not ika or not ikb:
                continue
            pair_key = (min(a,b), max(a,b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            tan = compute_tanimoto(smiles_list[a], smiles_list[b])
            if tan < 0:
                continue
            pairs.append((a, b))
            labels.append(tan)
            isomer_count += 1
            if isomer_count >= n_isomer:
                break
        if isomer_count >= n_isomer:
            break
    print(f'     Got {isomer_count} isomer pairs')

    # --- Type 3: Random different molecules (50%) ---
    n_random = args.n_pairs - len(pairs)
    print(f'   Type 3: Random diff-mol (target {n_random})...')
    attempts = 0
    while len(pairs) < args.n_pairs and attempts < n_random * 5:
        a, b = rng.choice(n_read, 2, replace=False)
        ika, ikb = inchikeys[a], inchikeys[b]
        if ika and ikb and ika == ikb:
            attempts += 1
            continue  # Skip same molecule
        tan = compute_tanimoto(smiles_list[a], smiles_list[b])
        if tan < 0:
            attempts += 1
            continue
        pairs.append((a, b))
        labels.append(tan)
        attempts += 1
    print(f'     Got {len(pairs) - n_same - isomer_count} random pairs')

    labels = np.array(labels, dtype=np.float32)

    # ---- Statistics ----
    print(f'\n[6] Dataset statistics:')
    print(f'   Total pairs: {len(pairs)}')
    print(f'   Tanimoto: mean={labels.mean():.4f}  std={labels.std():.4f}  '
          f'median={np.median(labels):.4f}')
    for lo, hi, label in [(0, 0.2, '<0.2'), (0.2, 0.5, '0.2-0.5'),
                           (0.5, 0.8, '0.5-0.8'), (0.8, 1.01, '>0.8')]:
        n = ((labels >= lo) & (labels < hi)).sum()
        print(f'     Tanimoto {label}: {n} ({n/len(labels)*100:.1f}%)')

    # ---- Save ----
    data = {
        'pairs': pairs,
        'labels': labels,
        'match_vecs_cache': {k: v.cpu() for k, v in match_vecs_cache.items()},
        'inchikeys': inchikeys,
        'smiles_list': smiles_list,
        'formulas': formulas,
    }
    out_path = output_dir / 'mil_balanced_data.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nSaved: {out_path}')
    print('Done!')


if __name__ == '__main__':
    main()
