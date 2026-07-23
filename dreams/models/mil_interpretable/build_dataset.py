"""
build_dataset.py — 测例库构造

将所有谱图源（MassBank+MoNA+MassSpecGym）的测例按类型分类保存：

data_library/
├── sources/              数据源统计
├── pairs/
│   ├── same_molecule/    同分子对 (Tanimoto ≈ 1.0)
│   ├── isomers/          同分异构体 (Tanimoto 0.3-0.9)
│   ├── random_different/ 随机不同分子 (Tanimoto < 0.2)
│   └── all_balanced/     混合平衡集
├── match_vectors/        预计算的规则匹配向量
├── statistics/           统计报告
└── versions/             各版本数据快照

用法: python -m dreams.models.mil_interpretable.build_dataset
"""

import torch, numpy as np, json, pickle
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm

from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto
from rdkit import Chem


def main():
    LIB = Path('data_library')
    LIB.mkdir(exist_ok=True)
    for d in ['sources','pairs/same_molecule','pairs/isomers','pairs/random_different',
              'pairs/all_balanced','match_vectors','statistics','versions/A0','versions/A1',
              'versions/A2','versions/A3']:
        (LIB/d).mkdir(exist_ok=True)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')
    import dreams.utils.dformats as dformats; import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    sp = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    # ===== 1. Parse ALL sources =====
    print('\n' + '='*60)
    print('  1. Parsing ALL data sources')
    print('='*60)
    spectra = []

    # MassBank NIST
    s = parse_msp('data/MassBank_NIST.msp', max_spectra=50000)
    print(f'  MassBank NIST: {len(s)}')
    for x in s: x['_source'] = 'massbank_nist'
    spectra.extend(s)

    # MoNA Pos
    s = parse_msp('data/MoNA-export-LC-MS-MS_Spectra.msp', max_spectra=50000)
    print(f'  MoNA Positive: {len(s)}')
    for x in s: x['_source'] = 'mona_pos'
    spectra.extend(s)

    # MoNA Neg
    s = parse_msp('data/MoNA-export-LC-MS-MS_Negative_Mode.msp', max_spectra=50000)
    print(f'  MoNA Negative: {len(s)}')
    for x in s: x['_source'] = 'mona_neg'
    spectra.extend(s)

    # MassSpecGym
    msdata = du.MSData.load('data/MassSpecGym_MurckoHist_split.hdf5')
    n_msg = min(50000, len(msdata))
    print(f'  MassSpecGym: reading {n_msg}...')
    for i in tqdm(range(n_msg), desc='  MassSpecGym'):
        try:
            smi = msdata.get_values('smiles', i)
            if isinstance(smi, bytes): smi = smi.decode('utf-8')
            ik = msdata.get_values('INCHIKEY', i)
            if isinstance(ik, bytes): ik = ik.decode('utf-8')
            fm = msdata.get_values('FORMULA', i)
            if isinstance(fm, bytes): fm = fm.decode('utf-8')
            pm = msdata.get_values('precursor_mz', i)
            spec_raw = torch.as_tensor(msdata.get_spectra(i), dtype=torch.float32)
            peaks = [(float(spec_raw[0, j]), float(spec_raw[1, j]))
                     for j in range(spec_raw.shape[1]) if spec_raw[0, j] > 0]
            spectra.append({'SMILES': str(smi).strip(), 'InChIKey': str(ik).strip(),
                           'PrecursorMZ': float(pm) if pm else 0,
                           '_formula': str(fm).strip(), 'peaks': peaks, '_source': 'massspecgym'})
        except Exception:
            pass
    print(f'  Total raw: {len(spectra)}')

    # ===== 2. Filter =====
    print('\n' + '='*60)
    print('  2. Filtering valid spectra')
    print('='*60)
    valid = []
    for s in spectra:
        smi = s.get('SMILES', '').strip()
        ik = s.get('InChIKey', '').strip()
        if not smi or not ik or len(smi) < 2:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if '_formula' not in s or not s['_formula']:
            s['_formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
        valid.append(s)
    print(f'  Valid: {len(valid)}')

    # Source distribution
    src_counts = Counter(s['_source'] for s in valid)
    print('  By source:')
    for src, n in src_counts.most_common():
        print(f'    {src}: {n}')
    with open(LIB/'sources'/'source_distribution.json', 'w') as f:
        json.dump({'total_raw': len(spectra), 'total_valid': len(valid),
                   'by_source': dict(src_counts)}, f, indent=2)

    # ===== 3. Match vectors =====
    print('\n' + '='*60)
    print('  3. Computing match vectors')
    print('='*60)
    mvs = {}
    for i, s in enumerate(tqdm(valid, desc='  Vectors')):
        vec = spectrum_to_match_vec(s, engine, sp)
        if vec is not None:
            mvs[i] = vec
    vidx = [i for i in range(len(valid)) if i in mvs]
    print(f'  {len(vidx)} spectra with rule vectors ({len(vidx)/len(valid)*100:.1f}%)')

    # Save match vectors
    mv_data = {'n_spectra': len(vidx), 'n_rules': len(engine.rules),
               'vectors': {str(i): mvs[i].tolist() for i in list(vidx)[:100]}}  # first 100 for reference
    with open(LIB/'match_vectors'/'match_vectors_sample.json', 'w') as f:
        json.dump(mv_data, f, indent=2)

    # ===== 4. Build pairs by type =====
    print('\n' + '='*60)
    print('  4. Building pairs by type')
    print('='*60)
    rng = np.random.RandomState(42)

    # Group
    ik2 = defaultdict(list); fm2 = defaultdict(list)
    for i in vidx:
        ik2[valid[i]['InChIKey']].append(i)
        fm = valid[i].get('_formula', '')
        if fm: fm2[fm].append(i)
    multi_ik = {k: v for k, v in ik2.items() if len(v) >= 2}
    multi_fm = {k: v for k, v in fm2.items() if len(v) >= 2}
    print(f'  Multi-IK molecules: {len(multi_ik)}')
    print(f'  Multi-FM groups: {len(multi_fm)}')

    all_report = {}

    # --- Same molecule ---
    print('\n  --- Same molecule pairs ---')
    same_pairs, same_labels = [], []
    for ik in sorted(multi_ik.keys(), key=lambda x: -len(multi_ik[x])):
        idxs = multi_ik[ik]
        for _ in range(min(5, len(idxs) * (len(idxs) - 1) // 2)):
            a, b = rng.choice(idxs, 2, replace=False)
            same_pairs.append((a, b)); same_labels.append(1.0)
        if len(same_pairs) >= 900: break
    print(f'  Built: {len(same_pairs)} pairs')
    with open(LIB/'pairs'/'same_molecule'/'pairs.json', 'w') as f:
        json.dump({'pairs': same_pairs, 'labels': same_labels, 'n': len(same_pairs),
                   'description': 'Same InChIKey, different acquisition conditions'}, f, indent=2)
    all_report['same_molecule'] = {'n': len(same_pairs), 'tanimoto': '≈1.0'}

    # --- Isomers ---
    print('\n  --- Isomer pairs ---')
    iso_pairs, iso_labels, iso_fm = [], [], Counter()
    for fm in sorted(multi_fm.keys(), key=lambda x: -len(multi_fm[x])):
        idxs = multi_fm[fm]
        if len(idxs) < 2: continue
        seen = set()
        for _ in range(min(80, len(idxs) * 5)):
            a, b = rng.choice(idxs, 2, replace=False)
            if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
            pk = (min(a, b), max(a, b))
            if pk in seen: continue; seen.add(pk)
            tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
            if 0.3 <= tan <= 0.9:
                iso_pairs.append((a, b)); iso_labels.append(tan); iso_fm[fm] += 1
            if len(iso_pairs) >= 1800: break
        if len(iso_pairs) >= 1800: break
    iso_tans = np.array(iso_labels)
    print(f'  Built: {len(iso_pairs)} pairs from {len(iso_fm)} unique formulas')
    print(f'  Tanimoto: mean={iso_tans.mean():.4f} std={iso_tans.std():.4f} '
          f'[0.3-0.5):{(iso_tans<0.5).sum()} [0.5-0.7):{((iso_tans>=0.5)&(iso_tans<0.7)).sum()} '
          f'[0.7-0.9):{(iso_tans>=0.7).sum()}')
    with open(LIB/'pairs'/'isomers'/'pairs.json', 'w') as f:
        json.dump({'pairs': iso_pairs, 'labels': iso_labels, 'n': len(iso_pairs),
                   'n_formulas': len(iso_fm), 'tanimoto_mean': float(iso_tans.mean()),
                   'tanimoto_std': float(iso_tans.std()),
                   'top_formulas': iso_fm.most_common(10),
                   'description': 'Same formula, different InChIKey, Tanimoto 0.3-0.9'}, f, indent=2)
    all_report['isomers'] = {'n': len(iso_pairs), 'n_formulas': len(iso_fm),
                              'tanimoto_mean': float(iso_tans.mean()),
                              'tanimoto_std': float(iso_tans.std())}

    # --- Random different ---
    print('\n  --- Random different pairs ---')
    rnd_pairs, rnd_labels = [], []
    for _ in range(20000):
        a, b = rng.choice(vidx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        pm_a = float(valid[a].get('PrecursorMZ', 0) or 0)
        pm_b = float(valid[b].get('PrecursorMZ', 0) or 0)
        if abs(pm_a - pm_b) <= 1.0: continue
        tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
        if 0 <= tan < 0.2:
            rnd_pairs.append((a, b)); rnd_labels.append(tan)
        if len(rnd_pairs) >= 1300: break
    rnd_tans = np.array(rnd_labels)
    print(f'  Built: {len(rnd_pairs)} pairs, Tanimoto mean={rnd_tans.mean():.4f}')
    with open(LIB/'pairs'/'random_different'/'pairs.json', 'w') as f:
        json.dump({'pairs': rnd_pairs, 'labels': rnd_labels, 'n': len(rnd_pairs),
                   'tanimoto_mean': float(rnd_tans.mean()),
                   'description': 'Random different molecules, mass diff > 1Da, Tanimoto < 0.2'}, f, indent=2)
    all_report['random_different'] = {'n': len(rnd_pairs), 'tanimoto_mean': float(rnd_tans.mean())}

    # --- All balanced ---
    all_pairs = same_pairs + iso_pairs + rnd_pairs
    all_labels = same_labels + iso_labels + rnd_labels
    print(f'\n  --- All balanced ---')
    print(f'  Total: {len(all_pairs)} pairs ({len(same_pairs)}+{len(iso_pairs)}+{len(rnd_pairs)})')
    all_tans = np.array(all_labels)
    print(f'  Tanimoto: mean={all_tans.mean():.4f} std={all_tans.std():.4f} median={np.median(all_tans):.4f}')
    # Bag-size check
    bsz = [((mvs[a] * mvs[b]) > 0).sum().item() for a, b in all_pairs]
    r_bag, _ = pearsonr(bsz, all_labels)
    print(f'  Bag-Tanimoto r = {r_bag:.4f}')
    # Tanimoto histogram
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    hist = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((all_tans >= lo) & (all_tans < hi)).sum())
        hist[f'[{lo:.1f},{hi:.1f})'] = n
    with open(LIB/'pairs'/'all_balanced'/'pairs.json', 'w') as f:
        json.dump({'pairs': all_pairs, 'labels': all_labels, 'n': len(all_pairs),
                   'composition': {'same': len(same_pairs), 'isomer': len(iso_pairs),
                                   'random': len(rnd_pairs)},
                   'tanimoto_mean': float(all_tans.mean()),
                   'tanimoto_std': float(all_tans.std()),
                   'tanimoto_median': float(np.median(all_tans)),
                   'tanimoto_histogram': hist,
                   'bag_tanimoto_r': float(r_bag),
                   'description': 'Balanced: 900 same + 1800 isomer + 1300 random, Tanimoto 0-1'}, f, indent=2)
    all_report['all_balanced'] = {'n': len(all_pairs), 'tanimoto_mean': float(all_tans.mean()),
                                   'tanimoto_std': float(all_tans.std()), 'bag_tanimoto_r': float(r_bag),
                                   'histogram': hist}

    # ===== 5. Statistics =====
    print('\n' + '='*60)
    print('  5. Final statistics')
    print('='*60)
    stats = {
        'created': datetime.now().isoformat(),
        'engine_rules': len(engine.rules),
        'sources': dict(src_counts),
        'total_raw_spectra': len(spectra),
        'total_valid_spectra': len(valid),
        'spectra_with_vectors': len(vidx),
        'multi_ik_molecules': len(multi_ik),
        'multi_fm_groups': len(multi_fm),
        'pairs': all_report,
        'weighted_jaccard_levels': {'L0': 1, 'L1': 2, 'L2': 4},
    }
    with open(LIB/'statistics'/'data_report.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  Saved to {LIB}/statistics/data_report.json')
    print(f'\n  Data library built successfully.')
    print(f'  Total pairs: {len(all_pairs)} ({len(same_pairs)} same + {len(iso_pairs)} isomer + {len(rnd_pairs)} random)')


if __name__ == '__main__':
    main()
