"""
train_mil_massbank.py — MassBank MSP → MIL 全流程

用法（在 Anaconda Prompt 里）：
  conda activate D:/dreams_env
  cd d:/DreaMS
  python -m dreams.models.mil_interpretable.train_mil_massbank
"""

import torch, torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
from pathlib import Path

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


def parse_msp(filepath, max_spectra=50000):
    """解析 MSP 文件，提取谱图元数据和峰列表。"""
    spectra = []
    current = {}
    peaks = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc='Parsing MSP', total=7000000):
            line = line.strip()
            if not line:
                if current and peaks:
                    current['peaks'] = peaks
                    spectra.append(current)
                    if len(spectra) >= max_spectra:
                        break
                current = {}
                peaks = []
                continue

            if ': ' in line:
                key, val = line.split(': ', 1)
                if key in ('Name', 'InChIKey', 'InChI', 'SMILES', 'Formula',
                           'PrecursorMZ', 'Ion_mode', 'Precursor_type',
                           'Collision_energy', 'Instrument_type'):
                    current[key] = val
                # MoNA format: SMILES embedded in Comments
                if key == 'Comments' and 'SMILES' not in current:
                    import re
                    m = re.search(r'SMILES="?([^"]+?)"?\s', val)
                    if m:
                        current['SMILES'] = m.group(1).strip()
            else:
                # Peak: "m/z intensity"
                parts = line.split()
                if len(parts) == 2:
                    try:
                        mz, intensity = float(parts[0]), float(parts[1])
                        if mz > 0 and intensity > 0:
                            peaks.append((mz, intensity))
                    except ValueError:
                        pass

    # Don't forget the last spectrum
    if current and peaks and len(spectra) < max_spectra:
        current['peaks'] = peaks
        spectra.append(current)

    return spectra


def spectrum_to_match_vec(spectrum, engine, spec_preproc):
    """将谱图转换为规则匹配向量。"""
    peaks = spectrum.get('peaks', [])
    if len(peaks) < 3:
        return None

    # 构造 (n_peaks, 2) 数组
    arr = np.array(peaks, dtype=np.float32)
    # 按 m/z 排序
    arr = arr[arr[:, 0].argsort()]
    # 取前 128 个峰
    if len(arr) > 128:
        arr = arr[:128]

    try:
        spec_pp = spec_preproc(arr.T, high_form=False)
        spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)
        mz = spec_t[:, 0].unsqueeze(0)
        pad = mz[:, 0] == 0
        mz_diffs = torch.abs(mz.unsqueeze(-1) - mz.unsqueeze(-2))
        vec = engine.get_rule_match_vectors(
            mz_diffs, mz_values=mz,
            precursor_mz=mz[:, 0].unsqueeze(0),
            padding_mask=pad, categories=['NL', 'CF', 'ISO', 'HR'])
        return vec.squeeze(0)
    except Exception:
        return None


def main():
    print('=' * 60)
    print('MassBank → MIL Full Pipeline')
    print('=' * 60)

    msp_path = 'data/MassBank_NIST.msp'
    engine = ChemicalRuleEngine(tolerance=0.02)

    # ---- Step 1: Parse MSP ----
    print('\n[1] Parsing MSP...')
    spectra = parse_msp(msp_path, max_spectra=50000)
    print(f'   Parsed {len(spectra)} spectra')

    # ---- Step 2: Filter valid spectra ----
    print('\n[2] Filtering spectra with valid SMILES...')
    valid = []
    for s in spectra:
        smi = s.get('SMILES', '').strip()
        ik = s.get('InChIKey', '').strip()
        fm = s.get('Formula', '').strip()
        if smi and ik and fm and len(smi) > 2:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid.append(s)
    print(f'   Valid spectra: {len(valid)}')

    # Group by InChIKey and Formula
    ik_to_idx = defaultdict(list)
    fm_to_idx = defaultdict(list)
    for i, s in enumerate(valid):
        ik = s['InChIKey']
        fm = s['Formula']
        if ik: ik_to_idx[ik].append(i)
        if fm: fm_to_idx[fm].append(i)

    multi_ik = {k: v for k, v in ik_to_idx.items() if len(v) >= 2}
    multi_fm = {k: v for k, v in fm_to_idx.items() if len(v) >= 2}
    print(f'   Multi-spectrum InChIKeys: {len(multi_ik)}')
    print(f'   Multi-spectrum Formulas: {len(multi_fm)}')

    # ---- Step 3: Compute match vectors ----
    print('\n[3] Computing rule match vectors...')
    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None:
            match_vecs[i] = vec

    n_with_vecs = len(match_vecs)
    print(f'   Spectra with rule vectors: {n_with_vecs}')

    # ---- Step 4: Build pairs ----
    print('\n[4] Building balanced pairs...')
    rng = np.random.RandomState(42)
    pairs = []  # (idx, idx)
    labels = []  # Tanimoto

    # Same molecule pairs
    ik_list = list(multi_ik.keys())
    rng.shuffle(ik_list)
    n_same = 0
    for ik in ik_list:
        idxs = [i for i in multi_ik[ik] if i in match_vecs]
        if len(idxs) >= 2:
            a, b = rng.choice(idxs, 2, replace=False)
            tan = DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(valid[a]['SMILES']), 2, 2048),
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(valid[b]['SMILES']), 2, 2048))
            pairs.append((a, b)); labels.append(tan); n_same += 1
        if n_same >= 800: break
    print(f'   Same molecule: {n_same} pairs')

    # Isomer pairs (same formula, different InChIKey)
    fm_list = list(multi_fm.keys())
    rng.shuffle(fm_list)
    n_isomer = 0
    for fm in fm_list:
        idxs = [i for i in multi_fm[fm] if i in match_vecs]
        if len(idxs) < 2: continue
        seen = set()
        for _ in range(min(30, len(idxs)*2)):
            a, b = rng.choice(idxs, 2, replace=False)
            if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
            pk = (min(a,b), max(a,b))
            if pk in seen: continue
            seen.add(pk)
            tan = DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(valid[a]['SMILES']), 2, 2048),
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(valid[b]['SMILES']), 2, 2048))
            if tan >= 0:
                pairs.append((a, b)); labels.append(tan); n_isomer += 1
            if n_isomer >= 800: break
        if n_isomer >= 800: break
    print(f'   Isomers: {n_isomer} pairs')

    # Random different molecules
    valid_idx = list(match_vecs.keys())
    n_random = 0
    target = 2000
    while n_random < target - n_same - n_isomer:
        a, b = rng.choice(valid_idx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        tan = DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(valid[a]['SMILES']), 2, 2048),
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(valid[b]['SMILES']), 2, 2048))
        if tan >= 0:
            pairs.append((a, b)); labels.append(tan); n_random += 1
    print(f'   Random: {n_random} pairs')

    labels = np.array(labels, dtype=np.float32)
    print(f'   Total: {len(pairs)} pairs')
    print(f'   Tanimoto: mean={labels.mean():.4f} std={labels.std():.4f}')

    # ---- Step 5: Build features ----
    print('\n[5] Building features...')
    X_agg, instances_list = [], []
    for a, b in tqdm(pairs, desc='Features'):
        va, vb = match_vecs[a], match_vecs[b]
        inter = (va * vb).sum().float()
        union = ((va + vb) > 0).float().sum()
        ov = (inter / union.clamp(min=1)).item()
        common = (va * vb) > 0
        nc = common.sum().item()
        nl = common[:214].sum().item()
        cf = common[214:214+102].sum().item()
        iso_hr = (common[214+102:214+102+8].sum() + common[-9:].sum()).item()
        X_agg.append([ov, float(nc), iso_hr/max(nc,1), nl/max(nc,1), cf/max(nc,1)])

        inst_feats = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category == 'HR': level = 2
                elif rule.category in ('NR','EE'): level = 0
                elif rule.category == 'ISO': level = 2
                inst_feats.append(build_instance_features({
                    'level':level, 'category':rule.category,
                    'match_type':rule.match_type, 'mass_diff_precision':0.5}))
        instances_list.append(torch.tensor(np.stack(inst_feats), dtype=torch.float32) if inst_feats else torch.zeros(0,12))

    X_agg = np.array(X_agg, dtype=np.float32)

    # ---- Step 6: 5-fold CV ----
    print('\n[6] 5-fold CV (molecule-level split)...')
    inchikey_list = [s.get('InChIKey','') for s in valid]
    pair_mols = []
    for a, b in pairs:
        m = set()
        if inchikey_list[a]: m.add(inchikey_list[a])
        if inchikey_list[b]: m.add(inchikey_list[b])
        pair_mols.append(m)

    all_mols = list(set().union(*pair_mols))
    rng.shuffle(all_mols)
    mpf = len(all_mols) // 5

    lr_rs, mil_rs = [], []
    for k in range(5):
        vs, ve = k*mpf, (k+1)*mpf if k < 4 else len(all_mols)
        vm = set(all_mols[vs:ve])
        tm = set(all_mols[:vs]) | set(all_mols[ve:])
        tr = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        assert not (set().union(*[pair_mols[p] for p in tr] if tr else []) &
                    set().union(*[pair_mols[p] for p in va] if va else [])), f'Fold {k} leak!'

        # LR-agg
        lr = Ridge(alpha=1.0); lr.fit(X_agg[tr], labels[tr])
        yp = lr.predict(X_agg[va]); lr_r, _ = pearsonr(labels[va], yp); lr_rs.append(max(lr_r,0))

        # MIL — 加强版参数
        model = RuleAttentionMIL(hidden_dim=128)  # 从 32→128
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_r, best_st, counter = 0, None, 0
        for ep in range(400):  # 从 200→400
            model.train(); loss_sum, n, batch_loss = 0, 0, 0.0
            for pi in tr:
                bag = instances_list[pi]
                if bag.shape[0]==0: continue
                pred, _ = model(bag)
                loss = F.mse_loss(pred, torch.tensor(labels[pi], dtype=torch.float32).unsqueeze(0))
                batch_loss += loss; loss_sum += loss.item(); n += 1
                if n % 4 == 0:  # batch_size=4 累积
                    batch_loss.backward(); opt.step(); opt.zero_grad(); batch_loss = 0.0
            if n % 4 != 0:  # 最后一批
                batch_loss.backward(); opt.step(); opt.zero_grad()
            if ep % 30 == 0:
                model.eval()
                with torch.no_grad():
                    pr, lb = [], []
                    for pi in va:
                        bag = instances_list[pi]
                        pr.append(model(bag)[0].item() if bag.shape[0]>0 else 0.0)
                        lb.append(labels[pi])
                r,_ = pearsonr(pr,lb); r=max(r,0)
                if r>best_r: best_r=r; best_st={k:v.clone() for k,v in model.state_dict().items()}; counter=0
                else: counter+=1
                if counter>=60: break  # 从 30→60，给更多时间
        if best_st: model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            pr, lb = [], []
            for pi in va:
                bag = instances_list[pi]
                pr.append(model(bag)[0].item() if bag.shape[0]>0 else 0.0)
                lb.append(labels[pi])
        mil_r, _ = pearsonr(pr,lb); mil_rs.append(max(mil_r,0))
        print(f'   Fold {k+1}: LR-agg r={lr_rs[-1]:.4f}  MIL r={mil_rs[-1]:.4f}')

    print(f'\n{"="*60}')
    print(f'MassBank RESULTS')
    print(f'{"="*60}')
    print(f'  LR-agg:        r = {np.mean(lr_rs):.4f} +/- {np.std(lr_rs):.4f}')
    print(f'  Attention MIL: r = {np.mean(mil_rs):.4f} +/- {np.std(mil_rs):.4f}')


if __name__ == '__main__':
    main()
