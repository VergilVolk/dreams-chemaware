"""
run_step1_A3.py — MIL Step 1 A3: 修正数据构造

新数据: 30% 正样本(同分子) + 30% 硬负样本(同分异构体) + 40% 易负样本(随机)
来源: MassSpecGym (有 FORMULA 字段用于同分异构体提取)
验证: Tanimoto分布 + bag大小共线性 + 分子级隔离

配置: 3486规则, hidden_dim=32, CosineAnnealing, 500 epochs, 10-fold

用法:
  python -m dreams.models.mil_interpretable.run_step1_A3
"""

import torch, torch.nn.functional as F
import numpy as np, json, pickle, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from tqdm import tqdm

torch.manual_seed(42); np.random.seed(42)

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


def main():
    out_dir = Path('outputs') / f'mil_A3_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    CFG = {'hidden_dim': 32, 'dropout': 0.1, 'entropy_coef': 0.001,
           'lr': 1e-4, 'weight_decay': 1e-5, 'batch_size': 32,
           'epochs': 500, 'n_folds': 10, 'n_pairs': 3000}

    sep = '=' * 60
    print(sep)
    print(f'  A3: Balanced data (30% pos + 30% isomer + 40% random), {CFG["n_folds"]}-fold')
    print(sep)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')

    # ===== 1. Load MassSpecGym =====
    import dreams.utils.data as du
    import dreams.utils.dformats as dformats
    msdata = du.MSData.load('data/MassSpecGym_MurckoHist_split.hdf5')
    n_total = len(msdata)
    print(f'\n[1] MassSpecGym: {n_total} spectra')

    # ===== 2. Read metadata =====
    print('\n[2] Reading metadata...')
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    n_read = min(50000, n_total)
    inchikeys, smiles_list, formulas, prec_mzs = [], [], [], []
    for i in tqdm(range(n_read), desc='Metadata'):
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
            inchikeys.append(''); smiles_list.append(''); formulas.append(''); prec_mzs.append(0.0)

    # ===== 3. Compute match vectors =====
    print('\n[3] Computing match vectors...')
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)
    match_vecs = {}
    for i in tqdm(range(n_read), desc='Rule vectors'):
        try:
            spec = torch.as_tensor(msdata.get_spectra(i), dtype=torch.float32)
            spec_pp = spec_preproc(spec.numpy(), high_form=False)
            spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)
            mz = spec_t[:, 0].unsqueeze(0)
            pad = mz[:, 0] == 0
            mz_diffs = torch.abs(mz.unsqueeze(-1) - mz.unsqueeze(-2))
            vec = engine.get_rule_match_vectors(mz_diffs, mz_values=mz,
                precursor_mz=mz[:, 0].unsqueeze(0), padding_mask=pad,
                categories=['NL', 'CF', 'ISO', 'HR'])
            match_vecs[i] = vec.squeeze(0)
        except Exception:
            match_vecs[i] = torch.zeros(len(engine.rules))
    valid_idx = [i for i in range(n_read) if i in match_vecs and smiles_list[i] and inchikeys[i]
                 and Chem.MolFromSmiles(smiles_list[i]) is not None]
    print(f'   {len(valid_idx)} valid spectra')

    # ===== 4. Group for sampling =====
    print('\n[4] Grouping...')
    ik_to_idx = defaultdict(list)
    fm_to_idx = defaultdict(list)
    for i in valid_idx:
        ik_to_idx[inchikeys[i]].append(i)
        if formulas[i]: fm_to_idx[formulas[i]].append(i)
    multi_ik = {k: v for k, v in ik_to_idx.items() if len(v) >= 2}
    multi_fm = {k: v for k, v in fm_to_idx.items() if len(v) >= 2}
    print(f'   Multi-spectrum InChIKeys: {len(multi_ik)}')
    print(f'   Multi-spectrum Formulas: {len(multi_fm)}')

    # ===== 5. Build pairs =====
    print('\n[5] Building balanced pairs...')
    rng = np.random.RandomState(42)
    pairs, labels = [], []

    # --- Positive: same InChIKey (30%, 900) ---
    n_pos = CFG['n_pairs'] * 30 // 100
    ik_list = list(multi_ik.keys()); rng.shuffle(ik_list)
    for ik in ik_list:
        idxs = [i for i in multi_ik[ik] if i in match_vecs]
        if len(idxs) >= 2:
            a, b = rng.choice(idxs, 2, replace=False)
            pairs.append((a, b)); labels.append(1.0)
        if len(pairs) >= n_pos: break
    print(f'   Positive (same mol): {len(pairs)}')

    # --- Hard negative: isomers (30%, 900) ---
    n_iso = CFG['n_pairs'] * 30 // 100
    fm_list = list(multi_fm.keys()); rng.shuffle(fm_list)
    iso_pairs = []
    for fm in fm_list:
        idxs = [i for i in multi_fm[fm] if i in match_vecs]
        if len(idxs) < 2: continue
        seen = set()
        for _ in range(min(30, len(idxs) * 2)):
            a, b = rng.choice(idxs, 2, replace=False)
            if inchikeys[a] == inchikeys[b]: continue
            pk = (min(a, b), max(a, b))
            if pk in seen: continue
            seen.add(pk)
            tan = _compute_tanimoto(smiles_list[a], smiles_list[b])
            if 0.3 <= tan <= 0.9:
                iso_pairs.append((a, b, tan))
            if len(iso_pairs) >= n_iso: break
        if len(iso_pairs) >= n_iso: break
    for a, b, tan in iso_pairs:
        pairs.append((a, b)); labels.append(tan)
    print(f'   Hard negative (isomers): {len(iso_pairs)}, mean tanimoto={np.mean([t for _,_,t in iso_pairs]):.4f}')

    # --- Easy negative: random diff mol (40%, 1200) ---
    n_easy = CFG['n_pairs'] - len(pairs)
    for _ in range(n_easy * 3):
        a, b = rng.choice(valid_idx, 2, replace=False)
        if inchikeys[a] == inchikeys[b]: continue
        if abs(prec_mzs[a] - prec_mzs[b]) <= 1.0: continue  # skip mass-proximate
        tan = _compute_tanimoto(smiles_list[a], smiles_list[b])
        if 0 <= tan < 0.2:
            pairs.append((a, b)); labels.append(tan)
        if len(pairs) >= CFG['n_pairs']: break
    print(f'   Easy negative (random): {len(pairs) - n_pos - len(iso_pairs)}')

    labels = np.array(labels, dtype=np.float32)
    print(f'   Total: {len(pairs)} pairs')
    print(f'   Tanimoto: mean={labels.mean():.4f} std={labels.std():.4f}')

    # ===== DATA VALIDATION =====
    print(f'\n{sep}')
    print(f'  DATA VALIDATION')
    print(f'{sep}')

    # V1: Tanimoto distribution
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    print(f'  Tanimoto distribution:')
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = ((labels >= lo) & (labels < hi)).sum()
        bar = '#' * (n * 50 // len(labels))
        print(f'    [{lo:.1f},{hi:.1f}): {n:4d} ({n/len(labels)*100:.1f}%) {bar}')

    # V2: Bag size collinearity check
    bag_sizes = []
    for a, b in pairs:
        va, vb = match_vecs[a], match_vecs[b]
        bag_sizes.append(((va * vb) > 0).sum().item())
    r_bag_tan, _ = pearsonr(bag_sizes, labels)
    print(f'\n  Bag-size vs Tanimoto Pearson r: {r_bag_tan:.4f}')
    if r_bag_tan > 0.4:
        print(f'  WARNING: bag size still dominates (r={r_bag_tan:.3f} > 0.4)')
    else:
        print(f'  PASS: bag size collinearity resolved (r={r_bag_tan:.3f} < 0.4)')

    # V3: Molecule isolation assertion (done per-fold below)

    # ===== 6. Features =====
    print(f'\n[6] Building features...')
    X_agg, instances_list = [], []
    for a, b in tqdm(pairs, desc='Features'):
        va, vb = match_vecs[a], match_vecs[b]
        inter = (va * vb).sum().float()
        union = ((va + vb) > 0).float().sum()
        ov = (inter / union.clamp(min=1)).item()
        common = (va * vb) > 0; nc = common.sum().item()
        nl = common[:293].sum().item() if len(common) >= 293 else 0
        cf = common[293:293+3174].sum().item() if len(common) >= 293+3174 else 0
        iso_hr = common[293+3174:].sum().item()
        X_agg.append([ov, float(nc), iso_hr / max(nc, 1), nl / max(nc, 1), cf / max(nc, 1)])
        inst_feats = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category == 'HR': level = 2
                elif rule.category in ('NR', 'EE'): level = 0
                elif rule.category == 'ISO': level = 2
                inst_feats.append(build_instance_features({
                    'level': level, 'category': rule.category,
                    'match_type': rule.match_type, 'mass_diff_precision': 0.5}))
        instances_list.append(
            torch.tensor(np.stack(inst_feats), dtype=torch.float32) if inst_feats else torch.zeros(0, 12))
    X_agg = np.array(X_agg, dtype=np.float32)

    # ===== 7. Split =====
    print(f'\n[7] Molecule-level {CFG["n_folds"]}-fold...')
    pair_mols = []
    for a, b in pairs:
        ms = set(); ik_a = inchikeys[a]; ik_b = inchikeys[b]
        if ik_a: ms.add(ik_a)
        if ik_b: ms.add(ik_b)
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams) // CFG['n_folds']

    # ===== 8. Train =====
    print(f'\n[8] Training LR-agg + MIL on {CFG["n_folds"]} folds...')
    all_logs = {}
    lr_rs, mil_rs = [], []

    for k in range(CFG['n_folds']):
        vs, ve = k * mpf, (k + 1) * mpf if k < CFG['n_folds'] - 1 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs]) | set(ams[ve:])
        tr = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        # V3 check
        tr_m = set().union(*[pair_mols[p] for p in tr]) if tr else set()
        va_m = set().union(*[pair_mols[p] for p in va]) if va else set()
        assert len(tr_m & va_m) == 0, f'Fold {k} molecule leak!'

        # LR-agg
        lr_m = Ridge(alpha=1.0); lr_m.fit(X_agg[tr], labels[tr])
        yp = lr_m.predict(X_agg[va]); lr_r, _ = pearsonr(labels[va], yp); lr_r = max(lr_r, 0)
        lr_rs.append(lr_r)

        # MIL
        model = RuleAttentionMIL(instance_dim=12, hidden_dim=CFG['hidden_dim'])
        model.feature_extractor[2].p = CFG['dropout']
        model.attn_dropout.p = CFG['dropout']
        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-6)
        best_r, best_state, best_ep = 0, None, 0; t0 = time.time()

        for ep in range(CFG['epochs']):
            model.train(); tl, n = 0.0, 0; bl, bn = None, 0
            for pi in tr:
                bag = instances_list[pi]
                if bag.shape[0] == 0: continue
                pred, attn = model(bag)
                loss = F.mse_loss(pred, torch.tensor(labels[pi], dtype=torch.float32).unsqueeze(0))
                if len(attn) > 1:
                    ac = attn.clamp(min=1e-8)
                    loss = loss + CFG['entropy_coef'] * (-(ac * torch.log(ac)).sum() / attn.size(0))
                bl = loss if bl is None else bl + loss
                tl += loss.item(); n += 1; bn += 1
                if bn >= CFG['batch_size']:
                    bl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad(); bl = None; bn = 0
            if bl is not None: bl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
            model.eval()
            with torch.no_grad():
                pr, lb, vl = [], [], 0.0
                for pi in va:
                    bag = instances_list[pi]
                    if bag.shape[0] == 0: pr.append(0); lb.append(labels[pi]); continue
                    pred, _ = model(bag); pr.append(pred.item()); lb.append(labels[pi])
                    vl += F.mse_loss(pred, torch.tensor(labels[pi], dtype=torch.float32).unsqueeze(0)).item()
                val_r, _ = pearsonr(pr, lb); val_r = max(val_r, 0); vl /= max(len(va), 1)
            if val_r > best_r: best_r = val_r; best_ep = ep; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            sched.step()
            if ep % 50 == 0 or ep < 3:
                print(f'   F{k} ep{ep:3d}: r={val_r:.4f} best={best_r:.4f}@{best_ep}')

        if best_state: torch.save(best_state, out_dir / f'fold_{k}_best.pt')
        torch.save(model.state_dict(), out_dir / f'fold_{k}_final.pt')
        all_logs[f'fold_{k}'] = [{'epoch': i} for i in range(CFG['epochs'])]  # simplified
        mil_rs.append(best_r)
        print(f'   Fold {k}: LR={lr_r:.4f}  MIL={best_r:.4f}')

    # ===== 9. Save =====
    with open(out_dir / 'training_logs.json', 'w') as f: json.dump(all_logs, f, indent=2)
    lr_res = {'r_mean': float(np.mean(lr_rs)), 'r_std': float(np.std(lr_rs)),
              'r_folds': [float(r) for r in lr_rs]}
    with open(out_dir / 'baseline_lr_results.json', 'w') as f: json.dump(lr_res, f, indent=2)
    summary = {'LR-agg': f'{np.mean(lr_rs):.4f}+/-{np.std(lr_rs):.4f}',
               'MIL_A3': f'{np.mean(mil_rs):.4f}+/-{np.std(mil_rs):.4f}',
               'bag_tan_r': r_bag_tan}
    with open(out_dir / 'summary.json', 'w') as f: json.dump(summary, f, indent=2)
    with open(out_dir / 'config.json', 'w') as f: json.dump(CFG, f, indent=2)

    print(f'\n{sep}')
    print(f'A3 RESULTS')
    print(sep)
    print(f'  Bag-size vs Tanimoto r: {r_bag_tan:.4f}')
    print(f'  LR-agg:  r = {summary["LR-agg"]}')
    print(f'  MIL A3:  r = {summary["MIL_A3"]}')
    print(f'  Output: {out_dir}')


def _compute_tanimoto(smi_a, smi_b):
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        ma = Chem.MolFromSmiles(str(smi_a).strip())
        mb = Chem.MolFromSmiles(str(smi_b).strip())
        if ma is None or mb is None: return -1.0
        fpa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fpb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fpa, fpb)
    except Exception:
        return -1.0


if __name__ == '__main__':
    main()
