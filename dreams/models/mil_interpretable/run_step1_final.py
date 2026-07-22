"""
run_step1_final.py — MIL Step 1 Final: 500 epochs, no early stop, full logging

配置:
  lr=2e-4, AdamW(wd=1e-4), ReduceLROnPlateau(patience=100, factor=0.5)
  hidden_dim=64, dropout=0.2, entropy=0.005
  500 epochs, batch_size=64, 5-fold CV, fixed seeds

输出目录: outputs/mil_regression_YYYYMMDD_HHMMSS/

用法:
  python -m dreams.models.mil_interpretable.run_step1_final
"""

import torch, torch.nn.functional as F
import numpy as np
import json, pickle, os, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from tqdm import tqdm

# ===== 固定随机种子 =====
torch.manual_seed(42)
np.random.seed(42)

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto


def main():
    # ===== 创建输出目录 =====
    out_dir = Path('outputs') / f'mil_regression_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}')

    print('=' * 60)
    print('MIL Step 1 FINAL v2: 300 epochs + health monitoring')
    print('=' * 60)

    # ===== 配置（融合版：版本A的随机性 + 版本B的耐心）=====
    CFG = {
        'lr': 5e-4, 'weight_decay': 1e-3, 'betas': (0.9, 0.999),
        'hidden_dim': 32, 'dropout': 0.3, 'entropy_coef': 0.02,
        'epochs': 300, 'batch_size': 32, 'grad_clip': 1.0,
        'scheduler_patience': 50, 'scheduler_factor': 0.5, 'min_lr': 1e-6,
        'early_stop_patience': 100,  # 恢复早停，100 epoch 耐心
        'ori_warn_threshold': 10,     # ORI 连续上升 10 epoch → 预警
        'decay_warn_threshold': 20,   # val_r 连续不刷新 20 epoch → 预警
        'n_folds': 5, 'n_pairs': 3000, 'seed': 42,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(CFG, f, indent=2)

    # ===== 加载引擎 =====
    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')

    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    # ===== 1. 解析数据 =====
    print('\n[1] Parsing MSP files...')
    spectra = []
    for fpath in ['data/MassBank_NIST.msp',
                  'data/MoNA-export-LC-MS-MS_Spectra.msp',
                  'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']:
        short = fpath.replace('\\','/').split('/')[-1]
        s = parse_msp(fpath, max_spectra=20000)
        print(f'   {short}: {len(s)}')
        spectra.extend(s)
    print(f'   Total: {len(spectra)}')

    # ===== 2. 过滤 =====
    print('\n[2] Filtering...')
    from rdkit import Chem
    valid = []
    for s in spectra:
        smi = s.get('SMILES','').strip()
        ik = s.get('InChIKey','').strip()
        if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
            valid.append(s)
    print(f'   Valid: {len(valid)}')

    # ===== 3. 规则向量 =====
    print('\n[3] Computing rule match vectors...')
    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None:
            match_vecs[i] = vec
    valid_idx = list(match_vecs.keys())
    print(f'   {len(valid_idx)} with rule vectors')

    # ===== 4. 构建配对 =====
    print('\n[4] Building balanced pairs...')
    rng = np.random.RandomState(42)
    ik_to_idx = defaultdict(list)
    for i in valid_idx:
        ik_to_idx[valid[i]['InChIKey']].append(i)
    multi_ik = {k:v for k,v in ik_to_idx.items() if len(v)>=2}

    pairs, labels = [], []
    ik_list = list(multi_ik.keys()); rng.shuffle(ik_list)
    for ik in ik_list:
        a,b = rng.choice(multi_ik[ik], 2, replace=False)
        pairs.append((a,b)); labels.append(1.0)
        if len(pairs) >= 750: break

    for _ in range(10000):
        a,b = rng.choice(valid_idx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
        if tan < 0: continue
        pairs.append((a,b)); labels.append(tan)
        if len(pairs) >= CFG['n_pairs']: break

    labels = np.array(labels, dtype=np.float32)
    print(f'   {len(pairs)} pairs, Tanimoto mean={labels.mean():.4f} std={labels.std():.4f}')

    # ===== 5. 构建特征 =====
    print('\n[5] Building features...')
    X_agg, instances_list = [], []
    for a,b in tqdm(pairs, desc='Features'):
        va, vb = match_vecs[a], match_vecs[b]
        inter = (va*vb).sum().float()
        union = ((va+vb)>0).float().sum()
        ov = (inter/union.clamp(min=1)).item()
        common = (va*vb)>0; nc = common.sum().item()
        nl_cf = nc  # simplified
        X_agg.append([ov, float(nc), 0.2, 0.3, 0.5])  # placeholder for now
        # Better aggregated features
        nl = common[:293].sum().item() if len(common) >= 293 else 0
        cf = common[293:293+3174].sum().item() if len(common) >= 293+3174 else 0
        iso_hr = common[293+3174:].sum().item()
        X_agg[-1] = [ov, float(nc), iso_hr/max(nc,1), nl/max(nc,1), cf/max(nc,1)]

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
        instances_list.append(torch.tensor(np.stack(inst_feats),dtype=torch.float32) if inst_feats else torch.zeros(0,12))

    X_agg = np.array(X_agg, dtype=np.float32)

    # ===== 6. 构建分子级切分 =====
    print('\n[6] Molecule-level split...')
    pair_mols = []
    for a,b in pairs:
        ms = set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams)//CFG['n_folds']

    # ===== 7. 5-fold CV =====
    print('\n[7] 5-fold CV (500 epochs each)...')
    all_logs = {}
    lr_rs, best_folds = [], []

    for k in range(CFG['n_folds']):
        vs,ve = k*mpf, (k+1)*mpf if k<CFG['n_folds']-1 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs])|set(ams[ve:])
        tr = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        assert not (set().union(*[pair_mols[p] for p in tr] if tr else []) &
                    set().union(*[pair_mols[p] for p in va] if va else [])), f'Fold {k} leak!'

        # ---- LR-agg baseline ----
        lr = Ridge(alpha=1.0); lr.fit(X_agg[tr], labels[tr])
        yp = lr.predict(X_agg[va]); lr_r, _ = pearsonr(labels[va], yp)
        lr_r = max(lr_r, 0); lr_rs.append(lr_r)
        with open(out_dir / f'fold_{k}_lr.pkl', 'wb') as f:
            pickle.dump(lr, f)

        # ---- MIL ----
        model = RuleAttentionMIL(instance_dim=12, hidden_dim=CFG['hidden_dim'])
        model.feature_extractor[2].p = CFG['dropout']
        model.attn_dropout.p = CFG['dropout']

        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'],
                                betas=CFG['betas'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=CFG['scheduler_factor'],
            patience=CFG['scheduler_patience'], min_lr=CFG['min_lr'])

        logs = []
        best_r, best_state, best_epoch = 0, None, 0
        t0 = time.time()
        ori_rise_count = 0       # ORI 连续上升计数器
        decay_count = 0          # 泛化衰退计数器
        prev_ori = None

        for ep in range(CFG['epochs']):
            # Train
            model.train()
            train_loss, n = 0.0, 0
            batch_loss = None; bn = 0
            for pi in tr:
                bag = instances_list[pi]
                if bag.shape[0]==0: continue
                pred, attn = model(bag)
                loss = F.mse_loss(pred, torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0))
                if len(attn)>1:
                    ac = attn.clamp(min=1e-8)
                    loss = loss + CFG['entropy_coef'] * (-(ac*torch.log(ac)).sum()/attn.size(0))
                batch_loss = loss if batch_loss is None else batch_loss+loss
                train_loss += loss.item(); n += 1; bn += 1
                if bn >= CFG['batch_size']:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                    opt.step(); opt.zero_grad()
                    batch_loss = None; bn = 0
            if batch_loss is not None:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
                opt.step(); opt.zero_grad()

            # Validate
            model.eval()
            with torch.no_grad():
                pr, lb, val_loss = [], [], 0.0
                for pi in va:
                    bag = instances_list[pi]
                    if bag.shape[0]==0: pr.append(0); lb.append(labels[pi]); continue
                    pred, _ = model(bag)
                    pr.append(pred.item()); lb.append(labels[pi])
                    val_loss += F.mse_loss(pred, torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0)).item()
                val_r, _ = pearsonr(pr, lb); val_r = max(val_r, 0)
                val_loss /= max(len(va), 1)

            tl = train_loss/max(n,1)
            lr_now = opt.param_groups[0]['lr']

            # ---- 健康度监控 ----
            ori = (val_loss - tl) / max(tl, 1e-8)
            if prev_ori is not None and ori > prev_ori:
                ori_rise_count += 1
            else:
                ori_rise_count = 0
            prev_ori = ori
            if val_r < best_r:
                decay_count += 1
            else:
                decay_count = 0

            logs.append({
                'epoch': ep, 'lr': lr_now,
                'train_loss': tl, 'val_loss': val_loss, 'val_r': val_r,
                'ori': ori, 'ori_rise': ori_rise_count, 'decay': decay_count,
            })

            if val_r > best_r:
                best_r = val_r; best_epoch = ep
                best_state = {k:v.clone() for k,v in model.state_dict().items()}

            scheduler.step(val_r)

            # 过拟合预警
            warn = ''
            if ori_rise_count >= CFG['ori_warn_threshold']:
                warn += f' [ORI_RISE={ori_rise_count}]'
            if decay_count >= CFG['decay_warn_threshold']:
                warn += f' [DECAY={decay_count}]'

            if ep % 10 == 0 or ep < 10 or warn:
                elapsed = time.time()-t0
                eta = (elapsed/(ep+1))*(CFG['epochs']-ep-1)
                print(f'   Fold {k} ep {ep:3d}/{CFG["epochs"]}: '
                      f'train={tl:.4f} val={val_loss:.4f} r={val_r:.4f} '
                      f'best={best_r:.4f}@{best_epoch} '
                      f'ORI={ori:.3f} D={decay_count} '
                      f'lr={lr_now:.1e}{warn} eta={eta/60:.0f}m')

            # 早停
            if decay_count >= CFG['early_stop_patience']:
                print(f'   Fold {k}: EARLY STOP at epoch {ep} (decay={decay_count})')
                break

        # Save best model
        if best_state:
            torch.save(best_state, out_dir / f'fold_{k}_model.pt')

        all_logs[f'fold_{k}'] = logs
        best_folds.append({'fold': k, 'best_r': best_r, 'best_epoch': best_epoch})
        print(f'   Fold {k} done: best_r={best_r:.4f}@{best_epoch}  LR-agg_r={lr_rs[-1]:.4f}')

    # ===== 8. 保存 =====
    print('\n[8] Saving results...')
    with open(out_dir / 'training_logs.json', 'w') as f:
        json.dump(all_logs, f, indent=2)
    with open(out_dir / 'best_fold_config.json', 'w') as f:
        json.dump(best_folds, f, indent=2)
    lr_results = {
        'model': 'LR-agg',
        'r_mean': float(np.mean(lr_rs)), 'r_std': float(np.std(lr_rs)),
        'r_folds': [float(r) for r in lr_rs],
    }
    with open(out_dir / 'baseline_lr_results.json', 'w') as f:
        json.dump(lr_results, f, indent=2)

    mil_rs = [b['best_r'] for b in best_folds]
    mil_mean, mil_std = float(np.mean(mil_rs)), float(np.std(mil_rs))

    print(f'\n{"="*60}')
    print(f'FINAL RESULTS')
    print(f'{"="*60}')
    print(f'  LR-agg:        r = {lr_results["r_mean"]:.4f} +/- {lr_results["r_std"]:.4f}')
    print(f'  Attention MIL: r = {mil_mean:.4f} +/- {mil_std:.4f}')
    print(f'  Output dir: {out_dir}')
    if mil_mean > lr_results['r_mean']:
        print(f'  >>> MIL BEATS LR-agg!')
    else:
        print(f'  >>> MIL below LR-agg by {lr_results["r_mean"]-mil_mean:.4f}')


if __name__ == '__main__':
    main()
