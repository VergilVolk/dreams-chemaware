"""
run_step1_A1.py — MIL Step 1: A0 vs A1 vs LR-agg on same 10 folds

配置:
  LR-agg: Ridge on 5 aggregated features
  A0: hidden_dim=32, dropout=0.1, entropy=0.001, Plateau scheduler, early stop
  A1: hidden_dim=32, dropout=0.15, entropy=0.002, CosineAnnealing, 500 epochs

用法:
  python -m dreams.models.mil_interpretable.run_step1_A1
"""

import torch, torch.nn.functional as F
import numpy as np
import json, pickle, time
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
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto


def train_one_config(label, mdl_cfg, tr, va, instances_list, labels, out_dir, fold_idx):
    """Train one MIL config, return logs+best_r"""
    model = RuleAttentionMIL(instance_dim=12, hidden_dim=mdl_cfg['hidden_dim'])
    model.feature_extractor[2].p = mdl_cfg['dropout']
    model.attn_dropout.p = mdl_cfg['dropout']
    opt = torch.optim.AdamW(model.parameters(), lr=mdl_cfg['lr'], weight_decay=mdl_cfg['weight_decay'])
    if mdl_cfg['scheduler'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=20)

    logs = []; best_r, best_state, best_ep = 0, None, 0
    counter = 0; patience = mdl_cfg.get('patience', 9999); t0 = time.time()

    for ep in range(mdl_cfg['epochs']):
        model.train()
        tl, n = 0.0, 0
        bl, bn = None, 0
        for pi in tr:
            bag = instances_list[pi]
            if bag.shape[0] == 0: continue
            pred, attn = model(bag)
            loss = F.mse_loss(pred, torch.tensor(labels[pi], dtype=torch.float32).unsqueeze(0))
            if len(attn) > 1:
                ac = attn.clamp(min=1e-8)
                loss = loss + mdl_cfg['entropy_coef'] * (-(ac * torch.log(ac)).sum() / attn.size(0))
            bl = loss if bl is None else bl + loss
            tl += loss.item(); n += 1; bn += 1
            if bn >= mdl_cfg['batch_size']:
                bl.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                bl = None; bn = 0
        if bl is not None:
            bl.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        model.eval()
        with torch.no_grad():
            pr, lb, vl = [], [], 0.0
            for pi in va:
                bag = instances_list[pi]
                if bag.shape[0] == 0: pr.append(0); lb.append(labels[pi]); continue
                pred, _ = model(bag)
                pr.append(pred.item()); lb.append(labels[pi])
                vl += F.mse_loss(pred, torch.tensor(labels[pi], dtype=torch.float32).unsqueeze(0)).item()
            val_r, _ = pearsonr(pr, lb); val_r = max(val_r, 0); vl /= max(len(va), 1)

        if val_r > best_r:
            best_r = val_r; best_ep = ep; counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            counter += 1

        if mdl_cfg['scheduler'] == 'plateau': sched.step(val_r)
        else: sched.step()

        logs.append({'epoch': ep, 'lr': opt.param_groups[0]['lr'],
                     'train_loss': tl / max(n, 1), 'val_loss': vl,
                     'val_r': val_r, 'best_r': best_r, 'best_epoch': best_ep})

        if ep % 50 == 0 or ep < 3:
            print(f'     {label} ep{ep:3d}: r={val_r:.4f} best={best_r:.4f}@{best_ep}')

        if counter >= patience: break

    if best_state: torch.save(best_state, out_dir / f'fold_{fold_idx}_{label}_best.pt')
    torch.save(model.state_dict(), out_dir / f'fold_{fold_idx}_{label}_final.pt')
    return logs, best_r


def main():
    out_dir = Path('outputs') / f'mil_A0vsA1_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    N_FOLDS = 10

    # A0 config
    A0 = {'hidden_dim': 32, 'dropout': 0.1, 'entropy_coef': 0.001,
          'lr': 1e-4, 'weight_decay': 1e-5, 'batch_size': 64,
          'epochs': 300, 'patience': 30, 'scheduler': 'plateau', 'label': 'A0'}

    # A1 config
    A1 = {'hidden_dim': 32, 'dropout': 0.15, 'entropy_coef': 0.002,
          'lr': 1e-4, 'weight_decay': 1e-5, 'batch_size': 32,
          'epochs': 500, 'patience': 9999, 'scheduler': 'cosine', 'label': 'A1'}

    with open(out_dir / 'config.json', 'w') as f:
        json.dump({'A0': A0, 'A1': A1, 'n_folds': N_FOLDS}, f, indent=2)

    sep = '=' * 60
    print(sep)
    print(f'  A0 vs A1 vs LR-agg — {N_FOLDS}-fold CV')
    print(sep)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')

    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    # ===== 1-5: Parse, Filter, Match vecs, Build pairs, Features =====
    print('\n[1] Parsing MSP...')
    spectra = []
    for fp in ['data/MassBank_NIST.msp', 'data/MoNA-export-LC-MS-MS_Spectra.msp',
               'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']:
        s = parse_msp(fp, max_spectra=20000)
        print(f'   {fp.replace(chr(92),chr(47)).split(chr(47))[-1]}: {len(s)}')
        spectra.extend(s)
    print(f'   Total: {len(spectra)}')

    print('\n[2] Filtering...')
    from rdkit import Chem
    valid = []
    for s in spectra:
        smi = s.get('SMILES', '').strip()
        ik = s.get('InChIKey', '').strip()
        if smi and ik and len(smi) > 2 and Chem.MolFromSmiles(smi) is not None:
            valid.append(s)
    print(f'   Valid: {len(valid)}')

    print('\n[3] Match vectors...')
    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None: match_vecs[i] = vec
    vidx = list(match_vecs.keys())
    print(f'   {len(vidx)} with rule vectors')

    print('\n[4] Building pairs...')
    rng = np.random.RandomState(42)
    ik2idx = defaultdict(list)
    for i in vidx: ik2idx[valid[i]['InChIKey']].append(i)
    multi_ik = {k: v for k, v in ik2idx.items() if len(v) >= 2}
    pairs, labels = [], []
    for ik in sorted(multi_ik.keys(), key=lambda x: -len(multi_ik[x])):
        a, b = rng.choice(multi_ik[ik], 2, replace=False)
        pairs.append((a, b)); labels.append(1.0)
        if len(pairs) >= 750: break
    for _ in range(10000):
        a, b = rng.choice(vidx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
        if tan < 0: continue
        pairs.append((a, b)); labels.append(tan)
        if len(pairs) >= 3000: break
    labels = np.array(labels, dtype=np.float32)
    print(f'   {len(pairs)} pairs, tanimoto mean={labels.mean():.4f} std={labels.std():.4f}')

    print('\n[5] Features...')
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

    # ===== 6. Molecule split =====
    print(f'\n[6] Molecule-level {N_FOLDS}-fold...')
    pair_mols = []
    for a, b in pairs:
        ms = set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams) // N_FOLDS

    # ===== 7. Train =====
    print(f'\n[7] Training LR-agg + A0 + A1 on {N_FOLDS} folds...')
    all_logs = {}
    lr_rs, a0_rs, a1_rs = [], [], []

    for k in range(N_FOLDS):
        vs, ve = k * mpf, (k + 1) * mpf if k < N_FOLDS - 1 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs]) | set(ams[ve:])
        tr = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi, pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        assert not (set().union(*[pair_mols[p] for p in tr] if tr else []) &
                    set().union(*[pair_mols[p] for p in va] if va else [])), f'Fold {k} leak!'

        print(f'\n--- Fold {k} ---')
        # LR-agg
        lr = Ridge(alpha=1.0); lr.fit(X_agg[tr], labels[tr])
        yp = lr.predict(X_agg[va]); lr_r, _ = pearsonr(labels[va], yp); lr_r = max(lr_r, 0)
        lr_rs.append(lr_r)
        print(f'   LR-agg: r={lr_r:.4f}')

        # A0
        print(f'   A0 (Plateau, early stop)...')
        a0_logs, a0_r = train_one_config('A0', A0, tr, va, instances_list, labels, out_dir, k)
        a0_rs.append(a0_r)
        all_logs[f'fold_{k}_A0'] = a0_logs

        # A1
        print(f'   A1 (Cosine, 500 epochs)...')
        a1_logs, a1_r = train_one_config('A1', A1, tr, va, instances_list, labels, out_dir, k)
        a1_rs.append(a1_r)
        all_logs[f'fold_{k}_A1'] = a1_logs

        print(f'   Fold {k}: LR={lr_r:.4f}  A0={a0_r:.4f}  A1={a1_r:.4f}')

    # ===== 8. Save =====
    with open(out_dir / 'training_logs.json', 'w') as f: json.dump(all_logs, f, indent=2)
    lr_res = {'model': 'LR-agg', 'r_mean': float(np.mean(lr_rs)), 'r_std': float(np.std(lr_rs)),
              'r_folds': [float(r) for r in lr_rs]}
    with open(out_dir / 'baseline_lr_results.json', 'w') as f: json.dump(lr_res, f, indent=2)
    summary = {
        'n_folds': N_FOLDS,
        'LR-agg': f'{np.mean(lr_rs):.4f}+/-{np.std(lr_rs):.4f}',
        'MIL_A0': f'{np.mean(a0_rs):.4f}+/-{np.std(a0_rs):.4f}',
        'MIL_A1': f'{np.mean(a1_rs):.4f}+/-{np.std(a1_rs):.4f}',
        'a0_folds': [float(r) for r in a0_rs], 'a1_folds': [float(r) for r in a1_rs],
    }
    with open(out_dir / 'summary.json', 'w') as f: json.dump(summary, f, indent=2)

    print(f'\n{sep}')
    print(f'FINAL COMPARISON ({N_FOLDS}-fold)')
    print(sep)
    print(f'  LR-agg:  r = {summary["LR-agg"]}')
    print(f'  MIL A0:  r = {summary["MIL_A0"]}')
    print(f'  MIL A1:  r = {summary["MIL_A1"]}')
    print(f'  Output: {out_dir}')


if __name__ == '__main__':
    main()
