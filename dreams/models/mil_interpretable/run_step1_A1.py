"""
run_step1_A1.py — MIL Step 1 Version A1

A0→A1 改动: dropout 0.1→0.15, entropy 0.001→0.002, CosineAnnealingWarmRestarts,
             batch_size 32, 500 epochs no early stop, 10-fold CV

用法:
  python -m dreams.models.mil_interpretable.run_step1_A1
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

torch.manual_seed(42); np.random.seed(42)

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto


def main():
    out_dir = Path('outputs') / f'mil_A1_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    CFG = {
        'lr': 1e-4, 'weight_decay': 1e-5,
        'hidden_dim': 32, 'dropout': 0.15, 'entropy_coef': 0.002,
        'epochs': 500, 'batch_size': 32, 'grad_clip': 1.0,
        'n_folds': 10, 'n_pairs': 3000, 'seed': 42,
        'T_0': 50, 'T_mult': 2, 'eta_min': 1e-6,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(CFG, f, indent=2)

    sep = '=' * 60
    print(sep)
    print(f'  MIL Step 1 A1: {CFG["n_folds"]}-fold, {CFG["epochs"]} epochs, CosineAnnealing')
    print(sep)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')

    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    # ===== 1. Parse MSP =====
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

    # ===== 2. Filter =====
    print('\n[2] Filtering...')
    from rdkit import Chem
    valid = []
    for s in spectra:
        smi = s.get('SMILES','').strip()
        ik = s.get('InChIKey','').strip()
        if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
            valid.append(s)
    print(f'   Valid: {len(valid)}')

    # ===== 3. Match vectors =====
    print('\n[3] Rule match vectors...')
    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None: match_vecs[i] = vec
    valid_idx = list(match_vecs.keys())
    print(f'   {len(valid_idx)} with rule vectors')

    # ===== 4. Build pairs =====
    print('\n[4] Building balanced pairs...')
    rng = np.random.RandomState(42)
    ik_to_idx = defaultdict(list)
    for i in valid_idx: ik_to_idx[valid[i]['InChIKey']].append(i)
    multi_ik = {k:v for k,v in ik_to_idx.items() if len(v)>=2}
    pairs, labels = [], []
    ik_list = list(multi_ik.keys()); rng.shuffle(ik_list)
    for ik in ik_list:
        a,b = rng.choice(multi_ik[ik],2,replace=False)
        pairs.append((a,b)); labels.append(1.0)
        if len(pairs)>=750: break
    for _ in range(10000):
        a,b = rng.choice(valid_idx,2,replace=False)
        if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
        tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
        if tan<0: continue
        pairs.append((a,b)); labels.append(tan)
        if len(pairs)>=CFG['n_pairs']: break
    labels = np.array(labels, dtype=np.float32)
    print(f'   {len(pairs)} pairs, Tanimoto mean={labels.mean():.4f} std={labels.std():.4f}')

    # ===== 5. Build features =====
    print('\n[5] Building features...')
    X_agg, instances_list = [], []
    for a,b in tqdm(pairs, desc='Features'):
        va,vb = match_vecs[a], match_vecs[b]
        inter = (va*vb).sum().float()
        union = ((va+vb)>0).float().sum()
        ov = (inter/union.clamp(min=1)).item()
        common = (va*vb)>0; nc = common.sum().item()
        nl = common[:293].sum().item() if len(common)>=293 else 0
        cf = common[293:293+3174].sum().item() if len(common)>=293+3174 else 0
        iso_hr = common[293+3174:].sum().item()
        X_agg.append([ov, float(nc), iso_hr/max(nc,1), nl/max(nc,1), cf/max(nc,1)])
        inst_feats = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category=='HR': level=2
                elif rule.category in ('NR','EE'): level=0
                elif rule.category=='ISO': level=2
                inst_feats.append(build_instance_features({
                    'level':level,'category':rule.category,
                    'match_type':rule.match_type,'mass_diff_precision':0.5}))
        instances_list.append(torch.tensor(np.stack(inst_feats),dtype=torch.float32) if inst_feats else torch.zeros(0,12))
    X_agg = np.array(X_agg, dtype=np.float32)

    # ===== 6. Molecule-level split =====
    print(f'\n[6] Molecule-level {CFG["n_folds"]}-fold split...')
    pair_mols = []
    for a,b in pairs:
        ms = set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams)//CFG['n_folds']

    # ===== 7. Training =====
    print(f'\n[7] {CFG["n_folds"]}-fold CV ({CFG["epochs"]} epochs each)...')
    all_logs = {}
    lr_rs, mil_rs = [], []

    for k in range(CFG['n_folds']):
        vs,ve = k*mpf, (k+1)*mpf if k<CFG['n_folds']-1 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs])|set(ams[ve:])
        tr = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        assert not (set().union(*[pair_mols[p] for p in tr] if tr else []) &
                    set().union(*[pair_mols[p] for p in va] if va else [])), f'Fold {k} leak!'

        # LR-agg
        lr = Ridge(alpha=1.0); lr.fit(X_agg[tr], labels[tr])
        yp = lr.predict(X_agg[va]); lr_r,_ = pearsonr(labels[va], yp); lr_r=max(lr_r,0); lr_rs.append(lr_r)
        with open(out_dir / f'fold_{k}_lr.pkl', 'wb') as f: pickle.dump(lr, f)

        # MIL
        model = RuleAttentionMIL(instance_dim=12, hidden_dim=CFG['hidden_dim'])
        model.feature_extractor[2].p = CFG['dropout']
        model.attn_dropout.p = CFG['dropout']

        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=CFG['T_0'], T_mult=CFG['T_mult'], eta_min=CFG['eta_min'])

        logs = []
        best_r, best_state, best_epoch = 0, None, 0
        t0 = time.time()

        for ep in range(CFG['epochs']):
            model.train()
            train_loss, n = 0.0, 0
            batch_loss, bn = None, 0
            for pi in tr:
                bag = instances_list[pi]
                if bag.shape[0]==0: continue
                pred, attn = model(bag)
                loss = F.mse_loss(pred, torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0))
                if len(attn)>1:
                    ac = attn.clamp(min=1e-8)
                    loss = loss + CFG['entropy_coef']*(-(ac*torch.log(ac)).sum()/attn.size(0))
                batch_loss = loss if batch_loss is None else batch_loss+loss
                train_loss+=loss.item(); n+=1; bn+=1
                if bn>=CFG['batch_size']:
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['grad_clip'])
                    opt.step(); opt.zero_grad()
                    batch_loss=None; bn=0
            if batch_loss is not None:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['grad_clip'])
                opt.step(); opt.zero_grad()

            model.eval()
            with torch.no_grad():
                pr,lb,vl=[],[],0.0
                for pi in va:
                    bag=instances_list[pi]
                    if bag.shape[0]==0: pr.append(0); lb.append(labels[pi]); continue
                    pred,_=model(bag); pr.append(pred.item()); lb.append(labels[pi])
                    vl+=F.mse_loss(pred,torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0)).item()
                val_r,_=pearsonr(pr,lb); val_r=max(val_r,0); vl/=max(len(va),1)
            lr_now = opt.param_groups[0]['lr']
            if val_r>best_r:
                best_r=val_r; best_epoch=ep
                best_state={k:v.clone() for k,v in model.state_dict().items()}
            scheduler.step()
            logs.append({'epoch':ep,'lr':lr_now,'train_loss':train_loss/max(n,1),
                         'val_loss':vl,'val_r':val_r,'best_r':best_r,'best_epoch':best_epoch})

            if ep%10==0 or ep<5:
                elapsed=time.time()-t0
                eta=(elapsed/(ep+1))*(CFG['epochs']-ep-1)
                print(f'   F{k} ep{ep:3d}: train={logs[-1]["train_loss"]:.4f} '
                      f'val={vl:.4f} r={val_r:.4f} best={best_r:.4f}@{best_epoch} '
                      f'lr={lr_now:.1e} eta={eta/60:.0f}m')

        if best_state:
            torch.save(best_state, out_dir/f'fold_{k}_best_model.pt')
        torch.save(model.state_dict(), out_dir/f'fold_{k}_final_model.pt')
        all_logs[f'fold_{k}'] = logs
        mil_rs.append(best_r)
        print(f'   Fold {k}: best_r={best_r:.4f}@{best_epoch}  LR_r={lr_r:.4f}')

    # ===== 8. Save =====
    with open(out_dir/'training_logs.json','w') as f: json.dump(all_logs, f, indent=2)
    lr_res = {'model':'LR-agg','r_mean':float(np.mean(lr_rs)),'r_std':float(np.std(lr_rs)),
              'r_folds':[float(r) for r in lr_rs]}
    with open(out_dir/'baseline_lr_results.json','w') as f: json.dump(lr_res, f, indent=2)
    summary = {'n_folds':CFG['n_folds'],'epochs':CFG['epochs'],
               'mil_r_mean':float(np.mean(mil_rs)),'mil_r_std':float(np.std(mil_rs)),
               'mil_r_folds':[float(r) for r in mil_rs],
               'lr_r_mean':lr_res['r_mean'],'lr_r_std':lr_res['r_std']}
    with open(out_dir/'summary.json','w') as f: json.dump(summary, f, indent=2)

    print(f'\n{sep}')
    print(f'A1 RESULTS ({CFG["n_folds"]}-fold)')
    print(sep)
    print(f'  LR-agg:        r = {lr_res["r_mean"]:.4f} +/- {lr_res["r_std"]:.4f}')
    print(f'  Attention MIL: r = {summary["mil_r_mean"]:.4f} +/- {summary["mil_r_std"]:.4f}')
    print(f'  Output: {out_dir}')


if __name__ == '__main__':
    main()
