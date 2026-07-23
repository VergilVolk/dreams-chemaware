"""
run_step1_A3_benchmark.py — A3新数据上全版本基准对比

数据: 3 MSP文件, 30%pos + 30%isomer + 40%random
模型: LR-agg(5-feat) + LR-agg(bag-size) + MIL A0 + MIL A1 + MIL A2
全部在相同10折上对比

用法:
  python -m dreams.models.mil_interpretable.run_step1_A3_benchmark
"""

import torch, torch.nn.functional as F
import numpy as np, json, pickle, time, math
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

# ==============================================================================
# AdaCosine Scheduler (from A2)
# ==============================================================================

class AdaCosineScheduler:
    def __init__(self, optimizer, T_0=100, eta_min=1e-6,
                 patience_stagnant=25, patience_overfit=10, extend_threshold=20):
        self.opt = optimizer; self.T_0 = T_0; self.eta_min = eta_min
        self.eta_max = optimizer.param_groups[0]['lr']
        self.patience_stagnant = patience_stagnant
        self.patience_overfit = patience_overfit
        self.extend_threshold = extend_threshold
        self.T_cur = 0; self.best_r = 0.0
        self.epochs_no_improve = 0; self.ori_streak = 0
        self.prev_ori = None; self.cycle_extension = 0
        self.restart_triggered = False

    def step(self, val_r, train_loss, val_loss):
        self.T_cur += 1; self.restart_triggered = False
        if val_r > self.best_r: self.best_r = val_r; self.epochs_no_improve = 0; self.cycle_extension = self.extend_threshold
        else: self.epochs_no_improve += 1
        ori = (val_loss - train_loss) / max(train_loss, 1e-8)
        if self.prev_ori is not None and ori > self.prev_ori: self.ori_streak += 1
        else: self.ori_streak = 0
        self.prev_ori = ori
        early = (self.epochs_no_improve >= self.patience_stagnant or self.ori_streak >= self.patience_overfit)
        if self.cycle_extension > 0: self.cycle_extension -= 1; early = False
        if early: self.T_cur = 0; self.epochs_no_improve = 0; self.ori_streak = 0; self.restart_triggered = True
        eff_T = self.T_0 + max(0, self.cycle_extension)
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / eff_T))
        for pg in self.opt.param_groups: pg['lr'] = lr
        return lr


# ==============================================================================
# Training helper
# ==============================================================================

def train_mil(label, mdl_cfg, tr, va, instances_list, labels, out_dir, fold_idx):
    model = RuleAttentionMIL(instance_dim=12, hidden_dim=mdl_cfg['hidden_dim'])
    model.feature_extractor[2].p = mdl_cfg['dropout']
    model.attn_dropout.p = mdl_cfg['dropout']
    opt = torch.optim.AdamW(model.parameters(), lr=mdl_cfg['lr'], weight_decay=mdl_cfg['weight_decay'])

    if mdl_cfg['scheduler'] == 'adacosine':
        sched = AdaCosineScheduler(opt, T_0=mdl_cfg['T_0'], eta_min=1e-6,
                                    patience_stagnant=mdl_cfg['patience_stagnant'],
                                    patience_overfit=mdl_cfg['patience_overfit'],
                                    extend_threshold=mdl_cfg['extend_threshold'])
    elif mdl_cfg['scheduler'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=mdl_cfg['T_0'], T_mult=2, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=20)

    logs = []; best_r, best_state, best_ep = 0, None, 0
    counter = 0; patience = mdl_cfg.get('patience', 9999); t0 = time.time()

    for ep in range(mdl_cfg['epochs']):
        model.train(); tl, n = 0.0, 0; bl, bn = None, 0
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

        if val_r > best_r: best_r = val_r; best_ep = ep; counter = 0
        best_state_cur = {k: v.clone() for k, v in model.state_dict().items()} if val_r > best_r else best_state
        if val_r <= best_r: counter += 1

        lr_now = opt.param_groups[0]['lr']
        restart = False
        if mdl_cfg['scheduler'] == 'adacosine':
            lr_now = sched.step(val_r, tl/max(n,1), vl)
            restart = sched.restart_triggered
        elif mdl_cfg['scheduler'] == 'plateau': sched.step(val_r)
        else: sched.step()

        if ep % 10 == 0:
            logs.append({'epoch': ep, 'lr': lr_now, 'train_loss': tl/max(n,1),
                         'val_loss': vl, 'val_r': val_r, 'best_r': best_r, 'best_epoch': best_ep,
                         'restart': restart})
        if counter >= patience: break

    if best_state: torch.save(best_state, out_dir / f'{label}_fold{fold_idx}_best.pt')
    torch.save(model.state_dict(), out_dir / f'{label}_fold{fold_idx}_final.pt')
    return logs, best_r


# ==============================================================================
# Main
# ==============================================================================

def main():
    out_dir = Path('outputs') / f'A012_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    N_FOLDS = 10

    # Configs
    A0 = {'hidden_dim':32,'dropout':0.1,'entropy_coef':0.001,'lr':1e-4,'weight_decay':1e-5,
          'batch_size':32,'epochs':500,'patience':30,'scheduler':'plateau','label':'A0'}
    A1 = {'hidden_dim':32,'dropout':0.1,'entropy_coef':0.001,'lr':1e-4,'weight_decay':1e-5,
          'batch_size':32,'epochs':500,'patience':9999,'scheduler':'cosine','label':'A1','T_0':100}
    A2 = {'hidden_dim':32,'dropout':0.1,'entropy_coef':0.001,'lr':1e-4,'weight_decay':1e-5,
          'batch_size':32,'epochs':800,'patience':9999,'scheduler':'adacosine','label':'A2',
          'T_0':100,'patience_stagnant':25,'patience_overfit':10,'extend_threshold':20}

    with open(out_dir/'config.json','w') as f: json.dump({'A0':A0,'A1':A1,'A2':A2,'n_folds':N_FOLDS},f,indent=2)

    sep = '='*60
    print(sep)
    print(f'  A3 BENCHMARK — LR-agg + A0 + A1 + A2 ({N_FOLDS}-fold)')
    print(sep)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')
    import dreams.utils.dformats as dformats; import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    MSP_FILES = ['data/MassBank_NIST.msp','data/MoNA-export-LC-MS-MS_Spectra.msp',
                 'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']

    # ===== 1-5: Data prep (same as A3) =====
    print('\n[1] Parsing MSP...')
    spectra = []
    for fp in MSP_FILES:
        s = parse_msp(fp, max_spectra=20000)
        print(f'   {fp.replace(chr(92),chr(47)).split(chr(47))[-1]}: {len(s)}')
        spectra.extend(s)
    print(f'   Total: {len(spectra)}')

    print('\n[2] Filtering...')
    from rdkit import Chem
    valid = []
    for s in spectra:
        smi = s.get('SMILES','').strip(); ik = s.get('InChIKey','').strip()
        if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
            fm = Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi))
            s['_formula'] = fm; valid.append(s)
    print(f'   Valid: {len(valid)}')

    print('\n[3] Match vectors...')
    match_vecs = {}
    for i,s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None: match_vecs[i] = vec
    vidx = [i for i in range(len(valid)) if i in match_vecs]
    print(f'   {len(vidx)} with rule vectors')

    print('\n[4] Building pairs...')
    rng = np.random.RandomState(42)
    ik2idx = defaultdict(list); fm2idx = defaultdict(list)
    for i in vidx:
        ik2idx[valid[i]['InChIKey']].append(i)
        fm = valid[i].get('_formula','')
        if fm: fm2idx[fm].append(i)
    multi_ik = {k:v for k,v in ik2idx.items() if len(v)>=2}
    multi_fm = {k:v for k,v in fm2idx.items() if len(v)>=2}
    print(f'   Multi-IK: {len(multi_ik)}, Multi-FM: {len(multi_fm)}')

    N_TARGET = 3000; pairs, labels = [], []
    # Pos
    for ik in sorted(multi_ik.keys(), key=lambda x: -len(multi_ik[x])):
        a,b = rng.choice(multi_ik[ik],2,replace=False)
        pairs.append((a,b)); labels.append(1.0)
        if len(pairs)>=900: break
    # Hard neg
    for fm in sorted(multi_fm.keys(), key=lambda x: -len(multi_fm[x])):
        idxs = multi_fm[fm]
        if len(idxs)<2: continue
        seen = set()
        for _ in range(min(50,len(idxs)*3)):
            a,b = rng.choice(idxs,2,replace=False)
            if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
            pk = (min(a,b),max(a,b))
            if pk in seen: continue
            seen.add(pk)
            tan = compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
            if 0.3<=tan<=0.9: pairs.append((a,b)); labels.append(tan)
            if len(pairs)>=1800: break
        if len(pairs)>=1800: break
    # Easy neg
    for _ in range((N_TARGET-len(pairs))*5):
        a,b = rng.choice(vidx,2,replace=False)
        if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
        pm_a = float(valid[a].get('PrecursorMZ',0) or 0)
        pm_b = float(valid[b].get('PrecursorMZ',0) or 0)
        if abs(pm_a-pm_b)<=1.0: continue
        tan = compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
        if 0<=tan<0.2: pairs.append((a,b)); labels.append(tan)
        if len(pairs)>=N_TARGET: break
    labels = np.array(labels, dtype=np.float32)
    n_pos = 900; n_iso = (np.array(labels)>=0.3).sum()-900
    n_easy = len(pairs)-n_pos-n_iso
    print(f'   Total: {len(pairs)} (pos={n_pos}, iso={n_iso}, easy={n_easy})')
    print(f'   Tanimoto: mean={labels.mean():.4f} std={labels.std():.4f}')

    # V2 check
    bag_sizes = []
    for a,b in pairs:
        va,vb = match_vecs[a], match_vecs[b]
        bag_sizes.append(((va*vb)>0).sum().item())
    r_bag,_ = pearsonr(bag_sizes, labels)
    print(f'   Bag-Tanimoto r = {r_bag:.4f}')
    with open(out_dir/'data_report.json','w') as f:
        json.dump({'n_pairs':len(pairs),'n_pos':n_pos,'n_iso':n_iso,'n_easy':n_easy,
                   'tan_mean':float(labels.mean()),'tan_std':float(labels.std()),
                   'r_bag_tan':float(r_bag)},f,indent=2)

    # ===== 5. Features =====
    print('\n[5] Features...')
    X_agg, instances_list = [], []
    for a,b in tqdm(pairs, desc='Features'):
        va,vb = match_vecs[a], match_vecs[b]
        inter = (va*vb).sum().float(); union = ((va+vb)>0).float().sum()
        ov = (inter/union.clamp(min=1)).item()
        common = (va*vb)>0; nc = common.sum().item()
        nl = common[:293].sum().item() if len(common)>=293 else 0
        cf = common[293:293+3174].sum().item() if len(common)>=293+3174 else 0
        iso_hr = common[293+3174:].sum().item()
        X_agg.append([ov, float(nc), iso_hr/max(nc,1), nl/max(nc,1), cf/max(nc,1)])
        X_bag = [[float(nc)]]
        inst_feats = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category=='HR': level=2
                elif rule.category in ('NR','EE'): level=0
                elif rule.category=='ISO': level=2
                inst_feats.append(build_instance_features({'level':level,'category':rule.category,
                    'match_type':rule.match_type,'mass_diff_precision':0.5}))
        instances_list.append(torch.tensor(np.stack(inst_feats),dtype=torch.float32) if inst_feats else torch.zeros(0,12))
    X_agg = np.array(X_agg, dtype=np.float32)
    X_bag = np.array([[float(b)] for b in bag_sizes], dtype=np.float32)

    # ===== 6. Split =====
    print(f'\n[6] Molecule-level {N_FOLDS}-fold...')
    pair_mols = []
    for a,b in pairs:
        ms=set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams)//N_FOLDS

    # ===== 7. Train all =====
    print(f'\n[7] Training LR-agg + A0 + A1 + A2 on {N_FOLDS} folds...')
    results = {'lr5': [], 'lr_bag': [], 'A0': [], 'A1': [], 'A2': []}
    all_logs = {}

    for k in range(N_FOLDS):
        vs,ve = k*mpf, (k+1)*mpf if k<N_FOLDS-1 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs])|set(ams[ve:])
        tr = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(vm)]

        print(f'\n--- Fold {k} ---')
        # LR-agg 5-feature
        lr5 = Ridge(alpha=1.0); lr5.fit(X_agg[tr], labels[tr])
        r5,_ = pearsonr(labels[va], lr5.predict(X_agg[va])); r5 = max(r5,0); results['lr5'].append(float(r5))
        # LR-agg bag-only
        lrb = Ridge(alpha=1.0); lrb.fit(X_bag[tr], labels[tr])
        rb,_ = pearsonr(labels[va], lrb.predict(X_bag[va])); rb = max(rb,0); results['lr_bag'].append(float(rb))
        print(f'   LR-agg: 5feat={r5:.4f}  bag-only={rb:.4f}')

        # MIL versions
        for mdl_cfg, key in [(A0,'A0'),(A1,'A1'),(A2,'A2')]:
            print(f'   {key}...')
            subdir = out_dir/key; subdir.mkdir(exist_ok=True)
            logs, best_r = train_mil(key, mdl_cfg, tr, va, instances_list, labels, subdir, k)
            results[key].append(float(best_r))
            all_logs[f'{key}_fold{k}'] = logs
            print(f'   {key}: best_r={best_r:.4f}')

    # ===== 8. Save =====
    for ver in results:
        rs = results[ver]
        with open(out_dir/f'{ver}_results.json','w') as f:
            json.dump({'r_mean':float(np.mean(rs)),'r_std':float(np.std(rs)),'r_folds':rs},f,indent=2)

    print(f'\n{sep}')
    print(f'BENCHMARK RESULTS ({N_FOLDS}-fold)')
    print(sep)
    for ver in ['lr5','lr_bag','A0','A1','A2']:
        rs = results[ver]
        print(f'  {ver:12s}: r = {np.mean(rs):.4f} +/- {np.std(rs):.4f}')
    print(f'  Output: {out_dir}')


if __name__ == '__main__':
    main()
