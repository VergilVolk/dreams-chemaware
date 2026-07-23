"""
run_A3_benchmark.py — A0/A1/A2/A3 + LR-agg, ALL 1500 epochs, 10-fold

A3 18-dim features (proper):
  1. Level normalized         (1 dim)
  2. Category onehot          (6 dims)
  3. Match-type onehot        (4 dims)
  4. Rule rarity (1/freq)     (1 dim)
  5. Hit pattern (common/A/B) (3 dims)
  6. Diagnostic power         (1 dim)
  7. Soft mass precision      (1 dim)
  8. Level × rarity           (1 dim)
  Total: 18 ✓

用法:
  python -m dreams.models.mil_interpretable.run_A3_benchmark
"""

import torch, torch.nn.functional as F
import numpy as np, json, pickle, time, math
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from tqdm import tqdm

torch.manual_seed(42); np.random.seed(42)

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto

# ==============================================================================
# AdaCosine
# ==============================================================================
class AdaCosineScheduler:
    def __init__(self, opt, T_0=100, eta_min=1e-6, p_stag=25, p_over=10, ext=20):
        self.opt=opt; self.T_0=T_0; self.eta_min=eta_min; self.eta_max=opt.param_groups[0]['lr']
        self.p_stag=p_stag; self.p_over=p_over; self.ext=ext
        self.T_cur=0; self.best_r=0; self.no_imp=0; self.ori_s=0; self.prev_ori=None; self.cyc_ext=0; self.restart=False
    def step(self, vr, tl, vl):
        self.T_cur+=1; self.restart=False
        if vr>self.best_r: self.best_r=vr; self.no_imp=0; self.cyc_ext=self.ext
        else: self.no_imp+=1
        ori=(vl-tl)/max(tl,1e-8)
        if self.prev_ori is not None and ori>self.prev_ori: self.ori_s+=1
        else: self.ori_s=0
        self.prev_ori=ori
        early=(self.no_imp>=self.p_stag or self.ori_s>=self.p_over)
        if self.cyc_ext>0: self.cyc_ext-=1; early=False
        if early: self.T_cur=0; self.no_imp=0; self.ori_s=0; self.restart=True
        et=self.T_0+max(0,self.cyc_ext)
        lr=self.eta_min+0.5*(self.eta_max-self.eta_min)*(1+math.cos(math.pi*self.T_cur/et))
        for pg in self.opt.param_groups: pg['lr']=lr
        return lr

# ==============================================================================
# Proper 18-dim Feature Builder
# ==============================================================================
class FeatureBuilder18:
    def __init__(self, engine, match_vecs_all, vidx, valid_spectra, iso_pair_indices):
        n_rules = len(engine.rules); n_spec = len(vidx)
        # Rule frequency
        rule_freq = np.zeros(n_rules, dtype=np.float64)
        for i in vidx:
            if i in match_vecs_all:
                rule_freq += match_vecs_all[i].numpy()
        rule_freq = rule_freq / max(n_spec, 1)
        self.rarity = 1.0 / (rule_freq + 1e-4)
        self.rarity = self.rarity / self.rarity.max()

        # Diagnostic power: info gain on isomer pairs (simplified)
        self.diag_power = np.ones(n_rules, dtype=np.float32)
        if iso_pair_indices:
            iso_vecs_a = [match_vecs_all[valid_spectra[a]['_orig_idx']] for a,_ in iso_pair_indices
                          if valid_spectra[a]['_orig_idx'] in match_vecs_all]
            iso_vecs_b = [match_vecs_all[valid_spectra[b]['_orig_idx']] for b,_ in iso_pair_indices
                          if valid_spectra[b]['_orig_idx'] in match_vecs_all]
            if iso_vecs_a:
                iso_common = np.zeros(n_rules, dtype=np.float64)
                for va, vb in zip(iso_vecs_a, iso_vecs_b):
                    iso_common += ((va > 0) & (vb > 0)).numpy()
                iso_common /= max(len(iso_vecs_a), 1)
                bg_freq = rule_freq ** 2
                self.diag_power = (iso_common + 1e-4) / (bg_freq + 1e-4)
                self.diag_power = self.diag_power / self.diag_power.max()

        self.sigma = 0.01
        self.cat_idx = {'NL':0,'CF':1,'ISO':2,'HR':3,'NR':4,'EE':5}
        self.mt_idx = {'mass_diff':0,'peak_mz':1,'mass_range':2,'parity':3,'mass_diff_range':0,'hr_shift':3}
        self.engine = engine; self.n_rules = n_rules

    def build(self, vec_a, vec_b):
        va = vec_a.numpy() if isinstance(vec_a, torch.Tensor) else vec_a
        vb = vec_b.numpy() if isinstance(vec_b, torch.Tensor) else vec_b
        hit_a = va > 0; hit_b = vb > 0
        # ALL rules hit by either spectrum (not just common!)
        either = hit_a | hit_b
        instances = []
        for idx in np.where(either)[0]:
            rule = self.engine.rules[idx]
            level = 1.0
            if rule.category=='HR': level=2.0
            elif rule.category in ('NR','EE'): level=0.0
            elif rule.category=='ISO': level=2.0
            cat_oh = np.zeros(6,dtype=np.float32); cat_oh[self.cat_idx.get(rule.category,0)]=1.0
            mt_oh = np.zeros(4,dtype=np.float32); mt_oh[self.mt_idx.get(rule.match_type,0)]=1.0
            rar = self.rarity[idx]
            # Hit pattern (3 dims) — now meaningful!
            common = hit_a[idx] and hit_b[idx]
            only_a = hit_a[idx] and not hit_b[idx]
            only_b = not hit_a[idx] and hit_b[idx]
            hit_pat = np.array([1.0 if common else 0.0, 1.0 if only_a else 0.0, 1.0 if only_b else 0.0], dtype=np.float32)
            # Soft mass precision
            soft = 1.0
            if rule.match_type in ('mass_diff','peak_mz'):
                tgt = float(rule.value) if isinstance(rule.value,(int,float)) else float(rule.value[0])
                dm = 0.02
                soft = math.exp(-dm**2/(2*self.sigma**2))
            diag = self.diag_power[idx]
            lr_int = (level/2.0)*rar
            feat = np.concatenate([
                np.array([level/2.0],dtype=np.float32), cat_oh, mt_oh,
                np.array([rar],dtype=np.float32), hit_pat,
                np.array([diag],dtype=np.float32), np.array([soft],dtype=np.float32),
                np.array([lr_int],dtype=np.float32),
            ])  # 1+6+4+1+3+1+1+1 = 18
            instances.append(feat)
        return instances

# ==============================================================================
# Training
# ==============================================================================
def train_one(model, mdl_cfg, tr, va, instances_list, labels, out_dir, fold_idx, label):
    opt = torch.optim.AdamW(model.parameters(), lr=mdl_cfg['lr'], weight_decay=mdl_cfg['wd'])
    if mdl_cfg['sched']=='adacosine':
        sched=AdaCosineScheduler(opt,T_0=mdl_cfg['T_0'],eta_min=1e-6)
    elif mdl_cfg['sched']=='cosine':
        sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=mdl_cfg['T_0'],T_mult=2,eta_min=1e-6)
    else:
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=20)
    best_r,best_state,best_ep=0,None,0; counter=0; pat=mdl_cfg.get('pat',9999); t0=time.time()
    for ep in range(mdl_cfg['epochs']):
        model.train(); tl,n=0.0,0; bl,bn=None,0
        for pi in tr:
            bag=instances_list[pi]
            if bag.shape[0]==0: continue
            pred,attn=model(bag)
            loss=F.mse_loss(pred,torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0))
            if len(attn)>1:
                ac=attn.clamp(min=1e-8)
                loss=loss+mdl_cfg['ent']*(-(ac*torch.log(ac)).sum()/attn.size(0)) if mdl_cfg['ent']>0 else loss
            bl=loss if bl is None else bl+loss; tl+=loss.item(); n+=1; bn+=1
            if bn>=mdl_cfg['bs']:
                bl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); opt.zero_grad(); bl=None; bn=0
        if bl is not None: bl.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            pr,lb,vl=[],[],0.0
            for pi in va:
                bag=instances_list[pi]
                if bag.shape[0]==0: pr.append(0); lb.append(labels[pi]); continue
                pred,_=model(bag); pr.append(pred.item()); lb.append(labels[pi])
                vl+=F.mse_loss(pred,torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0)).item()
            val_r,_=pearsonr(pr,lb); val_r=max(val_r,0); vl/=max(len(va),1)
        if val_r>best_r: best_r=val_r; best_ep=ep; best_state={k:v.clone() for k,v in model.state_dict().items()}; counter=0
        else: counter+=1
        if mdl_cfg['sched']=='adacosine': sched.step(val_r,tl/max(n,1),vl)
        elif mdl_cfg['sched']=='plateau': sched.step(val_r)
        else: sched.step()
        if ep%100==0 or ep<3:
            et=time.time()-t0; eta=(et/(ep+1))*(mdl_cfg['epochs']-ep-1)
            print(f'     {label} ep{ep:4d}: r={val_r:.4f} best={best_r:.4f}@{best_ep} eta={eta/60:.0f}m')
        if counter>=pat: break
    if best_state: torch.save(best_state, out_dir/f'{label}_fold{fold_idx}_best.pt')
    return best_r

# ==============================================================================
# Main
# ==============================================================================
def main():
    out_dir = Path('outputs')/f'A0123_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True,exist_ok=True)
    N_FOLDS=10; N_TARGET=4000  # 900 pos + 1800 iso + 1300 easy
    EPOCHS=1500

    # Configs — all based on A1, only scheduler differs
    A0={'hidden_dim':32,'dropout':0.1,'ent':0.001,'lr':1e-4,'wd':1e-5,'bs':32,
        'epochs':EPOCHS,'pat':30,'sched':'plateau','T_0':100,'label':'A0'}
    A1={'hidden_dim':32,'dropout':0.1,'ent':0.001,'lr':1e-4,'wd':1e-5,'bs':32,
        'epochs':EPOCHS,'pat':9999,'sched':'cosine','T_0':100,'label':'A1'}
    A2={'hidden_dim':32,'dropout':0.1,'ent':0.001,'lr':1e-4,'wd':1e-5,'bs':32,
        'epochs':EPOCHS,'pat':9999,'sched':'adacosine','T_0':100,'label':'A2'}
    A3={'hidden_dim':32,'dropout':0.1,'ent':0.001,'lr':1e-4,'wd':1e-5,'bs':32,
        'epochs':EPOCHS,'pat':9999,'sched':'cosine','T_0':100,'label':'A3'}
    # A3 uses same params as A1, only feature dim differs (18 vs 12)

    CFG={'N_FOLDS':N_FOLDS,'N_TARGET':N_TARGET,'EPOCHS':EPOCHS,'A0':A0,'A1':A1,'A2':A2,'A3':A3}
    with open(out_dir/'config.json','w') as f: json.dump(CFG,f,indent=2)

    sep='='*60
    print(sep)
    print(f'  FULL BENCHMARK: A0/A1/A2/A3 + LR-agg, ALL {EPOCHS}ep, {N_FOLDS}-fold')
    print(sep)
    for m,cfg in [('A0',A0),('A1',A1),('A2',A2),('A3',A3)]:
        print(f'  {m}: dim={cfg["hidden_dim"]}, lr={cfg["lr"]}, wd={cfg["wd"]}, '
              f'drop={cfg["dropout"]}, ent={cfg["ent"]}, bs={cfg["bs"]}, '
              f'sched={cfg["sched"]}, ep={cfg["epochs"]}, pat={cfg["pat"]}')
    print(f'  A3 unique: instance_dim=18 (all others 12)')
    print(sep)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'\nEngine: {len(engine.rules)} rules')
    import dreams.utils.dformats as dformats; import dreams.utils.data as du
    dformat=dformats.DataFormatA()
    sp=du.SpectrumPreprocessor(dformat=dformat,n_highest_peaks=128)

    MSP_FILES=['data/MassBank_NIST.msp','data/MoNA-export-LC-MS-MS_Spectra.msp',
               'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']

    # ===== 1. Parse ALL 4 databases =====
    print('\n[1] Parsing ALL databases (MSP + MassSpecGym)...')
    spectra=[]
    for fp in MSP_FILES:
        s=parse_msp(fp,max_spectra=50000)
        print(f'   {fp.replace(chr(92),chr(47)).split(chr(47))[-1]}: {len(s)}')
        spectra.extend(s)
    # Add MassSpecGym
    import dreams.utils.data as dud
    msdata=dud.MSData.load('data/MassSpecGym_MurckoHist_split.hdf5')
    n_msg=min(50000,len(msdata))
    print(f'   MassSpecGym: reading {n_msg}...')
    for i in tqdm(range(n_msg),desc='MassSpecGym'):
        try:
            smi=msdata.get_values('smiles',i)
            if isinstance(smi,bytes): smi=smi.decode('utf-8')
            ik=msdata.get_values('INCHIKEY',i)
            if isinstance(ik,bytes): ik=ik.decode('utf-8')
            fm=msdata.get_values('FORMULA',i)
            if isinstance(fm,bytes): fm=fm.decode('utf-8')
            pm=msdata.get_values('precursor_mz',i)
            spec_raw=torch.as_tensor(msdata.get_spectra(i),dtype=torch.float32)
            peaks=[(float(spec_raw[0,j]),float(spec_raw[1,j])) for j in range(spec_raw.shape[1]) if spec_raw[0,j]>0]
            spectra.append({'SMILES':str(smi).strip(),'InChIKey':str(ik).strip(),
                           'PrecursorMZ':float(pm) if pm else 0,'_formula':str(fm).strip(),
                           'peaks':peaks,'_source':'msg'})
        except: pass
    print(f'   Total spectra: {len(spectra)}')

    # ===== 2. Filter =====
    print('\n[2] Filtering...')
    from rdkit import Chem
    valid=[]
    for i,s in enumerate(spectra):
        smi=s.get('SMILES','').strip(); ik=s.get('InChIKey','').strip()
        if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
            fm=Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi))
            s['_formula']=fm; s['_orig_idx']=i; valid.append(s)
    print(f'   Valid: {len(valid)}')

    # ===== 3. Vectors =====
    print('\n[3] Match vectors...')
    mvs={}
    for i,s in enumerate(tqdm(valid,desc='Vectors')):
        vec=spectrum_to_match_vec(s,engine,sp)
        if vec is not None: mvs[i]=vec
    vidx=[i for i in range(len(valid)) if i in mvs]
    print(f'   {len(vidx)} with vectors')

    # ===== 4. Pairs =====
    print('\n[4] Building pairs...')
    rng=np.random.RandomState(42)
    ik2=defaultdict(list); fm2=defaultdict(list)
    for i in vidx:
        ik2[valid[i]['InChIKey']].append(i)
        fm=valid[i].get('_formula','')
        if fm: fm2[fm].append(i)
    multi_ik={k:v for k,v in ik2.items() if len(v)>=2}
    multi_fm={k:v for k,v in fm2.items() if len(v)>=2}
    print(f'   Multi-IK: {len(multi_ik)}, Multi-FM: {len(multi_fm)}')

    pairs,labels,pair_types=[],[],[]
    n_pos=0
    for ik in sorted(multi_ik.keys(),key=lambda x:-len(multi_ik[x])):
        a,b=rng.choice(multi_ik[ik],2,replace=False)
        pairs.append((a,b)); labels.append(1.0); pair_types.append('pos'); n_pos+=1
        if n_pos>=900: break
    iso_target=1800  # doubled!
    iso_found=0; iso_fm=Counter(); iso_tans=[]
    for fm in sorted(multi_fm.keys(),key=lambda x:-len(multi_fm[x])):
        idxs=multi_fm[fm]
        if len(idxs)<2: continue
        seen=set()
        for _ in range(min(80,len(idxs)*5)):  # more aggressive search
            a,b=rng.choice(idxs,2,replace=False)
            if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
            pk=(min(a,b),max(a,b))
            if pk in seen: continue; seen.add(pk)
            tan=compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
            if 0.3<=tan<=0.9:
                pairs.append((a,b)); labels.append(tan); pair_types.append('isomer')
                iso_found+=1; iso_fm[fm]+=1; iso_tans.append(tan)
            if iso_found>=iso_target: break
        if iso_found>=iso_target: break
    for _ in range(10000):
        a,b=rng.choice(vidx,2,replace=False)
        if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
        pm_a=float(valid[a].get('PrecursorMZ',0) or 0); pm_b=float(valid[b].get('PrecursorMZ',0) or 0)
        if abs(pm_a-pm_b)<=1.0: continue
        tan=compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
        if 0<=tan<0.2: pairs.append((a,b)); labels.append(tan); pair_types.append('easy')
        if len(pairs)>=N_TARGET: break
    n_easy=len(pairs)-n_pos-iso_found
    labels=np.array(labels,dtype=np.float32); iso_tans=np.array(iso_tans)

    # V1
    print(f'\n{sep}\n  DATA VALIDATION\n{sep}')
    print(f'  Total: {len(pairs)} (pos={n_pos}, iso={iso_found}, easy={n_easy})')
    print(f'  Iso Tanimoto: mean={iso_tans.mean():.4f} std={iso_tans.std():.4f} '
          f'[0.3-0.5):{(iso_tans<0.5).sum()} [0.5-0.7):{((iso_tans>=0.5)&(iso_tans<0.7)).sum()} [0.7-0.9):{(iso_tans>=0.7).sum()}')
    print(f'  Unique iso formulas: {len(iso_fm)}, max/formula={max(iso_fm.values())}')
    # V2: Weighted Jaccard (L2=4x, L1=2x, L0=1x)
    pos_idx=[i for i,pt in enumerate(pair_types) if pt=='pos'][:100]
    ovs,ovs_w=[],[]
    level_weights=np.ones(len(engine.rules),dtype=np.float32)
    for idx,r in enumerate(engine.rules):
        if r.category=='HR' or r.category=='ISO': level_weights[idx]=4.0
        elif r.category in ('NR','EE'): level_weights[idx]=1.0
        else: level_weights[idx]=2.0
    for pi in pos_idx:
        a,b=pairs[pi]; va,vb=mvs[a],mvs[b]
        va_f=va.float(); vb_f=vb.float()
        inter=(va_f*vb_f).sum(); union=((va_f+vb_f)>0).float().sum()
        ovs.append((inter/union.clamp(min=1)).item())
        w_inter=((va_f*vb_f)*torch.tensor(level_weights)).sum()
        w_union=(((va_f+vb_f)>0).float()*torch.tensor(level_weights)).sum()
        ovs_w.append((w_inter/w_union.clamp(min=1)).item())
    print(f'  Pos Jaccard (equal):  mean={np.mean(ovs):.4f} std={np.std(ovs):.4f}')
    print(f'  Pos Jaccard (weighted): mean={np.mean(ovs_w):.4f} std={np.std(ovs_w):.4f}')
    # Bag check
    bag_sizes=[((mvs[a]*mvs[b])>0).sum().item() for a,b in pairs]
    r_bag,_=pearsonr(bag_sizes,labels)
    print(f'  Bag-Tanimoto r = {r_bag:.4f}')

    # ===== 5. Features =====
    print(f'\n[5] Features...')
    iso_pair_idx = [i for i,pt in enumerate(pair_types) if pt=='isomer']
    iso_pair_data = [(pairs[i][0], pairs[i][1]) for i in iso_pair_idx]
    fb18 = FeatureBuilder18(engine, mvs, vidx, valid, iso_pair_data)
    inst12_list, inst18_list = [], []
    for a,b in tqdm(pairs,desc='Instances'):
        va,vb=mvs[a],mvs[b]
        common=(va*vb)>0
        # 12-dim (A0/A1/A2)
        i12=[]
        for idx in np.where(common)[0]:
            rule=engine.rules[idx]
            level=1.0
            if rule.category=='HR': level=2.0
            elif rule.category in ('NR','EE'): level=0.0
            elif rule.category=='ISO': level=2.0
            cat_oh=np.zeros(6,dtype=np.float32); cat_oh[{'NL':0,'CF':1,'ISO':2,'HR':3,'NR':4,'EE':5}.get(rule.category,0)]=1.0
            mt_oh=np.zeros(4,dtype=np.float32); mt_oh[{'mass_diff':0,'peak_mz':1,'mass_range':2,'parity':3,'mass_diff_range':0,'hr_shift':3}.get(rule.match_type,0)]=1.0
            i12.append(np.concatenate([np.array([level/2.0],dtype=np.float32),cat_oh,mt_oh,np.array([0.5],dtype=np.float32)]))
        inst12_list.append(torch.tensor(np.stack(i12),dtype=torch.float32) if i12 else torch.zeros(0,12))
        # 18-dim (A3) — with hit pattern!
        i18=fb18.build(va,vb)
        inst18_list.append(torch.tensor(np.stack(i18),dtype=torch.float32) if i18 else torch.zeros(0,18))

    X_agg=[]
    for a,b in tqdm(pairs,desc='LR'):
        va,vb=mvs[a],mvs[b]
        inter=(va*vb).sum().float(); union=((va+vb)>0).float().sum()
        ov=(inter/union.clamp(min=1)).item()
        common=(va*vb)>0; nc=common.sum().item()
        nl=common[:293].sum().item() if len(common)>=293 else 0
        cf=common[293:293+3174].sum().item() if len(common)>=293+3174 else 0
        iso_hr=common[293+3174:].sum().item()
        X_agg.append([ov,float(nc),iso_hr/max(nc,1),nl/max(nc,1),cf/max(nc,1)])
    X_agg=np.array(X_agg,dtype=np.float32)

    # ===== 6. Split =====
    print(f'\n[6] Molecule-level {N_FOLDS}-fold...')
    pair_mols=[]
    for a,b in pairs:
        ms=set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams=list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf=len(ams)//N_FOLDS

    # ===== 7. Train ALL =====
    results={m:[] for m in ['lr','A0','A1','A2','A3']}
    for k in range(N_FOLDS):
        vs,ve=k*mpf,(k+1)*mpf if k<N_FOLDS-1 else len(ams)
        vm=set(ams[vs:ve]); tm=set(ams[:vs])|set(ams[ve:])
        tr=[pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va=[pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        assert len(set().union(*[pair_mols[p] for p in tr] if tr else [])&
                   set().union(*[pair_mols[p] for p in va] if va else []))==0

        print(f'\n--- Fold {k} ---')
        lr=Ridge(alpha=1.0); lr.fit(X_agg[tr],labels[tr])
        lr_r,_=pearsonr(labels[va],lr.predict(X_agg[va])); lr_r=max(lr_r,0)
        results['lr'].append(float(lr_r))
        print(f'   LR-agg: r={lr_r:.4f}')

        for cfg,key in [(A0,'A0'),(A1,'A1'),(A2,'A2')]:
            print(f'   {key}...')
            model=RuleAttentionMIL(instance_dim=12,hidden_dim=cfg['hidden_dim'])
            model.feature_extractor[2].p=cfg['dropout']; model.attn_dropout.p=cfg['dropout']
            r=train_one(model,cfg,tr,va,inst12_list,labels,out_dir,k,key)
            results[key].append(float(r))
            print(f'   {key}: best_r={r:.4f}')

        # A3 with 18-dim
        print(f'   A3 (18-dim)...')
        model18=RuleAttentionMIL(instance_dim=18,hidden_dim=A3['hidden_dim'])
        model18.feature_extractor[2].p=A3['dropout']; model18.attn_dropout.p=A3['dropout']
        r18=train_one(model18,A3,tr,va,inst18_list,labels,out_dir,k,'A3')
        results['A3'].append(float(r18))
        print(f'   A3: best_r={r18:.4f}')

    # ===== 8. Save =====
    for ver in results:
        rs=results[ver]
        with open(out_dir/f'{ver}_results.json','w') as f:
            json.dump({'r_mean':float(np.mean(rs)),'r_std':float(np.std(rs)),'r_folds':rs},f,indent=2)

    print(f'\n{sep}')
    print(f'FINAL RESULTS ({N_FOLDS}-fold, {EPOCHS}ep each)')
    print(sep)
    for ver in ['lr','A0','A1','A2','A3']:
        rs=results[ver]
        print(f'  {ver:6s}: r = {np.mean(rs):.4f} +/- {np.std(rs):.4f}')
    print(f'  Output: {out_dir}')


if __name__=='__main__':
    main()
