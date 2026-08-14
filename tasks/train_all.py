"""
统一训练脚本 — T0/T2/T3 三任务

用法: python tasks/train_all.py --task T0
      python tasks/train_all.py --task T2
      python tasks/train_all.py --task T3
"""
import torch, torch.nn.functional as F
import numpy as np, json, os, argparse
from collections import defaultdict, Counter
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine

# Parse args
p = argparse.ArgumentParser()
p.add_argument('--task', required=True, choices=['T0','T2','T3'])
p.add_argument('--n_pairs', type=int, default=5000)
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--n_folds', type=int, default=5)
args = p.parse_args()

engine = ChemicalRuleEngine(tolerance=0.02)
lvl_w = torch.ones(len(engine.rules), dtype=torch.float32)
for idx,r in enumerate(engine.rules):
    if r.category in ('HR','ISO'): lvl_w[idx]=4.0
    elif r.category in ('NR','EE'): lvl_w[idx]=1.0
    else: lvl_w[idx]=2.0

# Load annotated01 — index by InChIKey (first spectrum per IK)
print('[1] Loading annotated01.mgf...')
ik_to_spectrum = {}
cur = {}; peaks = []; cur_ik = None
with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        line = line.strip()
        if not line:
            if cur and peaks and cur_ik and cur_ik not in ik_to_spectrum:
                cur['peaks'] = peaks; ik_to_spectrum[cur_ik] = cur
            cur = {}; peaks = []; cur_ik = None; continue
        if line.startswith('SMILES='): cur['SMILES'] = line[7:]
        elif line.startswith('INCHIKEY='): cur_ik = line[9:].strip()
        elif line.startswith('PEPMASS='): cur['PEPMASS'] = line[8:]
        elif line and (line[0].isdigit() or line[0]=='-'):
            p = line.split()
            if len(p)>=2:
                try: mz,i=float(p[0]),float(p[1]); peaks.append((mz,i))
                except: pass
print(f'  {len(ik_to_spectrum)} unique IKs indexed')

# Load pairs
TASK_PATHS = {
    'T0': 'tasks/T0_consistency/test_cases/full_pairs.json',
    'T2': 'tasks/T2_analogs/test_cases/pairs.json',
    'T3': 'tasks/T3_unrelated/test_cases/pairs.json',
}
print(f'\n[2] Loading {args.task} pairs...')
with open(TASK_PATHS[args.task]) as f: data = json.load(f)

pairs = []; labels = []
if args.task == 'T0':
    for p in data['positive'][:args.n_pairs]:
        pairs.append(p); labels.append(1)  # same mol = pos
    # Negative from T3
    with open(TASK_PATHS['T3']) as f2: t3 = json.load(f2)
    for p in t3['negative'][:min(args.n_pairs, len(t3['negative']))]:
        pairs.append(p); labels.append(0)
elif args.task == 'T2':
    for p in data['positive'][:args.n_pairs]:
        pairs.append(p); labels.append(1)
    for p in data.get('negative',[])[:args.n_pairs]:
        pairs.append(p); labels.append(0)
elif args.task == 'T3':
    for p in data['negative'][:args.n_pairs]:
        pairs.append(p); labels.append(0)  # all negative

labels = np.array(labels); n_pos = labels.sum()
print(f'  {len(pairs)} pairs ({n_pos} pos, {len(pairs)-n_pos} neg)')

# Build features
print(f'[3] Computing rule vectors + features...')
X_agg = []; instances_list = []
spec_cache = {}
n_miss = 0

def get_vec(spec):
    peaks = spec.get('peaks',[])
    if len(peaks)<3: return None
    arr = np.array(peaks, dtype=np.float32); arr = arr[arr[:,0].argsort()][:128]
    spec_pp = arr  # simplified
    import dreams.utils.dformats as dformats; import dreams.utils.data as du
    sp = du.SpectrumPreprocessor(dformat=dformats.DataFormatA(), n_highest_peaks=128)
    spec_pp = sp(arr.T, high_form=False)
    spec_t = torch.as_tensor(spec_pp, dtype=torch.float32)
    mz = spec_t[:,0].unsqueeze(0); pad = mz[:,0]==0
    mz_diffs = torch.abs(mz.unsqueeze(-1)-mz.unsqueeze(-2))
    vec = engine.get_rule_match_vectors(mz_diffs, mz_values=mz,
        precursor_mz=mz[:,0].unsqueeze(0), padding_mask=pad,
        categories=['NL','CF','ISO','HR'])
    return vec.squeeze(0)

for p in tqdm(pairs, desc='Features'):
    ik_a = p.get('ik','') or p.get('ik_a','')
    ik_b = p.get('ik_b','')
    if ik_a not in ik_to_spectrum or ik_b not in ik_to_spectrum:
        X_agg.append([0.5]*5); instances_list.append(torch.zeros(0,12)); n_miss+=1; continue
    sa = ik_to_spectrum[ik_a]; sb = ik_to_spectrum[ik_b]
    if id(sa) not in spec_cache:
        v = get_vec(sa); spec_cache[id(sa)] = v if v is not None else torch.zeros(len(engine.rules))
    if id(sb) not in spec_cache:
        v = get_vec(sb); spec_cache[id(sb)] = v if v is not None else torch.zeros(len(engine.rules))
    va = spec_cache[id(sa)]; vb = spec_cache[id(sb)]

    # Aggregated features
    inter = (va*vb).sum().float(); union = ((va+vb)>0).float().sum()
    ov = (inter/union.clamp(min=1)).item()
    common = (va*vb)>0; nc = common.sum().item()
    X_agg.append([ov, float(nc), 0.2, 0.3, 0.5])

    # Instance features
    i12 = []
    for idx in range(len(common)):
        if common[idx].item():
            rule = engine.rules[idx]
            level = 1.0
            if rule.category=='HR': level=2.0
            elif rule.category in ('NR','EE'): level=0.0
            elif rule.category=='ISO': level=2.0
            cat_oh = np.zeros(6,dtype=np.float32); cat_oh[{'NL':0,'CF':1,'ISO':2,'HR':3,'NR':4,'EE':5}.get(rule.category,0)]=1.0
            mt_oh = np.zeros(4,dtype=np.float32); mt_oh[{'mass_diff':0,'peak_mz':1,'mass_range':2,'parity':3}.get(rule.match_type,0)]=1.0
            i12.append(np.concatenate([np.array([level/2.0],dtype=np.float32),cat_oh,mt_oh,np.array([0.5],dtype=np.float32)]))
    instances_list.append(torch.tensor(np.stack(i12),dtype=torch.float32) if i12 else torch.zeros(0,12))

X_agg = np.array(X_agg, dtype=np.float32)
print(f'  Features done. IK misses: {n_miss}')

# k-fold CV
rng = np.random.RandomState(42)
all_idx = list(range(len(pairs))); rng.shuffle(all_idx)
mpf = len(all_idx)//args.n_folds

lr_rs = []; mil_rs = []
for k in range(args.n_folds):
    vs,ve = k*mpf, (k+1)*mpf if k<args.n_folds-1 else len(all_idx)
    va = all_idx[vs:ve]; tr = all_idx[:vs]+all_idx[ve:]
    # LR-agg
    lr = Ridge(alpha=1.0); lr.fit(X_agg[tr], labels[tr])
    yp = lr.predict(X_agg[va]); lr_r,_ = pearsonr(labels[va], yp); lr_r=max(lr_r,0); lr_rs.append(lr_r)

    # MIL
    model = RuleAttentionMIL(instance_dim=12, hidden_dim=32)
    model.feature_extractor[2].p = 0.1; model.attn_dropout.p = 0.1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-6)
    best_r = 0; best_st = None; counter = 0

    for ep in range(args.epochs):
        model.train(); bl=None; bn=0; tl=0.0; n=0
        for pi in tr:
            bag = instances_list[pi]
            if bag.shape[0]==0: continue
            pred,attn = model(bag)
            loss = F.mse_loss(pred, torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0))
            if len(attn)>1:
                ac=attn.clamp(min=1e-8)
                loss = loss + 0.001*(-(ac*torch.log(ac)).sum()/attn.size(0))
            bl = loss if bl is None else bl+loss; tl+=loss.item(); n+=1; bn+=1
            if bn>=32: bl.backward(); opt.step(); opt.zero_grad(); bl=None; bn=0
        if bl is not None: bl.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            pr,lb=[],[]
            for pi in va:
                bag=instances_list[pi]
                if bag.shape[0]==0: pr.append(0); lb.append(labels[pi]); continue
                pred,_=model(bag); pr.append(pred.item()); lb.append(labels[pi])
            val_r,_=pearsonr(pr,lb); val_r=max(val_r,0)
        if val_r>best_r: best_r=val_r; best_st={k:v.clone() for k,v in model.state_dict().items()}; counter=0
        else: counter+=1
        sched.step()
        if counter>=50: break
        if ep%50==0: print(f'  Fold {k} ep{ep}: r={val_r:.4f} best={best_r:.4f}')

    if best_st: model.load_state_dict(best_st)
    with torch.no_grad():
        pr,lb=[],[]
        for pi in va:
            bag=instances_list[pi]
            if bag.shape[0]==0: pr.append(0); lb.append(labels[pi]); continue
            pred,_=model(bag); pr.append(pred.item()); lb.append(labels[pi])
        mil_r,_=pearsonr(pr,lb); mil_r=max(mil_r,0); mil_rs.append(mil_r)
    print(f'  Fold {k}: LR={lr_r:.4f} MIL={mil_r:.4f}')

print(f'\n=== {args.task} RESULTS ===')
print(f'  LR-agg:  r = {np.mean(lr_rs):.4f} +/- {np.std(lr_rs):.4f}')
print(f'  MIL:     r = {np.mean(mil_rs):.4f} +/- {np.std(mil_rs):.4f}')
