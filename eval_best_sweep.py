"""
快速评估 sweep 最佳模型 — 只做推理，不训练
用法: python eval_best_sweep.py
"""
import torch, json, numpy as np, os, sys, time
from collections import defaultdict
from argparse import Namespace
import torch.nn.functional as F
from sklearn import metrics

sys.path.insert(0, '.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ---- 1. Load base + fine-tuned models ----
print('[1] Loading models...')
pkg = torch.load('dreams/models/pretrained/ssl_model_server.pt', map_location='cpu', weights_only=False)
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor
from dreams.models.dreams.dreams import DreaMS

recon_args = Namespace(**pkg['args'])
recon_args.dformat = DataFormatA()
for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge','max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
    if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
recon_args.d_graphormer_params = 0

def build_model():
    sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
    m = DreaMS(recon_args, sp)
    state = m.state_dict()
    for k in state:
        if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
            state[k] = pkg['state_dict'][k].clone()
    m.load_state_dict(state, strict=False)
    return m.eval().to(device)

model_base = build_model()

ft = torch.load('triplet_sweep/v5_experience/best.pt', map_location='cpu', weights_only=False)
h = ft['history']
best_epoch = h['val_auc'].index(max(h['val_auc'])) + 1 if h.get('val_auc') else h['val_sep'].index(max(h['val_sep'])) + 1
print(f'  Fine-tuned: epoch={ft["epoch"]+1}  val_sep={ft["val_sep"]:.4f}  val_acc={ft.get("val_acc",0):.3f}')

model_ft = build_model()
ft_state = model_ft.state_dict()
for k in ft_state:
    if k in ft['model_state_dict'] and ft_state[k].shape == ft['model_state_dict'][k].shape:
        ft_state[k] = ft['model_state_dict'][k].clone()
model_ft.load_state_dict(ft_state, strict=False)
model_ft.eval().to(device)

# ---- 2. Build proper AUC eval set (multi-spectrum same-mol) ----
print('[2] Building AUC eval set...')
N_PEAKS = 128
def peaks_to_tensor(peaks):
    arr = np.array(peaks, dtype=np.float32); arr = arr[arr[:,0].argsort()]
    if len(arr) > N_PEAKS:
        idx = np.argpartition(arr[:,1], -N_PEAKS)[-N_PEAKS:]; arr = arr[idx]; arr = arr[arr[:,0].argsort()]
    max_i = arr[:,1].max()
    if max_i > 0: arr[:,1] /= max_i
    padded = np.zeros((N_PEAKS,2), dtype=np.float32)
    n = min(len(arr), N_PEAKS); padded[:n] = arr[:n]
    return torch.from_numpy(padded)

# Scan annotated01 for all multi-spectrum IKs
ik_all_peaks = {}
cur_ik=None; cur_peaks=[]
with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        line=line.strip()
        if not line:
            if cur_ik and len(cur_peaks)>=3:
                if cur_ik not in ik_all_peaks: ik_all_peaks[cur_ik]=[]
                ik_all_peaks[cur_ik].append(cur_peaks[:])
            cur_ik=None; cur_peaks=[]; continue
        if line.startswith('INCHIKEY='): cur_ik=line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
            p2=line.split()
            if len(p2)>=2:
                try:
                    mz,i=float(p2[0]),float(p2[1])
                    if mz>0 and i>0: cur_peaks.append((mz,i))
                except: pass

multi_iks = {ik:pks for ik,pks in ik_all_peaks.items() if len(pks)>=2}
print(f'  {len(multi_iks)} IKs with >=2 spectra')

# Build AUC spectra + pairs — sample max 2000 IKs to keep fast
rng = np.random.RandomState(42)
n_auc_iks = 2000
sampled_iks = rng.choice(sorted(multi_iks.keys()), min(n_auc_iks, len(multi_iks)), replace=False)

auc_specs = []; auc_ik_to_idx = {}
for ik in sampled_iks:
    auc_ik_to_idx[ik] = []
    for pk in multi_iks[ik][:3]:
        t = peaks_to_tensor(pk)
        if t is not None:
            auc_ik_to_idx[ik].append(len(auc_specs)); auc_specs.append(t)

multi_ik_list = [ik for ik in sampled_iks if len(auc_ik_to_idx[ik])>=2]
all_auc_iks = [ik for ik in auc_ik_to_idx if len(auc_ik_to_idx[ik])>=1]
n_each = 2000
pair_i, pair_j, labels = [], [], []

n_pos=0
while n_pos < n_each and multi_ik_list:
    ik = rng.choice(multi_ik_list)
    idxs = auc_ik_to_idx[ik]
    if len(idxs)>=2:
        a,b = rng.choice(idxs,2,replace=False)
        pair_i.append(a); pair_j.append(b); labels.append(1); n_pos+=1

n_neg=0
while n_neg < n_each and len(all_auc_iks)>=2:
    ika,ikb = rng.choice(all_auc_iks,2,replace=False)
    if ika==ikb: continue
    a=rng.choice(auc_ik_to_idx[ika]); b=rng.choice(auc_ik_to_idx[ikb])
    pair_i.append(a); pair_j.append(b); labels.append(0); n_neg+=1

pair_i=np.array(pair_i); pair_j=np.array(pair_j); labels=np.array(labels)
print(f'  AUC set: {n_pos}P + {n_neg}N = {len(pair_i)} pairs, {len(auc_specs)} spectra')

# ---- 3. Also evaluate on T1 triplet validation ----
print('[3] Loading T1 validation triplets...')
with open('tasks/T1_near_isomers/test_cases/triplets_val.json') as f:
    val_trip = json.load(f)
def ik14(x): return x[:14]
val_trip = [(ik14(t['anchor_ik']), ik14(t['pos_ik']), ik14(t['neg_ik'])) for t in val_trip]

# Load spectra for triplet IKs
needed = set()
for a,p,n in val_trip:
    needed.add(a); needed.add(p); needed.add(n)
ik_to_spec = {}
for ik, pks in ik_all_peaks.items():
    if ik in needed:
        ik_to_spec[ik] = peaks_to_tensor(pks[0])  # first spectrum per IK
val_trip = [(a,p,n) for a,p,n in val_trip if a in ik_to_spec and p in ik_to_spec and n in ik_to_spec]
print(f'  {len(val_trip)} valid triplets')

# ---- 4. Compute embeddings (batched) ----
print('[4] Computing embeddings...')

@torch.no_grad()
def get_embs(model, specs):
    embs = []
    for start in range(0, len(specs), 32):
        batch = torch.stack(specs[start:start+32]).to(device)
        embs.append(model(batch, None)[:,0,:].cpu())
    return torch.cat(embs, dim=0)

t0 = time.time()
# AUC embeddings
auc_embs_base = get_embs(model_base, auc_specs)
auc_embs_ft = get_embs(model_ft, auc_specs)

# Triplet embeddings
tri_specs = [ik_to_spec[ik] for ik in ik_to_spec]
tri_iks = list(ik_to_spec.keys())
ik_to_idx = {ik:i for i,ik in enumerate(tri_iks)}
tri_embs_base = get_embs(model_base, tri_specs)
tri_embs_ft = get_embs(model_ft, tri_specs)
print(f'  Embeddings done ({time.time()-t0:.0f}s)')

# ---- 5. AUC evaluation ----
print('[5] Computing AUC...')
for name, auc_embs in [('Pretrained', auc_embs_base), ('Fine-tuned (5ep)', auc_embs_ft)]:
    cos_sims = F.cosine_similarity(auc_embs[pair_i], auc_embs[pair_j], dim=-1).numpy()
    fpr,tpr,_ = metrics.roc_curve(labels, cos_sims)
    auc = float(metrics.auc(fpr, tpr))
    cp = cos_sims[labels==1].mean(); cn = cos_sims[labels==0].mean()
    print(f'  {name}: AUC={auc:.4f}  cos+={cp:.4f}  cos-={cn:.4f}  sep={cp-cn:.4f}')

# ---- 6. Triplet evaluation ----
print('\n[6] Triplet metrics...')
for name, tri_embs in [('Pretrained', tri_embs_base), ('Fine-tuned (5ep)', tri_embs_ft)]:
    seps = []; correct = 0
    for a,p,n in val_trip:
        ea = tri_embs[ik_to_idx[a]]; ep = tri_embs[ik_to_idx[p]]; en = tri_embs[ik_to_idx[n]]
        cp = F.cosine_similarity(ea.unsqueeze(0), ep.unsqueeze(0), dim=-1).item()
        cn = F.cosine_similarity(ea.unsqueeze(0), en.unsqueeze(0), dim=-1).item()
        seps.append(cp-cn); correct += (cp>cn)
    print(f'  {name}: sep={np.mean(seps):.4f}  acc={correct/len(val_trip):.4f}')

# ---- 7. Summary ----
print(f'\n{"="*60}')
print(f'MODULE 1 T1 TRIPLET — FINAL RESULTS')
print(f'{"="*60}')
print(f'  Model: DreaMS + triplet fine-tuning (α=0.05 β=0.02 margin=0.2)')
print(f'  Training: 5 epochs, 3000 triplets')
print(f'  AUC eval: {len(pair_i)} pairs ({n_pos}P+{n_neg}N), {len(auc_specs)} spectra')
print(f'  Triplet val: {len(val_trip)} triplets')
print(f'{"="*60}')
