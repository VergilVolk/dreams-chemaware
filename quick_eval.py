"""极速评估: 从已有checkpoint直接算AUC+Triplet"""
import torch, json, numpy as np, time
from collections import defaultdict
from argparse import Namespace
import torch.nn.functional as F
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# Load model (same as before)
pkg = torch.load('dreams/models/pretrained/ssl_model_server.pt', map_location='cpu', weights_only=False)
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor
from dreams.models.dreams.dreams import DreaMS

recon_args = Namespace(**pkg['args'])
recon_args.dformat = DataFormatA()
for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge','max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
    if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
recon_args.d_graphormer_params = 0

def build():
    sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
    m = DreaMS(recon_args, sp); state = m.state_dict()
    for k in state:
        if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
            state[k] = pkg['state_dict'][k].clone()
    m.load_state_dict(state, strict=False); return m.eval().to(device)

model_base = build()
ft = torch.load('triplet_sweep/v5_experience/best.pt', map_location='cpu', weights_only=False)
model_ft = build()
ft_state = model_ft.state_dict()
for k in ft_state:
    if k in ft['model_state_dict'] and ft_state[k].shape == ft['model_state_dict'][k].shape:
        ft_state[k] = ft['model_state_dict'][k].clone()
model_ft.load_state_dict(ft_state, strict=False); model_ft.eval().to(device)

print(f'Models loaded. Fine-tuned: epoch={ft["epoch"]+1} val_sep={ft["val_sep"]:.4f} val_acc={ft.get("val_acc",0):.3f}', flush=True)

# ---- Build AUC set: 500 IKs, multi-spectrum ----
print('Building AUC set...', flush=True)
N_PEAKS = 128
def peaks_to_tensor(peaks):
    arr=np.array(peaks,dtype=np.float32); arr=arr[arr[:,0].argsort()]
    if len(arr)>N_PEAKS:
        idx=np.argpartition(arr[:,1],-N_PEAKS)[-N_PEAKS:]; arr=arr[idx]; arr=arr[arr[:,0].argsort()]
    max_i=arr[:,1].max()
    if max_i>0: arr[:,1]/=max_i
    p=np.zeros((N_PEAKS,2),dtype=np.float32); n=min(len(arr),N_PEAKS); p[:n]=arr[:n]
    return torch.from_numpy(p)

ik_all_peaks=defaultdict(list)
cur=None; cur_peaks=[]
with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        line=line.strip()
        if not line:
            if cur and len(cur_peaks)>=3: ik_all_peaks[cur].append(cur_peaks[:])
            cur=None; cur_peaks=[]; continue
        if line.startswith('INCHIKEY='): cur=line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
            p2=line.split()
            if len(p2)>=2:
                try:
                    mz,i=float(p2[0]),float(p2[1])
                    if mz>0 and i>0: cur_peaks.append((mz,i))
                except: pass

multi={ik:pks for ik,pks in ik_all_peaks.items() if len(pks)>=2}
print(f'  {len(multi)} multi-spectrum IKs', flush=True)

rng=np.random.RandomState(42)
s_iks=rng.choice(sorted(multi.keys()),min(500,len(multi)),replace=False)
auc_specs=[]; auc_idx={}
for ik in s_iks:
    auc_idx[ik]=[]
    for pk in multi[ik][:3]:
        t=peaks_to_tensor(pk)
        if t is not None: auc_idx[ik].append(len(auc_specs)); auc_specs.append(t)

pi,pj,lb=[],[],[]
ml=[ik for ik in s_iks if len(auc_idx[ik])>=2]
al=[ik for ik in s_iks if len(auc_idx[ik])>=1]
n_each=2000
np_=0
while np_<n_each and ml:
    ik=rng.choice(ml); idxs=auc_idx[ik]
    if len(idxs)>=2: a,b=rng.choice(idxs,2,replace=False); pi.append(a);pj.append(b);lb.append(1);np_+=1
nn_=0
while nn_<n_each and len(al)>=2:
    ika,ikb=rng.choice(al,2,replace=False)
    if ika==ikb: continue
    a=rng.choice(auc_idx[ika]);b=rng.choice(auc_idx[ikb]);pi.append(a);pj.append(b);lb.append(0);nn_+=1
pi=np.array(pi);pj=np.array(pj);lb=np.array(lb)
print(f'  AUC: {np_}P+{nn_}N, {len(auc_specs)} spectra', flush=True)

# ---- Triplet val ----
with open('tasks/T1_near_isomers/test_cases/triplets_val.json') as f: vt=json.load(f)
needed=set(); ik14=lambda x:x[:14]
for t in vt:
    for k in('anchor_ik','pos_ik','neg_ik'): needed.add(ik14(t[k]))
ik2spec={}
for ik in needed:
    if ik in ik_all_peaks: st=peaks_to_tensor(ik_all_peaks[ik][0])
    else: st=None
    if st is not None: ik2spec[ik]=st
vtt=[(ik14(t['anchor_ik']),ik14(t['pos_ik']),ik14(t['neg_ik'])) for t in vt]
vtt=[(a,p,n) for a,p,n in vtt if a in ik2spec and p in ik2spec and n in ik2spec]
specs=list({ik:s for ik,s in ik2spec.items()}.values())
iks=sorted(ik2spec.keys())
ik2i={ik:i for i,ik in enumerate(iks)}
print(f'  Triplets: {len(vtt)} valid', flush=True)

# ---- Embeddings (batched) ----
print('Extracting embeddings...', flush=True)
@torch.no_grad()
def emb_batch(model,spec_list):
    embs=[]
    for s in range(0,len(spec_list),4):
        b=torch.stack(spec_list[s:s+4]).to(device)
        embs.append(model(b,None)[:,0,:].cpu())
    return torch.cat(embs,dim=0)

t0=time.time()
ae_b=emb_batch(model_base,auc_specs); ae_f=emb_batch(model_ft,auc_specs)
te_b=emb_batch(model_base,specs); te_f=emb_batch(model_ft,specs)
print(f'  Done ({time.time()-t0:.0f}s)', flush=True)

# ---- Compute ----
print('\n' + '='*60, flush=True)
print('RESULTS', flush=True)
print('='*60, flush=True)

for name,ae in [('Pretrained',ae_b),('Fine-tuned',ae_f)]:
    cs=F.cosine_similarity(ae[pi],ae[pj],dim=-1).numpy()
    fpr,tpr,_=metrics.roc_curve(lb,cs); auc=float(metrics.auc(fpr,tpr))
    cp=cs[lb==1].mean(); cn=cs[lb==0].mean()
    print(f'\n{name}:', flush=True)
    print(f'  AUC={auc:.4f}  cos+={cp:.4f}  cos-={cn:.4f}  sep={cp-cn:.4f}', flush=True)

for name,te in [('Pretrained',te_b),('Fine-tuned',te_f)]:
    seps=[]; cor=0
    for a,p,n in vtt:
        ea=te[ik2i[a]]; ep=te[ik2i[p]]; en=te[ik2i[n]]
        cp=F.cosine_similarity(ea.unsqueeze(0),ep.unsqueeze(0),dim=-1).item()
        cn=F.cosine_similarity(ea.unsqueeze(0),en.unsqueeze(0),dim=-1).item()
        seps.append(cp-cn); cor+=(cp>cn)
    print(f'\n{name} Triplet:', flush=True)
    print(f'  sep={np.mean(seps):.4f}  acc={cor/len(vtt):.4f}', flush=True)

print('\n'+'='*60, flush=True)
print('DONE', flush=True)
