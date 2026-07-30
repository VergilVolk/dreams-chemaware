"""
分析 T1 triplet 训练结果 — 对比预训练 vs 微调后模型

用法 (dreams_env): python analyze_triplet_training.py
"""
import torch, json, numpy as np
from collections import defaultdict
from tqdm import tqdm
from argparse import Namespace
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor
from dreams.models.dreams.dreams import DreaMS

print('[1] Loading models...')
pkg = torch.load('dreams/models/pretrained/ssl_model_server.pt', map_location='cpu', weights_only=False)
recon_args = Namespace(**pkg['args'])
recon_args.dformat = DataFormatA()
for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge','max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
    if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
recon_args.d_graphormer_params = 0
sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)

# Baseline
model_base = DreaMS(recon_args, sp)
state = model_base.state_dict()
for k in state:
    if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
        state[k] = pkg['state_dict'][k].clone()
model_base.load_state_dict(state, strict=False); model_base.eval()

# Fine-tuned
ft = torch.load('triplet_t1_checkpoints/best_model.pt', map_location='cpu', weights_only=False)
model_ft = DreaMS(recon_args, sp)
ft_state = model_ft.state_dict()
for k in ft_state:
    if k in ft['model_state_dict'] and ft_state[k].shape == ft['model_state_dict'][k].shape:
        ft_state[k] = ft['model_state_dict'][k].clone()
model_ft.load_state_dict(ft_state, strict=False); model_ft.eval()

print(f'  Training: epoch={ft["epoch"]}, val_loss={ft["val_loss"]:.4f}')
h = ft['history']
for i in range(len(h.get('epoch',[]))):
    print(f'  Epoch {h["epoch"][i]}: train_loss={h["train_loss"][i]:.4f} val_loss={h["val_loss"][i]:.4f} train_sep={h["train_sep"][i]:.4f} val_sep={h["val_sep"][i]:.4f}')

# Load all triplets (train+val) for comprehensive eval
print('\n[2] Loading triplets...')
all_trip = json.load(open('tasks/T1_near_isomers/test_cases/triplets_train.json'))
all_trip += json.load(open('tasks/T1_near_isomers/test_cases/triplets_val.json'))
rng = np.random.RandomState(42)
sample = rng.choice(all_trip, min(1000, len(all_trip)), replace=False)
print(f'  Evaluating on {len(sample)} sampled triplets')

# Load spectra
print('[3] Loading spectra...')
needed = set()
for t in sample:
    needed.add(t['anchor_ik'][:14]); needed.add(t['pos_ik'][:14]); needed.add(t['neg_ik'][:14])

ik_to_peaks = {}
cur_ik=None; cur_peaks=[]
with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        if len(ik_to_peaks)>=len(needed): break
        line=line.strip()
        if not line:
            if cur_ik and cur_ik in needed and cur_ik not in ik_to_peaks and len(cur_peaks)>=3:
                ik_to_peaks[cur_ik]=cur_peaks[:]
            cur_ik=None; cur_peaks=[]; continue
        if line.startswith('INCHIKEY='): cur_ik=line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
            p2=line.split()
            if len(p2)>=2:
                try:
                    mz,i=float(p2[0]),float(p2[1])
                    if mz>0 and i>0: cur_peaks.append((mz,i))
                except: pass

# Extract embeddings
print('[4] Extracting embeddings...')
ik_to_emb_base = {}; ik_to_emb_ft = {}
for ik, peaks in tqdm(ik_to_peaks.items(), desc='Embed'):
    arr=np.array(peaks,dtype=np.float32); arr=arr[arr[:,0].argsort()]
    try: spec_pp=sp(arr.T,high_form=False)
    except: continue
    spec_t=torch.as_tensor(spec_pp,dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        ik_to_emb_base[ik]=model_base(spec_t,None)[:,0,:]
        ik_to_emb_ft[ik]=model_ft(spec_t,None)[:,0,:]

# Filter (triplets have 27-char IKs, ik_to_emb has 14-char keys)
import torch.nn.functional as F
def ik14(x): return x[:14]
valid = [t for t in sample
         if ik14(t['anchor_ik']) in ik_to_emb_ft
         and ik14(t['pos_ik']) in ik_to_emb_ft
         and ik14(t['neg_ik']) in ik_to_emb_ft]
print(f'  {len(valid)}/{len(sample)} valid')

# Compute metrics
print('[5] Computing metrics...')
cp_b_all=[]; cn_b_all=[]; cp_f_all=[]; cn_f_all=[]
for t in tqdm(valid, desc='Eval'):
    aik=ik14(t['anchor_ik']); pik=ik14(t['pos_ik']); nik=ik14(t['neg_ik'])
    a_b=ik_to_emb_base[aik]; p_b=ik_to_emb_base[pik]; n_b=ik_to_emb_base[nik]
    a_f=ik_to_emb_ft[aik]; p_f=ik_to_emb_ft[pik]; n_f=ik_to_emb_ft[nik]
    cp_b_all.append(F.cosine_similarity(a_b,p_b,dim=-1).item())
    cn_b_all.append(F.cosine_similarity(a_b,n_b,dim=-1).item())
    cp_f_all.append(F.cosine_similarity(a_f,p_f,dim=-1).item())
    cn_f_all.append(F.cosine_similarity(a_f,n_f,dim=-1).item())

# Summary
print(f'\n{"="*65}')
print(f'T1 TRIPLET TRAINING — 1 Epoch Evaluation ({len(valid)} triplets)')
print(f'{"="*65}')
print(f'')
print(f'                    Pretrained DreaMS   |  Fine-tuned (1 epoch)')
print(f'                    ─────────────────   |  ────────────────────')
print(f'cos+ (pos):         {np.mean(cp_b_all):.4f} ± {np.std(cp_b_all):.4f}       |  {np.mean(cp_f_all):.4f} ± {np.std(cp_f_all):.4f}')
print(f'cos- (neg):         {np.mean(cn_b_all):.4f} ± {np.std(cn_b_all):.4f}       |  {np.mean(cn_f_all):.4f} ± {np.std(cn_f_all):.4f}')
print(f'Separation:         {np.mean(cp_b_all)-np.mean(cn_b_all):.4f}              |  {np.mean(cp_f_all)-np.mean(cn_f_all):.4f}')
acc_b=sum(1 for cp,cn in zip(cp_b_all,cn_b_all) if cp>cn)/len(valid)
acc_f=sum(1 for cp,cn in zip(cp_f_all,cn_f_all) if cp>cn)/len(valid)
print(f'Triplet Accuracy:   {acc_b:.4f}                  |  {acc_f:.4f}')
print(f'')
print(f'Δ Separation: {np.mean(cp_f_all)-np.mean(cn_f_all) - (np.mean(cp_b_all)-np.mean(cn_b_all)):+.4f}')
print(f'Δ Accuracy:   {acc_f-acc_b:+.4f}')
print(f'{"="*65}')
print(f'')
print(f'Interpretation:')
if acc_f > acc_b:
    print(f'  ✓ Fine-tuning improved triplet accuracy by {acc_f-acc_b:+.4f} ({100*(acc_f-acc_b):.1f}%)')
else:
    print(f'  ⚠ Fine-tuning did not improve triplet accuracy')
sep_delta = np.mean(cp_f_all)-np.mean(cn_f_all) - (np.mean(cp_b_all)-np.mean(cn_b_all))
if sep_delta > 0.01:
    print(f'  ✓ Separation increased by {sep_delta:+.4f} — model better distinguishes near-isomers')
elif sep_delta > 0:
    print(f'  ~ Separation marginally improved ({sep_delta:+.4f})')
else:
    print(f'  ⚠ Separation decreased ({sep_delta:+.4f}) — may need more epochs')
