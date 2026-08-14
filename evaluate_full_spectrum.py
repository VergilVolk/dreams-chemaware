"""
DreaMS 零样本全谱系评估 — MCES 0-10 完整覆盖

评估集: T1 pos(MCES 0-2) + T2 pos(MCES 3-5) + T1 neg(MCES 6-10)
输出: dreams_fullspectrum.png (4面板) + dreams_fullspectrum.json

用法: python evaluate_full_spectrum.py
"""
import torch, json, numpy as np, os, sys, time
from collections import defaultdict
from argparse import Namespace
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

BATCH_SIZE = 4
rng = np.random.RandomState(42)

# ===================================================================
# 1. Load DreaMS
# ===================================================================
print('[1] Loading DreaMS...', flush=True)
t0 = time.time()
pkg = torch.load('dreams/models/pretrained/ssl_model_server.pt', map_location='cpu', weights_only=False)
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor
from dreams.models.dreams.dreams import DreaMS

recon_args = Namespace(**pkg['args'])
recon_args.dformat = DataFormatA()
for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge','max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
    if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
recon_args.d_graphormer_params = 0
sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
model = DreaMS(recon_args, sp)
state = model.state_dict()
for k in state:
    if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
        state[k] = pkg['state_dict'][k].clone()
model.load_state_dict(state, strict=False)
model.eval().to(device)
print(f'  OK ({time.time()-t0:.0f}s)', flush=True)

# ===================================================================
# 2. Load all pairs: T1 pos (MCES 0-2) + T2 pos (MCES 3-5) + T1 neg (MCES 6-10)
# ===================================================================
print('[2] Loading MCES pairs...', flush=True)
with open('tasks/T1_near_isomers/test_cases/pairs.json') as f: t1 = json.load(f)
with open('tasks/T2_analogs/test_cases/pairs.json') as f: t2 = json.load(f)

pairs = []  # (ik_a, ik_b, mces, label: 1=pos, 0=neg)
for p in t1['positive']:
    pairs.append((p['ik_a'][:14], p['ik_b'][:14], p.get('mces_raw',0), 1, 'T1_pos'))
for p in t2['positive']:
    pairs.append((p['ik_a'][:14], p['ik_b'][:14], p.get('mces_raw',5), 1, 'T2_pos'))
for p in t1['negative_hard']:
    pairs.append((p['ik_a'][:14], p['ik_b'][:14], p.get('mces_raw',8), 0, 'T1_neg'))

_t1p = sum(1 for p in pairs if p[4]=='T1_pos')
_t2p = sum(1 for p in pairs if p[4]=='T2_pos')
_t1n = sum(1 for p in pairs if p[4]=='T1_neg')
print(f'  T1 pos (MCES 0-2): {_t1p}')
print(f'  T2 pos (MCES 3-5): {_t2p}')
print(f'  T1 neg (MCES 6-10):{_t1n}')
print(f'  Total: {len(pairs)}')
print(f'  Pos/Neg: {sum(1 for p in pairs if p[3])}/{sum(1 for p in pairs if not p[3])}', flush=True)

# ===================================================================
# 3. Scan annotated01 for spectra + SMILES
# ===================================================================
print('[3] Loading spectra...', flush=True)
needed = set()
for a,b,_,_,_ in pairs:
    needed.add(a); needed.add(b)

ik_smi = {}; ik_peaks = {}
cur_ik=None; cur_smi=None; cur_peaks=[]
with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        line=line.strip()
        if not line:
            if cur_ik and cur_ik in needed and cur_ik not in ik_peaks and len(cur_peaks)>=3:
                ik_peaks[cur_ik]=cur_peaks[:]
                if cur_smi: ik_smi[cur_ik]=cur_smi
            cur_ik=None; cur_smi=None; cur_peaks=[]
            if len(ik_peaks)>=len(needed): break
            continue
        if line.startswith('SMILES='): cur_smi=line[7:].strip()
        elif line.startswith('INCHIKEY='): cur_ik=line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
            p2=line.split()
            if len(p2)>=2:
                try:
                    mz,i=float(p2[0]),float(p2[1])
                    if mz>0 and i>0: cur_peaks.append((mz,i))
                except: pass

print(f'  {len(ik_peaks)}/{len(needed)} spectra, {len(ik_smi)} SMILES', flush=True)

# ===================================================================
# 4. Preprocess spectra
# ===================================================================
print('[4] Preprocessing spectra...', flush=True)
ik_to_spec = {}
for ik, peaks in ik_peaks.items():
    arr=np.array(peaks,dtype=np.float32)
    try:
        spec_pp=sp(arr.T,high_form=False)
        ik_to_spec[ik]=torch.as_tensor(spec_pp,dtype=torch.float32)
    except: pass
print(f'  {len(ik_to_spec)} valid spectra', flush=True)

# Filter pairs
valid = [(a,b,m,l,t) for a,b,m,l,t in pairs if a in ik_to_spec and b in ik_to_spec]
print(f'  {len(valid)} valid pairs', flush=True)

# ===================================================================
# 5. Compute Tanimoto for all valid pairs
# ===================================================================
print('[5] Computing Tanimoto...', flush=True)
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Pre-compute fingerprints
fp_iks = set()
for a,b,_,_,_ in valid:
    fp_iks.add(a); fp_iks.add(b)
ik_to_fp = {}
for ik in fp_iks:
    smi = ik_smi.get(ik,'')
    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol: ik_to_fp[ik] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

tani_vals = []
for a,b,_,_,_ in valid:
    if a in ik_to_fp and b in ik_to_fp:
        tani_vals.append(float(DataStructs.TanimotoSimilarity(ik_to_fp[a], ik_to_fp[b])))
    else:
        tani_vals.append(-1.0)
print(f'  {sum(1 for t in tani_vals if t>=0)}/{len(tani_vals)} Tanimoto computed', flush=True)

# ===================================================================
# 6. Extract embeddings
# ===================================================================
print('[6] Extracting embeddings...', flush=True)
spec_list = [ik_to_spec[ik] for ik in sorted(ik_to_spec.keys())]
ik_to_idx = {ik:i for i,ik in enumerate(sorted(ik_to_spec.keys()))}

@torch.no_grad()
def batch_embed(spec_list):
    embs=[]
    for s in range(0,len(spec_list),BATCH_SIZE):
        b=torch.stack(spec_list[s:s+BATCH_SIZE]).to(device)
        embs.append(model(b,None)[:,0,:].cpu())
    return torch.cat(embs,dim=0)

t_emb=time.time()
embs = batch_embed(spec_list)
print(f'  Done ({time.time()-t_emb:.0f}s)', flush=True)

# Compute cosine similarities
cos_sims = []
for a,b,_,_,_ in valid:
    ea = embs[ik_to_idx[a]]; eb = embs[ik_to_idx[b]]
    cos_sims.append(float(F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0), dim=-1)))
cos_sims = np.array(cos_sims)
tani_arr = np.array(tani_vals)

# ===================================================================
# 7. Per-bin analysis
# ===================================================================
print('[7] Computing metrics...', flush=True)

labels = np.array([p[3] for p in valid])
mces_vals = np.array([p[2] for p in valid])

# Overall AUC
fpr,tpr,_=metrics.roc_curve(labels, cos_sims)
auc = float(metrics.auc(fpr,tpr))

# By MCES group
groups = [
    (0, 0, 'MCES=0 (stereo/tautomer)'),
    (1, 2, 'MCES 1-2 (near-isomer)'),
    (3, 5, 'MCES 3-5 (analog, BOUNDARY)'),
    (6, 10, 'MCES 6-10 (different isomer)'),
]

print(f'\n{"="*70}')
print(f'DREAMS ZERO-SHOT — FULL MCES SPECTRUM EVALUATION')
print(f'{"="*70}')
print(f'  Total pairs: {len(valid)} ({labels.sum():.0f}P + {(1-labels).sum():.0f}N)')
print(f'  Overall AUC: {auc:.4f}')
print(f'')

group_results = []
for lo, hi, name in groups:
    mask = (mces_vals >= lo) & (mces_vals <= hi)
    if mask.sum() == 0: continue
    cs = cos_sims[mask]; ts = tani_arr[mask]
    ls = labels[mask]

    if len(np.unique(ls)) >= 2:
        g_auc = float(metrics.roc_auc_score(ls, cs))
    else:
        g_auc = float('nan')
    cs_pos = cs[ls==1].mean() if (ls==1).any() else float('nan')
    cs_neg = cs[ls==0].mean() if (ls==0).any() else float('nan')
    tani_mean = ts[ts>=0].mean() if (ts>=0).any() else float('nan')

    # Correlation (only for groups with enough spread)
    if mask.sum() >= 10 and ts[ts>=0].std() > 0.05:
        valid_ts = ts[ts>=0]; valid_cs = cs[ts>=0]
        pr, pp = pearsonr(valid_cs, valid_ts)
    else:
        pr, pp = float('nan'), float('nan')

    group_results.append((name, mask.sum(), g_auc, cs_pos, cs_neg, tani_mean, pr))
    print(f'  {name:35s}  n={mask.sum():5d}  AUC={g_auc}  cos+={cs_pos}  cos-={cs_neg}  Tani={tani_mean}  r={pr}', flush=True)

# Tanimoto ceiling AUC
tani_valid = tani_arr >= 0
if len(np.unique(labels[tani_valid])) >= 2:
    tani_auc = float(metrics.roc_auc_score(labels[tani_valid], tani_arr[tani_valid]))
else:
    tani_auc = float('nan')
print(f'\n  Tanimoto-alone AUC (structural ceiling): {tani_auc:.4f}', flush=True)

# Pearson/Spearman (all pairs with valid Tanimoto)
valid_mask = tani_valid
r_val, p_val = pearsonr(cos_sims[valid_mask], tani_arr[valid_mask])
rho, sp_val = spearmanr(cos_sims[valid_mask], tani_arr[valid_mask])
print(f'  Pearson r: {r_val:.4f} (p={p_val:.2e})')
print(f'  Spearman ρ: {rho:.4f} (p={sp_val:.2e})')
print(f'{"="*70}', flush=True)

# ===================================================================
# 8. Plot
# ===================================================================
print('[8] Plotting...', flush=True)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('DreaMS Zero-Shot — Full MCES Spectrum Evaluation\n'
             f'annotated01 · {len(valid)} pairs · MCES 0-10 · AUC={auc:.4f}',
             fontsize=14, fontweight='bold')

# (a) ROC by MCES group
ax=axes[0,0]
colors = {0: '#2ecc71', 1: '#3498db', 2: '#e74c3c', 3: '#95a5a6'}
for (lo,hi,name), color in zip(groups, colors.values()):
    mask = (mces_vals >= lo) & (mces_vals <= hi)
    if mask.sum() < 5 or len(np.unique(labels[mask])) < 2: continue
    fpr_g, tpr_g, _ = metrics.roc_curve(labels[mask], cos_sims[mask])
    gr = [r for r in group_results if r[0] == name][0]
    ax.plot(fpr_g, tpr_g, lw=2, color=color, label=f'{name} (n={gr[1]})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('(a) ROC by MCES Group'); ax.legend(fontsize=9)
ax.grid(True,alpha=0.3)

# (b) Cosine similarity by MCES bin
ax=axes[0,1]
bin_edges = [0,1,3,6,11]
bin_labels = ['MCES=0','MCES 1-2','MCES 3-5','MCES 6-10']
bin_colors = ['#2ecc71','#3498db','#e74c3c','#95a5a6']
for i in range(len(bin_edges)-1):
    mask = (mces_vals >= bin_edges[i]) & (mces_vals < bin_edges[i+1])
    if mask.sum() == 0: continue
    cs = cos_sims[mask]
    pos_cs = cs[labels[mask]==1] if (labels[mask]==1).any() else np.array([])
    neg_cs = cs[labels[mask]==0] if (labels[mask]==0).any() else np.array([])
    pos = i*2 - 0.2; neg = i*2 + 0.2
    bp = ax.boxplot([pos_cs, neg_cs] if len(pos_cs)>0 and len(neg_cs)>0 else ([pos_cs] if len(pos_cs)>0 else [neg_cs]),
                     positions=[pos, neg] if len(pos_cs)>0 and len(neg_cs)>0 else ([pos] if len(pos_cs)>0 else [neg]),
                     widths=0.3, patch_artist=True)
    for patch, c in zip(bp['boxes'], [bin_colors[i], bin_colors[i]]):
        patch.set_facecolor(c); patch.set_alpha(0.5)
ax.set_xticks([i*2 for i in range(len(bin_edges)-1)])
ax.set_xticklabels(bin_labels, rotation=30)
ax.set_ylabel('Cosine Similarity')
ax.set_title('(b) Cosine Similarity Distribution by MCES')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax.grid(True,alpha=0.3)

# (c) Cos vs Tanimoto (color by MCES)
ax=axes[1,0]
mces_colors = np.array(['#2ecc71']*len(valid))
for i, (lo,hi) in enumerate([(0,1),(1,3),(3,6),(6,11)]):
    mces_colors[(mces_vals>=lo)&(mces_vals<hi)] = bin_colors[i]
sc = ax.scatter(tani_arr[valid_mask], cos_sims[valid_mask], alpha=0.15, s=6,
                c=mces_colors[valid_mask], edgecolors='none', rasterized=True)
ax.set_xlabel('Morgan Tanimoto (r=2, 2048 bits)')
ax.set_ylabel('DreaMS Cosine Similarity')
ax.set_title(f'(c) Cos vs Tanimoto — r={r_val:.4f}, ρ={rho:.4f}')
# Binned means
be=np.linspace(0,1,21); bc=(be[:-1]+be[1:])/2
bm=[cos_sims[valid_mask][(tani_arr[valid_mask]>=lo)&(tani_arr[valid_mask]<hi)].mean()
    if np.any((tani_arr[valid_mask]>=lo)&(tani_arr[valid_mask]<hi)) else np.nan
    for lo,hi in zip(be[:-1],be[1:])]
vb=~np.isnan(bm)
ax.plot(bc[vb],np.array(bm)[vb],'o-',color='#e74c3c',lw=2.5,ms=8,label='Binned mean')
ax.plot([0,1],[0,1],'k--',alpha=0.2)
ax.legend(fontsize=9,loc='upper left'); ax.grid(True,alpha=0.3)

# (d) AUC per MCES bin + Tanimoto ceiling
ax=axes[1,1]
x_labels = []; x_pos = []; auc_vals = []; tani_vals_plot = []; n_vals = []
for i, (lo,hi,name) in enumerate(groups):
    mask = (mces_vals>=lo)&(mces_vals<=hi)
    if mask.sum() < 5: continue
    ls = labels[mask]; cs = cos_sims[mask]; ts = tani_arr[mask]
    if len(np.unique(ls))>=2:
        g_auc=float(metrics.roc_auc_score(ls,cs))
        t_auc=float(metrics.roc_auc_score(ls,ts)) if ts.min()!=ts.max() else float('nan')
    else:
        g_auc=float('nan'); t_auc=float('nan')
    x_labels.append(name.split('(')[0].strip()); x_pos.append(i)
    auc_vals.append(g_auc); tani_vals_plot.append(t_auc); n_vals.append(mask.sum())

x_pos = np.array(x_pos)
w=0.3
bars1=ax.bar(x_pos-w, auc_vals, w, color='#3498db', alpha=0.8, label='DreaMS')
bars2=ax.bar(x_pos+w, tani_vals_plot, w, color='#e74c3c', alpha=0.8, label='Tanimoto (ceiling)')
for i, (auc_v, tani_v, n) in enumerate(zip(auc_vals, tani_vals_plot, n_vals)):
    if not np.isnan(auc_v): ax.text(x_pos[i]-w, auc_v+0.01, f'{auc_v:.3f}', ha='center', fontsize=9)
    if not np.isnan(tani_v): ax.text(x_pos[i]+w, tani_v+0.01, f'{tani_v:.3f}', ha='center', fontsize=9)
    ax.text(x_pos[i], 0.02, f'n={n}', ha='center', fontsize=8, color='gray')
ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=20)
ax.set_ylabel('AUC'); ax.set_ylim(0,1.1)
ax.set_title('(d) AUC by MCES Group vs Tanimoto Ceiling')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3,axis='y')

plt.tight_layout()
plt.savefig('dreams_fullspectrum.png', dpi=150, bbox_inches='tight')
print('  Saved: dreams_fullspectrum.png', flush=True)

results = {
    'overall_auc': auc,
    'tanimoto_auc': tani_auc,
    'pearson_r': r_val, 'pearson_p': p_val,
    'spearman_rho': rho, 'spearman_p': sp_val,
    'n_pairs': len(valid), 'n_pos': int(labels.sum()), 'n_neg': int((1-labels).sum()),
    'groups': [{'name': n, 'n': int(nn), 'auc': float(a) if not np.isnan(a) else None,
                'cos_pos': float(cp) if not np.isnan(cp) else None,
                'cos_neg': float(cn) if not np.isnan(cn) else None,
                'tani_mean': float(tm) if not np.isnan(tm) else None,
                'pearson_r': float(pr) if not np.isnan(pr) else None}
               for n,nn,a,cp,cn,tm,pr in group_results]
}
with open('dreams_fullspectrum.json','w') as f: json.dump(results, f, indent=2)
print('  Saved: dreams_fullspectrum.json', flush=True)
print(f'\nDone in {time.time()-t0:.0f}s', flush=True)
