"""
DreaMS Figure 4a — 严格复现 (MassSpecGym + 论文协议)

论文方法 (mol_sim_corr.ipynb):
  ① 预计算 ALL pairwise Morgan Tanimoto (r=2, nBits=4096)
  ② 分层采样: bin_size=0.025, 40 bins × 2500, random.choices (有放回)
  ③ SMILES → 谱图查找
  ④ DreaMS 嵌入 → Cosine Similarity
  ⑤ Pearson/Spearman on ~100K stratified pairs
  ⑥ hist2d + LogNorm 绘图

数据: MassSpecGym_MurckoHist_split.hdf5
  - 231,104 total spectra → 过滤 [M+H]+ → ~195K spectra
  - ~31,600 unique SMILES → C(31600,2) ≈ 500M 全成对 Tanimoto

论文基线: DreaMS zero-shot Pearson=0.634 Spearman=0.629 AUC≈0.85
"""
import h5py, torch, numpy as np, os, sys, time, argparse, random
from collections import defaultdict, Counter
from argparse import Namespace
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr, linregress
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, '.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# 参数 (匹配论文)
# ===================================================================
THLD = 2500              # 每 bin pairs 数
BIN_SIZE = 0.025         # bin 宽度
N_BINS = 40
N_PEAKS = 60             # 论文 Cell 24
FP_BITS = 4096           # 论文 morgan_fp() 默认
BATCH_SIZE = 4
RESERVOIR_PER_BIN = 100000  # 每 bin 最多保留候选对数
AUC_N = 3000
HDF5_PATH = 'data/models/MassSpecGym_MurckoHist_split.hdf5'
CKPT_PATH = 'dreams/models/pretrained/ssl_model_server.pt'
rng_np = np.random.RandomState(42)
rng_py = random.Random(42)

# ===================================================================
# 1. 加载 DreaMS
# ===================================================================
print('=' * 65)
print('DreaMS Figure 4a — MassSpecGym + Strict Paper Protocol')
print('=' * 65)
print(f'Device: {device}', flush=True)

print('\n[1] Loading DreaMS (n_peaks=60)...', flush=True)
t_start = time.time()
pkg = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
from dreams.utils.dformats import DataFormatA
from dreams.utils.data import SpectrumPreprocessor
from dreams.models.dreams.dreams import DreaMS

recon_args = Namespace(**pkg['args'])
recon_args.dformat = DataFormatA()
for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge',
           'max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
    if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
recon_args.d_graphormer_params = 0
sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=N_PEAKS)
model = DreaMS(recon_args, sp)
state = model.state_dict()
for k in state:
    if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
        state[k] = pkg['state_dict'][k].clone()
model.load_state_dict(state, strict=False)
model.eval().to(device)
print(f'  OK ({time.time()-t_start:.0f}s)', flush=True)

# ===================================================================
# 2. 加载 MassSpecGym → 过滤 [M+H]+
# ===================================================================
print('\n[2] Loading MassSpecGym + filtering [M+H]+...', flush=True)
f = h5py.File(HDF5_PATH, 'r')

# 读取所有元数据 (字节串需解码)
def decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

all_adducts = decode(f['adduct'][:])
all_smiles = decode(f['smiles'][:])
all_inchi = decode(f['INCHIKEY'][:])
all_folds = decode(f['fold'][:])

# 过滤: [M+H]+ only (论文协议)
mh_mask = np.array([a == '[M+H]+' for a in all_adducts])
n_total = len(all_smiles)
n_mh = mh_mask.sum()
print(f'  Total: {n_total} | [M+H]+: {n_mh} ({n_mh/n_total*100:.1f}%)')

# 获取 [M+H]+ 子集的所有数据
mh_indices = np.where(mh_mask)[0]
mh_smiles = [all_smiles[i] for i in mh_indices]
mh_inchi = [all_inchi[i][:14] for i in mh_indices]
mh_folds = [all_folds[i] for i in mh_indices]

# 构建 SMILES → spectrum indices 映射
smiles_to_spec_idx = defaultdict(list)
for idx, smi in zip(mh_indices, mh_smiles):
    smiles_to_spec_idx[smi].append(idx)

# 统计
unique_smiles = sorted(smiles_to_spec_idx.keys())
n_unique = len(unique_smiles)
print(f'  Unique SMILES: {n_unique}')
print(f'  C({n_unique},2) = {n_unique*(n_unique-1)//2:,} all-pairs')

# Multi-spectrum IKs (for same-molecule pairs)
ik_to_spec_idx = defaultdict(list)
for idx, ik in zip(mh_indices, mh_inchi):
    ik_to_spec_idx[ik].append(idx)
multi_iks = {ik: idxs for ik, idxs in ik_to_spec_idx.items() if len(idxs) >= 2}
n_same_mol_pairs = sum(len(idxs)*(len(idxs)-1)//2 for idxs in multi_iks.values())
print(f'  IKs with >=2 spectra: {len(multi_iks)}')
print(f'  Same-molecule possible pairs: {n_same_mol_pairs:,}')

# ===================================================================
# 3. Morgan 指纹 + ALL pairwise Tanimoto
# ===================================================================
print(f'\n[3] Morgan fingerprints (nBits={FP_BITS}, r=2) + ALL pairwise Tanimoto...', flush=True)

# 3a. 计算指纹
invalid_smiles = set()
smiles_to_fp = {}
for i, smi in enumerate(unique_smiles):
    if i % 10000 == 0: print(f'    fp {i}/{n_unique}', flush=True)
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        smiles_to_fp[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, FP_BITS)
    else:
        invalid_smiles.add(smi)

# 只保留有有效指纹的 SMILES
valid_smiles = sorted(s for s in unique_smiles if s in smiles_to_fp)
N = len(valid_smiles)
fp_list = [smiles_to_fp[s] for s in valid_smiles]
total_pairs = N * (N - 1) // 2
print(f'  Valid SMILES: {N} | Pairs: {total_pairs:,} ({total_pairs/1e9:.3f}B)')

# 3b. 计算 ALL 上三角 Tanimoto → 直接 bin
print(f'  Computing all pairwise Tanimoto + binning (reservoir={RESERVOIR_PER_BIN}/bin)...', flush=True)
bins_reservoir = {i: [] for i in range(N_BINS)}  # bin → [(smi_a, smi_b, tan)]

ts = time.time()
total_computed = 0

for i in range(N - 1):
    anchor_smi = valid_smiles[i]
    anchor_fp = fp_list[i]
    n_targets = N - i - 1

    # Bulk similarity → numpy array
    sims = np.array(DataStructs.BulkTanimotoSimilarity(anchor_fp, fp_list[i+1:]), dtype=np.float64)

    # 批量 bin (用 numpy 向量化)
    bin_indices = np.minimum((sims / BIN_SIZE).astype(int), N_BINS - 1)

    for bi in range(N_BINS):
        if len(bins_reservoir[bi]) >= RESERVOIR_PER_BIN:
            continue  # bin 已满，跳过
        mask = bin_indices == bi
        n_match = mask.sum()
        if n_match == 0:
            continue
        n_space = RESERVOIR_PER_BIN - len(bins_reservoir[bi])
        if n_match <= n_space:
            # 全部加入
            for j_offset in np.where(mask)[0]:
                j = i + 1 + j_offset
                bins_reservoir[bi].append((anchor_smi, valid_smiles[j], float(sims[j_offset])))
        else:
            # 随机采样 n_space 个
            selected = rng_np.choice(np.where(mask)[0], n_space, replace=False)
            for j_offset in selected:
                j = i + 1 + j_offset
                bins_reservoir[bi].append((anchor_smi, valid_smiles[j], float(sims[j_offset])))

    total_computed += n_targets
    if i % 2000 == 0 and i > 0:
        elapsed = time.time() - ts
        done_frac = (i * (2*N - i - 1) / 2) / total_pairs
        rate = total_computed / elapsed
        eta = elapsed / done_frac * (1 - done_frac)
        n_full = sum(1 for bi in range(N_BINS) if len(bins_reservoir[bi]) >= THLD)
        n_empty = sum(1 for bi in range(N_BINS) if len(bins_reservoir[bi]) == 0)
        print(f'    {done_frac*100:.0f}% | {total_computed/1e6:.1f}M computed '
              f'| {rate/1e6:.1f}M/s | {n_full}/{N_BINS} bins full '
              f'| {n_empty} empty | ETA {eta/60:.0f}min', flush=True)

elapsed = time.time() - ts
print(f'  Done: {total_computed/1e6:.1f}M pairs in {elapsed/60:.1f}min ({total_computed/elapsed/1e6:.2f}M/s)')

# 3c. 添加 same-molecule pairs (Tanimoto=1.0) → bin 39
print('  Adding same-molecule pairs...', flush=True)
n_sm_added = 0
for ik, spec_idxs in multi_iks.items():
    if len(bins_reservoir[39]) >= RESERVOIR_PER_BIN: break
    smi = all_smiles[spec_idxs[0]]
    if smi not in smiles_to_fp: continue
    for si in range(min(len(spec_idxs), 5)):
        for sj in range(si + 1, min(len(spec_idxs), 5)):
            if len(bins_reservoir[39]) < RESERVOIR_PER_BIN:
                bins_reservoir[39].append((smi, smi, 1.0))
                n_sm_added += 1
print(f'    {n_sm_added} same-molecule pairs → bin 39')

# ===================================================================
# 4. 报告 + 分层采样
# ===================================================================
print(f'\n[4] Bin reservoir sizes (ALL {N_BINS} bins):')
empty_bins = []
for i in range(N_BINS):
    n = len(bins_reservoir[i])
    bar = '#' * min(50, n * 50 // max(1, RESERVOIR_PER_BIN))
    tag = 'EMPTY' if n == 0 else ('FEW' if n < THLD else 'OK')
    if n == 0: empty_bins.append(i)
    print(f'    bin {i:2d} [{i*BIN_SIZE:.3f},{(i+1)*BIN_SIZE:.3f}): {n:6d} {bar} {tag}', flush=True)

# 分层采样 (论文 Cell 7: random.choices)
print(f'\n[5] Stratified sampling: {N_BINS} bins × {THLD} (random.choices)...', flush=True)
sampled_pairs = []
for i in range(N_BINS):
    cand = bins_reservoir[i]
    if len(cand) == 0:
        print(f'    bin {i}: SKIPPED (empty)')
        continue
    if len(cand) >= THLD:
        idxs = rng_np.choice(len(cand), THLD, replace=False)
    else:
        idxs = rng_np.choice(len(cand), THLD, replace=True)
        print(f'    bin {i}: {len(cand)} unique → resampling with replacement')
    for idx in idxs:
        sampled_pairs.append(cand[idx])

rng_np.shuffle(sampled_pairs)
print(f'  {len(sampled_pairs)} total sampled pairs ({len(empty_bins)} empty bins)')

# ===================================================================
# 5. 加载谱图 + DreaMS 嵌入
# ===================================================================
print('\n[6] Loading spectra + extracting embeddings...', flush=True)

# 构建 (smiles, spec_hdf5_idx) 的唯一列表
# 对于 same-molecule pairs: 需要两个不同的谱图
needed_specs = set()  # (smi, spec_hdf5_idx)
pair_specs = []       # [(smiles_a, spec_idx_a, smiles_b, spec_idx_b, tanimoto)]

for smi_a, smi_b, tan in sampled_pairs:
    idxs_a = smiles_to_spec_idx.get(smi_a, [])
    idxs_b = smiles_to_spec_idx.get(smi_b, [])

    if smi_a == smi_b:
        # Same molecule: pick 2 different spectra
        if len(idxs_a) >= 2:
            sa, sb = rng_np.choice(idxs_a, 2, replace=False)
            needed_specs.add((smi_a, int(sa)))
            needed_specs.add((smi_b, int(sb)))
            pair_specs.append((smi_a, int(sa), smi_b, int(sb), tan))
    else:
        if len(idxs_a) >= 1 and len(idxs_b) >= 1:
            sa = rng_np.choice(idxs_a)
            sb = rng_np.choice(idxs_b)
            needed_specs.add((smi_a, int(sa)))
            needed_specs.add((smi_b, int(sb)))
            pair_specs.append((smi_a, int(sa), smi_b, int(sb), tan))

print(f'  {len(needed_specs)} unique spectra to load')
print(f'  {len(pair_specs)} valid pairs')

# 加载谱图
spec_idx_to_tensor = {}
for smi, h5_idx in needed_specs:
    s = f['spectrum'][h5_idx]  # HDF5: (2,128), SpectrumPreprocessor expects this with high_form=False
    spec_np = s.astype(np.float32)  # keep (2,128), SP does .T internally
    spec_pp = sp(spec_np, high_form=False)
    spec_idx_to_tensor[(smi, h5_idx)] = torch.as_tensor(spec_pp, dtype=torch.float32)

print(f'  {len(spec_idx_to_tensor)} spectra preprocessed')

# 构建嵌入输入
spec_list = []
key_to_emb_idx = {}
for (smi, h5_idx), tensor in spec_idx_to_tensor.items():
    key_to_emb_idx[(smi, h5_idx)] = len(spec_list)
    spec_list.append(tensor)

corr_i, corr_j, corr_tani = [], [], []
for smi_a, sa, smi_b, sb, tan in pair_specs:
    ia = key_to_emb_idx.get((smi_a, sa))
    ib = key_to_emb_idx.get((smi_b, sb))
    if ia is not None and ib is not None:
        corr_i.append(ia); corr_j.append(ib); corr_tani.append(tan)

print(f'  {len(corr_i)} final correlation pairs')

# 批量提取嵌入
@torch.no_grad()
def batch_embed(lst):
    embs = []
    for s in range(0, len(lst), BATCH_SIZE):
        b = torch.stack(lst[s:s+BATCH_SIZE]).to(device)
        embs.append(model(b, None)[:, 0, :].cpu())
    return torch.cat(embs, dim=0)

t_emb = time.time()
all_emb = batch_embed(spec_list)
print(f'  Embeddings: {time.time()-t_emb:.0f}s', flush=True)

# ===================================================================
# 6. AUC 评估集
# ===================================================================
print('\n[7] AUC evaluation set...', flush=True)
auc_iks = rng_np.choice(sorted(multi_iks.keys()), min(300, len(multi_iks)), replace=False)
auc_specs = []; auc_ik_idx = {}

for ik in auc_iks:
    spec_idxs = multi_iks[ik][:3]
    auc_ik_idx[ik] = []
    for si in spec_idxs:
        s = f['spectrum'][si]  # (2,128)
        try:
            spec_pp = sp(s.astype(np.float32), high_form=False)  # SP does .T internally
            auc_ik_idx[ik].append(len(auc_specs))
            auc_specs.append(torch.as_tensor(spec_pp, dtype=torch.float32))
        except Exception:
            pass

# 构建 AUC pairs
ml = [ik for ik in auc_iks if len(auc_ik_idx[ik]) >= 2]
al = [ik for ik in auc_iks if len(auc_ik_idx[ik]) >= 1]
pi, pj, lb_arr = [], [], []

for _ in range(AUC_N):
    if ml:
        ik = ml[rng_np.randint(0, len(ml))]
        idxs = auc_ik_idx[ik]
        if len(idxs) >= 2:
            a, b = rng_np.choice(len(idxs), 2, replace=False)
            pi.append(idxs[a]); pj.append(idxs[b]); lb_arr.append(1)

for _ in range(AUC_N):
    if len(al) >= 2:
        ia, ib = rng_np.choice(len(al), 2, replace=False)
        if ia != ib:
            a = rng_np.choice(auc_ik_idx[al[ia]])
            b = rng_np.choice(auc_ik_idx[al[ib]])
            pi.append(a); pj.append(b); lb_arr.append(0)

pi, pj, lb_arr = np.array(pi), np.array(pj), np.array(lb_arr)

# 嵌入 AUC 谱图
auc_emb = batch_embed(auc_specs)
print(f'  {lb_arr.sum():.0f}P+{(1-lb_arr).sum():.0f}N, {len(auc_specs)} spectra')

f.close()  # 释放 HDF5

# ===================================================================
# 7. 指标
# ===================================================================
print('\n[8] Computing metrics...', flush=True)

# AUC
auc_cos = F.cosine_similarity(auc_emb[pi], auc_emb[pj], dim=-1).numpy()
fpr, tpr, _ = metrics.roc_curve(lb_arr, auc_cos)
auc = float(metrics.auc(fpr, tpr))

# Correlation
corr_i_np = np.array(corr_i); corr_j_np = np.array(corr_j)
corr_cos = F.cosine_similarity(all_emb[corr_i_np], all_emb[corr_j_np], dim=-1).numpy()
tani_arr = np.array(corr_tani)

cos_clip = np.clip(corr_cos, 0, 1)

r_val, p_val = pearsonr(cos_clip, tani_arr)
rho, sp_val = spearmanr(cos_clip, tani_arr)
r_raw, _ = pearsonr(corr_cos, tani_arr)
rho_raw, _ = spearmanr(corr_cos, tani_arr)

cp = auc_cos[lb_arr == 1].mean(); cn = auc_cos[lb_arr == 0].mean()

print(f'\n{"="*65}')
print(f'DREAMS ZERO-SHOT — MassSpecGym + Strict Paper Protocol')
print(f'  fp_bits={FP_BITS}  n_peaks={N_PEAKS}  {N_BINS} bins × {THLD}')
print(f'  {N} molecules → {total_computed/1e6:.1f}M pairwise Tanimoto')
print(f'{"="*65}')
print(f'  AUC:              {auc:.4f}  (paper: ~0.85)')
print(f'  Pearson r:        {r_val:.4f}  (paper: 0.634)')
print(f'  Spearman ρ:       {rho:.4f}  (paper: 0.629)')
print(f'  Pearson (raw):    {r_raw:.4f}  Spearman (raw): {rho_raw:.4f}')
print(f'  cos+ = {cp:.4f}  cos- = {cn:.4f}  sep = {cp-cn:.4f}')
print(f'  Corr pairs: {len(corr_i)}  AUC pairs: {len(lb_arr)}')
if empty_bins: print(f'  Empty bins: {empty_bins}')
print(f'{"="*65}', flush=True)

# 分 Tanimoto 范围
for lo, hi, label in [(0, 0.3, 'Low [0,0.3)'), (0.3, 0.7, 'Mid [0.3,0.7)'),
                        (0.7, 1.01, 'High [0.7,1.0]')]:
    mask = (tani_arr >= lo) & (tani_arr < hi)
    if mask.sum() > 1:
        r_sub, _ = pearsonr(cos_clip[mask], tani_arr[mask])
        print(f'    {label}: {mask.sum()} pairs, r={r_sub:.4f}')

# ===================================================================
# 8. 绘图 (论文 Cell 28 风格)
# ===================================================================
print('\n[9] Plotting...', flush=True)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'DreaMS Zero-Shot — MassSpecGym + Paper Protocol\n'
             f'Pearson r={r_val:.4f}  Spearman ρ={rho:.4f}  AUC={auc:.4f}',
             fontsize=12, fontweight='bold')

# (a) ROC
ax = axes[0]
ax.plot(fpr, tpr, color='#2ecc71', lw=2.5, label=f'DreaMS (AUC={auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title(f'(a) ROC — AUC={auc:.4f}'); ax.legend(loc='lower right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (b) Distribution
ax = axes[1]
b = np.linspace(-1, 1, 50)
ax.hist(auc_cos[lb_arr == 1], bins=b, alpha=0.5, color='#2ecc71', label=f'Same mol (μ={cp:.3f})')
ax.hist(auc_cos[lb_arr == 0], bins=b, alpha=0.5, color='#e74c3c', label=f'Diff mol (μ={cn:.3f})')
ax.set_xlabel('Cosine'); ax.set_ylabel('Freq')
ax.set_title(f'(b) Distribution — Sep={cp-cn:.4f}'); ax.legend(); ax.grid(True, alpha=0.3)

# (c) Cos vs Tanimoto (hist2d + LogNorm — 论文 Cell 28)
ax = axes[2]
paper_cmap = mcolors.LinearSegmentedColormap.from_list('paper', ['#E3E0F9', '#3498db', '#1a6e3f'], N=256)
h = ax.hist2d(tani_arr, cos_clip, bins=80, norm=mcolors.LogNorm(), cmap=paper_cmap, alpha=1.0)
slope, intercept, _, _, _ = linregress(tani_arr, cos_clip)
ax.plot([0, 1], [intercept, intercept + slope], '#e74c3c', lw=1.5, alpha=0.9)

# Binned means
be = np.linspace(0, 1, N_BINS + 1); bc = (be[:-1] + be[1:]) / 2
bm = np.array([cos_clip[(tani_arr >= lo) & (tani_arr < hi)].mean()
               if np.any((tani_arr >= lo) & (tani_arr < hi)) else np.nan
               for lo, hi in zip(be[:-1], be[1:])])
vb = ~np.isnan(bm)
ax.plot(bc[vb], bm[vb], 'o-', color='#e74c3c', lw=2.5, ms=6, zorder=5)

cbar = plt.colorbar(h[3], ax=ax, shrink=0.8); cbar.set_label('Count (log)', fontsize=9)
ax.set_xlabel(f'Morgan Tanimoto (r=2, nBits={FP_BITS})')
ax.set_ylabel('DreaMS Cosine Similarity')
ax.set_title(f'(c) Cos vs Tanimoto — r={r_val:.4f}  ρ={rho:.4f}')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('dreams_zeroshot_baseline.png', dpi=150, bbox_inches='tight')
print('  Saved: dreams_zeroshot_baseline.png', flush=True)

results = {
    'dataset': 'MassSpecGym',
    'n_spectra_total': int(n_total),
    'n_spectra_mh': int(n_mh),
    'n_unique_smiles': int(N),
    'n_pairwise_tanimoto': int(total_computed),
    'auc': float(auc),
    'pearson_r': float(r_val),
    'pearson_p': float(p_val),
    'spearman_rho': float(rho),
    'spearman_p': float(sp_val),
    'pearson_r_raw': float(r_raw),
    'spearman_rho_raw': float(rho_raw),
    'n_corr_pairs': int(len(corr_i)),
    'n_auc_pairs': int(len(lb_arr)),
    'n_peaks': int(N_PEAKS),
    'n_bins': int(N_BINS),
    'bin_size': float(BIN_SIZE),
    'target_per_bin': int(THLD),
    'fp_bits': int(FP_BITS),
    'empty_bins': [int(b) for b in empty_bins],
    'cos_pos': float(cp),
    'cos_neg': float(cn),
    'paper_pearson': 0.634,
    'paper_spearman': 0.629,
    'paper_auc': 0.85
}
out_path = 'dreams_zeroshot_baseline.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'  Saved: {out_path} ({len(json.dumps(results))} bytes)', flush=True)
print(f'\nTotal time: {(time.time()-t_start)/60:.1f} min', flush=True)
