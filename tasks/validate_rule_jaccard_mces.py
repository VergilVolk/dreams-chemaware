"""
Task 0: Rule Jaccard vs MCES Correlation Validation (CRITICAL)

验证 335 条化学规则的 Jaccard 重叠度与分子结构 MCES 是否正相关。
结果决定模块一微调用"规则对齐"还是"分子对齐"路线。

步骤:
  1. annotated01 → 按 InChIKey 去重 → 每个 IK 选总强度最大的谱图
  2. 随机采样 5,000 对 → 覆盖不同 MCES 区间
  3. MCES: 复用 T1 已有数据 + RDKit MCS 近似
  4. 规则 Jaccard: ChemicalRuleEngine 335 维命中向量 → 交集/并集
  5. 统计 + 可视化

用法: python tasks/validate_rule_jaccard_mces.py
"""
import json, os, sys, csv, time
from collections import defaultdict, Counter
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from tasks.build_utils import load_indices
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, DataStructs
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)
OUT_DIR = 'data/validation/rule_mces_correlation'
os.makedirs(OUT_DIR, exist_ok=True)

N_SUBSET = 5000   # 子集大小, C(5000,2)=12.5M 足够覆盖所有 Tanimoto 区间
N_PEAKS = 60     # 论文默认, 60×59/2 = 1770 峰对 vs 128×127/2 = 8128
FP_BITS = 2048

# ===================================================================
# 1. Load indices + filter to valid IKs (no MGF scan needed!)
# ===================================================================
print('[1] Loading indices...')
idx = load_indices()
ik_to_smi = idx['ik_to_smi']

# Filter to IKs with valid SMILES and fingerprint-able
valid_iks = []
for ik in ik_to_smi:
    smi = ik_to_smi[ik]
    if not smi: continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    valid_iks.append(ik)
print(f'  Valid IKs: {len(valid_iks)}')

# ===================================================================
# 2. Load existing T1 MCES data (reuse!)
# ===================================================================
print('\n[2] Loading existing MCES data from T1...')
existing_mces = {}  # (ik_a, ik_b) sorted → mces_raw
t1_pairs_path = 'tasks/T1_near_isomers/test_cases/pairs.json'
t1_count = 0
if os.path.exists(t1_pairs_path):
    with open(t1_pairs_path) as f:
        t1_data = json.load(f)
    for p in t1_data['positive'] + t1_data['negative_hard'] + t1_data['negative_easy']:
        if 'mces_raw' not in p: continue
        a, b = p['ik_a'][:14], p['ik_b'][:14]
        key = (a, b) if a < b else (b, a)
        existing_mces[key] = p['mces_raw']
        t1_count += 1
print(f'  Loaded {t1_count} existing MCES values from T1 pairs.json')

# ===================================================================
# 3. Sample N_SUBSET molecules + stratified pair sampling
# ===================================================================
print(f'\n[3] Subset {N_SUBSET} molecules, stratified pair sampling...')

# Load rule vector cache first (to filter subset to only IKs with both SMILES + spectra)
CACHE_PATH = 'tasks/_cache/rule_vectors/ik_to_rvec.npz'
if not os.path.exists(CACHE_PATH):
    print(f'ERROR: Run first: python tasks/precompute_rule_vectors.py'); sys.exit(1)
cache = np.load(CACHE_PATH)
cached_iks = set(cache.keys())
print(f'  Rule vector cache: {len(cached_iks)} IKs')

# Pick subset from IKs that have BOTH valid SMILES AND rule vectors
eligible_iks = sorted(set(valid_iks) & cached_iks)
print(f'  Eligible (SMILES + rule vectors): {len(eligible_iks)}')
subset_iks = rng.choice(eligible_iks, min(N_SUBSET, len(eligible_iks)), replace=False)
subset_iks = sorted(set(subset_iks))
N_sub = len(subset_iks)
print(f'  Subset: {N_sub} molecules → C({N_sub},2) = {N_sub*(N_sub-1)//2:,} candidate pairs')

# Compute fingerprints for subset only
ik_fp = {}
for ik in tqdm(subset_iks, desc='Fingerprints'):
    mol = Chem.MolFromSmiles(ik_to_smi[ik])
    ik_fp[ik] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, FP_BITS)

# Stratified sampling from subset
strata = [(0, 0.05, 200), (0.05, 0.15, 250), (0.15, 0.35, 250), (0.35, 1.01, 300)]
sampled_pairs = []
for lo, hi, n_target in strata:
    n = 0; tried = 0
    while n < n_target and tried < 100000:
        ai, bi = rng.choice(N_sub, 2, replace=False)
        a, b = subset_iks[ai], subset_iks[bi]
        key = (a, b) if a < b else (b, a)
        if key in sampled_pairs: continue
        tan = DataStructs.TanimotoSimilarity(ik_fp[a], ik_fp[b])
        if lo <= tan < hi:
            sampled_pairs.append(key)
            n += 1
        tried += 1
    print(f'    [{lo:.2f},{hi:.2f}): {n}/{n_target} (tried {tried})')

print(f'  Sampled: {len(sampled_pairs)} pairs')

# ===================================================================
# 4. MCES computation (reuse T1 + RDKit MCS fallback)
# ===================================================================
print('\n[4] Computing MCES for sampled pairs...')

def compute_mces_approx(smi_a, smi_b):
    """RDKit MCS-based MCES approximation: n_bonds_A + n_bonds_B - 2*n_bonds_MCS"""
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None: return None
    try:
        mcs = rdFMCS.FindMCS([mol_a, mol_b], timeout=1,
                             bondCompare=rdFMCS.BondCompare.CompareOrderExact)
        if mcs.numBonds == 0:
            # No common substructure → return total bonds (maximum distance)
            return mol_a.GetNumBonds() + mol_b.GetNumBonds()
        # Get the MCS molecule to count bonds
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        n_bonds_mcs = mcs_mol.GetNumBonds() if mcs_mol else mcs.numBonds
        mces_approx = mol_a.GetNumBonds() + mol_b.GetNumBonds() - 2 * n_bonds_mcs
        return max(0, mces_approx)
    except Exception:
        return None

pair_mces = {}; pair_smiles = {}
t1_hits = 0; mcs_fallback = 0; mces_failed = 0

for a, b in tqdm(sampled_pairs):
    pair_smiles[(a, b)] = (ik_to_smi.get(a, ''), ik_to_smi.get(b, ''))

    # Check T1 cache
    if (a, b) in existing_mces:
        pair_mces[(a, b)] = existing_mces[(a, b)]
        t1_hits += 1
    else:
        smi_a = ik_to_smi.get(a, '')
        smi_b = ik_to_smi.get(b, '')
        mces_val = compute_mces_approx(smi_a, smi_b)
        if mces_val is not None:
            pair_mces[(a, b)] = mces_val
            mcs_fallback += 1
        else:
            mces_failed += 1

print(f'  MCES: {t1_hits} from T1 cache, {mcs_fallback} from RDKit MCS, {mces_failed} failed')

# ===================================================================
# 5. Rule Jaccard — use cache already loaded in step 3
# ===================================================================
print('\n[5] Building rule vector lookup from cache...')
ik_to_rvec = {ik: cache[ik] for ik in cache.keys()}
N_RULES = ik_to_rvec[list(ik_to_rvec.keys())[0]].shape[0]
print(f'  {len(ik_to_rvec)} rule vectors ({N_RULES} rules)')

# Filter pairs: only keep those with BOTH MCES and rule vectors
valid_pairs = []
n_miss_mces = 0; n_miss_rvec = 0
for a, b in sampled_pairs:
    if (a, b) not in pair_mces:
        n_miss_mces += 1; continue
    if a not in ik_to_rvec or b not in ik_to_rvec:
        n_miss_rvec += 1; continue
    valid_pairs.append((a, b))

# Compute Jaccard for valid pairs
print(f'  Valid pairs for Jaccard: {len(valid_pairs)} '
      f'(miss MCES={n_miss_mces}, miss rvec={n_miss_rvec})')

pair_jaccard = {}
for a, b in tqdm(valid_pairs, desc='Jaccard'):
    va = ik_to_rvec[a]; vb = ik_to_rvec[b]
    intersection = (va & vb).sum()
    union = (va | vb).sum()
    pair_jaccard[(a, b)] = float(intersection / union) if union > 0 else 0.0

# ===================================================================
# 6. Save CSV + Statistics
# ===================================================================
print('\n[6] Saving results...')

# sampled_pairs.csv
with open(f'{OUT_DIR}/sampled_pairs.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ik_a', 'ik_b', 'smiles_a', 'smiles_b', 'mces', 'jaccard'])
    for a, b in sampled_pairs:
        smi_a, smi_b = pair_smiles.get((a, b), ('', ''))
        mces = pair_mces.get((a, b))
        jac = pair_jaccard.get((a, b))
        writer.writerow([a, b, smi_a[:120], smi_b[:120],
                         f'{mces:.1f}' if mces is not None else '',
                         f'{jac:.4f}' if jac is not None else ''])

# pair_mces_jaccard.csv (only complete pairs)
mces_arr = []; jac_arr = []
with open(f'{OUT_DIR}/pair_mces_jaccard.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ik_a', 'ik_b', 'mces', 'jaccard'])
    for a, b in valid_pairs:
        mces = pair_mces[(a, b)]
        jac = pair_jaccard.get((a, b), 0.0)
        writer.writerow([a, b, f'{mces:.1f}', f'{jac:.4f}'])
        mces_arr.append(mces)
        jac_arr.append(jac)

mces_arr = np.array(mces_arr); jac_arr = np.array(jac_arr)
n_complete = len(mces_arr)
print(f'  Complete pairs (both MCES+Jaccard): {n_complete}')

# Statistics
r_val, p_val = pearsonr(mces_arr, jac_arr)
rho, sp_val = spearmanr(mces_arr, jac_arr)
print(f'\n  Pearson r:  {r_val:.4f} (p={p_val:.2e})')
print(f'  Spearman ρ: {rho:.4f} (p={sp_val:.2e})')

# By MCES group
groups = [(0, 2, 'MCES 0-2 (near-isomers)'),
          (3, 5, 'MCES 3-5 (analogs)'),
          (6, 10, 'MCES 6-10 (different isomers)'),
          (11, 999, 'MCES >10 (unrelated)')]
print(f'\n  By MCES group:')
group_stats = {}
for lo, hi, label in groups:
    mask = (mces_arr >= lo) & (mces_arr <= hi)
    n_g = mask.sum()
    if n_g > 0:
        mean_j = jac_arr[mask].mean()
        std_j = jac_arr[mask].std()
        print(f'    {label}: n={n_g}  Jaccard mean={mean_j:.4f} std={std_j:.4f}')
        group_stats[label] = {'n': int(n_g), 'jaccard_mean': float(mean_j), 'jaccard_std': float(std_j)}

# correlation_report.json
report = {
    'n_pairs_sampled': int(len(sampled_pairs)),
    'n_pairs_complete': int(n_complete),
    'n_rules': N_RULES,
    'pearson_r': float(r_val),
    'pearson_p': float(p_val),
    'spearman_rho': float(rho),
    'spearman_p': float(sp_val),
    'mces_range': [float(mces_arr.min()), float(mces_arr.max())],
    'jaccard_range': [float(jac_arr.min()), float(jac_arr.max())],
    'jaccard_mean': float(jac_arr.mean()),
    'jaccard_std': float(jac_arr.std()),
    'by_mces_group': group_stats,
    't1_cache_hits': t1_hits,
    'mcs_fallback': mcs_fallback,
    'mces_failed': mces_failed,
}
with open(f'{OUT_DIR}/correlation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ===================================================================
# 7. Plot
# ===================================================================
print('\n[7] Plotting...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Rule Jaccard vs MCES Correlation\n'
             f'Pearson r={r_val:.4f}  Spearman ρ={rho:.4f}  N={n_complete}',
             fontsize=13, fontweight='bold')

# (a) Scatter
ax = axes[0]
ax.scatter(mces_arr, jac_arr, alpha=0.3, s=8, c='#3498db', edgecolors='none', rasterized=True)
z = np.polyfit(mces_arr, jac_arr, 1)
x_line = np.linspace(mces_arr.min(), mces_arr.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r-', lw=2, label=f'Pearson r={r_val:.4f}')
ax.set_xlabel('MCES (structural distance)'); ax.set_ylabel('Rule Jaccard')
ax.set_title('(a) Rule Jaccard vs MCES')
ax.legend(); ax.grid(True, alpha=0.3)

# (b) Binned bar chart
ax = axes[1]
labels = []; means = []; stds = []
for lo, hi, label in groups:
    mask = (mces_arr >= lo) & (mces_arr <= hi)
    if mask.sum() > 0:
        labels.append(label.split('(')[0].strip())
        means.append(jac_arr[mask].mean())
        stds.append(jac_arr[mask].std())
x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, means, yerr=stds, color=['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'],
              capsize=5, alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel('Mean Rule Jaccard'); ax.set_title('(b) Jaccard by MCES Group')
ax.grid(True, alpha=0.3, axis='y')
# Add count labels
for i, (lo, hi, label) in enumerate(groups):
    mask = (mces_arr >= lo) & (mces_arr <= hi)
    if mask.sum() > 0:
        ax.text(i, means[i] + stds[i] + 0.005, f'n={mask.sum()}',
                ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/rule_jaccard_vs_mces.png', dpi=150, bbox_inches='tight')
print(f'  Saved: {OUT_DIR}/rule_jaccard_vs_mces.png')

print(f'\n=== VALIDATION COMPLETE ===')
print(f'  Output: {OUT_DIR}/')
print(f'  Key result: Pearson r={r_val:.4f}, Spearman ρ={rho:.4f}')
print(f'  Interpretation: ', end='')
if abs(r_val) > 0.5:
    print('STRONG correlation → rules CAN guide structure-aware training')
elif abs(r_val) > 0.3:
    print('MODERATE correlation → rules partially reflect structure, may need augmentation')
else:
    print('WEAK correlation → rules alone insufficient, need molecular alignment strategy')
