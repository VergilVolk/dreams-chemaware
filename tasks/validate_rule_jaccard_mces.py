"""
Task 0: Rule Jaccard vs MCES Correlation Validation — V4

用已有 T1 MCES 分层数据 + 随机补充，确保 MCES 全谱覆盖。

用法: python tasks/validate_rule_jaccard_mces.py                         # 335 主规则
       python tasks/validate_rule_jaccard_mces.py --use-massbank          # 全量 ~3,486 条规则
"""
import json, os, sys, csv, time, argparse
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')
from tasks.build_utils import load_indices
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--use-massbank', action='store_true',
                    help='Use full ~3,486 rule cache (default: 335 main rules)')
args = parser.parse_args()

rng = np.random.RandomState(42)
suffix = '_massbank' if args.use_massbank else ''
OUT_DIR = f'data/validation/rule_mces_correlation{suffix}'
os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================================
# 1. Load everything
# ===================================================================
print('[1] Loading data...')

# Indices → SMILES lookup (14-char)
idx = load_indices()
smi_lookup = {}
for ik, smi in idx['ik_to_smi'].items():
    smi_lookup[ik[:14]] = smi
_ = [smi_lookup.pop(k, None) for k in list(smi_lookup.keys()) if len(k) > 14]
print(f'  SMILES lookup: {len(smi_lookup)} IKs')

# Rule vector cache
CACHE_PATH = 'tasks/_cache/rule_vectors/ik_to_rvec_massbank.npz' if args.use_massbank \
             else 'tasks/_cache/rule_vectors/ik_to_rvec.npz'
cache = np.load(CACHE_PATH)
ik_to_rvec = {ik: cache[ik] for ik in cache.keys()}
N_RULES = ik_to_rvec[list(ik_to_rvec.keys())[0]].shape[0]
print(f'  Rule vectors: {len(ik_to_rvec)} IKs, {N_RULES} rules')

# T1 pairs
t1_path = 'tasks/T1_near_isomers/test_cases/pairs.json'
with open(t1_path) as f:
    t1 = json.load(f)
print(f'  T1: positive={len(t1["positive"])}, hard={len(t1["negative_hard"])}, easy={len(t1["negative_easy"])}')

# Intersection check
t1_iks = set()
for k in ['positive', 'negative_hard', 'negative_easy']:
    for p in t1[k]:
        t1_iks.add(p['ik_a'][:14])
        t1_iks.add(p['ik_b'][:14])
print(f'  T1 unique IKs: {len(t1_iks)}')
print(f'  T1 ∩ cache: {len(t1_iks & set(ik_to_rvec.keys()))}')
print(f'  T1 ∩ SMILES: {len(t1_iks & set(smi_lookup.keys()))}')

# ===================================================================
# 2. MCES computation helper
# ===================================================================
print('\n[2] Build formula groups + MCES helper...')

def compute_mces(smi_a, smi_b):
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None: return None
    try:
        mcs = rdFMCS.FindMCS([mol_a, mol_b], timeout=1,
                             bondCompare=rdFMCS.BondCompare.CompareOrderExact)
        if mcs.numBonds == 0:
            return mol_a.GetNumBonds() + mol_b.GetNumBonds()
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        n_mcs = mcs_mol.GetNumBonds() if mcs_mol else mcs.numBonds
        return max(0, mol_a.GetNumBonds() + mol_b.GetNumBonds() - 2 * n_mcs)
    except Exception:
        return None

# Eligible: IK has cache + valid SMILES + RDKit Mol
eligible = sorted(set(ik_to_rvec.keys()) & set(smi_lookup.keys()))
print(f'  Eligible IKs: {len(eligible)}')

# Build formula groups from eligible IKs (for dense isomer sampling)
from rdkit.Chem import rdMolDescriptors
fm_to_iks = defaultdict(list)
for ik in eligible:
    mol = Chem.MolFromSmiles(smi_lookup[ik])
    if mol is not None:
        fm = rdMolDescriptors.CalcMolFormula(mol)
        fm_to_iks[fm].append(ik)

large_groups = [(fm, iks) for fm, iks in fm_to_iks.items() if len(iks) >= 3]
large_groups.sort(key=lambda x: len(x[1]), reverse=True)
total_in_groups = sum(len(iks) for _, iks in large_groups)
print(f'  Formula groups (≥3 mols): {len(large_groups)}, {total_in_groups} molecules')

# ===================================================================
# 3. Collect pairs: T1 + formula-dense + random
# ===================================================================
print('\n[3] Collecting pairs...')

pair_mces = {}   # (ik_a, ik_b) → mces
seen = set()

def add(a, b, mces):
    key = (a, b) if a < b else (b, a)
    if key not in seen:
        seen.add(key)
        pair_mces[key] = mces

# 3a. T1 positive: known MCES [0,2]
for p in t1['positive']:
    if 'mces_raw' not in p: continue
    add(p['ik_a'][:14], p['ik_b'][:14], p['mces_raw'])

# 3b. T1 negative_hard: known MCES [6,10]
for p in t1['negative_hard']:
    if 'mces_raw' not in p: continue
    add(p['ik_a'][:14], p['ik_b'][:14], p['mces_raw'])

print(f'  T1 cached: {len(pair_mces)} pairs')

# 3c. V5: Formula-based dense sampling — fills MCES 0-5 (isomers/analogs)
# Use top 500 largest formula groups, sample up to 30 intra-group pairs each
N_FORMULA_GROUPS = min(500, len(large_groups))
N_PAIRS_PER_GROUP = 30
print(f'  [V5] Sampling from {N_FORMULA_GROUPS} largest formula groups...')

intra_pairs_to_compute = []
for fm, iks in large_groups[:N_FORMULA_GROUPS]:
    n_sample = min(N_PAIRS_PER_GROUP, len(iks) * (len(iks) - 1) // 2)
    sampled = 0; tried = 0
    while sampled < n_sample and tried < n_sample * 10:
        ai, bi = rng.choice(len(iks), 2, replace=False)
        key = (iks[ai], iks[bi]) if iks[ai] < iks[bi] else (iks[bi], iks[ai])
        if key not in seen:
            seen.add(key)
            intra_pairs_to_compute.append(key)
            sampled += 1
        tried += 1

print(f'  Intra-formula pairs to compute MCES: {len(intra_pairs_to_compute)}')

# 3d. T1 negative_easy + random supplement
easy_pairs = []
for p in t1['negative_easy']:
    a, b = p['ik_a'][:14], p['ik_b'][:14]
    key = (a, b) if a < b else (b, a)
    if key not in seen:
        easy_pairs.append(key)
        seen.add(key)

N_RANDOM = 500
random_pairs = []
while len(random_pairs) < N_RANDOM:
    ai, bi = rng.choice(len(eligible), 2, replace=False)
    key = (eligible[ai], eligible[bi])
    key = (key[0], key[1]) if key[0] < key[1] else (key[1], key[0])
    if key not in seen:
        seen.add(key)
        random_pairs.append(key)

print(f'  T1 easy: {len(easy_pairs)}, Random: {len(random_pairs)}')

# ===================================================================
# 4. Compute MCES for all pairs without known values
# ===================================================================
print('\n[4] Computing MCES via RDKit MCS...')
to_compute = intra_pairs_to_compute + easy_pairs + random_pairs
print(f'  Need RDKit MCS: {len(to_compute)} pairs')

for a, b in tqdm(to_compute, desc='RDKit MCS'):
    smi_a = smi_lookup.get(a, '')
    smi_b = smi_lookup.get(b, '')
    mces = compute_mces(smi_a, smi_b)
    if mces is not None:
        pair_mces[(a, b)] = mces

print(f'  Total pairs with MCES: {len(pair_mces)}')

# MCES distribution
print(f'\n  MCES distribution:')
bins = [(0, 2, 'MCES 0-2'), (3, 5, 'MCES 3-5'),
        (6, 10, 'MCES 6-10'), (11, 999, 'MCES >10')]
mces_vals = list(pair_mces.values())
for lo, hi, label in bins:
    n = sum(1 for m in mces_vals if lo <= m <= hi)
    print(f'    {label}: {n}')

# ===================================================================
# 5. Jaccard
# ===================================================================
print('\n[5] Computing Jaccard...')
pair_data = []
for (a, b), mces in pair_mces.items():
    rv_a = ik_to_rvec.get(a)
    rv_b = ik_to_rvec.get(b)
    if rv_a is None or rv_b is None: continue
    inter = (rv_a & rv_b).sum()
    union = (rv_a | rv_b).sum()
    jac = float(inter / union) if union > 0 else 0.0
    pair_data.append((a, b, mces, jac))

mces_arr = np.array([p[2] for p in pair_data])
jac_arr = np.array([p[3] for p in pair_data])
n = len(pair_data)
print(f'  Complete: {n} pairs')

# ===================================================================
# 6. Statistics
# ===================================================================
print('\n[6] Statistics...')
r_val, p_val = pearsonr(mces_arr, jac_arr)
rho, sp_val = spearmanr(mces_arr, jac_arr)
print(f'  Pearson r:  {r_val:.4f} (p={p_val:.2e})')
print(f'  Spearman ρ: {rho:.4f} (p={sp_val:.2e})')

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
        mj = jac_arr[mask].mean(); sj = jac_arr[mask].std()
        print(f'    {label}: n={n_g}  Jaccard mean={mj:.4f} std={sj:.4f}')
        group_stats[label] = {'n': int(n_g), 'jaccard_mean': float(mj), 'jaccard_std': float(sj)}
    else:
        print(f'    {label}: n=0')

# ===================================================================
# 7. Save
# ===================================================================
print('\n[7] Saving...')

with open(f'{OUT_DIR}/pair_mces_jaccard.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ik_a', 'ik_b', 'smiles_a', 'smiles_b', 'mces', 'jaccard'])
    for a, b, mces, jac in pair_data:
        writer.writerow([a, b, smi_lookup.get(a, '')[:120], smi_lookup.get(b, '')[:120],
                         f'{mces:.1f}', f'{jac:.4f}'])

report = {
    'n_pairs_total': int(n),
    'n_rules': N_RULES,
    'pearson_r': float(r_val), 'pearson_p': float(p_val),
    'spearman_rho': float(rho), 'spearman_p': float(sp_val),
    'mces_range': [float(mces_arr.min()), float(mces_arr.max())],
    'jaccard_range': [float(jac_arr.min()), float(jac_arr.max())],
    'jaccard_mean': float(jac_arr.mean()), 'jaccard_std': float(jac_arr.std()),
    'by_mces_group': group_stats,
    'mces_distribution': {
        label: int(((mces_arr >= lo) & (mces_arr <= hi)).sum())
        for lo, hi, label in groups
    },
}
with open(f'{OUT_DIR}/correlation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ===================================================================
# 8. Plot
# ===================================================================
print('[8] Plotting...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Rule Jaccard vs MCES Correlation ({N_RULES} rules)\n'
             f'Pearson r={r_val:.4f}  Spearman ρ={rho:.4f}  N={n}',
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(mces_arr, jac_arr, alpha=0.3, s=8, c='#3498db', edgecolors='none', rasterized=True)
z = np.polyfit(mces_arr, jac_arr, 1)
xl = np.linspace(mces_arr.min(), mces_arr.max(), 100)
ax.plot(xl, np.polyval(z, xl), 'r-', lw=2, label=f'Pearson r={r_val:.4f}')
ax.set_xlabel('MCES (structural distance)'); ax.set_ylabel('Rule Jaccard')
ax.set_title('(a) Rule Jaccard vs MCES')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
labels = []; means = []; stds = []
for lo, hi, label in groups:
    mask = (mces_arr >= lo) & (mces_arr <= hi)
    if mask.sum() > 0:
        labels.append(label.split('(')[0].strip())
        means.append(jac_arr[mask].mean())
        stds.append(jac_arr[mask].std())

bar_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'][:len(labels)]
xp = np.arange(len(labels))
ax.bar(xp, means, yerr=stds, color=bar_colors, capsize=5, alpha=0.8)
ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel('Mean Rule Jaccard'); ax.set_title('(b) Jaccard by MCES Group')
ax.grid(True, alpha=0.3, axis='y')

bi = 0
for lo, hi, label in groups:
    mask = (mces_arr >= lo) & (mces_arr <= hi)
    if mask.sum() > 0:
        ax.text(bi, means[bi] + stds[bi] + 0.005, f'n={mask.sum()}', ha='center', fontsize=8)
        bi += 1

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/rule_jaccard_vs_mces.png', dpi=150, bbox_inches='tight')
print(f'  Saved: {OUT_DIR}/rule_jaccard_vs_mces.png')

print(f'\n=== VALIDATION COMPLETE ===')
print(f'  Pearson r={r_val:.4f}, Spearman ρ={rho:.4f}')
if abs(r_val) > 0.5:
    print('  STRONG → rules CAN guide structure-aware training')
elif abs(r_val) > 0.3:
    print('  MODERATE → rules partially reflect structure')
else:
    print('  WEAK → need molecular alignment strategy')
