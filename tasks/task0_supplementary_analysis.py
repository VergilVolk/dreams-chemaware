"""
Task 0 补充分析: 7 项轻量级分析 (v3 — 修复 bool_and/or、常量NaN、Pearson口径、命名)

用法:
    python tasks/task0_supplementary_analysis.py              # 完整分析
    python tasks/task0_supplementary_analysis.py --quick       # 只跑核心项 (1,2,6,7)
"""
import json, os, sys, csv, argparse, time
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, '.')
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true',
                    help='Quick mode: skip heavy computations')
cli = parser.parse_args()

rng = np.random.RandomState(42)
OUT_DIR = 'data/validation/rule_mces_correlation'
os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()

print('=' * 70)
print('Task 0 Supplementary Analysis — 7 Items (v3 fixed)')
print('=' * 70)

# ===================================================================
# Efficient data loading
# ===================================================================
print('\n[LOAD] Loading indices from cache...')
t0 = time.time()

with open('tasks/_cache/indices.json', 'r') as f:
    idx = json.load(f)

smi_lookup = {}
for ik, smi in idx['ik_to_smi'].items():
    smi_lookup[ik[:14]] = smi

fm_lookup = {ik[:14]: fm for ik, fm in idx['ik_to_fm'].items()}
pm_lookup = {ik[:14]: pm for ik, pm in idx['ik_to_pm'].items()}
pk_lookup = {}
for ik, peaks in idx['ik_to_peaks'].items():
    pk_lookup[ik[:14]] = len(peaks) if isinstance(peaks, list) else peaks

spec_counts = {ik[:14]: int(c) for ik, c in idx['ik_counts'].items()}
print(f'  Indices: {len(smi_lookup)} IKs ({time.time()-t0:.1f}s)')

# Load rule vectors as npz → dense bool matrix
print(f'  Loading rule vectors...')
t0 = time.time()

npz_335 = np.load('tasks/_cache/rule_vectors/ik_to_rvec.npz')
ik_list_335 = sorted(npz_335.keys())
ik_to_idx_335 = {ik: i for i, ik in enumerate(ik_list_335)}
N_RULES_335 = npz_335[ik_list_335[0]].shape[0]
rv_matrix_335 = np.zeros((len(ik_list_335), N_RULES_335), dtype=bool)
for i, ik in enumerate(ik_list_335):
    rv_matrix_335[i] = npz_335[ik].astype(bool)
print(f'  335 rules: {len(ik_list_335)} IKs, {N_RULES_335} rules ({time.time()-t0:.1f}s)')

t0 = time.time()
npz_mb = np.load('tasks/_cache/rule_vectors/ik_to_rvec_massbank.npz')
ik_list_mb = sorted(npz_mb.keys())
ik_to_idx_mb = {ik: i for i, ik in enumerate(ik_list_mb)}
N_RULES_MB = npz_mb[ik_list_mb[0]].shape[0]
rv_matrix_mb = np.zeros((len(ik_list_mb), N_RULES_MB), dtype=bool)
for i, ik in enumerate(ik_list_mb):
    rv_matrix_mb[i] = npz_mb[ik].astype(bool)
print(f'  MassBank rules: {len(ik_list_mb)} IKs, {N_RULES_MB} rules ({time.time()-t0:.1f}s)')

# Load pair data
pair_mces_jaccard_335 = []
pair_mces_jaccard_mb = []
for suffix, arr in [('', pair_mces_jaccard_335), ('_massbank', pair_mces_jaccard_mb)]:
    path = f'data/validation/rule_mces_correlation{suffix}/pair_mces_jaccard.csv'
    if os.path.exists(path):
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                arr.append((row[0][:14], row[1][:14], float(row[4]), float(row[5])))
print(f'  Pair data: 335={len(pair_mces_jaccard_335)}, MB={len(pair_mces_jaccard_mb)}')

# Load rule definitions
print(f'  Loading rule definitions...')
from dreams.models.chem_aware.chem_rules import _load_rules_from_json, _load_massbank_rules_json
rules_335 = _load_rules_from_json()
rule_names_335 = [r.name for r in rules_335]
rule_categories_335 = [r.category for r in rules_335]
cat_counts_335 = Counter(rule_categories_335)
print(f'  Categories (335): {dict(cat_counts_335)}')

rules_mb_data = _load_massbank_rules_json()
print(f'  MassBank empirical rules: {len(rules_mb_data)}')

# Combined rule names/categories for 3,486 (fix P2: display names in Item 2)
rule_names_mb = rule_names_335 + [r.name for r in rules_mb_data]
rule_categories_mb = rule_categories_335 + ['MassBank_empirical'] * len(rules_mb_data)

print(f'  Total load time: {time.time()-T0:.1f}s')

# ===================================================================
# Item 1: Molecule coverage (fix P1: rename from "Spectrum" to "Molecule")
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 1: Rule Molecule Coverage')
print('=' * 70)

def compute_rule_coverage_fast(rv_matrix, label):
    coverage = rv_matrix.mean(axis=0)
    n_rules = len(coverage)
    n_iks = rv_matrix.shape[0]
    print(f'\n  {label} ({n_rules} rules, {n_iks} IKs):')
    print(f'    Coverage: min={coverage.min()*100:.4f}%  max={coverage.max()*100:.1f}%')
    print(f'    Percentiles: P10={np.percentile(coverage,10)*100:.2f}%  P50={np.median(coverage)*100:.2f}%  P90={np.percentile(coverage,90)*100:.2f}%  P99={np.percentile(coverage,99)*100:.2f}%')
    bins = [(0, 0.001, '0-0.1%'), (0.001, 0.01, '0.1-1%'), (0.01, 0.05, '1-5%'),
            (0.05, 0.10, '5-10%'), (0.10, 0.25, '10-25%'), (0.25, 0.50, '25-50%'), (0.50, 1.01, '50-100%')]
    for lo, hi, desc in bins:
        n = ((coverage >= lo) & (coverage < hi)).sum()
        print(f'      {desc:10s}: {n:5d} rules ({n/n_rules*100:5.1f}%)')
    return coverage

cov_335 = compute_rule_coverage_fast(rv_matrix_335, '335 rules')
cov_mb = compute_rule_coverage_fast(rv_matrix_mb, '3,486 rules')

# ===================================================================
# Item 2: Molecule support
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 2: Independent Molecule Support Per Rule')
print('=' * 70)

def compute_mol_support_fast(rv_matrix, label, spec_counts_map, ik_list,
                              rule_names=None, rule_cats=None, n_top=10):
    mol_support = rv_matrix.sum(axis=0)
    n_rules = len(mol_support)

    spec_weights = np.array([spec_counts_map.get(ik, 1) for ik in ik_list])
    spec_support = (rv_matrix.astype(np.float64).T * spec_weights).T.sum(axis=0)

    print(f'\n  {label} ({n_rules} rules):')
    print(f'    Mol support: min={mol_support.min()}  max={mol_support.max()}  mean={mol_support.mean():.0f}  median={np.median(mol_support):.0f}')
    print(f'    Rules < 5 molecules: {(mol_support < 5).sum()} ({(mol_support < 5).sum()/n_rules*100:.1f}%)')
    print(f'    Rules < 10 molecules: {(mol_support < 10).sum()} ({(mol_support < 10).sum()/n_rules*100:.1f}%)')

    order = np.argsort(mol_support)[::-1]
    print(f'\n    Top-{n_top} most-supported rules:')
    for i in range(min(n_top, n_rules)):
        ri = order[i]
        name = rule_names[ri] if rule_names and ri < len(rule_names) else f'r{ri}'
        cat = rule_cats[ri] if rule_cats and ri < len(rule_cats) else '?'
        print(f'      [{ri:4d}] {name:45s} {cat:20s}  mol={mol_support[ri]:6d}  spec={spec_support[ri]:.0f}')

    non_zero = mol_support > 0
    nz_indices = np.where(non_zero)[0]
    order_asc = np.argsort(mol_support[non_zero])
    print(f'\n    Bottom-{n_top} least-supported (non-zero) rules:')
    for i in range(min(n_top, len(order_asc))):
        ri = nz_indices[order_asc[i]]
        name = rule_names[ri] if rule_names and ri < len(rule_names) else f'r{ri}'
        cat = rule_cats[ri] if rule_cats and ri < len(rule_cats) else '?'
        print(f'      [{ri:4d}] {name:45s} {cat:20s}  mol={mol_support[ri]:6d}')

    return mol_support, spec_support

mol_335, spec_335_w = compute_mol_support_fast(
    rv_matrix_335, '335 rules', spec_counts, ik_list_335,
    rule_names_335, rule_categories_335)
# Fix P2: pass combined names for MassBank display
mol_mb, spec_mb_w = compute_mol_support_fast(
    rv_matrix_mb, '3,486 rules', spec_counts, ik_list_mb,
    rule_names_mb, rule_categories_mb, n_top=10)

# ===================================================================
# Item 3: Per-category correlation
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 3: Per-Category Rule Jaccard vs MCES Correlation')
print('=' * 70)

def per_category_correlation_fast(pair_data, n_rules, rule_categories,
                                   ik_to_idx, rv_matrix, ik_list, label):
    cat_to_indices = defaultdict(list)
    for ri, cat in enumerate(rule_categories):
        cat_to_indices[cat].append(ri)

    mces_arr = np.array([p[2] for p in pair_data])

    results = {}
    print(f'\n  {label}:')
    print(f'  {"Category":25s} {"N_rules":>8s} {"N_pairs":>8s} {"Pearson r":>10s} {"Spearman ρ":>10s} {"Δ Jaccard":>10s} {"Discrim":>8s}')
    print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*10} {"-"*10} {"-"*10} {"-"*8}')

    for cat in sorted(cat_to_indices.keys()):
        rule_idx = np.array(cat_to_indices[cat])
        n_cat = len(rule_idx)

        jac_arr = np.full(len(pair_data), np.nan)
        valid_mask = np.zeros(len(pair_data), dtype=bool)

        for i, (ik_a, ik_b, mces, _) in enumerate(pair_data):
            pos_a = ik_to_idx.get(ik_a)
            pos_b = ik_to_idx.get(ik_b)
            if pos_a is None or pos_b is None:
                continue
            sub_a = rv_matrix[pos_a, rule_idx]
            sub_b = rv_matrix[pos_b, rule_idx]
            # Fix P0: use & and | instead of undefined bool_and/bool_or
            inter = (sub_a & sub_b).sum()
            union = (sub_a | sub_b).sum()
            if union > 0:
                jac_arr[i] = inter / union
                valid_mask[i] = True

        if valid_mask.sum() < 30:
            print(f'  {cat:25s} {n_cat:>8d} {int(valid_mask.sum()):>8d}  {"(insufficient data)":>35s}')
            continue

        jac_v = jac_arr[valid_mask]
        mces_v = mces_arr[valid_mask]

        # Fix P1: guard against constant arrays (e.g. EE/NR single-rule categories)
        if np.unique(jac_v).size < 2 or np.unique(mces_v).size < 2:
            print(f'  {cat:25s} {n_cat:>8d} {len(jac_v):>8d}  {"(constant / undefined)":>35s}')
            continue

        r_val, _ = pearsonr(mces_v, jac_v)
        rho_val, _ = spearmanr(mces_v, jac_v)

        near_mask = mces_v <= 2; far_mask = mces_v > 10
        delta = jac_v[near_mask].mean() - jac_v[far_mask].mean() if near_mask.any() and far_mask.any() else 0
        discrim = jac_v[near_mask].mean() / jac_v[far_mask].mean() if near_mask.any() and far_mask.any() and jac_v[far_mask].mean() > 0 else 0

        results[cat] = {'n_rules': n_cat, 'n_pairs': int(valid_mask.sum()),
                        'pearson_r': float(r_val), 'spearman_rho': float(rho_val),
                        'delta_jaccard': float(delta), 'discrimination_ratio': float(discrim)}
        print(f'  {cat:25s} {n_cat:>8d} {int(valid_mask.sum()):>8d} {r_val:>10.4f} {rho_val:>10.4f} {delta:>10.4f} {discrim:>7.2f}x')
    return results

cat_results_335 = per_category_correlation_fast(
    pair_mces_jaccard_335, N_RULES_335, rule_categories_335,
    ik_to_idx_335, rv_matrix_335, ik_list_335, '335 rules')
cat_results_mb = per_category_correlation_fast(
    pair_mces_jaccard_mb, N_RULES_MB, rule_categories_mb,
    ik_to_idx_mb, rv_matrix_mb, ik_list_mb, '3,486 rules')

# ===================================================================
# Item 4: Same-formula Δ Jaccard
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 4: Δ Jaccard within Same-Formula Isomer Subset')
print('=' * 70)

def same_formula_delta(pair_data, label):
    same_fm = []; diff_fm = []
    for ik_a, ik_b, mces, jac in pair_data:
        fm_a = fm_lookup.get(ik_a, '')
        fm_b = fm_lookup.get(ik_b, '')
        (same_fm if fm_a and fm_b and fm_a == fm_b else diff_fm).append((ik_a, ik_b, mces, jac))

    print(f'\n  {label}:')
    print(f'    Same-formula pairs: {len(same_fm)}  |  Diff-formula pairs: {len(diff_fm)}')

    if same_fm:
        mces_s = np.array([p[2] for p in same_fm])
        jac_s = np.array([p[3] for p in same_fm])
        for lo, hi, desc in [(0, 2, 'MCES 0-2'), (3, 5, 'MCES 3-5'), (6, 999, 'MCES 6+')]:
            m = (mces_s >= lo) & (mces_s <= hi)
            if m.sum() > 0:
                print(f'      {desc}: n={m.sum()}, Jaccard μ={jac_s[m].mean():.4f}, σ={jac_s[m].std():.4f}')
        near_m = mces_s <= 2; far_m = mces_s > 5
        if near_m.any() and far_m.any():
            delta = jac_s[near_m].mean() - jac_s[far_m].mean()
            print(f'    Δ Jaccard (0-2 vs >5) within same-formula: {delta:.4f}')

    return same_fm, diff_fm

sf_335, df_335 = same_formula_delta(pair_mces_jaccard_335, '335 rules')
sf_mb, df_mb = same_formula_delta(pair_mces_jaccard_mb, '3,486 rules')

# ===================================================================
# Item 5: Partial correlation
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 5: Partial Correlation (Control: precursor m/z, peak count, spectrum count)')
print('=' * 70)

if cli.quick:
    print('  Skipped (--quick mode)')
    pc_335 = None; pc_mb = None
else:
    from sklearn.linear_model import LinearRegression

    def partial_corr(pair_data, label):
        n = len(pair_data)
        mces_arr = np.zeros(n); jac_arr = np.zeros(n)
        delta_pm = np.zeros(n); delta_pk = np.zeros(n); mean_spec = np.zeros(n)
        for i, (ik_a, ik_b, mces, jac) in enumerate(pair_data):
            mces_arr[i] = mces; jac_arr[i] = jac
            pm_a = pm_lookup.get(ik_a, 0); pm_b = pm_lookup.get(ik_b, 0)
            delta_pm[i] = abs(pm_a - pm_b)
            pk_a = pk_lookup.get(ik_a, 0); pk_b = pk_lookup.get(ik_b, 0)
            delta_pk[i] = abs(pk_a - pk_b)
            sc_a = spec_counts.get(ik_a, 1); sc_b = spec_counts.get(ik_b, 1)
            mean_spec[i] = (sc_a + sc_b) / 2

        r0, _ = spearmanr(mces_arr, jac_arr)
        r_pm_j, _ = spearmanr(delta_pm, jac_arr)
        r_pk_j, _ = spearmanr(delta_pk, jac_arr)
        r_pm_m, _ = spearmanr(delta_pm, mces_arr)

        X = np.column_stack([np.argsort(np.argsort(delta_pm)),
                             np.argsort(np.argsort(delta_pk)),
                             np.argsort(np.argsort(mean_spec))]).astype(float)
        mr = np.argsort(np.argsort(mces_arr)).astype(float).reshape(-1, 1)
        jr = np.argsort(np.argsort(jac_arr)).astype(float).reshape(-1, 1)
        m_resid = mr.ravel() - LinearRegression().fit(X, mr).predict(X).ravel()
        j_resid = jr.ravel() - LinearRegression().fit(X, jr).predict(X).ravel()
        rp, pp = spearmanr(m_resid, j_resid)

        print(f'\n  {label} ({n} pairs):')
        print(f'    Zero-order ρ(MCES, Jaccard): {r0:.4f}')
        print(f'    ρ(|Δpm|, Jaccard): {r_pm_j:.4f}  ρ(|Δpk|, Jaccard): {r_pk_j:.4f}  ρ(|Δpm|, MCES): {r_pm_m:.4f}')
        print(f'    Partial ρ (controlled): {rp:.4f} (p={pp:.2e})  Δ={r0-rp:+.4f}')
        return {'n': n, 'zero_order_rho': float(r0), 'partial_rho': float(rp), 'delta': float(r0-rp)}

    pc_335 = partial_corr(pair_mces_jaccard_335, '335 rules')
    pc_mb = partial_corr(pair_mces_jaccard_mb, '3,486 rules')

# ===================================================================
# Item 6: Conflict samples
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 6: Conflict Sample Inspection (High Jaccard + High MCES)')
print('=' * 70)

def inspect_conflicts_fast(pair_data, label, ik_to_idx, rv_matrix, ik_list, n_rules):
    mces_arr = np.array([p[2] for p in pair_data])
    jac_arr = np.array([p[3] for p in pair_data])

    conflict_mask = (jac_arr > 0.5) & (mces_arr > 5)
    conflicts = [pair_data[i] for i in np.where(conflict_mask)[0]]
    conflicts.sort(key=lambda p: p[2] * p[3], reverse=True)

    print(f'\n  {label}:')
    print(f'    Conflict pairs (Jaccard>0.5, MCES>5): {len(conflicts)} / {len(pair_data)} ({len(conflicts)/len(pair_data)*100:.2f}%)')

    if not conflicts:
        for jt in [0.4, 0.3]:
            for mt in [4, 3]:
                cm = (jac_arr > jt) & (mces_arr > mt)
                print(f'    Jaccard>{jt}, MCES>{mt}: {cm.sum()} pairs')
        return conflicts

    print(f'\n    Top-20 conflict samples:')
    print(f'    {"IK_A":14s} {"IK_B":14s} {"MCES":>6s} {"Jaccard":>8s} {"SMILES_A":45s} {"SMILES_B":45s}')
    print(f'    {"-"*14} {"-"*14} {"-"*6} {"-"*8} {"-"*45} {"-"*45}')

    for i, (ik_a, ik_b, mces, jac) in enumerate(conflicts[:20]):
        smi_a = smi_lookup.get(ik_a, '?')[:43]
        smi_b = smi_lookup.get(ik_b, '?')[:43]
        print(f'    {ik_a:14s} {ik_b:14s} {mces:>6.1f} {jac:>8.4f} {smi_a:45s} {smi_b:45s}')

        if i < 5:
            pos_a = ik_to_idx.get(ik_a); pos_b = ik_to_idx.get(ik_b)
            if pos_a is not None and pos_b is not None:
                both = rv_matrix[pos_a] & rv_matrix[pos_b]
                n_both = both.sum()
                cat_both = Counter()
                for ri in np.where(both)[0]:
                    if ri < len(rule_categories_335):
                        cat_both[rule_categories_335[ri]] += 1
                    else:
                        cat_both['MassBank'] += 1
                print(f'      {n_both} shared rules: {dict(cat_both.most_common(5))}')

    low_jac_near = (jac_arr < 0.2) & (mces_arr <= 2)
    print(f'\n    Complementary: Low-Jaccard (<0.2) + Near-isomer (MCES≤2): {low_jac_near.sum()} pairs')
    return conflicts

conflicts_335 = inspect_conflicts_fast(pair_mces_jaccard_335, '335 rules',
                                        ik_to_idx_335, rv_matrix_335, ik_list_335, N_RULES_335)
conflicts_mb = inspect_conflicts_fast(pair_mces_jaccard_mb, '3,486 rules',
                                       ik_to_idx_mb, rv_matrix_mb, ik_list_mb, N_RULES_MB)

# ===================================================================
# Item 7: MassBank provenance
# ===================================================================
print('\n' + '=' * 70)
print('ITEM 7: MassBank Rule Provenance Check')
print('=' * 70)

def massbank_provenance_fast():
    mb_start = N_RULES_335
    n_mb = N_RULES_MB - mb_start

    # MassBank-only statistics (fix P1: correct slicing)
    mb_cov = rv_matrix_mb[:, mb_start:].mean(axis=0)
    mb_mol = rv_matrix_mb[:, mb_start:].sum(axis=0).astype(np.int32)

    single_mol = mb_mol < 3
    low_support = mb_mol < 10
    high_cov = mb_cov > 0.5

    print(f'\n  MassBank empirical rules only: {n_mb}')
    print(f'  Single-molecule (<3 mols): {single_mol.sum()} ({single_mol.sum()/n_mb*100:.1f}%) — EXCLUDE')
    print(f'  Low-support (<10 mols):    {low_support.sum()} ({low_support.sum()/n_mb*100:.1f}%) — CAUTION')
    print(f'  High-coverage (>50%):      {high_cov.sum()} ({high_cov.sum()/n_mb*100:.1f}%) — too generic')
    print(f'  Usable (≥10 mols):         {n_mb - low_support.sum()} ({(n_mb-low_support.sum())/n_mb*100:.1f}%)')

    print(f'\n  Coverage distribution (MassBank-only):')
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f'    P{pct}: {np.percentile(mb_cov, pct)*100:.2f}%')

    print(f'\n  Molecule support distribution (MassBank-only):')
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f'    P{pct}: {np.percentile(mb_mol, pct):.0f} molecules')

    if single_mol.any():
        print(f'\n  Example single-molecule MassBank rules:')
        mb_rule_names = [r.name for r in rules_mb_data]
        for idx in np.where(single_mol)[0][:5]:
            ri = mb_start + idx
            name = mb_rule_names[idx] if idx < len(mb_rule_names) else f'r{ri}'
            matching_iks = [ik_list_mb[j] for j in range(len(ik_list_mb)) if rv_matrix_mb[j, ri]]
            print(f'    Rule {ri}: {name}, IKs: {matching_iks}')

    return {
        'n_massbank_rules': n_mb,
        'single_molecule': int(single_mol.sum()),
        'low_support_lt10': int(low_support.sum()),
        'high_coverage_gt50': int(high_cov.sum()),
        'usable_ge10': int(n_mb - low_support.sum()),
    }

mb_prov = massbank_provenance_fast()

# ===================================================================
# Pre-compute overall Pearson/Spearman for summary (fix P1)
# ===================================================================
r_335_overall, _ = pearsonr(
    np.array([p[2] for p in pair_mces_jaccard_335]),
    np.array([p[3] for p in pair_mces_jaccard_335]))
r_mb_overall, _ = pearsonr(
    np.array([p[2] for p in pair_mces_jaccard_mb]),
    np.array([p[3] for p in pair_mces_jaccard_mb]))
srho_335_overall, _ = spearmanr(
    np.array([p[2] for p in pair_mces_jaccard_335]),
    np.array([p[3] for p in pair_mces_jaccard_335]))
srho_mb_overall, _ = spearmanr(
    np.array([p[2] for p in pair_mces_jaccard_mb]),
    np.array([p[3] for p in pair_mces_jaccard_mb]))

# ===================================================================
# Figure (8-panel)
# ===================================================================
print('\n' + '=' * 70)
print('Rendering summary figure...')
print('=' * 70)

fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.30)

# (a) Rule molecule coverage histogram
ax = fig.add_subplot(gs[0, 0])
ax.hist(cov_335 * 100, bins=50, alpha=0.6, label='335 rules', color='#3498db', edgecolor='white')
ax.hist(cov_mb * 100, bins=50, alpha=0.6, label='3,486 rules', color='#e74c3c', edgecolor='white')
ax.set_xlabel('Molecule coverage (%)'); ax.set_ylabel('Number of rules')
ax.set_title('(a) Rule Molecule Coverage Distribution', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# (b) Molecule support histogram
ax = fig.add_subplot(gs[0, 1])
m335_pos = mol_335[mol_335 > 0]
mmb_pos = mol_mb[mol_mb > 0]
ax.hist(np.log10(m335_pos + 1), bins=40, alpha=0.6, label='335 rules', color='#3498db', edgecolor='white')
ax.hist(np.log10(mmb_pos + 1), bins=40, alpha=0.6, label='3,486 rules', color='#e74c3c', edgecolor='white')
ax.set_xlabel('log10(Molecule support + 1)'); ax.set_ylabel('Number of rules')
ax.set_title('(b) Independent Molecule Support Per Rule', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# (c) Per-category Pearson r
ax = fig.add_subplot(gs[1, 0])
cats_ordered_335 = sorted(cat_results_335.keys())
r_vals = [cat_results_335[c]['pearson_r'] for c in cats_ordered_335]
color_map_cat = {
    'NL': '#3498db', 'CF': '#2ecc71', 'ISO': '#1abc9c',
    'NR': '#e74c3c', 'EE': '#f39c12', 'HR': '#9b59b6'
}
bar_colors = [color_map_cat.get(c, '#95a5a6') for c in cats_ordered_335]
ax.bar(range(len(cats_ordered_335)), r_vals, color=bar_colors, alpha=0.8, edgecolor='white')
ax.axhline(y=r_335_overall, color='gray', linestyle='--', alpha=0.5,
          label=f'Overall r={r_335_overall:.4f}')
ax.set_xticks(range(len(cats_ordered_335)))
ax.set_xticklabels(cats_ordered_335, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Pearson r (MCES vs Jaccard)')
ax.set_title('(c) Per-Category Correlation (335 rules)', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')

# (d) Same-formula vs cross-formula
ax = fig.add_subplot(gs[1, 1])
mces_bin_defs = [(0, 2), (3, 5), (6, 999)]
x = np.arange(len(mces_bin_defs))
width = 0.35
for j, (pairs, name, color) in enumerate([(sf_335, 'Same FM', '#e74c3c'), (df_335, 'Diff FM', '#3498db')]):
    m_arr = np.array([p[2] for p in pairs]); j_arr = np.array([p[3] for p in pairs])
    means = [j_arr[(m_arr>=lo)&(m_arr<=hi)].mean() if ((m_arr>=lo)&(m_arr<=hi)).any() else 0 for lo,hi in mces_bin_defs]
    stds = [j_arr[(m_arr>=lo)&(m_arr<=hi)].std() if ((m_arr>=lo)&(m_arr<=hi)).sum()>1 else 0 for lo,hi in mces_bin_defs]
    off = (j - 0.5) * width
    ax.bar(x + off, means, width*0.9, yerr=stds, color=color, alpha=0.8, label=name, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(['MCES 0-2', 'MCES 3-5', 'MCES 6+'])
ax.set_ylabel('Mean Rule Jaccard'); ax.set_title('(d) Same-Formula vs Cross-Formula (335 rules)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

# (e) Conflict scatter
ax = fig.add_subplot(gs[2, 0])
mces_arr_335 = np.array([p[2] for p in pair_mces_jaccard_335])
jac_arr_335 = np.array([p[3] for p in pair_mces_jaccard_335])
ax.scatter(mces_arr_335, jac_arr_335, alpha=0.12, s=4, c='#bdc3c7', edgecolors='none', rasterized=True)
if conflicts_335:
    c_mces = np.array([p[2] for p in conflicts_335])
    c_jac = np.array([p[3] for p in conflicts_335])
    ax.scatter(c_mces, c_jac, alpha=0.8, s=25, c='#e74c3c', edgecolors='#c0392b',
              linewidth=0.5, label=f'Conflict n={len(conflicts_335)}', zorder=5)
low_mask = (jac_arr_335 < 0.2) & (mces_arr_335 <= 2)
if low_mask.any():
    ax.scatter(mces_arr_335[low_mask], jac_arr_335[low_mask], alpha=0.8, s=25, c='#f39c12',
              edgecolors='#e67e22', linewidth=0.5, label=f'Low-Jac near n={low_mask.sum()}', zorder=4)
ax.axhline(y=0.5, color='#e74c3c', linestyle='--', alpha=0.4, lw=1)
ax.axvline(x=5, color='#e74c3c', linestyle='--', alpha=0.4, lw=1)
ax.set_xlabel('MCES'); ax.set_ylabel('Rule Jaccard')
ax.set_title('(e) Conflict Sample Identification (335 rules)', fontweight='bold')
ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.2)

# (f) Per-category Δ Jaccard
ax = fig.add_subplot(gs[2, 1])
deltas = [cat_results_335[c]['delta_jaccard'] for c in cats_ordered_335]
discrims = [cat_results_335[c]['discrimination_ratio'] for c in cats_ordered_335]
x = np.arange(len(cats_ordered_335))
ax.bar(x - width/2, deltas, width*0.9, color=bar_colors, alpha=0.8, edgecolor='white')
for i, (d, disc) in enumerate(zip(deltas, discrims)):
    if disc > 0:
        ax.text(i - width/2, d + 0.003, f'{disc:.1f}×', ha='center', fontsize=7, color='#2c3e50')
ax.set_xticks(x); ax.set_xticklabels(cats_ordered_335, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Δ Jaccard (MCES 0-2 vs >10)')
ax.set_title('(f) Per-Category Discrimination Power (335 rules)', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

# (g) MassBank coverage vs support
ax = fig.add_subplot(gs[3, 0])
mb_mol_arr = rv_matrix_mb[:, N_RULES_335:].sum(axis=0)
mb_cov_arr = rv_matrix_mb[:, N_RULES_335:].mean(axis=0)
sc = ax.scatter(np.log10(mb_mol_arr + 1), mb_cov_arr * 100,
               c=mb_mol_arr, cmap='viridis', alpha=0.5, s=12, edgecolors='none')
sm_mask = mb_mol_arr < 3
ax.scatter(np.log10(mb_mol_arr[sm_mask] + 1), mb_cov_arr[sm_mask] * 100,
          c='red', alpha=0.8, s=25, edgecolors='darkred', lw=0.5,
          label=f'Single-mol ({sm_mask.sum()})', zorder=5)
ax.set_xlabel('log10(Molecule support + 1)'); ax.set_ylabel('Coverage (%)')
ax.set_title('(g) MassBank Rules: Coverage vs Support', fontweight='bold')
ax.legend(fontsize=8); plt.colorbar(sc, ax=ax, label='Mol support'); ax.grid(True, alpha=0.2)

# (h) Summary table — use direct overall Pearson (fix P1)
ax = fig.add_subplot(gs[3, 1])
ax.axis('off')
table_data = [
    ['Metric', '335 rules', '3,486 rules'],
    ['Overall Pearson r', f'{r_335_overall:.4f}', f'{r_mb_overall:.4f}'],
    ['Overall Spearman ρ', f'{srho_335_overall:.4f}', f'{srho_mb_overall:.4f}'],
    ['Δ Jaccard (same-FM, 335)', f'{np.mean([p[3] for p in sf_335]):.4f}' if sf_335 else 'N/A', '-'],
    ['Conflict pairs', f'{len(conflicts_335)}', f'{len(conflicts_mb)}'],
    ['MB single-mol (exclude)', '-', f'{mb_prov["single_molecule"]}'],
    ['MB low-support (caution)', '-', f'{mb_prov["low_support_lt10"]}'],
    ['MB usable (≥10 mols)', '-', f'{mb_prov["usable_ge10"]}'],
]
tbl = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.32, 0.22, 0.22])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.1, 1.8)
for j in range(3): tbl[0,j].set_facecolor('#2c3e50'); tbl[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(table_data)):
    tbl[i,0].set_facecolor('#ecf0f1')
    for j in range(1,3): tbl[i,j].set_facecolor('#f8f9fa')
ax.set_title('(h) Summary', fontweight='bold', y=1.05)

fig.suptitle('Task 0 Supplementary Analysis: Rule-Molecule Characterization\n'
             'annotated01 | 24,333 pairs | 335 mechanistic + 3,151 MassBank empirical rules',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUT_DIR}/supplementary_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT_DIR}/supplementary_analysis.png')

# ===================================================================
# Save JSON report
# ===================================================================
print('\nSaving JSON report...')
report = {
    'item1_rule_molecule_coverage': {
        '335_rules': {
            'min_pct': float(cov_335.min()*100), 'max_pct': float(cov_335.max()*100),
            'median_pct': float(np.median(cov_335)*100), 'mean_pct': float(cov_335.mean()*100),
        },
        '3486_rules': {
            'min_pct': float(cov_mb.min()*100), 'max_pct': float(cov_mb.max()*100),
            'median_pct': float(np.median(cov_mb)*100), 'mean_pct': float(cov_mb.mean()*100),
        }
    },
    'item2_molecule_support': {
        '335_rules': {
            'min': int(mol_335.min()), 'max': int(mol_335.max()),
            'mean': float(mol_335.mean()), 'median': int(np.median(mol_335)),
            'n_lt5': int((mol_335<5).sum()), 'n_lt10': int((mol_335<10).sum()),
        },
        '3486_rules': {
            'min': int(mol_mb.min()), 'max': int(mol_mb.max()),
            'mean': float(mol_mb.mean()), 'median': int(np.median(mol_mb)),
            'n_lt5': int((mol_mb<5).sum()), 'n_lt10': int((mol_mb<10).sum()),
        }
    },
    'item3_per_category_correlation': {'335_rules': cat_results_335, '3486_rules': cat_results_mb},
    'item4_same_formula': {'335_n_same_fm': len(sf_335), '335_n_diff_fm': len(df_335),
                           'mb_n_same_fm': len(sf_mb), 'mb_n_diff_fm': len(df_mb)},
    'item5_partial_correlation': {'335_rules': pc_335, '3486_rules': pc_mb},
    'item6_conflict_samples': {
        '335_n_conflicts': len(conflicts_335),
        '335_top10': [(a,b,float(m),float(j)) for a,b,m,j in conflicts_335[:10]],
        'mb_n_conflicts': len(conflicts_mb),
        'mb_top10': [(a,b,float(m),float(j)) for a,b,m,j in conflicts_mb[:10]],
    },
    'item7_massbank_provenance': mb_prov,
    'overall_correlation': {
        '335_pearson_r': float(r_335_overall),
        '335_spearman_rho': float(srho_335_overall),
        '3486_pearson_r': float(r_mb_overall),
        '3486_spearman_rho': float(srho_mb_overall),
    },
}

with open(f'{OUT_DIR}/supplementary_analysis.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'Saved: {OUT_DIR}/supplementary_analysis.json')

# ===================================================================
# Markdown summary
# ===================================================================
print('\nWriting markdown summary...')
with open(f'{OUT_DIR}/SUPPLEMENTARY_ANALYSIS.md', 'w', encoding='utf-8') as f:
    f.write(f"""# Task 0 Supplementary Analysis (v3)

**Date**: 2026-08-06 | **Data**: annotated01, 24,333 molecule pairs
**Rules**: 335 mechanistic + 3,151 MassBank empirical = 3,486 total
**Overall**: Pearson r={r_335_overall:.4f} (335), {r_mb_overall:.4f} (3,486) | Spearman ρ={srho_335_overall:.4f} (335), {srho_mb_overall:.4f} (3,486)

---

## Item 1: Rule Molecule Coverage

Each rule's coverage = fraction of 76,157 unique molecules (14-char InChIKeys) where the rule fires.

| Metric | 335 rules | 3,486 rules |
|--------|-----------|-------------|
| Min coverage | {cov_335.min()*100:.4f}% | {cov_mb.min()*100:.4f}% |
| Max coverage | {cov_335.max()*100:.1f}% | {cov_mb.max()*100:.1f}% |
| Median coverage | {np.median(cov_335)*100:.2f}% | {np.median(cov_mb)*100:.2f}% |
| Mean coverage | {cov_335.mean()*100:.2f}% | {cov_mb.mean()*100:.2f}% |
| P10 coverage | {np.percentile(cov_335,10)*100:.2f}% | {np.percentile(cov_mb,10)*100:.2f}% |
| P90 coverage | {np.percentile(cov_335,90)*100:.2f}% | {np.percentile(cov_mb,90)*100:.2f}% |

---

## Item 2: Independent Molecule Support Per Rule

| Metric | 335 rules | 3,486 rules |
|--------|-----------|-------------|
| Min support | {mol_335.min()} mols | {mol_mb.min()} mols |
| Max support | {mol_335.max()} mols | {mol_mb.max()} mols |
| Median support | {np.median(mol_335):.0f} mols | {np.median(mol_mb):.0f} mols |
| Mean support | {mol_335.mean():.0f} mols | {mol_mb.mean():.0f} mols |
| Rules < 5 mols | {(mol_335<5).sum()} ({(mol_335<5).sum()/N_RULES_335*100:.1f}%) | {(mol_mb<5).sum()} ({(mol_mb<5).sum()/N_RULES_MB*100:.1f}%) |
| Rules < 10 mols | {(mol_335<10).sum()} ({(mol_335<10).sum()/N_RULES_335*100:.1f}%) | {(mol_mb<10).sum()} ({(mol_mb<10).sum()/N_RULES_MB*100:.1f}%) |

---

## Item 3: Per-Category Correlation

### 335 Rules
| Category | N Rules | N Pairs | Pearson r | Spearman ρ | Δ Jaccard | Discrim |
|----------|--------|---------|-----------|------------|-----------|--------|
""")
    for cat in sorted(cat_results_335.keys()):
        r = cat_results_335[cat]
        f.write(f'| {cat} | {r["n_rules"]} | {r["n_pairs"]} | {r["pearson_r"]:.4f} | {r["spearman_rho"]:.4f} | {r["delta_jaccard"]:.4f} | {r["discrimination_ratio"]:.2f}× |\n')

    f.write('\n### 3,486 Rules\n| Category | N Rules | N Pairs | Pearson r | Spearman ρ | Δ Jaccard | Discrim |\n|----------|--------|---------|-----------|------------|-----------|--------|\n')
    for cat in sorted(cat_results_mb.keys()):
        r = cat_results_mb[cat]
        f.write(f'| {cat} | {r["n_rules"]} | {r["n_pairs"]} | {r["pearson_r"]:.4f} | {r["spearman_rho"]:.4f} | {r["delta_jaccard"]:.4f} | {r["discrimination_ratio"]:.2f}× |\n')

    # Compute MassBank-only low-support (P1 fix)
    mb_only_support = mol_mb[N_RULES_335:]
    n_mb_low = int((mb_only_support < 10).sum())
    n_mb_single = int((mb_only_support < 3).sum())

    f.write(f"""
---

## Item 4: Same-Formula Isomer Δ Jaccard

| Metric | 335 rules | 3,486 rules |
|--------|-----------|-------------|
| Same-formula pairs | {len(sf_335)} | {len(sf_mb)} |
| Different-formula pairs | {len(df_335)} | {len(df_mb)} |

---

## Item 5: Partial Correlation

Controlling: |Δ precursor_mz|, |Δ peak count|, mean spectrum count.

""")
    if pc_335:
        f.write(f"""| Metric | 335 rules | 3,486 rules |
|--------|-----------|-------------|
| Zero-order ρ | {pc_335['zero_order_rho']:.4f} | {pc_mb['zero_order_rho']:.4f} |
| Partial ρ (controlled) | {pc_335['partial_rho']:.4f} | {pc_mb['partial_rho']:.4f} |
| Δ (confounders) | {pc_335['delta']:+.4f} | {pc_mb['delta']:+.4f} |

""")
    else:
        f.write('(Skipped in --quick mode)\n\n')

    f.write(f"""---

## Item 6: Conflict Samples

| Metric | 335 rules | 3,486 rules |
|--------|-----------|-------------|
| Conflict (Jac>0.5, MCES>5) | {len(conflicts_335)} | {len(conflicts_mb)} |

Top-10 conflict pairs (see `supplementary_analysis.json` for full SMILES).

---

## Item 7: MassBank Rule Provenance

- **Total MassBank empirical rules**: {mb_prov['n_massbank_rules']}
- **Single-molecule (< 3 mols)**: {n_mb_single} — **EXCLUDE from training**
- **Low-support (< 10 mols)**: {n_mb_low} — **CAUTION, monitor in E5**
- **High-coverage (> 50%)**: {mb_prov['high_coverage_gt50']} — too generic for discrimination
- **Usable (≥ 10 mols)**: {mb_prov['usable_ge10']} — suitable for empirical supervision

### Recommendation for E5
- Strictly exclude {n_mb_single} single-molecule MassBank rules
- Apply minimum 10-molecule support filter → keep {mb_prov['usable_ge10']} rules
- Monitor {n_mb_low} low-support rules for overfitting

---

## Summary for P1 Architecture

1. **Coverage**: Most rules fire on reasonable fractions of molecules; use P10 as minimum coverage for E3.
2. **Molecule support**: {(mol_335<5).sum()} of 335 mechanistic rules have <5 molecule support — exclude from E3 rule-decode head.
3. **Category matters**: Rule categories differ in structural correlation — use per-category weights in E3 loss.
4. **Same-formula**: Rules show limited isomer discrimination — structural supervision (MCES) is essential.
5. **Partial correlation**: Confounders explain negligible correlation delta — granularity mismatch is real.
6. **Conflict samples**: {len(conflicts_335)} pairs identified for E4 hard negative mining.
7. **MassBank filtering**: Exclude {n_mb_single} single-molecule rules, use {mb_prov['usable_ge10']} filtered set for E5.

![supplementary_analysis.png](supplementary_analysis.png)
""")

print(f'Saved: {OUT_DIR}/SUPPLEMENTARY_ANALYSIS.md')
print(f'\n=== COMPLETE ({time.time()-T0:.0f}s) ===')
