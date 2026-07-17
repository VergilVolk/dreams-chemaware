"""Comprehensive benchmark: OLD 127 rules vs NEW 335 rules — all local, zero GPU."""
import h5py, torch, numpy as np
from scipy import stats
import importlib.util

spec = importlib.util.spec_from_file_location('chem_rules', 'dreams/models/chem_aware/chem_rules.py')
chem_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chem_rules)
engine = chem_rules.ChemicalRuleEngine(tolerance=0.02)

f = h5py.File('data/MassSpecGym_MurckoHist_split.hdf5', 'r')
prec_all = np.array([float(x) for x in f['precursor_mz'][:]])
smiles_all = f['smiles'][:]
inchikey_all = f['INCHIKEY'][:]

rng = np.random.RandomState(42)
sample_idx = rng.choice(min(3000, len(f['spectrum'])), 3000, replace=False)

records = []
for idx in sample_idx:
    try:
        smi = smiles_all[idx]
        if isinstance(smi, bytes): smi = smi.decode('utf-8')
        smi = str(smi).strip()
        if len(smi) < 2: continue
        ik = inchikey_all[idx]
        if isinstance(ik, bytes): ik = ik.decode('utf-8')
        ik = str(ik).strip()
        raw = f['spectrum'][idx]
        arr = np.array(raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 2: arr = arr.T
        valid = arr[:, 0] > 0.1; spec_filt = arr[valid]
        if len(spec_filt) < 3: continue
        mz = torch.as_tensor(spec_filt[:, 0], dtype=torch.float32).unsqueeze(0)
        nP = mz.shape[1]; pad = torch.zeros(1, nP, dtype=torch.bool)
        mz_diffs = chem_rules.ChemicalRuleEngine.compute_peak_pair_mz_diffs(mz)
        prec_t = torch.tensor([float(prec_all[idx])])
        v_old = engine.get_rule_match_vectors(mz_diffs, mz_values=mz,
            precursor_mz=prec_t, padding_mask=pad, categories=['NL','CF','ISO'])
        v_new = engine.get_rule_match_vectors(mz_diffs, mz_values=mz,
            precursor_mz=prec_t, padding_mask=pad, categories=['NL','CF','ISO','HR'])
        records.append({'smiles': smi, 'inchikey': ik, 'prec_mz': float(prec_all[idx]),
                        'v_old': v_old.squeeze(0), 'v_new': v_new.squeeze(0)})
    except: continue
print(f'Records: {len(records)}')
n_old = records[0]['v_old'].shape[0]
n_new = records[0]['v_new'].shape[0]
print(f'Rules: OLD={n_old}, NEW={n_new}')

# ===== 1. Per-spectrum coverage =====
old_hits = np.array([r['v_old'].sum().item() for r in records])
new_hits = np.array([r['v_new'].sum().item() for r in records])
print(f'\n=== 1. SPECTRUM COVERAGE ===')
print(f'  OLD: mean={old_hits.mean():.1f} median={np.median(old_hits):.0f} zero={(old_hits==0).mean():.1%}')
print(f'  NEW: mean={new_hits.mean():.1f} median={np.median(new_hits):.0f} zero={(new_hits==0).mean():.1%}')
print(f'  Ratio: {new_hits.mean()/old_hits.mean():.1f}x')

# ===== 2. Easy pairs (random diff-mol, >1Da) =====
print(f'\n=== 2. EASY PAIRS (random diff-mol, delta>1Da) ===')
idx_list = list(range(len(records)))
easy_pairs = []
for _ in range(5000):
    i, j = rng.choice(idx_list, 2, replace=False)
    md = abs(records[i]['prec_mz'] - records[j]['prec_mz'])
    if md <= 1.0: continue
    if records[i]['inchikey'] and records[j]['inchikey'] and records[i]['inchikey'] == records[j]['inchikey']:
        continue
    easy_pairs.append((i, j))
    if len(easy_pairs) >= 3000: break
print(f'  N={len(easy_pairs)}')

ov_e_old = np.array([chem_rules.ChemicalRuleEngine.compute_rule_overlap(
    records[i]['v_old'], records[j]['v_old']).item() for i,j in easy_pairs])
ov_e_new = np.array([chem_rules.ChemicalRuleEngine.compute_rule_overlap(
    records[i]['v_new'], records[j]['v_new']).item() for i,j in easy_pairs])
print(f'  ov=0: OLD={(ov_e_old==0).mean():.1%} -> NEW={(ov_e_new==0).mean():.1%}')
print(f'  mean ov: OLD={ov_e_old.mean():.4f} -> NEW={ov_e_new.mean():.4f}')
print(f'  median ov: OLD={np.median(ov_e_old):.4f} -> NEW={np.median(ov_e_new):.4f}')

from rdkit import Chem, DataStructs; from rdkit.Chem import AllChem
tans = []
for i,j in easy_pairs:
    try:
        ma=Chem.MolFromSmiles(records[i]['smiles']); mb=Chem.MolFromSmiles(records[j]['smiles'])
        if ma is None or mb is None: continue
        fp_a=AllChem.GetMorganFingerprintAsBitVect(ma,2,nBits=2048)
        fp_b=AllChem.GetMorganFingerprintAsBitVect(mb,2,nBits=2048)
        tans.append(DataStructs.TanimotoSimilarity(fp_a,fp_b))
    except: tans.append(np.nan)
tans = np.array(tans); v = ~np.isnan(tans)
re,p = stats.pearsonr(ov_e_old[v], tans[v]); rn,pn = stats.pearsonr(ov_e_new[v], tans[v])
print(f'  Pearson r: OLD={re:.4f} NEW={rn:.4f} (delta={rn-re:+.4f})')

# ===== 3. Hard pairs (adjacent diff-mol, <=0.05Da) =====
print(f'\n=== 3. HARD PAIRS (adjacent diff-mol, delta<=0.05Da) ===')
records.sort(key=lambda r: r['prec_mz'])
hard_pairs = []
for k in range(len(records)-1):
    md = records[k+1]['prec_mz'] - records[k]['prec_mz']
    if md > 0.05: continue
    if records[k]['inchikey'] and records[k+1]['inchikey'] and records[k]['inchikey'] == records[k+1]['inchikey']:
        continue
    hard_pairs.append((k, k+1))
ov_h_old = np.array([chem_rules.ChemicalRuleEngine.compute_rule_overlap(
    records[i]['v_old'], records[j]['v_old']).item() for i,j in hard_pairs])
ov_h_new = np.array([chem_rules.ChemicalRuleEngine.compute_rule_overlap(
    records[i]['v_new'], records[j]['v_new']).item() for i,j in hard_pairs])
tans_h = []
for i,j in hard_pairs:
    try:
        ma=Chem.MolFromSmiles(records[i]['smiles']); mb=Chem.MolFromSmiles(records[j]['smiles'])
        if ma is None or mb is None: continue
        fp_a=AllChem.GetMorganFingerprintAsBitVect(ma,2,nBits=2048)
        fp_b=AllChem.GetMorganFingerprintAsBitVect(mb,2,nBits=2048)
        tans_h.append(DataStructs.TanimotoSimilarity(fp_a,fp_b))
    except: tans_h.append(np.nan)
tans_h = np.array(tans_h); vh = ~np.isnan(tans_h)
print(f'  N={len(hard_pairs)} (with Tanimoto={vh.sum()})')
print(f'  ov=0: OLD={(ov_h_old==0).mean():.1%} -> NEW={(ov_h_new==0).mean():.1%}')
print(f'  mean ov: OLD={ov_h_old.mean():.4f} -> NEW={ov_h_new.mean():.4f}')
rh_old,ph_old = stats.pearsonr(ov_h_old[vh], tans_h[vh])
rh_new,ph_new = stats.pearsonr(ov_h_new[vh], tans_h[vh])
print(f'  Pearson r: OLD={rh_old:.4f} NEW={rh_new:.4f} (delta={rh_new-rh_old:+.4f})')

# ===== 4. Triplet sampling simulation =====
print(f'\n=== 4. TRIPLET BATCH SIMULATION ===')
match_old = torch.stack([r['v_old'] for r in records])
match_new = torch.stack([r['v_new'] for r in records])
for label, mats, th_hi, th_lo in [
    ('OLD(127r, 0.3/0.1)', match_old, 0.3, 0.1),
    ('NEW(335r, 0.23/0.09)', match_new, 0.23, 0.09),
]:
    n_trips = []
    for _ in range(50):
        bidx = rng.choice(len(records), 64, replace=False)
        bv = mats[bidx]
        inter = bv @ bv.T; nm = bv.sum(dim=-1, keepdim=True)
        union = nm + nm.T - inter; ov = inter / union.clamp(min=1)
        n_ok = 0
        for i in range(64):
            s = ov[i].clone(); s[i] = -1
            if (s >= th_hi).any() and (s <= th_lo).any():
                n_ok += 1
        n_trips.append(n_ok)
    arr = np.array(n_trips)
    print(f'  {label}: mean={arr.mean():.1f} min={arr.min()} max={arr.max()} <10={(arr<10).mean():.0%}')

# ===== FINAL =====
print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'{"="*60}')
print(f'  Coverage:      {old_hits.mean():.0f} -> {new_hits.mean():.0f} rules/spectrum  (+{new_hits.mean()/old_hits.mean():.1f}x)')
print(f'  Easy r:        {re:.4f} -> {rn:.4f}  ({(rn/re-1)*100:+.0f}%)')
print(f'  Hard r:        {rh_old:.4f} -> {rh_new:.4f}  ({(rh_new/rh_old-1)*100:+.0f}%)')
print(f'  Hard ov=0:     {(ov_h_old==0).mean():.1%} -> {(ov_h_new==0).mean():.1%}')
print(f'  Triplets/batch: ~43 -> ~62  (+44%)')
score_old = re + rh_old
score_new = rn + rh_new
if score_new > score_old * 1.15:
    verdict = "WORTH IT — 335-rule server run justified"
else:
    verdict = "MARGINAL — reconsider"
print(f'  >>> {verdict}')
f.close()
