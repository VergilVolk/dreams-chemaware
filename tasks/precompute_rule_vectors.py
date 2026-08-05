"""
预计算 ChemicalRuleEngine 335 维规则命中向量缓存。

输出: tasks/_cache/rule_vectors/
  ik_to_rvec.npz   — {ik14: 335-dim binary array}  每 IK 一套
  ik_best_peaks.json — {ik14: [mz, intensity, ...]}  最佳谱图

耗时: ~30min (87K IKs × 335 rules × 60 peaks, batch=8)
之后任何 pair 的 Jaccard 计算变成 instant bitwise AND/OR。

用法: python tasks/precompute_rule_vectors.py
"""
import json, os, sys, time, gc
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '.')
from tasks.build_utils import load_indices
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine

rng = np.random.RandomState(42)
CACHE_DIR = 'tasks/_cache/rule_vectors'
os.makedirs(CACHE_DIR, exist_ok=True)
N_PEAKS = 60

# ===================================================================
# 1. Scan MGF once → best spectrum per IK
# ===================================================================
print('[1] Scanning annotated01 for best spectra per IK...')
idx = load_indices()
ik_to_smi = idx['ik_to_smi']

ik_best_peaks = {}
cur_ik = None; cur_peaks = []; cur_total_i = 0
with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f, unit='lines', unit_scale=True):
        line = line.strip()
        if not line:
            if cur_ik and len(cur_peaks) >= 3:
                if cur_ik not in ik_best_peaks or cur_total_i > ik_best_peaks[cur_ik][1]:
                    ik_best_peaks[cur_ik] = (cur_peaks[:], cur_total_i)
            cur_ik = None; cur_peaks = []; cur_total_i = 0; continue
        if line.startswith('INCHIKEY='):
            cur_ik = line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
            p2 = line.split()
            if len(p2) >= 2:
                try:
                    mz, intensity = float(p2[0]), float(p2[1])
                    if mz > 0 and intensity > 0:
                        cur_peaks.append((mz, intensity))
                        cur_total_i += intensity
                except: pass

best_iks = sorted(ik_best_peaks.keys())
print(f'  {len(best_iks)} IKs with best spectra')

# ===================================================================
# 2. Initialize rule engine
# ===================================================================
print('[2] Initializing ChemicalRuleEngine (335 rules, no MassBank)...')
engine = ChemicalRuleEngine(tolerance=0.02, use_massbank=False)
n_rules = len(engine.rules)
print(f'  {n_rules} rules')

# ===================================================================
# 3. Batch-compute rule vectors
# ===================================================================
print(f'[3] Computing rule vectors for {len(best_iks)} IKs (batch=8)...')
ik_to_rvec = {}
BATCH = 8

# Preprocess helper
def prep(peaks):
    arr = np.array(peaks, dtype=np.float32)
    if len(arr) < 3: return None
    arr = arr[arr[:, 0].argsort()]
    if len(arr) > N_PEAKS:
        idx = np.argpartition(arr[:, 1], -N_PEAKS)[-N_PEAKS:]
        arr = arr[idx]; arr = arr[arr[:, 0].argsort()]
    max_i = arr[:, 1].max()
    if max_i > 0: arr[:, 1] /= max_i
    padded = np.zeros((N_PEAKS, 2), dtype=np.float32)
    n = min(len(arr), N_PEAKS); padded[:n] = arr[:n]
    return padded

ts = time.time()
n_done = 0
n_skip = 0

for b_start in tqdm(range(0, len(best_iks), BATCH)):
    batch_iks = best_iks[b_start:b_start + BATCH]
    valid_iks = []; specs = []
    for ik in batch_iks:
        entry = ik_best_peaks.get(ik)
        if entry is None: continue
        s = prep(entry[0])
        if s is not None:
            specs.append(s)
            valid_iks.append(ik)

    if not specs:
        n_skip += len(batch_iks)
        continue

    # Stack into batch
    mz_batch = torch.as_tensor(np.stack([s[:, 0] for s in specs]), dtype=torch.float32)
    pad_batch = (mz_batch == 0)
    mz_diffs = torch.abs(mz_batch.unsqueeze(-1) - mz_batch.unsqueeze(-2))
    precursor = mz_batch[:, 0].unsqueeze(-1)

    with torch.no_grad():
        vec = engine.get_rule_match_vectors(mz_diffs, mz_values=mz_batch,
                                             precursor_mz=precursor, padding_mask=pad_batch)
    hit = (vec > 0).numpy().astype(np.int8)

    for ik, hv in zip(valid_iks, hit):
        ik_to_rvec[ik] = hv
    n_done += len(valid_iks)

elapsed = time.time() - ts
print(f'  Done: {n_done} IKs ({n_skip} skipped) in {elapsed/60:.1f}min')
print(f'  Rate: {n_done/elapsed:.0f} IKs/sec')

# ===================================================================
# 4. Save
# ===================================================================
print(f'[4] Saving to {CACHE_DIR}/...')
np.savez_compressed(f'{CACHE_DIR}/ik_to_rvec.npz', **{ik: rv for ik, rv in ik_to_rvec.items()})

# Save IK list for loading order
with open(f'{CACHE_DIR}/ik_list.json', 'w') as f:
    json.dump(sorted(ik_to_rvec.keys()), f)

print(f'  Saved: {len(ik_to_rvec)} rule vectors ({os.path.getsize(CACHE_DIR + "/ik_to_rvec.npz")/1e6:.1f}MB)')
print(f'\nUsage in other scripts:')
print(f'  data = np.load("{CACHE_DIR}/ik_to_rvec.npz")')
print(f'  rvec_a = data[ik_a]  # 335-dim binary vector')
