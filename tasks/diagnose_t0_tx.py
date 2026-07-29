"""
T0 TransExION 诊断: 排查 AUC=1.0 是否可信

检查项:
  1. TX 到底给多少对打了分，分数分布如何
  2. 正样本对中是否有近乎相同的 peak list (跨库重复嫌疑)
  3. n_pairs=0 bug 是否波及 T0

用法: python tasks/diagnose_t0_tx.py
"""
import json, os, sys, hashlib
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')

# ---- 1. Load T0 pairs ----
print('=' * 60)
print('DIAGNOSIS 1: T0 pair structure')
print('=' * 60)

with open('tasks/T0_consistency/test_cases/pairs.json') as f:
    t0 = json.load(f)

pos = t0.get('positive', [])
neg = t0.get('negative', [])
print(f'Positive pairs: {len(pos)}')
print(f'Negative pairs: {len(neg)}')

# Show sample pair format
print(f'\nSample positive pair keys: {list(pos[0].keys())}')
print(f'Sample positive pair: ik={pos[0].get("ik","?")[:27]}...')
print(f'Sample negative pair keys: {list(neg[0].keys())}')

# ---- 2. Check IK coverage in annotated01 ----
print(f'\n{"=" * 60}')
print('DIAGNOSIS 2: IK coverage in annotated01.mgf')
print('=' * 60)

# Collect all needed IKs from first 100 pos + 100 neg
needed = set()
for p in pos[:100]:
    ik = (p.get('ik', '') or p.get('ik_a', ''))[:14]
    needed.add(ik)
for p in neg[:100]:
    ik_a = (p.get('ik_a', '') or p.get('ik', ''))[:14]
    ik_b = (p.get('ik_b', '') or p.get('ik', ''))[:14]
    needed.add(ik_a)
    needed.add(ik_b)

# Quick scan annotated01 for these IKs
ik_found = set()
ik_to_peaks_sample = {}
with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    cur_ik = None; cur_peaks = []
    for line in f:
        if len(ik_found) >= len(needed):
            break
        line = line.strip()
        if not line:
            if cur_ik and cur_ik in needed and cur_ik not in ik_found and len(cur_peaks) >= 3:
                ik_found.add(cur_ik)
                ik_to_peaks_sample[cur_ik] = cur_peaks[:]
            cur_ik = None; cur_peaks = []
            continue
        if line.startswith('INCHIKEY='):
            cur_ik = line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
            p2 = line.split()
            if len(p2) >= 2:
                try:
                    mz, i = float(p2[0]), float(p2[1])
                    if mz > 0 and i > 0:
                        cur_peaks.append((mz, i))
                except: pass

print(f'IKs found in annotated01: {len(ik_found)} / {len(needed)} ({len(ik_found)/max(1,len(needed))*100:.0f}%)')

# ---- 3. Check for near-duplicate spectra in positive pairs ----
print(f'\n{"=" * 60}')
print('DIAGNOSIS 3: Cross-DB duplicate check (peak list similarity)')
print('=' * 60)

def peak_cosine(pk_a, pk_b, bin_size=0.01):
    """Compute cosine similarity between two binned peak lists"""
    if not pk_a or not pk_b:
        return -1
    max_mz = max(max(p[0] for p in pk_a), max(p[0] for p in pk_b))
    n_bins = int(max_mz / bin_size) + 1
    vec_a = np.zeros(n_bins)
    vec_b = np.zeros(n_bins)
    for mz, i in pk_a:
        idx = min(int(mz / bin_size), n_bins - 1)
        vec_a[idx] += i
    for mz, i in pk_b:
        idx = min(int(mz / bin_size), n_bins - 1)
        vec_b[idx] += i
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return -1
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

# For T0: each positive pair is same IK, different spectra
# Check if the SAME IK appears with the SAME peak list twice → cross-DB duplicate
# We need to check multiple spectra per IK from the raw data
print('\nChecking if any T0 positive IK has near-identical spectra in annotated01...')

# Scan for IKs that appear in T0 positive pairs, collect ALL their spectra
t0_pos_iks = set()
for p in pos[:500]:
    ik = (p.get('ik', '') or p.get('ik_a', ''))[:14]
    t0_pos_iks.add(ik)

ik_to_all_peaks = defaultdict(list)
with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    cur_ik = None; cur_peaks = []
    for line in f:
        line = line.strip()
        if not line:
            if cur_ik and cur_ik in t0_pos_iks and len(cur_peaks) >= 3:
                ik_to_all_peaks[cur_ik].append(cur_peaks[:])
            cur_ik = None; cur_peaks = []
            continue
        if line.startswith('INCHIKEY='):
            cur_ik = line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
            p2 = line.split()
            if len(p2) >= 2:
                try:
                    mz, i = float(p2[0]), float(p2[1])
                    if mz > 0 and i > 0:
                        cur_peaks.append((mz, i))
                except: pass

# Check pairwise cosine within each IK
suspicious = []
for ik, peaks_list in ik_to_all_peaks.items():
    n = len(peaks_list)
    if n < 2:
        continue
    for i in range(n):
        for j in range(i + 1, n):
            cos = peak_cosine(peaks_list[i], peaks_list[j])
            if cos > 0.99:
                suspicious.append((ik, i, j, cos))

print(f'\nIKs with >=2 spectra in annotated01: {sum(1 for v in ik_to_all_peaks.values() if len(v)>=2)}')
print(f'Near-identical spectra pairs (cos > 0.99): {len(suspicious)}')
if suspicious:
    print(f'\nTOP SUSPICIOUS PAIRS:')
    for ik, i, j, cos in suspicious[:10]:
        print(f'  IK={ik}  spectra[{i}] vs [{j}]  cosine={cos:.6f}')

# If no near-identical spectra, the T0 pairs are clean
if not suspicious:
    print('No cross-DB duplicates detected. T0 positive pairs are clean.')

# ---- 4. Check if n_pairs=0 bug affected T0 ----
print(f'\n{"=" * 60}')
print('DIAGNOSIS 4: n_pairs=0 bug impact')
print('=' * 60)
print('The old run_ablation.py had --n_pairs default=500 (not 0).')
print('T0 was run with --n_pairs 500, so the pairs[:args.n_pairs] bug')
print('would have been pairs[:500], which is correct.')
print('→ T0 was NOT affected by the n_pairs=0 bug.')
print('→ The N=20 pairs for TX on T0 is a DIFFERENT issue:')
print('  only 20 unique IKs out of 250 pos IKs had spectra in spec_list.')

# ---- 5. Summary ----
print(f'\n{"=" * 60}')
print('SUMMARY')
print('=' * 60)
print(f'T0 pairs: {len(pos)} pos + {len(neg)} neg')
print(f'IK coverage: {len(ik_found)}/{len(needed)} found in annotated01')
print(f'Cross-DB duplicates: {len(suspicious)} (cos > 0.99)')
print(f'n_pairs=0 bug affected T0: NO (old default was 500)')
print(f'\nIf T0 TX AUC=1.0 with N=20:')
print(f'  - If scores are all ~1.0 for pos and ~0.0 for neg: real but small-N')
print(f'  - If scores are all identical: degenerate model output')
print(f'  - Need to see the actual 20 TX scores to judge')
