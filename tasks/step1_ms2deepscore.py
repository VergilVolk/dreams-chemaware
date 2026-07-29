"""
Step 1: MS2DeepScore 批量计算
加载转换后的 PyTorch 模型 + Keras HDF5 中的 spectrum_binner 元数据
输出: tasks/ms2ds_scores_{task}.json

用法 (任意有 torch + numpy 的环境):
  python tasks/step1_ms2deepscore.py --task T3
  python tasks/step1_ms2deepscore.py --task T1
  python tasks/step1_ms2deepscore.py --task T2
  python tasks/step1_ms2deepscore.py --task T0
"""
import json, os, sys, argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

p = argparse.ArgumentParser()
p.add_argument('--task', required=True, choices=['T0', 'T1', 'T2', 'T3'])
p.add_argument('--n_pairs', type=int, default=0, help='最大 pair 数 (0=全部)')
args = p.parse_args()

TASK_PATHS = {
    'T0': 'tasks/T0_consistency/test_cases/pairs.json',
    'T1': 'tasks/T1_near_isomers/test_cases/pairs.json',
    'T2': 'tasks/T2_analogs/test_cases/pairs.json',
    'T3': 'tasks/T3_unrelated/test_cases/pairs.json',
}

# ---- 1. Load spectrum binner from HDF5 ----
print('[1] Loading spectrum binner from HDF5...')
import h5py
with h5py.File('data/models/MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5', 'r') as f:
    binner_str = f.attrs['spectrum_binner']
binner = json.loads(binner_str)
mz_min = binner['mz_min']          # 10.0
mz_max = binner['mz_max']          # 1000.0
d_bins = binner['d_bins']          # 0.099
peak_scaling = binner['peak_scaling']  # 0.5
peak_to_position = {int(k): int(v) for k, v in binner['peak_to_position'].items()}
input_dim = len(peak_to_position)   # 9948

print(f'  mz_range: [{mz_min}, {mz_max}], bin_width: {d_bins}, peak_scaling: {peak_scaling}')
print(f'  Input dimension: {input_dim} (after removing always-empty bins)')

# ---- 2. Load pairs ----
print(f'[2] Loading {args.task} pairs...')
with open(TASK_PATHS[args.task]) as f:
    data = json.load(f)

if args.task == 'T1':
    pairs = data.get('positive', []) + data.get('negative_hard', []) + data.get('negative_easy', [])
else:
    pairs = data.get('positive', []) + data.get('negative', []) + data.get('negative_hard', []) + data.get('negative_easy', [])

if args.n_pairs > 0 and len(pairs) > args.n_pairs:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(pairs), args.n_pairs, replace=False)
    pairs = [pairs[i] for i in idx]
print(f'  {len(pairs)} pairs')

# ---- 3. Load spectra from annotated01 ----
print('[3] Loading spectra from annotated01.mgf...')
needed_iks = set()
for p in pairs:
    ik_a = (p.get('ik', '') or p.get('ik_a', ''))[:14]
    ik_b = (p.get('ik_b', '') or p.get('ik', '') or p.get('ik_a', ''))[:14]
    needed_iks.add(ik_a)
    needed_iks.add(ik_b)
print(f'  {len(needed_iks)} unique IKs needed')

ik_to_peaks = {}
cur_ik = None; cur_peaks = []
with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if len(ik_to_peaks) >= len(needed_iks):
            break
        line = line.strip()
        if not line:
            if cur_ik and cur_ik in needed_iks and cur_ik not in ik_to_peaks and len(cur_peaks) >= 3:
                ik_to_peaks[cur_ik] = cur_peaks[:]
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

print(f'  {len(ik_to_peaks)} spectra loaded')

# ---- 4. Convert spectra to vectors (replicating MS2DeepScore binning) ----
print('[4] Binning spectra...')

def spectrum_to_vector(peaks):
    """Convert peak list to 9948-dim vector using the exact trained binner"""
    vector = np.zeros(input_dim, dtype=np.float32)
    for mz, intensity in peaks:
        if mz_min <= mz < mz_max:
            bin_idx = int((mz - mz_min) / d_bins)
            if bin_idx in peak_to_position:
                pos = peak_to_position[bin_idx]
                scaled = intensity ** peak_scaling
                vector[pos] = max(vector[pos], scaled)
    return vector

ik_to_vec = {}
for ik, peaks in ik_to_peaks.items():
    ik_to_vec[ik] = spectrum_to_vector(peaks)

print(f'  {len(ik_to_vec)} spectra binned')

# ---- 5. Load converted PyTorch model ----
print('[5] Loading MS2DeepScore PyTorch model...')
import torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    'convert_ms2ds_model',
    os.path.join(os.path.dirname(__file__), 'convert_ms2ds_model.py'))
cvt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvt)
MS2DSSiamese = cvt.MS2DSSiamese

ckpt = torch.load('data/models/MS2DeepScore_allGNPSpositive_10k_500_500_200.pt',
                  map_location='cpu', weights_only=True)
model = MS2DSSiamese(ckpt['input_dim'], ckpt['hidden'], ckpt['embedding'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Model loaded: {ckpt["input_dim"]} → {ckpt["hidden"]} → {ckpt["embedding"]}')

# ---- 6. Compute scores in batches ----
print('[6] Computing MS2DeepScore...')
scores = {}
batch_size = 256
n_computed = 0

# Prepare pairs that have spectra for both IKs
valid_pairs = []
for p in pairs:
    ik_a = (p.get('ik', '') or p.get('ik_a', ''))[:14]
    ik_b = (p.get('ik_b', '') or p.get('ik', '') or p.get('ik_a', ''))[:14]
    if ik_a in ik_to_vec and ik_b in ik_to_vec:
        valid_pairs.append((ik_a, ik_b))

print(f'  {len(valid_pairs)} pairs with both spectra available')

for batch_start in range(0, len(valid_pairs), batch_size):
    batch_end = min(batch_start + batch_size, len(valid_pairs))
    batch_pairs = valid_pairs[batch_start:batch_end]

    vecs_a = np.array([ik_to_vec[ik_a] for ik_a, _ in batch_pairs], dtype=np.float32)
    vecs_b = np.array([ik_to_vec[ik_b] for _, ik_b in batch_pairs], dtype=np.float32)

    with torch.no_grad():
        t_a = torch.from_numpy(vecs_a)
        t_b = torch.from_numpy(vecs_b)
        sims = model(t_a, t_b).numpy()

    for (ik_a, ik_b), sim in zip(batch_pairs, sims):
        scores[f'{ik_a}|{ik_b}'] = float(sim)
        n_computed += 1

    if (batch_start // batch_size) % 10 == 0:
        print(f'  {n_computed}/{len(valid_pairs)}')

# ---- 7. Save ----
out_path = f'tasks/ms2ds_scores_{args.task}.json'
with open(out_path, 'w') as f:
    json.dump(scores, f, indent=2)
print(f'\nSaved {n_computed} scores to {out_path}')

# Quick stats
if scores:
    vals = list(scores.values())
    print(f'  Score range: [{min(vals):.4f}, {max(vals):.4f}]')
    print(f'  Mean: {np.mean(vals):.4f}  Std: {np.std(vals):.4f}')
print(f'=== DONE ===')
