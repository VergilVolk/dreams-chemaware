"""
T0 TX 分数打印 — 输出每对的原始 TransExION 分数
用法: python tasks/print_t0_tx_scores.py
"""
import json, os, sys, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'TransExION')

print('[1] Loading T0 pairs...')
with open('tasks/T0_consistency/test_cases/pairs.json') as f:
    t0 = json.load(f)

# Use first 40 pos + 40 neg (matching the old n_pairs=500 default: 250 pos T0 + 250 neg T3)
# But TX only scored subset → let's see how many
pos = t0['positive'][:100]
neg = t0['negative'][:100]
pairs = pos + neg
labels = [1] * len(pos) + [0] * len(neg)

print(f'  {len(pos)} pos + {len(neg)} neg = {len(pairs)} pairs')

print('[2] Loading spectra from annotated01...')
needed = set()
for p in pairs:
    ik_a = (p.get('ik', '') or p.get('ik_a', ''))[:14]
    ik_b = (p.get('ik_b', '') or p.get('ik', '') or p.get('ik_a', ''))[:14]
    needed.add(ik_a)
    needed.add(ik_b)

ik_to_peaks = {}; ik_to_smi = {}
cur_ik = None; cur_peaks = []; cur_smi = None
with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if len(ik_to_peaks) >= len(needed): break
        line = line.strip()
        if not line:
            if cur_ik and cur_ik in needed and cur_ik not in ik_to_peaks and len(cur_peaks) >= 3:
                ik_to_peaks[cur_ik] = cur_peaks[:]
                if cur_smi: ik_to_smi[cur_ik] = cur_smi
            cur_ik = None; cur_peaks = []; cur_smi = None; continue
        if line.startswith('SMILES='): cur_smi = line[7:].strip()
        elif line.startswith('INCHIKEY='): cur_ik = line[9:].strip()[:14]
        elif line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
            p2 = line.split()
            if len(p2) >= 2:
                try:
                    mz, i = float(p2[0]), float(p2[1])
                    if mz > 0 and i > 0: cur_peaks.append((mz, i))
                except: pass

print(f'  {len(ik_to_peaks)} spectra loaded (of {len(needed)} needed)')

# Export to MGF for TX
mgf_path = 'data/_tx_diag_T0.mgf'
hdf5_path = 'data/_tx_diag_T0.hdf5'
spec_list = []
ik_to_idx = {}
for ik in needed:
    if ik in ik_to_peaks:
        ik_to_idx[ik] = len(spec_list)
        spec_list.append((ik, ik_to_smi.get(ik, ''), ik_to_peaks[ik]))

if not spec_list:
    print('ERROR: No spectra available for any needed IK!')
    sys.exit(1)

print(f'  {len(spec_list)} unique spectra for TX')

with open(mgf_path, 'w', encoding='utf-8') as f:
    for ik, smi, pk in spec_list:
        f.write('BEGIN IONS\n')
        f.write(f'SMILES={smi}\nINCHIKEY={ik}\n')
        f.write('PEPMASS=400\nIONMODE=POSITIVE\nMSLEVEL=2\n')
        for mz, i in pk: f.write(f'{mz:.4f} {i:.4f}\n')
        f.write('END IONS\n\n')

print('[3] Running TransExION inference...')
from spectrum.io import load_mgf_file, convert_raw2refined_spectra
from common.io import save_data_in_hdf5_format
ms_data = load_mgf_file(mgf_path, mol_id_key=None, use_drug=False)
transformed = convert_raw2refined_spectra(ms_data)
save_data_in_hdf5_format(hdf5_path, transformed)

# Build pairs HDF5
pairs_data = []
for p in pairs:
    ik_a = (p.get('ik', '') or p.get('ik_a', ''))[:14]
    ik_b = (p.get('ik_b', '') or p.get('ik', '') or p.get('ik_a', ''))[:14]
    if ik_a in ik_to_idx and ik_b in ik_to_idx:
        pairs_data.append(json.dumps([ik_to_idx[ik_a], ik_to_idx[ik_b], 0]))

import h5py
pairs_h5 = 'data/_tx_diag_pairs_T0.hdf5'
with h5py.File(pairs_h5, 'w') as f:
    f.create_dataset('data', data=np.array(pairs_data, dtype=h5py.string_dtype()))
print(f'  TX pairs: {len(pairs_data)}')

from lrp.data import C_MAX_PEAK_DIFF, C_NUM_DEFFECT_BIN, MSPairSet, MSDataset, mspair_collate_fn
from lrp.model import relMSSimilarityModel
from lrp.functional import evaluate_spectral_similarity_measure
from torch.utils.data.dataloader import DataLoader
import torch

net = relMSSimilarityModel(C_MAX_PEAK_DIFF + 1, C_NUM_DEFFECT_BIN + 2, hidden_dim=128, nclasses=1, dropout=0.1)
net.load_state_dict(torch.load('data/gnps/TransExION_GNPS_MassBank.ms.model', map_location='cpu'))
net = net.double()
db = MSDataset(hdf5_path)
test_set = MSPairSet(db, db, pairs_h5)
loader = DataLoader(test_set, 128, shuffle=False, num_workers=0, collate_fn=mspair_collate_fn)
pred_vals, _ = evaluate_spectral_similarity_measure(loader, net, 'cpu')

# Print ALL pairs with scores
print(f'\n{"=" * 80}')
print(f'INDIVIDUAL TX SCORES ({len(pred_vals)} pairs)')
print(f'{"=" * 80}')
print(f'{"#":>4s}  {"label":>5s}  {"TX_score":>10s}  {"ik_a":>16s}  {"ik_b":>16s}  {"same_ik?":>8s}')
print(f'{"-" * 80}')

pair_info = test_set.get_full_info_all_pairs()
pos_scores = []; neg_scores = []

for idx, (qi, ri, _, _, _, _, _) in enumerate(pair_info):
    ik_a = spec_list[int(qi)][0]
    ik_b = spec_list[int(ri)][0]
    score = float(pred_vals[idx])
    same = ik_a == ik_b
    label = 1 if same else 0
    if label == 1: pos_scores.append(score)
    else: neg_scores.append(score)
    flag = ' <<<' if score > 0.95 else ''
    print(f'{idx:4d}  {label:5d}  {score:10.6f}  {ik_a:16s}  {ik_b:16s}  {str(same):>8s}{flag}')

# Summary
print(f'\n{"=" * 80}')
print(f'SUMMARY')
print(f'{"=" * 80}')
if pos_scores:
    print(f'Positive pairs (same IK, N={len(pos_scores)}):')
    print(f'  TX scores: min={min(pos_scores):.6f}  max={max(pos_scores):.6f}  '
          f'mean={np.mean(pos_scores):.6f}  std={np.std(pos_scores):.6f}')
    print(f'  Scores histogram:')
    for lo, hi in [(0, 0.5), (0.5, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0)]:
        n = sum(1 for s in pos_scores if lo < s <= hi)
        if n > 0:
            print(f'    ({lo:.2f}, {hi:.2f}]: {n}')

if neg_scores:
    print(f'Negative pairs (diff IK, N={len(neg_scores)}):')
    print(f'  TX scores: min={min(neg_scores):.6f}  max={max(neg_scores):.6f}  '
          f'mean={np.mean(neg_scores):.6f}  std={np.std(neg_scores):.6f}')

# Verdict
print(f'\nVERDICT:')
if pos_scores:
    if min(pos_scores) > 0.99:
        print('  ⚠ ALL positive scores > 0.99 → POSSIBLE DEGENERATE MODEL OUTPUT')
    elif np.std(pos_scores) < 0.001:
        print('  ⚠ Near-zero variance in positive scores → scores may be degenerate')
    else:
        print('  ✓ Positive scores show reasonable variance')

if len(pos_scores) < 10:
    print(f'  ⚠ Only {len(pos_scores)} pos pairs scored → N too small, AUC unreliable')
else:
    print(f'  N={len(pos_scores)} is sufficient for stable AUC')

# Cleanup
for f in [mgf_path, hdf5_path, pairs_h5]:
    try: os.remove(f)
    except: pass
