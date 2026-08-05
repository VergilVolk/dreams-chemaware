"""
MassSpecGym HDF5 → MGF + metadata.csv

Input:  data/models/MassSpecGym_MurckoHist_split.hdf5
Output: data/massspecgym/
  train.mgf      (~185K, official train split)
  val.mgf        (~23K, val split first half → evaluation)
  test.mgf       (~22K, val split second half → held-out test)
  metadata.csv   (IK, SMILES, FORMULA, adduct, precursor_mz, fold)

Usage: python tasks/convert_massspecgym_to_mgf.py
"""
import h5py, csv, os, numpy as np
from collections import defaultdict

HDF5_PATH = 'data/models/MassSpecGym_MurckoHist_split.hdf5'
OUT_DIR = 'data/massspecgym'
os.makedirs(OUT_DIR, exist_ok=True)

print('[1] Loading MassSpecGym HDF5...')
f = h5py.File(HDF5_PATH, 'r')

def decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]

smiles_all = decode(f['smiles'][:])
inchi_all = decode(f['INCHIKEY'][:])
formula_all = decode(f['FORMULA'][:])
adduct_all = decode(f['adduct'][:])
precursor_all = f['precursor_mz'][:]
fold_all = decode(f['fold'][:])
spectra = f['spectrum']  # (N, 2, 128)

# Split val into val + test (50/50)
import random
rng = random.Random(42)
val_indices = [i for i, fl in enumerate(fold_all) if fl == 'val']
rng.shuffle(val_indices)
n_val_half = len(val_indices) // 2
val_test_split = {
    'val': set(val_indices[:n_val_half]),
    'test': set(val_indices[n_val_half:]),
}

# Build metadata
ik_to_meta = {}
for i in range(len(smiles_all)):
    ik14 = inchi_all[i][:14]
    if ik14 not in ik_to_meta:
        ik_to_meta[ik14] = {
            'smiles': smiles_all[i],
            'formula': formula_all[i],
            'adducts': set(),
        }
    ik_to_meta[ik14]['adducts'].add(adduct_all[i])

print(f'  Train: {sum(1 for fl in fold_all if fl=="train")}')
print(f'  Val:   {len(val_test_split["val"])}')
print(f'  Test:  {len(val_test_split["test"])}')
print(f'  Unique IKs: {len(ik_to_meta)}')

# Write metadata.csv
print('\n[2] Writing metadata.csv...')
with open(f'{OUT_DIR}/metadata.csv', 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(['inchikey14', 'smiles', 'formula', 'adducts', 'n_spectra_train', 'n_spectra_val', 'n_spectra_test'])
    for ik14, meta in sorted(ik_to_meta.items()):
        # Count spectra per fold per IK
        n_train = 0; n_val = 0; n_test = 0
        # (We'll count as we write MGFs)
        writer.writerow([
            ik14, meta['smiles'], meta['formula'],
            '|'.join(sorted(meta['adducts'])),
            0, 0, 0  # placeholder, updated after MGF write
        ])

# Write MGFs
for fold_name, index_set in [('train', {i for i, fl in enumerate(fold_all) if fl == 'train'}),
                               ('val', val_test_split['val']),
                               ('test', val_test_split['test'])]:
    out_path = f'{OUT_DIR}/{fold_name}.mgf'
    ik_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

    print(f'\n[3] Writing {out_path} ({len(index_set)} spectra)...')
    with open(out_path, 'w', encoding='utf-8') as fout:
        count = 0
        for i in sorted(index_set):
            s = spectra[i]  # (2, 128)
            mz_row = s[0]
            int_row = s[1]

            # Only write peaks where m/z > 0 (non-padding)
            valid = mz_row > 0

            fout.write('BEGIN IONS\n')
            fout.write(f'SMILES={smiles_all[i]}\n')
            fout.write(f'INCHIKEY={inchi_all[i]}\n')
            fout.write(f'FORMULA={formula_all[i]}\n')
            fout.write(f'PEPMASS={precursor_all[i]:.5f}\n')
            fout.write(f'ADDUCT={adduct_all[i]}\n')
            fout.write('IONMODE=POSITIVE\n')
            fout.write('MSLEVEL=2\n')

            for j in range(128):
                if valid[j]:
                    fout.write(f'{mz_row[j]:.4f} {int_row[j]:.6f}\n')
            fout.write('END IONS\n\n')

            ik14 = inchi_all[i][:14]
            ik_counts[ik14][fold_name] += 1
            count += 1

            if count % 50000 == 0:
                print(f'    {count}/{len(index_set)}...', flush=True)

    print(f'  Wrote {count} spectra')

    # Update metadata.csv counts for this fold
    for ik14, cnts in ik_counts.items():
        pass  # counted; will update below

f.close()

# Update metadata.csv with actual counts (re-read and rewrite)
print('\n[4] Updating metadata.csv with spectrum counts...')
# Count spectra from MGF files
ik_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
fold_all_list = list(fold_all)
for i, fl in enumerate(fold_all_list):
    ik14 = inchi_all[i][:14]
    if fl == 'train':
        ik_counts[ik14]['train'] += 1
    elif i in val_test_split['val']:
        ik_counts[ik14]['val'] += 1
    else:
        ik_counts[ik14]['test'] += 1

rows = []
with open(f'{OUT_DIR}/metadata.csv', 'r', encoding='utf-8') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    for row in reader:
        ik14 = row[0]
        cnts = ik_counts.get(ik14, {'train': 0, 'val': 0, 'test': 0})
        row[4] = str(cnts['train'])
        row[5] = str(cnts['val'])
        row[6] = str(cnts['test'])
        rows.append(row)

with open(f'{OUT_DIR}/metadata.csv', 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(header)
    writer.writerows(rows)

# Report
total_mgf = sum(1 for fn in os.listdir(OUT_DIR) if fn.endswith('.mgf'))
total_size = sum(os.path.getsize(f'{OUT_DIR}/{fn}') for fn in os.listdir(OUT_DIR))
print(f'\n=== DONE ===')
print(f'  Files: {total_mgf} MGF + metadata.csv')
print(f'  Total size: {total_size/1e9:.2f} GB')
for fn in ['train.mgf', 'val.mgf', 'test.mgf', 'metadata.csv']:
    path = f'{OUT_DIR}/{fn}'
    if os.path.exists(path):
        n_lines = sum(1 for _ in open(path, encoding='utf-8'))
        print(f'  {fn}: {os.path.getsize(path)/1e6:.1f} MB ({n_lines} lines)')
