"""
Prep Task 1: MassSpecGym test fold exclusion list

防止 annotated01 中出现在 MassSpecGym test fold 的分子被用于训练。
使用 metadata.csv 中的 n_spectra_test 字段标识 test 分子。

输出:
  data/exclusion/
    massspecgym_test_iks.txt      — MassSpecGym test 的 IK14 列表
    annotated01_exclusion.txt     — annotated01 中需要排除的 IK14 (交集)
    exclusion_report.json

用法: python tasks/build_exclusion_list.py
"""
import csv, json, os, sys

sys.path.insert(0, '.')
from tasks.build_utils import load_indices

OUT_DIR = 'data/exclusion'
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Extract test IK14 from metadata.csv
print('[1] Extracting test IKs from MassSpecGym metadata.csv...')
test_iks = set()
with open('data/massspecgym/metadata.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['n_spectra_test']) > 0:
            test_iks.add(row['inchikey14'])

print(f'  MassSpecGym test IKs: {len(test_iks)}')

# Save
with open(f'{OUT_DIR}/massspecgym_test_iks.txt', 'w') as f:
    for ik in sorted(test_iks):
        f.write(f'{ik}\n')

print(f'  Saved: {OUT_DIR}/massspecgym_test_iks.txt')

# 2. Intersect with annotated01
print('[2] Intersecting with annotated01...')
idx = load_indices()
anno_iks = set(idx['ik_to_smi'].keys())
print(f'  annotated01 IKs: {len(anno_iks)}')

overlap = test_iks & anno_iks
overlap_pct = len(overlap) / len(anno_iks) * 100

print(f'  Overlap: {len(overlap)} ({overlap_pct:.1f}% of annotated01)')

with open(f'{OUT_DIR}/annotated01_exclusion.txt', 'w') as f:
    for ik in sorted(overlap):
        f.write(f'{ik}\n')

print(f'  Saved: {OUT_DIR}/annotated01_exclusion.txt')

# 3. Report
report = {
    'massspecgym_test_iks': len(test_iks),
    'annotated01_iks': len(anno_iks),
    'overlap': len(overlap),
    'overlap_pct': round(overlap_pct, 2),
    'leakage_severity': 'LOW' if overlap_pct < 20 else ('MEDIUM' if overlap_pct < 50 else 'HIGH'),
    'recommendation': (
        'Safe to use annotated01 for training, exclude these IKs'
        if overlap_pct < 20 else
        'Significant overlap — consider using MassSpecGym train fold only'
    )
}

with open(f'{OUT_DIR}/exclusion_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f'\n=== EXCLUSION REPORT ===')
for k, v in report.items():
    print(f'  {k}: {v}')
