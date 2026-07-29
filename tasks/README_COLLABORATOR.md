# T0-T3 任务数据 + 训练 — 快速上手指南

## 环境要求

```bash
conda create -n dreams_env python=3.10 -y
conda activate dreams_env
pip install torch rdkit numpy scipy scikit-learn tqdm h5py
pip install myopic-mces  # MCES 计算
```

## 数据准备

### 1. 构建 annotated01.mgf（一次性，~30分钟）

```bash
python build_annotated01.py
```

这会扫描 `data/` 下所有 MGF/MSP 文件，去重，输出 `data/annotated01.mgf`。

### 2. 构建索引缓存（一次性，~2分钟）

```bash
python -c "from tasks.build_utils import load_indices; idx = load_indices(force_rebuild=True)"
```

输出: `tasks/_cache/indices.json`

### 3. 构建 T0-T3 pairs

```bash
# T0（同分子一致性，秒级）
python tasks/T0_consistency/code/build_test_cases.py

# T1（近同分异构体，需要 MCES，~1-2小时）
python tasks/T1_near_isomers/code/build_test_cases.py --validate --n_groups 100
# 确认分布后:
python tasks/T1_near_isomers/code/build_test_cases.py --max_groups 2000

# T2（类似物，需要 MCES，~1-2小时）
python tasks/T2_analogs/code/build_test_cases.py --validate --n_groups 100
python tasks/T2_analogs/code/build_test_cases.py --max_groups 2000

# T3（不相关基线，秒级）
python tasks/T3_unrelated/code/build_test_cases.py
```

### 4. 构建 T1 triplets（基于已有 pairs，秒级）

```bash
python -c "
import sys; sys.path.insert(0,'.')
from tasks.build_utils import load_indices
import json, numpy as np
from collections import defaultdict

idx = load_indices()
with open('tasks/T1_near_isomers/test_cases/pairs.json') as f:
    t1 = json.load(f)
with open('tasks/T3_unrelated/test_cases/pairs.json') as f:
    t3 = json.load(f)

pos = t1['positive']       # MCES [0,2]
neg_hard = t1['negative_hard']  # MCES [6,10]
neg_easy = t3['negative']  # 不同分子式

rng = np.random.RandomState(42)
ik_to_fm = idx['ik_to_fm']
ik_to_peaks = idx['ik_to_peaks']

# Index negatives by formula
fm_to_neg = defaultdict(list)
for p in neg_hard:
    for k in ('ik_a','ik_b'):
        fm = ik_to_fm.get(p[k], '')
        if fm: fm_to_neg[fm].append(p[k])
t3_iks = list(set(p.get('ik_a','')[:14] for p in neg_easy if p.get('ik_a','')))

triplets = []
for pp in pos:
    ik_a, ik_b = pp['ik_a'], pp['ik_b']
    fm = pp.get('fm','') or ik_to_fm.get(ik_a,'')
    if ik_a not in ik_to_peaks or ik_b not in ik_to_peaks: continue
    for anchor, pos_ik in [(ik_a, ik_b), (ik_b, ik_a)]:
        negs = [n for n in fm_to_neg.get(fm,[]) if n not in (anchor, pos_ik)]
        negs += [n for n in t3_iks if n not in (anchor, pos_ik)]
        if not negs: continue
        chosen = rng.choice(list(set(negs)), min(5, len(set(negs))), replace=False)
        for n in chosen:
            if n in ik_to_peaks:
                triplets.append({'anchor_ik': anchor, 'pos_ik': pos_ik, 'neg_ik': n})

# Split by anchor IK
anchors = sorted(set(t['anchor_ik'] for t in triplets))
rng.shuffle(anchors)
val_anchors = set(anchors[:max(1, int(len(anchors)*0.1))])
train = [t for t in triplets if t['anchor_ik'] not in val_anchors]
val = [t for t in triplets if t['anchor_ik'] in val_anchors]

for name, data in [('triplets_train', train), ('triplets_val', val)]:
    with open(f'tasks/T1_near_isomers/test_cases/{name}.json', 'w') as f:
        json.dump(data, f)
print(f'Triplets: {len(train)} train + {len(val)} val')
"
```

### 5. MS2DeepScore 分数（可选）

首先转换模型:
```bash
python tasks/convert_ms2ds_model.py  # Keras→PyTorch
```

然后计算分数:
```bash
python tasks/step1_ms2deepscore.py --task T1
python tasks/step1_ms2deepscore.py --task T2
python tasks/step1_ms2deepscore.py --task T3
```

### 6. 消融实验

```bash
python tasks/run_ablation.py --task T1
python tasks/run_ablation.py --task T2
```

### 7. Triplet 训练（需要 DreaMS 预训练权重）

```bash
python -m dreams.models.chem_aware.train_triplet_t1 \
    --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
    --epochs 10 --batch_size 32 --alpha 0.5 --beta 0.01 --margin 0.3
```

---

## 输出格式

### pairs.json（T1 示例）

```json
{
  "positive": [
    {
      "mces_raw": 0.0,
      "mces_norm": 0.0,
      "tanimoto": 1.0,
      "ik_a": "UJNSFDHVIBGEJZ-CMRIBGNTSA-N",
      "ik_b": "UJNSFDHVIBGEJZ-UHFFFAOYSA-N",
      "smi_a": "C/C1=C/CC[C@@]2(C)O...",
      "smi_b": "CC1=CCCC2(C)OC2...",
      "fm": "C17H27NO3"
    }
  ],
  "negative_hard": [
    {
      "mces_raw": 6.0,
      "ik_a": "...",
      "ik_b": "...",
      "fm": "C17H27NO3"
    }
  ],
  "negative_easy": [...]
}
```

### triplets_train.json

```json
[
  {
    "anchor_ik": "UJNSFDHVIBGEJZ",
    "pos_ik": "UJNSFDHVIBGEJZ",
    "neg_ik": "OWYMWLCFHDHVAH"
  }
]
```

**注意**: IKs 已截断为 14 字符（前 14 位），匹配 annotated01.mgf 格式。

### 消融实验输出

```
=== T1 ABLATION ===
  Baseline: Rule Engine wJaccard     : AUC=0.6908  r=0.2819
  A: MS2DeepScore                    : AUC=0.8008  r=0.3494
  B: TransExION                      : AUC=0.8079  r=0.3205
  A+B: Rule+TX+MS2DS (dynamic)       : AUC=0.7587  r=0.3739
```

---

## 当前数据统计

| 任务 | Pos pairs | Neg pairs | 说明 |
|------|-----------|-----------|------|
| T0 | 128,760 | 10,000 | 同分子不同谱图 |
| T1 | 781 | 4,087 hard + 4,087 easy | MCES [0,2] pos, [6,10] neg |
| T2 | 452 | 875 | MCES [3,5] pos, >5 neg |
| T3 | 0 | 10,000 | 不同分子式，纯负样本 |

T1 Triplets: 7,030 train + 780 val（按 anchor IK 分组防泄漏）

## 已知结果

| 任务 | 规则引擎 | MS2DeepScore | TransExION | A+B |
|------|---------|-------------|------------|-----|
| T1 | 0.6908 | 0.8008 | 0.8079 | 0.7587 |
| T2 | 0.6198 | 0.6557 | 0.6143 | 0.6377 |
