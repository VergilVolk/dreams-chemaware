"""
T0 训练: 一致性任务 — 预测同一 InChIKey 的谱图对
用法: python tasks/T0_consistency/code/train.py
"""
import torch, numpy as np, json, os
from collections import defaultdict
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine

# Config
CFG = {'hidden_dim':32, 'dropout':0.1, 'entropy':0.001, 'lr':1e-4, 'wd':1e-5,
       'bs':32, 'epochs':500, 'T_0':100, 'n_folds':5}

engine = ChemicalRuleEngine(tolerance=0.02)
lvl_w = torch.ones(len(engine.rules), dtype=torch.float32)
for idx,r in enumerate(engine.rules):
    if r.category in ('HR','ISO'): lvl_w[idx]=4.0
    elif r.category in ('NR','EE'): lvl_w[idx]=1.0
    else: lvl_w[idx]=2.0

# Load pairs
print('Loading T0 pairs...')
with open('tasks/T0_consistency/test_cases/full_pairs.json') as f:
    data = json.load(f)
pos = data['positive']

# Build simple features: for each pair, compute wJaccard from match vectors
# Use annotated01.mgf for spectra
print(f'{len(pos)} positive pairs')

# Quick baseline: train LR on overlap features
# For now, report data readiness
print(f'T0 ready: {len(pos)} positive pairs for training')
print(f'Config: {CFG}')
print(f'Next: compute rule vectors for all pairs, train MIL')
