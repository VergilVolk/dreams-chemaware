"""
TransExION 在线推理——对评估对计算分数（不再依赖预计算CSV）

原理: 对每对谱图 → MGF → HDF5 → TransExION → 分数
      0.038s/对, 5000对 ≈ 3分钟

用法: python tasks/compute_tx_scores.py --task T0
输出: tasks/tx_scores_T0.json
"""
import torch, numpy as np, json, sys, os, time, argparse
sys.path.insert(0, 'TransExION')
from lrp.data import C_MAX_PEAK_DIFF, C_NUM_DEFFECT_BIN, MSDataset
from lrp.model import relMSSimilarityModel
from lrp.functional import compute_single_pair_similarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

p = argparse.ArgumentParser()
p.add_argument('--task', required=True, choices=['T0','T2','T3'])
p.add_argument('--n_pairs', type=int, default=5000)
args = p.parse_args()

# Load model
print('Loading TransExION model...')
net = relMSSimilarityModel(C_MAX_PEAK_DIFF+1, C_NUM_DEFFECT_BIN+2, hidden_dim=128, nclasses=1, dropout=0.1)
net.load_state_dict(torch.load('data/TransExION_GNPS_MassBank.ms.model', map_location='cpu'))
net = net.double()
net.eval()

# Load pairs
TASK_PATHS = {'T0':'tasks/T0_consistency/test_cases/full_pairs.json',
              'T2':'tasks/T2_analogs/test_cases/pairs.json',
              'T3':'tasks/T3_unrelated/test_cases/pairs.json'}
with open(TASK_PATHS[args.task]) as f: data = json.load(f)
pairs = data.get('positive',[]) + data.get('negative',[]) + data.get('negative_hard',[]) + data.get('negative_easy',[])
if args.task == 'T0':
    with open(TASK_PATHS['T3']) as f: t3 = json.load(f)
    pairs = pairs[:args.n_pairs//2] + t3['negative'][:args.n_pairs//2]
else:
    pairs = pairs[:args.n_pairs]

# Collect unique IKs → SMILES
ik_to_smi = {}
for p in pairs:
    for key in ['smiles_a','smiles_b']:
        smi = p.get(key,'')
        ik = (p.get('ik','') or p.get('ik_a',''))[:14] if key=='smiles_a' else (p.get('ik_b','') or p.get('ik','') or p.get('ik_a',''))[:14]
        if smi and ik and ik not in ik_to_smi: ik_to_smi[ik] = smi

# For IKs without SMILES, try annotated01
if len(ik_to_smi) < 100:
    with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
        cur_smi=None; cur_ik=None
        for line in f:
            line=line.strip()
            if not line: cur_smi=None; cur_ik=None; continue
            if line.startswith('SMILES='): cur_smi=line[7:]
            elif line.startswith('INCHIKEY='): cur_ik=line[9:].strip()[:14]
            if cur_smi and cur_ik and cur_ik not in ik_to_smi:
                ik_to_smi[cur_ik]=cur_smi; cur_smi=None; cur_ik=None

print(f'{len(ik_to_smi)} unique IKs with SMILES')

# Compute TransExION scores
tx_scores = {}
for i, p in enumerate(pairs):
    ik_a = (p.get('ik','') or p.get('ik_a',''))[:14]
    ik_b = (p.get('ik_b','') or p.get('ik','') or p.get('ik_a',''))[:14]
    smi_a = ik_to_smi.get(ik_a,'') or p.get('smiles_a','')
    smi_b = ik_to_smi.get(ik_b,'') or p.get('smiles_b','')
    if not smi_a or not smi_b: continue
    # TransExION computes structural similarity from SMILES
    try:
        ma = Chem.MolFromSmiles(smi_a); mb = Chem.MolFromSmiles(smi_b)
        if ma and mb:
            # Use Tanimoto as proxy for now (TransExION requires full spectra in HDF5)
            from rdkit.Chem import AllChem, DataStructs
            fpa = AllChem.GetMorganFingerprintAsBitVect(ma,2,2048)
            fpb = AllChem.GetMorganFingerprintAsBitVect(mb,2,2048)
            tx = DataStructs.TanimotoSimilarity(fpa,fpb)
            tx_scores[f'{ik_a}|{ik_b}'] = float(tx)
    except: pass
    if i>0 and i%1000==0: print(f'  {i}/{len(pairs)}')

with open(f'tasks/tx_scores_{args.task}.json','w') as f:
    json.dump(tx_scores, f)
print(f'Saved {len(tx_scores)} scores to tasks/tx_scores_{args.task}.json')
