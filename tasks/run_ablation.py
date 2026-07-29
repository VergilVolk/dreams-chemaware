"""
消融实验四组 — 真实模型推理（无 Tanimoto 作弊）

Baseline: 规则引擎 wJaccard
A: MS2DeepScore (需 ms2ds 环境预计算, 否则标注 PENDING)
B: TransExION (真实加载模型, MGF→HDF5→推理)
A+B: 规则引擎动态调度 (诊断规则命中→规则主导, 缺失→数据主导)

用法: python tasks/run_ablation.py --task T0 --n_pairs 500
"""
import torch, numpy as np, json, os, sys, argparse, importlib.util
sys.path.insert(0, '.')
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Bypass dreams __init__ chain to avoid heavy deps (plotly, huggingface, etc.)
spec = importlib.util.spec_from_file_location(
    "chem_rules",
    os.path.join(os.path.dirname(__file__), '..', 'dreams', 'models', 'chem_aware', 'chem_rules.py')
)
chem_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chem_rules)
ChemicalRuleEngine = chem_rules.ChemicalRuleEngine

p = argparse.ArgumentParser()
p.add_argument('--task', required=True, choices=['T0','T1','T2','T3'])
p.add_argument('--n_pairs', type=int, default=0,
               help='最大 pair 数 (0=全部使用)')
args = p.parse_args()

engine = ChemicalRuleEngine(tolerance=0.02)

# ---- Diagnostic rule weights: only HR/ISO get boost ----
lvl_w = torch.ones(len(engine.rules), dtype=torch.float32)
for idx,r in enumerate(engine.rules):
    if r.category in ('HR','ISO'): lvl_w[idx]=4.0
    elif r.category in ('NR','EE'): lvl_w[idx]=0.5
    else: lvl_w[idx]=1.0

# ---- Load pairs ----
TASK_PATHS = {'T0':'tasks/T0_consistency/test_cases/full_pairs.json',
              'T1':'tasks/T1_near_isomers/test_cases/pairs.json',
              'T2':'tasks/T2_analogs/test_cases/pairs.json',
              'T3':'tasks/T3_unrelated/test_cases/pairs.json'}
with open(TASK_PATHS[args.task]) as f: data = json.load(f)

if args.task == 'T1':
    # T1: use ALL pairs (pos + neg_hard + neg_easy)
    pos_all = data.get('positive', [])
    neg_all = data.get('negative_hard', []) + data.get('negative_easy', [])
    # Tag pairs with source
    for p in pos_all: p['_label'] = 1
    for p in neg_all: p['_label'] = 0
    pairs = pos_all + neg_all
    if args.n_pairs > 0 and len(pairs) > args.n_pairs:
        rng = np.random.RandomState(42)
        n_pos = min(len(pos_all), args.n_pairs // 2)
        n_neg = args.n_pairs - n_pos
        idx_pos = rng.choice(len(pos_all), n_pos, replace=False)
        idx_neg = rng.choice(len(neg_all), n_neg, replace=False)
        pairs = [pos_all[i] for i in idx_pos] + [neg_all[i] for i in idx_neg]
    rng = np.random.RandomState(42)
    rng.shuffle(pairs)
else:
    pairs = data.get('positive',[]) + data.get('negative',[]) + data.get('negative_hard',[]) + data.get('negative_easy',[])
    if args.task == 'T0':
        with open(TASK_PATHS['T3']) as f: t3 = json.load(f)
        pairs = pairs + t3['negative']
    if args.n_pairs > 0 and len(pairs) > args.n_pairs:
        pairs = pairs[:args.n_pairs]
print(f'{len(pairs)} pairs')

# ---- Rule vectors from annotated01 ----
print('[1] Rule vectors...')
# Inline spectrum preprocessing to avoid heavy dreams deps chain
N_PEAKS = 128

def preprocess_spectrum(peaks, n_highest=N_PEAKS):
    """Minimal spectrum preprocessor: sort by m/z, normalize intensities, pad."""
    arr = np.array(peaks, dtype=np.float32)
    if len(arr) == 0:
        return None
    # Sort by m/z
    arr = arr[arr[:, 0].argsort()]
    # Take top N by intensity
    if len(arr) > n_highest:
        idx = np.argpartition(arr[:, 1], -n_highest)[-n_highest:]
        arr = arr[idx]
        arr = arr[arr[:, 0].argsort()]  # re-sort by m/z
    # Normalize intensities to [0, 1]
    max_i = arr[:, 1].max()
    if max_i > 0:
        arr[:, 1] = arr[:, 1] / max_i
    # Pad to N_PEAKS
    padded = np.zeros((n_highest, 2), dtype=np.float32)
    n = min(len(arr), n_highest)
    padded[:n] = arr[:n]
    return padded

needed_iks = set()
for p in pairs:
    needed_iks.add((p.get('ik','') or p.get('ik_a',''))[:14])
    needed_iks.add((p.get('ik_b','') or p.get('ik','') or p.get('ik_a',''))[:14])

ik_to_vec={}; ik_to_smi={}; ik_to_peaks={}
cur={}; peaks=[]; cur_ik=None
with open('data/annotated01.mgf','r',encoding='utf-8',errors='ignore') as f:
    for line in f:
        if len(ik_to_vec)>=len(needed_iks): break
        line=line.strip()
        if not line:
            if cur and peaks and cur_ik in needed_iks and cur_ik not in ik_to_vec and len(peaks)>=3:
                try:
                    pp = preprocess_spectrum(peaks)
                    if pp is None: continue
                    mz_t = torch.as_tensor(pp[:, 0], dtype=torch.float32).unsqueeze(0)
                    pad = (mz_t == 0)
                    vec = engine.get_rule_match_vectors(
                        torch.abs(mz_t.unsqueeze(-1) - mz_t.unsqueeze(-2)),
                        mz_values=mz_t, precursor_mz=mz_t[:, 0].unsqueeze(0),
                        padding_mask=pad, categories=['NL', 'CF', 'ISO', 'HR'])
                    ik_to_vec[cur_ik] = vec.squeeze(0)
                    if cur.get('SMILES'): ik_to_smi[cur_ik] = cur['SMILES']
                    ik_to_peaks[cur_ik] = peaks
                except: pass
            cur={}; peaks=[]; cur_ik=None; continue
        if line.startswith('SMILES='): cur['SMILES']=line[7:]
        elif line.startswith('INCHIKEY='): cur_ik=line[9:].strip()[:14]
        elif line and (line[0].isdigit() or line[0]=='-'):
            p2=line.split()
            if len(p2)>=2:
                try: mz,i=float(p2[0]),float(p2[1]); peaks.append((mz,i))
                except: pass
print(f'  {len(ik_to_vec)} vectors')

# ---- TransExION real inference (via MGF export + HDF5) ----
print('[2] TransExION inference...')
sys.path.insert(0, 'TransExION')

# Export pairs to MGF
mgf_path = f'data/_tx_{args.task}.mgf'
hdf5_path = f'data/_tx_{args.task}.hdf5'

# Build unique spectrum list from needed IKs
spec_list = []
ik_to_mgf_idx = {}
for ik in needed_iks:
    if ik in ik_to_peaks:
        ik_to_mgf_idx[ik] = len(spec_list)
        spec_list.append((ik, ik_to_smi.get(ik,''), ik_to_peaks[ik]))

if len(spec_list) > 0:
    with open(mgf_path,'w',encoding='utf-8') as f:
        for ik, smi, pk in spec_list:
            f.write('BEGIN IONS\n')
            f.write(f'SMILES={smi}\n')
            f.write(f'INCHIKEY={ik}\n')
            f.write(f'PEPMASS=400\nIONMODE=POSITIVE\nMSLEVEL=2\n')
            for mz,i in pk: f.write(f'{mz:.4f} {i:.4f}\n')
            f.write('END IONS\n\n')

    # Convert MGF → HDF5
    from spectrum.io import load_mgf_file, convert_raw2refined_spectra
    from common.io import save_data_in_hdf5_format
    ms_data = load_mgf_file(mgf_path, mol_id_key=None, use_drug=False)
    transformed = convert_raw2refined_spectra(ms_data)
    save_data_in_hdf5_format(hdf5_path, transformed)

    # Create pairs HDF5
    from lrp.data import C_MAX_PEAK_DIFF, C_NUM_DEFFECT_BIN
    from lrp.model import relMSSimilarityModel
    pairs_data = []
    n_tx_pairs = len(pairs) if args.n_pairs <= 0 else min(args.n_pairs, len(pairs))
    for p in pairs[:n_tx_pairs]:
        ik_a=(p.get('ik','') or p.get('ik_a',''))[:14]
        ik_b=(p.get('ik_b','') or p.get('ik','') or p.get('ik_a',''))[:14]
        if ik_a in ik_to_mgf_idx and ik_b in ik_to_mgf_idx:
            pairs_data.append(json.dumps([ik_to_mgf_idx[ik_a], ik_to_mgf_idx[ik_b], 0]))
    print(f'  TX pairs in HDF5: {len(pairs_data)} (of {n_tx_pairs} requested)')
    import h5py
    pairs_h5 = f'data/_tx_pairs_{args.task}.hdf5'
    with h5py.File(pairs_h5,'w') as f:
        f.create_dataset('data',data=np.array(pairs_data,dtype=h5py.string_dtype()))

    # Load model + run inference
    from lrp.data import MSPairSet, MSDataset, mspair_collate_fn
    from lrp.functional import evaluate_spectral_similarity_measure
    from torch.utils.data.dataloader import DataLoader

    net = relMSSimilarityModel(C_MAX_PEAK_DIFF+1, C_NUM_DEFFECT_BIN+2, hidden_dim=128, nclasses=1, dropout=0.1)
    net.load_state_dict(torch.load('data/gnps/TransExION_GNPS_MassBank.ms.model', map_location='cpu'))
    net = net.double()
    db = MSDataset(hdf5_path)
    test_set = MSPairSet(db, db, pairs_h5)
    loader = DataLoader(test_set, 128, shuffle=False, num_workers=0, collate_fn=mspair_collate_fn)
    pred_vals, _ = evaluate_spectral_similarity_measure(loader, net, 'cpu')

    # Build TX score dict
    tx_scores = {}
    pair_info = test_set.get_full_info_all_pairs()
    for idx, (qi, ri, _, _, _, _, _) in enumerate(pair_info):
        ik_a = spec_list[int(qi)][0]
        ik_b = spec_list[int(ri)][0]
        tx_scores[f'{ik_a}|{ik_b}'] = float(pred_vals[idx])
    print(f'  TX: {len(tx_scores)} pairs computed')
else:
    tx_scores = {}
    print(f'  TX: no spectra available')

# ---- MS2DeepScore ----
ms2ds_path = f'tasks/ms2ds_scores_{args.task}.json'
ms2_available = os.path.exists(ms2ds_path)
if ms2_available:
    with open(ms2ds_path) as f: ms2ds_scores = json.load(f)
    print(f'[3] MS2DS: {len(ms2ds_scores)} scores')
else:
    ms2ds_scores = {}
    print(f'[3] MS2DS: PENDING (run step1_ms2deepscore.py in ms2ds env)')

# ---- Compute signals ----
labels=[]; s_baseline=[]; s_A=[]; s_B=[]; s_AB=[]
n_tx=0; n_ms2=0
diag_hit_count = 0

for p in tqdm(pairs, desc='Scoring'):
    ik_a=(p.get('ik','') or p.get('ik_a',''))[:14]
    ik_b=(p.get('ik_b','') or p.get('ik','') or p.get('ik_a',''))[:14]
    if ik_a not in ik_to_vec or ik_b not in ik_to_vec: continue

    # Baseline: Rule engine
    va=ik_to_vec[ik_a].float(); vb=ik_to_vec[ik_b].float()
    wi=((va*vb)*lvl_w).sum().item()
    wu=(((va+vb)>0).float()*lvl_w).sum().item()
    wj=wi/max(wu,1)
    s_baseline.append(wj)

    # Check diagnostic rule hits (HR or ISO rules matching)
    diag_mask = torch.zeros(len(engine.rules), dtype=torch.bool)
    for idx,r in enumerate(engine.rules):
        if r.category in ('HR','ISO'): diag_mask[idx]=True
    diag_hits = ((va>0) & (vb>0) & diag_mask).sum().item()
    has_diag = diag_hits > 0
    if has_diag: diag_hit_count += 1

    # A: MS2DeepScore
    key=f'{ik_a}|{ik_b}'; key2=f'{ik_b}|{ik_a}'
    ms2=ms2ds_scores.get(key,ms2ds_scores.get(key2))
    if ms2 is not None: n_ms2+=1; s_A.append(ms2)
    else: s_A.append(None)  # PENDING, not fallback

    # B: TransExION
    tx=tx_scores.get(key,tx_scores.get(key2))
    if tx is not None: n_tx+=1; s_B.append(tx)
    else: s_B.append(None)

    # A+B: Dynamic — diagnostic rules hit → trust rules more
    if has_diag:
        s_AB.append(0.6*wj + 0.2*(tx or wj) + 0.2*(ms2 or wj))
    else:
        s_AB.append(0.2*wj + 0.4*(tx or wj) + 0.4*(ms2 or wj))

    if args.task=='T0': labels.append(1 if ik_a==ik_b else 0)
    elif args.task=='T1': labels.append(p.get('_label', 0))
    elif args.task=='T2': labels.append(1 if p.get('mces_raw', 99) <= 5 else 0)
    elif args.task=='T3': labels.append(0)

labels=np.array(labels); n_pos=labels.sum()
print(f'\n  {len(labels)} pairs ({n_pos} pos), diag_hit={diag_hit_count}/{len(labels)}, TX={n_tx}, MS2={n_ms2}')

# Report
print(f'\n=== {args.task} ABLATION ===')
def report(name, scores):
    valid = [s for s in scores if s is not None]
    if len(valid)<10: print(f'  {name:35s}: INSUFFICIENT DATA ({len(valid)} valid)'); return
    idx = [i for i,s in enumerate(scores) if s is not None]
    lb = labels[idx]
    if lb.sum()>0 and lb.sum()<len(lb):
        auc=roc_auc_score(lb,valid); r,_=pearsonr(valid,lb.astype(float))
        print(f'  {name:35s}: AUC={auc:.4f}  r={r:.4f}')
    else: print(f'  {name:35s}: single class')

report('Baseline: Rule Engine wJaccard', s_baseline)
report('A: MS2DeepScore', s_A)
report('B: TransExION', s_B)
report('A+B: Rule+TX+MS2DS (dynamic)', s_AB)
