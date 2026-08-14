"""
run_ablation.py — 消融实验：直接用 TransExION CSV 做评估

数据: explainability_output/transexion_massbank.csv (10K pairs)
A: 规则引擎 weighted Jaccard
B: A → 待 TransExION 同数据对齐

用法: python -m dreams.models.mil_interpretable.run_ablation
"""
import torch, numpy as np, json, csv, re
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from rdkit import Chem; from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs

# Bypass DreaMS import chain
import importlib.util
spec = importlib.util.spec_from_file_location('chem_rules', 'dreams/models/chem_aware/chem_rules.py')
chem_rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(chem_rules)
ChemicalRuleEngine = chem_rules.ChemicalRuleEngine

def parse_msp_simple(filepath, max_spec=50000):
    spectra=[]; cur={}; peaks=[]
    with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line:
                if cur and peaks: cur['peaks']=peaks; spectra.append(cur)
                if len(spectra)>=max_spec: break
                cur={}; peaks=[]; continue
            if ': ' in line:
                k,v=line.split(': ',1)
                if k in ('Name','InChIKey','SMILES','Formula','PrecursorMZ','Ion_mode','Precursor_type'): cur[k]=v
                if k=='Comments' and 'SMILES' not in cur:
                    m=re.search(r'SMILES="?([^"]+?)"?\s',v)
                    if m: cur['SMILES']=m.group(1).strip()
            else:
                p=line.split()
                if len(p)==2:
                    try: mz,i=float(p[0]),float(p[1])
                    except: continue
                    if mz>0 and i>0: peaks.append((mz,i))
    if cur and peaks and len(spectra)<max_spec: cur['peaks']=peaks; spectra.append(cur)
    return spectra

class MinimalSP:
    def __init__(self): self.n_highest_peaks=128
    def __call__(self,spec,_=False):
        arr=np.array(spec,dtype=np.float32)
        if arr.ndim==2 and arr.shape[0]==2: arr=arr.T
        valid=arr[:,0]>0.1; arr=arr[valid]
        if len(arr)>self.n_highest_peaks:
            idx=np.argsort(arr[:,1])[-self.n_highest_peaks:]; arr=arr[idx]
        return arr[arr[:,0].argsort()]

def compute_match_vec(spectrum,engine,sp):
    peaks=spectrum.get('peaks',[])
    if len(peaks)<3: return None
    arr=np.array(peaks,dtype=np.float32); arr=arr[arr[:,0].argsort()][:128]
    try:
        spec_pp=sp(arr.T); spec_t=torch.as_tensor(spec_pp,dtype=torch.float32)
        mz=spec_t[:,0].unsqueeze(0); pad=mz[:,0]==0
        mz_diffs=torch.abs(mz.unsqueeze(-1)-mz.unsqueeze(-2))
        vec=engine.get_rule_match_vectors(mz_diffs,mz_values=mz,
            precursor_mz=mz[:,0].unsqueeze(0),padding_mask=pad,
            categories=['NL','CF','ISO','HR'])
        return vec.squeeze(0)
    except: return None

def main():
    engine=ChemicalRuleEngine(tolerance=0.02); sp=MinimalSP()

    # Load ALL MassBank spectra once, index by InChIKey
    print('[1] Loading MassBank spectra...')
    spectra=parse_msp_simple('data/MassBank_NIST.msp',50000)
    ik_to_spec=defaultdict(list)
    for s in spectra:
        ik=s.get('InChIKey','').strip()[:14]  # TransExION uses short InChIKey (first 14 chars)
        if ik: ik_to_spec[ik].append(s)
    print(f'  {len(spectra)} spectra, {len(ik_to_spec)} unique InChIKeys')

    # Compute rule vectors for ALL spectra
    print('[2] Computing rule vectors...')
    vec_cache={}
    for ik,ss in ik_to_spec.items():
        for s in ss:  # ALL spectra per InChIKey
            vec=compute_match_vec(s,engine,sp)
            if vec is not None: vec_cache[id(s)]=vec
    print(f'  {len(vec_cache)} vectors computed')

    # Level weights
    lvl_w=np.ones(len(engine.rules),dtype=np.float32)
    for idx,r in enumerate(engine.rules):
        if r.category in ('HR','ISO'): lvl_w[idx]=4.0
        elif r.category in ('NR','EE'): lvl_w[idx]=1.0
        else: lvl_w[idx]=2.0

    # Load TransExION scores (match by short InChIKey)
    print('[3] Loading TransExION scores...')
    tx_scores={}
    with open('explainability_output/transexion_massbank.csv') as f:
        for r in csv.DictReader(f):
            tx_scores[(r['query_key'], r['ref_key'])]=float(r['predict score'])
    print(f'  {len(tx_scores)} TransExION pairs')

    # Build BALANCED evaluation pairs from our parsed data
    print('[4] Building balanced evaluation pairs...')
    rng=np.random.RandomState(42)
    multi_ik={k:v for k,v in ik_to_spec.items() if len(v)>=2}
    pairs=[]; labels=[]

    # Positive: same InChIKey, different spectra
    seen_pos=set()
    for ik in sorted(multi_ik.keys(),key=lambda x:-len(multi_ik[x])):
        idxs=list(range(len(multi_ik[ik])))
        for _ in range(min(5,len(idxs)*(len(idxs)-1)//2)):
            a,b=rng.choice(idxs,2,replace=False)
            pk=(min(a,b),max(a,b))
            if pk in seen_pos: continue
            seen_pos.add(pk)
            pairs.append((multi_ik[ik][a], multi_ik[ik][b])); labels.append(1)
    n_pos=len(pairs)

    # Negative: different InChIKey (random)
    all_iks=list(ik_to_spec.keys())
    seen_neg=set()
    for _ in range(50000):
        ika,ikb=rng.choice(all_iks,2,replace=False)
        if ika==ikb: continue
        pk=(ika,ikb)
        if pk in seen_neg: continue
        seen_neg.add(pk)
        pairs.append((ik_to_spec[ika][0], ik_to_spec[ikb][0])); labels.append(0)
        if len(pairs)-n_pos>=10000: break

    labels=np.array(labels)
    n_pos=labels.sum(); n_neg=len(labels)-n_pos
    print(f'  {len(pairs)} pairs ({n_pos} pos, {n_neg} neg)')

    # Compute signals
    print('[5] Computing signals...')
    scores_A=[]; scores_B=[]; valid_labels=[]
    n_tx=0
    for p_idx,(sa,sb) in enumerate(pairs):
        if id(sa) not in vec_cache or id(sb) not in vec_cache: continue
        va=vec_cache[id(sa)].float(); vb=vec_cache[id(sb)].float()
        wi=((va*vb)*torch.tensor(lvl_w)).sum().item()
        wu=(((va+vb)>0).float()*torch.tensor(lvl_w)).sum().item()
        wj=wi/max(wu,1)
        scores_A.append(wj)
        # Match TransExION
        ik_a=sa.get('InChIKey','')[:14]; ik_b=sb.get('InChIKey','')[:14]
        tx=tx_scores.get((ik_a,ik_b),tx_scores.get((ik_b,ik_a),None))
        if tx is not None: n_tx+=1; scores_B.append(0.5*wj+0.5*tx)
        else: scores_B.append(wj)
        valid_labels.append(labels[p_idx])

    labels=np.array(valid_labels)
    n_pos=labels.sum(); n_neg=len(labels)-n_pos
    print(f'  {len(labels)} valid pairs ({n_pos} pos, {n_neg} neg), TX matched: {n_tx}')

    # AUC
    auc_A=roc_auc_score(labels,scores_A); r_A,_=pearsonr(scores_A,labels.astype(float))
    auc_B=roc_auc_score(labels,scores_B); r_B,_=pearsonr(scores_B,labels.astype(float))

    sep='='*60
    print(f'\n{sep}')
    print(f'  ABLATION RESULTS')
    print(f'{sep}')
    print(f'  Pairs: {len(labels)} ({n_pos} pos, {n_neg} neg)')
    print(f'  A: Rule Engine (wJaccard)     AUC={auc_A:.4f}  r={r_A:.4f}')
    print(f'  B: A + TransExION (0.5/0.5)  AUC={auc_B:.4f}  r={r_B:.4f}')
    print(f'')
    print(f'  Delta: B improves over A by {auc_B-auc_A:+.4f} AUC')
    print(f'{sep}')

    json.dump({'A':scores_A,'B':scores_B,'labels':labels.tolist(),
               'auc_A':auc_A,'auc_B':auc_B},
              open('explainability_output/ablation_results.json','w'))

if __name__=='__main__': main()
