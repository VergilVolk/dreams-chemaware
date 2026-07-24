"""V1/V2/V3 数据质量验证 — 对 A3 构造的 3000 对进行三项检查。
用法: python validate_A3_data.py"""
import torch, numpy as np
from collections import defaultdict, Counter
from scipy.stats import pearsonr
from tqdm import tqdm

from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto
from rdkit import Chem

engine = ChemicalRuleEngine(tolerance=0.02)
import dreams.utils.dformats as dformats; import dreams.utils.data as du
dformat = dformats.DataFormatA()
spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

MSP_FILES = ['data/MassBank_NIST.msp', 'data/MoNA-export-LC-MS-MS_Spectra.msp',
             'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']

spectra = []
for fp in MSP_FILES:
    s = parse_msp(fp, max_spectra=20000); spectra.extend(s)

valid = []
for s in spectra:
    smi = s.get('SMILES','').strip(); ik = s.get('InChIKey','').strip()
    if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
        fm = Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi))
        s['_formula'] = fm; valid.append(s)

match_vecs = {}
for i,s in enumerate(tqdm(valid, desc='Vectors')):
    vec = spectrum_to_match_vec(s, engine, spec_preproc)
    if vec is not None: match_vecs[i] = vec
vidx = [i for i in range(len(valid)) if i in match_vecs]

rng = np.random.RandomState(42)
ik2idx = defaultdict(list); fm2idx = defaultdict(list)
for i in vidx:
    ik2idx[valid[i]['InChIKey']].append(i)
    fm2idx[valid[i].get('_formula','')].append(i)
multi_ik = {k:v for k,v in ik2idx.items() if len(v)>=2}
multi_fm = {k:v for k,v in fm2idx.items() if len(v)>=2}

pairs, labels, pair_mtypes = [], [], []
n_pos = 0
for ik in sorted(multi_ik.keys(), key=lambda x: -len(multi_ik[x])):
    a,b = rng.choice(multi_ik[ik],2,replace=False)
    pairs.append((a,b)); labels.append(1.0); pair_mtypes.append('pos')
    n_pos += 1
    if n_pos >= 900: break

iso_count = 0; iso_formulas = Counter(); iso_tans = []
for fm in sorted(multi_fm.keys(), key=lambda x: -len(multi_fm[x])):
    idxs = multi_fm[fm]
    if len(idxs)<2: continue
    seen = set()
    for _ in range(min(50,len(idxs)*3)):
        a,b = rng.choice(idxs,2,replace=False)
        if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
        pk = (min(a,b),max(a,b))
        if pk in seen: continue; seen.add(pk)
        tan = compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
        if 0.3<=tan<=0.9:
            pairs.append((a,b)); labels.append(tan); pair_mtypes.append('isomer')
            iso_count += 1; iso_formulas[fm] += 1; iso_tans.append(tan)
        if iso_count >= 900: break
    if iso_count >= 900: break

for _ in range(5000):
    a,b = rng.choice(vidx,2,replace=False)
    if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
    pm_a=float(valid[a].get('PrecursorMZ',0) or 0); pm_b=float(valid[b].get('PrecursorMZ',0) or 0)
    if abs(pm_a-pm_b)<=1.0: continue
    tan = compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
    if 0<=tan<0.2: pairs.append((a,b)); labels.append(tan); pair_mtypes.append('easy')
    if len(pairs)>=3000: break

labels=np.array(labels,dtype=np.float32); pair_mtypes=np.array(pair_mtypes)
iso_tans=np.array(iso_tans)

sep = '='*60
# V1
print(f'\n{sep}\n  V1: Isomer pair diversity\n{sep}')
print(f'  Total isomer pairs: {iso_count}')
print(f'  Tanimoto: min={iso_tans.min():.4f} max={iso_tans.max():.4f} mean={iso_tans.mean():.4f} std={iso_tans.std():.4f}')
print(f'  [0.3-0.5): {(iso_tans<0.5).sum()}  [0.5-0.7): {((iso_tans>=0.5)&(iso_tans<0.7)).sum()}  [0.7-0.9): {(iso_tans>=0.7).sum()}')
print(f'  Unique formulas: {len(iso_formulas)}')
print(f'  Pairs/formula: mean={iso_count/len(iso_formulas):.1f}, max={max(iso_formulas.values())}')
top5 = iso_formulas.most_common(5)
for fm,n in top5: print(f'    {fm}: {n} ({n/iso_count*100:.1f}%)')

# V2
print(f'\n{sep}\n  V2: Positive sample Jaccard consistency\n{sep}')
pos_idx = [i for i,mt in enumerate(pair_mtypes) if mt=='pos'][:100]
ovs = []
for pi in pos_idx:
    a,b = pairs[pi]; va,vb = match_vecs[a], match_vecs[b]
    inter=(va*vb).sum().float(); union=((va+vb)>0).float().sum()
    ovs.append((inter/union.clamp(min=1)).item())
ovs=np.array(ovs)
print(f'  Jaccard: mean={ovs.mean():.4f} std={ovs.std():.4f} min={ovs.min():.4f} max={ovs.max():.4f}')
print(f'  Status: {"OK" if ovs.mean()>0.3 else "LOW — fragmentation inconsistent across conditions"}')

# V3
print(f'\n{sep}\n  V3: Signal strength by pair type\n{sep}')
iso_idx = [i for i,mt in enumerate(pair_mtypes) if mt=='isomer']
easy_idx = [i for i,mt in enumerate(pair_mtypes) if mt=='easy']
ov_iso,tan_iso=[],[]
for pi in iso_idx: a,b=pairs[pi]; va,vb=match_vecs[a],match_vecs[b]; inter=(va*vb).sum().float(); union=((va+vb)>0).float().sum(); ov_iso.append((inter/union.clamp(min=1)).item()); tan_iso.append(labels[pi])
ov_easy,tan_easy=[],[]
for pi in easy_idx: a,b=pairs[pi]; va,vb=match_vecs[a],match_vecs[b]; inter=(va*vb).sum().float(); union=((va+vb)>0).float().sum(); ov_easy.append((inter/union.clamp(min=1)).item()); tan_easy.append(labels[pi])
r_iso,_=pearsonr(ov_iso,tan_iso); r_easy,_=pearsonr(ov_easy,tan_easy)
print(f'  Isomer subset: r(overlap,T) = {r_iso:.4f} (n={len(iso_idx)})')
print(f'  Easy neg subset: r(overlap,T) = {r_easy:.4f} (n={len(easy_idx)})')
print(f'  Verdict: {"Isomers MORE informative" if r_iso>r_easy else "Easy neg similar or better"}')
print(f'\n{sep}\n  ALL VALIDATIONS COMPLETE\n{sep}')
