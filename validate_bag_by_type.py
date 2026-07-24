"""快速验证：分类型检查bag大小与Tanimoto的共线性 + Godden检查"""
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
sp = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

MSP_FILES = ['data/MassBank_NIST.msp', 'data/MoNA-export-LC-MS-MS_Spectra.msp',
             'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']
spectra = []
for fp in MSP_FILES:
    s = parse_msp(fp, max_spectra=30000); spectra.extend(s)

valid = []
for s in spectra:
    smi = s.get('SMILES','').strip(); ik = s.get('InChIKey','').strip()
    if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
        fm = Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smi))
        s['_formula'] = fm; valid.append(s)

mvs = {}
for i,s in enumerate(tqdm(valid, desc='Vectors')):
    vec = spectrum_to_match_vec(s, engine, sp)
    if vec is not None: mvs[i] = vec
vidx = [i for i in range(len(valid)) if i in mvs]

rng = np.random.RandomState(42)
ik2 = defaultdict(list); fm2 = defaultdict(list)
for i in vidx:
    ik2[valid[i]['InChIKey']].append(i)
    fm2[valid[i].get('_formula','')].append(i)
multi_ik = {k:v for k,v in ik2.items() if len(v)>=2}
multi_fm = {k:v for k,v in fm2.items() if len(v)>=2}

pairs,labels,types=[],[],[]
for ik in sorted(multi_ik.keys(),key=lambda x:-len(multi_ik[x])):
    a,b=rng.choice(multi_ik[ik],2,replace=False)
    pairs.append((a,b)); labels.append(1.0); types.append('same_mol')
    if len(pairs)>=300: break
for fm in sorted(multi_fm.keys(),key=lambda x:-len(multi_fm[x])):
    idxs=multi_fm[fm];
    if len(idxs)<2: continue
    seen=set()
    for _ in range(min(50,len(idxs)*3)):
        a,b=rng.choice(idxs,2,replace=False)
        if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
        pk=(min(a,b),max(a,b))
        if pk in seen: continue; seen.add(pk)
        tan=compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
        if 0.3<=tan<=0.9: pairs.append((a,b)); labels.append(tan); types.append('isomer')
        if len(types)-300>=600: break
    if len(types)-300>=600: break
for _ in range(5000):
    a,b=rng.choice(vidx,2,replace=False)
    if valid[a]['InChIKey']==valid[b]['InChIKey']: continue
    pm_a=float(valid[a].get('PrecursorMZ',0) or 0); pm_b=float(valid[b].get('PrecursorMZ',0) or 0)
    if abs(pm_a-pm_b)<=1.0: continue
    tan=compute_tanimoto(valid[a]['SMILES'],valid[b]['SMILES'])
    if 0<=tan<0.2: pairs.append((a,b)); labels.append(tan); types.append('random')
    if len(pairs)>=1200: break

sep='='*60
print(f'\n{sep}\n  BAG-SIZE vs TANIMOTO — BY PAIR TYPE\n{sep}')
for pt in ['same_mol','isomer','random']:
    idx=[i for i,t in enumerate(types) if t==pt]
    bsz=[((mvs[pairs[i][0]]*mvs[pairs[i][1]])>0).sum().item() for i in idx]
    tans=[labels[i] for i in idx]
    r,p=pearsonr(bsz,tans)
    print(f'  {pt:12s}: n={len(idx):4d}  r(bag_size,T)={r:.4f}  p={p:.2e}  mean_bsz={np.mean(bsz):.1f}')

iso_idx=[i for i,t in enumerate(types) if t=='isomer']
iso_ovs=[]
for i in iso_idx:
    a,b=pairs[i]; va,vb=mvs[a],mvs[b]
    inter=(va*vb).sum().float(); union=((va+vb)>0).float().sum()
    iso_ovs.append((inter/union.clamp(min=1)).item())
iso_tans=[labels[i] for i in iso_idx]
r_ov,_=pearsonr(iso_ovs,iso_tans)
print(f'\n  Isomer detail: r(overlap,T)={r_ov:.4f}, unique formulas={len(set(valid[pairs[i][0]]["_formula"] for i in iso_idx))}')

rnd_idx=[i for i,t in enumerate(types) if t=='random']
rnd_tans=np.array([labels[i] for i in rnd_idx])
in_01_03=((rnd_tans>=0.1)&(rnd_tans<0.3)).mean()
print(f'\n  Godden check: random pairs with T in [0.1,0.3): {in_01_03*100:.1f}%')
print(f'  This is a mathematical baseline, NOT functional group overlap.')
