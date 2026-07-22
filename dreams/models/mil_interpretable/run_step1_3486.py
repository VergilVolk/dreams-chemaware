"""
run_step1_3486.py — Step 1: 数据修复验证

用 3486 规则引擎重建数据，旧 MIL 配置（不做任何参数修改），对比 LR-agg。

用法：
  python -m dreams.models.mil_interpretable.run_step1_3486
"""

import torch, torch.nn.functional as F
import numpy as np
import pickle
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from tqdm import tqdm
from pathlib import Path

from dreams.models.mil_interpretable.mil_model import RuleAttentionMIL
from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec
from dreams.models.mil_interpretable.build_balanced_data import compute_tanimoto


def main():
    print('=' * 60)
    print('Step 1: Data Fix Verification (3486 rules, old MIL config)')
    print('=' * 60)

    engine = ChemicalRuleEngine(tolerance=0.02)
    print(f'Engine: {len(engine.rules)} rules')

    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    # ===== 1. Parse MSP files =====
    print('\n[1] Parsing MSP files...')
    spectra = []
    for fpath in ['data/MassBank_NIST.msp',
                  'data/MoNA-export-LC-MS-MS_Spectra.msp',
                  'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']:
        name = fpath.replace('\\','/').split('/')[-1]
        s = parse_msp(fpath, max_spectra=20000)
        print(f'   {name}: {len(s)}')
        spectra.extend(s)
    print(f'   Total: {len(spectra)}')

    # ===== 2. Filter =====
    print('\n[2] Filtering...')
    from rdkit import Chem
    valid = []
    for s in spectra:
        smi = s.get('SMILES','').strip()
        ik = s.get('InChIKey','').strip()
        if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
            valid.append(s)
    print(f'   Valid: {len(valid)}')

    # ===== 3. Match vectors =====
    print('\n[3] Computing rule match vectors...')
    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None:
            match_vecs[i] = vec
    valid_idx = list(match_vecs.keys())
    print(f'   {len(valid_idx)} with rule vectors')

    # ===== 4. Build balanced pairs =====
    print('\n[4] Building balanced pairs (3000, Tanimoto 0-1)...')
    rng = np.random.RandomState(42)

    # Group
    ik_to_idx = defaultdict(list)
    for i in valid_idx:
        ik = valid[i]['InChIKey']
        ik_to_idx[ik].append(i)
    multi_ik = {k:v for k,v in ik_to_idx.items() if len(v)>=2}

    pairs, labels = [], []

    # Same molecule: 750
    ik_list = list(multi_ik.keys()); rng.shuffle(ik_list)
    for ik in ik_list:
        idxs = multi_ik[ik]
        a, b = rng.choice(idxs, 2, replace=False)
        pairs.append((a,b)); labels.append(1.0)
        if len(pairs) >= 750: break

    # Random diverse: 2250
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    for _ in range(10000):
        a, b = rng.choice(valid_idx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        tan = compute_tanimoto(valid[a]['SMILES'], valid[b]['SMILES'])
        if tan < 0: continue
        pairs.append((a,b)); labels.append(tan)
        if len(pairs) >= 3000: break

    labels = np.array(labels, dtype=np.float32)
    print(f'   {len(pairs)} pairs, Tanimoto mean={labels.mean():.4f} std={labels.std():.4f}')

    # ===== 5. Build features =====
    print('\n[5] Building features...')
    X_agg, instances_list = [], []
    for a, b in tqdm(pairs, desc='Features'):
        va, vb = match_vecs[a], match_vecs[b]
        inter = (va*vb).sum().float()
        union = ((va+vb)>0).float().sum()
        ov = (inter/union.clamp(min=1)).item()
        common = (va*vb)>0; nc = common.sum().item()
        nl = common[:293].sum().item()
        cf = common[293:293+3174].sum().item()
        iso_hr = common[293+3174:].sum().item()
        X_agg.append([ov, float(nc), iso_hr/max(nc,1), nl/max(nc,1), cf/max(nc,1)])

        inst_feats = []
        for idx in range(len(common)):
            if common[idx].item():
                rule = engine.rules[idx]
                level = 1
                if rule.category == 'HR': level = 2
                elif rule.category in ('NR','EE'): level = 0
                elif rule.category == 'ISO': level = 2
                inst_feats.append(build_instance_features({
                    'level':level, 'category':rule.category,
                    'match_type':rule.match_type, 'mass_diff_precision':0.5}))
        instances_list.append(torch.tensor(np.stack(inst_feats),dtype=torch.float32) if inst_feats else torch.zeros(0,12))

    X_agg = np.array(X_agg, dtype=np.float32)

    # ===== 6. Molecule-level 5-fold CV =====
    print('\n[6] 5-fold CV...')
    pair_mols = []
    for a,b in pairs:
        ms = set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams)//5

    lr_rs, mil_rs = [], []
    for k in range(5):
        vs,ve = k*mpf, (k+1)*mpf if k<4 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs])|set(ams[ve:])
        tr = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        assert not (set().union(*[pair_mols[p] for p in tr] if tr else []) &
                    set().union(*[pair_mols[p] for p in va] if va else [])), f'Fold {k} leak!'

        # LR-agg
        lr = Ridge(alpha=1.0); lr.fit(X_agg[tr], labels[tr])
        yp = lr.predict(X_agg[va])
        lr_r, _ = pearsonr(labels[va], yp); lr_rs.append(max(lr_r,0))

        # MIL — OLD config: hidden_dim=32, lr=1e-4, dropout=0.1, wd=1e-5, entropy=0.001
        model = RuleAttentionMIL(instance_dim=12, hidden_dim=32)
        # Override dropout
        model.feature_extractor[2].p = 0.1
        model.attn_dropout.p = 0.1
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        best_r, best_st, counter = 0, None, 0

        for ep in range(500):
            model.train(); batch_loss, bn = None, 0
            for pi in tr:
                bag = instances_list[pi]
                if bag.shape[0]==0: continue
                pred, attn = model(bag)
                loss = F.mse_loss(pred, torch.tensor(labels[pi],dtype=torch.float32).unsqueeze(0))
                if len(attn)>1:
                    ac = attn.clamp(min=1e-8)
                    loss = loss + 0.001 * (-(ac*torch.log(ac)).sum()/attn.size(0))
                batch_loss = loss if batch_loss is None else batch_loss+loss; bn+=1
                if bn%8==0: batch_loss.backward(); opt.step(); opt.zero_grad(); batch_loss=None
            if batch_loss is not None: batch_loss.backward(); opt.step(); opt.zero_grad()

            if ep%20==0:
                model.eval()
                with torch.no_grad():
                    pr,lb=[],[]
                    for pi in va:
                        bag=instances_list[pi]
                        pr.append(model(bag)[0].item() if bag.shape[0]>0 else 0)
                        lb.append(labels[pi])
                r,_=pearsonr(pr,lb); r=max(r,0)
                if r>best_r: best_r=r; best_st={k:v.clone() for k,v in model.state_dict().items()}; counter=0
                else: counter+=1
                if counter>=30: break

        if best_st: model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            pr,lb=[],[]
            for pi in va:
                bag=instances_list[pi]
                pr.append(model(bag)[0].item() if bag.shape[0]>0 else 0)
                lb.append(labels[pi])
        r,_=pearsonr(pr,lb); mil_rs.append(max(r,0))
        print(f'   Fold {k+1}: LR-agg r={lr_rs[-1]:.4f}  MIL r={mil_rs[-1]:.4f}')

    sep = '='*60
    print(f'\n{sep}')
    print(f'RESULTS (3486 rules, OLD MIL config)')
    print(f'{sep}')
    print(f'  LR-agg:        r = {np.mean(lr_rs):.4f} +/- {np.std(lr_rs):.4f}')
    print(f'  MIL (old cfg): r = {np.mean(mil_rs):.4f} +/- {np.std(mil_rs):.4f}')
    if np.mean(mil_rs) > np.mean(lr_rs):
        print(f'  >>> MIL BEATS LR-agg! Problem solved.')
    else:
        print(f'  >>> MIL still below LR-agg. Need parameter tuning.')


if __name__ == '__main__':
    main()
