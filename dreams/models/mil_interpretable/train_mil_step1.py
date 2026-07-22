"""
train_mil_step1.py — MIL Step 1: 化学一致性独立验证

任务: 判断两张谱图的碎裂模式是否来自同一个分子
正样本: 同一 InChIKey 的不同谱图 → MIL 应输出高化学一致性分数
负样本: 不同 InChIKey 的谱图对 → MIL 应输出低化学一致性分数
训练: 对比损失（MSE: score→1 for pos, score→0 for neg）
评估: 测试集区分 AUC + 注意力权重 L2 vs L0

用法:
  python -m dreams.models.mil_interpretable.train_mil_step1
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from dreams.models.mil_interpretable.build_data import build_instance_features
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.mil_interpretable.train_mil_massbank import parse_msp, spectrum_to_match_vec


class ChemicalConsistencyMIL(nn.Module):
    """输出化学一致性分数 [0,1] + attention weights"""
    def __init__(self, instance_dim=12, hidden_dim=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(instance_dim, hidden_dim), nn.ReLU(),
        )
        self.attn_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid(),
        )
        self.no_evidence = nn.Parameter(torch.zeros(hidden_dim))

    def encode_bag(self, instances):
        if instances.shape[0] == 0:
            return self.no_evidence, torch.zeros(0)
        h = self.feature_extractor(instances)
        a_raw = self.attn_w(self.attn_V(h) * self.attn_U(h))
        a = torch.softmax(a_raw, dim=0)
        bag = (a * h).sum(dim=0)
        return bag, a.squeeze(-1)

    def forward(self, instances):
        bag, attn = self.encode_bag(instances)
        score = self.scorer(bag)  # [0,1] 化学一致性
        return score, attn


def main():
    print('=' * 60)
    print('MIL Step 1: Chemical Consistency Validation')
    print('=' * 60)

    engine = ChemicalRuleEngine(tolerance=0.02)
    import dreams.utils.dformats as dformats
    import dreams.utils.data as du
    dformat = dformats.DataFormatA()
    spec_preproc = du.SpectrumPreprocessor(dformat=dformat, n_highest_peaks=128)

    # ===== 1. 解析数据（MassBank + MoNA）=====
    print('\n[1] Parsing MSP files...')
    spectra = []
    msp_files = ['data/MassBank_NIST.msp',
                 'data/MoNA-export-LC-MS-MS_Spectra.msp',
                 'data/MoNA-export-LC-MS-MS_Negative_Mode.msp']
    for fpath in msp_files:
        s = parse_msp(fpath, max_spectra=40000)
        print(f'   {fpath}: {len(s)} spectra')
        spectra.extend(s)
    print(f'   Total: {len(spectra)} spectra')

    # ===== 2. 过滤 =====
    print('\n[2] Filtering valid spectra...')
    from rdkit import Chem
    valid = []
    for s in spectra:
        smi = s.get('SMILES','').strip()
        ik = s.get('InChIKey','').strip()
        if smi and ik and len(smi)>2 and Chem.MolFromSmiles(smi) is not None:
            valid.append(s)
    print(f'   Valid: {len(valid)}')

    ik_to_idx = defaultdict(list)
    for i, s in enumerate(valid):
        ik_to_idx[s['InChIKey']].append(i)
    multi_ik = {k:v for k,v in ik_to_idx.items() if len(v)>=2}
    print(f'   Multi-spectrum molecules: {len(multi_ik)}')

    # ===== 3. 规则向量 =====
    print('\n[3] Computing rule match vectors...')
    match_vecs = {}
    for i, s in enumerate(tqdm(valid, desc='Rule vectors')):
        vec = spectrum_to_match_vec(s, engine, spec_preproc)
        if vec is not None:
            match_vecs[i] = vec
    valid_idx = list(match_vecs.keys())
    print(f'   {len(valid_idx)} spectra with rule vectors')

    # ===== 4. 构造配对 =====
    print('\n[4] Building pairs...')
    rng = np.random.RandomState(42)

    # Positive: same InChIKey
    ik_list = list(multi_ik.keys()); rng.shuffle(ik_list)
    pos_pairs = []
    for ik in ik_list:
        idxs = [i for i in multi_ik[ik] if i in match_vecs]
        if len(idxs) >= 2:
            a, b = rng.choice(idxs, 2, replace=False)
            pos_pairs.append((a, b))
        if len(pos_pairs) >= 1000: break

    # Negative: different InChIKey (random)
    neg_pairs = []
    for _ in range(5000):
        a, b = rng.choice(valid_idx, 2, replace=False)
        if valid[a]['InChIKey'] == valid[b]['InChIKey']: continue
        neg_pairs.append((a, b))
        if len(neg_pairs) >= 1000: break

    print(f'   Positive (same mol):  {len(pos_pairs)}')
    print(f'   Negative (random diff): {len(neg_pairs)}')

    # ===== 5. 构造 bag =====
    print('\n[5] Building bag features...')
    def build_bag(a, b):
        va, vb = match_vecs[a], match_vecs[b]
        common = (va * vb) > 0
        inst_feats, levels = [], []
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
                levels.append(level)
        if inst_feats:
            return (torch.tensor(np.stack(inst_feats), dtype=torch.float32),
                    torch.tensor(levels, dtype=torch.long))
        return torch.zeros(0,12), torch.zeros(0,dtype=torch.long)

    all_pairs = pos_pairs + neg_pairs
    all_bags, all_levels = [], []
    for a,b in tqdm(all_pairs, desc='Bags'):
        bag, lvls = build_bag(a,b)
        all_bags.append(bag)
        all_levels.append(lvls)

    all_labels = np.array([1.0]*len(pos_pairs) + [0.0]*len(neg_pairs), dtype=np.float32)
    print(f'   Total: {len(all_bags)} bags')

    # ===== 6. 分子级切分 + 5折CV =====
    print('\n[6] Molecule-level 5-fold CV...')
    pair_mols = []
    for a,b in all_pairs:
        ms = set()
        if valid[a]['InChIKey']: ms.add(valid[a]['InChIKey'])
        if valid[b]['InChIKey']: ms.add(valid[b]['InChIKey'])
        pair_mols.append(ms)
    ams = list(set().union(*pair_mols)); rng.shuffle(ams)
    mpf = len(ams)//5

    aucs, l2_means, l0_means = [], [], []
    for k in range(5):
        vs,ve = k*mpf, (k+1)*mpf if k<4 else len(ams)
        vm = set(ams[vs:ve]); tm = set(ams[:vs])|set(ams[ve:])
        tr = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(tm)]
        va = [pi for pi,pm in enumerate(pair_mols) if pm and pm.issubset(vm)]
        # Assert clean split
        tr_m = set().union(*[pair_mols[p] for p in tr]) if tr else set()
        va_m = set().union(*[pair_mols[p] for p in va]) if va else set()
        assert len(tr_m & va_m) == 0, f'Fold {k} molecule leak!'

        # Train
        model = ChemicalConsistencyMIL(hidden_dim=64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_auc, best_st, counter = 0, None, 0

        for ep in range(500):  # 充分训练
            model.train(); batch_loss, bn = 0, 0
            for pi in tr:
                bag = all_bags[pi]
                if bag.shape[0] == 0: continue
                score, _ = model(bag)
                target = torch.tensor(all_labels[pi], dtype=torch.float32).unsqueeze(0)
                loss = F.mse_loss(score, target)
                batch_loss += loss; bn += 1
                if bn % 8 == 0:
                    batch_loss.backward(); opt.step(); opt.zero_grad(); batch_loss = 0
            if bn % 8:
                batch_loss.backward(); opt.step(); opt.zero_grad()

            if ep % 10 == 0 or ep == 299:
                model.eval()
                with torch.no_grad():
                    preds, trues = [], []
                    for pi in va:
                        bag = all_bags[pi]
                        preds.append(model(bag)[0].item() if bag.shape[0]>0 else 0.5)
                        trues.append(all_labels[pi])
                auc = roc_auc_score(trues, preds)
                if auc > best_auc:
                    best_auc = auc; best_st = {k:v.clone() for k,v in model.state_dict().items()}; counter=0
                else: counter+=1
                if counter >= 8: break  # patience=8 × 10epochs = 80 epochs
                if ep % 50 == 0:
                    print(f'     ep {ep:3d}: AUC={auc:.4f} best={best_auc:.4f}')

        # Final eval
        if best_st: model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            preds, trues, l2w, l0w = [], [], [], []
            for pi in va:
                bag = all_bags[pi]
                if bag.shape[0] == 0:
                    preds.append(0.5)
                else:
                    score, attn = model(bag)
                    preds.append(score.item())
                    lvls = all_levels[pi]
                    if len(lvls)>0 and len(attn)>0:
                        for j in range(len(attn)):
                            lv = lvls[j].item()
                            if lv==2: l2w.append(attn[j].item())
                            elif lv==0: l0w.append(attn[j].item())
                trues.append(all_labels[pi])
            auc = roc_auc_score(trues, preds)
            aucs.append(auc)
            l2_means.append(np.mean(l2w) if l2w else 0)
            l0_means.append(np.mean(l0w) if l0w else 0)
        print(f'   Fold {k+1}: AUC={auc:.4f}  L2_attn={l2_means[-1]:.4f}  L0_attn={l0_means[-1]:.4f}  ratio={l2_means[-1]/max(l0_means[-1],1e-8):.1f}x')

    print()
    print(f'STEP 1 RESULTS')
    print()
    print(f'  AUC (same vs random diff): {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')
    print(f'  L2 attention: {np.mean(l2_means):.4f}')
    print(f'  L0 attention: {np.mean(l0_means):.4f}')
    print(f'  L2/L0 ratio:  {np.mean(l2_means)/max(np.mean(l0_means),1e-8):.1f}x')
    print(f'')
    if np.mean(aucs) > 0.7:
        print(f'  >>> PASS: AUC={np.mean(aucs):.3f} > 0.7, MIL ready for Step 2')
    else:
        print(f'  >>> FAIL: AUC={np.mean(aucs):.3f} < 0.7, MIL needs improvement')


if __name__ == '__main__':
    main()
