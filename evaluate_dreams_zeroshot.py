"""
DreaMS 零样本评估 — 用我们自己的 annotated01.mgf 超大谱库

输出:
  1. ROC AUC 曲线 (T0: 同分子 vs 不同分子)
  2. Cosine Similarity vs Tanimoto 相关性散点图 (DreaMS 论文 Figure 4a 风格)

用法 (在 dreams_env):
  python evaluate_dreams_zeroshot.py --n_spectra 5000 --n_pairs 20000
"""
import torch, numpy as np, json, os, sys, argparse
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import pearsonr

from argparse import Namespace
import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS
from dreams.definitions import PRETRAINED


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_spectra', type=int, default=5000)
    p.add_argument('--n_pairs', type=int, default=20000)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_model(ckpt_path, device):
    """加载 DreaMS 预训练模型 (与 evaluate_retrieval.py 同逻辑)"""
    pkg = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    recon_args = Namespace(**pkg['args'])
    recon_args.dformat = dformats.DataFormatA()
    for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
               'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
               'min_intensity_ampl', 'max_ms_level']:
        if da in pkg['args']:
            setattr(recon_args.dformat, da, pkg['args'][da])
    recon_args.chem_attn = False

    sp = du.SpectrumPreprocessor(dformat=recon_args.dformat,
                                 n_highest_peaks=recon_args.max_peaks_n)
    model = DreaMS(recon_args, sp)
    state = model.state_dict()
    for k in state:
        if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
            state[k] = pkg['state_dict'][k].clone()
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, sp


def load_from_annotated01(model, n_spectra, spec_preproc, device, seed=42):
    """从 annotated01.mgf 加载谱图 + SMILES, 提取 DreaMS 嵌入"""
    # Pass 1: scan IK → SMILES mapping
    print('  Scanning annotated01.mgf for IK→SMILES...')
    ik_to_smi = {}
    cur_ik = None; cur_smi = None
    with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_ik and cur_smi and cur_ik not in ik_to_smi:
                    ik_to_smi[cur_ik] = cur_smi
                cur_ik = None; cur_smi = None; continue
            if line.startswith('SMILES='): cur_smi = line[7:].strip()
            elif line.startswith('INCHIKEY='): cur_ik = line[9:].strip()[:14]

    # Sample IKs
    all_iks = sorted(ik_to_smi.keys())
    rng = np.random.RandomState(seed)
    sampled = rng.choice(all_iks, min(n_spectra, len(all_iks)), replace=False)
    sampled_set = set(sampled)
    print(f'  Sampled {len(sampled)} IKs (from {len(all_iks)} total)')

    # Pass 2: load spectra for sampled IKs
    print('  Loading spectra...')
    ik_to_peaks = {}
    cur_ik = None; cur_peaks = []
    with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_ik and cur_ik in sampled_set and len(cur_peaks) >= 3:
                    if cur_ik not in ik_to_peaks:
                        ik_to_peaks[cur_ik] = cur_peaks[:]
                cur_ik = None; cur_peaks = []; continue
            if line.startswith('INCHIKEY='): cur_ik = line[9:].strip()[:14]
            elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
                p2 = line.split()
                if len(p2) >= 2:
                    try:
                        mz, i = float(p2[0]), float(p2[1])
                        if mz > 0 and i > 0: cur_peaks.append((mz, i))
                    except: pass

    # Extract embeddings
    print('  Extracting DreaMS embeddings...')
    embeddings = []
    smi_list = []
    ik_list = []

    for ik in tqdm(sampled, desc='Embedding'):
        if ik not in ik_to_peaks: continue
        smi = ik_to_smi.get(ik, '')
        if not smi: continue

        arr = np.array(ik_to_peaks[ik], dtype=np.float32)
        arr = arr[arr[:, 0].argsort()]
        try:
            spec_pp = spec_preproc(arr.T, high_form=False)
        except Exception:
            continue
        spec_t = torch.as_tensor(spec_pp, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            emb = model(spec_t, None)

        embeddings.append(emb[:, 0, :].cpu())
        smi_list.append(smi)
        ik_list.append(ik)

    embeddings = torch.cat(embeddings, dim=0)
    print(f'  Got {len(embeddings)} embeddings, {len(set(smi_list))} unique SMILES')
    return embeddings, smi_list, ik_list


def compute_tanimoto(smi_a, smi_b):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    ma = Chem.MolFromSmiles(smi_a); mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None: return None
    fpa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fpb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    return float(DataStructs.TanimotoSimilarity(fpa, fpb))


def build_eval_pairs(embeddings, smi_list, n_pairs, seed=42):
    """构建平衡评估对: 同分子=pos, 不同分子=neg"""
    mol_to_idx = defaultdict(list)
    for i, smi in enumerate(smi_list):
        mol_to_idx[smi].append(i)

    multi = {k: v for k, v in mol_to_idx.items() if len(v) >= 2}
    all_mols = list(mol_to_idx.keys())
    rng = np.random.RandomState(seed)

    n_pos = 0; n_neg = 0
    target_pos = n_pairs // 2; target_neg = n_pairs - target_pos
    pair_i, pair_j, pair_labels = [], [], []
    tani_scores = []

    # Pos
    if multi:
        mol_list = list(multi.keys())
        while n_pos < target_pos:
            mol = mol_list[rng.randint(0, len(mol_list))]
            idxs = multi[mol]
            a, b = rng.choice(idxs, 2, replace=False)
            pair_i.append(a); pair_j.append(b); pair_labels.append(1)
            tani_scores.append(1.0)  # same mol → Tanimoto=1
            n_pos += 1

    # Neg
    while n_neg < target_neg:
        m1, m2 = rng.choice(all_mols, 2, replace=False)
        if m1 == m2: continue
        a = rng.choice(mol_to_idx[m1])
        b = rng.choice(mol_to_idx[m2])
        tan = compute_tanimoto(smi_list[a], smi_list[b])
        if tan is None: continue
        pair_i.append(a); pair_j.append(b); pair_labels.append(0)
        tani_scores.append(tan)
        n_neg += 1

    return (np.array(pair_i), np.array(pair_j), np.array(pair_labels), np.array(tani_scores))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt = args.ckpt or str(PRETRAINED / 'ssl_model_server.pt')
    print(f'Device: {device}\nCheckpoint: {ckpt}')

    # 1. Load model
    print('\n[1] Loading DreaMS...')
    model, spec_preproc = load_model(ckpt, device)

    # 2. Load T0/T3 pairs
    print('\n[2] Loading T0/T3 pairs...')
    with open('tasks/T0_consistency/test_cases/pairs.json') as f:
        t0 = json.load(f)
    with open('tasks/T3_unrelated/test_cases/pairs.json') as f:
        t3 = json.load(f)

    rng = np.random.RandomState(42)
    n_pairs = args.n_pairs

    # Sample T0 positive pairs (same IK, different spectra)
    t0_pos_sample = rng.choice(t0['positive'], min(n_pairs//2, len(t0['positive'])), replace=False)
    # Sample T3 negative pairs (different IK, different formula)
    t3_neg_sample = rng.choice(t3['negative'], min(n_pairs//2, len(t3['negative'])), replace=False)

    # 3. Collect needed IKs → SMILES mapping
    needed_iks = set()
    for p in t0_pos_sample:
        needed_iks.add(p['ik'][:14])
    for p in t3_neg_sample:
        needed_iks.add(p['ik_a'][:14])
        needed_iks.add(p['ik_b'][:14])

    # Scan annotated01: IK→SMILES + collect ALL spectra per IK
    print(f'  {len(needed_iks)} unique IKs needed')
    print('  Scanning annotated01.mgf for spectra + SMILES...')
    ik_to_smi = {}
    ik_to_all_peaks = defaultdict(list)
    cur_ik = None; cur_smi = None; cur_peaks = []
    with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_ik and cur_ik in needed_iks and len(cur_peaks) >= 3:
                    ik_to_all_peaks[cur_ik].append(cur_peaks[:])
                    if cur_smi and cur_ik not in ik_to_smi:
                        ik_to_smi[cur_ik] = cur_smi
                cur_ik = None; cur_smi = None; cur_peaks = []; continue
            if line.startswith('SMILES='): cur_smi = line[7:].strip()
            elif line.startswith('INCHIKEY='): cur_ik = line[9:].strip()[:14]
            elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
                p2 = line.split()
                if len(p2) >= 2:
                    try:
                        mz, i = float(p2[0]), float(p2[1])
                        if mz > 0 and i > 0: cur_peaks.append((mz, i))
                    except: pass

    multi_spec = {ik: pks for ik, pks in ik_to_all_peaks.items() if len(pks) >= 2}
    print(f'  {len(ik_to_smi)} IKs with SMILES, {len(multi_spec)} with >=2 spectra')

    # 4. Extract embeddings in batches
    print(f'\n[3] Extracting DreaMS embeddings (batch_size=16)...')
    spec_records = []  # [(ik, smi, embedding), ...]
    total_specs = 0

    # Collect all spectra into a flat list
    all_specs_flat = []  # [(ik, smi, peaks), ...]
    for ik, peaks_list in ik_to_all_peaks.items():
        smi = ik_to_smi.get(ik, '')
        if not smi: continue
        for peaks in peaks_list:
            all_specs_flat.append((ik, smi, peaks))

    # Batch process
    batch_size = 16
    for start in tqdm(range(0, len(all_specs_flat), batch_size), desc='Embedding'):
        batch = all_specs_flat[start:start+batch_size]
        batch_tensors = []
        for ik, smi, peaks in batch:
            arr = np.array(peaks, dtype=np.float32)
            arr = arr[arr[:, 0].argsort()]
            try:
                spec_pp = spec_preproc(arr.T, high_form=False)
            except Exception: continue
            batch_tensors.append(torch.as_tensor(spec_pp, dtype=torch.float32))
            spec_records.append({'ik': ik, 'smi': smi, 'emb': None, '_idx': len(spec_records)})

        if not batch_tensors: continue
        spec_batch = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            embs = model(spec_batch, None)
        for j, (ik, smi, peaks) in enumerate(batch):
            if j < len(embs):
                spec_records[total_specs + j]['emb'] = embs[j, 0, :].cpu()
        total_specs += len(batch_tensors)

    # Remove records without embeddings
    spec_records = [r for r in spec_records if r['emb'] is not None]
    total_specs = len(spec_records)
    print(f'  {total_specs} embeddings from {len(ik_to_all_peaks)} IKs')

    # Build IK→indices mapping
    ik_to_indices = defaultdict(list)
    for i, r in enumerate(spec_records):
        ik_to_indices[r['ik']].append(i)

    # 5. Build evaluation pairs with Tanimoto
    print(f'\n[4] Building evaluation pairs...')
    pair_i, pair_j, pair_labels, tani_scores = [], [], [], []
    n_pos, n_neg = 0, 0

    # T0 positive: different spectra of same IK
    target_pos = n_pairs // 2
    for p in t0_pos_sample:
        ik = p['ik'][:14]
        indices = ik_to_indices.get(ik, [])
        if len(indices) < 2: continue
        if n_pos >= target_pos: break
        a, b = rng.choice(indices, 2, replace=False)
        pair_i.append(a); pair_j.append(b); pair_labels.append(1)
        tani_scores.append(1.0)  # same molecule
        n_pos += 1

    # T3 negative: different IK → compute Tanimoto
    target_neg = n_pairs - n_pos
    for p in t3_neg_sample:
        ik_a = p['ik_a'][:14]; ik_b = p['ik_b'][:14]
        idx_a = ik_to_indices.get(ik_a, [])
        idx_b = ik_to_indices.get(ik_b, [])
        if not idx_a or not idx_b: continue
        if n_neg >= target_neg: break
        a = rng.choice(idx_a); b = rng.choice(idx_b)
        smi_a = spec_records[a]['smi']; smi_b = spec_records[b]['smi']
        tan = compute_tanimoto(smi_a, smi_b)
        if tan is None: continue
        pair_i.append(a); pair_j.append(b); pair_labels.append(0)
        tani_scores.append(tan)
        n_neg += 1

    print(f'  Built: {n_pos} pos + {n_neg} neg = {n_pos+n_neg} pairs')

    # 6. Cosine similarity
    print('\n[5] Computing cosine similarities...')
    cos_sims = []
    for a, b in zip(pair_i, pair_j):
        ea = spec_records[a]['emb']; eb = spec_records[b]['emb']
        cos = float(torch.nn.functional.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0), dim=-1))
        cos_sims.append(cos)

    cos_sims = np.array(cos_sims)
    pair_labels = np.array(pair_labels)
    tani_scores = np.array(tani_scores)

    # 7. Metrics
    fpr, tpr, _ = metrics.roc_curve(pair_labels, cos_sims)
    auc = metrics.auc(fpr, tpr)
    r, p_val = pearsonr(cos_sims, tani_scores)

    print(f'\n{"=" * 60}')
    print(f'RESULTS (DreaMS Zero-Shot on annotated01)')
    print(f'{"=" * 60}')
    print(f'  Spectra:    {total_specs}')
    print(f'  Pairs:      {len(pair_labels)} ({n_pos}P + {n_neg}N)')
    print(f'  AUC:        {auc:.4f}')
    print(f'  Pearson r:  {r:.4f} (p={p_val:.2e})')
    print(f'  Cos+ mean:  {cos_sims[pair_labels==1].mean():.4f}')
    print(f'  Cos- mean:  {cos_sims[pair_labels==0].mean():.4f}')
    print(f'  Separation: {cos_sims[pair_labels==1].mean() - cos_sims[pair_labels==0].mean():.4f}')
    print(f'{"=" * 60}')

    # 6. Plot
    print('\n[6] Plotting...')
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('DreaMS Zero-Shot Evaluation on annotated01 (87K molecules, 3.2M spectra)',
                 fontsize=14, fontweight='bold')

    # (a) ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color='#2ecc71', linewidth=2.5, label=f'DreaMS (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2ecc71')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'(a) ROC Curve — Same vs Different Molecule\nAUC = {auc:.4f}')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # (b) Cosine similarity distribution
    ax = axes[1]
    bins = np.linspace(-1, 1, 60)
    ax.hist(cos_sims[pair_labels == 1], bins=bins, alpha=0.5, color='#2ecc71',
            label=f'Same mol (n={n_pos}, μ={cos_sims[pair_labels==1].mean():.3f})')
    ax.hist(cos_sims[pair_labels == 0], bins=bins, alpha=0.5, color='#e74c3c',
            label=f'Diff mol (n={n_neg}, μ={cos_sims[pair_labels==0].mean():.3f})')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(b) Embedding Similarity Distribution')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) Cosine Similarity vs Tanimoto (DreaMS Figure 4a style)
    ax = axes[2]
    # Only use neg pairs for correlation (pos pairs all have tani=1.0)
    neg_mask = pair_labels == 0
    ax.scatter(tani_scores[neg_mask], cos_sims[neg_mask], alpha=0.15, s=8,
               color='#3498db', edgecolors='none', rasterized=True)
    # Add bin statistics
    bin_edges = np.linspace(0, 1, 21)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = [cos_sims[neg_mask][(tani_scores[neg_mask] >= lo) & (tani_scores[neg_mask] < hi)].mean()
                 if np.any((tani_scores[neg_mask] >= lo) & (tani_scores[neg_mask] < hi)) else np.nan
                 for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], np.array(bin_means)[valid], 'o-', color='#e74c3c',
            linewidth=2.5, markersize=7, label=f'Binned mean (r={r:.3f})')
    ax.set_xlabel('Morgan Fingerprint Tanimoto Similarity', fontsize=12)
    ax.set_ylabel('DreaMS Embedding Cosine Similarity', fontsize=12)
    ax.set_title(f'(c) Cosine Similarity vs Tanimoto\nPearson r = {r:.4f} (p={p_val:.1e})')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'dreams_zeroshot_annotated01.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {out_path}')

    # Save results
    results = {
        'n_spectra': total_specs,
        'n_pairs': int(len(pair_labels)),
        'n_pos': int(n_pos), 'n_neg': int(n_neg),
        'auc': float(auc), 'pearson_r': float(r), 'p_value': float(p_val),
        'cos_pos_mean': float(cos_sims[pair_labels == 1].mean()),
        'cos_neg_mean': float(cos_sims[pair_labels == 0].mean()),
    }
    with open('dreams_zeroshot_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: dreams_zeroshot_results.json')


if __name__ == '__main__':
    main()
