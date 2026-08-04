"""
T1 Triplet 微调 v2 — 用预构建的 MCES triplet 做对比学习

Triplet 来源: tasks/T1_near_isomers/test_cases/triplets_*.json
  - Anchor + Positive: 同分子式, MCES [0,2] (近同分异构体)
  - Negative: 同分子式, MCES [6,10] (明确不同异构体)

损失: L = L_mask + α·L_triplet + β·L_preservation
按分子式分组 train/val split，无数据泄漏

用法 (在 dreams_env):
  python -m dreams.models.chem_aware.train_triplet_t1 \
      --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
      --epochs 10 --batch_size 32 --alpha 0.5 --beta 0.01
"""
import torch, torch.nn.functional as F, json, os, sys, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# 化学规则由 ChemAwareDreaMS 内部管理，无需单独导入


def parse_args():
    p = argparse.ArgumentParser(description='T1 MCES triplet fine-tuning')
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--alpha', type=float, default=0.05, help='Triplet loss weight (v5 verified: 0.05)')
    p.add_argument('--beta', type=float, default=0.02, help='Preservation loss weight (v5 verified: 0.02)')
    p.add_argument('--margin', type=float, default=0.2, help='Triplet margin (v5 verified: 0.2)')
    p.add_argument('--save_dir', type=str, default='./triplet_t1_checkpoints')
    p.add_argument('--n_peaks', type=int, default=128)
    p.add_argument('--val_every', type=int, default=1, help='Validate every N epochs')
    p.add_argument('--save_best_only', action='store_true', help='Only save best checkpoint')
    p.add_argument('--auc_pairs', type=int, default=2000, help='Number of AUC eval pairs (0=disable)')
    return p.parse_args()


def load_spectra_for_triplets(triplets, n_peaks=128):
    """Load only needed spectra from annotated01.mgf"""
    needed = set()
    for t in triplets:
        needed.add(t['anchor_ik'][:14])
        needed.add(t['pos_ik'][:14])
        needed.add(t['neg_ik'][:14])

    ik_to_peaks = {}
    cur_ik = None; cur_peaks = []
    with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(ik_to_peaks) >= len(needed): break
            line = line.strip()
            if not line:
                if cur_ik and cur_ik in needed and cur_ik not in ik_to_peaks and len(cur_peaks) >= 3:
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

    print(f'  Loaded {len(ik_to_peaks)}/{len(needed)} spectra')
    return ik_to_peaks


def spectrum_to_tensor(peaks, n_peaks=128):
    """Convert peak list to (n_peaks, 2) tensor"""
    arr = np.array(peaks, dtype=np.float32)
    if len(arr) == 0: return None
    arr = arr[arr[:, 0].argsort()]
    if len(arr) > n_peaks:
        idx = np.argpartition(arr[:, 1], -n_peaks)[-n_peaks:]
        arr = arr[idx]; arr = arr[arr[:, 0].argsort()]
    max_i = arr[:, 1].max()
    if max_i > 0: arr[:, 1] /= max_i
    padded = np.zeros((n_peaks, 2), dtype=np.float32)
    n = min(len(arr), n_peaks)
    padded[:n] = arr[:n]
    return torch.from_numpy(padded)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- 1. Load triplets ----
    print('[1] Loading triplets...')
    with open('tasks/T1_near_isomers/test_cases/triplets_train.json') as f:
        train_trip = json.load(f)
    with open('tasks/T1_near_isomers/test_cases/triplets_val.json') as f:
        val_trip = json.load(f)
    print(f'  Train: {len(train_trip)}  Val: {len(val_trip)}')

    # ---- 2. Load spectra ----
    print('[2] Loading spectra...')
    # Truncate all IKs to 14-char (matching annotated01.mgf format)
    def ik14(ik): return ik[:14] if len(ik) > 14 else ik
    # Truncate IKs in triplets
    for t in train_trip + val_trip:
        t['anchor_ik'] = ik14(t['anchor_ik'])
        t['pos_ik'] = ik14(t['pos_ik'])
        t['neg_ik'] = ik14(t['neg_ik'])

    all_trip = train_trip + val_trip
    ik_to_peaks = load_spectra_for_triplets(all_trip, args.n_peaks)

    # Pre-compute tensors
    print('  Pre-computing spectral tensors...')
    ik_to_spec = {}
    for ik, peaks in ik_to_peaks.items():
        st = spectrum_to_tensor(peaks, args.n_peaks)
        if st is not None: ik_to_spec[ik] = st
    print(f'  {len(ik_to_spec)} valid spectral tensors')

    # Filter triplets
    def filter_trip(triplets):
        return [t for t in triplets
                if t['anchor_ik'] in ik_to_spec and t['pos_ik'] in ik_to_spec
                and t['neg_ik'] in ik_to_spec]
    train_trip = filter_trip(train_trip)
    val_trip = filter_trip(val_trip)
    print(f'  After filtering: train={len(train_trip)} val={len(val_trip)}')

    if len(train_trip) == 0:
        print('ERROR: No valid training triplets! Check spectra loading.')
        return

    # ---- 2b. Build fixed AUC evaluation set (T0 pos + T3 neg) ----
    auc_eval = None
    if args.auc_pairs > 0:
        print(f'\n[2b] Building AUC eval set ({args.auc_pairs} pairs)...')
        rng_auc = np.random.RandomState(12345)
        with open('tasks/T0_consistency/test_cases/pairs.json') as f:
            t0 = json.load(f)
        with open('tasks/T3_unrelated/test_cases/pairs.json') as f:
            t3 = json.load(f)

        n_each = args.auc_pairs // 2
        t0_pos = rng_auc.choice(t0['positive'], min(n_each, len(t0['positive'])), replace=False)
        t3_neg = rng_auc.choice(t3['negative'], min(n_each, len(t3['negative'])), replace=False)

        # Collect needed IKs + load spectra
        auc_iks = set()
        for p in t0_pos: auc_iks.add(p['ik'][:14])
        for p in t3_neg:
            auc_iks.add(p['ik_a'][:14]); auc_iks.add(p['ik_b'][:14])

        # Build pairs: (ik_a, ik_b, label)
        auc_pairs = []
        for p in t0_pos:
            ik = p['ik'][:14]
            if ik in ik_to_spec: auc_pairs.append((ik, ik, 1))
        for p in t3_neg:
            ika = p['ik_a'][:14]; ikb = p['ik_b'][:14]
            if ika in ik_to_spec and ikb in ik_to_spec: auc_pairs.append((ika, ikb, 0))

        # Deduplicate IKs for AUC (unique spectra only)
        auc_spec_iks = sorted(set(ik for a, b, _ in auc_pairs for ik in (a, b)))
        auc_ik_to_idx = {ik: i for i, ik in enumerate(auc_spec_iks)}
        auc_pair_indices = [(auc_ik_to_idx[a], auc_ik_to_idx[b]) for a, b, _ in auc_pairs]
        auc_labels = np.array([l for _, _, l in auc_pairs])

        auc_eval = {
            'spec_iks': auc_spec_iks,   # ordered list of IKs
            'pair_i': np.array([p[0] for p in auc_pair_indices]),
            'pair_j': np.array([p[1] for p in auc_pair_indices]),
            'labels': auc_labels,
        }
        print(f'  AUC eval: {len(auc_pairs)} pairs, {len(auc_spec_iks)} unique IKs, '
              f'{auc_labels.sum():.0f}P + {len(auc_labels)-auc_labels.sum():.0f}N')

    # ---- 3. Load ChemAwareDreaMS (化学规则注入, 模块一核心) ----
    print(f'[3] Loading ChemAwareDreaMS from {args.ckpt_path}...')
    pkg = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

    if 'pytorch-lightning_version' in pkg:
        raise ValueError('Lightning checkpoint not supported for ChemAwareDreaMS. Use ssl_model_server.pt.')
    elif 'args' in pkg and 'state_dict' in pkg:
        from argparse import Namespace
        from dreams.utils.dformats import DataFormatA
        from dreams.utils.data import SpectrumPreprocessor
        from dreams.models.chem_aware.chem_aware_dreams import ChemAwareDreaMS

        state_dict = pkg['state_dict']
        recon_args = Namespace(**pkg['args'])
        recon_args.dformat = DataFormatA()
        for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
                   'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
                   'min_intensity_ampl', 'max_ms_level']:
            if da in pkg['args']:
                setattr(recon_args.dformat, da, pkg['args'][da])
        recon_args.d_graphormer_params = 0
        # 启用化学注意力注入 (模块一核心)
        recon_args.chem_attn = True
        recon_args.chem_attn_tolerance = 0.02
        recon_args.chem_attn_layer = -1  # 仅注入最后一层

        spec_preproc = SpectrumPreprocessor(dformat=recon_args.dformat,
                                            n_highest_peaks=recon_args.max_peaks_n)
        model = ChemAwareDreaMS(recon_args, spec_preproc)
        model.load_state_dict(state_dict, strict=False)
        # 新参数: chem_rule_engine.rule_weights_raw (335 dims) + chem_residual_scale (标量)
        print(f'   Loaded {len(state_dict)} pretrained params + chemical attention (335 rules)')
    else:
        raise ValueError(f'Unknown checkpoint format. Keys: {list(pkg.keys())}')

    # Frozen copy for preservation loss
    import copy
    model_frozen = copy.deepcopy(model)
    model_frozen.eval()
    for p in model_frozen.parameters(): p.requires_grad = False

    model = model.to(device)
    model_frozen = model_frozen.to(device)
    for p in model.parameters(): p.requires_grad = True
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

    # ---- 4. Training (化学规则由 ChemAwareDreaMS 内部注入) ----
    print(f'\n[5] Training: α={args.alpha} β={args.beta} margin={args.margin}')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, len(train_trip)//args.batch_size), T_mult=2, eta_min=1e-7)

    history = {'epoch':[], 'train_loss':[], 'val_loss':[], 'train_sep':[], 'val_sep':[], 'lr':[]}
    best_val_loss = float('inf')
    rng = np.random.RandomState(42)

    for epoch in range(args.epochs):
        model.train()
        idx = rng.permutation(len(train_trip))
        losses, seps = [], []

        pbar = tqdm(range(0, len(idx), args.batch_size), desc=f'Epoch {epoch+1}')
        for start in pbar:
            end = min(start + args.batch_size, len(idx))
            batch = [train_trip[i] for i in idx[start:end]]
            B = len(batch)

            a_t = torch.stack([ik_to_spec[t['anchor_ik']] for t in batch]).to(device)
            p_t = torch.stack([ik_to_spec[t['pos_ik']] for t in batch]).to(device)
            n_t = torch.stack([ik_to_spec[t['neg_ik']] for t in batch]).to(device)

            # Forward all at once
            all_t = torch.cat([a_t, p_t, n_t], dim=0)
            embs = model(all_t, None)
            emb = embs[:, 0, :]

            a_emb = emb[:B]; p_emb = emb[B:2*B]; n_emb = emb[2*B:]

            # Triplet loss
            loss_trip = F.triplet_margin_loss(a_emb, p_emb, n_emb, margin=args.margin, p=2)

            # Preservation loss (on anchors only)
            if args.beta > 0:
                with torch.no_grad():
                    frozen_emb = model_frozen(a_t, None)[:, 0, :]
                loss_pres = F.mse_loss(a_emb, frozen_emb)
            else:
                loss_pres = torch.tensor(0.0, device=device)

            # Mask prediction loss (on anchors)
            pad = a_t[:, :, 0] == 0
            N_peaks = a_t.shape[1]
            mask_b = torch.zeros(B, N_peaks, dtype=torch.bool, device=device)
            spec_masked = a_t.clone()
            for b in range(B):
                vc = (~pad[b]).sum().item()
                nm = max(1, int(vc * 0.15))
                vi = torch.where(~pad[b])[0]
                if len(vi) > 1:
                    pm = torch.randperm(len(vi)-1, device=device)[:nm] + 1
                    mi = vi[pm]
                else:
                    mi = vi[:1]
                mask_b[b, mi] = True
                spec_masked[b, mi, :] = 0.0
            loss_mask, _, _, _ = model.spec_ssl_step(spec_masked, a_t, mask_b, None)
            loss_mask = loss_mask.sum() / loss_mask.numel()

            loss = loss_mask + args.alpha * loss_trip + args.beta * loss_pres

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                cp = F.cosine_similarity(a_emb, p_emb, dim=-1).mean()
                cn = F.cosine_similarity(a_emb, n_emb, dim=-1).mean()

            losses.append(loss.item()); seps.append((cp-cn).item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'sep': f'{cp-cn:.3f}',
                              'cos+': f'{cp:.3f}', 'cos-': f'{cn:.3f}'})

        # Validation
        model.eval()
        v_losses, v_seps, v_correct = [], [], []
        with torch.no_grad():
            for start in range(0, len(val_trip), args.batch_size):
                end = min(start + args.batch_size, len(val_trip))
                batch = val_trip[start:end]
                Bv = len(batch)
                a_t = torch.stack([ik_to_spec[t['anchor_ik']] for t in batch]).to(device)
                p_t = torch.stack([ik_to_spec[t['pos_ik']] for t in batch]).to(device)
                n_t = torch.stack([ik_to_spec[t['neg_ik']] for t in batch]).to(device)
                all_t = torch.cat([a_t, p_t, n_t], dim=0)
                emb = model(all_t, None)[:, 0, :]
                loss_t = F.triplet_margin_loss(emb[:Bv], emb[Bv:2*Bv], emb[2*Bv:],
                                                margin=args.margin, p=2)
                # Per-triplet metrics
                cp_all = F.cosine_similarity(emb[:Bv], emb[Bv:2*Bv], dim=-1)
                cn_all = F.cosine_similarity(emb[:Bv], emb[2*Bv:], dim=-1)
                v_losses.append(loss_t.item())
                v_seps.extend((cp_all - cn_all).tolist())
                v_correct.extend((cp_all > cn_all).tolist())

        tl, ts = np.mean(losses), np.mean(seps)
        vl, vs = np.mean(v_losses) if v_losses else 0, np.mean(v_seps) if v_seps else 0

        # Validation triplet accuracy (cos+ > cos-)
        v_acc = sum(v_correct) / len(v_correct) if v_correct else 0.0

        # AUC evaluation (fixed T0+T3 set, every val_every epochs)
        auc_val = 0.0
        if auc_eval is not None and (epoch + 1) % args.val_every == 0:
            with torch.no_grad():
                # Extract embeddings for AUC spectra
                auc_embs = []
                for ik in auc_eval['spec_iks']:
                    st = ik_to_spec[ik].unsqueeze(0).to(device)
                    auc_embs.append(model(st, None)[:, 0, :].cpu())
                auc_embs = torch.cat(auc_embs, dim=0)

                # Cosine similarity for all pairs
                emb_i = auc_embs[auc_eval['pair_i']]
                emb_j = auc_embs[auc_eval['pair_j']]
                cos_sims = F.cosine_similarity(emb_i, emb_j, dim=-1).numpy()

                try:
                    from sklearn import metrics
                    fpr, tpr, _ = metrics.roc_curve(auc_eval['labels'], cos_sims)
                    auc_val = float(metrics.auc(fpr, tpr))
                except Exception:
                    auc_val = 0.5

        print(f'Epoch {epoch+1}: train_loss={tl:.4f} sep={ts:.4f} | '
              f'val_loss={vl:.4f} sep={vs:.4f} acc={v_acc:.3f} auc={auc_val:.4f}')

        for arr, val in [(history['train_loss'], tl), (history['val_loss'], vl),
                          (history['train_sep'], ts), (history['val_sep'], vs),
                          (history['epoch'], epoch+1), (history['lr'], optimizer.param_groups[0]['lr'])]:
            arr.append(val)
        if 'val_acc' not in history: history['val_acc'] = []
        if 'val_auc' not in history: history['val_auc'] = []
        history['val_acc'].append(v_acc)
        history['val_auc'].append(auc_val)

        # Save best by AUC when available, else by val_sep
        best_metric = auc_val if auc_eval is not None else vs
        is_best = best_metric > best_val_loss if best_val_loss != float('inf') else True
        if is_best:
            best_val_loss = best_metric
            ckpt_data = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_sep': vs, 'val_acc': v_acc, 'val_auc': auc_val,
                        'history': history}
            torch.save(ckpt_data, save_dir/'best.pt')
            metric_name = 'auc' if auc_eval is not None else 'sep'
            print(f'  → Best ({metric_name}={best_metric:.4f}, sep={vs:.4f}, acc={v_acc:.3f})')

        if not args.save_best_only:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'history': history}, save_dir/f'epoch_{epoch+1}.pt')

    torch.save({'model_state_dict': model.state_dict(), 'history': history}, save_dir/'final_model.pt')
    print(f'\nDone. Best val_sep={best_val_loss:.4f}. Models in {save_dir}/')


if __name__ == '__main__':
    main()
