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

# Bypass dreams __init__ chain (import only what we need)
import importlib.util
_dreams_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_dreams_dir, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_chem_rules = _load_module('chem_rules', 'models/chem_aware/chem_rules.py')
ChemicalRuleEngine = _chem_rules.ChemicalRuleEngine


def parse_args():
    p = argparse.ArgumentParser(description='T1 MCES triplet fine-tuning')
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--alpha', type=float, default=0.5, help='Triplet loss weight')
    p.add_argument('--beta', type=float, default=0.01, help='Preservation loss weight')
    p.add_argument('--margin', type=float, default=0.3, help='Triplet margin')
    p.add_argument('--save_dir', type=str, default='./triplet_t1_checkpoints')
    p.add_argument('--n_peaks', type=int, default=128)
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

    # ---- 3. Load model (exact same logic as train_contrastive_v5.py main) ----
    print(f'[3] Loading DreaMS from {args.ckpt_path}...')
    pkg = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)

    if 'pytorch-lightning_version' in pkg:
        from dreams.models.dreams.dreams import DreaMS
        model = DreaMS.load_from_checkpoint(args.ckpt_path, map_location='cpu')
    elif 'args' in pkg and 'state_dict' in pkg:
        from argparse import Namespace
        from dreams.utils.dformats import DataFormatA
        from dreams.utils.data import SpectrumPreprocessor
        from dreams.models.dreams.dreams import DreaMS

        state_dict = pkg['state_dict']
        recon_args = Namespace(**pkg['args'])
        recon_args.dformat = DataFormatA()
        for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
                   'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
                   'min_intensity_ampl', 'max_ms_level']:
            if da in pkg['args']:
                setattr(recon_args.dformat, da, pkg['args'][da])
        recon_args.d_graphormer_params = 0

        spec_preproc = SpectrumPreprocessor(dformat=recon_args.dformat,
                                            n_highest_peaks=recon_args.max_peaks_n)
        model = DreaMS(recon_args, spec_preproc)
        model.load_state_dict(state_dict, strict=False)
        print(f'   Loaded {len(state_dict)} params')
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

    # ---- 4. Rule Engine ----
    print('[4] Initializing ChemicalRuleEngine...')
    engine = ChemicalRuleEngine(tolerance=0.02)
    engine = engine.to(device)

    # ---- 5. Training ----
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
        v_losses, v_seps = [], []
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
                cp = F.cosine_similarity(emb[:Bv], emb[Bv:2*Bv], dim=-1).mean()
                cn = F.cosine_similarity(emb[:Bv], emb[2*Bv:], dim=-1).mean()
                v_losses.append(loss_t.item()); v_seps.append((cp-cn).item())

        tl, ts = np.mean(losses), np.mean(seps)
        vl, vs = np.mean(v_losses) if v_losses else 0, np.mean(v_seps) if v_seps else 0
        print(f'Epoch {epoch+1}: train_loss={tl:.4f} sep={ts:.4f} | val_loss={vl:.4f} sep={vs:.4f}')

        for arr, val in [(history['train_loss'], tl), (history['val_loss'], vl),
                          (history['train_sep'], ts), (history['val_sep'], vs),
                          (history['epoch'], epoch+1), (history['lr'], optimizer.param_groups[0]['lr'])]:
            arr.append(val)

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': vl, 'history': history}, save_dir/'best_model.pt')
            print(f'  → Best (val_loss={vl:.4f})')

        # Per-epoch checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'history': history}, save_dir/f'epoch_{epoch+1}.pt')

    torch.save({'model_state_dict': model.state_dict(), 'history': history}, save_dir/'final_model.pt')
    print(f'\nDone. Best val_loss={best_val_loss:.4f}. Models in {save_dir}/')


if __name__ == '__main__':
    main()
