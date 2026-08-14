"""
T1 Triplet 参数扫描 — 本地高效运行，3组 × 5 epochs，约 1-2 小时 (GPU) / 4-6 小时 (CPU)

用法: python sweep_triplet_params.py
"""
import torch, json, os, sys, time, numpy as np
from pathlib import Path
from argparse import Namespace

sys.path.insert(0, '.')

COMBO = [
    ('conservative', 0.01, 0.05, 0.1),
    ('v5_experience', 0.05, 0.02, 0.2),
    ('moderate', 0.1, 0.02, 0.2),
]

EPOCHS = 5
BATCH_SIZE = 16
LR = 5e-6
AUC_PAIRS = 500
N_TRAIN = 3000  # Subset of training triplets for speed
N_VAL = 780     # All validation triplets


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Train triplets: {N_TRAIN}')

    # ---- Load data once ----
    print('\n[1] Loading data...')
    with open('tasks/T1_near_isomers/test_cases/triplets_train.json') as f:
        train_all = json.load(f)
    with open('tasks/T1_near_isomers/test_cases/triplets_val.json') as f:
        val_all = json.load(f)

    rng = np.random.RandomState(42)
    if len(train_all) > N_TRAIN:
        train_all = [train_all[i] for i in rng.choice(len(train_all), N_TRAIN, replace=False)]

    # Load spectra
    def ik14(x): return x[:14]
    needed = set()
    for t in train_all + val_all:
        needed.add(ik14(t['anchor_ik'])); needed.add(ik14(t['pos_ik'])); needed.add(ik14(t['neg_ik']))

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

    # Pre-compute spectral tensors
    print('  Preprocessing spectra...')
    N_PEAKS = 128

    def peaks_to_tensor(peaks):
        arr = np.array(peaks, dtype=np.float32)
        arr = arr[arr[:, 0].argsort()]
        if len(arr) > N_PEAKS:
            idx = np.argpartition(arr[:, 1], -N_PEAKS)[-N_PEAKS:]
            arr = arr[idx]; arr = arr[arr[:, 0].argsort()]
        max_i = arr[:, 1].max()
        if max_i > 0: arr[:, 1] /= max_i
        padded = np.zeros((N_PEAKS, 2), dtype=np.float32)
        n = min(len(arr), N_PEAKS); padded[:n] = arr[:n]
        return torch.from_numpy(padded)

    ik_to_spec = {}
    for ik, peaks in ik_to_peaks.items():
        ik_to_spec[ik] = peaks_to_tensor(peaks)

    # Filter triplets and truncate IKs
    train_trip = []
    for t in train_all:
        aik, pik, nik = ik14(t['anchor_ik']), ik14(t['pos_ik']), ik14(t['neg_ik'])
        if aik in ik_to_spec and pik in ik_to_spec and nik in ik_to_spec:
            train_trip.append((aik, pik, nik))
    val_trip = []
    for t in val_all:
        aik, pik, nik = ik14(t['anchor_ik']), ik14(t['pos_ik']), ik14(t['neg_ik'])
        if aik in ik_to_spec and pik in ik_to_spec and nik in ik_to_spec:
            val_trip.append((aik, pik, nik))
    print(f'  Train: {len(train_trip)}  Val: {len(val_trip)}')

    # ---- Build AUC eval set (multi-spectrum same-molecule pairs) ----
    print('  Building AUC eval set...')
    # Find IKs with >=2 spectra, collect all their spectra
    ik_all_peaks = {}  # ik → [(peaks), ...] all spectra
    cur_ik = None; cur_peaks = []
    with open('data/annotated01.mgf', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_ik and len(cur_peaks) >= 3:
                    if cur_ik not in ik_all_peaks: ik_all_peaks[cur_ik] = []
                    ik_all_peaks[cur_ik].append(cur_peaks[:])
                cur_ik = None; cur_peaks = []; continue
            if line.startswith('INCHIKEY='): cur_ik = line[9:].strip()[:14]
            elif line[0].isdigit() or (line[0]=='-' and len(line)>1 and line[1].isdigit()):
                p2 = line.split()
                if len(p2) >= 2:
                    try:
                        mz, i = float(p2[0]), float(p2[1])
                        if mz > 0 and i > 0: cur_peaks.append((mz, i))
                    except: pass

    multi_iks = {ik: pks for ik, pks in ik_all_peaks.items() if len(pks) >= 2}
    print(f'  IKs with >=2 spectra: {len(multi_iks)}')

    # Build AUC spectra: for each multi-IK, take up to 3 spectra
    auc_specs = []  # [(ik, tensor), ...]
    auc_ik_to_indices = {}
    for ik, pks_list in multi_iks.items():
        auc_ik_to_indices[ik] = []
        for pk in pks_list[:3]:  # max 3 spectra per IK
            t = peaks_to_tensor(pk)
            if t is not None:
                auc_ik_to_indices[ik].append(len(auc_specs))
                auc_specs.append(t)

    # Build pairs: same IK different spectra = pos, different IK = neg
    multi_ik_list = [ik for ik in multi_iks if len(auc_ik_to_indices[ik]) >= 2]
    all_auc_iks = [ik for ik in auc_ik_to_indices if len(auc_ik_to_indices[ik]) >= 1]
    n_each = AUC_PAIRS // 2
    auc_pair_i, auc_pair_j, auc_labels_list = [], [], []

    # Pos: different spectra of same IK
    n_pos = 0
    while n_pos < n_each and multi_ik_list:
        ik = rng.choice(multi_ik_list)
        idxs = auc_ik_to_indices[ik]
        if len(idxs) >= 2:
            a, b = rng.choice(idxs, 2, replace=False)
            auc_pair_i.append(a); auc_pair_j.append(b); auc_labels_list.append(1)
            n_pos += 1

    # Neg: different IK
    n_neg = 0
    while n_neg < n_each and len(all_auc_iks) >= 2:
        ik_a, ik_b = rng.choice(all_auc_iks, 2, replace=False)
        if ik_a == ik_b: continue
        a = rng.choice(auc_ik_to_indices[ik_a])
        b = rng.choice(auc_ik_to_indices[ik_b])
        auc_pair_i.append(a); auc_pair_j.append(b); auc_labels_list.append(0)
        n_neg += 1

    auc_pair_i = np.array(auc_pair_i); auc_pair_j = np.array(auc_pair_j)
    auc_labels = np.array(auc_labels_list)
    print(f'  AUC: {len(auc_pair_i)} pairs ({n_pos}P+{n_neg}N), {len(auc_specs)} spectra from {len(multi_ik_list)} IKs')

    # ---- Load base model once, clone for each run ----
    print('\n[2] Loading base DreaMS...', flush=True)
    t_load = time.time()
    pkg = torch.load('dreams/models/pretrained/ssl_model_server.pt', map_location='cpu', weights_only=False)
    print(f'  Checkpoint loaded ({time.time()-t_load:.0f}s), building model...', flush=True)
    from dreams.utils.dformats import DataFormatA
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.models.dreams.dreams import DreaMS

    recon_args = Namespace(**pkg['args'])
    recon_args.dformat = DataFormatA()
    for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge','max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
        if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
    recon_args.d_graphormer_params = 0
    print(f'  Args built, creating model...', flush=True)

    def make_fresh_model():
        sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
        m = DreaMS(recon_args, sp)
        state = m.state_dict()
        for k in state:
            if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
                state[k] = pkg['state_dict'][k].clone()
        m.load_state_dict(state, strict=False)
        m.eval().to(device)
        return m

    # Warm-up: build first model to trigger any lazy init
    print(f'  Building first model instance...', flush=True)
    _ = make_fresh_model()
    print(f'  Model ready ({time.time()-t_load:.0f}s total)', flush=True)

    import copy
    import torch.nn.functional as F
    from sklearn import metrics

    results = {}

    for name, alpha, beta, margin in COMBO:
        print(f'\n{"="*60}', flush=True)
        print(f'  {name}: α={alpha} β={beta} margin={margin}', flush=True)
        print(f'{"="*60}', flush=True)

        t_combo = time.time()
        model = make_fresh_model()
        print(f'  Model ready ({time.time()-t_combo:.0f}s), starting training...', flush=True)
        model_frozen = copy.deepcopy(model)
        model_frozen.eval()
        for p in model_frozen.parameters(): p.requires_grad = False
        for p in model.parameters(): p.requires_grad = True
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        history = {'epoch':[], 'train_loss':[], 'val_loss':[], 'train_sep':[], 'val_sep':[], 'val_acc':[], 'val_auc':[], 'lr':[]}
        best_auc = 0.0

        for epoch in range(EPOCHS):
            # ---- Train ----
            model.train()
            rng.shuffle(train_trip)
            losses, seps = [], []
            t0 = time.time()

            for start in range(0, len(train_trip), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(train_trip))
                batch = train_trip[start:end]
                B = len(batch)

                a_t = torch.stack([ik_to_spec[a] for a, p, n in batch]).to(device)
                p_t = torch.stack([ik_to_spec[p] for a, p, n in batch]).to(device)
                n_t = torch.stack([ik_to_spec[n] for a, p, n in batch]).to(device)
                all_t = torch.cat([a_t, p_t, n_t], dim=0)
                emb = model(all_t, None)[:, 0, :]
                a_emb, p_emb, n_emb = emb[:B], emb[B:2*B], emb[2*B:]

                loss_trip = F.triplet_margin_loss(a_emb, p_emb, n_emb, margin=margin, p=2)

                if beta > 0:
                    with torch.no_grad():
                        frozen_emb = model_frozen(a_t, None)[:, 0, :]
                    loss_pres = F.mse_loss(a_emb, frozen_emb)
                else:
                    loss_pres = torch.tensor(0.0, device=device)

                # Simplified mask loss
                pad = a_t[:,:,0] == 0; Np = a_t.shape[1]
                mask_b = torch.zeros(B, Np, dtype=torch.bool, device=device)
                spec_m = a_t.clone()
                for b in range(B):
                    vc = (~pad[b]).sum().item(); nm = max(1, int(vc*0.15))
                    vi = torch.where(~pad[b])[0]
                    if len(vi) > 1:
                        pm = torch.randperm(len(vi)-1, device=device)[:nm] + 1
                        mi = vi[pm]
                    else: mi = vi[:1]
                    mask_b[b, mi] = True; spec_m[b, mi, :] = 0.0
                loss_mask, _, _, _ = model.spec_ssl_step(spec_m, a_t, mask_b, None)
                loss_mask = loss_mask.sum() / loss_mask.numel()

                loss = loss_mask + alpha * loss_trip + beta * loss_pres

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    cp = F.cosine_similarity(a_emb, p_emb, dim=-1).mean()
                    cn = F.cosine_similarity(a_emb, n_emb, dim=-1).mean()
                losses.append(loss.item()); seps.append((cp-cn).item())

            # ---- Val ----
            model.eval()
            v_losses, v_seps, v_correct = [], [], []
            with torch.no_grad():
                for start in range(0, len(val_trip), BATCH_SIZE):
                    end = min(start + BATCH_SIZE, len(val_trip))
                    batch = val_trip[start:end]; Bv = len(batch)
                    a_t = torch.stack([ik_to_spec[a] for a,p,n in batch]).to(device)
                    p_t = torch.stack([ik_to_spec[p] for a,p,n in batch]).to(device)
                    n_t = torch.stack([ik_to_spec[n] for a,p,n in batch]).to(device)
                    all_t = torch.cat([a_t,p_t,n_t], dim=0)
                    emb = model(all_t, None)[:,0,:]
                    lt = F.triplet_margin_loss(emb[:Bv], emb[Bv:2*Bv], emb[2*Bv:], margin=margin, p=2)
                    cp_all = F.cosine_similarity(emb[:Bv], emb[Bv:2*Bv], dim=-1)
                    cn_all = F.cosine_similarity(emb[:Bv], emb[2*Bv:], dim=-1)
                    v_losses.append(lt.item())
                    v_seps.extend((cp_all - cn_all).tolist())
                    v_correct.extend((cp_all > cn_all).tolist())

            # ---- AUC (multi-spectrum same-mol vs diff-mol pairs) ----
            with torch.no_grad():
                auc_embs = []
                for spec_t in auc_specs:
                    auc_embs.append(model(spec_t.unsqueeze(0).to(device), None)[:,0,:].cpu())
                auc_embs = torch.cat(auc_embs, dim=0)
                cos_sims = F.cosine_similarity(auc_embs[auc_pair_i], auc_embs[auc_pair_j], dim=-1).detach().numpy()
            try:
                fpr, tpr, _ = metrics.roc_curve(auc_labels, cos_sims)
                auc_val = float(metrics.auc(fpr, tpr))
            except: auc_val = 0.5

            tl = np.mean(losses); ts = np.mean(seps)
            vl = np.mean(v_losses); vs = np.mean(v_seps); va = sum(v_correct)/len(v_correct)
            print(f'  Epoch {epoch+1}: loss={tl:.4f}/{vl:.4f} sep={ts:.4f}/{vs:.4f} acc={va:.3f} auc={auc_val:.4f}  ({time.time()-t0:.0f}s)', flush=True)

            for arr, val in [(history['train_loss'],tl),(history['val_loss'],vl),
                              (history['train_sep'],ts),(history['val_sep'],vs),
                              (history['val_acc'],va),(history['val_auc'],auc_val),
                              (history['epoch'],epoch+1),(history['lr'],LR)]:
                arr.append(val)

            if auc_val > best_auc:
                best_auc = auc_val
                Path(f'./triplet_sweep/{name}').mkdir(parents=True, exist_ok=True)
                torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(),
                           'history':history, 'name':name, 'alpha':alpha, 'beta':beta, 'margin':margin},
                          f'./triplet_sweep/{name}/best.pt')

        results[name] = {'history': history, 'best_auc': best_auc,
                         'alpha': alpha, 'beta': beta, 'margin': margin}

    # ---- Report ----
    print(f'\n{"="*60}')
    print(f'PARAMETER SWEEP RESULTS')
    print(f'{"="*60}')
    print(f'{"Combo":<20s} {"α":>6s} {"β":>6s} {"m":>6s} {"Best AUC":>10s} {"Final Sep":>10s} {"Final Acc":>10s}')
    print(f'{"-"*60}')
    for name, r in results.items():
        h = r['history']
        ba = max(h['val_auc'])
        be = h['epoch'][h['val_auc'].index(ba)]
        fs = h['val_sep'][-1]; fa = h['val_acc'][-1]
        print(f'{name:<20s} {r["alpha"]:>6.3f} {r["beta"]:>6.3f} {r["margin"]:>6.2f} {ba:>10.4f} {fs:>10.4f} {fa:>10.3f}')

    best_combo = max(results, key=lambda n: results[n]['best_auc'])
    print(f'\n  >>> BEST: {best_combo} (AUC={results[best_combo]["best_auc"]:.4f})')

    # Save
    summary = {n: {'best_auc': r['best_auc'], 'alpha': r['alpha'], 'beta': r['beta'], 'margin': r['margin'],
                    'final_sep': r['history']['val_sep'][-1], 'final_acc': r['history']['val_acc'][-1]}
               for n, r in results.items()}
    with open('./triplet_sweep/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: ./triplet_sweep/summary.json')


if __name__ == '__main__':
    main()
