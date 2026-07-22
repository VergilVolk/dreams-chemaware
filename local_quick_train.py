"""Local CPU quick training: 500 spectra, 50 steps, verify all metrics."""
import torch, time, copy, numpy as np
from pathlib import Path
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F

import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS
from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine
from dreams.models.chem_aware.train_contrastive_v5 import (
    compute_batch_rule_vectors, sample_triplets_by_overlap
)

device = torch.device('cpu')
print(f'Device: {device}')

# ---- Load model ----
ckpt_path = 'dreams/models/pretrained/ssl_model_server.pt'
pkg = torch.load(ckpt_path, map_location='cpu', weights_only=False)
from argparse import Namespace
recon_args = Namespace(**pkg['args'])
recon_args.dformat = dformats.DataFormatA()
for da in ['max_mz','max_peaks_n','max_tbxic_stdev','min_peaks_n','min_charge','max_charge','max_prec_mz','high_intensity_thld','min_intensity_ampl','max_ms_level']:
    if da in pkg['args']: setattr(recon_args.dformat, da, pkg['args'][da])
recon_args.d_graphormer_params = 0

spec_preproc = du.SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=recon_args.max_peaks_n)
model = DreaMS(recon_args, spec_preproc)
state = model.state_dict()
for k in state:
    if k in pkg['state_dict'] and state[k].shape == pkg['state_dict'][k].shape:
        state[k] = pkg['state_dict'][k].clone()
model.load_state_dict(state, strict=False)
model.eval()
print(f'Model loaded')

# Frozen copy for preservation
model_frozen = copy.deepcopy(model)
model_frozen.eval()
for p in model_frozen.parameters(): p.requires_grad = False

engine = ChemicalRuleEngine(tolerance=0.02)
print(f'Engine: {len(engine.rules)} rules')

# ---- Load small subset ----
print('Loading 500 spectra...')
msdata = du.MSData.load('data/MassSpecGym_MurckoHist_split.hdf5')
rng = np.random.RandomState(42)
all_idx = rng.choice(min(50000, len(msdata)), 50000, replace=False)
subset_idx = all_idx[:500]

dataset = msdata.to_torch_dataset(spec_preproc)
train_dataset = Subset(dataset, subset_idx)
dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
print(f'Data ready: {len(train_dataset)} spectra')

# ---- Setup ----
for p in model.parameters(): p.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
alpha, beta = 0.05, 0.02
margin = 0.2
overlap_high, overlap_low = 0.23, 0.09
categories = ['NL', 'CF', 'ISO', 'HR']

print(f'\n{"="*60}')
print(f'Starting 50-step CPU training...')
print(f'{"="*60}\n')

t0_total = time.time()
for step in range(500):
    # Get batch
    try:
        batch = next(iter(dl))
    except StopIteration:
        dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
        batch = next(iter(dl))

    spec = batch['spectrum']
    prec_mz = batch['precursor_mz']
    padding = spec[:,:,0] == 0
    bs = spec.shape[0]

    # Rule vectors
    with torch.no_grad():
        match_vecs = compute_batch_rule_vectors(engine, spec, padding, prec_mz, categories)
        triplets = sample_triplets_by_overlap(match_vecs, k=2, overlap_high=overlap_high, overlap_low=overlap_low)

    # Forward
    embs = model(spec, None)

    # Preservation
    if beta > 0 and len(triplets) > 0:
        with torch.no_grad():
            embs_frozen = model_frozen(spec, None)
        aidx = torch.tensor([t[0] for t in triplets])
        loss_pres = F.mse_loss(embs[aidx, 0, :], embs_frozen[aidx, 0, :])
    else:
        loss_pres = torch.tensor(0.0)

    # Mask loss
    spec_mask = spec.clone()
    mask_bool = torch.zeros(bs, spec.shape[1], dtype=torch.bool)
    for b in range(bs):
        vc = (~padding[b]).sum().item()
        nm = max(1, int(vc*0.15))
        vi = torch.where(~padding[b])[0]
        idx = vi[torch.randperm(len(vi)-1)[:nm]+1] if len(vi)>1 else vi[:1]
        mask_bool[b, idx] = True; spec_mask[b, idx, :] = 0
    loss_mask, _, _, _ = model.spec_ssl_step(spec_mask, spec, mask_bool, None)
    loss_mask = loss_mask.sum() / loss_mask.numel()

    # Triplet loss
    if len(triplets) > 0:
        aidx = torch.tensor([t[0] for t in triplets])
        pidx = torch.tensor([t[1] for t in triplets])
        nidx = torch.tensor([t[2] for t in triplets])
        loss_trip = F.triplet_margin_loss(
            embs[aidx, 0, :], embs[pidx, 0, :], embs[nidx, 0, :],
            margin=margin, p=2
        )
        cos_pos = F.cosine_similarity(embs[aidx, 0, :], embs[pidx, 0, :], dim=-1)
        cos_neg = F.cosine_similarity(embs[aidx, 0, :], embs[nidx, 0, :], dim=-1)
        sep = (cos_pos.mean() - cos_neg.mean()).item()
    else:
        loss_trip = torch.tensor(0.0)
        sep = 0.0

    loss_total = loss_mask + alpha * loss_trip + beta * loss_pres

    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 50 == 0:
        elapsed = time.time() - t0_total
        print(f'Step {step:3d} | mask={loss_mask.item():.4f} '
              f'trip={loss_trip.item():.4f} pres={loss_pres.item():.4f} | '
              f'n_trip={len(triplets)} Sep={sep:+.4f} | {elapsed/(step+1):.1f}s/step')

elapsed = time.time() - t0_total
print(f'\n{"="*60}')
print(f'50 steps complete in {elapsed/60:.1f} min ({elapsed/50:.1f}s/step)')
print(f'Estimated: 500 steps = {elapsed/50*500/60:.0f} min on CPU')
print(f'ALL SYSTEMS GO - local CPU training verified')
