"""
E0: DreaMS Zero-Shot Baseline Retrieval Evaluation (v2 — fixed candidate indexing)

Protocols:
  Primary:   Strict 10-ppm, same adduct, [M+H]+ preferred
  Secondary: Same-polarity, per-adduct
  Control:   Legacy random-negative (historical reference only)

Metrics:
  Pair-level:  ROC-AUC, Average Precision (AP), query-clustered bootstrap 95% CI
  Query-level: Recall@1/5/10, MRR, median rank (molecule-aggregated by 14-char IK)

Data: MassSpecGym_MurckoHist_split.hdf5 (val fold)
Model: ssl_model_server.pt (DreaMS pretrained)

Usage:
  python tasks/eval_e0_baseline.py                              # val fold (default)
  python tasks/eval_e0_baseline.py --fold val --reuse-embeddings # skip inference, re-evaluate only
  python tasks/eval_e0_baseline.py --dry-run                    # quick check
"""
import argparse, hashlib, json, os, sys, time, warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from e1_checkpoint_io import (
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    torch_load_compat,
)

# ── Path setup ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CKPT = REPO_ROOT / 'dreams' / 'models' / 'pretrained' / 'ssl_model_server.pt'

# ── CLI ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='E0: DreaMS Zero-Shot Baseline Retrieval')
    p.add_argument('--data', type=str,
                   default='data/models/MassSpecGym_MurckoHist_split.hdf5')
    p.add_argument('--ckpt', type=str, default=None,
                   help='DreaMS checkpoint (default: ssl_model_server.pt)')
    p.add_argument('--fold', type=str, default='val')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='cpu',
                   help='Device for inference (cpu/cuda). Default: cpu.')
    p.add_argument('--output-dir', type=str, default='data/validation/e0_baseline')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true',
                   help='Process only 500 spectra for quick syntax check')
    p.add_argument('--ppm-tol', type=float, default=10.0,
                   help='ppm tolerance for precursor m/z matching')
    p.add_argument('--peak-hash-tol', type=float, default=0.01,
                   help='m/z rounding tolerance for peak dedup hash (Da)')
    p.add_argument('--n-bootstrap', type=int, default=1000,
                   help='Bootstrap iterations for CI (0 = skip)')
    p.add_argument('--primary-adduct', type=str, default='[M+H]+')
    p.add_argument('--reuse-embeddings', action='store_true',
                   help='Skip inference; load e0_embeddings.npy + e0_manifest.json from output-dir')
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────

def decode_bytes(x):
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='ignore')
    return str(x)

def sha256_prefix(path, length=16):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:length]

def ik14(ik: str) -> str:
    return ik[:14] if len(ik) > 14 else ik

def peak_hash(spec_2_n, tol: float = 0.01) -> str:
    mzs = spec_2_n[0]; intens = spec_2_n[1]
    mask = mzs > 0
    mzs = mzs[mask]; intens = intens[mask]
    if len(mzs) == 0:
        return "empty"
    mz_binned = np.round(mzs / tol).astype(np.int32)
    int_binned = np.round(intens / 0.01).astype(np.int32)
    order = np.argsort(mz_binned)
    packed = np.stack([mz_binned[order], int_binned[order]], axis=-1)
    return hashlib.blake2b(packed.tobytes(), digest_size=8).hexdigest()

def derive_polarity(adduct: str) -> str:
    if '+' in adduct: return 'POSITIVE'
    elif '-' in adduct: return 'NEGATIVE'
    return 'UNKNOWN'


def rowwise_cosine_chunked(embeddings: np.ndarray,
                           row_i: np.ndarray,
                           row_j: np.ndarray,
                           chunk_size: int = 8192) -> np.ndarray:
    """Compute row-wise dot products without materializing two huge matrices."""
    out = np.empty(len(row_i), dtype=np.float32)
    for start in range(0, len(row_i), chunk_size):
        end = min(start + chunk_size, len(row_i))
        left = embeddings[row_i[start:end]]
        right = embeddings[row_j[start:end]]
        out[start:end] = np.einsum('ij,ij->i', left, right, optimize=True)
    return out


def query_auc_ap(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """AUC and AP for one query with at least one positive and negative."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    differences = pos_scores[:, None] - neg_scores[None, :]
    auc = float(
        (np.count_nonzero(differences > 0) +
         0.5 * np.count_nonzero(differences == 0)) /
        differences.size
    )
    order = np.argsort(-scores, kind='stable')
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    positive_ranks = np.flatnonzero(sorted_labels == 1)
    ap = float(np.mean(
        cumulative_positive[positive_ranks] / (positive_ranks + 1)))
    return auc, ap


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Fix P1: explicit error if CUDA requested but unavailable; support CPU by default
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but this environment has a CPU-only PyTorch build.\n'
            'Install a CUDA-compatible PyTorch, or use --device cpu.'
        )
    device = torch.device(args.device)
    print(f'Device: {device}')
    print(f'Fold: {args.fold}')
    print(f'Output: {args.output_dir}')

    # ══════════════════════════════════════════════════════════════
    # 1. Load Data
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('[1/8] Loading data...')
    t0 = time.time()

    if args.reuse_embeddings:
        manifest_path = os.path.join(args.output_dir, 'e0_manifest.json')
        emb_path = os.path.join(args.output_dir, 'e0_embeddings.npy')
        if not os.path.exists(manifest_path) or not os.path.exists(emb_path):
            raise FileNotFoundError(
                f'Embedding cache is incomplete in {args.output_dir}.')
        with open(manifest_path) as fm:
            cached_manifest = json.load(fm)
        cached_folds = sorted({m['fold'] for m in cached_manifest})
        if cached_folds != [args.fold]:
            raise ValueError(
                f'Cached manifest fold(s) {cached_folds} do not match --fold {args.fold}.')
        eval_fold = args.fold
        records = [{
            'identifier': m['spectrum_id'],
            'inchikey': m['inchikey_14'],
            'smiles': m['smiles'],
            'precursor_mz': m['precursor_mz'],
            'adduct': m['adduct'],
            'polarity': m['polarity'],
            'peak_hash': m['peak_hash'],
            'n_peaks': m['n_peaks'],
            'ce': m.get('ce', 0),
            'instrument': m.get('instrument', ''),
            'fold': m['fold'],
        } for m in cached_manifest]
        n_total = len(records)
        print(f'  Reusing cached metadata: {n_total:,} {eval_fold} spectra')
    else:
        f = h5py.File(args.data, 'r')
        all_folds = np.array([decode_bytes(x) for x in f['fold'][:]])

        available_folds = sorted(set(all_folds))
        print(f'  Available folds: {available_folds}')
        if args.fold not in available_folds:
            raise ValueError(f'Fold "{args.fold}" not in {available_folds}')
        eval_fold = args.fold

        fold_mask = all_folds == eval_fold
        fold_indices = np.where(fold_mask)[0]
        print(f'  {eval_fold} fold: {len(fold_indices):,} spectra')

        n_total = len(fold_indices)
        if args.dry_run:
            rng_dry = np.random.RandomState(args.seed)
            fold_indices = rng_dry.choice(fold_indices, min(500, n_total), replace=False)
            n_total = len(fold_indices)
            print(f'  DRY RUN: using {n_total} spectra')

        print(f'  Loading {n_total:,} spectra metadata...')
        records = []
        for idx in tqdm(fold_indices, desc='  Loading'):
            rec = {
                'identifier':   decode_bytes(f['IDENTIFIER'][idx]),
                'inchikey':     decode_bytes(f['INCHIKEY'][idx]),
                'smiles':       decode_bytes(f['smiles'][idx]),
                'formula':      decode_bytes(f['FORMULA'][idx]),
                'precursor_mz': float(f['precursor_mz'][idx]),
                'adduct':       decode_bytes(f['adduct'][idx]),
                'polarity':     derive_polarity(decode_bytes(f['adduct'][idx])),
                'ce':           float(f['COLLISION_ENERGY'][idx]),
                'instrument':   decode_bytes(f['INSTRUMENT_TYPE'][idx]),
                'fold':         eval_fold,
            }
            rec['inchikey'] = ik14(rec['inchikey'])
            records.append(rec)

        print(f'  Computing peak hashes...')
        for i, idx in enumerate(tqdm(fold_indices, desc='  Hashing')):
            spec = f['spectrum'][idx]
            records[i]['peak_hash'] = peak_hash(spec, args.peak_hash_tol)
            records[i]['n_peaks'] = int((spec[0] > 0).sum())

        f.close()
    print(f'  Data loaded in {time.time() - t0:.1f}s')

    hash_counts = Counter(r['peak_hash'] for r in records)
    dup_hashes = {h: c for h, c in hash_counts.items() if c > 1}
    n_dup_spectra = sum(c - 1 for c in dup_hashes.values())
    print(f'  Peak dedup: {len(dup_hashes)} duplicate hashes, {n_dup_spectra} redundant spectra')

    # ══════════════════════════════════════════════════════════════
    # 2-3. Load Model & Extract Embeddings (or reuse)
    # ══════════════════════════════════════════════════════════════
    precomputed = args.output_dir

    if args.reuse_embeddings:
        emb_path_reuse = os.path.join(precomputed, 'e0_embeddings.npy')
        manifest_path_reuse = os.path.join(precomputed, 'e0_manifest.json')
        if not os.path.exists(emb_path_reuse) or not os.path.exists(manifest_path_reuse):
            raise FileNotFoundError(
                f'Embeddings not found at {precomputed}/. Run without --reuse-embeddings first.')

        print('\n' + '=' * 60)
        print('[2-3/8] Reusing existing embeddings...')
        emb_array = np.load(emb_path_reuse)
        manifest = cached_manifest
        if emb_array.ndim != 2 or emb_array.shape[0] != len(manifest):
            raise ValueError(
                f'Embedding cache shape {emb_array.shape} does not match '
                f'manifest length {len(manifest)}.')
        records = []
        for m in manifest:
            records.append({
                'identifier': m['spectrum_id'],
                'inchikey': m['inchikey_14'],
                'smiles': m['smiles'],
                'precursor_mz': m['precursor_mz'],
                'adduct': m['adduct'],
                'polarity': m['polarity'],
                'peak_hash': m['peak_hash'],
                'n_peaks': m['n_peaks'],
                'ce': m.get('ce', 0),
                'instrument': m.get('instrument', ''),
                'fold': m['fold'],
            })
        n_valid = len(records)
        # Re-derive arrays
        pmzs = np.array([r['precursor_mz'] for r in records])
        iks = np.array([r['inchikey'] for r in records])
        hashes = np.array([r['peak_hash'] for r in records])
        adducts = np.array([r['adduct'] for r in records])
        polarities = np.array([r['polarity'] for r in records])
        d_model = emb_array.shape[1]
        ckpt_path = args.ckpt or str(DEFAULT_CKPT)
        previous_report_path = os.path.join(precomputed, 'e0_report.json')
        if os.path.exists(previous_report_path):
            with open(previous_report_path) as fr:
                previous_report = json.load(fr)
            ckpt_hash = previous_report.get('checkpoint_sha256', 'reused')
        else:
            ckpt_hash = 'reused'
        n_failed = 0
        print(f'  Loaded {n_valid:,} spectra, {d_model}-dim embeddings from cache')
    else:
        # ── 2. Load Model ──
        print('\n' + '=' * 60)
        print('[2/8] Loading DreaMS model...')

        # Lazy imports keep cache-only evaluation lightweight.
        from dreams.utils.data import SpectrumPreprocessor
        from dreams.utils.dformats import DataFormatA
        from dreams.models.dreams.dreams import DreaMS

        ckpt_path = args.ckpt or str(DEFAULT_CKPT)
        print(f'  Checkpoint: {ckpt_path}')
        pkg = torch_load_compat(ckpt_path, map_location='cpu')
        checkpoint_format = checkpoint_kind(pkg)
        head_state = None

        if checkpoint_format == 'causal_chemmask_head':
            base_path = Path(pkg.get('base_checkpoint', ''))
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Causal head base checkpoint is unavailable: {base_path}"
                )
            base_package = torch_load_compat(base_path, map_location='cpu')
            architecture_path = Path(
                pkg.get('config', {}).get('architecture_ckpt', DEFAULT_CKPT)
            )
            if not architecture_path.exists():
                architecture_path = DEFAULT_CKPT
            args_package = torch_load_compat(architecture_path, map_location='cpu')
            state_dict = official_backbone_state(base_package)
            head_state = pkg['head_state_dict']
            n_highest_peaks = int(pkg.get('config', {}).get('n_highest_peaks', 100))
            print('  Checkpoint format: causal ChemMask head on frozen official DreaMS')
            print(f'  Frozen backbone checkpoint: {base_path}')
        elif checkpoint_format in ('e1_identity', 'counterfactual_dreams'):
            architecture_path = Path(pkg.get('architecture_checkpoint', DEFAULT_CKPT))
            if not architecture_path.exists():
                architecture_path = DEFAULT_CKPT
            args_package = torch_load_compat(architecture_path, map_location='cpu')
            state_dict = pkg['backbone_state_dict']
            head_state = pkg['head_state_dict']
            n_highest_peaks = int(pkg.get('config', {}).get('n_highest_peaks', 100))
            print(f'  Checkpoint format: trained embedding ({checkpoint_format})')
            print(f'  Architecture checkpoint: {architecture_path}')
        elif checkpoint_format in ('official_embedding', 'official_embedding_slim'):
            args_package = torch_load_compat(DEFAULT_CKPT, map_location='cpu')
            state_dict = official_backbone_state(pkg)
            head_state = official_head_state(pkg)
            n_highest_peaks = 100
            print('  Checkpoint format: official DreaMS contrastive embedding')
        elif checkpoint_format == 'raw_ssl':
            args_package = pkg
            state_dict = pkg['state_dict']
            n_highest_peaks = None
            print('  Checkpoint format: raw self-supervised DreaMS')
        else:
            raise ValueError(f'Unknown checkpoint format: {checkpoint_format}')

        from argparse import Namespace
        recon_args = Namespace(**args_package['args'])
        recon_args.dformat = DataFormatA()
        for da in ['max_mz', 'max_peaks_n', 'max_tbxic_stdev', 'min_peaks_n',
                   'min_charge', 'max_charge', 'max_prec_mz', 'high_intensity_thld',
                   'min_intensity_ampl', 'max_ms_level']:
            if da in args_package['args']:
                setattr(recon_args.dformat, da, args_package['args'][da])
        recon_args.d_graphormer_params = 0

        model_dim = getattr(recon_args, 'd_model', None)
        print(f'  Model dim: {model_dim}')
        print(f'  Charge feature: {getattr(recon_args, "charge_feature", "unknown")}')
        print(f'  N layers: {getattr(recon_args, "n_layers", "unknown")}')
        print(f'  N heads: {getattr(recon_args, "n_heads", "unknown")}')

        spec_preproc = SpectrumPreprocessor(
            dformat=recon_args.dformat,
            n_highest_peaks=(n_highest_peaks if n_highest_peaks is not None
                             else getattr(recon_args, 'max_peaks_n', 128)),
        )

        backbone = DreaMS(recon_args, spec_preproc)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'  Missing keys: {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')

        if head_state is not None:
            class ProjectedEvaluationModel(torch.nn.Module):
                def __init__(self, backbone_, dimension, head_state):
                    super().__init__()
                    self.backbone = backbone_
                    self.head = torch.nn.Linear(dimension, dimension, bias=True)
                    self.head.load_state_dict(head_state, strict=True)

                def forward(self, spectra, charge=None):
                    precursor = self.backbone(spectra, charge)[:, 0, :]
                    return self.head(precursor)

            model = ProjectedEvaluationModel(backbone, int(recon_args.d_model), head_state)
        else:
            model = backbone

        model = model.to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Model loaded: {n_params:,} params')

        ckpt_hash = sha256_prefix(ckpt_path)
        print(f'  Checkpoint SHA256: {ckpt_hash}')

        # ── 3. Extract Embeddings ──
        print('\n' + '=' * 60)
        print(f'[3/8] Extracting embeddings for {n_total:,} spectra...')

        n_failed = 0
        failed_indices = []

        h5_raw = h5py.File(args.data, 'r')
        all_raw_specs = h5_raw['spectrum'][fold_indices]
        h5_raw.close()

        print('  Preprocessing spectra...')
        preprocessed_specs = []
        for i, rec in enumerate(tqdm(records, desc='  Preproc')):
            spec_raw = all_raw_specs[i]
            try:
                spec_pp = spec_preproc(spec_raw, prec_mz=rec['precursor_mz'], high_form=False)
                preprocessed_specs.append(torch.as_tensor(spec_pp, dtype=torch.float32))
            except Exception:
                n_failed += 1
                failed_indices.append(i)
                preprocessed_specs.append(None)

        if n_failed > 0:
            print(f'  ⚠ {n_failed} spectra failed preprocessing; will skip')
        records = [r for i, r in enumerate(records) if i not in failed_indices]
        preprocessed_specs = [s for s in preprocessed_specs if s is not None]
        n_valid = len(records)
        print(f'  Valid spectra: {n_valid:,}')

        print(f'  Running inference (batch_size={args.batch_size})...')
        all_embs = []
        for start in tqdm(range(0, n_valid, args.batch_size), desc='  Embedding'):
            end = min(start + args.batch_size, n_valid)
            batch = torch.stack(preprocessed_specs[start:end]).to(device)
            with torch.no_grad():
                emb = model(batch, None)
                if emb.ndim == 3:
                    emb = emb[:, 0, :]
                emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu())

        all_embs = torch.cat(all_embs, dim=0)
        d_model = all_embs.shape[1]
        emb_array = all_embs.numpy()
        print(f'  Embeddings: {emb_array.shape} (dim={d_model})')

    # ══════════════════════════════════════════════════════════════
    # Derive shared arrays from records (only if not set by reuse path)
    # ══════════════════════════════════════════════════════════════
    if not args.reuse_embeddings:
        for i, rec in enumerate(records):
            rec['embedding_idx'] = i
        pmzs = np.array([r['precursor_mz'] for r in records])
        iks = np.array([r['inchikey'] for r in records])
        hashes = np.array([r['peak_hash'] for r in records])
        adducts = np.array([r['adduct'] for r in records])
        polarities = np.array([r['polarity'] for r in records])

    adduct_types = sorted(set(adducts))
    print(f'\n  Adducts: {adduct_types}')

    # ══════════════════════════════════════════════════════════════
    # 4. Build 10-ppm Candidate Index
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(f'[4/8] Building 10-ppm candidate index (tol={args.ppm_tol} ppm)...')

    # ══════════════════════════════════════════════════════════════
    # 5. Pair-Level Evaluation: Strict 10-ppm (FIXED)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('[5/8] Pair-level evaluation (Strict 10-ppm)...')

    def evaluate_pairs(adduct_filter: str, same_adduct: bool = True,
                       same_polarity: bool = True) -> dict:
        """
        Build all valid 10-ppm candidate pairs and compute metrics.

        FIX: Use sorted_global positions directly (not sort_idx).
        Keep directed query-candidate pairs so every query is evaluated on
        its complete candidate set. Report unordered unique-pair count as a
        separate diagnostic.
        Bootstrap by query cluster, preserving resampling multiplicity.
        Use average_precision_score for AP.
        """
        if same_adduct:
            mask = adducts == adduct_filter
            label = f'10ppm, [{adduct_filter}]'
        elif same_polarity:
            target_pol = derive_polarity(adduct_filter) if '+' in adduct_filter else 'POSITIVE'
            mask = np.array([derive_polarity(a) == target_pol for a in adducts])
            label = '10ppm, same-polarity'
        else:
            mask = np.ones(n_valid, dtype=bool)
            label = '10ppm, all'

        indices = np.where(mask)[0]
        if len(indices) < 2:
            return {'label': label, 'error': f'Too few spectra: {len(indices)}'}

        pmzs_sub = pmzs[indices]
        iks_sub = iks[indices]
        hashes_sub = hashes[indices]

        # Sort by precursor m/z for efficient window search
        sort_idx = np.argsort(pmzs_sub)
        sorted_pmzs = pmzs_sub[sort_idx]
        sorted_global = indices[sort_idx]   # global indices in sorted order

        # Per-query pair accumulation
        queries_pos = []   # list of lists of global_j for each eligible query
        queries_neg = []
        query_global_i = []  # global_i for each eligible query

        n_skipped_no_pos = 0
        n_skipped_no_neg = 0
        n_skipped_duplicate = 0

        for qpos, global_i in enumerate(tqdm(sorted_global, desc=f'  Pairs ({label})')):
            mz_q = pmzs[global_i]
            ik_q = iks[global_i]
            hash_q = hashes[global_i]

            # Find 10-ppm window in sorted array (FIX: use sorted_pmzs)
            ppm_da = args.ppm_tol * 1e-6 * mz_q
            lo = np.searchsorted(sorted_pmzs, mz_q - ppm_da, side='left')
            hi = np.searchsorted(sorted_pmzs, mz_q + ppm_da, side='right')

            pos_candidates = []
            neg_candidates = []

            # FIX: iterate over cpos in range(lo, hi), use sorted_global[cpos] directly
            for cpos in range(lo, hi):
                if cpos == qpos:
                    continue
                global_j = sorted_global[cpos]
                # Peak dedup
                if hashes[global_i] == hashes[global_j]:
                    continue
                if iks[global_i] == iks[global_j]:
                    pos_candidates.append(global_j)
                else:
                    neg_candidates.append(global_j)

            if len(pos_candidates) == 0:
                n_skipped_no_pos += 1
                continue
            if len(neg_candidates) == 0:
                n_skipped_no_neg += 1
                continue

            query_global_i.append(global_i)
            queries_pos.append(pos_candidates)
            queries_neg.append(neg_candidates)

        n_eligible_queries = len(query_global_i)

        if n_eligible_queries == 0:
            return {'label': label, 'error': 'No eligible queries'}

        # Flatten to arrays
        pair_i_all, pair_j_all, pair_labels_all = [], [], []
        query_ids = []  # which query each pair belongs to
        for qi in range(n_eligible_queries):
            gi = query_global_i[qi]
            for gj in queries_pos[qi]:
                pair_i_all.append(gi)
                pair_j_all.append(gj)
                pair_labels_all.append(1)
                query_ids.append(qi)
            for gj in queries_neg[qi]:
                pair_i_all.append(gi)
                pair_j_all.append(gj)
                pair_labels_all.append(0)
                query_ids.append(qi)

        pair_i_all = np.array(pair_i_all)
        pair_j_all = np.array(pair_j_all)
        pair_labels_all = np.array(pair_labels_all)
        query_ids = np.array(query_ids)
        n_pos = int(pair_labels_all.sum())
        n_neg = len(pair_labels_all) - n_pos
        unique_pair_keys = {
            (min(int(i), int(j)), max(int(i), int(j)))
            for i, j in zip(pair_i_all, pair_j_all)
        }
        n_unique_pairs = len(unique_pair_keys)

        print(f'    Eligible queries: {n_eligible_queries}')
        print(f'    Skipped (no pos): {n_skipped_no_pos}, (no neg): {n_skipped_no_neg}')
        print(f'    Directed pairs: {len(pair_labels_all):,} ({n_pos}P + {n_neg}N)')
        print(f'    Unordered unique pairs: {n_unique_pairs:,}')

        # Cosine similarities
        cos_sims = rowwise_cosine_chunked(
            emb_array, pair_i_all, pair_j_all)

        # AUC and Average Precision (FIX: use average_precision_score)
        try:
            from sklearn import metrics
            fpr, tpr, _ = metrics.roc_curve(pair_labels_all, cos_sims)
            auc_roc = float(metrics.auc(fpr, tpr))
            ap = float(metrics.average_precision_score(pair_labels_all, cos_sims))
        except Exception as e:
            auc_roc = 0.5
            ap = float(np.mean(pair_labels_all))
            print(f'    ⚠ sklearn metrics failed: {e}')

        # Query-macro metrics ensure that molecules with many replicate
        # spectra do not dominate the point estimate. Bootstrap the query
        # metrics themselves, which is a true query-cluster bootstrap.
        query_counts = np.bincount(query_ids, minlength=n_eligible_queries)
        query_offsets = np.concatenate([[0], np.cumsum(query_counts)])
        query_aucs = np.empty(n_eligible_queries, dtype=np.float64)
        query_aps = np.empty(n_eligible_queries, dtype=np.float64)
        for qid in range(n_eligible_queries):
            start, end = query_offsets[qid], query_offsets[qid + 1]
            query_aucs[qid], query_aps[qid] = query_auc_ap(
                pair_labels_all[start:end], cos_sims[start:end])
        macro_auc = float(query_aucs.mean())
        macro_ap = float(query_aps.mean())

        if args.n_bootstrap > 0 and n_eligible_queries >= 10:
            rng_bs = np.random.RandomState(args.seed)
            bootstrap_auc = np.empty(args.n_bootstrap, dtype=np.float64)
            bootstrap_ap = np.empty(args.n_bootstrap, dtype=np.float64)
            for bootstrap_idx in range(args.n_bootstrap):
                sampled = rng_bs.randint(
                    0, n_eligible_queries, size=n_eligible_queries)
                bootstrap_auc[bootstrap_idx] = query_aucs[sampled].mean()
                bootstrap_ap[bootstrap_idx] = query_aps[sampled].mean()
            macro_auc_ci = tuple(
                float(v) for v in np.percentile(bootstrap_auc, [2.5, 97.5]))
            macro_ap_ci = tuple(
                float(v) for v in np.percentile(bootstrap_ap, [2.5, 97.5]))
        else:
            macro_auc_ci = (macro_auc, macro_auc)
            macro_ap_ci = (macro_ap, macro_ap)

        return {
            'label': label,
            'n_queries_eligible': n_eligible_queries,
            'n_queries_skipped_no_pos': n_skipped_no_pos,
            'n_queries_skipped_no_neg': n_skipped_no_neg,
            'n_pairs_directed': len(pair_labels_all),
            'n_pairs_unique': n_unique_pairs,
            'n_pairs_positive': int(n_pos),
            'n_pairs_negative': int(n_neg),
            'auc_roc': auc_roc,
            'ap': ap,
            'macro_query_auc': macro_auc,
            'macro_query_auc_95ci': macro_auc_ci,
            'macro_query_ap': macro_ap,
            'macro_query_ap_95ci': macro_ap_ci,
            'pos_cos_mean': float(cos_sims[pair_labels_all == 1].mean()) if n_pos > 0 else 0.0,
            'neg_cos_mean': float(cos_sims[pair_labels_all == 0].mean()) if n_neg > 0 else 0.0,
            'separation': float(cos_sims[pair_labels_all == 1].mean() - cos_sims[pair_labels_all == 0].mean()),
            'pair_indices': (pair_i_all, pair_j_all, pair_labels_all, cos_sims, query_ids),
        }

    pair_results = {}
    computed_adducts = set()

    primary_adduct = args.primary_adduct
    if primary_adduct not in adduct_types:
        print(f'  ⚠ Primary adduct [{primary_adduct}] not found! Available: {adduct_types}')
        primary_adduct = Counter(adducts).most_common(1)[0][0]
        print(f'  → Using most common adduct: [{primary_adduct}]')

    pair_results['primary'] = evaluate_pairs(primary_adduct, same_adduct=True)
    computed_adducts.add(primary_adduct)

    for ad in adduct_types:
        if ad not in computed_adducts:
            pair_results[f'adduct_{ad}'] = evaluate_pairs(ad, same_adduct=True)
            computed_adducts.add(ad)

    # ══════════════════════════════════════════════════════════════
    # 6. Query-Level Evaluation (FIXED)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('[6/8] Query-level evaluation (molecule-aggregated by 14-char IK)...')

    def evaluate_query_level(adduct_filter: str = None) -> dict:
        """
        Query-level metrics with molecule-level aggregation.

        FIX: Require both positive AND negative candidates for eligibility.
        FIX: Use normal dict instead of defaultdict(float) for max-sim aggregation.
        """
        if adduct_filter:
            mask = adducts == adduct_filter
            label = f'[{adduct_filter}]'
        else:
            mask = np.ones(n_valid, dtype=bool)
            label = 'all'

        q_indices = np.where(mask)[0]
        if len(q_indices) < 2:
            return {'label': label, 'error': 'Too few spectra'}

        recalls = {1: [], 5: [], 10: []}
        mrrs = []
        ranks = []
        n_eligible = 0
        n_skipped_no_pos = 0
        n_skipped_no_neg = 0
        n_skipped_no_candidate = 0

        for qi in tqdm(q_indices, desc=f'  Query ({label})'):
            mz_q = pmzs[qi]
            ik_q = iks[qi]
            hash_q = hashes[qi]

            # 10-ppm candidate window
            ppm_da = args.ppm_tol * 1e-6 * mz_q
            candidate_mask = (
                (np.abs(pmzs - mz_q) <= ppm_da) &
                (np.arange(n_valid) != qi) &
                (hashes != hash_q)
            )
            if adduct_filter:
                candidate_mask &= (adducts == adduct_filter)

            candidates = np.where(candidate_mask)[0]

            if len(candidates) == 0:
                n_skipped_no_candidate += 1
                continue

            cand_iks = iks[candidates]

            # FIX: require both positive and negative
            pos_mask = cand_iks == ik_q
            neg_mask = cand_iks != ik_q
            if not pos_mask.any():
                n_skipped_no_pos += 1
                continue
            if not neg_mask.any():
                n_skipped_no_neg += 1
                continue

            q_emb = emb_array[qi:qi+1]
            cand_embs = emb_array[candidates]
            cand_sims = (q_emb * cand_embs).sum(axis=-1)

            # Molecule-level aggregation: max cosine per IK (FIX: normal dict)
            ik_to_maxsim = {}
            for c_idx in range(len(candidates)):
                c_ik = cand_iks[c_idx]
                sim = float(cand_sims[c_idx])
                if c_ik not in ik_to_maxsim or sim > ik_to_maxsim[c_ik]:
                    ik_to_maxsim[c_ik] = sim

            sorted_mols = sorted(ik_to_maxsim.items(), key=lambda x: x[1], reverse=True)
            sorted_mol_iks = [m[0] for m in sorted_mols]

            try:
                rank = sorted_mol_iks.index(ik_q) + 1
            except ValueError:
                rank = len(sorted_mols) + 1

            ranks.append(rank)
            mrrs.append(1.0 / rank)
            for k in [1, 5, 10]:
                recalls[k].append(1.0 if rank <= k else 0.0)
            n_eligible += 1

        if n_eligible == 0:
            return {
                'label': label,
                'error': 'No eligible queries',
                'n_eligible_queries': 0,
                'n_skipped_no_candidate': n_skipped_no_candidate,
                'n_skipped_no_pos': n_skipped_no_pos,
                'n_skipped_no_neg': n_skipped_no_neg,
            }

        ranks_arr = np.array(ranks)
        return {
            'label': label,
            'n_eligible_queries': n_eligible,
            'n_skipped_no_candidate': n_skipped_no_candidate,
            'n_skipped_no_pos': n_skipped_no_pos,
            'n_skipped_no_neg': n_skipped_no_neg,
            'recall@1': float(np.mean(recalls[1])),
            'recall@5': float(np.mean(recalls[5])),
            'recall@10': float(np.mean(recalls[10])),
            'mrr': float(np.mean(mrrs)),
            'median_rank': float(np.median(ranks_arr)),
            'mean_rank': float(ranks_arr.mean()),
            'n_molecules_in_gallery': len(set(iks[q_indices])),
        }

    query_results = {}
    query_results['primary'] = evaluate_query_level(args.primary_adduct)
    for ad in adduct_types:
        if ad != args.primary_adduct:
            query_results[f'adduct_{ad}'] = evaluate_query_level(ad)

    # ══════════════════════════════════════════════════════════════
    # 7. Legacy Random-Negative Protocol (Historical Control)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('[7/8] Legacy random-negative protocol (historical control)...')

    rng_legacy = np.random.RandomState(args.seed)
    n_pairs_legacy = 20000
    legacy_i, legacy_j, legacy_labels = [], [], []

    ik_to_indices = defaultdict(list)
    for i in range(n_valid):
        ik_to_indices[iks[i]].append(i)

    multi_iks = {ik: idxs for ik, idxs in ik_to_indices.items() if len(idxs) >= 2}
    pos_pool = []
    for ik, idxs in multi_iks.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if hashes[idxs[a]] != hashes[idxs[b]]:
                    # FIX: symmetric dedup
                    pair_key = (min(idxs[a], idxs[b]), max(idxs[a], idxs[b]))
                    pos_pool.append(pair_key)

    neg_seen = set()

    n_pos = min(n_pairs_legacy // 2, len(pos_pool))
    if n_pos > 0:
        chosen_pos = rng_legacy.choice(len(pos_pool), n_pos, replace=False)
        for idx in chosen_pos:
            a, b = pos_pool[idx]
            legacy_i.append(a)
            legacy_j.append(b)
            legacy_labels.append(1)
            neg_seen.add((min(a, b), max(a, b)))

    n_neg = min(n_pairs_legacy - n_pos, n_valid * 10)
    neg_count = 0
    max_attempts = n_neg * 100
    attempts = 0
    while neg_count < n_neg and attempts < max_attempts:
        a, b = rng_legacy.randint(0, n_valid, 2)
        attempts += 1
        if a == b: continue
        if iks[a] == iks[b]: continue
        if hashes[a] == hashes[b]: continue
        pk = (min(a, b), max(a, b))
        if pk in neg_seen: continue
        neg_seen.add(pk)
        legacy_i.append(a)
        legacy_j.append(b)
        legacy_labels.append(0)
        neg_count += 1

    legacy_i = np.array(legacy_i)
    legacy_j = np.array(legacy_j)
    legacy_labels = np.array(legacy_labels)
    legacy_sims = rowwise_cosine_chunked(emb_array, legacy_i, legacy_j)

    try:
        from sklearn import metrics
        fpr_l, tpr_l, _ = metrics.roc_curve(legacy_labels, legacy_sims)
        legacy_auc = float(metrics.auc(fpr_l, tpr_l))
        legacy_ap = float(metrics.average_precision_score(legacy_labels, legacy_sims))
    except Exception:
        legacy_auc = 0.5
        legacy_ap = float(np.mean(legacy_labels))

    legacy_result = {
        'label': 'Random negative (legacy control)',
        'n_pairs': len(legacy_labels),
        'n_pos': int(legacy_labels.sum()),
        'n_neg': int(len(legacy_labels) - legacy_labels.sum()),
        'auc_roc': legacy_auc,
        'ap': legacy_ap,
        'pos_cos_mean': float(legacy_sims[legacy_labels == 1].mean()),
        'neg_cos_mean': float(legacy_sims[legacy_labels == 0].mean()),
        'separation': float(legacy_sims[legacy_labels == 1].mean() - legacy_sims[legacy_labels == 0].mean()),
    }

    # ══════════════════════════════════════════════════════════════
    # Pre-report: quantity audit
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('Quantity Audit')
    print('=' * 60)

    for ad in [primary_adduct] + [a for a in adduct_types if a != primary_adduct]:
        result_key = 'primary' if ad == primary_adduct else f'adduct_{ad}'
        pr = pair_results.get(result_key)
        qr = query_results.get(result_key)
        if pr is None or 'error' in pr:
            continue
        n_pq = pr.get('n_queries_eligible', 0)
        n_qq = qr.get('n_eligible_queries', 0) if qr and 'error' not in qr else 0
        # Both implementations apply the same eligibility definition and
        # should agree exactly.
        match = 'OK' if n_pq == n_qq else '⚠ MISMATCH'
        print(f'  {ad:20s}: pair-builder queries={n_pq:>6d}, query-level eligible={n_qq:>6d}  {match}')

    # Overall stats
    print(f'\n  Spectra evaluated:       {n_valid:,}')
    print(f'  Unique 14-char IKs:      {len(set(iks)):,}')
    multi_spec_iks = sum(1 for ik, idxs in ik_to_indices.items() if len(idxs) >= 2)
    print(f'  IKs with ≥2 spectra:     {multi_spec_iks:,}')

    # ══════════════════════════════════════════════════════════════
    # 8. Save Results
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print('[8/8] Saving results...')

    # ── 8a. Summary table (FIX: use query_results eligible count for Recall rows) ──
    summary_rows = []
    summary_rows.append({
        'Protocol': 'Random negative (legacy)',
        'ROC-AUC': f'{legacy_result["auc_roc"]:.4f}',
        'Macro-AUC': '-',
        'AP': f'{legacy_result["ap"]:.4f}',
        'Separation': f'{legacy_result["separation"]:.4f}',
        'Directed pairs': legacy_result['n_pairs'],
        'Unique pairs': legacy_result['n_pairs'],
        'Recall@1': '-',
        'MRR': '-',
        'Queries (pair)': '-',
        'Queries (query-level)': '-',
    })

    for proto_key in ['primary'] + sorted(k for k in pair_results if k != 'primary'):
        pr = pair_results.get(proto_key)
        qr = query_results.get(proto_key)
        if pr is None or 'error' in pr:
            continue
        row = {
            'Protocol': pr['label'],
            'ROC-AUC': f'{pr["auc_roc"]:.4f}',
            'Macro-AUC': f'{pr["macro_query_auc"]:.4f}',
            'AP': f'{pr["ap"]:.4f}',
            'Separation': f'{pr["separation"]:.4f}',
            'Directed pairs': pr['n_pairs_directed'],
            'Unique pairs': pr['n_pairs_unique'],
            'Recall@1': '-',
            'MRR': '-',
            'Queries (pair)': pr['n_queries_eligible'],
            'Queries (query-level)': '-',
        }
        if qr and 'error' not in qr:
            row['Recall@1'] = f'{qr["recall@1"]:.4f}'
            row['MRR'] = f'{qr["mrr"]:.4f}'
            row['Queries (query-level)'] = qr['n_eligible_queries']
        summary_rows.append(row)

    # Print table
    print('\n' + '─' * 110)
    header = (f'{"Protocol":<35} {"ROC-AUC":>9} {"MacroAUC":>9} {"AP":>9} {"Sep":>8} '
              f'{"Pairs":>10} {"Recall@1":>9} {"MRR":>9} {"Q(pair)":>9} {"Q(query)":>9}')
    print(header)
    print('─' * 110)
    for row in summary_rows:
        print(f'{row["Protocol"]:<35} {row["ROC-AUC"]:>9} {row["Macro-AUC"]:>9} {row["AP"]:>9} '
              f'{row["Separation"]:>8} {row["Directed pairs"]:>10} '
              f'{row["Recall@1"]:>9} {row["MRR"]:>9} '
              f'{row["Queries (pair)"]:>9} {row["Queries (query-level)"]:>9}')
    print('─' * 110)
    print(f'  * Primary row = strict 10-ppm, [{args.primary_adduct}], molecule-aggregated ranking')
    print(f'  * Legacy row = random negative pairs, for historical reference only')
    print(f'  * AP = Average Precision (equivalent to PR-AUC)')
    print(f'  * Q(pair) = eligible queries in pair-builder, Q(query) = eligible queries in query-level')
    print(f'  * All results on {eval_fold} fold ({n_valid:,} spectra)')

    # ── 8b. Full JSON report ──
    report = {
        'e0_version': '2.0',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': str(ckpt_path),
        'checkpoint_sha256': ckpt_hash,
        'data': str(args.data),
        'fold': eval_fold,
        'n_spectra_evaluated': n_valid,
        'n_failed_preprocessing': n_failed,
        'model_dim': d_model,
        'config': {
            'ppm_tol': args.ppm_tol,
            'peak_hash_tol': args.peak_hash_tol,
            'primary_adduct': args.primary_adduct,
            'seed': args.seed,
            'n_bootstrap': args.n_bootstrap,
            'device': str(device),
        },
        'pair_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'pair_indices'}
            for k, v in pair_results.items()
        },
        'query_results': {
            k: v for k, v in query_results.items()
        },
        'legacy_control': legacy_result,
        'summary_table': summary_rows,
        'data_statistics': {
            'n_spectra_total': int(n_total),
            'n_unique_iks': int(len(set(iks))),
            'n_unique_smiles': int(len(set(r['smiles'] for r in records))),
            'n_iks_with_multi_spectra': multi_spec_iks,
            'n_duplicate_peaks': int(n_dup_spectra),
            'adduct_distribution': dict(Counter(adducts)),
            'precursor_mz_range': [float(pmzs.min()), float(pmzs.max())],
        },
    }

    report_path = os.path.join(args.output_dir, 'e0_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\n  Report: {report_path}')

    # ── 8c-e. Embedding manifest + embeddings + pair/query manifests ──
    if not args.reuse_embeddings:
        manifest = []
        for i, rec in enumerate(records):
            manifest.append({
                'spectrum_id': rec['identifier'],
                'embedding_idx': i,
                'inchikey_14': rec['inchikey'],
                'smiles': rec['smiles'],
                'precursor_mz': float(rec['precursor_mz']),
                'adduct': rec['adduct'],
                'polarity': rec['polarity'],
                'fold': rec['fold'],
                'peak_hash': rec['peak_hash'],
                'n_peaks': rec['n_peaks'],
                'ce': rec['ce'],
                'instrument': rec['instrument'],
            })

        manifest_path = os.path.join(args.output_dir, 'e0_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'  Manifest: {manifest_path}')

        emb_path = os.path.join(args.output_dir, 'e0_embeddings.npy')
        np.save(emb_path, emb_array)
        print(f'  Embeddings: {emb_path} ({emb_array.shape})')

    # Pair manifest: keep JSON small and store large numeric arrays in NPZ.
    pair_manifest = {}
    pair_npz_payload = {}
    for k, pr in pair_results.items():
        if 'pair_indices' in pr:
            pi, pj, pl, ps, qids = pr['pair_indices']
            pair_manifest[k] = {
                'n_pairs': len(pl),
                'n_eligible_queries': pr.get('n_queries_eligible', 0),
                'npz_prefix': k,
            }
            pair_npz_payload[f'{k}__pair_i'] = pi.astype(np.int32, copy=False)
            pair_npz_payload[f'{k}__pair_j'] = pj.astype(np.int32, copy=False)
            pair_npz_payload[f'{k}__labels'] = pl.astype(np.int8, copy=False)
            pair_npz_payload[f'{k}__scores'] = ps.astype(np.float32, copy=False)
            pair_npz_payload[f'{k}__query_ids'] = qids.astype(np.int32, copy=False)
    pair_manifest_path = os.path.join(args.output_dir, 'e0_pair_manifest.json')
    with open(pair_manifest_path, 'w') as f:
        json.dump(pair_manifest, f, indent=2)
    print(f'  Pair manifest: {pair_manifest_path}')
    pair_npz_path = os.path.join(args.output_dir, 'e0_pair_arrays.npz')
    np.savez_compressed(pair_npz_path, **pair_npz_payload)
    print(f'  Pair arrays: {pair_npz_path}')

    # Query manifest
    query_manifest_path = os.path.join(args.output_dir, 'e0_query_manifest.json')
    with open(query_manifest_path, 'w') as f:
        json.dump(query_results, f, indent=2)
    print(f'  Query manifest: {query_manifest_path}')

    print('\n' + '=' * 60)
    print('E0 baseline evaluation complete (v2).')
    print(f'All outputs in: {args.output_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
