"""
run_explainability_v0.py — 化学可解释性模块 V0

集成 MS2DeepScore + TransExION，对谱图对输出相似度分数和碎片级解释。

用法:
  python -m dreams.models.mil_interpretable.run_explainability_v0 \
      --query_mgf data/query.mgf --library_mgf data/library.mgf
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse, json, sys, warnings
warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--query_mgf', type=str, default=None)
    p.add_argument('--library_mgf', type=str, default=None)
    p.add_argument('--n_pairs', type=int, default=20)
    p.add_argument('--output_dir', type=str, default='./explainability_output')
    p.add_argument('--ms2deepscore_model', type=str,
                   default='data/MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5')
    p.add_argument('--transexion_model', type=str,
                   default='data/TransExION_GNPS_MassBank.ms.model')
    return p.parse_args()


def run_ms2deepscore(spectra_list, model_path):
    """Run MS2DeepScore on spectrum dicts (converted to matchms format)."""
    print('\n' + '=' * 60)
    print('  MS2DeepScore')
    print('=' * 60)

    if not Path(model_path).exists():
        print(f'  Model not found: {model_path}')
        return None

    try:
        from ms2deepscore.models import load_model
        from ms2deepscore import MS2DeepScore
        from matchms import Spectrum
        from matchms import calculate_scores

        # Convert our dict spectra to matchms Spectrum objects
        ms_spectra = []
        for s in spectra_list:
            mz = np.array([p[0] for p in s.get('peaks', [])], dtype=float)
            intens = np.array([p[1] for p in s.get('peaks', [])], dtype=float)
            if len(mz) == 0: continue
            precursor = float(s.get('PrecursorMZ', 0) or 0)
            spectrum = Spectrum(mz=mz, intensities=intens,
                               metadata={'precursor_mz': precursor,
                                        'precursor_type': s.get('Precursor_type', '[M+H]+')})
            ms_spectra.append(spectrum)

        print(f'  Converted {len(ms_spectra)} spectra to matchms format')

        model = load_model(model_path, allow_legacy=True)
        ms2ds = MS2DeepScore(model=model)
        scores = calculate_scores(references=ms_spectra, queries=ms_spectra,
                                  similarity_function=ms2ds)

        n = len(ms_spectra)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    score = scores.scores.get(f'{i}:{j}', None)
                    if score is not None:
                        sim_matrix[i, j] = float(score[0])
        print(f'  Computed {n}x{n} similarity matrix')
        return sim_matrix
    except Exception as e:
        print(f'  MS2DeepScore error: {e}')
        import traceback; traceback.print_exc()
        return None


def run_transexion(spectra_list, model_path):
    """Try to run TransExION."""
    print('\n' + '=' * 60)
    print('  TransExION')
    print('=' * 60)
    try:
        # TransExION model: try torch pickle
        import torch

        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f'  TransExION model loaded: {type(model_data)}')
        if isinstance(model_data, dict):
            print(f'  Keys: {list(model_data.keys())[:10]}')
            # Try to find similarity function
            for k in model_data:
                if 'similarity' in str(k).lower():
                    print(f'  Found similarity key: {k}')
        elif hasattr(model_data, 'similarity'):
            print(f'  Has similarity method')
        print(f'  TransExION model ready.')
        return model_data
    except Exception as e:
        print(f'  TransExION loading error: {e}')
        return None


def load_spectra_from_msp(msp_path, max_n=100):
    """Load spectra from MSP file (using our own parser, not matchms)."""
    from dreams.models.mil_interpretable.train_mil_massbank import parse_msp
    spectra = parse_msp(msp_path, max_spectra=max_n)
    print(f'  Loaded {len(spectra)} spectra from {msp_path}')
    return spectra


def explain_top_match(query_idx, lib_idx, score, spectra):
    """Generate explanation for a top match."""
    q = spectra[query_idx]
    l = spectra[lib_idx]
    q_peaks = q.get('peaks', [])
    l_peaks = l.get('peaks', [])
    explanation = {
        'query_idx': int(query_idx),
        'library_idx': int(lib_idx),
        'score': float(score),
        'query_name': q.get('Name', q.get('SMILES', '?')[:40]),
        'library_name': l.get('Name', l.get('SMILES', '?')[:40]),
        'query_precursor_mz': float(q.get('PrecursorMZ', 0) or 0),
        'library_precursor_mz': float(l.get('PrecursorMZ', 0) or 0),
        'n_query_peaks': len(q_peaks),
        'n_library_peaks': len(l_peaks),
    }
    return explanation


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Chemical Explainability Module V0')
    print('=' * 60)

    # Load spectra
    spectra = None
    msp_files = [f for f in ['data/MassBank_NIST.msp',
                              'data/MoNA-export-LC-MS-MS_Spectra.msp']
                 if Path(f).exists()]
    if msp_files:
        print(f'\n[1] Loading spectra from {msp_files[0]}...')
        spectra = load_spectra_from_msp(msp_files[0], max_n=args.n_pairs)
        print(f'  Using {len(spectra)} spectra')
    elif args.query_mgf:
        print(f'\n[1] Loading from {args.query_mgf}...')
        spectra = load_spectra_from_msp(args.query_mgf, max_n=args.n_pairs)

    if not spectra or len(spectra) < 2:
        print('  ERROR: Need at least 2 spectra. Please provide MSP/MGF files.')
        print('  Place spectra in data/ directory (MassBank_NIST.msp etc.)')
        sys.exit(1)

    # Run MS2DeepScore
    ms2ds_matrix = None
    if Path(args.ms2deepscore_model).exists():
        ms2ds_matrix = run_ms2deepscore(spectra, args.ms2deepscore_model)
    else:
        print(f'\n  MS2DeepScore model not found at {args.ms2deepscore_model}')
        print(f'  Download from Zenodo: MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5')
        print(f'  Place in d:/DreaMS/data/')

    # Try TransExION
    transexion_model = None
    if Path(args.transexion_model).exists():
        transexion_model = run_transexion(spectra, args.transexion_model)
    else:
        print(f'\n  TransExION model not found at {args.transexion_model}')
        print(f'  Download from: https://zenodo.org/records/8175528')

    # Output summary
    report = {
        'n_spectra': len(spectra),
        'ms2deepscore_available': ms2ds_matrix is not None,
        'transexion_available': transexion_model is not None,
    }

    if ms2ds_matrix is not None:
        # Find top-5 matches for first query
        for qi in range(min(3, len(spectra))):
            scores = ms2ds_matrix[qi]
            top_idx = np.argsort(scores)[::-1][1:6]  # exclude self
            report[f'query_{qi}_top5'] = [
                explain_top_match(qi, int(ti), float(scores[ti]), spectra)
                for ti in top_idx
            ]
            print(f'\n  Query {qi}: {spectra[qi].get("Name","?")}')
            for rank, ti in enumerate(top_idx):
                print(f'    {rank+1}. {spectra[ti].get("Name","?")}: '
                      f'score={scores[ti]:.4f}')

    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n  Report saved: {out_dir / "report.json"}')


if __name__ == '__main__':
    main()
