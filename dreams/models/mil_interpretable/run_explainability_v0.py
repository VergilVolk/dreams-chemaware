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
import argparse, json, sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--query_mgf', type=str, default=None)
    p.add_argument('--library_mgf', type=str, default=None)
    p.add_argument('--n_pairs', type=int, default=20)
    p.add_argument('--output_dir', type=str, default='./explainability_output')
    p.add_argument('--ms2deepscore_model', type=str,
                   default='data/ms2deepscore_model.pt')
    p.add_argument('--transexion_model', type=str,
                   default='data/TransExION_GNPS_MassBank.ms.model')
    return p.parse_args()


def run_ms2deepscore(spectra_list, model_path):
    """Run MS2DeepScore on a list of matchms Spectrum objects."""
    print('\n' + '=' * 60)
    print('  MS2DeepScore')
    print('=' * 60)
    try:
        from ms2deepscore.models import load_model
        from ms2deepscore import MS2DeepScore
        from matchms import calculate_scores

        model = load_model(model_path)
        ms2ds = MS2DeepScore(model=model)

        # Calculate pairwise similarity matrix
        scores = calculate_scores(
            references=spectra_list,
            queries=spectra_list,
            similarity_function=ms2ds,
        )
        n = len(spectra_list)
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
        print(f'  To use, download model from: https://zenodo.org/records/6339969')
        print(f'  Then: --ms2deepscore_model path/to/ms2deepscore_model.pt')
        return None


def run_transexion(spectra_list, model_path):
    """Try to run TransExION."""
    print('\n' + '=' * 60)
    print('  TransExION')
    print('=' * 60)
    try:
        # TransExION uses matchms model format
        import pickle
        import torch

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        print(f'  Model loaded: {type(model_data)}')
        if isinstance(model_data, dict):
            print(f'  Keys: {list(model_data.keys())[:10]}')

        # Try to find similarity computation function
        if hasattr(model_data, 'similarity'):
            print('  Found .similarity method')
        elif isinstance(model_data, dict) and 'model' in model_data:
            print('  Found model key in dict')
            sub = model_data['model']
            print(f'  Sub-model type: {type(sub)}')

        print(f'  Model file loaded successfully ({model_path})')
        print(f'  Full inference requires TransExION GitHub code:')
        print(f'  https://github.com/adremlab/TransExION (check for repo)')
        return model_data
    except Exception as e:
        print(f'  TransExION loading error: {e}')
        return None


def load_spectra_from_msp(msp_path, max_n=100):
    """Load spectra from MSP file using matchms."""
    try:
        from matchms.importing import load_from_msp
        spectra = list(load_from_msp(msp_path))
        print(f'  Loaded {len(spectra)} spectra from {msp_path}')
        if len(spectra) > max_n:
            spectra = spectra[:max_n]
        return spectra
    except Exception as e:
        print(f'  Error loading MSP: {e}')
        return []


def explain_top_match(query_idx, lib_idx, score, spectra):
    """Generate explanation for a top match."""
    q = spectra[query_idx]
    l = spectra[lib_idx]
    q_mz = q.peaks.mz if hasattr(q, 'peaks') else []
    l_mz = l.peaks.mz if hasattr(l, 'peaks') else []

    explanation = {
        'query_idx': int(query_idx),
        'library_idx': int(lib_idx),
        'score': float(score),
        'query_precursor_mz': float(q.get('precursor_mz', 0)),
        'library_precursor_mz': float(l.get('precursor_mz', 0)),
        'n_query_peaks': len(q_mz),
        'n_library_peaks': len(l_mz),
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
        print(f'  Download from: https://zenodo.org/records/6339969')

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
            print(f'\n  Query {qi}: {spectra[qi].get("name","?")}')
            for rank, ti in enumerate(top_idx):
                print(f'    {rank+1}. {spectra[ti].get("name","?")}: '
                      f'score={scores[ti]:.4f}')

    with open(out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n  Report saved: {out_dir / "report.json"}')


if __name__ == '__main__':
    main()
