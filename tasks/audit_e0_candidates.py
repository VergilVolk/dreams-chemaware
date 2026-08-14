"""Audit candidate-set difficulty for the cached E0 evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values):
    return {
        'min': int(np.min(values)),
        'p10': float(np.percentile(values, 10)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90)),
        'max': int(np.max(values)),
        'mean': float(np.mean(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir', default='data/validation/e0_baseline')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    with (output_dir / 'e0_manifest.json').open() as f:
        manifest = json.load(f)
    inchikeys = np.array([row['inchikey_14'] for row in manifest])
    arrays = np.load(output_dir / 'e0_pair_arrays.npz')

    results = {}
    prefixes = sorted({key.split('__', 1)[0] for key in arrays.files})
    for prefix in prefixes:
        query_ids = arrays[f'{prefix}__query_ids']
        pair_j = arrays[f'{prefix}__pair_j']
        labels = arrays[f'{prefix}__labels']
        n_queries = int(query_ids.max()) + 1

        pair_counts = np.bincount(query_ids, minlength=n_queries)
        positive_counts = np.bincount(
            query_ids, weights=labels, minlength=n_queries).astype(int)
        negative_counts = pair_counts - positive_counts
        candidate_molecules = np.empty(n_queries, dtype=int)
        negative_molecules = np.empty(n_queries, dtype=int)
        offsets = np.concatenate([[0], np.cumsum(pair_counts)])
        for query_id in range(n_queries):
            start, end = offsets[query_id], offsets[query_id + 1]
            candidate_iks = inchikeys[pair_j[start:end]]
            candidate_molecules[query_id] = len(set(candidate_iks))
            negative_molecules[query_id] = max(
                candidate_molecules[query_id] - 1, 0)

        results[prefix] = {
            'n_queries': n_queries,
            'pairs_per_query': summarize(pair_counts),
            'positive_spectra_per_query': summarize(positive_counts),
            'negative_spectra_per_query': summarize(negative_counts),
            'candidate_molecules_per_query': summarize(candidate_molecules),
            'negative_molecules_per_query': summarize(negative_molecules),
            'fraction_one_negative_molecule': float(
                np.mean(negative_molecules == 1)),
            'fraction_at_most_two_negative_molecules': float(
                np.mean(negative_molecules <= 2)),
        }

    output_path = output_dir / 'e0_candidate_audit.json'
    with output_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
