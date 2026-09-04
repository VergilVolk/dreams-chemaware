from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from build_bioaware_context_evidence_tensor import build_one


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate_path = root / "candidates.csv.gz"
        path_path = root / "paths.csv.gz"
        output = root / "out"
        pd.DataFrame([
            {"query_id": "q", "candidate_id": "a", "spectral_score": 0.7, "truth_candidate_id": "a", "truth_formula": "F"},
            {"query_id": "q", "candidate_id": "b", "spectral_score": 0.6, "truth_candidate_id": "a", "truth_formula": "F"},
        ]).to_csv(candidate_path, index=False)
        base = {
            "query_id": "q", "candidate_id": "a", "seed_compound_id": "s", "seed_query_id": "sq",
            "seed_score": 0.9, "contribution": 0.2, "source_side_complete": False,
            "source_side_completeness": 0.5, "target_side_completeness": 1.0,
            "missing_source_signature": "x", "curated_direction_supported": False,
            "curated_direction_conflicted": False, "candidate_specificity": 1.0,
            "specificity_weighted_contribution": 0.2,
        }
        pd.DataFrame([
            {**base, "reaction_id": 1},
            {**base, "reaction_id": 2, "contribution": 0.1, "specificity_weighted_contribution": 0.1},
        ]).to_csv(path_path, index=False)
        report = build_one(candidate_path, path_path, output, "toy")
        assert report["raw_paths"] == 2 and report["dependency_collapsed_edges"] == 1
        candidates = pd.read_csv(output / "toy__candidates.csv.gz")
        edges = pd.read_csv(output / "toy__edges.csv.gz")
        labels = pd.read_csv(output / "toy__labels.csv.gz")
        assert not any("truth" in column for column in candidates.columns)
        assert not any("truth" in column for column in edges.columns)
        assert "truth_candidate_id" in labels.columns
        assert int(candidates.loc[candidates.candidate_id == "a", "context_edges"].iloc[0]) == 1
    print("[test_bioaware_context_evidence_tensor] PASS")


if __name__ == "__main__":
    main()
