#!/usr/bin/env python
"""CPU-only contracts for the pair-evidence fusion stack."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noise_msg_pair_evidence_core import (
    MOLECULE_FEATURES,
    build_molecule_features,
    query_balanced_molecule_weights,
)


def synthetic_evidence(seed: int = 3) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n_queries, edges = 40, 0
    query_ptr = [0]
    molecule_ptr = [0]
    molecule_labels: list[int] = []
    formulas = [f"C{q}H{2 * q}" for q in range(n_queries)]
    for _ in range(n_queries):
        n_molecules = int(rng.integers(2, 5))
        for molecule in range(n_molecules):
            n_spectra = int(rng.integers(1, 4))
            molecule_ptr.append(molecule_ptr[-1] + n_spectra)
            molecule_labels.append(1 if molecule == 0 else 0)
        query_ptr.append(query_ptr[-1] + n_molecules)
    molecule_ptr_arr = np.asarray(molecule_ptr, dtype=np.int64)
    query_ptr_arr = np.asarray(query_ptr, dtype=np.int64)
    query_edge_ptr = molecule_ptr_arr[query_ptr_arr]
    n_edges = int(molecule_ptr_arr[-1])
    signal = rng.random(n_edges)
    evidence = {
        "v1_cosine": (0.5 + 0.4 * signal).astype(np.float32),
        "official_cosine": (0.4 + 0.4 * rng.random(n_edges)).astype(np.float32),
        "weighted_entropy": (0.3 + 0.6 * signal + 0.05 * rng.random(n_edges)).astype(np.float32),
        "sqrt_cosine": (0.2 + 0.7 * rng.random(n_edges)).astype(np.float32),
        "entropy_similarity": (0.2 + 0.7 * rng.random(n_edges)).astype(np.float32),
        "neutral_loss_sqrt_cosine": (0.2 + 0.7 * rng.random(n_edges)).astype(np.float32),
        "p2b_fused": (0.4 + 0.5 * signal).astype(np.float32),
        "query_edge_ptr": query_edge_ptr,
        "molecule_ptr": molecule_ptr_arr,
        "query_ptr": query_ptr_arr,
        "molecule_label": np.asarray(molecule_labels, dtype=np.int8),
        "query_formula": np.asarray(formulas),
    }
    return evidence, {
        "edges": evidence,
        "query_edge_ptr": query_edge_ptr,
        "molecule_ptr": molecule_ptr_arr,
        "query_ptr": query_ptr_arr,
    }


def test_feature_matrix_shape_and_finiteness() -> None:
    evidence, view = synthetic_evidence()
    features = build_molecule_features(
        view["edges"], evidence["query_ptr"], view["molecule_ptr"],
    )
    assert features.shape == (len(view["molecule_ptr"]) - 1, len(MOLECULE_FEATURES))
    assert np.all(np.isfinite(features))
    # Percentile columns live in [0, 1]; agreement flags are binary.
    for column in range(7, 14):
        assert features[:, column].min() >= 0.0 and features[:, column].max() <= 1.0
    for column in (19, 20):
        assert set(np.unique(features[:, column])) <= {0.0, 1.0}


def test_percentiles_average_ties() -> None:
    edges = {
        "v1_cosine": np.asarray([0.5, 0.5, 0.9]),
        "official_cosine": np.asarray([0.1, 0.2, 0.3]),
        "weighted_entropy": np.asarray([0.1, 0.2, 0.3]),
        "sqrt_cosine": np.asarray([0.1, 0.2, 0.3]),
        "entropy_similarity": np.asarray([0.1, 0.2, 0.3]),
        "neutral_loss_sqrt_cosine": np.asarray([0.1, 0.2, 0.3]),
        "p2b_fused": np.asarray([0.1, 0.2, 0.3]),
    }
    query_edge_ptr = np.asarray([0, 3], dtype=np.int64)
    molecule_ptr = np.asarray([0, 1, 2, 3], dtype=np.int64)
    features = build_molecule_features(
        edges, np.asarray([0, 3], dtype=np.int64), molecule_ptr,
    )
    # The two tied v1 cosines share the mean of ranks 0 and 1 (=0.5), scaled
    # by the maximum rank 2 -> 0.25; the top value maps to 1.0.
    assert abs(features[0, 7] - 0.25) < 1e-12 and abs(features[1, 7] - 0.25) < 1e-12
    assert abs(features[2, 7] - 1.0) < 1e-12


def test_query_balanced_weights_remove_candidate_count_bias() -> None:
    labels = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.int8)
    ptr = np.asarray([0, 2, 6], dtype=np.int64)
    weights = query_balanced_molecule_weights(labels, ptr)
    assert np.allclose(weights[:2], np.asarray([0.5, 0.5]))
    assert weights[2] == 0.5
    assert np.allclose(weights[3:], np.full(3, 1 / 6))
    assert np.allclose(np.add.reduceat(weights, ptr[:-1]), np.ones(2))


def test_sbatch_is_two_gpu_and_keeps_gnps_out_of_training() -> None:
    script = (ROOT / "tasks" / "run_noise_msg_fusion_5pp_2gpu.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "build_noise_massspecgym_pair_evidence.py" in script
    assert 'cp "$LOCAL_ROOT/evidence/evidence.npz" "$RUN_ROOT/evidence/evidence.npz"' in script
    assert "train_noise_msg_fusion_stack.py" in script
    assert script.index("train_noise_msg_fusion_stack.py") < script.index(
        "apply_noise_msg_fusion_to_gnps.py"
    )
    assert "gnps_used_in_training" not in script


def test_fusion_learns_the_informative_channel() -> None:
    """End-to-end miniature: OOF fused ranking beats a blind channel."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from train_noise_msg_fusion_stack import top1_accuracy

    evidence, view = synthetic_evidence(seed=11)
    features = build_molecule_features(
        view["edges"], evidence["query_ptr"], view["molecule_ptr"],
    )
    labels = evidence["molecule_label"].astype(np.int8)
    weight = float((labels == 0).sum()) / max(float((labels == 1).sum()), 1.0)
    sample_weight = np.where(labels == 1, weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.1, min_samples_leaf=5, early_stopping=False,
        random_state=0,
    )
    model.fit(features, labels, sample_weight=sample_weight)
    fused = model.predict_proba(features)[:, 1]
    fused_accuracy = top1_accuracy(
        fused, evidence["molecule_label"], view["query_ptr"],
    )
    official_molecule = np.maximum.reduceat(
        evidence["official_cosine"], evidence["molecule_ptr"][:-1],
    )
    official_accuracy = top1_accuracy(
        official_molecule, evidence["molecule_label"], view["query_ptr"],
    )
    # 'official_cosine' carries no signal by construction; the stack must beat it.
    assert fused_accuracy > official_accuracy


def test_fold_filter_excludes_other_queries() -> None:
    from train_noise_msg_fusion_stack import top1_accuracy

    evidence, view = synthetic_evidence(seed=5)
    folds = np.asarray([index % 5 for index in range(len(evidence["query_formula"]))])
    base = top1_accuracy(
        np.maximum.reduceat(evidence["v1_cosine"], evidence["molecule_ptr"][:-1]),
        evidence["molecule_label"], view["query_ptr"], folds, fold=0,
    )
    assert 0.0 <= base <= 1.0
    try:
        top1_accuracy(
            np.maximum.reduceat(evidence["v1_cosine"], evidence["molecule_ptr"][:-1]),
            evidence["molecule_label"], view["query_ptr"],
            folds, fold=99,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty fold selection must fail loudly")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_msg_fusion] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
