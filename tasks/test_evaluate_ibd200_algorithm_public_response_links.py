from pathlib import Path

import numpy as np

from evaluate_ibd200_algorithm_public_response_links import benjamini_hochberg, parse_score_spec


def test_parse_score_spec_preserves_windows_drive_path():
    method, path, field = parse_score_spec("integrated=D:/DreaMS/result.npz:integrated_score")
    assert method == "integrated"
    assert path == Path("D:/DreaMS/result.npz")
    assert field == "integrated_score"


def test_top1_tie_contract_is_candidate_id_order():
    scores = np.asarray([0.5, 0.5, 0.2])
    candidates = np.asarray(["B", "A", "C"])
    indices = np.flatnonzero(np.isfinite(scores))
    order = np.lexsort((candidates[indices], -scores[indices]))
    assert candidates[indices[order[0]]] == "A"


def test_benjamini_hochberg_is_monotone_in_rank():
    adjusted = benjamini_hochberg(np.asarray([0.04, 0.001, 0.03]))
    np.testing.assert_allclose(adjusted, [0.04, 0.003, 0.04])
