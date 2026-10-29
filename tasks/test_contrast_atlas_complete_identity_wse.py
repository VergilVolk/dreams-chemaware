from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from score_contrast_atlas_complete_identity_wse import matched_fragments  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matched_fragments_is_one_to_one() -> None:
    left = np.asarray([[100.0, 0.6], [100.05, 0.4], [200.0, 0.2]], dtype=np.float32)
    right = np.asarray([[100.01, 0.7], [100.06, 0.3], [300.0, 0.1]], dtype=np.float32)
    assert matched_fragments(left, right, 0.02) == 2


def test_compact_release_preserves_full_denominator() -> None:
    directory = ROOT / "data/validation/contrast_atlas_complete_identity_wse_release_v1_20261029"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert report["data_role"] == "development_consumed"
    assert report["truth_blinding"] == {
        "response_beta_se_read": False,
        "phenotype_read": False,
        "registered_relations_read": False,
        "identity_truth_read": False,
        "candidate_admission_changed": False,
    }
    assert report["denominators"]["query_blocks"] == 768_054
    assert report["denominators"]["directed_candidate_pairs"] == 291_330
    assert report["denominators"]["zero_candidate_blocks"] == 628_925
    for filename, expected in report["outputs"].items():
        assert digest(directory / filename) == expected
    with np.load(directory / "wse_scores_aligned.npz", allow_pickle=False) as body:
        assert str(body["schema"].item()) == "contrast_atlas_complete_identity_wse_aligned_v1"
        assert len(body["query_ptr"]) == 768_055
        assert int(body["query_ptr"][-1]) == 291_330
        assert body["wse_score"].shape == (291_330,)
        assert bool(np.all(body["wse_availability"]))


def test_ibd_adjudication_keeps_claim_boundaries() -> None:
    path = ROOT / "data/validation/contrast_atlas_ibd_wse_adjudication_v2_20261029/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    findings = report["findings"]
    control = findings["wse_5asa_identity_control"]
    assert control["expected_candidate"] == "IBD200_GROUP::1547"
    assert control["expected_rank"] == 3
    assert control["wrong_top1_candidate"] == "IBD200_GROUP::1527"
    assert findings["stercobilin_like_response_direction"]["all_three_negative"] is True
    assert findings["late_mz595_pair"]["both_rank1"] is True
    assert any("not external" in boundary.lower() for boundary in report["claim_boundary"])
