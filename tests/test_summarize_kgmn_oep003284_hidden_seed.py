from pathlib import Path


def test_external_summary_keeps_primary_and_claim_boundaries() -> None:
    text = Path("tasks/summarize_kgmn_oep003284_hidden_seed.py").read_text(encoding="utf-8")
    assert '"primary_arm": "author_official_intersection"' in text
    assert '"failed_primary_is_not_rescued_by_secondary": True' in text
    assert '"thresholds_tuned_on_oep003284": False' in text
    assert "open-world SOTA" in text
