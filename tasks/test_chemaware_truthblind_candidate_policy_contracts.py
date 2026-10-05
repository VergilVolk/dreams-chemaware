from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.index(f"def {next_name}(", start)
    return source[start:end]


def main() -> None:
    core = (ROOT / "tasks/chemaware_truthblind_candidate_core.py").read_text(encoding="utf-8")
    scorer = (ROOT / "tasks/audit_chemaware_mass_kernel_embedding.py").read_text(encoding="utf-8")
    audit = (ROOT / "tasks/audit_chemaware_orthogonal_rule_residual_policy.py").read_text(encoding="utf-8")
    replay = (ROOT / "tasks/replay_chemaware_orthogonal_policy_truthblind.py").read_text(encoding="utf-8")
    sbatch = (ROOT / "tasks/run_chemaware_truthblind_candidate_policy.sbatch").read_text(encoding="utf-8")

    build = function_body(core, "build_truthblind_candidate_features", "rank_candidates_from_utility")
    assert "labels" not in build
    assert "old_rank" not in build
    assert "query_ik14" not in build
    score = function_body(scorer, "score_queries_truthblind", "score_queries")
    assert "molecule_label" not in score
    assert "old_rank" not in score
    assert '"formula"' not in score

    assert '"schema": "chemaware_truthblind_candidate_policy_v1"' in audit
    assert '"truth_fields_accepted_by_inference": []' in audit
    assert '"mechanism_ablation": mechanism_ablation' in audit
    assert '"same_feature_direct"' in audit
    assert '"nuisance_only"' in audit
    assert '"no_abstention"' in audit
    candidate = (ROOT / "tasks/audit_chemaware_candidate_evidence_policy.py").read_text(encoding="utf-8")
    wrapper = function_body(candidate, "build_candidate_table", "max_only_indices")
    assert "truthblind_scored_view(scored, rule_key)" in wrapper
    assert "build_truthblind_candidate_features(scored," not in wrapper
    assert "truthblind_body" in replay
    prediction_freeze = replay.index("# Freeze predictions before opening the truth array")
    truth_open = replay.index('loaded["molecule_label"]', prediction_freeze)
    assert prediction_freeze < truth_open

    directives = [line.strip() for line in sbatch.splitlines() if line.startswith("#SBATCH")]
    assert directives.count("#SBATCH --gpus=1") == 1
    assert not any("--mem" in line for line in directives)
    assert "--array" not in sbatch
    assert "audit_chemaware_orthogonal_rule_residual_policy.py" in sbatch
    assert "replay_chemaware_orthogonal_policy_truthblind.py" in sbatch
    assert "CHEMAWARE_TRUTHBLIND_POLICY_REPLAY_PASS" in sbatch
    locked = re.findall(r"^verify_sha (\S+) ([0-9a-f]{64})$", sbatch, flags=re.MULTILINE)
    assert len(locked) == 5
    for relative, expected in locked:
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, (relative, observed, expected)
    print("PASS: ChemAware truth-blind policy and mechanism-ablation contracts")


if __name__ == "__main__":
    main()
