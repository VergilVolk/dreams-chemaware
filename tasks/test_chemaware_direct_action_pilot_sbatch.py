"""Static cost and causal contracts for the ChemAware direct-action pilot."""
import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    body = (ROOT / "tasks/run_chemaware_direct_action_views_pilot.sbatch").read_text()
    trainer = (ROOT / "tasks/train_chemaware_iceberg_direct_shared.py").read_text()
    assert "#SBATCH --gpus=1" in body
    assert not any(line.startswith("#SBATCH --array=") for line in body.splitlines())
    assert "set -euo pipefail" in body
    assert "direct_projected_guarded" in body
    assert "--maximum-action-gradient-ratio 0.25" in body
    assert "--lambda-margin-floor 2 --lambda-preserve 20" in body
    assert "--max-safety-identities 512" in body
    assert "--lambda-consistency 0" in body and "--lambda-peak-contrast 0" in body
    clean = body.index("run_arm clean_duplicate")
    correct = body.index("run_arm correct_synthetic")
    stage1 = body.index("--stage primary")
    swapped = body.index("run_arm candidate_swapped")
    permuted = body.index("run_arm peak_permuted")
    full = body.index("--stage full")
    assert clean < correct < stage1 < swapped < permuted < full
    assert "pass_to_matched_controls" in body
    assert "STOPPED_EARLY" in body
    assert '[[ ! -e "$ROOT" ]]' in body
    assert "COMMON_ARGS=(" in body
    assert '--output "$ROOT/cpu_preflight" --preflight-only' in body
    assert 'test -f "$out/COMPLETE.json"' in body
    assert "trap '" in body and "stopped before completion" in body
    assert "[example.query_row for example in action_examples]" in trainer
    assert "[example.query_row for example in safety_examples]" in trainer
    assert "missing_training_rows" in trainer
    assert "validate_teacher_arrays(" in trainer
    assert "validate_clean_data_and_embedding_cache(" in trainer
    assert "report_text = json.dumps(report" in trainer
    parser_tree = ast.parse(trainer)
    trainer_options = {
        argument.value
        for node in ast.walk(parser_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args[:1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }
    # argparse.BooleanOptionalAction registered for --amp also creates --no-amp.
    assert "--amp" in trainer_options
    trainer_options.add("--no-amp")
    common = body[body.index("COMMON_ARGS=("):body.index(")\n\n# Exercise")]
    invoked_trainer_options = set(re.findall(r"--[a-z][a-z0-9-]*", common)) | {
        "--arm", "--output", "--preflight-only",
    }
    unknown = invoked_trainer_options - trainer_options
    assert not unknown, f"SBATCH passes unrecognized trainer options: {sorted(unknown)}"
    for module in (
        "audit_chemaware_iceberg_synthetic_embedding.py",
        "chemaware_iceberg_direct_core.py", "chemaware_iceberg_peak_action_core.py",
        "chemaware_direct_training_core.py", "chemaware_direct_action_core.py",
        "chemaware_retrieval_graph.py", "chemaware_shared_v2_core.py",
        "noise_final_core.py", "train_e1_identity.py",
        "train_noise_final_e4a_direct_augmentation.py",
        "train_noise_final_r2_shared_encoder.py", "chemaware_frozen_prefix_cache.py",
    ):
        assert (ROOT / "tasks" / module).is_file(), f"missing trainer dependency: {module}"
    print("PASS: direct-action pilot is single-GPU, staged, guarded, and fail-closed")


if __name__ == "__main__":
    main()
