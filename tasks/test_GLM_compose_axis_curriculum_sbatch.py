"""Static contracts for the GLM composition orchestrator.

Locks the pre-registered design so a later edit cannot silently change the
experiment: three arms from one warm start, the cluster's submission policy,
the fail-closed floors and hash pin, the single-universe role-2 readout, the
axis contrast, the GNPS dual-panel chain, the additivity verdict and the
secondary fold-0 stage that must never abort the primary result.
"""
from __future__ import annotations

from pathlib import Path

SBATCH = Path(__file__).with_name("run_GLM_compose_axis_curriculum.sbatch")


def text() -> str:
    return SBATCH.read_text(encoding="utf-8")


def test_cluster_policy() -> None:
    body = text()
    assert "#SBATCH --gpus=3" in body
    assert "--partition" not in body, "the cluster forbids an explicit partition"
    assert "--mem" not in body, "the cluster forbids an explicit memory request"
    assert "conda activate dreams" in body


def test_every_stage_runs_inside_the_allocation() -> None:
    body = text()
    assert 'CUDA_VISIBLE_DEVICES:-' in body
    assert "${#ALLOCATED_GPUS[@]} != 3" in body
    for tool in ("GLM_build_compose_axis_curriculum.py",
                 "GLM_summarize_compose_additivity.py",
                 "train_noise_dreams_native_residual_stage2.py",
                 "evaluate_chemaware_v2_direct_triplet.py",
                 "select_chemaware_residual_checkpoint.py",
                 "encode_gnps_gold_silver_10ppm_checkpoint.py",
                 "evaluate_gnps_gold_silver_10ppm_embeddings.py",
                 "evaluate_noise_dreams_native.py"):
        assert f"tasks/{tool}" in body, tool


def test_contracts_before_gpu_work_and_warm_start_pin() -> None:
    body = text()
    assert "tasks/test_GLM_build_compose_axis_curriculum.py" in body
    assert "tasks/test_GLM_summarize_compose_additivity.py" in body
    assert "tasks/test_GLM_compose_axis_curriculum_sbatch.py" in body
    assert "py_compile" in body
    assert 'PHASEA_SHA="a8428329' in body
    assert "GLM_COMPOSE_WARM_START_DRIFT" in body
    assert "sha256sum" in body


def test_both_curricula_are_built_from_the_chemistry_champion() -> None:
    body = text()
    assert body.count("--axis-balance balanced") == 1
    assert body.count("--axis-balance advantage-max") == 1
    assert body.count('--warm-start-checkpoint "$PHASEA"') >= 2
    assert body.count('--expected-warm-start-sha256 "$PHASEA_SHA"') == 2


def test_floor_gate_stops_before_training() -> None:
    body = text()
    assert "MIN_AXIS_QUERIES=60" in body
    assert "MIN_TOTAL_QUERIES=200" in body
    assert "GLM_COMPOSE_FLOOR_GATE" in body
    assert "axis_selected_queries" in body
    assert "positive_deficit" in body and "negative_excess" in body
    assert "sys.exit(0 if ok else 6)" in body


def test_three_native_arms_share_one_warm_start_and_one_dose() -> None:
    body = text()
    assert 'train_arm "${ALLOCATED_GPUS[0]}" "$BUILD_A" targeted balanced' in body
    assert 'train_arm "${ALLOCATED_GPUS[1]}" "$BUILD_A" control control' in body
    assert 'train_arm "${ALLOCATED_GPUS[2]}" "$BUILD_N" targeted advantage_max' in body
    assert body.count("--lr 1e-6") == 1, "one shared learning rate for all arms"
    assert "--max-epochs 1" in body
    assert body.count("--warm-start-checkpoint \"$PHASEA\"") >= 3
    for name in ("balanced", "control", "advantage_max"):
        assert f"$TRAIN/{name}/final.ckpt" in body


def test_single_universe_role2_readout_and_axis_contrast() -> None:
    body = text()
    for name in ("official=", "phasea_2pp=", "noise_stage1=",
                 "compose_balanced=", "compose_advantage_max=", "compose_control="):
        assert f'--checkpoint "{name}' in body, name
    assert "--paired-reference phasea_2pp --formula-role 2" in body
    assert "--paired-reference compose_advantage_max --formula-role 2" in body
    assert "--base-name phasea_2pp --require-positive-formula-ci" in body
    assert "GLM_COMPOSE_ROLE2_STOP" in body


def test_role3_protection_is_gated_on_selection() -> None:
    body = text()
    assert 'if [[ "$SELECTED_STEP" != "0" ]]; then' in body
    assert "--formula-role 3" in body
    assert "confirmation_triplet_evidence.npz" in body


def test_gnps_dual_panel_chain() -> None:
    body = text()
    assert "--label official" in body
    loop_list = ("for pair in phasea_2pp noise_stage1 compose_balanced "
                 "compose_advantage_max compose_control; do")
    assert loop_list in body, "every champion must be encoded on GNPS"
    assert '--label "$pair"' in body
    for call in ("evaluate_gnps phasea_2pp compose_balanced compose_balanced_vs_phasea",
                 "evaluate_gnps phasea_2pp compose_control compose_control_vs_phasea",
                 "evaluate_gnps compose_advantage_max compose_balanced "
                 "compose_balanced_vs_advantage_max",
                 "evaluate_gnps official compose_balanced compose_balanced_vs_official",
                 "evaluate_gnps official phasea_2pp phasea_vs_official",
                 "evaluate_gnps official noise_stage1 noise_stage1_vs_official"):
        assert call in body, call
    assert '--output "$GNPS/${name}"' in body
    assert "--bootstrap-resamples 10000" in body


def test_additivity_verdict_receives_every_required_report() -> None:
    body = text()
    assert "GLM_summarize_compose_additivity.py" in body
    assert '--role2 "$EVAL/role2_composition.json"' in body
    assert '--role2-axis "$EVAL/role2_axis_contrast.json"' in body
    assert "ROLE3_ARGS" in body
    assert '--output "$ADD"' in body


def test_fold0_secondary_cannot_abort_the_primary_result() -> None:
    body = text()
    marker = body.index("secondary: corrected-graph fold-0")
    tail = body[marker:]
    assert "set +e" in tail and "set -e" in tail
    assert "GLM_COMPOSE_FOLD0_SECONDARY_FAILED" in tail
    assert "fold0_secondary" in body


def test_provenance_and_status_are_written() -> None:
    body = text()
    assert "SHA256SUMS" in body
    assert "run_status.json" in body
    assert "COMPLETE: GLM composition experiment" in body


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("GLM composition sbatch contracts passed")
