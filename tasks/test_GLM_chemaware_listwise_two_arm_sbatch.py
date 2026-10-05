"""Static contracts for run_GLM_chemaware_listwise_two_arm.sbatch.

Guards the pre-registered two-arm experiment chain of
docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md:
- cluster header discipline (gpus=1, no partition, no mem);
- the 2347152 bug class: no comment line may follow a backslash-continued
  command line;
- frozen hyperparameters and sealed V16 relation anchors appear exactly as
  pre-registered;
- both arms are trained by ONE loop whose only per-arm variables are the arm
  name and the output directory (structural identity guarantee);
- selection/confirmation/arm-pair/GNPS/baseline invocations and the reported
  G1/G2/G3 gates match the pre-registration.
"""
from __future__ import annotations

import re
from pathlib import Path

SBATCH = Path(__file__).resolve().parent / "run_GLM_chemaware_listwise_two_arm.sbatch"
RECOVERY = (
    Path(__file__).resolve().parent
    / "resume_GLM_chemaware_listwise_two_arm_gnps_finalize.sbatch"
)


def test_cluster_header_discipline() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "--mem" not in text


def test_no_comment_after_backslash_continuation() -> None:
    lines = SBATCH.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.rstrip().endswith("\\"):
            stripped = lines[index + 1].lstrip()
            assert not stripped.startswith("#"), (
                f"comment splices a continued command at line {index + 2}: "
                f"{lines[index + 1]!r}"
            )


def test_frozen_hyperparameters_and_anchors() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    frozen = [
        "--tau 0.1",
        "--temperature 0.05",
        "--delta-main 0.05",
        "--delta-contested 0.025",
        "--maximum-candidates-per-query 8",
        "--maximum-reference-spectra-per-molecule 3",
        "--batch-size 1",
        "--max-steps 3600",
        "--save-every-n-steps 900",
        "--maximum-spectra-per-batch 32",
        "--lr 2e-6",
        "--weight-decay 0",
        "--seed 3407",
        "--expected-qualified-relations 1684",
        "--expected-admitted-relations 1202",
        "--expected-uncontested-relations 602",
        "--expected-recovered-relations 600",
        "--expected-contested-relations 482",
    ]
    for fragment in frozen:
        assert text.count(fragment) == 1, fragment


def test_both_arms_share_one_identical_training_loop() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    loop = re.search(
        r"for ARM in arm1 arm2; do\n(.*?)\ndone\n", text, re.DOTALL,
    )
    assert loop is not None, "the two arms must be trained by one shared loop"
    body = loop.group(1)
    assert '--arm "$ARM"' in body
    assert '--output "$OUT/${ARM}_training"' in body
    assert "--expected-shared-sha256" in body and "--expected-nonzero-margins" in body
    # No other per-arm differentiation is allowed inside the loop.
    differentiated = set(re.findall(r"\$\{?ARM\}?", body))
    assert differentiated <= {"$ARM", "${ARM}"}


def test_initialization_is_protected_phase_a_for_both_arms() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "chemaware_phasea_max_boundary_step2000.ckpt" in text
    assert "--official-checkpoint \"$PHASEA\"" in text
    assert "INIT_CKPT" not in text
    assert "step-750" not in text and "step-*750" not in text


def test_selection_and_paired_references() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.count("GLM_select_chemaware_listwise_shared_step.py") == 2
    for step in (900, 1800, 2700, 3600):
        assert f"--expected-step {step}" in text
        assert f'--checkpoint "arm1_step{step}=' in text
        assert f'--checkpoint "arm2_step{step}=' in text
    assert "shared_step_selection.arm1_checkpoint.txt" in text
    assert "shared_step_selection.arm2_checkpoint.txt" in text
    assert text.count("--paired-reference phaseA_2pp") == 2
    assert text.count("--paired-reference arm1_selected") == 2
    assert "--formula-role 2" in text and "--formula-role 3" in text
    # Every evaluator checkpoint must use its required argparse flag.
    for fragment in (
        '"official=$OFFICIAL"', '"phaseA_2pp=$PHASEA"',
        '"arm1_selected=$ARM1_CKPT"', '"arm2_selected=$ARM2_CKPT"',
    ):
        for line in (row for row in text.splitlines() if fragment in row):
            assert "--checkpoint" in line


def test_gnps_chain_and_spectrum_free_baseline() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert text.count("prepare_official_embedding_checkpoint.py") == 3
    # GNPS encoding runs once per model through a shared loop.
    assert text.count("encode_gnps_gold_silver_10ppm_checkpoint.py") == 1
    assert "for MODEL in arm1_selected arm2_selected phasea; do" in text
    assert text.count("evaluate_gnps_gold_silver_10ppm_embeddings.py") == 3
    assert "_embeddings.npy" not in text
    assert text.count("_embeddings.npz") == 7
    for output in (
        "gnps_arm1_vs_phasea", "gnps_arm2_vs_phasea", "gnps_arm2_vs_arm1",
    ):
        assert f'--output "$OUT/{output}"' in text
        assert f'cp -a "$OUT/{output}"' in text
    assert "GLM_spectrum_free_candidate_baseline.py" in text
    assert "--graph-panel" in text and "--manifest-panel" in text
    for panel in ("role2_selection", "role3_confirmation",
                  "gnps_identity_disjoint", "gnps_formula_disjoint"):
        assert panel in text


def test_decision_gates_are_reported_not_enforced() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "G1_objective_benefit_arm1" in text
    assert "G2_chemical_attribution_arm2_vs_arm1" in text
    assert "G3_protection_role3_regression" in text
    assert "-0.005" in text
    assert '"arm1_confirmation_ci_lower"' in text
    assert '"arm2_confirmation_ci_lower"' in text
    assert "ci_upper" not in text
    assert "reported, never enforced" in text
    assert "GLM_LISTWISE_TWO_ARM_DECISION_SUMMARY" in text


def test_sealed_provenance_and_no_retired_route() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "run_2347164/evidence_class_diagnostic_v2.json" in text
    assert "minimum-launch-boundaries" not in text
    assert "1000/500 gates" not in text or "retired" in text
    assert "build_chemaware_layered_10k_native_pool.py" not in text
    assert "GLM_build_chemaware_v16_dynamic_triplets.py \\" not in text
    assert 'ln "$OUT/listwise_pool/groups.npz"' in text
    assert 'ln "$ARM1_CKPT"' in text and 'ln "$ARM2_CKPT"' in text


def test_preflight_contracts() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "python -m py_compile" in text
    # Contract tests are a LOCAL discipline; the server dreams env has no
    # pytest and the sbatch must not invoke it there.
    assert "pytest" not in text.replace(
        "the server dreams environment intentionally carries no pytest", "",
    )
    assert "Output collision" in text
    assert "set -euo pipefail" in text


def test_gnps_recovery_reuses_training_and_encoding() -> None:
    text = RECOVERY.read_text(encoding="utf-8")
    assert "run_2347317" in text
    assert "GLM_train_chemaware_listwise.py" not in text
    assert "encode_gnps_gold_silver_10ppm_checkpoint.py" not in text
    assert "phasea_embeddings.npy" in text
    assert "arm1_selected_embeddings.npy" in text
    assert "arm2_selected_embeddings.npy" in text
    assert text.count("evaluate_gnps_gold_silver_10ppm_embeddings.py") == 2
    assert '"training_repeated": False' in text
    assert '"embedding_encoding_repeated": False' in text
    assert "protected_corpus_recovery_staging" in text
    assert text.count("#SBATCH --gpus=1") == 1
    assert "#SBATCH --mem" not in text
    assert "Incomplete protected corpus already exists" in text
    assert "set -euo pipefail" in text
