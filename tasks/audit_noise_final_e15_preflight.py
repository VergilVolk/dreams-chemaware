"""Fail-closed audit of the rejected E14 implementation and frozen E15 contract."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E14_TEACHER = ROOT / "tasks/build_noise_final_e14_crossfit_p_teacher.py"
E14_TRAINER = ROOT / "tasks/train_noise_final_e4a_direct_augmentation.py"
CONTRACT_DOC = ROOT / "docs/NOISE_E14_IMPLEMENTATION_FAILURE_AND_E15_CONTRACT_20260831.md"

E15_MANDATORY_CONTRACT = {
    "multiple_corrective_actions_per_query": True,
    "harmful_actions_materialized_separately": True,
    "no_op_always_available": True,
    "corrective_and_risk_losses_are_distinct": True,
    "risk_branch_forbids_corrective_self_transfer": True,
    "within_epoch_action_recycling_forbidden": True,
    "maximum_exposure_report_required": True,
    "gradient_calibration_minimum_stratified_microbatches": 32,
    "gradient_calibration_minimum_identity_action_observations": 128,
    "branch_gradient_cosines_required": True,
    "overfit_gate_before_identity_holdout": True,
    "identity_holdout_before_formula_fold": True,
    "formula_fold_before_multifold": True,
    "P2b_forbidden": True,
    "P3_consumed": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_tokens(source: str, tokens: tuple[str, ...], label: str) -> list[str]:
    missing = [token for token in tokens if token not in source]
    if missing:
        raise RuntimeError(f"{label} audit signatures changed or disappeared: {missing}")
    return list(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/g8r_noise_final_e15_preflight.json"),
    )
    args = parser.parse_args()
    # The Markdown document is a human-readable record, not a runtime input.
    # Cluster jobs must remain reproducible when only executable task files are
    # synchronized.
    paths = (E14_TEACHER, E14_TRAINER)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    teacher_source = E14_TEACHER.read_text(encoding="utf-8")
    trainer_source = E14_TRAINER.read_text(encoding="utf-8")
    ast.parse(teacher_source)
    ast.parse(trainer_source)

    single_action = require_tokens(teacher_source, (
        "correcting = np.flatnonzero((result_rank[local] == 1) & eligible_action)",
        "best = int(correcting[np.argmax(result_margin[local, correcting])])",
        'if selected["query_index"].duplicated().any():',
        '"one_selected_action_per_query": True',
    ), "single-action collapse")
    global_filter = require_tokens(teacher_source, (
        "load_prior_safe_definitions(args)",
        "eligible_action = np.zeros(len(definitions), dtype=bool)",
        "outer_train_multifold_action_safety_filter_applied",
    ), "global action filter")
    shared_risk_loss = require_tokens(trainer_source, (
        "guided_batch = guided_batch + guided_risk_epoch[",
        "guided_loss, guided_log = guided_noise_loss(",
        'example.supervision_kind != "corrective" for example in examples',
    ), "risk-control loss")
    recycling = require_tokens(trainer_source, (
        "if guided_cursor + guided_size > len(guided_epoch):",
        "guided_rng.shuffle(guided_epoch)",
        "guided_cursor = 0",
    ), "within-epoch guided recycling")
    four_sample = require_tokens(trainer_source, (
        '"n_action": action_examples[: min(4, len(action_examples))]',
        '"safety": safety_examples[: min(4, len(safety_examples))]',
        '"p_corrective": guided_examples[: min(4, len(guided_examples))]',
        '"p_risk": guided_risk_examples[: min(4, len(guided_risk_examples))]',
    ), "four-example gradient calibration")
    scope_mismatch = require_tokens(trainer_source, (
        'args.guided_noise_policy not in {"none", "selected"}',
        'and args.guided_query_scope == "positive_deficit_errors"',
    ), "selected-scope mismatch")

    audit = {
        "status": "noise_final_e15_preflight_audit_complete",
        "formal": True,
        "legacy_e14_disposition": {
            "safe_to_continue_multifold": False,
            "route_failure_established": False,
            "implementation_failure_established": True,
            "detected_failures": {
                "single_action_collapse": single_action,
                "global_filter_erases_conditional_actions": global_filter,
                "risk_controls_share_corrective_loss": shared_risk_loss,
                "guided_examples_recycled_within_epoch": recycling,
                "gradient_calibration_uses_first_four_examples": four_sample,
                "selected_scope_does_not_apply_positive_deficit_filter": scope_mismatch,
            },
        },
        "e15_mandatory_contract": E15_MANDATORY_CONTRACT,
        "next_authorized_operation": (
            "implement and unit-test the E15 multi-action ledger, separate risk loss, "
            "bounded sampler, and stratified gradient calibration; no large training"
        ),
        "provenance": {
            "e14_teacher_sha256": sha256_file(E14_TEACHER),
            "e14_trainer_sha256": sha256_file(E14_TRAINER),
            "contract_document_sha256": (
                sha256_file(CONTRACT_DOC) if CONTRACT_DOC.is_file() else None
            ),
            "contract_document_present": CONTRACT_DOC.is_file(),
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
        "pass": True,
        "claim_limit": (
            "This audit rejects the current E14 implementation and freezes E15 engineering "
            "requirements. It is not a trained embedding result."
        ),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E15 preflight audit: {output}")
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
