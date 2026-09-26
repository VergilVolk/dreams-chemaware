"""Atomically protect one evaluated true-support ChemAware checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "tasks/build_chemaware_dense_true_support_native_triplets.py",
    "tasks/build_chemaware_true_support_native_triplets.py",
    "tasks/encode_chemaware_formula_role_checkpoint_rows.py",
    "tasks/build_chemaware_multicondition_max_boundary_triplets.py",
    "tasks/build_chemaware_max_boundary_native_triplets.py",
    "tasks/build_chemaware_dreams_native_triplets.py",
    "tasks/chemaware_numpy_sampling.py",
    "tasks/encode_chemaware_checkpoint_manifest_rows.py",
    "tasks/train_chemaware_weighted_native.py",
    "tasks/train_chemaware_dreams_native.py",
    "tasks/train_chemaware_specific_replay_native.py",
    "tasks/evaluate_chemaware_v2_direct_triplet.py",
    "tasks/evaluate_chemaware_full_role_native.py",
    "tasks/chemaware_v2_triplet_eval_core.py",
    "tasks/select_chemaware_residual_checkpoint.py",
    "tasks/freeze_chemaware_true_support_artifact.py",
    "tasks/test_chemaware_dense_true_support_native.py",
    "tasks/run_chemaware_dense_true_support_native.sbatch",
    "tasks/run_chemaware_true_support_native.sbatch",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--role2-evaluation", type=Path, required=True)
    parser.add_argument("--role3-evaluation", type=Path, required=True)
    parser.add_argument("--full-evaluation", type=Path, required=True)
    parser.add_argument("--triplet-report", type=Path, required=True)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--val-pool", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def row_by_name(report: dict[str, object], name: str) -> dict[str, object]:
    rows = [row for row in report["results"] if row["name"] == name]
    if len(rows) != 1:
        raise RuntimeError(f"evaluation must contain exactly one {name!r} row")
    return rows[0]


def paired_gates(
    base: dict[str, object], model: dict[str, object], paired: dict[str, object],
) -> dict[str, bool]:
    base_metrics = base["metrics"]
    metrics = model["metrics"]
    corrected = int(paired["corrected_at_1"])
    introduced = int(paired["introduced_at_1"])
    ci = list(map(float, paired["formula_cluster_bootstrap_delta_recall1_ci95"]))
    return {
        "recall1_strictly_improves": float(paired["delta_recall1"]) > 0.0,
        "formula_cluster_ci_strictly_positive": ci[0] > 0.0,
        "risk_utility_positive": corrected - 2 * introduced > 0,
        "mrr_strictly_improves": float(paired["delta_mrr"]) > 0.0,
        "recall3_nonnegative": float(metrics["recall3"]) >= float(base_metrics["recall3"]),
        "micro_auc_nonnegative": float(metrics["micro_auc"]) >= float(base_metrics["micro_auc"]),
        "macro_auc_nonnegative": float(metrics["macro_auc"]) >= float(base_metrics["macro_auc"]),
    }


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite protected artifact: {args.output}")
    inputs = {
        "checkpoint_selection.json": args.selection,
        "role2_checkpoint_evaluation.json": args.role2_evaluation,
        "role3_evaluation.json": args.role3_evaluation,
        "role2_role3_full_evaluation.json": args.full_evaluation,
        "triplet_report.json": args.triplet_report,
        "train_pool.npz": args.train_pool,
        "val_pool.npz": args.val_pool,
        "training_report.json": args.training_report,
    }
    for path in (args.checkpoint, *inputs.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    role2 = json.loads(args.role2_evaluation.read_text(encoding="utf-8"))
    role3 = json.loads(args.role3_evaluation.read_text(encoding="utf-8"))
    full = json.loads(args.full_evaluation.read_text(encoding="utf-8"))
    triplets = json.loads(args.triplet_report.read_text(encoding="utf-8"))
    selected = selection["selected"]
    selected_name = str(selected["name"])
    role2_base = row_by_name(role2, "phaseA_base")
    role2_model = row_by_name(role2, selected_name)
    role3_base = row_by_name(role3, "phaseA_base")
    role3_model = row_by_name(role3, "true_support")
    triplet_status = triplets.get("status")
    legacy_triplet_contract = bool(
        triplet_status == "CHEMAWARE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE"
        and len(triplets.get("settings", {}).get("rule_metrics", [])) == 4
    )
    dense_triplet_contract = bool(
        triplet_status == "CHEMAWARE_DENSE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE"
        and triplets.get("cache_kind") == "protected_phasea"
        and int(triplets.get("events", {}).get("unique_correction_triplets", 0)) >= 1000
        and int(triplets.get("coverage", {}).get("correction_queries", 0)) >= 200
        and int(triplets.get("coverage", {}).get("distinct_identity_false_boundaries", 0)) >= 200
        and triplets.get("sampler", {}).get("custom_sampling_weight_present") is False
    )
    provenance_gates = {
        "selection_advanced_beyond_phase_a": bool(
            selection.get("advanced_beyond_base")
            and selection.get("base_name") == "phaseA_base"
        ),
        "selected_checkpoint_matches_input": same_path(selected["checkpoint"], args.checkpoint),
        "role2_checkpoint_matches_input": same_path(role2_model["checkpoint"], args.checkpoint),
        "role3_checkpoint_matches_input": same_path(role3_model["checkpoint"], args.checkpoint),
        "role2_is_formula_role_2": int(role2.get("formula_role", -1)) == 2,
        "role3_is_formula_role_3": int(role3.get("formula_role", -1)) == 3,
        "full_evaluation_is_roles_2_and_3": list(full.get("formula_roles", [])) == [2, 3],
        "role4_never_accessed": not any(bool(report.get("outer_role_4_accessed")) for report in (role2, role3, full)),
        "triplet_status_is_supported": triplet_status in {
            "CHEMAWARE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE",
            "CHEMAWARE_DENSE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE",
        },
        "triplet_gates_passed": bool(triplets.get("gates")) and all(triplets["gates"].values()),
        "triplet_method_contract_passed": legacy_triplet_contract or dense_triplet_contract,
    }
    if not all(provenance_gates.values()):
        raise RuntimeError(f"artifact provenance gates failed: {provenance_gates}")
    role2_paired = role2_model.get("paired_vs_phaseA_base")
    role3_paired = role3_model.get("paired_vs_phaseA_base")
    if not isinstance(role2_paired, dict) or not isinstance(role3_paired, dict):
        raise RuntimeError("evaluations lack paired comparisons against Phase A")
    role2_gates = paired_gates(role2_base, role2_model, role2_paired)
    if not all(role2_gates.values()):
        raise RuntimeError(f"selected checkpoint no longer passes role-2 gates: {role2_gates}")
    role3_gates = paired_gates(role3_base, role3_model, role3_paired)
    role3_confirmed = all(role3_gates.values())

    checkpoint_name = "chemaware_true_support_native.ckpt"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_true_support_protected_", dir=args.output.parent))
    try:
        storage = link_or_copy(args.checkpoint, temporary / checkpoint_name)
        for name, source in inputs.items():
            shutil.copy2(source, temporary / name)
        source_directory = temporary / "source"
        for relative in SOURCE_FILES:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = source_directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest = {
            "status": (
                "CHEMAWARE_TRUE_SUPPORT_ROLE3_CONFIRMED_ARTIFACT_PROTECTED"
                if role3_confirmed else
                "CHEMAWARE_TRUE_SUPPORT_ROLE2_ONLY_ARTIFACT_PROTECTED"
            ),
            "claim_status": (
                "ROLE3_CONFIRMED_CANDIDATE_NOT_OUTER_TESTED"
                if role3_confirmed else "ROLE2_ADVANCE_NOT_ROLE3_CONFIRMED"
            ),
            "shared_embedding": True,
            "reranker": False,
            "distillation": False,
            "checkpoint_file": checkpoint_name,
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_storage": storage,
            "selected_name": selected_name,
            "selected_step": int(selected["step"]),
            "role2_vs_phase_a": role2_paired,
            "role3_vs_phase_a": role3_paired,
            "role2_gates": role2_gates,
            "role3_gates": role3_gates,
            "role3_confirmed": role3_confirmed,
            "outer_role4_accessed": False,
            "provenance_gates": provenance_gates,
            "triplet_status": triplet_status,
            "triplet_method_contract": (
                "dense_true_support" if dense_triplet_contract else "legacy_true_support"
            ),
            "source_files": list(SOURCE_FILES),
            "allowed_claim": (
                "The true-support ChemAware shared embedding passed frozen role-2 "
                "selection and independent role-3 confirmation."
                if role3_confirmed else
                "The true-support ChemAware shared embedding advanced on role 2 "
                "but did not pass independent role-3 confirmation."
            ),
            "forbidden_claim": (
                "Do not describe this artifact as outer-tested or as a five-point "
                "improvement unless a separate frozen evaluation establishes it."
            ),
        }
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
        protected_files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (temporary / "SHA256SUMS").write_text(
            "".join(
                f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in protected_files
            ),
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
