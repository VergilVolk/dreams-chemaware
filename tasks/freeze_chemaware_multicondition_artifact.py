"""Atomically protect one evaluated ChemAware multi-condition checkpoint.

The artifact may be a role-2-only development advance or a role-3-confirmed
candidate.  The manifest states that distinction explicitly; neither state is
allowed to imply access to formula role 4.
"""
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
    "tasks/encode_chemaware_formula_role_checkpoint_rows.py",
    "tasks/build_chemaware_phasea_residual_consensus_triplets.py",
    "tasks/build_chemaware_multicondition_max_boundary_triplets.py",
    "tasks/build_chemaware_max_boundary_native_triplets.py",
    "tasks/build_chemaware_action_hard_native_triplets.py",
    "tasks/build_chemaware_dreams_native_triplets.py",
    "tasks/chemaware_numpy_sampling.py",
    "tasks/train_chemaware_weighted_native.py",
    "tasks/train_chemaware_dreams_native.py",
    "tasks/train_chemaware_specific_replay_native.py",
    "tasks/evaluate_chemaware_v2_direct_triplet.py",
    "tasks/evaluate_chemaware_full_role_native.py",
    "tasks/chemaware_v2_triplet_eval_core.py",
    "tasks/select_chemaware_residual_checkpoint.py",
    "tasks/freeze_chemaware_multicondition_artifact.py",
    "tasks/run_chemaware_multicondition_max_boundary.sbatch",
    "tasks/run_chemaware_phasea_residual_consensus.sbatch",
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


def row_by_name(report: dict[str, object], name: str) -> dict[str, object]:
    rows = [row for row in report["results"] if row["name"] == name]
    if len(rows) != 1:
        raise RuntimeError(f"evaluation must contain exactly one {name!r} row")
    return rows[0]


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


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
        "recall3_nonnegative": (
            float(metrics["recall3"]) >= float(base_metrics["recall3"])
        ),
        "micro_auc_nonnegative": (
            float(metrics["micro_auc"]) >= float(base_metrics["micro_auc"])
        ),
        "macro_auc_nonnegative": (
            float(metrics["macro_auc"]) >= float(base_metrics["macro_auc"])
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite protected artifact: {args.output}")
    inputs = {
        "checkpoint": args.checkpoint,
        "checkpoint_selection.json": args.selection,
        "role2_checkpoint_evaluation.json": args.role2_evaluation,
        "role3_evaluation.json": args.role3_evaluation,
        "role2_role3_full_evaluation.json": args.full_evaluation,
        "triplet_report.json": args.triplet_report,
        "train_pool.npz": args.train_pool,
        "val_pool.npz": args.val_pool,
        "training_report.json": args.training_report,
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    role2 = json.loads(args.role2_evaluation.read_text(encoding="utf-8"))
    role3 = json.loads(args.role3_evaluation.read_text(encoding="utf-8"))
    full = json.loads(args.full_evaluation.read_text(encoding="utf-8"))
    triplets = json.loads(args.triplet_report.read_text(encoding="utf-8"))
    selected = selection["selected"]
    selected_name = str(selected["name"])

    provenance_gates = {
        "selection_advanced_beyond_phase_a": bool(
            selection.get(
                "advanced_beyond_base", selection.get("advanced_beyond_stage1")
            )
            and selection.get("base_name") == "phaseA_base"
        ),
        "selected_checkpoint_matches_input": same_path(
            selected["checkpoint"], args.checkpoint,
        ),
        "role2_is_formula_role_2": int(role2.get("formula_role", -1)) == 2,
        "role3_is_formula_role_3": int(role3.get("formula_role", -1)) == 3,
        "full_evaluation_is_roles_2_and_3": list(full.get("formula_roles", [])) == [2, 3],
        "role4_never_accessed": not any(bool(report.get("outer_role_4_accessed")) for report in (
            role2, role3, full,
        )),
        "triplets_are_approved_residual_curriculum": triplets.get("status") in {
            "CHEMAWARE_MULTICONDITION_MAX_BOUNDARY_TRIPLETS_COMPLETE",
            "CHEMAWARE_PHASEA_RESIDUAL_CONSENSUS_TRIPLETS_COMPLETE",
        },
        "phase_a_prefix_was_immutable": bool(
            triplets.get("gates", {}).get("phase_a_base_event_prefix_immutable")
            or triplets.get("gates", {}).get("phase_a_base_prefix_immutable")
        ),
    }
    role2_base = row_by_name(role2, "phaseA_base")
    role2_model = row_by_name(role2, selected_name)
    role3_base = row_by_name(role3, "phaseA_base")
    role3_model = row_by_name(role3, "multicondition")
    provenance_gates.update({
        "role2_checkpoint_matches_input": same_path(
            role2_model["checkpoint"], args.checkpoint,
        ),
        "role3_checkpoint_matches_input": same_path(
            role3_model["checkpoint"], args.checkpoint,
        ),
    })
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

    checkpoint_name = "chemaware_multicondition_max_boundary.ckpt"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_mcmax_protected_", dir=args.output.parent))
    try:
        storage = link_or_copy(args.checkpoint, temporary / checkpoint_name)
        for output_name, source in inputs.items():
            if output_name == "checkpoint":
                continue
            shutil.copy2(source, temporary / output_name)
        source_directory = temporary / "source"
        for relative in SOURCE_FILES:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = source_directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        method_label = (
            "PHASEA_RESIDUAL_CONSENSUS"
            if triplets.get("status")
            == "CHEMAWARE_PHASEA_RESIDUAL_CONSENSUS_TRIPLETS_COMPLETE"
            else "MULTICONDITION"
        )
        status = (
            f"CHEMAWARE_{method_label}_ROLE3_CONFIRMED_ARTIFACT_PROTECTED"
            if role3_confirmed
            else f"CHEMAWARE_{method_label}_ROLE2_ONLY_ARTIFACT_PROTECTED"
        )
        manifest = {
            "status": status,
            "claim_status": (
                "ROLE3_CONFIRMED_CANDIDATE_NOT_OUTER_TESTED"
                if role3_confirmed
                else "ROLE2_ADVANCE_NOT_ROLE3_CONFIRMED"
            ),
            "shared_embedding": True,
            "reranker": False,
            "distillation": False,
            "checkpoint_file": checkpoint_name,
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_storage": storage,
            "source_files": list(SOURCE_FILES),
            "selected_name": selected_name,
            "selected_step": int(selected["step"]),
            "role2_vs_phase_a": role2_paired,
            "role3_vs_phase_a": role3_paired,
            "role2_gates": role2_gates,
            "role3_gates": role3_gates,
            "role3_confirmed": role3_confirmed,
            "outer_role4_accessed": False,
            "provenance_gates": provenance_gates,
            "allowed_claim": (
                "The selected ChemAware multi-condition shared embedding passed "
                "the frozen role-2 selection and independent role-3 confirmation gates."
                if role3_confirmed else
                "The selected ChemAware multi-condition shared embedding advanced "
                "on role 2 but did not pass the frozen role-3 confirmation gates."
            ),
            "forbidden_claim": (
                "Do not describe this artifact as outer-tested or as a five-point "
                "improvement unless those results appear in a separate frozen evaluation."
            ),
        }
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
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
