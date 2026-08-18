"""Run and audit the complete ChemAware pipeline without repeating finished work.

The pipeline separates deterministic CPU preparation from the two expensive
model stages.  Every stage has a semantic completion check, so the command can
be safely resumed after interruption.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
PIPELINE_DIR = ROOT / "data/pipeline"
MCES_DIR = ROOT / "data/e2/mces_local_rank"
DOUBLE_DIR = ROOT / "data/validation/double_mapping"
MODULE2_DIR = ROOT / "data/validation/module2_evidence_records"


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def files_exist(*paths: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def evaluation_complete(root: Path, checkpoint: Path | None) -> bool:
    report = read_json(root / "evaluation_manifest.json")
    if report.get("status") != "complete" or checkpoint is None:
        return False
    try:
        return Path(report.get("checkpoint", "")).resolve() == checkpoint.resolve()
    except OSError:
        return False


def mces_targets() -> dict[str, int]:
    audit = read_json(MCES_DIR / "manifest_audit.json")
    return {
        split: int(values.get("unique_molecule_pairs", 0))
        for split, values in audit.get("splits", {}).items()
    }


def mces_counts() -> dict[str, dict[str, int]]:
    cache = MCES_DIR / "mces_cache.sqlite"
    if not cache.is_file():
        return {}
    try:
        connection = sqlite3.connect(f"file:{cache.as_posix()}?mode=ro", uri=True, timeout=2)
        rows = connection.execute(
            "SELECT split, status, COUNT(*) FROM mces_pair GROUP BY split, status"
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        return {}
    result: dict[str, dict[str, int]] = {}
    for split, status, count in rows:
        result.setdefault(str(split), {})[str(status)] = int(count)
    return result


def mces_cache_complete() -> bool:
    targets = mces_targets()
    counts = mces_counts()
    if not targets or not counts:
        return False
    for split, target in targets.items():
        attempted = sum(counts.get(split, {}).values())
        ok = counts.get(split, {}).get("ok", 0)
        # Solver/time-limit failures are retained as unavailable edges.  They do
        # not become labels, and the rank builder ignores them.  Requiring 95%
        # usable pairs prevents a few hard MCS failures from blocking the whole
        # resumable pipeline indefinitely.
        if attempted < target or ok < int(0.95 * target):
            return False
    return True


def mces_process_running() -> bool:
    try:
        import psutil
        own_pid = psutil.Process().pid
        for process in psutil.process_iter(["pid", "cmdline"]):
            if process.info["pid"] == own_pid:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if "compute_mces_local_rank_cache.py" in command:
                return True
    except Exception:
        return False
    return False


def mces_triplets_complete() -> bool:
    report = read_json(MCES_DIR / "mces_rank_triplet_report.json")
    targets = mces_targets()
    splits = report.get("splits", {})
    if not targets or not splits:
        return False
    return all(
        int(splits.get(split, {}).get("computed_pairs", 0)) >= int(0.99 * target)
        and int(splits.get(split, {}).get("rank_triplets", 0)) > 0
        for split, target in targets.items()
    )


@dataclass
class Stage:
    name: str
    group: str
    command: list[str]
    complete: Callable[[], bool]
    requires: tuple[str, ...] = ()
    expensive: bool = False


def build_stages(args: argparse.Namespace) -> list[Stage]:
    p1_root = args.p1_output_root
    multitask_root = args.multitask_output_dir
    multitask_checkpoint = multitask_root / "best_chemaware_multitask.pt"
    final_eval_root = PIPELINE_DIR / "final_chemaware_locked_evaluation"
    return [
        Stage(
            "mces_manifest", "cpu",
            [PYTHON, "tasks/build_mces_local_rank_manifest.py"],
            lambda: files_exist(
                MCES_DIR / "manifest_audit.json",
                MCES_DIR / "train_unique_molecule_pairs.csv",
                MCES_DIR / "val_unique_molecule_pairs.csv",
            ),
        ),
        Stage(
            "mces_cache", "cpu",
            [
                PYTHON, "tasks/compute_mces_local_rank_cache.py", "--splits", "val", "train",
                "--workers", str(args.mces_workers), "--export-every", "500",
            ],
            mces_cache_complete, ("mces_manifest",), expensive=True,
        ),
        Stage(
            "mces_rank_triplets", "cpu",
            [PYTHON, "tasks/build_mces_rank_triplets.py"],
            mces_triplets_complete, ("mces_cache",),
        ),
        Stage(
            "spectrum_rule_labels", "cpu",
            [PYTHON, "tasks/build_spectrum_rule_label_cache.py"],
            lambda: files_exist(
                DOUBLE_DIR / "spectrum_rule_labels.npz",
                DOUBLE_DIR / "spectrum_rule_labels.json",
            ),
        ),
        Stage(
            "frozen_spectrum_concept_probe", "cpu",
            [PYTHON, "tasks/train_frozen_spectrum_concept_probe.py"],
            lambda: read_json(DOUBLE_DIR / "frozen_concept_probe/report.json").get("status")
            == "frozen_spectrum_concept_probe_complete",
            ("spectrum_rule_labels",),
        ),
        Stage(
            "structure_environment_data", "cpu",
            [PYTHON, "tasks/build_structure_environment_probe_data.py"],
            lambda: files_exist(
                DOUBLE_DIR / "structure_environment_probe_data.npz",
                DOUBLE_DIR / "structure_environment_probe_data.json",
            ),
        ),
        Stage(
            "frozen_structure_probe", "cpu",
            [PYTHON, "tasks/train_frozen_structure_environment_probe.py"],
            lambda: read_json(DOUBLE_DIR / "frozen_structure_probe/report.json").get("status")
            == "frozen_structure_environment_probe_complete",
            ("structure_environment_data",),
        ),
        Stage(
            "structure_spectrum_links", "cpu",
            [PYTHON, "tasks/link_structure_environments_to_spectral_concepts.py"],
            lambda: read_json(DOUBLE_DIR / "structure_spectrum_links/report.json").get("status")
            == "structure_spectrum_linking_complete",
            ("frozen_spectrum_concept_probe", "frozen_structure_probe"),
        ),
        Stage(
            "existing_factor_catalog", "cpu",
            [PYTHON, "tasks/assemble_existing_double_mapping_catalog.py"],
            lambda: read_json(DOUBLE_DIR / "existing_factor_catalog/report.json").get("status")
            == "existing_double_mapping_catalog_complete",
        ),
        Stage(
            "module2_evidence_records", "cpu",
            [PYTHON, "tasks/build_module2_evidence_records.py"],
            lambda: read_json(MODULE2_DIR / "report.json").get("status")
            == "module2_evidence_materialized",
        ),
        Stage(
            "module2_evidence_summary", "cpu",
            [PYTHON, "tasks/summarize_module2_evidence.py"],
            lambda: read_json(MODULE2_DIR / "mechanism_evidence_summary.json").get("status")
            == "module2_evidence_summary_complete",
            ("module2_evidence_records",),
        ),
        Stage(
            "p1_locked_evaluation", "gpu",
            (
                [
                    PYTHON, "tasks/run_chemmask_locked_evaluation.py",
                    "--checkpoint", str(args.p1_checkpoint),
                    "--output-root", str(p1_root), "--device", args.device,
                    "--batch-size", str(args.batch_size),
                ] if args.p1_checkpoint else []
            ),
            lambda: evaluation_complete(p1_root, args.p1_checkpoint),
            expensive=True,
        ),
        Stage(
            "chemaware_multitask_training", "gpu",
            (
                [
                    PYTHON, "tasks/train_chemaware_multitask_head.py",
                    "--p1-checkpoint", str(args.p1_checkpoint),
                    "--output-dir", str(multitask_root),
                    "--device", args.device, "--batch-size", str(args.batch_size),
                ] if args.p1_checkpoint else []
            ),
            lambda: (
                read_json(multitask_root / "report.json").get("status")
                == "chemaware_multitask_training_complete"
                and multitask_checkpoint.is_file()
            ),
            ("mces_rank_triplets", "p1_locked_evaluation"), expensive=True,
        ),
        Stage(
            "final_chemaware_locked_evaluation", "gpu",
            [
                PYTHON, "tasks/run_chemmask_locked_evaluation.py",
                "--checkpoint", str(multitask_checkpoint),
                "--output-root", str(final_eval_root), "--device", args.device,
                "--batch-size", str(args.batch_size),
            ],
            lambda: evaluation_complete(final_eval_root, multitask_checkpoint),
            ("chemaware_multitask_training",), expensive=True,
        ),
        Stage(
            "concept_peak_causality", "gpu",
            [
                PYTHON, "tasks/validate_double_mapping_concepts.py",
                "--head-checkpoint", str(multitask_checkpoint),
                "--device", args.device, "--batch-size", str(args.batch_size),
            ],
            lambda: (
                read_json(DOUBLE_DIR / "causal_concept_mapping/report.json").get("status")
                == "double_mapping_causal_pilot_complete"
                and multitask_checkpoint.name in str(
                    read_json(DOUBLE_DIR / "causal_concept_mapping/report.json").get(
                        "initialization", ""
                    )
                )
            ),
            ("chemaware_multitask_training", "frozen_spectrum_concept_probe"), expensive=True,
        ),
    ]


def summarize(stages: list[Stage], states: dict[str, str], args: argparse.Namespace) -> dict:
    concept = read_json(DOUBLE_DIR / "frozen_concept_probe/report.json")
    structure = read_json(DOUBLE_DIR / "frozen_structure_probe/report.json")
    links = read_json(DOUBLE_DIR / "structure_spectrum_links/report.json")
    causal = read_json(DOUBLE_DIR / "causal_concept_mapping/report.json")
    module2 = read_json(MODULE2_DIR / "mechanism_evidence_summary.json")
    rank = read_json(MCES_DIR / "mces_rank_triplet_report.json")
    confirmation_rows = [
        row for row in module2.get("rows", []) if row.get("split") == "confirmation"
    ]
    gates = {
        "identity_counterfactual_model_formally_evaluated": states.get("p1_locked_evaluation") == "complete",
        "mces_local_rank_supervision_ready": states.get("mces_rank_triplets") == "complete",
        "embedding_chemical_concepts_decodable": (
            float(concept.get("test_macro_auprc", 0))
            > 2 * float(concept.get("test_macro_prevalence_baseline", 1))
        ),
        "embedding_structure_environments_decodable": (
            float(structure.get("test_macro_auprc", 0))
            > 2 * float(structure.get("test_macro_prevalence_baseline", 1))
        ),
        "population_double_mapping_replicated": int(links.get("embedding_aligned_replicated_links", 0)) > 0,
        "concept_to_peak_causality_tested": causal.get("status") == "double_mapping_causal_pilot_complete",
        "module2_peak_interventions_replicated": any(
            float(row.get("directional_support_positive_fraction", 0)) > 0.5
            for row in confirmation_rows
        ),
        "final_multitask_model_trained": states.get("chemaware_multitask_training") == "complete",
        "final_multitask_model_formally_evaluated": (
            states.get("final_chemaware_locked_evaluation") == "complete"
        ),
    }
    unfinished = [name for name, state in states.items() if state not in {"complete", "skipped_optional"}]
    return {
        "status": "complete" if not unfinished else "in_progress",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "python": PYTHON,
        "stage_states": states,
        "gates": gates,
        "key_metrics": {
            "spectrum_concept_test_macro_auprc": concept.get("test_macro_auprc"),
            "spectrum_concept_prevalence_baseline": concept.get("test_macro_prevalence_baseline"),
            "structure_environment_test_macro_auprc": structure.get("test_macro_auprc"),
            "structure_environment_prevalence_baseline": structure.get("test_macro_prevalence_baseline"),
            "embedding_aligned_replicated_links": links.get("embedding_aligned_replicated_links"),
            "causal_closed_loop_concepts": causal.get("closed_loop_candidates"),
            "mces_rank_triplets": rank.get("splits"),
        },
        "next_blockers": unfinished,
        "p1_checkpoint": str(args.p1_checkpoint.resolve()) if args.p1_checkpoint else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["status", "cpu", "gpu", "all"], default="status")
    parser.add_argument("--mces-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--p1-checkpoint", type=Path)
    parser.add_argument(
        "--p1-output-root", type=Path,
        default=ROOT / "data/pipeline/p1_locked_evaluation",
    )
    parser.add_argument(
        "--multitask-output-dir", type=Path,
        default=ROOT / "data/e3/chemaware_multitask_head",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    stages = build_stages(args)
    states: dict[str, str] = {}

    for stage in stages:
        if stage.complete() and not args.force:
            states[stage.name] = "complete"
            print(f"[OK]   {stage.name}")
            continue
        if stage.name == "mces_cache" and mces_process_running():
            states[stage.name] = "running_external"
            print(f"[RUN]  {stage.name} (an existing resumable process owns this stage)")
            continue
        if stage.name in {"p1_locked_evaluation", "chemaware_multitask_training"} and not args.p1_checkpoint:
            states[stage.name] = "waiting_for_p1_checkpoint"
            print(f"[WAIT] {stage.name} (supply --p1-checkpoint after P1 training)")
            continue
        missing = [dependency for dependency in stage.requires if states.get(dependency) != "complete"]
        if missing:
            states[stage.name] = "blocked_by:" + ",".join(missing)
            print(f"[WAIT] {stage.name} <- {', '.join(missing)}")
            continue
        should_run = args.mode == "all" or args.mode == stage.group
        if not should_run:
            states[stage.name] = "pending"
            print(f"[TODO] {stage.name} ({stage.group})")
            continue
        print(f"[EXEC] {subprocess.list2cmdline(stage.command)}", flush=True)
        if args.dry_run:
            states[stage.name] = "planned"
            continue
        subprocess.run(stage.command, cwd=ROOT, check=True)
        states[stage.name] = "complete" if stage.complete() else "failed_completion_check"
        if states[stage.name] != "complete":
            raise RuntimeError(f"Stage ran but did not pass completion check: {stage.name}")

    report = summarize(stages, states, args)
    output = PIPELINE_DIR / "chemaware_pipeline_status.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"\nPipeline report: {output}")


if __name__ == "__main__":
    main()
