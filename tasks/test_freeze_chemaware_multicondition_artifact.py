"""CPU contracts for atomic ChemAware multi-condition artifact protection."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def metrics(recall1: float, recall3: float, mrr: float, micro: float, macro: float):
    return {
        "queries": 100,
        "recall1": recall1,
        "recall3": recall3,
        "recall5": 1.0,
        "recall10": 1.0,
        "recall20": 1.0,
        "recall50": 1.0,
        "mrr": mrr,
        "mean_positive_margin": 0.3,
        "micro_auc": micro,
        "macro_auc": macro,
    }


def paired(delta: float, corrected: int, introduced: int):
    return {
        "delta_recall1": delta,
        "delta_mrr": delta / 2,
        "corrected_at_1": corrected,
        "introduced_at_1": introduced,
        "formula_cluster_bootstrap_delta_recall1_ci95": [0.001, 0.03],
    }


def write(path: Path, body: dict[str, object]) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "step-000500.ckpt"
        checkpoint.write_bytes(b"evaluated checkpoint")
        selection = root / "selection.json"
        role2_path = root / "role2.json"
        role3_path = root / "role3.json"
        full_path = root / "full.json"
        triplet_path = root / "triplets.json"
        train_pool = root / "train_pool.npz"
        val_pool = root / "val_pool.npz"
        training_report = root / "training_report.json"
        output = root / "protected"
        selected_name = "mcmax_step-000500"
        base_metrics = metrics(0.90, 0.98, 0.94, 0.93, 0.95)
        model_metrics = metrics(0.92, 0.98, 0.95, 0.94, 0.96)
        write(selection, {
            "base_name": "phaseA_base",
            "advanced_beyond_base": True,
            "selected": {
                "name": selected_name,
                "checkpoint": str(checkpoint),
                "step": 500,
            },
        })
        write(role2_path, {
            "formula_role": 2,
            "outer_role_4_accessed": False,
            "results": [
                {"name": "phaseA_base", "checkpoint": "phasea.ckpt", "metrics": base_metrics},
                {
                    "name": selected_name, "checkpoint": str(checkpoint),
                    "metrics": model_metrics,
                    "paired_vs_phaseA_base": paired(0.02, 5, 1),
                },
            ],
        })
        write(role3_path, {
            "formula_role": 3,
            "outer_role_4_accessed": False,
            "results": [
                {"name": "phaseA_base", "checkpoint": "phasea.ckpt", "metrics": base_metrics},
                {
                    "name": "multicondition", "checkpoint": str(checkpoint),
                    "metrics": model_metrics,
                    "paired_vs_phaseA_base": paired(0.02, 5, 1),
                },
            ],
        })
        write(full_path, {
            "formula_roles": [2, 3],
            "outer_role_4_accessed": False,
            "results": [],
        })
        write(triplet_path, {
            "status": "CHEMAWARE_MULTICONDITION_MAX_BOUNDARY_TRIPLETS_COMPLETE",
            "gates": {"phase_a_base_event_prefix_immutable": True},
        })
        train_pool.write_bytes(b"train pool")
        val_pool.write_bytes(b"validation pool")
        write(training_report, {"status": "CHEMAWARE_DREAMS_NATIVE_TRAINING_COMPLETE"})
        subprocess.run([
            sys.executable, str(ROOT / "tasks/freeze_chemaware_multicondition_artifact.py"),
            "--checkpoint", str(checkpoint),
            "--selection", str(selection),
            "--role2-evaluation", str(role2_path),
            "--role3-evaluation", str(role3_path),
            "--full-evaluation", str(full_path),
            "--triplet-report", str(triplet_path),
            "--train-pool", str(train_pool),
            "--val-pool", str(val_pool),
            "--training-report", str(training_report),
            "--output", str(output),
        ], check=True, capture_output=True, text=True)
        manifest = json.loads((output / "artifact_manifest.json").read_text())
        assert manifest["status"] == (
            "CHEMAWARE_MULTICONDITION_ROLE3_CONFIRMED_ARTIFACT_PROTECTED"
        )
        assert manifest["role3_confirmed"] is True
        assert manifest["outer_role4_accessed"] is False
        assert (output / "chemaware_multicondition_max_boundary.ckpt").is_file()
        assert (output / "train_pool.npz").read_bytes() == b"train pool"
        assert (output / "source/tasks/train_chemaware_weighted_native.py").is_file()
        assert (output / "SHA256SUMS").is_file()
        for line in (output / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected

        # A role-3 failure remains valuable evidence and must still be archived,
        # but it must never receive the confirmation label.
        output_role2_only = root / "protected_role2_only"
        harmful_metrics = metrics(0.89, 0.97, 0.93, 0.92, 0.94)
        write(role3_path, {
            "formula_role": 3,
            "outer_role_4_accessed": False,
            "results": [
                {"name": "phaseA_base", "checkpoint": "phasea.ckpt", "metrics": base_metrics},
                {
                    "name": "multicondition", "checkpoint": str(checkpoint),
                    "metrics": harmful_metrics,
                    "paired_vs_phaseA_base": paired(-0.01, 0, 1),
                },
            ],
        })
        subprocess.run([
            sys.executable, str(ROOT / "tasks/freeze_chemaware_multicondition_artifact.py"),
            "--checkpoint", str(checkpoint),
            "--selection", str(selection),
            "--role2-evaluation", str(role2_path),
            "--role3-evaluation", str(role3_path),
            "--full-evaluation", str(full_path),
            "--triplet-report", str(triplet_path),
            "--train-pool", str(train_pool),
            "--val-pool", str(val_pool),
            "--training-report", str(training_report),
            "--output", str(output_role2_only),
        ], check=True, capture_output=True, text=True)
        manifest = json.loads(
            (output_role2_only / "artifact_manifest.json").read_text()
        )
        assert manifest["status"] == (
            "CHEMAWARE_MULTICONDITION_ROLE2_ONLY_ARTIFACT_PROTECTED"
        )
        assert manifest["role3_confirmed"] is False

        output_phasea_residual = root / "protected_phasea_residual"
        write(triplet_path, {
            "status": "CHEMAWARE_PHASEA_RESIDUAL_CONSENSUS_TRIPLETS_COMPLETE",
            "gates": {"phase_a_base_prefix_immutable": True},
        })
        subprocess.run([
            sys.executable, str(ROOT / "tasks/freeze_chemaware_multicondition_artifact.py"),
            "--checkpoint", str(checkpoint),
            "--selection", str(selection),
            "--role2-evaluation", str(role2_path),
            "--role3-evaluation", str(role3_path),
            "--full-evaluation", str(full_path),
            "--triplet-report", str(triplet_path),
            "--train-pool", str(train_pool),
            "--val-pool", str(val_pool),
            "--training-report", str(training_report),
            "--output", str(output_phasea_residual),
        ], check=True, capture_output=True, text=True)
        manifest = json.loads(
            (output_phasea_residual / "artifact_manifest.json").read_text()
        )
        assert manifest["status"] == (
            "CHEMAWARE_PHASEA_RESIDUAL_CONSENSUS_ROLE2_ONLY_ARTIFACT_PROTECTED"
        )
    print("PASS: ChemAware multi-condition artifact protection contracts", flush=True)


if __name__ == "__main__":
    main()
