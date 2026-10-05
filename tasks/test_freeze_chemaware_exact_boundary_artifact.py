"""CPU contracts for exact-boundary ChemAware artifact protection."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def metrics(recall1: float, recall3: float, mrr: float, micro: float, macro: float):
    return {
        "queries": 100, "recall1": recall1, "recall3": recall3,
        "recall5": 1.0, "recall10": 1.0, "recall20": 1.0,
        "recall50": 1.0, "mrr": mrr, "mean_positive_margin": 0.3,
        "micro_auc": micro, "macro_auc": macro,
    }


def paired(delta: float, corrected: int, introduced: int):
    return {
        "delta_recall1": delta, "delta_mrr": delta / 2,
        "corrected_at_1": corrected, "introduced_at_1": introduced,
        "formula_cluster_bootstrap_delta_recall1_ci95": [0.001, 0.03],
    }


def write(path: Path, body: dict[str, object]) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def freeze(root: Path, role3_passes: bool) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = root / "step-000100.ckpt"; checkpoint.write_bytes(b"exact boundary")
    base = metrics(0.90, 0.98, 0.94, 0.93, 0.95)
    improved = metrics(0.92, 0.98, 0.95, 0.94, 0.96)
    role3_model = improved if role3_passes else metrics(0.89, 0.97, 0.93, 0.92, 0.94)
    role3_pair = paired(0.02, 5, 1) if role3_passes else paired(-0.01, 0, 1)
    selected_name = "exact_step-000100"
    selection = root / "selection.json"; role2 = root / "role2.json"
    role3 = root / "role3.json"; full = root / "full.json"
    triplets = root / "triplets.json"; training = root / "training.json"
    train_pool = root / "train_pool.npz"; val_pool = root / "val_pool.npz"
    write(selection, {
        "base_name": "phaseA_base", "advanced_beyond_base": True,
        "selected": {"name": selected_name, "checkpoint": str(checkpoint), "step": 100},
    })
    write(role2, {
        "formula_role": 2, "outer_role_4_accessed": False,
        "results": [
            {"name": "phaseA_base", "checkpoint": "phasea.ckpt", "metrics": base},
            {"name": selected_name, "checkpoint": str(checkpoint), "metrics": improved,
             "paired_vs_phaseA_base": paired(0.02, 5, 1)},
        ],
    })
    write(role3, {
        "formula_role": 3, "outer_role_4_accessed": False,
        "results": [
            {"name": "phaseA_base", "checkpoint": "phasea.ckpt", "metrics": base},
            {"name": "exact_boundary", "checkpoint": str(checkpoint),
             "metrics": role3_model, "paired_vs_phaseA_base": role3_pair},
        ],
    })
    write(full, {"formula_roles": [2, 3], "outer_role_4_accessed": False, "results": []})
    write(triplets, {
        "status": "CHEMAWARE_EXACT_BOUNDARY_RESIDUAL_TRIPLETS_COMPLETE",
        "cache_kind": "protected_phasea",
        "coverage": {"correction_events": 120, "correction_queries": 60,
                     "correction_formulas": 50},
        "gates": {"no_identity_broadcast": True,
                  "no_custom_sampling_weights": True, "all": True},
    })
    write(training, {"status": "CHEMAWARE_DREAMS_NATIVE_TRAINING_COMPLETE"})
    train_pool.write_bytes(b"train"); val_pool.write_bytes(b"val")
    output = root / "protected"
    command = [
        sys.executable, str(ROOT / "tasks/freeze_chemaware_exact_boundary_artifact.py"),
        "--checkpoint", str(checkpoint), "--selection", str(selection),
        "--role2-evaluation", str(role2), "--role3-evaluation", str(role3),
        "--full-evaluation", str(full), "--triplet-report", str(triplets),
        "--train-pool", str(train_pool), "--val-pool", str(val_pool),
        "--training-report", str(training), "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"freezer failed\n{completed.stdout}\n{completed.stderr}")
    return output


def main() -> None:
    from freeze_chemaware_exact_boundary_artifact import SOURCE_FILES
    assert SOURCE_FILES and all(Path(path).parts[0] == "tasks" for path in SOURCE_FILES)
    with tempfile.TemporaryDirectory(prefix="chem_exact_freeze_") as raw:
        root = Path(raw)
        confirmed = freeze(root / "confirmed", True)
        manifest = json.loads((confirmed / "artifact_manifest.json").read_text())
        assert manifest["status"] == "CHEMAWARE_EXACT_BOUNDARY_ROLE3_CONFIRMED_ARTIFACT_PROTECTED"
        assert manifest["triplet_method_contract"] == "exact_query_candidate_boundary"
        assert (confirmed / "chemaware_exact_boundary_native.ckpt").is_file()
        assert (confirmed / "source/tasks/build_chemaware_exact_boundary_residual_triplets.py").is_file()
        for line in (confirmed / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256((confirmed / name).read_bytes()).hexdigest() == expected
        role2 = freeze(root / "role2", False)
        manifest = json.loads((role2 / "artifact_manifest.json").read_text())
        assert manifest["status"] == "CHEMAWARE_EXACT_BOUNDARY_ROLE2_ONLY_ARTIFACT_PROTECTED"
        assert manifest["role3_confirmed"] is False
    print("PASS: ChemAware exact-boundary artifact-protection contracts")


if __name__ == "__main__":
    main()
