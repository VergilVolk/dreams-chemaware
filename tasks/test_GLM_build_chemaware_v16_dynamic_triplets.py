"""CPU contracts for the GLM V16 pre-registered dynamic-triplet builder.

Covers: the symmetric-significance veto recovering a production-rejected
relation, contested routing to the separate arm pool, no-dominant-support
rejection, the executable launch gate (threshold, anchor match, manifest and
ledger hash provenance), Phase-A prefix preservation, role codes, ledger
markers, layered-pool report compatibility, and fail-closed refusals.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

TASKS = Path(__file__).resolve().parent
sys.path.insert(0, str(TASKS))

from GLM_build_chemaware_v16_dynamic_triplets import (  # noqa: E402
    v16_registry,
    v16_relation_verdict,
)
from GLM_diagnose_chemaware_evidence_class_errors import (  # noqa: E402
    array_sha256,
    file_sha256,
)
from build_chemaware_multisource_native_triplets import pair_evidence  # noqa: E402

SCRIPT = TASKS / "GLM_build_chemaware_v16_dynamic_triplets.py"

ROWS = (
    # query 0: anchor 0 (1.0), true ref 1 (0.68), false A ref 2 (0.6),
    # false B ref 3 (0.3);
    1.0, 0.68, 0.6, 0.3,
    # query 1: anchor 4 (1.0), true ref 5 (0.5), false ref 6 (0.55) -> error;
    1.0, 0.5, 0.55,
    # query 2: anchor 7 (1.0), true ref 8 (0.7), false ref 9 (0.75) -> error.
    1.0, 0.7, 0.75,
)
IK = (
    "Q0TRUE00000001-QQ", "Q0TRUE00000001-FF",  # rows 0, 1 share identity
    "Q0FALSEA0000001-QQ", "Q0FALSEB0000001-QQ",
    "Q1TRUE00000001-QQ", "Q1TRUE00000001-FF",
    "Q1FALSEA000001-QQ",
    "Q2TRUE00000001-QQ", "Q2TRUE00000001-FF",
    "Q2FALSEA000001-QQ",
)


def unit(x: float) -> tuple[float, float]:
    return (x, math.sqrt(max(0.0, 1.0 - x * x)))


def build_world(root: Path) -> dict[str, Path]:
    cache_dir = root / "cache"
    cache_dir.mkdir()
    embeddings = np.asarray(
        [unit(v) for v in ROWS], dtype=np.float64,
    ).astype(np.float32)
    rows = np.arange(len(ROWS), dtype=np.int64)
    np.save(cache_dir / "rows.npy", rows, allow_pickle=False)
    np.save(cache_dir / "embeddings_f32.npy", embeddings, allow_pickle=False)
    checkpoint = root / "phasea.ckpt"
    checkpoint.write_bytes(b"synthetic-phase-a")

    manifest = {
        "query_row": np.asarray([0, 4, 7], dtype=np.int64),
        "query_ptr": np.asarray([0, 3, 5, 7], dtype=np.int64),
        "molecule_label": np.asarray(
            [1, 0, 0, 1, 0, 1, 0], dtype=np.int8,
        ),
        "molecule_formula": np.asarray([
            "C10H10NO", "C10H10NO", "C10H10NO",
            "C6H12O6", "C6H12O6",
            "C9H9NO3", "C9H9NO3",
        ]),
        "molecule_ik14": np.asarray([
            value[:14] for value in (
                IK[0], IK[2], IK[3], IK[4], IK[6], IK[7], IK[9],
            )
        ]),
        "molecule_ptr": np.asarray(
            [0, 2, 3, 4, 6, 7, 9, 10], dtype=np.int64,
        ),
        "pair_candidate_row": np.asarray(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64,
        ),
        "query_formula": np.asarray(["C10H10NO", "C6H12O6", "C9H9NO3"]),
        "query_ik14": np.asarray([
            IK[0][:14], IK[4][:14], IK[7][:14],
        ]),
    }
    manifest_path = root / "manifest.npz"
    np.savez(manifest_path, **manifest)

    (cache_dir / "report.json").write_text(json.dumps({
        "status": "CHEMAWARE_CHECKPOINT_MANIFEST_EMBEDDINGS_COMPLETE",
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest_path),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(embeddings),
        "formula_role_4_accessed": False,
    }), encoding="utf-8")

    data_path = root / "data.hdf5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("INCHIKEY", data=np.asarray(IK, dtype="S27"))

    # Phase-A prefix: one event (anchor 0, positive row 1, negative row 3).
    phasea_pool = root / "phasea_pool.npz"
    np.savez(
        phasea_pool,
        anchor_idx=np.asarray([0], dtype=np.int64),
        positive_ptr=np.asarray([0, 1], dtype=np.int64),
        positive_idx=np.asarray([1], dtype=np.int64),
        negative_ptr=np.asarray([0, 1], dtype=np.int64),
        negative_idx=np.asarray([3], dtype=np.int64),
        source_query=np.asarray([0], dtype=np.int64),
        negative_candidate=np.asarray([2], dtype=np.int32),
        source_tag=np.asarray([8], dtype=np.int16),
        curriculum_role=np.asarray([8], dtype=np.int8),
    )
    validation_pool = root / "val_pool.npz"
    np.savez(validation_pool, anchor_idx=np.asarray([], dtype=np.int64))

    ledger = root / "ledger"
    ledger.mkdir()
    families = {
        "fam_pi": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
        "fam_nl": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
    }

    def body(query: int, candidate: int, fam: str, score: float,
             ca: float, cb: float):
        molecule = int(manifest["query_ptr"][query]) + candidate
        return (
            f"{query}\t{candidate}\t{manifest['molecule_ik14'][molecule]}\t"
            f"{manifest['molecule_formula'][molecule]}\t{fam}\tall_candidates\t"
            f"{score}\t{ca}\t{cb}\t1"
        )

    lines = [
        "manifest_query\tlocal_candidate\tik14\tformula\tsource_family\tscope\t"
        "source_score\tcontrol_a_score\tcontrol_b_score\tcontrols_available",
        # q0 pair (0,1): PI support; NL sub-significant opposition -> vetoed
        # under production rules, recovered by the symmetric veto.
        body(0, 0, "fam_pi", 0.9, 0.5, 0.2),
        body(0, 1, "fam_pi", 0.5, 0.5, 0.2),
        body(0, 0, "fam_nl", 0.3, 0.5, 0.1),
        body(0, 1, "fam_nl", 0.5, 0.2, 0.4),
        # q0 pair (0,2): PI positive but sub-dominant (0.05 < control 0.3,
        # abstain), NL dominant opposition and no support ->
        # dominant_opposition_only rejection.
        body(0, 2, "fam_pi", 0.85, 0.3, 0.3),
        body(0, 2, "fam_nl", 0.95, 0.0, 0.0),
        # q1 pair (0,1): both families support -> unanimous.
        body(1, 0, "fam_pi", 0.8, 0.4, 0.4),
        body(1, 1, "fam_pi", 0.4, 0.4, 0.4),
        body(1, 0, "fam_nl", 0.7, 0.3, 0.3),
        body(1, 1, "fam_nl", 0.4, 0.3, 0.3),
        # q2 pair (0,1): PI support, NL dominant opposition -> contested arm.
        body(2, 0, "fam_pi", 0.8, 0.3, 0.3),
        body(2, 1, "fam_pi", 0.4, 0.3, 0.3),
        body(2, 0, "fam_nl", 0.7, 0.3, 0.3),
        body(2, 1, "fam_nl", 0.9, 0.0, 0.0),
    ]
    (ledger / "candidate_scores.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    (ledger / "report.json").write_text(json.dumps({
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE",
        "truth_fields_exported": False,
        "candidate_scores_sha256": file_sha256(ledger / "candidate_scores.tsv"),
        "source_families": families,
    }), encoding="utf-8")

    return {
        "manifest": manifest_path, "cache": cache_dir, "checkpoint": checkpoint,
        "data": data_path, "phasea_pool": phasea_pool,
        "validation_pool": validation_pool, "ledger": ledger,
    }


def write_diagnostic(
    root: Path, world: dict[str, Path], *, vetoed: int = 30, contested: int = 25,
    all_match: bool = True, tamper_ledger_hash: bool = False,
    tamper_manifest_hash: bool = False,
) -> Path:
    ledger = world["ledger"]
    report_sha = file_sha256(ledger / "report.json")
    tsv_sha = file_sha256(ledger / "candidate_scores.tsv")
    if tamper_ledger_hash:
        tsv_sha = "0" * 64
    manifest_sha = (
        "0" * 64 if tamper_manifest_hash
        else file_sha256(world["manifest"])
    )
    diagnostic = {
        "status": "GLM_CHEMAWARE_EVIDENCE_CLASS_DIAGNOSTIC_COMPLETE",
        "sanity_anchors": {"all_match": all_match},
        "decision_summary": {"decision_eligible": True},
        "error_boundary_counts": {
            "vetoed_by_subsignificant_oppose": vetoed,
            "contested_dominant": contested,
        },
        "provenance": {
            "manifest_sha256": manifest_sha,
            "ledgers": [{
                "ledger": str(ledger.resolve()),
                "report_sha256": report_sha,
                "tsv_sha256": tsv_sha,
            }],
        },
    }
    path = root / f"diagnostic_{abs(hash((vetoed, contested, all_match)))}.json"
    path.write_text(json.dumps(diagnostic), encoding="utf-8")
    return path


def run_builder(
    world: dict[str, Path], diagnostic: Path, output: Path,
    contested_output: Path, extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-X", "utf8", str(SCRIPT),
            "--phasea-pool", str(world["phasea_pool"]),
            "--validation-pool", str(world["validation_pool"]),
            "--source-ledger", str(world["ledger"]),
            "--manifest", str(world["manifest"]),
            "--embedding-rows", str(world["cache"] / "rows.npy"),
            "--phasea-embeddings", str(world["cache"] / "embeddings_f32.npy"),
            "--embedding-report", str(world["cache"] / "report.json"),
            "--geometry-checkpoint", str(world["checkpoint"]),
            "--data", str(world["data"]),
            "--evidence-class-diagnostic", str(diagnostic),
            "--output", str(output),
            "--contested-output", str(contested_output),
            *(extra or []),
        ],
        capture_output=True, text=True,
    )


def verdict_contract() -> None:
    world_ledger_families = {
        "fam_pi": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
        "fam_nl": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
    }
    registry = v16_registry(world_ledger_families)

    def candidate_body(score: float, ca: float, cb: float):
        return {
            "ik14": "IK", "formula": "C", "scope": "all_candidates",
            "score": score, "control_scores": (ca, cb),
            "control_names": ("control_a_score", "control_b_score"),
            "controls": True,
        }

    candidates = {
        0: {
            "fam_pi": candidate_body(0.9, 0.5, 0.2),
            "fam_nl": candidate_body(0.3, 0.5, 0.1),
        },
        1: {
            "fam_pi": candidate_body(0.5, 0.5, 0.2),
            "fam_nl": candidate_body(0.5, 0.2, 0.4),
        },
    }
    verdict = v16_relation_verdict(candidates, registry, 0, 1)
    assert verdict["pair_class"] == "vetoed_by_subsignificant_oppose"
    assert verdict["support_names"] == ["fam_pi"]
    assert verdict["blocking_subsignificant"] == ["fam_nl"]
    assert verdict["blocking_dominant"] == []
    # The production builder rejects this exact pair: symmetric recovery.
    supports, blocking, _ = pair_evidence(candidates, world_ledger_families, 0, 1)
    assert supports and blocking == ["fam_nl"]

    contested = {
        0: {
            "fam_pi": candidate_body(0.8, 0.3, 0.3),
            "fam_nl": candidate_body(0.7, 0.3, 0.3),
        },
        1: {
            "fam_pi": candidate_body(0.4, 0.3, 0.3),
            "fam_nl": candidate_body(0.9, 0.0, 0.0),
        },
    }
    verdict = v16_relation_verdict(contested, registry, 0, 1)
    assert verdict["pair_class"] == "contested_dominant"
    assert verdict["blocking_dominant"] == ["fam_nl"]

    unanimous = {
        0: {
            "fam_pi": candidate_body(0.8, 0.4, 0.4),
            "fam_nl": candidate_body(0.7, 0.3, 0.3),
        },
        1: {
            "fam_pi": candidate_body(0.4, 0.4, 0.4),
            "fam_nl": candidate_body(0.4, 0.3, 0.3),
        },
    }
    verdict = v16_relation_verdict(unanimous, registry, 0, 1)
    assert verdict["pair_class"] == "unanimous_admitted_style"


def load_pool(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def end_to_end_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_v16_") as name:
        root = Path(name)
        world = build_world(root)
        diagnostic = write_diagnostic(root, world)
        output = root / "v16_triplets"
        contested_output = root / "v16_contested"
        result = run_builder(world, diagnostic, output, contested_output)
        assert result.returncode == 0, result.stderr
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE"
        assert report["variant"] == "GLM_v16_symmetric_significance_pre_registered"
        assert all(report["gates"].values())
        # Layered-pool compatibility keys are truthful.
        pool = load_pool(output / "train_pool.npz")
        chemical = len(pool["anchor_idx"]) - report["phasea_events_preserved"]
        assert chemical == report["dynamic_chemical_events_added"] == 2
        assert report["dynamic_chemical_queries"] == 2
        # Arm A: q1 unanimous error (role 12) + q0 recovered margin (role 13).
        chemical_roles = sorted(
            pool["curriculum_role"][report["phasea_events_preserved"]:].tolist()
        )
        assert chemical_roles == [12, 13]
        audit = report["v16_audit"]
        assert audit["relations_recovered_from_subsignificant_veto"] == 1
        assert audit["relations_uncontested_admitted"] == 1
        assert audit["relations_routed_to_contested_arm"] == 1
        assert audit["relations_rejected_no_dominant_support"] == 1
        assert report["source_correctable_current_winners"] == 1
        # Ledger marks the recovered relation with its class and veto family.
        with (output / "dynamic_source_event_ledger.tsv").open(
            encoding="utf-8",
        ) as handle:
            ledger_rows = list(csv.DictReader(handle, delimiter="\t"))
        by_query = {int(row["manifest_query"]): row for row in ledger_rows}
        assert by_query[0]["pair_evidence_class"] == "vetoed_by_subsignificant_oppose"
        assert by_query[0]["v16_confidence_tier"] == "recovered_subsignificant_veto"
        assert by_query[0]["blocking_subsignificant_opposition"] == "fam_nl"
        assert by_query[1]["pair_evidence_class"] == "unanimous_admitted_style"
        assert float(by_query[0]["source_deltas"].split(",")[0]) > 0.0
        # Phase-A prefix preserved byte-exactly in both pools.
        contested_pool = load_pool(contested_output / "train_pool.npz")
        phasea = load_pool(world["phasea_pool"])
        assert np.array_equal(pool["anchor_idx"][:1], phasea["anchor_idx"])
        assert np.array_equal(
            pool["curriculum_role"][:1], phasea["curriculum_role"],
        )
        assert np.array_equal(
            contested_pool["anchor_idx"][:1], phasea["anchor_idx"],
        )
        contested_chemical = contested_pool["curriculum_role"][1:].tolist()
        assert contested_chemical == [14]
        contested_report = json.loads(
            (contested_output / "report.json").read_text(encoding="utf-8"),
        )
        assert contested_report["status"] == "GLM_V16_CONTESTED_ARM_POOL_COMPLETE"
        assert contested_report["contested_events"] == 1
        # The rejected no-dominant-support relation is auditable.
        with (output / "rejected_relation_ledger.tsv").open(
            encoding="utf-8",
        ) as handle:
            rejected = list(csv.DictReader(handle, delimiter="\t"))
        assert any(
            row["reason"] == "no_dominant_support"
            and row["pair_evidence_class"] == "dominant_opposition_only"
            for row in rejected
        )
        # Launch gate provenance is recorded.
        assert report["launch_gate"]["diagnostic_recoverable_boundaries"] == 30
        assert report["launch_gate"]["diagnostic_contested_boundaries"] == 25


def gate_refusal_contracts() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_v16_ref_") as name:
        root = Path(name)
        world = build_world(root)
        output = root / "v16_triplets"
        contested_output = root / "v16_contested"
        cases = {
            "threshold": write_diagnostic(root, world, vetoed=10, contested=5),
            "anchors": write_diagnostic(root, world, all_match=False),
            "ledger_hash": write_diagnostic(root, world, tamper_ledger_hash=True),
            "manifest_hash": write_diagnostic(
                root, world, tamper_manifest_hash=True,
            ),
        }
        for name_case, diagnostic in cases.items():
            result = run_builder(world, diagnostic, output, contested_output)
            assert result.returncode != 0, name_case
            assert not output.exists() and not contested_output.exists(), name_case
        # Passing gate still refuses to overwrite an existing output.
        diagnostic = write_diagnostic(root, world)
        output.mkdir()
        result = run_builder(world, diagnostic, output, contested_output)
        assert result.returncode != 0
        assert "refusing to overwrite" in (result.stderr + result.stdout)


def main() -> None:
    verdict_contract()
    end_to_end_contract()
    gate_refusal_contracts()
    print("GLM V16 dynamic-triplet contracts passed")


if __name__ == "__main__":
    main()
