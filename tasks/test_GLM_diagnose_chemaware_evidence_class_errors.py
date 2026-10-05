"""Semantic contracts for GLM_diagnose_chemaware_evidence_class_errors.

Covers: builder-faithful stance semantics (support / positive-subdominant /
dominant opposition / sub-significant opposition / inapplicable scope /
unassessed controls), pair classification including tier arbitration, anchor
row exclusion, the tie-unfavorable error policy, multi-winner tie ambiguity,
no-ledger-coverage boundaries, end-to-end cross-tab numbers, multi-ledger
merge equivalence, family-gate exclusion, within-formula-cluster permutation
structure, decision eligibility, and fail-closed cache provenance.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from GLM_diagnose_chemaware_evidence_class_errors import (  # noqa: E402
    classify_pair,
    family_stance,
    file_sha256,
    array_sha256,
)

SCRIPT = (
    Path(__file__).resolve().parent
    / "GLM_diagnose_chemaware_evidence_class_errors.py"
)


def unit(x: float) -> tuple[float, float]:
    return (x, math.sqrt(max(0.0, 1.0 - x * x)))


def body(
    score: float, ca: float, cb: float, formula: str,
    scope: str = "all_candidates",
) -> dict[str, object]:
    return {
        "ik14": "IK",
        "formula": formula,
        "scope": scope,
        "score": score,
        "control_scores": (ca, cb),
        "control_names": ("control_a_score", "control_b_score"),
        "controls": True,
    }


def registry_entry(
    scope: str = "all_candidates", tier: str = "C",
) -> dict[str, object]:
    ranks = {"A": 3, "B": 2, "C": 1, "Q": 0}
    return {
        "scope": scope,
        "confidence_tier": (
            "C_calibration_required" if tier == "C" else tier + "_tier"
        ),
        "tier_rank": ranks[tier],
        "matched_controls": ["rule_mass_cyclic"],
        "larger_is_better": True,
    }


def stance_contract() -> None:
    reg = {"fam": registry_entry()}
    # delta > 0 and beats every control delta -> support.
    assert family_stance(
        body(0.9, 0.5, 0.2, "C10H10NO"), body(0.5, 0.5, 0.2, "C10H10NO"), "fam", reg,
    ) == "support"
    # delta > 0 but one control delta >= delta -> positive subdominant abstain.
    assert family_stance(
        body(0.6, 0.7, 0.2, "C10H10NO"), body(0.5, 0.5, 0.2, "C10H10NO"), "fam", reg,
    ) == "abstain_positive_subdominant"
    # delta <= 0 with every control delta above it -> dominant opposition.
    assert family_stance(
        body(0.2, 0.5, 0.5, "C10H10NO"), body(0.6, 0.0, 0.0, "C10H10NO"), "fam", reg,
    ) == "oppose_dominant"
    # delta <= 0 with one control delta below it -> sub-significant opposition.
    assert family_stance(
        body(0.4, 0.1, 0.1, "C10H10NO"), body(0.5, 0.1, 0.5, "C10H10NO"), "fam", reg,
    ) == "oppose_subsignificant"
    # delta == 0 is a raw opposition under builder semantics (delta <= 0).
    assert family_stance(
        body(0.5, 0.5, 0.2, "C10H10NO"), body(0.5, 0.5, 0.2, "C10H10NO"), "fam", reg,
    ) == "oppose_subsignificant"
    # delta <= 0 without controls -> unassessed raw opposition.
    unassessed = body(0.3, 0.0, 0.0, "C10H10NO")
    unassessed["controls"] = False
    assert family_stance(
        unassessed, body(0.5, 0.1, 0.1, "C10H10NO"), "fam", reg,
    ) == "oppose_unassessed"
    # cross_formula scope family is inapplicable on a same-formula pair
    # (scope is carried by the ledger row, as in read_source_ledgers).
    cross = {"fam": registry_entry(scope="cross_formula")}
    assert family_stance(
        body(0.9, 0.0, 0.0, "C10H10NO", scope="cross_formula"),
        body(0.1, 0.0, 0.0, "C10H10NO", scope="cross_formula"), "fam", cross,
    ) == "inapplicable"
    assert family_stance(
        body(0.9, 0.0, 0.0, "C6H12O6", scope="cross_formula"),
        body(0.1, 0.0, 0.0, "C6H10O6", scope="cross_formula"), "fam", cross,
    ) == "support"


def classify_contract() -> None:
    assert classify_pair({"a": "support"}, {"a": registry_entry()}) == \
        "unanimous_admitted_style"
    assert classify_pair(
        {"a": "support", "b": "oppose_subsignificant"},
        {"a": registry_entry(), "b": registry_entry()},
    ) == "vetoed_by_subsignificant_oppose"
    assert classify_pair(
        {"a": "support", "b": "oppose_dominant"},
        {"a": registry_entry(), "b": registry_entry()},
    ) == "contested_dominant"
    assert classify_pair(
        {"a": "support", "b": "oppose_unassessed"},
        {"a": registry_entry(), "b": registry_entry()},
    ) == "contested_dominant"
    assert classify_pair(
        {"a": "oppose_dominant", "b": "oppose_subsignificant"},
        {"a": registry_entry(), "b": registry_entry()},
    ) == "dominant_opposition_only"
    assert classify_pair(
        {"a": "abstain_positive_subdominant"}, {"a": registry_entry()},
    ) == "no_dominant_signal"
    assert classify_pair({}, {}) == "no_dominant_signal"
    # Tier arbitration: a lower-tier opposition never blocks a stronger support.
    mixed = {"b": registry_entry(tier="B"), "c": registry_entry(tier="C")}
    assert classify_pair({"b": "support", "c": "oppose_dominant"}, mixed) == \
        "unanimous_admitted_style"
    # Same tier blocks.
    assert classify_pair(
        {"b": "support", "c": "oppose_subsignificant"},
        {"b": registry_entry(tier="B"), "c": registry_entry(tier="B")},
    ) == "vetoed_by_subsignificant_oppose"


def build_world(root: Path) -> tuple[Path, Path, Path]:
    """Four queries exercising every boundary class.

    q0: plain error, vetoed_by_subsignificant boundary (anchor exclusion is
        load-bearing: the true candidate only wins if its own anchor row is
        wrongly counted as its reference).
    q1: tie error (best false == true) with unanimous boundary.
    q2: error with two tied winning falses whose pair classes disagree ->
        ambiguous_tie.
    q3: error whose winning false candidate has no ledger rows ->
        no_ledger_coverage.
    """
    cache_dir = root / "cache"
    cache_dir.mkdir()
    cosines = (1.0, 0.9, 0.99, 1.0, 0.5, 0.5, 0.3, 0.2,
               1.0, 0.4, 0.6, 0.6, 1.0, 0.4, 0.35, 0.6)
    embeddings = np.asarray(
        [unit(v) for v in cosines], dtype=np.float64,
    ).astype(np.float32)
    rows = np.arange(len(cosines), dtype=np.int64)
    np.save(cache_dir / "rows.npy", rows, allow_pickle=False)
    np.save(cache_dir / "embeddings_f32.npy", embeddings, allow_pickle=False)

    manifest = {
        "query_row": np.asarray([0, 3, 8, 12], dtype=np.int64),
        "query_ptr": np.asarray([0, 2, 6, 9, 12], dtype=np.int64),
        "molecule_label": np.asarray(
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8,
        ),
        "molecule_formula": np.asarray([
            "C10H10NO", "C10H10NO",
            "C6H12O6", "C6H10O6", "C7H8N4O2", "C5H4N4",
            "C9H9NO3", "C8H8NO3", "C7H7NO3",
            "C8H9NO2", "C8H9NO2", "C4H4N2O2",
        ]),
        "molecule_ik14": np.asarray([
            "IK00", "IK01",
            "IK10", "IK11", "IK12", "IK13",
            "IK20", "IK21", "IK22",
            "IK30", "IK31", "IK32",
        ]),
        "molecule_ptr": np.asarray(
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64,
        ),
        "pair_candidate_row": np.asarray(
            [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15], dtype=np.int64,
        ),
        "query_formula": np.asarray(
            ["C10H10NO", "C6H12O6", "C9H9NO3", "C8H9NO2"],
        ),
    }
    manifest_path = root / "manifest.npz"
    np.savez(manifest_path, **manifest)

    (cache_dir / "report.json").write_text(json.dumps({
        "status": "CHEMAWARE_CHECKPOINT_MANIFEST_EMBEDDINGS_COMPLETE",
        "checkpoint": "synthetic://phasea",
        "checkpoint_sha256": "0" * 64,
        "checkpoint_kind": "official_embedding",
        "manifest_sha256": file_sha256(manifest_path),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(embeddings),
        "formula_role_4_accessed": False,
    }), encoding="utf-8")

    ledger = root / "ledger"
    ledger.mkdir()
    families = {
        "synthetic_product_ion": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
        "synthetic_neutral_loss": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
        "synthetic_cross_only": {
            "scope": "cross_formula", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": ["rule_mass_cyclic"],
            "specificity_gate_passed": True,
        },
        "synthetic_gate_failed": {
            "scope": "all_candidates", "larger_is_better": True,
            "confidence_tier": "C_calibration_required",
            "matched_controls": [],
            "specificity_gate_passed": False,
        },
    }
    (ledger / "report.json").write_text(
        json.dumps({
            "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE",
            "truth_fields_exported": False,
            "source_families": families,
        }), encoding="utf-8",
    )

    header = (
        "manifest_query\tlocal_candidate\tik14\tformula\tsource_family\tscope\t"
        "source_score\tcontrol_a_score\tcontrol_b_score\tcontrols_available"
    )

    def line(q: int, c: int, fam: str, score: float, ca: float, cb: float,
             formula: str, scope: str = "all_candidates"):
        ik = f"IK{q}{c}"
        return (
            f"{q}\t{c}\t{ik}\t{formula}\t{fam}\t{scope}\t{score}\t"
            f"{ca}\t{cb}\t1"
        )

    lines = [header]
    # q0: (0,1) = PI support + NL sub-significant opposition -> vetoed.
    lines += [
        line(0, 0, "synthetic_product_ion", 0.9, 0.5, 0.2, "C10H10NO"),
        line(0, 1, "synthetic_product_ion", 0.5, 0.5, 0.2, "C10H10NO"),
        line(0, 0, "synthetic_neutral_loss", 0.3, 0.5, 0.1, "C10H10NO"),
        line(0, 1, "synthetic_neutral_loss", 0.5, 0.2, 0.4, "C10H10NO"),
        line(0, 0, "synthetic_cross_only", 0.9, 0.0, 0.0, "C10H10NO",
             scope="cross_formula"),
        line(0, 1, "synthetic_cross_only", 0.1, 0.0, 0.0, "C10H10NO",
             scope="cross_formula"),
        line(0, 0, "synthetic_gate_failed", 0.9, 0.0, 0.0, "C10H10NO"),
        line(0, 1, "synthetic_gate_failed", 0.1, 0.0, 0.0, "C10H10NO"),
        # q1: (0,1) unanimous; (0,2) dominant opposition; (0,3) no signal.
        line(1, 0, "synthetic_product_ion", 0.8, 0.4, 0.4, "C6H12O6"),
        line(1, 1, "synthetic_product_ion", 0.4, 0.4, 0.4, "C6H10O6"),
        line(1, 2, "synthetic_product_ion", 0.9, 0.3, 0.3, "C7H8N4O2"),
        line(1, 3, "synthetic_product_ion", 0.75, 0.3, 0.3, "C5H4N4"),
        line(1, 0, "synthetic_neutral_loss", 0.7, 0.3, 0.3, "C6H12O6"),
        line(1, 1, "synthetic_neutral_loss", 0.3, 0.3, 0.3, "C6H10O6"),
        line(1, 2, "synthetic_neutral_loss", 0.9, 0.3, 0.3, "C7H8N4O2"),
        line(1, 3, "synthetic_neutral_loss", 0.68, 0.25, 0.25, "C5H4N4"),
        line(1, 0, "synthetic_cross_only", 0.8, 0.0, 0.0, "C6H12O6",
             scope="cross_formula"),
        line(1, 1, "synthetic_cross_only", 0.2, 0.0, 0.0, "C6H10O6",
             scope="cross_formula"),
        line(1, 2, "synthetic_cross_only", 0.85, 0.1, 0.0, "C7H8N4O2",
             scope="cross_formula"),
        line(1, 3, "synthetic_cross_only", 0.85, 0.1, 0.0, "C5H4N4",
             scope="cross_formula"),
        # q2: (0,1) unanimous; (0,2) dominant opposition -> ambiguous tie.
        line(2, 0, "synthetic_product_ion", 0.8, 0.3, 0.3, "C9H9NO3"),
        line(2, 1, "synthetic_product_ion", 0.4, 0.3, 0.3, "C8H8NO3"),
        line(2, 2, "synthetic_product_ion", 0.9, 0.0, 0.0, "C7H7NO3"),
        line(2, 0, "synthetic_neutral_loss", 0.7, 0.3, 0.3, "C9H9NO3"),
        line(2, 1, "synthetic_neutral_loss", 0.4, 0.3, 0.3, "C8H8NO3"),
        line(2, 2, "synthetic_neutral_loss", 0.9, 0.0, 0.0, "C7H7NO3"),
        # q3: (0,1) no dominant signal; candidate 2 has no ledger rows.
        line(3, 0, "synthetic_product_ion", 0.5, 0.3, 0.3, "C8H9NO2"),
        line(3, 1, "synthetic_product_ion", 0.5, 0.3, 0.3, "C8H9NO2"),
        line(3, 0, "synthetic_neutral_loss", 0.5, 0.3, 0.3, "C8H9NO2"),
        line(3, 1, "synthetic_neutral_loss", 0.5, 0.3, 0.3, "C8H9NO2"),
    ]
    (ledger / "candidate_scores.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    # The registry hash check reads candidate_scores_sha256.
    report = json.loads((ledger / "report.json").read_text(encoding="utf-8"))
    report["candidate_scores_sha256"] = file_sha256(
        ledger / "candidate_scores.tsv",
    )
    (ledger / "report.json").write_text(
        json.dumps(report), encoding="utf-8",
    )
    return manifest_path, cache_dir, ledger


def run_diagnostic(
    root: Path, manifest: Path, cache: Path, output: Path,
    ledgers: list[Path], extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable, "-X", "utf8", str(SCRIPT),
        "--manifest", str(manifest),
        "--cache-dir", str(cache),
        "--output", str(output),
    ]
    for ledger in ledgers:
        command += ["--ledger", str(ledger)]
    return subprocess.run(
        command + list(extra or []),
        capture_output=True, text=True, cwd=str(root),
    )


def end_to_end_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_diag_") as name:
        root = Path(name)
        manifest_path, cache_dir, ledger = build_world(root)
        output = root / "diagnostic.json"
        result = run_diagnostic(
            root, manifest_path, cache_dir, output, [ledger],
            extra=[
                "--expected-current-errors", "4",
                "--expected-admitted-winners", "1",
                "--permutation-draws", "200",
            ],
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "GLM_CHEMAWARE_EVIDENCE_CLASS_DIAGNOSTIC_COMPLETE"
        assert report["formula_role_4_accessed"] is False
        assert set(report["families"]) == {
            "synthetic_product_ion", "synthetic_neutral_loss", "synthetic_cross_only",
        }
        assert "synthetic_gate_failed" not in report["families"]
        assert len(report["class_definitions"]) == 7
        assert len(report["caveats"]) == 4
        assert report["classified_queries"] == 4
        assert report["total_pairs"] == 7
        assert report["pairs_without_joint_family_coverage"] == 0
        assert report["pair_class_counts"] == {
            "unanimous_admitted_style": 2,
            "vetoed_by_subsignificant_oppose": 1,
            "contested_dominant": 0,
            "dominant_opposition_only": 2,
            "no_dominant_signal": 2,
        }
        assert report["current_errors"] == 4
        assert report["tie_errors"] == 1
        assert report["multiple_top_false_ties"] == 1
        assert report["error_boundary_counts"] == {
            "unanimous_admitted_style": 1,
            "vetoed_by_subsignificant_oppose": 1,
            "contested_dominant": 0,
            "dominant_opposition_only": 0,
            "no_dominant_signal": 0,
            "no_ledger_coverage": 1,
            "ambiguous_tie": 1,
        }
        # Anchor exclusion held: query 0 is an error only because its anchor
        # row was dropped from the true candidate's references.
        by_query = {d["query"]: d for d in report["error_boundary_details"]}
        assert by_query[0]["boundary_class"] == "vetoed_by_subsignificant_oppose"
        assert by_query[0]["boundary_tie"] is False
        assert by_query[0]["boundary_classes_per_winner"] == {
            "1": "vetoed_by_subsignificant_oppose",
        }
        # Query 1: tie error, single winner, unanimous boundary.
        assert by_query[1]["boundary_tie"] is True
        assert by_query[1]["boundary_class"] == "unanimous_admitted_style"
        # Query 2: two tied winners with disagreeing classes -> ambiguous.
        assert by_query[2]["winning_false_candidates"] == [1, 2]
        assert by_query[2]["boundary_class"] == "ambiguous_tie"
        assert by_query[2]["boundary_classes_per_winner"] == {
            "1": "unanimous_admitted_style",
            "2": "dominant_opposition_only",
        }
        # Query 3: winning false has no ledger rows -> absent coverage bucket.
        assert by_query[3]["boundary_class"] == "no_ledger_coverage"
        assert by_query[3]["boundary_classes_per_winner"] == {
            "2": "no_ledger_coverage",
        }
        # Uniform expectation: four queries, one per populated class, four
        # errors -> observed equals expectation exactly in every class.
        for entry in report["uniform_baseline"].values():
            if entry["queries_with_this_top_false_boundary"]:
                assert entry["error_boundaries"] == entry["uniform_expected"]
        # Singleton formula clusters: permutation cannot move any label.
        permutation = report["cluster_permutation_test"]
        assert permutation["draws"] == 200
        assert permutation["clusters"] == 4
        for entry in permutation["per_class"].values():
            assert entry["p_depleted"] == 1.0
            assert entry["p_enriched"] == 1.0
        decision = report["decision_summary"]
        assert decision["teachable_under_current_rules"] == 1
        assert decision["recoverable_by_symmetric_significance"] == 1
        assert decision["no_chemical_direction_available"] == 0
        assert decision["boundary_without_ledger_coverage"] == 1
        assert decision["ambiguous_tie_boundaries"] == 1
        assert decision["decision_eligible"] is True


def multi_ledger_merge_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_diag_merge_") as name:
        root = Path(name)
        manifest_path, cache_dir, ledger = build_world(root)
        output_single = root / "single.json"
        assert run_diagnostic(
            root, manifest_path, cache_dir, output_single, [ledger],
        ).returncode == 0
        # Split the tsv into two ledgers over the SAME queries; merged they
        # must reproduce the single-ledger classes exactly.
        source_lines = (
            ledger / "candidate_scores.tsv"
        ).read_text(encoding="utf-8").splitlines()
        pi_lines = [source_lines[0]] + [
            line for line in source_lines[1:]
            if "synthetic_product_ion" in line
        ]
        other_lines = [source_lines[0]] + [
            line for line in source_lines[1:]
            if "synthetic_product_ion" not in line
        ]
        second = root / "ledger_b"
        second.mkdir()
        (second / "candidate_scores.tsv").write_text(
            "\n".join(other_lines) + "\n", encoding="utf-8",
        )
        (ledger / "candidate_scores.tsv").write_text(
            "\n".join(pi_lines) + "\n", encoding="utf-8",
        )
        base_report = json.loads(
            (ledger / "report.json").read_text(encoding="utf-8"),
        )
        families = base_report["source_families"]
        report_a = dict(
            base_report,
            source_families={"synthetic_product_ion": families["synthetic_product_ion"]},
            candidate_scores_sha256=file_sha256(ledger / "candidate_scores.tsv"),
        )
        (ledger / "report.json").write_text(json.dumps(report_a), encoding="utf-8")
        report_b = dict(
            base_report,
            source_families={
                name: body for name, body in families.items()
                if name != "synthetic_product_ion"
            },
            candidate_scores_sha256=file_sha256(second / "candidate_scores.tsv"),
        )
        (second / "report.json").write_text(json.dumps(report_b), encoding="utf-8")
        output_merged = root / "merged.json"
        result = run_diagnostic(
            root, manifest_path, cache_dir, output_merged, [ledger, second],
            extra=["--permutation-draws", "50"],
        )
        assert result.returncode == 0, result.stderr
        single_report = json.loads(output_single.read_text(encoding="utf-8"))
        merged_report = json.loads(output_merged.read_text(encoding="utf-8"))
        assert (
            merged_report["pair_class_counts"]
            == single_report["pair_class_counts"]
        )
        assert (
            merged_report["error_boundary_counts"]
            == single_report["error_boundary_counts"]
        )
        assert set(merged_report["families"]) == set(single_report["families"])


def provenance_drift_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_diag_drift_") as name:
        root = Path(name)
        manifest_path, cache_dir, ledger = build_world(root)
        report_path = cache_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["manifest_sha256"] = "f" * 64
        report_path.write_text(json.dumps(report), encoding="utf-8")
        output = root / "diagnostic.json"
        result = run_diagnostic(root, manifest_path, cache_dir, output, [ledger])
        assert result.returncode != 0
        assert "provenance drift" in (result.stderr + result.stdout)
        assert not output.exists()


def decision_ineligible_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_diag_inel_") as name:
        root = Path(name)
        manifest_path, cache_dir, ledger = build_world(root)
        output = root / "diagnostic.json"
        result = run_diagnostic(
            root, manifest_path, cache_dir, output, [ledger],
            extra=["--expected-current-errors", "999"],
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["decision_summary"]["decision_eligible"] is False


def refuse_overwrite_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="glm_diag_over_") as name:
        root = Path(name)
        manifest_path, cache_dir, ledger = build_world(root)
        output = root / "diagnostic.json"
        output.write_text("sentinel", encoding="utf-8")
        result = run_diagnostic(root, manifest_path, cache_dir, output, [ledger])
        assert result.returncode != 0
        assert output.read_text(encoding="utf-8") == "sentinel"


def main() -> None:
    stance_contract()
    classify_contract()
    end_to_end_contract()
    multi_ledger_merge_contract()
    provenance_drift_contract()
    decision_ineligible_contract()
    refuse_overwrite_contract()
    print("GLM evidence-class diagnostic contracts passed")


if __name__ == "__main__":
    main()
