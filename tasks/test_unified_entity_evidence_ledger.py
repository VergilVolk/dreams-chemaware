#!/usr/bin/env python
"""Dependency-light contract test for the layered entity evidence ledger."""
from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks/build_unified_entity_evidence_ledger.py"


def read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="unified_entity_ledger_") as raw:
        root = Path(raw)
        benchmark = root / "benchmark"
        cache = root / "chem"
        benchmark.mkdir()
        cache.mkdir()
        # Two queries, two candidate molecules each, one spectrum edge per molecule.
        np.savez_compressed(
            benchmark / "panel_identity_disjoint.npz",
            query_row=np.asarray([100, 200], dtype=np.int64),
            query_ptr=np.asarray([0, 2, 4], dtype=np.int64),
            molecule_ptr=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            candidate_row=np.asarray([10, 11, 12, 13], dtype=np.int64),
            molecule_label=np.asarray([1, 0, 1, 0], dtype=np.int8),
            molecule_ik14=np.asarray(["A", "B", "C", "D"]),
            query_ik14=np.asarray(["A", "C"]),
            query_formula=np.asarray(["F1", "F2"]),
            near_query=np.asarray([True, False]),
        )
        methods = np.asarray([
            "noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen",
        ])
        # Noise misses q0, WSE is right on both, P2b misses q1.
        np.savez_compressed(
            root / "scores.npz",
            method_names=methods,
            scores_identity_disjoint=np.asarray([
                [0.4, 0.8, 0.9, 0.2],
                [0.9, 0.3, 0.8, 0.4],
                [0.7, 0.2, 0.3, 0.6],
            ], dtype=np.float32),
        )
        # ChemAware corrects Noise q0 and abstains on q1.
        np.savez_compressed(
            cache / "action_ledger_identity_disjoint.npz",
            query_index=np.asarray([0, 1], dtype=np.int32),
            abstained=np.asarray([False, True]),
            selected_candidate=np.asarray([0, -1], dtype=np.int32),
            deployment_top_candidate=np.asarray([1, 0], dtype=np.int32),
            changed_deployment_top1=np.asarray([True, False]),
        )
        manifest = root / "entities.csv"
        manifest.write_text(
            "panel,query_index,entity_id,qc_pass,reference_status,orthogonal_structure_status\n"
            "identity_disjoint,0,E0,true,known,unavailable\n"
            "identity_disjoint,1,E1,true,unknown,authentic_standard_confirmed\n",
            encoding="utf-8",
        )
        events = root / "events.csv"
        events.write_text(
            "panel,query_index,candidate_id,event_id,event_type,event_score,event_quality_pass\n"
            "identity_disjoint,1,C,EV1,perturbation,0.8,true\n"
            "identity_disjoint,1,D,EV_BAD,coabundance,2.0,false\n",
            encoding="utf-8",
        )
        output = root / "output"
        subprocess.run([
            sys.executable, str(SCRIPT),
            "--benchmark", str(benchmark),
            "--score-bundle", str(root / "scores.npz"),
            "--chemaware-action-cache", str(cache),
            "--entity-manifest", str(manifest),
            "--bioaware-events", str(events),
            "--panel", "identity_disjoint",
            "--output", str(output),
        ], check=True, capture_output=True, text=True)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        panel = report["panels"]["identity_disjoint"]
        assert report["algorithm"]["fusion_model_fitted"] is False
        assert report["algorithm"]["truth_used_for_output_tier"] is False
        assert panel["module_ablation_same_denominator"]["noise_v1"]["recall@1"] == 0.5
        assert panel["module_ablation_same_denominator"]["weighted_spectral_entropy"]["recall@1"] == 1.0
        assert panel["chemaware"]["applicable_queries"] == 1
        assert panel["chemaware"]["corrected_vs_noise"] == 1
        assert panel["chemaware"]["introduced_vs_noise"] == 0
        assert panel["bioaware"]["entities_with_quality_passed_events"] == 1
        assert panel["known_only_control"]["status"] == "partition_materialized_biology_endpoint_required"
        candidates = read_rows(output / "candidate_evidence_identity_disjoint.csv.gz")
        entities = read_rows(output / "entity_evidence_identity_disjoint.csv.gz")
        truth = read_rows(output / "evaluation_truth_identity_disjoint.csv.gz")
        assert len(candidates) == 4 and len(entities) == 2
        assert len(truth) == 4
        assert "evaluation_is_true_candidate" not in candidates[0]
        assert "query_identity" not in candidates[0]
        assert truth[0]["evaluation_is_true_candidate"] == "True"
        assert candidates[0]["chemaware_evidence_status"] == "selected_candidate"
        assert candidates[2]["bioaware_applicable"] == "True"
        assert entities[0]["output_tier"] == "ranked_structure_hypothesis"
        assert entities[1]["output_tier"] == "trusted_structure"
    print("PASS: unified entity evidence ledger contracts")


if __name__ == "__main__":
    main()
