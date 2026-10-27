#!/usr/bin/env python3
"""Integrity tests for the 2026-10-27 Contrast Atlas biology handoff."""

from __future__ import annotations

import csv
import gzip
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ContrastAtlasBiologyDeliveryTest(unittest.TestCase):
    def test_common_identity_partition_is_the_method_partition(self) -> None:
        path = ROOT / "data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/identity_partition.tsv.gz"
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 54861)
        self.assertEqual(
            Counter(row["identity_status"] for row in rows),
            Counter({"unresolved_entity": 53882, "library_supported": 979}),
        )

    def test_eligibility_denominators_and_truth_boundary(self) -> None:
        report = json.loads((ROOT / "data/validation/contrast_atlas_contrast_eligibility_v3_20261027/report.json").read_text())
        denominator = report["denominators"]
        self.assertEqual(denominator["source_families"], 15)
        self.assertEqual(denominator["study_level_contrasts"], 51)
        self.assertEqual(denominator["response_rows"], 238412)
        self.assertEqual(denominator["cross_project_primary_connected_components"], 2)
        self.assertEqual(denominator["contrasts_with_signed_truth"], 0)
        self.assertEqual(denominator["external_validation_contrasts"], 0)

    def test_complete_axis_is_model_independent(self) -> None:
        directory = ROOT / "data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"
        report = json.loads((directory / "report.json").read_text())
        definition = report["candidate_definition"]
        self.assertFalse(definition["model_scores_read"])
        self.assertTrue(definition["response_table_read_for_entity_lineage"])
        self.assertFalse(definition["response_beta_se_or_relation_truth_used_for_admission"])
        self.assertEqual(report["denominators"]["directed_candidate_pairs"], 291330)
        self.assertEqual(report["denominators"]["query_target_source_blocks"], 768054)

        previous_end = 0
        with gzip.open(directory / "query_target_source_blocks.tsv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            self.assertIn("query_block_id", reader.fieldnames or [])
            self.assertIn("candidate_pair_offset_start", reader.fieldnames or [])
            for expected_index, row in enumerate(reader):
                self.assertEqual(int(row["query_block_index"]), expected_index)
                self.assertEqual(int(row["candidate_pair_offset_start"]), previous_end)
                self.assertEqual(
                    int(row["candidate_pair_offset_end"]) - int(row["candidate_pair_offset_start"]),
                    int(row["eligible_candidates"]),
                )
                previous_end = int(row["candidate_pair_offset_end"])
        self.assertEqual(previous_end, 291330)

    def test_ibd_handoff_is_truth_and_response_blind(self) -> None:
        directory = ROOT / "data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027"
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["denominators"]["complete_block_candidate_rows"], 24)
        self.assertEqual(report["denominators"]["stercobilin_like_bh_q_lt_0_05"], 1)
        self.assertEqual(len(report["provenance"]["supporting_evidence_sha256_by_path"]), 9)
        with gzip.open(directory / "complete_block_algorithm_handoff.tsv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            rows = list(reader)
        self.assertTrue(rows)
        forbidden = ("beta", "se", "candidate_set", "expected", "role", "score")
        self.assertFalse([field for field in fields if any(token in field.lower() for token in forbidden)])
        with gzip.open(directory / "pair_adjudication_evidence.tsv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
            evidence_rows = list(csv.DictReader(handle, delimiter="\t"))
        roles = Counter(row["registered_pair_role"] for row in evidence_rows)
        self.assertGreater(roles["expected_identity_pair"], 0)
        self.assertGreater(roles["known_same_mass_competitor_pair"], 0)

    def test_ledger_reserves_both_query_directions_and_missing_scores_are_empty(self) -> None:
        path = ROOT / "data/validation/contrast_atlas_entity_change_ledger_v7_20261027/entity_change_ledger.tsv.gz"
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            rows = list(reader)
        self.assertIn("full23_v1_left_to_right_score", fields)
        self.assertIn("full23_v1_right_to_left_score", fields)
        self.assertIn("left_to_right_directed_pair_id", fields)
        self.assertIn("right_to_left_directed_pair_id", fields)
        self.assertNotIn("full23_v1_score", fields)
        score_fields = [field for field in fields if field.endswith("_score") and field.startswith(("noise_", "wse_", "full23_"))]
        self.assertTrue(score_fields)
        self.assertTrue(all(row[field] == "" for row in rows for field in score_fields))
        self.assertEqual({row["full23_v1_score_availability"] for row in rows}, {"not_delivered_on_complete_block_axis"})


if __name__ == "__main__":
    unittest.main()
