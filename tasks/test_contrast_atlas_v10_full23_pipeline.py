from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from package_contrast_atlas_v10_method_scores import package_method, validate_bundle_axis
from preflight_contrast_atlas_v10_full23 import (
    EXPECTED_MS_ENTROPY_WHEEL_SHA256,
    EXPECTED_RELEASE_MANIFEST_SHA256,
    EXPECTED_RUNTIME_EQUIVALENCE_ADDENDUM_SHA256,
    parse_artifacts,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ContrastAtlasV10Full23PipelineTest(unittest.TestCase):
    def make_axis(self, root: Path) -> tuple[Path, Path]:
        axis = root / "axis"
        axis.mkdir()
        pairs = pd.DataFrame([
            ("q0", 0, 0, "p0", "A", "A::1", "B", "B::1"),
            ("q0", 0, 1, "p1", "A", "A::1", "B", "B::2"),
            ("q1", 1, 0, "p2", "A", "A::2", "B", "B::1"),
            ("q2", 2, 0, "p3", "B", "B::1", "A", "A::1"),
            ("q2", 2, 1, "p4", "B", "B::1", "A", "A::2"),
            ("q3", 3, 0, "p5", "B", "B::2", "A", "A::1"),
        ], columns=[
            "query_block_id", "query_block_index", "candidate_index_within_block",
            "directed_pair_id", "query_study_id", "query_entity_id",
            "candidate_study_id", "candidate_entity_id",
        ])
        blocks = pd.DataFrame([
            ("q0", 0, 0, 2, 2), ("q1", 1, 2, 3, 1),
            ("q2", 2, 3, 5, 2), ("q3", 3, 5, 6, 1),
        ], columns=[
            "query_block_id", "query_block_index", "candidate_pair_offset_start",
            "candidate_pair_offset_end", "eligible_candidates",
        ])
        registry = pd.DataFrame([
            ("A", "1", "A::1"), ("A", "2", "A::2"),
            ("B", "1", "B::1"), ("B", "2", "B::2"),
        ], columns=["study_id", "raw_entity_id", "entity_id"])
        coverage = pd.DataFrame([("A", 2), ("B", 2)], columns=["study_id", "entities"])
        outputs = {
            "observed_ion_registry.tsv.gz": registry,
            "directed_complete_candidate_pairs.tsv.gz": pairs,
            "query_target_source_blocks.tsv.gz": blocks,
            "study_observed_ion_coverage.tsv": coverage,
        }
        hashes = {}
        for name, frame in outputs.items():
            path = axis / name
            if name.endswith(".gz"):
                frame.to_csv(path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
            else:
                frame.to_csv(path, sep="\t", index=False)
            hashes[name] = sha256(path)
        report = {
            "schema": "contrast_atlas_model_independent_identity_axis_v1",
            "status": "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN",
            "data_role": "development_consumed",
            "denominators": {
                "studies": 2, "entities": 4, "query_target_source_blocks": 4,
                "nonempty_candidate_blocks": 4, "abstain_no_candidate_blocks": 0,
                "directed_candidate_pairs": 6,
            },
            "outputs": hashes,
        }
        report_path = axis / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        contract = json.loads(
            (Path(__file__).with_name("contrast_atlas_v10_method_pair_score_contract_v1.json"))
            .read_text(encoding="utf-8")
        )
        contract["axis"].update({
            "axis_report_sha256": sha256(report_path),
            "directed_pair_axis_sha256": hashes["directed_complete_candidate_pairs.tsv.gz"],
            "query_block_axis_sha256": hashes["query_target_source_blocks.tsv.gz"],
            "directed_candidate_pairs": 6,
            "query_blocks_total": 4,
            "nonempty_candidate_blocks": 4,
            "zero_candidate_blocks": 0,
        })
        contract_path = root / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        return axis, contract_path

    def test_each_method_forms_its_own_complete_block_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, contract = self.make_axis(root)
            left = np.asarray([0.9, 0.1, 0.2, 0.8, 0.3, 0.7])
            right = np.asarray([0.1, 0.9, 0.8, 0.2, 0.7, 0.9])
            package_method(axis, "noise_v1", left, "a" * 64, root / "left", contract)
            package_method(axis, "official_dreams", right, "b" * 64, root / "right", contract)
            left_decisions = pd.read_csv(root / "left/method_block_decisions.tsv.gz", sep="\t")
            right_decisions = pd.read_csv(root / "right/method_block_decisions.tsv.gz", sep="\t")
            self.assertEqual(left_decisions.loc[0, "selected_directed_pair_id"], "p0")
            self.assertEqual(right_decisions.loc[0, "selected_directed_pair_id"], "p1")
            self.assertNotEqual(
                set(pd.read_csv(root / "left/reciprocal_edges.tsv.gz", sep="\t")["reciprocal_edge_id"]),
                set(pd.read_csv(root / "right/reciprocal_edges.tsv.gz", sep="\t")["reciprocal_edge_id"]),
            )

    def test_one_missing_candidate_abstains_whole_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, contract = self.make_axis(root)
            values = np.asarray([0.9, np.nan, 0.2, 0.8, 0.3, 0.7])
            package_method(axis, "full23_v1", values, "c" * 64, root / "full23", contract)
            decisions = pd.read_csv(root / "full23/method_block_decisions.tsv.gz", sep="\t")
            first = decisions.loc[decisions["query_block_id"].eq("q0")].iloc[0]
            self.assertEqual(first["decision_status"], "abstain_incomplete_block_evidence")
            self.assertFalse(bool(first["answered"]))

    def test_wse_zero_evidence_is_method_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, contract = self.make_axis(root)
            values = np.asarray([0.0, 0.0, 0.2, 0.8, 0.3, 0.7])
            matched = np.asarray([0, 0, 1, 1, 1, 1])
            package_method(
                axis, "weighted_spectral_entropy", values, "d" * 64,
                root / "wse", contract, wse_matched_fragments=matched,
            )
            decisions = pd.read_csv(root / "wse/method_block_decisions.tsv.gz", sep="\t")
            first = decisions.loc[decisions["query_block_id"].eq("q0")].iloc[0]
            self.assertEqual(first["decision_status"], "abstain_tied_top1")
            singleton = decisions.loc[decisions["query_block_id"].eq("q1")].iloc[0]
            self.assertEqual(singleton["decision_status"], "selected_unique_top1")

    def test_artifact_parser_rejects_duplicate_names(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_artifacts(["noise_v1=a", "noise_v1=b"])

    def test_release_manifest_addendum_and_runtime_wheel_are_pinned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            sha256(root / "deploy/unified_additive_full23_v1/manifest.json"),
            EXPECTED_RELEASE_MANIFEST_SHA256,
        )
        self.assertEqual(
            sha256(root / "deploy/unified_additive_full23_v1/runtime_equivalence_addendum_20261031.json"),
            EXPECTED_RUNTIME_EQUIVALENCE_ADDENDUM_SHA256,
        )
        sbatch = (root / "tasks/run_contrast_atlas_v10_complete_axis_full23_1gpu.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn(EXPECTED_MS_ENTROPY_WHEEL_SHA256, sbatch)
        self.assertIn("python -m pip install --no-deps --target", sbatch)
        self.assertIn("--runtime-equivalence-addendum", sbatch)
        self.assertIn("--runtime-wheel", sbatch)

    def test_score_bundle_axis_order_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis, _ = self.make_axis(root)
            pairs = pd.read_csv(
                axis / "directed_complete_candidate_pairs.tsv.gz", sep="\t",
                dtype={"directed_pair_id": str},
            )
            blocks = pd.read_csv(
                axis / "query_target_source_blocks.tsv.gz", sep="\t",
                dtype={"query_block_id": str},
            )
            pair_ids = pairs["directed_pair_id"].astype(str).to_numpy(dtype="U")
            query_ids = blocks["query_block_id"].astype(str).to_numpy(dtype="U")
            starts = blocks["candidate_pair_offset_start"].to_numpy(np.int64)
            stops = blocks["candidate_pair_offset_end"].to_numpy(np.int64)
            query_ptr = np.concatenate((starts[:1], stops))
            validate_bundle_axis(axis, pair_ids, query_ptr, query_ids)
            swapped = pair_ids.copy()
            swapped[[0, 1]] = swapped[[1, 0]]
            with self.assertRaisesRegex(RuntimeError, "directed-pair order"):
                validate_bundle_axis(axis, swapped, query_ptr, query_ids)


if __name__ == "__main__":
    unittest.main()
