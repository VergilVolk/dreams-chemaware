#!/usr/bin/env python
"""Fail-closed replay of frozen BioAware B2 outer-study artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "tasks"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from annotation.bioaware_context_adapter import (
    BiologicalEvidenceContextAdapter,
    MonotoneEvidenceTangentLift,
)
from train_bioaware_b2_leave_study_out import Dataset, score_query, sha256, strict_rank


def load_artifact(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch releases do not expose weights_only
        return torch.load(path, map_location="cpu")


def build_artifact_model(configuration: dict, device: torch.device) -> torch.nn.Module:
    adapter_type = str(configuration.get("adapter_type", "free"))
    if adapter_type == "free":
        return BiologicalEvidenceContextAdapter(
            configuration["embedding_dim"], configuration["evidence_dim"],
            hidden_dim=configuration["hidden_dim"],
            update_rank=configuration["update_rank"],
            delta_bound=configuration["delta_bound"],
        ).to(device)
    if adapter_type == "monotone_evidence_lift":
        return MonotoneEvidenceTangentLift(
            configuration["embedding_dim"], configuration["evidence_dim"],
            delta_bound=configuration["delta_bound"],
        ).to(device)
    raise RuntimeError(f"unknown artifact adapter type: {adapter_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    destination = args.input_dir / "replay.json"
    if destination.exists():
        raise RuntimeError(f"fail-closed: replay already exists: {destination}")

    report_path = args.input_dir / "report.json"
    transitions_path = args.input_dir / "query_oof_transitions.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["provenance"]["dataset_sha256"] != sha256(args.dataset):
        raise RuntimeError("dataset hash differs from training report")
    if report["provenance"]["transitions_sha256"] != sha256(transitions_path):
        raise RuntimeError("transition hash differs from training report")
    expected = pd.read_csv(transitions_path).set_index("query_id")
    device = torch.device(args.device)
    data = Dataset(args.dataset, device, report["configuration"]["evidence_mode"])

    by_study: dict[str, list[dict]] = {}
    for item in report["run_reports"]:
        by_study.setdefault(str(item["outer_study"]), []).append(item)
    rank_mismatches = 0
    baseline_rank_mismatches = 0
    replayed = 0
    artifact_hashes: dict[str, str] = {}
    for outer_study in sorted(by_study):
        heldout = np.flatnonzero(data.study_ids == outer_study)
        ensemble: dict[int, list[np.ndarray]] = {int(index): [] for index in heldout}
        for item in sorted(by_study[outer_study], key=lambda value: int(value["seed"])):
            artifact_name = Path(str(item["artifact"])).name
            artifact_path = args.input_dir / artifact_name
            if not artifact_path.exists():
                raise FileNotFoundError(artifact_path)
            observed_hash = sha256(artifact_path)
            if observed_hash != item["artifact_sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {artifact_name}")
            artifact_hashes[artifact_name] = observed_hash
            body = load_artifact(artifact_path)
            configuration = body["configuration"]
            if configuration["evidence_mode"] != data.evidence_mode:
                raise RuntimeError(f"artifact evidence mode mismatch: {artifact_name}")
            model = build_artifact_model(configuration, device)
            model.load_state_dict(body["state_dict"], strict=True)
            model.eval()
            mean = torch.from_numpy(np.asarray(body["evidence_mean"], dtype=np.float32)).to(device)
            scale = torch.from_numpy(np.asarray(body["evidence_scale"], dtype=np.float32)).to(device)
            for index in heldout:
                scores, _, _ = score_query(
                    model, data, int(index), mean, scale,
                    str(configuration.get("adapter_type", "free")),
                )
                ensemble[int(index)].append(scores)
        for index in heldout:
            index = int(index)
            scores = np.mean(np.stack(ensemble[index]), axis=0)
            section = data.section(index)
            baseline = (data.candidate[section] @ data.query[index]).cpu().numpy()
            positive = int(data.positive[index])
            query_id = str(data.query_ids[index])
            if query_id not in expected.index:
                raise RuntimeError(f"replayed query absent from transitions: {query_id}")
            row = expected.loc[query_id]
            baseline_rank_mismatches += int(strict_rank(baseline, positive) != int(row.baseline_rank))
            rank_mismatches += int(strict_rank(scores, positive) != int(row.final_rank))
            replayed += 1

    result = {
        "status": "bioaware_b2_artifact_replay_passed",
        "queries": replayed,
        "expected_queries": int(len(expected)),
        "artifacts": len(artifact_hashes),
        "baseline_rank_mismatches": baseline_rank_mismatches,
        "final_rank_mismatches": rank_mismatches,
        "artifact_sha256": artifact_hashes,
        "dataset_sha256": sha256(args.dataset),
    }
    if replayed != len(expected) or baseline_rank_mismatches or rank_mismatches:
        raise RuntimeError(f"artifact replay failed: {result}")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
