#!/usr/bin/env python
"""Split the pooled negative DreaMS benchmark into immutable unit inputs.

The existing MetDNA3 candidate-path implementation is reused verbatim.  This
adapter only materializes its three required tables for each external unit; it
does not calculate network evidence or alter spectral scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_dreams_v1"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_units_v2"),
    )
    parser.add_argument(
        "--external-root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
    )
    args = parser.parse_args()
    inputs = {
        "report": args.benchmark_dir / "report.json",
        "queries": args.benchmark_dir / "queries.csv.gz",
        "scores": args.benchmark_dir / "candidate_scores.csv.gz",
        "transitions": args.benchmark_dir / "transitions.csv.gz",
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    source_report = json.loads(inputs["report"].read_text(encoding="utf-8"))
    if not source_report.get("formal") or not source_report.get("pass_to_negative_bioaware_increment"):
        raise RuntimeError("negative official-DreaMS benchmark did not pass")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    queries = pd.read_csv(inputs["queries"])
    scores = pd.read_csv(inputs["scores"])
    transitions = pd.read_csv(inputs["transitions"])
    unit_reports = []
    for unit_id in sorted(queries["unit_id"].unique()):
        output = args.output_root / str(unit_id)
        output.mkdir()
        local_queries = queries[queries["unit_id"].eq(unit_id)].copy()
        query_ids = set(local_queries["query_id"].astype(str))
        local_scores = scores[scores["query_id"].astype(str).isin(query_ids)].copy()
        local_transition = transitions[
            transitions["query_id"].astype(str).isin(query_ids)
        ].copy()
        if len(local_queries) != len(local_transition):
            raise RuntimeError(f"query/transition mismatch: {unit_id}")
        baseline = local_transition[["query_id", "truth_ik14", "truth_formula"]].copy()
        baseline["baseline_top_candidate"] = local_transition["top_candidate_id"].astype(str)
        baseline["baseline_correct"] = local_transition["baseline_correct"].astype(bool)
        external_manifest_path = args.external_root / str(unit_id) / "cache" / "external_spectra.csv.gz"
        external_tensor_path = args.external_root / str(unit_id) / "cache" / "external_tensors.npz"
        for path in (external_manifest_path, external_tensor_path):
            if not path.exists():
                raise FileNotFoundError(path)
        external_manifest = pd.read_csv(external_manifest_path)
        external_tensors = np.load(external_tensor_path, allow_pickle=False)["external_tensor"]
        if len(external_manifest) != len(external_tensors):
            raise RuntimeError(f"external manifest/tensor mismatch: {unit_id}")
        tensor_position = {
            str(key): position for position, key in enumerate(external_manifest["spectrum_key"])
        }
        missing = set(local_queries["spectrum_key"].astype(str)) - set(tensor_position)
        if missing:
            raise RuntimeError(f"selected query tensors missing for {unit_id}: {len(missing)}")
        query_tensors = np.stack([
            external_tensors[tensor_position[str(key)]]
            for key in local_queries["spectrum_key"]
        ]).astype(np.float32)
        files = {
            "queries": output / "queries.csv.gz",
            "scores": output / "candidate_scores.csv.gz",
            "baseline": output / "raw_transitions.csv.gz",
            "query_tensors": output / "query_tensors.npz",
        }
        local_queries.to_csv(files["queries"], index=False, compression="gzip")
        local_scores.to_csv(files["scores"], index=False, compression="gzip")
        baseline.to_csv(files["baseline"], index=False, compression="gzip")
        np.savez_compressed(files["query_tensors"], query_tensor=query_tensors)
        unit_reports.append({
            "unit_id": str(unit_id), "queries": int(len(local_queries)),
            "identities": int(local_queries["truth_ik14"].nunique()),
            "candidate_pairs": int(len(local_scores)),
            "files": {key: sha256(path) for key, path in files.items()},
        })
    report = {
        "status": "bioaware_metdna3_negative_unit_inputs_complete",
        "formal": True, "units": unit_reports,
        "contracts": {
            "scores_changed": False, "query_selection_changed": False,
            "network_evidence_calculated": False,
        },
        "source": {key: sha256(path) for key, path in inputs.items()},
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
