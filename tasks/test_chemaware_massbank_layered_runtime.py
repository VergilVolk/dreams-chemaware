"""Dependency-minimal runtime contracts for the frozen MassBank 12k corpus."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    return parser.parse_args()


def load_pool(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    args = arguments()
    report = json.loads((args.corpus / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "CHEMAWARE_LAYERED_10K_NATIVE_CORPUS_COMPLETE":
        raise RuntimeError("frozen 12k corpus status is invalid")
    if int(report.get("events", -1)) != 12000 or not all(report.get("gates", {}).values()):
        raise RuntimeError("frozen 12k corpus count or gates are invalid")
    chemical_events = int(report["strata"]["qualified_chemical_boundary"])
    if chemical_events < 1000 or int(report["qualified_chemical_queries"]) < 500:
        raise RuntimeError("frozen chemistry coverage is below the release contract")

    train = load_pool(args.corpus / "train_pool.npz")
    validation = load_pool(args.corpus / "val_pool.npz")
    native_required = {
        "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx",
    }
    annotated_required = {
        "source_query", "negative_candidate", "source_tag", "curriculum_role",
    }
    if native_required.difference(train) or annotated_required.difference(train):
        raise RuntimeError("annotated native train-pool schema is incomplete")
    if native_required.difference(validation):
        raise RuntimeError("native validation-pool schema is incomplete")
    events = len(train["anchor_idx"])
    if events != 12000:
        raise RuntimeError(f"train pool has {events} events instead of 12000")
    if len(train["positive_ptr"]) != events + 1 or len(train["negative_ptr"]) != events + 1:
        raise RuntimeError("native pointer cardinality is invalid")
    if int(train["positive_ptr"][-1]) != len(train["positive_idx"]):
        raise RuntimeError("positive pointer boundary is invalid")
    if int(train["negative_ptr"][-1]) != len(train["negative_idx"]):
        raise RuntimeError("negative pointer boundary is invalid")
    if np.any(np.diff(train["positive_ptr"]) < 1) or np.any(np.diff(train["negative_ptr"]) < 1):
        raise RuntimeError("empty reference list reached the native pool")
    dynamic_mask = np.isin(train["curriculum_role"], [12, 13])
    if int(np.sum(dynamic_mask)) != chemical_events:
        raise RuntimeError("chemical event count differs between report and native pool")
    if count_rows(args.corpus / "chemical_source_event_ledger.tsv") != chemical_events:
        raise RuntimeError("chemical event ledger cardinality is invalid")
    if count_rows(args.corpus / "corpus_event_manifest.tsv") != events:
        raise RuntimeError("corpus event manifest cardinality is invalid")
    print(
        "PASS: frozen ChemAware MassBank-layered runtime contracts "
        f"events={events} chemical_events={chemical_events}",
        flush=True,
    )


if __name__ == "__main__":
    main()
