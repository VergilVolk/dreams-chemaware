"""Merge disjoint fragment-graph shards without changing any candidate score."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"table has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    reports = []
    score_fields: list[str] = []
    explanation_fields: list[str] = []
    exclusion_fields: list[str] = []
    scores: list[dict[str, str]] = []
    explanations: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    seen_queries: set[int] = set()
    observed_indices = set()
    expected_count = None
    reference_contract = None
    reference_provenance = None
    reference_configuration = None
    for shard_path in args.shard:
        report = json.loads((shard_path / "report.json").read_text(encoding="utf-8"))
        if report.get("status") != "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED":
            raise RuntimeError(f"shard is not an unqualified source ledger: {shard_path}")
        shard = report.get("shard", {})
        count, index = int(shard.get("count", 0)), int(shard.get("index", -1))
        if expected_count is None:
            expected_count = count
        if count != expected_count or index in observed_indices:
            raise RuntimeError("fragment-graph shard index/count drift")
        observed_indices.add(index)
        contract = report["source_families"]
        provenance = {
            key: value for key, value in report["provenance"].items()
            if key not in {"candidate_scores_sha256", "connected_explanations_sha256"}
        }
        configuration = report["configuration"]
        if reference_contract is None:
            reference_contract = contract
            reference_provenance = provenance
            reference_configuration = configuration
        if contract != reference_contract or provenance != reference_provenance:
            raise RuntimeError("fragment-graph shard source/provenance drift")
        if configuration != reference_configuration:
            raise RuntimeError("fragment-graph shard configuration drift")
        fields, rows = read_table(shard_path / "candidate_scores.tsv")
        if score_fields and fields != score_fields:
            raise RuntimeError("candidate score schema drift between shards")
        score_fields = fields
        shard_queries = {int(row["manifest_query"]) for row in rows}
        if seen_queries.intersection(shard_queries):
            raise RuntimeError("fragment-graph shards overlap in manifest query")
        seen_queries.update(shard_queries)
        scores.extend(rows)
        fields, rows = read_table(shard_path / "connected_explanations.tsv")
        if explanation_fields and fields != explanation_fields:
            raise RuntimeError("connected explanation schema drift between shards")
        explanation_fields = fields or explanation_fields
        explanations.extend(rows)
        fields, rows = read_table(shard_path / "exclusions.tsv")
        if exclusion_fields and fields and fields != exclusion_fields:
            raise RuntimeError("exclusion schema drift between shards")
        exclusion_fields = fields or exclusion_fields
        exclusions.extend(rows)
        reports.append(report)
    if expected_count is None or observed_indices != set(range(expected_count)):
        raise RuntimeError(
            f"incomplete shard set: expected={expected_count} observed={sorted(observed_indices)}"
        )
    scores.sort(key=lambda row: (
        int(row["manifest_query"]), int(row["local_candidate"]), row["source_family"],
    ))
    explanations.sort(key=lambda row: (
        int(row["manifest_query"]), int(row["local_candidate"]), row["arm"],
    ))
    exclusions.sort(key=lambda row: (
        int(row["manifest_query"]), int(row["local_candidate"]),
    ))
    report = json.loads(json.dumps(reports[0]))
    report["queries"] = len(seen_queries)
    report["candidates"] = len(scores)
    report["candidate_source_rows"] = len(scores)
    report["active_queries"] = sum(int(body["active_queries"]) for body in reports)
    report["unique_fragment_graphs_shard_sum_upper_bound"] = sum(
        int(body["unique_fragment_graphs"]) for body in reports
    )
    report.pop("unique_fragment_graphs", None)
    report["excluded_candidate_rows"] = len(exclusions)
    audit_keys = sorted({key for body in reports for key in body.get("audit", {})})
    report["audit"] = {
        key: sum(int(body.get("audit", {}).get(key, 0)) for body in reports)
        for key in audit_keys
    }
    report["shard"] = {
        "count": expected_count,
        "indices": sorted(observed_indices),
        "merged": True,
        "input_reports_sha256": [sha256_file(path / "report.json") for path in args.shard],
    }
    report["formal_triplet_mining_authorized"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_fragment_graph_merge_", dir=args.output.parent))
    try:
        write_table(temporary / "candidate_scores.tsv", score_fields, scores)
        write_table(
            temporary / "connected_explanations.tsv", explanation_fields, explanations,
        )
        write_table(temporary / "exclusions.tsv", exclusion_fields, exclusions)
        report["candidate_scores_sha256"] = sha256_file(temporary / "candidate_scores.tsv")
        if explanations:
            report["connected_explanations_sha256"] = sha256_file(
                temporary / "connected_explanations.tsv"
            )
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
