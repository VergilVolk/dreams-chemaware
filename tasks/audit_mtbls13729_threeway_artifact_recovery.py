"""Audit what can and cannot be recovered from the MTBLS13729 three-way run.

The completed server run produced candidate-level official-DreaMS, E6 and P2b
tables.  A local checkout may contain only the stdout log.  Aggregate JSON in
that log is useful provenance, but it cannot reconstruct feature identities.
This audit makes that boundary machine-readable and emits the minimum list of
server artifacts required for candidate-level biological attribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def json_objects(text: str) -> list[dict[str, Any]]:
    """Extract balanced JSON objects from mixed stdout without guessing lines."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    position = 0
    while True:
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        position = start + length
    return objects


def relative_file_state(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ROOT / "mtbls13729_p2b_2326596.out")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/mtbls13729/threeway_artifact_recovery_audit_v1",
    )
    args = parser.parse_args()
    if not args.log.is_file():
        raise FileNotFoundError(args.log)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    objects = json_objects(args.log.read_text(encoding="utf-8", errors="replace"))
    by_status: dict[str, list[dict[str, Any]]] = {}
    for value in objects:
        status = value.get("status")
        if isinstance(status, str):
            by_status.setdefault(status, []).append(value)

    expected_statuses = {
        "p2b": "mtbls13729_p2b_vs_dreams_inference_complete",
        "e6": "mtbls13729_embedding_retrieval_complete",
        "threeway": "mtbls13729_threeway_annotation_comparison_complete",
    }
    for label, status in expected_statuses.items():
        if status not in by_status:
            raise RuntimeError(f"missing {label} aggregate status in {args.log}: {status}")

    required_relative = [
        "data/mtbls13729/p2b_application_v1/neg_rp__report.json",
        "data/mtbls13729/p2b_application_v1/neg_rp__per_query.csv.gz",
        "data/mtbls13729/p2b_application_v1/neg_rp__dreams_features.csv.gz",
        "data/mtbls13729/p2b_application_v1/neg_rp__p2b_features.csv.gz",
        "data/mtbls13729/p2b_application_v1/pos_rp__report.json",
        "data/mtbls13729/p2b_application_v1/pos_rp__per_query.csv.gz",
        "data/mtbls13729/p2b_application_v1/pos_rp__dreams_features.csv.gz",
        "data/mtbls13729/p2b_application_v1/pos_rp__p2b_features.csv.gz",
        "data/mtbls13729/e6_embedding_application_v1/neg_rp__e6_fixed_v2_sw2__report.json",
        "data/mtbls13729/e6_embedding_application_v1/neg_rp__e6_fixed_v2_sw2__per_query.csv.gz",
        "data/mtbls13729/e6_embedding_application_v1/neg_rp__e6_fixed_v2_sw2__features.csv.gz",
        "data/mtbls13729/e6_embedding_application_v1/pos_rp__e6_fixed_v2_sw2__report.json",
        "data/mtbls13729/e6_embedding_application_v1/pos_rp__e6_fixed_v2_sw2__per_query.csv.gz",
        "data/mtbls13729/e6_embedding_application_v1/pos_rp__e6_fixed_v2_sw2__features.csv.gz",
        "data/mtbls13729/threeway_application_v1/neg_rp__threeway_features.csv.gz",
        "data/mtbls13729/threeway_application_v1/neg_rp__threeway_per_query.csv.gz",
        "data/mtbls13729/threeway_application_v1/pos_rp__threeway_features.csv.gz",
        "data/mtbls13729/threeway_application_v1/pos_rp__threeway_per_query.csv.gz",
        "data/mtbls13729/threeway_application_v1/report.json",
    ]
    files = [relative_file_state(ROOT / relative) for relative in required_relative]
    missing = [item["path"] for item in files if not item["exists"]]

    p2b_reports = by_status[expected_statuses["p2b"]]
    e6_reports = by_status[expected_statuses["e6"]]
    threeway_report = by_status[expected_statuses["threeway"]][-1]
    aggregate = {
        "p2b_panels": {
            str(item.get("panel")): {
                "queries_with_candidates": item.get("queries_with_candidates"),
                "features_with_candidates": item.get("features_with_candidates"),
                "decisions": item.get("decisions"),
                "systems": item.get("systems"),
            }
            for item in p2b_reports
        },
        "e6_panels": {
            str(item.get("panel")): {
                "selected_query_spectra": item.get("selected_query_spectra"),
                "scored_query_spectra": item.get("scored_query_spectra"),
                "linked_features": item.get("linked_features"),
                "annotated_features": item.get("annotated_features"),
                "model": item.get("model"),
                "provenance": item.get("provenance"),
            }
            for item in e6_reports
        },
        "threeway": threeway_report,
    }
    report = {
        "status": "mtbls13729_threeway_artifact_recovery_audit_complete",
        "formal": True,
        "log": args.log.as_posix(),
        "aggregate_json_recovered": True,
        "candidate_level_artifacts_complete": not missing,
        "required_candidate_files": files,
        "missing_candidate_files": missing,
        "aggregate_results": aggregate,
        "permitted_claims_without_candidate_tables": [
            "method-specific annotated-feature counts on the frozen application protocol",
            "aggregate retained/changed/abstained counts",
            "three-way consensus feature counts",
        ],
        "forbidden_claims_without_candidate_tables": [
            "which metabolite identity was corrected",
            "which biological module was uniquely enabled by E6 or P2b",
            "candidate-level accuracy improvement in an application without structure truth",
            "candidate-level overlap with the frozen abundance results",
        ],
        "recovery_action": (
            "Copy the listed result files from the server run. Re-encoding is unnecessary; "
            "candidate-level attribution becomes reproducible once these small CSV/JSON outputs are present."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "server_sync_manifest.txt").write_text("\n".join(required_relative) + "\n", encoding="utf-8")

    lines = [
        "# MTBLS13729 three-way artifact recovery audit",
        "",
        f"- Aggregate JSON recovered from the completed log: **yes**",
        f"- Candidate-level files present locally: **{len(files) - len(missing)}/{len(files)}**",
        f"- Candidate-level biological attribution currently allowed: **{'yes' if not missing else 'no'}**",
        "",
        "The log verifies denominators and aggregate method changes, but it cannot identify which feature or metabolite changed. "
        "Until the candidate-level files are restored, E6/P2b results are engineering coverage evidence only.",
        "",
        "## Missing files",
        "",
        *[f"- `{path}`" for path in missing],
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
