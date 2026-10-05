#!/usr/bin/env python
"""Freeze the B47 B1 provenance chain without opening annotation truth.

The registry is deliberately separate from model evaluation.  It verifies the
candidate graph, official shared embeddings, truth-blind seeds and U0 nuisance
audit as immutable files, checks their cross-report hashes, and records missing
stages explicitly.  No candidate accuracy or BioAware score is computed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402


EXPECTED = {
    "graph_report": "a47ebd3e3cff35ad8a135e72fbf14a7b72013c21d7248f6b84c7d825141bce5d",
    "queries": "c9551bc5d32fcec93405bd80cbe62880e3ac309c3df18e120123fb5fbf4133be",
    "candidate_references": "64f5940669876db01d17a4b6cd36977d043a930ac471e53d7af3c5018bf67c56",
    "queries_mgf": "302d3ff12beaa4b09ad3555b64c2682e82336597135be34031359c5f9f80a4a0",
    "query_embeddings": "afbd3a029433ab6b0f981d4d8cc41f6c1592de3ebecbd98c60496499db15f798",
    "reference_embeddings": "634e6db40998768bacee337d48a57583f46665f1f5e650606959a541405d01b2",
    "reference_rows": "32b5909b8a69f4e218edabe3aa70a027c3627e50a2d2c700271a09c80a199c9c",
    "query_index": "8ba4e698e0bc4f072baa4ee84fefc7757dc7b84a38fdb005e87affa8e180a745",
    "official_checkpoint": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    "architecture_checkpoint": "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
}
FORBIDDEN_COLUMN_TOKENS = (
    "truth", "label", "correct", "annotation", "phenotype", "disease", "case_control",
)


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def artifact_row(stage: str, name: str, path: Path, expected: str | None = None) -> dict:
    present = path.is_file() and path.stat().st_size > 0
    observed = sha256_file(path) if present else None
    return {
        "stage": stage,
        "name": name,
        "path": str(path),
        "present": bool(present),
        "bytes": int(path.stat().st_size) if present else 0,
        "sha256": observed,
        "expected_sha256": expected,
        "hash_matches_expected": None if expected is None or not present else observed == expected,
    }


def forbidden_headers(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    columns = pd.read_csv(path, nrows=0).columns.astype(str)
    return sorted(
        column for column in columns
        if any(token in column.casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    )


def load_report(directory: Path, status: str) -> dict:
    path = directory / "report.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != status or report.get("formal") is not True:
        raise RuntimeError(f"unexpected formal status in {path}")
    return report


def registry_decisions(
    *, graph_valid: bool, embedding_valid: bool, seed_artifact_valid: bool,
    seed_policy_pass: bool, u0_valid: bool, headers_valid: bool,
) -> dict[str, bool]:
    """Keep provenance completion separate from downstream scientific gates."""
    pass_to_u0 = bool(graph_valid and embedding_valid and headers_valid)
    pass_b1_provenance = bool(pass_to_u0 and seed_artifact_valid and u0_valid)
    return {
        "pass_to_u0": pass_to_u0,
        "pass_b1_provenance": pass_b1_provenance,
        "pass_to_b2_exact_event": bool(pass_b1_provenance and seed_policy_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--u0-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite artifact registry: {output}")

    graph = args.graph_dir.resolve()
    embedding = args.embedding_dir.resolve()
    seeds = args.seed_dir.resolve()
    u0 = args.u0_dir.resolve()
    rows: list[dict] = []
    problems: list[str] = []

    graph_files = {
        "report": (graph / "report.json", EXPECTED["graph_report"]),
        "queries": (graph / "queries.csv.gz", EXPECTED["queries"]),
        "candidate_references": (
            graph / "candidate_references.csv.gz", EXPECTED["candidate_references"]
        ),
        "queries_mgf": (graph / "queries.mgf", EXPECTED["queries_mgf"]),
    }
    rows.extend(artifact_row("graph", key, *value) for key, value in graph_files.items())
    graph_valid = all(row["present"] and row["hash_matches_expected"] is not False for row in rows)
    graph_report: dict = {}
    if graph_valid:
        graph_report = load_report(graph, "bioaware_b47_truthblind_candidate_graph_frozen")
        graph_valid = bool(graph_report.get("pass_to_query_embedding_and_seed_construction"))
        graph_valid &= graph_report.get("contracts") == {
            "P2b_used": False, "algorithm_outputs_opened": False,
            "phenotype_used": False, "truth_opened": False,
        }
    if not graph_valid:
        problems.append("candidate graph is absent, changed, or outside the truth-blind contract")

    embedding_files = {
        "report": (embedding / "report.json", None),
        "query_embeddings": (embedding / "query_embeddings.npy", EXPECTED["query_embeddings"]),
        "reference_embeddings": (
            embedding / "reference_embeddings.npy", EXPECTED["reference_embeddings"]
        ),
        "reference_rows": (embedding / "reference_rows.npy", EXPECTED["reference_rows"]),
        "query_index": (embedding / "query_index.csv", EXPECTED["query_index"]),
    }
    embedding_rows = [artifact_row("embeddings", key, *value) for key, value in embedding_files.items()]
    rows.extend(embedding_rows)
    embedding_valid = all(
        row["present"] and row["hash_matches_expected"] is not False for row in embedding_rows
    )
    embedding_report: dict = {}
    if embedding_valid:
        embedding_report = load_report(
            embedding, "bioaware_b47_truthblind_embeddings_complete"
        )
        contract = embedding_report.get("contracts", {})
        embedding_valid &= contract.get("one_shared_official_encoder") is True
        embedding_valid &= all(
            contract.get(key) is False for key in (
                "truth_opened", "phenotype_used", "algorithm_outputs_opened",
                "candidate_scores_computed", "seed_selection_performed", "model_fitted",
                "P2b_used",
            )
        )
        provenance = embedding_report.get("provenance", {})
        embedding_valid &= provenance.get("graph_report_sha256") == EXPECTED["graph_report"]
        embedding_valid &= provenance.get("queries_sha256") == EXPECTED["queries"]
        embedding_valid &= (
            provenance.get("candidate_references_sha256") == EXPECTED["candidate_references"]
        )
        embedding_valid &= provenance.get("official_checkpoint_sha256") == EXPECTED["official_checkpoint"]
        embedding_valid &= (
            provenance.get("architecture_checkpoint_sha256") == EXPECTED["architecture_checkpoint"]
        )
    if not embedding_valid:
        problems.append("official shared embeddings are absent, changed, or provenance-inconsistent")

    seed_names = (
        "candidate_scores", "query_summaries", "feature_consensus", "seeds_primary", "seeds_strict"
    )
    seed_files = {name: seeds / f"{name}.csv.gz" for name in seed_names}
    seed_rows = [artifact_row("seeds", "report", seeds / "report.json")]
    seed_rows.extend(artifact_row("seeds", name, path) for name, path in seed_files.items())
    rows.extend(seed_rows)
    seed_artifact_valid = all(row["present"] for row in seed_rows)
    seed_report: dict = {}
    seed_policy_pass = False
    seed_failed_gates: list[str] = []
    if seed_artifact_valid:
        seed_report = load_report(seeds, "bioaware_b47_truthblind_seed_construction_complete")
        provenance = seed_report.get("provenance", {})
        seed_artifact_valid &= provenance.get("graph_report_sha256") == EXPECTED["graph_report"]
        seed_artifact_valid &= provenance.get("embedding_report_sha256") == sha256_file(
            embedding / "report.json"
        )
        seed_artifact_valid &= all(
            provenance.get(f"{name}_sha256") == sha256_file(path)
            for name, path in seed_files.items()
        )
        seed_policy_pass = bool(seed_report.get("pass_to_truthblind_event_graph"))
        seed_failed_gates = sorted(
            key for key, value in seed_report.get("gates", {}).items() if not bool(value)
        )
    if not seed_artifact_valid:
        problems.append("truth-blind seed artifact is absent or does not close its provenance chain")

    u0_files = {
        "candidate_aggregates": u0 / "candidate_aggregates.csv.gz",
        "per_query": u0 / "per_query.csv.gz",
    }
    u0_rows = [artifact_row("u0", "report", u0 / "report.json")]
    u0_rows.extend(artifact_row("u0", name, path) for name, path in u0_files.items())
    rows.extend(u0_rows)
    u0_valid = all(row["present"] for row in u0_rows)
    u0_report: dict = {}
    if u0_valid:
        u0_report = load_report(
            u0, "bioaware_b47_u0_reference_multiplicity_adduct_audit_complete"
        )
        provenance = u0_report.get("provenance", {})
        u0_valid &= bool(u0_report.get("pass_to_u1_strong_unary_bakeoff"))
        u0_valid &= provenance.get("graph_report_sha256") == EXPECTED["graph_report"]
        u0_valid &= provenance.get("embedding_report_sha256") == sha256_file(
            embedding / "report.json"
        )
        u0_valid &= all(
            provenance.get(f"{name}_sha256") == sha256_file(path)
            for name, path in u0_files.items()
        )
    if not u0_valid:
        problems.append("U0 nuisance audit is absent or does not close its provenance chain")

    tabular = [
        graph / "queries.csv.gz", graph / "candidate_references.csv.gz",
        embedding / "query_index.csv", *seed_files.values(), *u0_files.values(),
    ]
    forbidden = {
        str(path): forbidden_headers(path) for path in tabular if path.is_file()
    }
    forbidden = {path: columns for path, columns in forbidden.items() if columns}
    headers_valid = not forbidden
    if not headers_valid:
        problems.append(f"forbidden truth/phenotype-like headers found: {forbidden}")

    decisions = registry_decisions(
        graph_valid=graph_valid,
        embedding_valid=embedding_valid,
        seed_artifact_valid=seed_artifact_valid,
        seed_policy_pass=seed_policy_pass,
        u0_valid=u0_valid,
        headers_valid=headers_valid,
    )
    pass_to_u0 = decisions["pass_to_u0"]
    pass_b1_provenance = decisions["pass_b1_provenance"]
    pass_to_b2 = decisions["pass_to_b2_exact_event"]
    scientific_blocks = []
    if seed_artifact_valid and not seed_policy_pass:
        scientific_blocks.append(
            "truth-blind seed policy gate failed: " + ", ".join(seed_failed_gates)
        )
    if args.require_complete and not pass_b1_provenance:
        raise RuntimeError("B47 B1 complete-registry gate failed: " + "; ".join(problems))

    output.mkdir(parents=True, exist_ok=False)
    registry_path = output / "artifact_registry.csv.gz"
    pd.DataFrame(rows).to_csv(
        registry_path, index=False, compression={"method": "gzip", "mtime": 0}
    )
    report = {
        "status": "bioaware_b47_b1_artifact_registry_complete",
        "formal": True,
        "stages": {
            "candidate_graph": {"valid": bool(graph_valid)},
            "official_shared_embeddings": {"valid": bool(embedding_valid)},
            "truthblind_seeds": {
                "artifact_valid": bool(seed_artifact_valid),
                "scientific_policy_pass": bool(seed_policy_pass),
                "failed_policy_gates": seed_failed_gates,
            },
            "u0_reference_bias": {"valid": bool(u0_valid)},
        },
        "artifact_files": int(len(rows)),
        "artifact_bytes": int(sum(row["bytes"] for row in rows)),
        "forbidden_headers": forbidden,
        "gates": {
            "graph_valid": bool(graph_valid),
            "embeddings_valid": bool(embedding_valid),
            "seed_artifact_valid": bool(seed_artifact_valid),
            "u0_valid": bool(u0_valid),
            "truth_and_phenotype_headers_absent": bool(headers_valid),
        },
        "pass_to_u0": pass_to_u0,
        "pass_b1_provenance": pass_b1_provenance,
        "pass_to_b2_exact_event": pass_to_b2,
        "problems": problems,
        "scientific_blocks": scientific_blocks,
        "contracts": {
            "truth_opened": False,
            "phenotype_used": False,
            "performance_computed": False,
            "model_fitted": False,
            "reaction_network_scored": False,
            "P2b_used": False,
        },
        "provenance": {
            "graph_report_sha256": (
                sha256_file(graph / "report.json") if (graph / "report.json").is_file() else None
            ),
            "embedding_report_sha256": (
                sha256_file(embedding / "report.json")
                if (embedding / "report.json").is_file() else None
            ),
            "seed_report_sha256": (
                sha256_file(seeds / "report.json") if (seeds / "report.json").is_file() else None
            ),
            "u0_report_sha256": (
                sha256_file(u0 / "report.json") if (u0 / "report.json").is_file() else None
            ),
            "artifact_registry_sha256": sha256_file(registry_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Provenance and truth-blind denominator registry only. It does not measure "
            "annotation accuracy, reaction-event gain, shared-embedding gain, or SOTA. "
            "A complete B1 registry can coexist with a failed seed policy gate; in that "
            "case exact-event work remains blocked."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
