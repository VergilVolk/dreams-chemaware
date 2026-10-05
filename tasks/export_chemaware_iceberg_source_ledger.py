"""Export the qualified ICEBERG teacher as a truth-blind candidate-source ledger.

The existing ICEBERG audit already computed query-conditioned predicted-spectrum
distances and two matched controls.  This exporter only maps those frozen values
back to corrected-manifest candidate indices.  It never reads candidate labels.
Distances are negated so every exported source uses the convention "larger is
better".
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--graph-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--teacher-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_teacher_700_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    graph_report = json.loads((args.graph_dir / "report.json").read_text(encoding="utf-8"))
    teacher_report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if graph_report.get("status") != "chemaware_full_manifest_iceberg_graph_complete":
        raise RuntimeError("ICEBERG graph is not complete")
    if teacher_report.get("status") != "PASS":
        raise RuntimeError("ICEBERG candidate-specificity gate did not pass")
    controls = teacher_report["metrics"]["paired_formula_cluster_bootstrap"]
    required_control_gates = (
        controls["hit1_correct_minus_candidate_swapped"]["formula_cluster_bootstrap_95ci"][0] > 0,
        controls["hit1_correct_minus_peak_permuted"]["formula_cluster_bootstrap_95ci"][0] > 0,
        controls["margin_correct_minus_candidate_swapped"]["formula_cluster_bootstrap_95ci"][0] > 0,
        controls["margin_correct_minus_peak_permuted"]["formula_cluster_bootstrap_95ci"][0] > 0,
    )
    if not all(required_control_gates):
        raise RuntimeError("ICEBERG source did not beat every frozen matched control")

    with np.load(args.manifest, allow_pickle=True) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.graph_dir / "graph.npz", allow_pickle=True) as loaded:
        graph = {key: np.asarray(loaded[key]) for key in loaded.files}
    source_query = np.load(args.graph_dir / "source_query_index.npy").astype(np.int64)
    selected = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    teacher_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    with np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True) as loaded:
        correct = -np.asarray(loaded["correct_score"], dtype=np.float64)
        swapped = -np.asarray(loaded["candidate_swapped_score"], dtype=np.float64)
        permuted = -np.asarray(loaded["peak_permuted_score"], dtype=np.float64)
    if len(source_query) != len(graph["query_row"]):
        raise RuntimeError("ICEBERG graph source-query registry is misaligned")
    if len(teacher_ptr) != len(selected) + 1 or int(teacher_ptr[-1]) != len(correct):
        raise RuntimeError("ICEBERG teacher score registry is misaligned")

    rows: list[dict[str, object]] = []
    manifest_queries: set[int] = set()
    for position, graph_query in enumerate(selected):
        graph_query = int(graph_query)
        manifest_query = int(source_query[graph_query])
        if int(manifest["query_row"][manifest_query]) != int(graph["query_row"][graph_query]):
            raise RuntimeError(f"ICEBERG query row drift at graph query {graph_query}")
        gleft, gright = map(int, graph["query_ptr"][graph_query:graph_query + 2])
        mleft, mright = map(int, manifest["query_ptr"][manifest_query:manifest_query + 2])
        tleft, tright = map(int, teacher_ptr[position:position + 2])
        if gright - gleft != tright - tleft:
            raise RuntimeError(f"ICEBERG candidate-score count drift at graph query {graph_query}")
        manifest_lookup: dict[tuple[str, str], int] = {}
        for local, molecule in enumerate(range(mleft, mright)):
            key = (
                str(manifest["molecule_ik14"][molecule]),
                str(manifest["molecule_formula"][molecule]),
            )
            if key in manifest_lookup:
                raise RuntimeError(f"manifest repeats candidate key {key} at query {manifest_query}")
            manifest_lookup[key] = local
        for offset, molecule in enumerate(range(gleft, gright)):
            key = (
                str(graph["molecule_ik14"][molecule]),
                str(graph["molecule_formula"][molecule]),
            )
            if key not in manifest_lookup:
                raise RuntimeError(f"ICEBERG candidate absent from manifest: query={manifest_query} key={key}")
            score_position = tleft + offset
            rows.append({
                "manifest_query": manifest_query,
                "local_candidate": manifest_lookup[key],
                "ik14": key[0],
                "formula": key[1],
                "source_family": "iceberg_v1_entropy",
                "scope": "all_candidates",
                "source_score": f"{correct[score_position]:.17g}",
                "source_available": 1,
                "control_a_score": f"{swapped[score_position]:.17g}",
                "control_b_score": f"{permuted[score_position]:.17g}",
                "controls_available": 1,
            })
        manifest_queries.add(manifest_query)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_iceberg_source_", dir=args.output.parent))
    try:
        table = temporary / "candidate_scores.tsv"
        with table.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE",
            "truth_fields_exported": False,
            "training_formula_only": True,
            "source_families": {
                "iceberg_v1_entropy": {
                    "scope": "all_candidates",
                    "larger_is_better": True,
                    "matched_controls": ["candidate_swapped", "peak_permuted"],
                    "specificity_gate_passed": True,
                }
            },
            "queries": len(manifest_queries),
            "candidate_rows": len(rows),
            "provenance": {
                "manifest": str(args.manifest),
                "graph": str(args.graph_dir / "graph.npz"),
                "teacher": str(args.teacher_dir / "scores_and_ranks.npz"),
                "graph_sha256": sha256(args.graph_dir / "graph.npz"),
                "teacher_sha256": sha256(args.teacher_dir / "scores_and_ranks.npz"),
            },
        }
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
