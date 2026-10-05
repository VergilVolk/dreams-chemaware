"""GLM spectrum-free candidate baseline (pre-registered 2026-09-30).

MassSpecGym-in-the-Wild shortcut check for
docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md Section 8:
rank each query's candidate molecules WITHOUT any spectral information --
purely by reference-spectrum count (more library spectra = higher rank), ties
broken by IK14 lexicographic order (truth-blind; the manifest stores the
positive first, so any order-preserving tiebreak would leak).  Reports
recall@1 of this baseline next to the expected recall of uniform random
ranking and reference-count asymmetry diagnostics.  Context for interpreting
absolute recall on the frozen panels; NOT a release gate.

Two panel kinds are supported with one code path:
- ``--graph-panel NAME=PATH``: self-contained panel graphs (GNPS gold/silver
  ``panel_<name>.npz`` with query_ptr/molecule_ptr/molecule_label/
  molecule_ik14/candidate_row).
- ``--manifest-panel NAME=PATH``: ChemAware role-2/role-3 evidence files
  (query/formula arrays) resolved against the corrected candidate manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

BASELINE_STATUS = "GLM_SPECTRUM_FREE_CANDIDATE_BASELINE_COMPLETE"
GRAPH_ARRAYS = (
    "query_ptr", "molecule_ptr", "molecule_label", "molecule_ik14",
)
RANKING_RULE = "reference_count_desc_then_ik14_lexicographic_asc_truth_blind"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-panel", action="append", default=[],
        metavar="NAME=PATH",
        help="Self-contained panel graph npz (e.g. GNPS panel files).",
    )
    parser.add_argument(
        "--manifest-panel", action="append", default=[],
        metavar="NAME=PATH",
        help="Role evidence npz (query/formula) resolved against --manifest.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_named(values: list[str], flag: str) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} entries must be NAME=PATH: {value!r}")
        name, raw = value.split("=", 1)
        if not name.strip() or not raw.strip():
            raise ValueError(f"{flag} entries must be NAME=PATH: {value!r}")
        parsed.append((name.strip(), Path(raw)))
    return parsed


def graph_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = [name for name in GRAPH_ARRAYS if name not in loaded.files]
        if missing:
            raise RuntimeError(f"panel graph {path} lacks arrays: {missing}")
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def manifest_view(
    manifest: dict[str, np.ndarray], queries: np.ndarray,
) -> dict[str, np.ndarray]:
    """Gather a panel subgraph; panel queries are scattered through the
    manifest, so molecules and reference rows are re-indexed cumulatively
    instead of sliced."""
    query_ptr = np.asarray(manifest["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(manifest["molecule_ptr"], dtype=np.int64)
    view_query_ptr = [0]
    molecule_indices: list[int] = []
    for query in queries:
        left, right = map(int, query_ptr[int(query):int(query) + 2])
        molecule_indices.extend(range(left, right))
        view_query_ptr.append(len(molecule_indices))
    indices = np.asarray(molecule_indices, dtype=np.int64)
    reference_counts = molecule_ptr[indices + 1] - molecule_ptr[indices]
    if np.any(reference_counts < 1):
        raise RuntimeError("a manifest molecule carries no reference spectrum")
    view_molecule_ptr = np.concatenate((
        np.zeros(1, dtype=np.int64), np.cumsum(reference_counts),
    ))
    rows = np.concatenate([
        np.asarray(
            manifest["pair_candidate_row"][
                int(molecule_ptr[m]):int(molecule_ptr[m + 1])
            ], dtype=np.int64,
        )
        for m in indices
    ]) if len(indices) else np.asarray([], dtype=np.int64)
    return {
        "query_ptr": np.asarray(view_query_ptr, dtype=np.int64),
        "molecule_ptr": view_molecule_ptr,
        "molecule_label": np.asarray(manifest["molecule_label"])[indices],
        "molecule_ik14": np.asarray(manifest["molecule_ik14"])[indices],
        "candidate_row": rows,
    }


def spectrum_free_metrics(graph: dict[str, np.ndarray]) -> dict[str, object]:
    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    label = np.asarray(graph["molecule_label"])
    ik14 = np.asarray(graph["molecule_ik14"]).astype(str)
    counts = np.diff(molecule_ptr)
    if np.any(counts < 1):
        raise RuntimeError("a candidate molecule carries no reference spectrum")
    ranks: list[int] = []
    candidate_counts: list[int] = []
    truth_counts: list[int] = []
    negative_mean_counts: list[float] = []
    for query in range(len(query_ptr) - 1):
        left, right = map(int, query_ptr[query:query + 2])
        truth_positions = np.flatnonzero(label[left:right] == 1)
        if len(truth_positions) != 1 or right - left < 2:
            raise RuntimeError(
                f"query {query} does not carry exactly one positive among >=2 candidates"
            )
        truth_position = int(truth_positions[0])
        local_counts = counts[left:right]
        local_ik14 = ik14[left:right]
        order = sorted(
            range(right - left),
            key=lambda index: (-int(local_counts[index]), str(local_ik14[index])),
        )
        ranks.append(order.index(truth_position) + 1)
        candidate_counts.append(right - left)
        truth_counts.append(int(local_counts[truth_position]))
        negatives = [
            int(local_counts[index])
            for index in range(right - left) if index != truth_position
        ]
        negative_mean_counts.append(float(np.mean(negatives)))
    ranks_a = np.asarray(ranks, dtype=np.int64)
    candidate_counts_a = np.asarray(candidate_counts, dtype=np.int64)
    return {
        "queries": int(len(ranks_a)),
        "ranking_rule": RANKING_RULE,
        "spectrum_free_recall1": float(np.mean(ranks_a == 1)),
        "random_ranking_expected_recall1": float(
            np.mean(1.0 / candidate_counts_a)
        ),
        "candidates_per_query": {
            "min": int(candidate_counts_a.min()),
            "median": float(np.median(candidate_counts_a)),
            "max": int(candidate_counts_a.max()),
        },
        "truth_reference_count_mean": float(np.mean(truth_counts)),
        "negative_reference_count_mean": float(np.mean(negative_mean_counts)),
        "reference_count_asymmetry": (
            "truth candidates carry more library references than negatives; a "
            "high spectrum-free recall@1 marks database-prior shortcut mass, "
            "not spectral ability"
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    graph_panels = parse_named(args.graph_panel, "--graph-panel")
    manifest_panels = parse_named(args.manifest_panel, "--manifest-panel")
    if not graph_panels and not manifest_panels:
        raise RuntimeError("at least one panel is required")
    if manifest_panels and args.manifest is None:
        raise RuntimeError("--manifest-panel requires --manifest")

    report: dict[str, object] = {
        "status": BASELINE_STATUS,
        "preregistration": (
            "docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md"
        ),
        "panels": {},
        "interpretation": (
            "context only: quantifies how much of each panel is solvable "
            "without spectra; not a release gate for either arm"
        ),
    }
    for name, path in graph_panels:
        report["panels"][name] = spectrum_free_metrics(graph_arrays(path))
    if manifest_panels:
        with np.load(args.manifest, allow_pickle=False) as loaded:
            manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
        for name, path in manifest_panels:
            with np.load(path, allow_pickle=False) as loaded:
                if "query" not in loaded.files:
                    raise RuntimeError(f"role panel lacks query registry: {path}")
                queries = np.asarray(loaded["query"], dtype=np.int64)
            if len(np.unique(queries)) != len(queries):
                raise RuntimeError(f"role panel has duplicate queries: {path}")
            view = manifest_view(manifest, queries)
            report["panels"][name] = spectrum_free_metrics(view)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "status": BASELINE_STATUS,
        "panels": {
            name: {
                "recall1": body["spectrum_free_recall1"],
                "random": body["random_ranking_expected_recall1"],
            }
            for name, body in report["panels"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
