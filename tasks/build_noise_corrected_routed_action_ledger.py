"""Combine routed N/P/A4/V4 actions into one bounded multi-action ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd

from noise_corrected_action_routing import select_diverse_routed_actions
from noise_corrected_action_routing_v3 import select_diverse_routed_actions_v3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ALLOWED_STATUSES = {
    "noise_corrected_full_p_router_audit_complete",
    "noise_corrected_n_router_audit_complete",
    "noise_corrected_a4_router_audit_complete",
    "noise_corrected_v4_action_router_audit_complete",
}


def _validate_shared_clean_ranks(table: pd.DataFrame) -> dict[str, object]:
    """Require every route source to reference one immutable E8 geometry."""
    required = {"query_index", "source", "clean_rank"}
    if missing := required - set(table.columns):
        raise RuntimeError(
            f"routed action table misses clean geometry: {sorted(missing)}"
        )
    numeric = pd.to_numeric(table.clean_rank, errors="coerce")
    if numeric.isna().any() or (numeric < 1).any():
        raise RuntimeError("routed clean rank is missing or invalid")
    geometry = table.assign(_clean_rank=numeric.astype(np.int64)).groupby(
        "query_index", sort=True,
    ).agg(
        clean_rank_values=(
            "_clean_rank", lambda values: tuple(sorted(set(values))),
        ),
        source_values=(
            "source", lambda values: tuple(sorted(set(map(str, values)))),
        ),
    )
    disagreements = geometry.loc[geometry.clean_rank_values.map(len).gt(1)]
    if len(disagreements):
        sample = {
            str(int(query)): {
                "clean_ranks": list(map(int, row.clean_rank_values)),
                "sources": list(row.source_values),
            }
            for query, row in disagreements.head(10).iterrows()
        }
        raise RuntimeError(
            "routed sources disagree on immutable current-E8 clean rank: "
            + json.dumps({
                "queries": int(len(disagreements)), "sample": sample,
            }, sort_keys=True)
        )
    return {
        "queries": int(len(geometry)),
        "multi_source_queries": int(
            geometry.source_values.map(len).gt(1).sum()
        ),
        "rank_disagreement_queries": 0,
        "passed": True,
    }


def _validate_control_semantics(table: pd.DataFrame, route_dir: Path) -> None:
    allowed = {
        "N_mature": {"matched_neutral", "clean_fallback"},
        "A4_exact": {"matched_neutral", "clean_fallback"},
        "V4_gradient_path": {"matched_neutral"},
        "P_guided_original": {"wrong_identity_direction"},
        "E10B": {"wrong_identity_direction"},
        "E11": {"wrong_identity_direction"},
        "E12B": {"wrong_identity_direction"},
    }
    for source, block in table.groupby("source", sort=True):
        source = str(source)
        observed = set(block.control_semantic.astype(str))
        if source not in allowed or not observed or not observed <= allowed[source]:
            raise RuntimeError(
                f"control semantics do not match source {source}: {observed}; {route_dir}"
            )
    kind_semantic = table[["control_kind", "control_semantic"]].astype(str)
    invalid = (
        kind_semantic.control_kind.eq("wrong_identity_direction")
        != kind_semantic.control_semantic.eq("wrong_identity_direction")
    ) | (
        kind_semantic.control_kind.eq("clean_fallback")
        != kind_semantic.control_semantic.eq("clean_fallback")
    )
    if invalid.any():
        raise RuntimeError(f"control kind/semantic mapping drifted: {route_dir}")


def _read_npy_header(stream: object) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read a small NPY header without materializing the array payload."""
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise RuntimeError(f"unsupported NPY member version: {version}")
    return tuple(map(int, shape)), bool(fortran_order), np.dtype(dtype)


def _read_exact(stream: object, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("truncated NPY payload inside routed action archive")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_selected_npz_rows(
    path: Path,
    member: str,
    selected_indices: np.ndarray,
    *,
    expected_ids: pd.Series | None = None,
    chunk_rows: int = 8192,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Stream one compressed NPY member and retain only requested leading rows.

    ``numpy.load`` inflates a complete compressed member.  A formal P route can
    contain millions of rows, so inflating action, control and ID arrays before
    selection creates a multi-tens-of-GB transient.  NPY rows are fixed width;
    reading them sequentially keeps the peak buffer bounded while preserving
    exact tensor indices and validating the complete ID ledger.
    """
    requested = np.asarray(selected_indices, dtype=np.int64)
    if requested.ndim != 1 or len(np.unique(requested)) != len(requested):
        raise RuntimeError("selected source tensor indices must be a unique vector")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    archive_member = f"{member}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        if archive_member not in archive.namelist():
            raise RuntimeError(f"routed action spectra schema misses {member}: {path}")
        with archive.open(archive_member, "r") as stream:
            shape, fortran_order, dtype = _read_npy_header(stream)
            if fortran_order or not shape:
                raise RuntimeError(f"unsupported routed tensor layout for {member}: {path}")
            rows = int(shape[0])
            if np.any((requested < 0) | (requested >= rows)):
                raise RuntimeError(f"selected tensor index is outside {member}: {path}")
            if expected_ids is not None and len(expected_ids) != rows:
                raise RuntimeError(f"routed action ID count differs from table: {path}")
            row_shape = shape[1:]
            values_per_row = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1
            bytes_per_row = values_per_row * dtype.itemsize
            output = np.empty((len(requested), *row_shape), dtype=dtype)
            if not len(requested) and expected_ids is None:
                return output, shape
            order = np.argsort(requested, kind="stable")
            sorted_requested = requested[order]
            for left in range(0, rows, chunk_rows):
                right = min(rows, left + chunk_rows)
                raw = _read_exact(stream, (right - left) * bytes_per_row)
                block = np.frombuffer(raw, dtype=dtype).reshape((right - left, *row_shape))
                if expected_ids is not None:
                    observed = np.asarray(block, dtype=str).reshape(right - left)
                    expected = expected_ids.iloc[left:right].astype(str).to_numpy()
                    if not np.array_equal(observed, expected):
                        raise RuntimeError(f"routed action tensor alignment failed: {path}")
                begin = int(np.searchsorted(sorted_requested, left, side="left"))
                end = int(np.searchsorted(sorted_requested, right, side="left"))
                if begin < end:
                    output[order[begin:end]] = block[sorted_requested[begin:end] - left]
    return output, shape


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--route-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum-corrective-per-query", type=int, default=16)
    parser.add_argument("--maximum-risk-per-query", type=int, default=8)
    parser.add_argument("--maximum-robust-per-query", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_ledger(
    graph_dir: Path,
    route_dirs: list[Path],
    output_dir: Path,
    *,
    maximum_corrective_per_query: int,
    maximum_risk_per_query: int,
    maximum_robust_per_query: int = 0,
) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    graph_path = graph_dir / "candidate_graph.npz"
    graph_report_path = graph_dir / "report.json"
    if not graph_path.is_file() or not graph_report_path.is_file():
        raise FileNotFoundError("corrected candidate graph is incomplete")
    graph_sha = sha256_file(graph_path)
    graph_report_sha = sha256_file(graph_report_path)
    frames: list[pd.DataFrame] = []
    reports: list[tuple[Path, dict[str, object]]] = []
    for route_source_index, route_dir in enumerate(route_dirs):
        table_path = route_dir / "routed_actions.csv.gz"
        spectra_path = route_dir / "action_spectra.npz"
        report_path = route_dir / "report.json"
        if not all(path.is_file() for path in (table_path, spectra_path, report_path)):
            raise FileNotFoundError(f"routed action source is incomplete: {route_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") not in ALLOWED_STATUSES:
            raise RuntimeError(f"unregistered routed action source: {route_dir}")
        if (
            report.get("provenance", {}).get("candidate_graph_sha256") != graph_sha
            or report.get("provenance", {}).get("graph_report_sha256") != graph_report_sha
            or report.get("provenance", {}).get("action_spectra_sha256") != sha256_file(spectra_path)
            or report.get("contracts", {}).get("outer_held_formula_consumed") is not False
            or report.get("contracts", {}).get("teacher_embedding_target_used") is not False
            or report.get("contracts", {}).get("P3_consumed") is not False
            or report.get("contracts", {}).get("control_semantics_explicit") is not True
            or report.get("contracts", {}).get(
                "registered_formal_route_configuration_verified"
            ) is not True
        ):
            raise RuntimeError(f"routed action provenance failed: {route_dir}")
        with zipfile.ZipFile(spectra_path, "r") as archive:
            expected_members = {
                "action_ids.npy", "action_spectra.npy", "control_spectra.npy",
            }
            if set(archive.namelist()) != expected_members:
                raise RuntimeError(f"routed action spectra schema failed: {route_dir}")
        table = pd.read_csv(table_path, low_memory=False)
        required = {
            "action_id", "query_index", "query_row", "query_ik14", "query_formula",
            "formula_fold", "source", "family", "route", "margin_change",
            "paired_advantage", "recipe_id", "action_tensor_index", "control_kind",
            "control_semantic",
        }
        if missing := required - set(table.columns):
            raise RuntimeError(f"routed action table misses {sorted(missing)}")
        _validate_control_semantics(table, route_dir)
        indices = table.action_tensor_index.to_numpy(np.int64)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"routed action tensor indices are not unique: {route_dir}")
        # Preserve the source-local tensor index until bounded selection has
        # finished.  Loading any spectra here was the formal-scale OOM bug.
        table["_route_source_index"] = int(route_source_index)
        table["_source_action_tensor_index"] = indices
        frames.append(table)
        reports.append((route_dir, report))
    if not frames:
        raise RuntimeError("no routed action sources were supplied")
    initial_hashes = {
        str(report.get("model_provenance", {}).get("initial_student_checkpoint_sha256"))
        for _, report in reports
    }
    if len(initial_hashes) != 1 or "None" in initial_hashes:
        raise RuntimeError("routed sources do not share one initialization")
    outer_folds = {int(report["outer_formula_fold"]) for _, report in reports}
    if len(outer_folds) != 1:
        raise RuntimeError("routed sources use different outer folds")
    outer_fold = next(iter(outer_folds))
    formula_fold_seeds = {
        int(report.get("configuration", {}).get("formula_fold_seed", -1))
        for _, report in reports
    }
    if formula_fold_seeds != {20260825}:
        raise RuntimeError(
            f"routed sources do not share the registered formula fold seed: "
            f"{formula_fold_seeds}"
        )
    candidate_switch_sources = [
        str(route_dir) for route_dir, source_report in reports
        if source_report.get("contracts", {}).get(
            "exact_action_control_candidate_switch_rows_recorded"
        ) is True
    ]
    frontier_sources = []
    for route_dir, source_report in reports:
        if source_report.get("contracts", {}).get(
            "selector_frontier_is_lossless_for_global_caps",
        ) is True:
            frontier = source_report.get("selector_frontier", {})
            required_limits = {
                "maximum_corrective_per_query": maximum_corrective_per_query,
                "maximum_harmful_per_query": maximum_risk_per_query,
                "maximum_robust_per_query": maximum_robust_per_query,
            }
            if any(int(frontier.get(key, -1)) < value for key, value in required_limits.items()):
                raise RuntimeError(
                    f"source selector frontier is narrower than global caps: {route_dir}"
                )
            frontier_sources.append(str(route_dir))
    routed = pd.concat(frames, ignore_index=True, sort=False)
    if routed.action_id.astype(str).duplicated().any():
        raise RuntimeError("combined routed action IDs are not unique")
    clean_geometry = _validate_shared_clean_ranks(routed)
    if routed.formula_fold.eq(outer_fold).any():
        raise RuntimeError("outer-held formula leaked into routed ledger")
    if maximum_robust_per_query:
        routed = select_diverse_routed_actions_v3(
            routed,
            maximum_corrective_per_query=maximum_corrective_per_query,
            maximum_harmful_per_query=maximum_risk_per_query,
            maximum_robust_per_query=maximum_robust_per_query,
        )
        # Keep the old risk name as a compatibility alias.  The v3 trainer
        # consumes the explicit three-way supervision kind.
        routed["selected_risk"] = routed.selected_harmful
        selected = routed.loc[
            routed.selected_corrective | routed.selected_harmful | routed.selected_robust
        ].copy()
        selected["supervision_kind"] = np.select(
            [
                selected.selected_corrective,
                selected.selected_harmful,
                selected.selected_robust,
            ],
            ["corrective", "harmful", "robust"],
            default="invalid",
        )
    else:
        routed = select_diverse_routed_actions(
            routed,
            maximum_corrective_per_query=maximum_corrective_per_query,
            maximum_risk_per_query=maximum_risk_per_query,
        )
        selected = routed.loc[routed.selected_corrective | routed.selected_risk].copy()
        selected["supervision_kind"] = np.where(
            selected.selected_corrective, "corrective", "harmful",
        )
    selected = selected.sort_values(
        ["supervision_kind", "query_index", "source", "family", "action_id"],
        kind="stable",
    ).reset_index(drop=True)
    if not len(selected) or not selected.supervision_kind.eq("corrective").any():
        raise RuntimeError("routed ledger has no corrective supervision")
    selected_action: np.ndarray | None = None
    selected_control: np.ndarray | None = None
    selected_ids = selected.action_id.astype(str).to_numpy()
    for route_source_index, (route_dir, _) in enumerate(reports):
        source_mask = selected._route_source_index.eq(route_source_index).to_numpy()
        source_positions = np.flatnonzero(source_mask)
        source_indices = selected.loc[
            source_mask, "_source_action_tensor_index"
        ].to_numpy(np.int64)
        source_table = frames[route_source_index]
        spectra_path = route_dir / "action_spectra.npz"
        streamed_ids, id_shape = _stream_selected_npz_rows(
            spectra_path,
            "action_ids",
            source_indices,
            expected_ids=source_table.action_id,
        )
        if not np.array_equal(
            np.asarray(streamed_ids, dtype=str), selected_ids[source_positions],
        ):
            raise RuntimeError(f"selected routed action IDs drifted: {route_dir}")
        source_action, action_shape = _stream_selected_npz_rows(
            spectra_path, "action_spectra", source_indices,
        )
        source_control, control_shape = _stream_selected_npz_rows(
            spectra_path, "control_spectra", source_indices,
        )
        if (
            id_shape[0] != action_shape[0]
            or action_shape != control_shape
            or len(action_shape) != 3
        ):
            raise RuntimeError(f"routed action spectra shape failed: {route_dir}")
        if selected_action is None:
            selected_action = np.empty(
                (len(selected), *action_shape[1:]), dtype=np.float32,
            )
            selected_control = np.empty_like(selected_action)
        elif selected_action.shape[1:] != action_shape[1:]:
            raise RuntimeError("routed action sources use different spectrum shapes")
        selected_action[source_positions] = source_action.astype(np.float32, copy=False)
        assert selected_control is not None
        selected_control[source_positions] = source_control.astype(np.float32, copy=False)
    if selected_action is None or selected_control is None:
        raise RuntimeError("selected routed action spectra were not materialized")

    internal_columns = ["_route_source_index", "_source_action_tensor_index"]
    routed_output = routed.drop(columns=internal_columns)
    selected = selected.drop(columns=internal_columns)
    selected["action_tensor_index"] = np.arange(len(selected), dtype=np.int64)

    formal = all(
        report.get("formal_training_authorized") is True for _, report in reports
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        routed_output.to_csv(
            staging / "routing_ledger.csv.gz", index=False, compression="gzip",
        )
        selected.to_csv(staging / "training_actions.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=np.asarray(selected.action_id.astype(str), dtype=str),
            action_spectra=selected_action,
            control_spectra=selected_control,
        )
        report: dict[str, object] = {
            "status": "noise_corrected_routed_action_ledger_complete",
            "formal": formal, "formal_training_authorized": formal,
            "outer_formula_fold": outer_fold,
            "routed_rows": int(len(routed)), "selected_rows": int(len(selected)),
            "selected_queries": int(selected.query_index.nunique()),
            "selected_corrective_rows": int(selected.supervision_kind.eq("corrective").sum()),
            "selected_risk_rows": int(selected.supervision_kind.eq("harmful").sum()),
            "selected_robust_rows": int(selected.supervision_kind.eq("robust").sum()),
            "corrective_queries": int(selected.loc[
                selected.supervision_kind.eq("corrective"), "query_index"
            ].nunique()),
            "risk_queries": int(selected.loc[
                selected.supervision_kind.eq("harmful"), "query_index"
            ].nunique()),
            "robust_queries": int(selected.loc[
                selected.supervision_kind.eq("robust"), "query_index"
            ].nunique()),
            "source_kind_rows": {
                f"{source}|{kind}": int(count)
                for (source, kind), count in selected.groupby(
                    ["source", "supervision_kind"]
                ).size().items()
            },
            "control_semantic_rows": {
                str(semantic): int(count)
                for semantic, count in selected.groupby("control_semantic").size().items()
            },
            "source_control_semantic_rows": {
                f"{source}|{semantic}": int(count)
                for (source, semantic), count in selected.groupby(
                    ["source", "control_semantic"]
                ).size().items()
            },
            "source_action_panels": [
                {
                    "directory": str(path),
                    "status": source_report.get("status"),
                    "action_panel": source_report.get("action_panel"),
                    "evaluated_route_counts": source_report.get("route_counts"),
                    "selector_frontier": source_report.get("selector_frontier"),
                }
                for path, source_report in reports
                if source_report.get("action_panel") is not None
            ],
            "shared_current_E8_clean_geometry": clean_geometry,
            "contracts": {
                "maximum_corrective_per_query": maximum_corrective_per_query,
                "maximum_risk_per_query": maximum_risk_per_query,
                "maximum_robust_per_query": maximum_robust_per_query,
                "multiple_corrective_actions_preserved": bool(
                    selected.loc[selected.supervision_kind.eq("corrective")]
                    .groupby("query_index").size().max() > 1
                ),
                "mechanism_then_source_family_round_robin_selection": bool(
                    maximum_robust_per_query
                ),
                "source_then_family_exposure_balanced_before_recipe_score": True,
                "source_family_round_robin_selection": True,
                "action_multiplicity_is_not_training_dose": True,
                "nonselected_actions_retained_in_routing_ledger": True,
                "spectra_loaded_only_after_bounded_selection": True,
                "compressed_source_arrays_streamed_with_bounded_memory": True,
                "source_selector_frontiers_composed_losslessly": bool(frontier_sources),
                "source_selector_frontier_directories": frontier_sources,
                "exact_action_control_candidate_switch_rows_recorded": bool(
                    len(candidate_switch_sources) == len(reports)
                ),
                "candidate_switch_source_directories": candidate_switch_sources,
                "three_semantics_are_separate": bool(maximum_robust_per_query),
                "control_semantics_explicit_and_source_validated": True,
                "all_route_formal_configurations_verified": True,
                "all_route_formula_fold_seeds_match": True,
                "all_route_clean_ranks_match": clean_geometry["passed"],
                "outer_held_formula_consumed": False,
                "teacher_embedding_target_used": False,
                "P3_consumed": False,
            },
            "model_provenance": {
                "initial_student_checkpoint_sha256": next(iter(initial_hashes)),
            },
            "provenance": {
                "candidate_graph_sha256": graph_sha,
                "graph_report_sha256": graph_report_sha,
                "route_sources": [
                    {
                        "directory": str(path),
                        "report_sha256": sha256_file(path / "report.json"),
                        "actions_sha256": sha256_file(path / "routed_actions.csv.gz"),
                        "spectra_sha256": sha256_file(path / "action_spectra.npz"),
                    }
                    for path, _ in reports
                ],
                "routing_ledger_sha256": sha256_file(staging / "routing_ledger.csv.gz"),
                "training_actions_sha256": sha256_file(staging / "training_actions.csv.gz"),
                "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": "Action routing and bounded selection only; not encoder performance.",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    args = arguments()
    print(json.dumps(build_ledger(
        args.graph_dir, list(args.route_dirs), args.output_dir,
        maximum_corrective_per_query=args.maximum_corrective_per_query,
        maximum_risk_per_query=args.maximum_risk_per_query,
        maximum_robust_per_query=args.maximum_robust_per_query,
    ), indent=2))


if __name__ == "__main__":
    main()
