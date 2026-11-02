#!/usr/bin/env python
"""Assemble strict source-aware identity panels without confusing comparisons with queries.

The input JSON has this intentionally small schema::

    {
      "schema": "mega_identity_source_manifest_v1",
      "dataset_id": "new_external_identity_v1",
      "sources": [
        {
          "source_id": "source_a",
          "panel": "prepared/source_a_panel.npz",
          "panel_sha256": "optional exact SHA-256",
          "spectrum_row_namespace": {"id": "source_a_rows", "size": 123456}
        }
      ]
    }

Panel paths are resolved relative to the manifest.  Every source panel must
already satisfy the formal three-axis identity contract used by
``evaluate_mega_identity_benchmark.py``.  Source-local query/reference row IDs
are shifted by disjoint cumulative namespace offsets.  The output source
vectors are compact integer indices into ``source_ids`` rather than repeated
strings, which matters when N_P contains millions of spectrum comparisons.

``N_Q`` is labelled queries, ``N_E`` is query-candidate-molecule edges, and
``N_P`` is query-reference-spectrum comparisons.  They are deliberately kept
separate in both the NPZ and report; N_P must never be described as the number
of labelled queries.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluate_mega_identity_benchmark import (
    NpzMemmapStore,
    canonical_axis_digest,
    load_and_validate_panel,
    sha256_file,
)


MANIFEST_SCHEMA = "mega_identity_source_manifest_v1"
OUTPUT_SCHEMA = "mega_identity_benchmark_panel_v1"
SOURCE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
INT64_MAX = int(np.iinfo(np.int64).max)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _text_scalar(array: np.ndarray, field: str) -> str:
    if array.ndim == 0:
        value = array.item()
    elif array.size == 1:
        value = array.reshape(-1)[0].item()
    else:
        raise RuntimeError(f"{field} must be scalar")
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"{field} is not valid UTF-8") from error
    return str(value)


def _integer_scalar(array: np.ndarray, field: str) -> int:
    if array.ndim == 0:
        value = array.item()
    elif array.size == 1:
        value = array.reshape(-1)[0].item()
    else:
        raise RuntimeError(f"{field} must be scalar")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"{field} must be an integer scalar")
    return int(value)


def _required_token(value: object, field: str) -> str:
    if not isinstance(value, str) or SOURCE_TOKEN.fullmatch(value) is None:
        raise RuntimeError(
            f"{field} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
        )
    return value


def _load_manifest(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid manifest JSON: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise RuntimeError("manifest dataset_id must be a non-empty string")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("manifest sources must be a non-empty list")

    normalized: list[dict[str, object]] = []
    source_ids: set[str] = set()
    namespace_ids: set[str] = set()
    panel_paths: set[Path] = set()
    for index, raw in enumerate(sources):
        if not isinstance(raw, dict):
            raise RuntimeError(f"sources[{index}] must be an object")
        source_id = _required_token(raw.get("source_id"), f"sources[{index}].source_id")
        if source_id in source_ids:
            raise RuntimeError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)

        raw_panel = raw.get("panel")
        if not isinstance(raw_panel, str) or not raw_panel:
            raise RuntimeError(f"sources[{index}].panel must be a non-empty path")
        panel_path = Path(raw_panel).expanduser()
        if not panel_path.is_absolute():
            panel_path = path.parent / panel_path
        panel_path = panel_path.resolve()
        if panel_path in panel_paths:
            raise RuntimeError(f"duplicate source panel path: {panel_path}")
        panel_paths.add(panel_path)
        if not panel_path.is_file():
            raise FileNotFoundError(panel_path)

        namespace = raw.get("spectrum_row_namespace")
        if not isinstance(namespace, dict):
            raise RuntimeError(
                f"sources[{index}].spectrum_row_namespace must be an object"
            )
        namespace_id = _required_token(
            namespace.get("id"), f"sources[{index}].spectrum_row_namespace.id"
        )
        if namespace_id in namespace_ids:
            raise RuntimeError(f"duplicate spectrum row namespace id: {namespace_id}")
        namespace_ids.add(namespace_id)
        namespace_size = namespace.get("size")
        if (
            isinstance(namespace_size, bool)
            or not isinstance(namespace_size, int)
            or namespace_size <= 0
        ):
            raise RuntimeError(
                f"sources[{index}].spectrum_row_namespace.size must be a positive integer"
            )

        expected_sha = raw.get("panel_sha256")
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
                raise RuntimeError(f"sources[{index}].panel_sha256 must be lowercase SHA-256")
            observed_sha = sha256_file(panel_path)
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"source {source_id} panel SHA-256 mismatch: {observed_sha}"
                )

        normalized.append({
            "source_id": source_id,
            "panel": panel_path,
            "namespace_id": namespace_id,
            "namespace_size": namespace_size,
        })
    return payload, normalized


def _validate_embedded_contract(store: NpzMemmapStore, panel: object) -> None:
    expected = {
        "N_Q": panel.n_queries,
        "N_I": panel.n_identities,
        "N_F": panel.n_formulas,
        "N_E": panel.n_molecule_edges,
        "N_P": panel.n_pair_edges,
    }
    for field, value in expected.items():
        if store.has(field) and _integer_scalar(store.array(field), field) != value:
            raise RuntimeError(f"input {field} metadata disagrees with its formal axis")
    if store.has("axis_digest"):
        observed = _text_scalar(store.array("axis_digest"), "axis_digest")
        if observed != panel.axis_digest:
            raise RuntimeError("input axis_digest disagrees with its formal axis")


def _source_vector(length: int, source_index: int) -> np.ndarray:
    return np.full(length, source_index, dtype=np.int32)


def assemble(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest, sources = _load_manifest(manifest_path)
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    extraction = staging / "_extracted"
    extraction.mkdir()

    stores: list[NpzMemmapStore] = []
    panels: list[object] = []
    try:
        row_offset = 0
        source_reports: list[dict[str, object]] = []
        for source_index, source in enumerate(sources):
            namespace_size = int(source["namespace_size"])
            if row_offset > INT64_MAX or namespace_size > INT64_MAX - row_offset:
                raise RuntimeError("combined spectrum row namespace exceeds signed int64")
            store = NpzMemmapStore(
                Path(source["panel"]), extraction, f"source_{source_index:04d}"
            )
            stores.append(store)
            panel = load_and_validate_panel(store)
            panels.append(panel)
            _validate_embedded_contract(store, panel)

            local_rows = np.concatenate((
                np.asarray(panel.query_row, dtype=np.int64),
                np.asarray(panel.pair_candidate_row, dtype=np.int64),
            ))
            if len(local_rows) == 0 or int(local_rows.min()) < 0:
                raise RuntimeError(f"source {source['source_id']} has invalid local row IDs")
            if int(local_rows.max()) >= namespace_size:
                raise RuntimeError(
                    f"source {source['source_id']} row ID exceeds declared namespace size"
                )
            source["row_offset"] = row_offset
            source_reports.append({
                "source_index": source_index,
                "source_id": source["source_id"],
                "spectrum_row_namespace": {
                    "id": source["namespace_id"],
                    "size": namespace_size,
                    "global_offset": row_offset,
                },
                "panel": str(Path(source["panel"])),
                "panel_sha256": sha256_file(Path(source["panel"])),
                "axis_digest": panel.axis_digest,
                "counts": {
                    "N_Q_labelled_queries": panel.n_queries,
                    "N_I_query_identity_clusters": panel.n_identities,
                    "N_F_query_formula_clusters": panel.n_formulas,
                    "N_E_query_candidate_molecule_edges": panel.n_molecule_edges,
                    "N_P_query_reference_spectrum_comparisons": panel.n_pair_edges,
                },
            })
            row_offset += namespace_size

        query_ptr_parts = [np.asarray([0], dtype=np.int64)]
        molecule_ptr_parts = [np.asarray([0], dtype=np.int64)]
        edge_offset = 0
        pair_offset = 0
        for panel in panels:
            query_ptr_parts.append(np.asarray(panel.query_ptr[1:], dtype=np.int64) + edge_offset)
            molecule_ptr_parts.append(
                np.asarray(panel.molecule_ptr[1:], dtype=np.int64) + pair_offset
            )
            edge_offset += panel.n_molecule_edges
            pair_offset += panel.n_pair_edges

        query_ik14 = np.concatenate([np.asarray(panel.query_ik14) for panel in panels])
        molecule_ik14 = np.concatenate([np.asarray(panel.molecule_ik14) for panel in panels])
        query_formula = np.concatenate([np.asarray(panel.query_formula) for panel in panels])
        molecule_formula = np.concatenate([np.asarray(panel.molecule_formula) for panel in panels])
        query_row = np.concatenate([
            np.asarray(panel.query_row, dtype=np.int64) + int(source["row_offset"])
            for panel, source in zip(panels, sources)
        ])
        candidate_row = np.concatenate([
            np.asarray(panel.pair_candidate_row, dtype=np.int64) + int(source["row_offset"])
            for panel, source in zip(panels, sources)
        ])
        molecule_label = np.concatenate([
            np.asarray(panel.molecule_label, dtype=bool) for panel in panels
        ])
        near_query = np.concatenate([
            np.asarray(panel.near_query, dtype=bool) for panel in panels
        ])
        query_source = np.concatenate([
            _source_vector(panel.n_queries, index) for index, panel in enumerate(panels)
        ])
        molecule_source = np.concatenate([
            _source_vector(panel.n_molecule_edges, index)
            for index, panel in enumerate(panels)
        ])
        reference_source = np.concatenate([
            _source_vector(panel.n_pair_edges, index) for index, panel in enumerate(panels)
        ])

        n_q = len(query_ik14)
        n_e = len(molecule_ik14)
        n_p = len(candidate_row)
        n_i = len(np.unique(query_ik14))
        n_f = len(np.unique(query_formula))
        payload = {
            "query_ptr": np.concatenate(query_ptr_parts),
            "molecule_ptr": np.concatenate(molecule_ptr_parts),
            "candidate_row": candidate_row,
            "molecule_label": molecule_label,
            "query_ik14": query_ik14,
            "molecule_ik14": molecule_ik14,
            "query_formula": query_formula,
            "molecule_formula": molecule_formula,
            "near_query": near_query,
            "query_row": query_row,
            "query_source": query_source,
            "molecule_source": molecule_source,
            "reference_source": reference_source,
            "source_ids": np.asarray([source["source_id"] for source in sources]),
            "source_row_namespace_ids": np.asarray(
                [source["namespace_id"] for source in sources]
            ),
            "source_row_offsets": np.asarray(
                [source["row_offset"] for source in sources], dtype=np.int64
            ),
            "source_row_namespace_sizes": np.asarray(
                [source["namespace_size"] for source in sources], dtype=np.int64
            ),
            "N_Q": np.asarray(n_q, dtype=np.int64),
            "N_I": np.asarray(n_i, dtype=np.int64),
            "N_F": np.asarray(n_f, dtype=np.int64),
            "N_E": np.asarray(n_e, dtype=np.int64),
            "N_P": np.asarray(n_p, dtype=np.int64),
        }

        # Validate the assembled axis independently before publishing it.
        provisional_path = staging / "panel.provisional.npz"
        np.savez_compressed(provisional_path, **payload)
        combined_store = NpzMemmapStore(provisional_path, extraction, "combined")
        try:
            combined_panel = load_and_validate_panel(combined_store)
            if (
                combined_panel.n_queries != n_q
                or combined_panel.n_molecule_edges != n_e
                or combined_panel.n_pair_edges != n_p
            ):
                raise RuntimeError("assembled panel counts drifted during serialization")
            axis_digest = canonical_axis_digest(combined_panel)
        finally:
            combined_store.close()

        panel_path = staging / "panel.npz"
        np.savez_compressed(
            panel_path,
            **payload,
            axis_digest=np.asarray(axis_digest),
            schema=np.asarray(OUTPUT_SCHEMA),
        )
        provisional_path.unlink()
        report = {
            "schema": OUTPUT_SCHEMA,
            "status": "MEGA_IDENTITY_BENCHMARK_ASSEMBLY_COMPLETE",
            "dataset_id": manifest["dataset_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "panel": "panel.npz",
            "panel_sha256": sha256_file(panel_path),
            "axis_digest": axis_digest,
            "counts": {
                "N_Q_labelled_queries": n_q,
                "N_I_query_identity_clusters": n_i,
                "N_F_query_formula_clusters": n_f,
                "N_E_query_candidate_molecule_edges": n_e,
                "N_P_query_reference_spectrum_comparisons": n_p,
            },
            "unit_definitions": {
                "N_Q": "labelled query spectra",
                "N_I": "unique query identity clusters (IK14)",
                "N_F": "unique query formula clusters",
                "N_E": "query-candidate-molecule edges",
                "N_P": "query-reference-spectrum comparisons; not labelled queries",
            },
            "source_index_contract": (
                "query_source, molecule_source, and reference_source are zero-based "
                "indices into source_ids"
            ),
            "sources": source_reports,
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for store in stores:
            store.close()
        stores.clear()
        panels.clear()
        gc.collect()
        shutil.rmtree(extraction)
        staging.replace(output_dir)
        return report
    except Exception:
        for store in stores:
            store.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    report = assemble(args.manifest, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output_dir),
        "counts": report["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
