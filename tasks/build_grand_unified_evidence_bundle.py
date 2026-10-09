"""Assemble aligned candidate evidence for the all-module unified model.

This adapter accepts the existing GNPS pair-score bundle plus any number of
server-produced aligned score arrays.  Every requested module is retained in
the output.  An unavailable module receives neutral scores and availability=0,
so downstream code can distinguish missing evidence from evidence against a
candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel  # noqa: E402


DEFAULT_MODULES = (
    "official_dreams",
    "noise_v1",
    "chemaware_stage1_encoder",
    "weighted_spectral_entropy",
    "p2b_noise_v1_frozen",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_array_spec(spec: str) -> tuple[str, Path, str | None]:
    """Parse NAME=PATH[::KEY], using :: to remain Windows-drive safe."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError("array spec must be NAME=PATH[::KEY]")
    name, value = spec.split("=", 1)
    if "::" in value:
        path, key = value.rsplit("::", 1)
    else:
        path, key = value, None
    return name, Path(path), key


def load_named_arrays(path: Path, panel: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "method_names" not in data:
            raise ValueError(f"{path} has no method_names")
        names = [str(x) for x in data["method_names"]]
        key = f"scores_{panel}"
        if key not in data:
            raise ValueError(f"{path} has no {key}")
        values = np.asarray(data[key], dtype=np.float32)
    if values.shape[0] != len(names):
        raise ValueError(f"method/score mismatch in {path}")
    return {name: values[i] for i, name in enumerate(names)}


def reduce_to_molecules(values: np.ndarray, graph) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n_molecules = len(graph.molecule_label)
    n_pairs = len(graph.pair_candidate_row)
    if len(values) == n_molecules:
        return values
    if len(values) == n_pairs:
        return np.maximum.reduceat(values, graph.molecule_ptr[:-1])
    raise ValueError(
        f"score length {len(values)} is neither molecules={n_molecules} nor pairs={n_pairs}"
    )


def load_extra(path: Path, key: str | None, panel: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        candidates = [key] if key else [f"scores_{panel}", "scores", panel]
        chosen = next((candidate for candidate in candidates if candidate in data), None)
        if chosen is None:
            raise ValueError(f"cannot infer array key in {path}; tried {candidates}")
        result = {"scores": np.asarray(data[chosen], dtype=np.float32)}
        for name in ("query_ids", "candidate_ids", "availability"):
            if name in data:
                result[name] = np.asarray(data[name])
        return result


def validate_candidate_keys(payload: dict[str, np.ndarray], query_ids: np.ndarray, candidate_ids: np.ndarray, module: str) -> None:
    if "query_ids" not in payload or "candidate_ids" not in payload:
        raise ValueError(f"{module}: extra arrays must include query_ids and candidate_ids")
    if not np.array_equal(payload["query_ids"].astype("U"), query_ids.astype("U")):
        raise ValueError(f"{module}: query_ids do not match frozen panel order")
    if not np.array_equal(payload["candidate_ids"].astype("U"), candidate_ids.astype("U")):
        raise ValueError(f"{module}: candidate_ids do not match frozen panel order")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, choices=("identity_disjoint", "formula_disjoint"))
    parser.add_argument("--panel-npz", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=PATH[::KEY]")
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    graph = graph_from_panel(args.panel_npz)
    raw = load_named_arrays(args.base_scores, args.panel)
    extras = {}
    for spec in args.extra:
        name, path, key = parse_array_spec(spec)
        if name in raw or name in extras:
            raise ValueError(f"duplicate module source: {name}")
        extras[name] = (path, load_extra(path, key, args.panel))

    modules = tuple(args.module) if args.module else DEFAULT_MODULES
    if len(set(modules)) != len(modules):
        raise ValueError("duplicate requested module")
    q_count = graph.n_queries
    n_molecules = len(graph.molecule_label)
    with np.load(args.panel_npz, allow_pickle=False) as panel:
        group_ids = np.asarray(panel["query_formula"]).astype("U")
        query_ids = np.asarray(panel["query_ik14"]).astype("U")
        candidate_ids = np.asarray(panel["molecule_ik14"]).astype("U")
        near_query = np.asarray(panel["near_query"], dtype=bool)
    score_rows = []
    availability_rows = []
    provenance = {}
    for module in modules:
        if module not in raw and module not in extras:
            score_rows.append(np.zeros(n_molecules, dtype=np.float32))
            availability_rows.append(np.zeros(n_molecules, dtype=np.float32))
            provenance[module] = {"status": "MISSING_NEUTRAL"}
            continue
        if module in extras:
            source_path, payload = extras[module]
            validate_candidate_keys(payload, query_ids, candidate_ids, module)
            values = reduce_to_molecules(payload["scores"], graph)
            observed = np.asarray(payload.get("availability", np.ones(n_molecules)), dtype=np.float32).reshape(-1)
            if len(observed) != n_molecules:
                raise ValueError(f"{module}: availability must be candidate-level length {n_molecules}")
            provenance[module] = {"status": "PRESENT_KEYED", "source": str(source_path), "sha256": sha256(source_path)}
        else:
            values = reduce_to_molecules(raw[module], graph)
            observed = np.ones(n_molecules, dtype=np.float32)
            provenance[module] = {"status": "PRESENT_BASE_FROZEN"}
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite scores for {module}")
        if np.any((observed < 0) | (observed > 1)):
            raise ValueError(f"{module}: availability values must be in [0, 1]")
        score_rows.append(values)
        availability_rows.append(observed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        schema=np.asarray("grand_unified_evidence_bundle_v2"),
        dataset_id=np.asarray("gnps_gold_silver_10ppm"),
        evaluation_role=np.asarray("development"),
        truth_status=np.asarray("consumed"),
        allow_final_claim=np.asarray(0, dtype=np.int8),
        panel=np.asarray(args.panel),
        module_names=np.asarray(modules),
        scores=np.stack(score_rows).astype(np.float32),
        availability=np.stack(availability_rows).astype(np.float32),
        query_ptr=np.asarray(graph.query_ptr, dtype=np.int64),
        labels=np.asarray(graph.molecule_label, dtype=np.int8),
        group_ids=group_ids,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        near_query=near_query,
    )
    manifest = {
        "schema": "grand_unified_evidence_bundle_manifest_v2",
        "dataset_id": "gnps_gold_silver_10ppm",
        "evaluation_role": "development",
        "truth_status": "consumed",
        "allow_final_claim": False,
        "boundary": (
            "GNPS Gold/Silver has already been inspected during model development. "
            "This builder cannot create a final external-test bundle."
        ),
        "panel": args.panel,
        "queries": int(q_count),
        "candidates": int(n_molecules),
        "modules": list(modules),
        "present_modules": [m for m in modules if m in raw or m in extras],
        "missing_modules": [m for m in modules if m not in raw and m not in extras],
        "panel_sha256": sha256(args.panel_npz),
        "base_scores_sha256": sha256(args.base_scores),
        "candidate_axis": "query_ids + candidate_ids exact equality required for every extra module",
        "provenance": provenance,
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
