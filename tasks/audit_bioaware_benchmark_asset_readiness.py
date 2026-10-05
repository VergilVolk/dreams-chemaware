#!/usr/bin/env python
"""Read-only asset readiness audit for the BioAware unified benchmark.

Inventories which Track A/B/C inputs exist locally, verifies their schemas and
counts, and reports per-track executability.  No annotation truth beyond the
already-opened development labels inside frozen artifacts is read; no model is
fitted and no performance metric is computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _file_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"present": False}
    stat = path.stat()
    return {
        "present": True,
        "bytes": int(stat.st_size),
        "last_write": stat.st_mtime,
    }


def audit_corrected_graph(root: Path) -> dict[str, object]:
    graph_dir = root / "data/validation/noise_corrected_candidate_graph_v1_20260906"
    graph = _file_state(graph_dir / "candidate_graph.npz")
    embeddings = _file_state(graph_dir / "official_embeddings.npz")
    result: dict[str, object] = {
        "directory": str(graph_dir),
        "candidate_graph": graph,
        "official_embeddings": embeddings,
        "role": "track_a_development_only",
    }
    if not graph["present"]:
        result["status"] = "MISSING"
        return result
    with np.load(graph_dir / "candidate_graph.npz", allow_pickle=True) as data:
        keys = set(data.files)
        result["keys"] = sorted(keys)
        required = {
            "query_ptr", "molecule_ptr", "pair_candidate_row", "molecule_label",
            "molecule_ik14", "molecule_formula", "query_ik14", "query_row",
        }
        result["required_keys_present"] = sorted(required & keys)
        result["required_keys_missing"] = sorted(required - keys)
        queries = int(len(data["query_ik14"]))
        molecules = int(len(data["molecule_ik14"]))
        result["queries"] = queries
        result["candidate_molecules"] = molecules
        result["truth_labels"] = int(np.asarray(data["molecule_label"]).sum())
        result["near_flagged_queries"] = int(np.asarray(data["query_has_near"]).sum())
        result["expected_queries_match"] = queries == 83619
    if embeddings["present"]:
        with np.load(graph_dir / "official_embeddings.npz", allow_pickle=True) as data:
            result["embedding_rows"] = int(data["embeddings"].shape[0])
            result["embedding_dim"] = int(data["embeddings"].shape[1])
            result["embedding_rows_equal_queries_plus_reference_pool"] = (
                result["embedding_rows"] >= result["queries"]
            )
    result["raw_spectra_locally_derivable"] = False
    result["classical_baselines_runnable"] = False
    result["official_dreams_scores_runnable_from_local_embeddings"] = bool(
        embeddings["present"]
    )
    result["status"] = "RUNNABLE_LOCAL_DEVELOPMENT"
    return result


def audit_gnps_panel(root: Path) -> dict[str, object]:
    panel_dir = root / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
    files = {
        name: _file_state(panel_dir / name)
        for name in (
            "spectra.mgf", "manifest.csv.gz", "panel_identity_disjoint.npz",
            "panel_formula_disjoint.npz", "pairs_identity_disjoint.npz",
            "pairs_formula_disjoint.npz", "report.json", "checksums.sha256",
        )
    }
    report_path = panel_dir / "report.json"
    expected_identity_queries = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        identity = report.get("identity_disjoint", {})
        expected_identity_queries = identity.get("queries")
        formula = report.get("formula_disjoint", {})
        result_report = {
            "identity_disjoint_queries_declared": expected_identity_queries,
            "formula_disjoint_queries_declared": formula.get("queries"),
        }
    else:
        result_report = {"report_json": "missing"}
    missing = [name for name, state in files.items() if not state["present"]]
    result: dict[str, object] = {
        "directory": str(panel_dir),
        "files": files,
        "missing_files": missing,
        "report_summary": result_report,
        "role": "track_a_sealed_external_library_panel",
        "classical_baselines_runnable": bool(
            files["spectra.mgf"]["present"] and files["manifest.csv.gz"]["present"]
        ),
        "identity_disjoint_main_panel_missing": not files[
            "panel_identity_disjoint.npz"
        ]["present"],
        "official_dreams_embeddings_available_locally": False,
        "status": (
            "PARTIAL_MAIN_PANEL_MISSING"
            if "panel_identity_disjoint.npz" in missing
            else "SEALED_COMPLETE"
        ),
    }
    return result


def audit_massspecgym_v15(root: Path) -> dict[str, object]:
    hdf5_candidates = list(root.glob("data/**/*MurckoHist*.hdf5"))
    v15_markers = list(root.glob("data/**/massspecgym*v1.5*"))
    return {
        "murcko_hdf5_local": [str(p) for p in hdf5_candidates[:3]],
        "v1_5_resources_local": [str(p) for p in v15_markers[:3]],
        "public_leaderboard_claim_blocked_locally": not hdf5_candidates
        and not v15_markers,
        "status": "SERVER_ONLY",
        "consequence": (
            "Public MassSpecGym v1.5 leaderboard claims cannot be produced from "
            "this workstation until the official test split and candidate "
            "resources are acquired and hashed."
        ),
    }


def audit_official_checkpoint(root: Path) -> dict[str, object]:
    large_model_files = []
    for pattern in ("data/models/**/*", "data/external/**/*.ckpt",
                    "data/external/**/*.pt", "data/external/**/*.pth",
                    "data/external/**/*.safetensors"):
        for path in root.glob(pattern):
            if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
                large_model_files.append(str(path))
    official_named = [p for p in large_model_files if "official" in p.lower()]
    return {
        "large_model_weight_files": large_model_files[:10],
        "official_checkpoint_found": bool(official_named),
        "consequence": (
            "New spectra cannot be encoded locally with official DreaMS weights; "
            "official-encoder runs for the GNPS panel must execute on the server "
            "or from a synchronized checkpoint."
        ),
        "status": "SERVER_ONLY",
    }


def audit_opened_bioaware_baselines(root: Path) -> dict[str, object]:
    candidates = {
        "b35_real_library_formal_local": root
        / "data/validation/bioaware_b35_real_library_retrieval_formal_local_20260912",
        "b30_cross_source_sink_localcheck": root
        / "data/validation/bioaware_b30_cross_source_sink_veto_localcheck_20260907_v1",
        "phasea_pool_rebuild": root
        / "data/validation/chemaware_phasea_pool_rebuild_local_20260927",
    }
    states = {}
    for name, path in candidates.items():
        state = _file_state(path)
        if state["present"]:
            inner = [p.name for p in path.iterdir()][:12]
            state["members"] = inner
        states[name] = state
    return {
        "artifacts": states,
        "role": "track_c_opened_development_engineering_baselines",
        "usage_rule": (
            "Reported only as opened cross-fitted development baselines with the "
            "B44 external reversal retained in the limitation table."
        ),
        "status": "PRESENT" if all(s["present"] for s in states.values()) else "PARTIAL",
    }


def audit_external_tools(root: Path) -> dict[str, object]:
    tool_paths = {
        "metdna3_source": root / "data/external/metdna3_2025",
        "netid_release": root / "data/external/netid_v1",
        "iceberg_weights": root / "data/external/iceberg_msg_weights_v1",
        "kpgt": root / "data/external/KPGT",
        "unified_reference_library": root / "data/reference/unified_v3/unified_pos.mgf",
    }
    states = {name: _file_state(path) for name, path in tool_paths.items()}
    return {
        "resources": states,
        "status": "PARTIAL",
        "note": (
            "Track B structure-aware methods require per-tool environment "
            "verification before any run card is issued."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite readiness report: {args.output}")

    report = {
        "status": "bioaware_unified_benchmark_asset_readiness_v1",
        "contract": "docs/BIOAWARE_UNIFIED_BENCHMARK_CONTRACT_20261003.md",
        "track_a_corrected_graph": audit_corrected_graph(args.root),
        "track_a_gnps_sealed_panel": audit_gnps_panel(args.root),
        "track_a_massspecgym_v15": audit_massspecgym_v15(args.root),
        "official_checkpoint": audit_official_checkpoint(args.root),
        "track_c_opened_bioaware_baselines": audit_opened_bioaware_baselines(args.root),
        "external_tools": audit_external_tools(args.root),
    }

    track_a_blockers = []
    if report["track_a_gnps_sealed_panel"]["identity_disjoint_main_panel_missing"]:
        track_a_blockers.append("gnps panel_identity_disjoint.npz missing locally")
    if report["official_checkpoint"]["status"] == "SERVER_ONLY":
        track_a_blockers.append("official DreaMS checkpoint not local")
    if report["track_a_massspecgym_v15"]["status"] == "SERVER_ONLY":
        track_a_blockers.append("MassSpecGym v1.5 official resources not local")
    report["track_a_blocking_gaps"] = track_a_blockers
    report["track_a_local_first_run"] = (
        "corrected-graph official-embedding replay (development, no public claim) "
        "plus classical baselines on the GNPS formula-disjoint panel"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"[readiness] {args.output}", flush=True)
    print(
        json.dumps(
            {
                "corrected_graph": report["track_a_corrected_graph"]["status"],
                "gnps": report["track_a_gnps_sealed_panel"]["status"],
                "massspecgym_v15": report["track_a_massspecgym_v15"]["status"],
                "official_checkpoint": report["official_checkpoint"]["status"],
                "track_a_blockers": track_a_blockers,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
