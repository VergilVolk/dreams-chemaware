"""Combine outcome-free mature N and fixed P action banks without changing dose.

The output preserves all action rows.  Training is responsible for equalising
queries, action families within a query, and actions within each family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--n-action-bank-dir", type=Path, required=True)
    parser.add_argument("--p-action-bank-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_bank(path: Path, status: str) -> tuple[pd.DataFrame, dict[str, object]]:
    action_path = path / "training_actions.csv.gz"
    report_path = path / "report.json"
    if not action_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"action bank is incomplete: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != status
        or report.get("contracts", {}).get("action_outcomes_computed") is not False
        or report.get("contracts", {}).get("P3_consumed") is not False
    ):
        raise RuntimeError(f"action bank contract failed: {path}")
    return pd.read_csv(action_path, low_memory=False), report


def combine(
    graph_dir: Path, n_dir: Path, p_dir: Path, output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    graph_path = graph_dir / "candidate_graph.npz"
    cache_path = graph_dir / "official_embeddings.npz"
    graph_report_path = graph_dir / "report.json"
    for path in (graph_path, cache_path, graph_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    n, n_report = _load_bank(n_dir, "noise_corrected_full_action_bank_complete")
    p, p_report = _load_bank(p_dir, "noise_corrected_fixed_p_action_bank_complete")
    formal = bool(
        n_report.get("formal_training_authorized") is True
        and p_report.get("formal_training_authorized") is True
    )
    if n_report.get("outer_formula_fold") != p_report.get("outer_formula_fold"):
        raise RuntimeError("N/P outer folds differ")
    for key, actual in (
        ("candidate_graph_sha256", sha256_file(graph_path)),
        ("embedding_cache_sha256", sha256_file(cache_path)),
        ("graph_report_sha256", sha256_file(graph_report_path)),
    ):
        if n_report.get("provenance", {}).get(key) != actual or p_report.get("provenance", {}).get(key) != actual:
            raise RuntimeError(f"N/P corrected-graph provenance differs: {key}")
    for key in ("initial_student_checkpoint_sha256", "official_checkpoint_sha256"):
        if n_report.get("model_provenance", {}).get(key) != p_report.get("model_provenance", {}).get(key):
            raise RuntimeError(f"N/P initialization differs: {key}")
    if n_report.get("model_provenance", {}).get("initialization") != "mature_e4_current_geometry":
        raise RuntimeError("N action bank was not generated in mature E4 geometry")
    if p_report.get("model_provenance", {}).get("initialization") != "mature_e4_current_geometry":
        raise RuntimeError("P action bank was not generated in mature E4 geometry")

    required = {
        "action_id", "query_index", "query_row", "query_ik14", "query_formula",
        "formula_fold", "selector", "attenuation", "step", "target_path",
    }
    if required - set(n.columns) or required - set(p.columns):
        raise RuntimeError("N/P action schema is incomplete")
    n = n.copy()
    n["action_kind"] = "attenuation_path"
    n["action_tensor_index"] = -1
    if not {"action_kind", "action_tensor_index"} <= set(p.columns):
        raise RuntimeError("P precomputed action schema is incomplete")
    if set(p.action_kind.astype(str)) != {"precomputed_spectrum"}:
        raise RuntimeError("P bank has an unexpected action kind")
    combined = pd.concat([n, p], ignore_index=True, sort=False).sort_values(
        ["query_index", "selector", "step", "action_id"], kind="stable",
    ).reset_index(drop=True)
    if combined.action_id.duplicated().any():
        raise RuntimeError("combined N/P action IDs are not unique")
    if np.any(combined.formula_fold.to_numpy(np.int8) == int(n_report["outer_formula_fold"])):
        raise RuntimeError("outer-held formula leaked into combined action bank")

    p_spectrum_path = p_dir / "action_spectra.npz"
    if (
        not p_spectrum_path.is_file()
        or p_report.get("provenance", {}).get("action_spectra_sha256") != sha256_file(p_spectrum_path)
    ):
        raise RuntimeError("P action spectrum provenance failed")
    with np.load(p_spectrum_path, allow_pickle=False) as body:
        if set(body.files) != {"action_ids", "action_spectra"}:
            raise RuntimeError("P action spectrum schema failed")
        ids = np.asarray(body["action_ids"], dtype=str)
        spectra = np.asarray(body["action_spectra"], dtype=np.float32)
    p_indices = combined.loc[
        combined.action_kind.eq("precomputed_spectrum"), "action_tensor_index"
    ].to_numpy(np.int64)
    p_ids = combined.loc[
        combined.action_kind.eq("precomputed_spectrum"), "action_id"
    ].astype(str).to_numpy()
    if np.any((p_indices < 0) | (p_indices >= len(ids))) or not np.array_equal(p_ids, ids[p_indices]):
        raise RuntimeError("combined P tensor indices do not align")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        combined.to_csv(staging / "training_actions.csv.gz", index=False, compression="gzip")
        np.savez_compressed(staging / "action_spectra.npz", action_ids=ids, action_spectra=spectra)
        report: dict[str, object] = {
            "status": "noise_corrected_combined_np_action_bank_complete",
            "formal": formal,
            "formal_training_authorized": formal,
            "outer_formula_fold": int(n_report["outer_formula_fold"]),
            "source_queries": int(combined.query_index.nunique()),
            "action_rows": int(len(combined)),
            "family_rows": {
                str(key): int(value) for key, value in combined.groupby("selector").size().items()
            },
            "model_provenance": n_report["model_provenance"],
            "contracts": {
                "action_outcomes_computed": False,
                "outer_held_formulas_published": False,
                "action_multiplicity_is_not_training_dose": True,
                "query_family_action_equal_required": True,
                "teacher_embedding_or_margin_used": False,
                "P2b": "forbidden",
                "P3_consumed": False,
            },
            "provenance": {
                "candidate_graph_sha256": sha256_file(graph_path),
                "embedding_cache_sha256": sha256_file(cache_path),
                "graph_report_sha256": sha256_file(graph_report_path),
                "n_action_bank_report_sha256": sha256_file(n_dir / "report.json"),
                "n_action_bank_sha256": sha256_file(n_dir / "training_actions.csv.gz"),
                "p_action_bank_report_sha256": sha256_file(p_dir / "report.json"),
                "p_action_bank_sha256": sha256_file(p_dir / "training_actions.csv.gz"),
                "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": "Outcome-free bank combination only; no encoder improvement is claimed.",
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    args = arguments()
    print(json.dumps(combine(
        args.graph_dir, args.n_action_bank_dir, args.p_action_bank_dir, args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
