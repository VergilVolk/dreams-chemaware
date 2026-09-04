"""E3: architecture-independent gradient compatibility of E2 noise actions.

For L(z)=1-cos(z, stopgrad(z_action)) at a unit clean embedding z, the desired
negative gradient in the unit-sphere tangent space is
    g = z_action - (z_action dot z) z.
This is the exact first-order embedding update that an E4 shared adapter must
learn.  E3 measures its magnitude, candidate-margin alignment and pairwise
compatibility before losses are combined in a trainable model.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_noise_v3_a4_exact_peak_scan import load_embeddings, query_candidate_block  # noqa: E402
from build_g8r_real_error_atlas import Cache, load_p3_identities, sha256_file  # noqa: E402
from diagnose_noise_v3_a4b_positive_evidence import cluster_bootstrap  # noqa: E402
from noise_v3_core import attenuate_sequence, candidate_representatives  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_sensitivity"))
    parser.add_argument("--scan-dir", type=Path, default=Path("data/validation/g8r_noise_final_e2_corrective_scan"))
    parser.add_argument("--cache", type=Path, default=Path("data/validation/g8r_error_atlas_listwise_cache.npz"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/validation/g8r_p2_official_embeddings.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--official-checkpoint", type=Path, default=Path("data/e1/official_embedding_slim.pt"))
    parser.add_argument("--architecture-checkpoint", type=Path, default=Path("dreams/models/pretrained/ssl_model_server.pt"))
    parser.add_argument("--p3-dir", type=Path, default=Path("data/validation/g8r_p3_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e3_gradient_compatibility"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--minimum-overlap", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-actions", type=int, default=0, help="smoke only")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_tokens(value: Any) -> np.ndarray:
    return np.asarray([int(token) for token in str(value).split(",") if token != ""], dtype=np.int64)


def tangent_direction(clean: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    clean = np.asarray(clean, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    clean /= max(float(np.linalg.norm(clean)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    tangent = target - float(target @ clean) * clean
    magnitude = float(np.linalg.norm(tangent))
    if magnitude < 1e-10:
        return np.zeros_like(clean, dtype=np.float32), magnitude
    return (tangent / magnitude).astype(np.float32), magnitude


def pairwise_compatibility(
    left: pd.DataFrame, right: pd.DataFrame, bootstrap: int, seed: int,
) -> dict[str, float | int]:
    left_map = left.set_index("query_index")
    right_map = right.set_index("query_index")
    common = left_map.index.intersection(right_map.index)
    if not len(common):
        return {"overlap": 0, "mean_cosine": np.nan, "median_cosine": np.nan,
                "p10_cosine": np.nan, "negative_fraction": np.nan,
                "formula_ci_low": np.nan, "formula_ci_high": np.nan}
    left_vector = np.stack(left_map.loc[common, "direction"].to_numpy())
    right_vector = np.stack(right_map.loc[common, "direction"].to_numpy())
    cosine = np.einsum("ij,ij->i", left_vector, right_vector)
    local = pd.DataFrame({
        "query_formula": left_map.loc[common, "query_formula"].astype(str).to_numpy(),
        "cosine": cosine,
    })
    if local["query_formula"].nunique() >= 2:
        ci = cluster_bootstrap(local, cosine, "query_formula", bootstrap, seed)
    else:
        ci = {"ci_low": np.nan, "ci_high": np.nan}
    return {
        "overlap": int(len(common)),
        "mean_cosine": float(np.mean(cosine)),
        "median_cosine": float(np.median(cosine)),
        "p10_cosine": float(np.quantile(cosine, 0.10)),
        "negative_fraction": float(np.mean(cosine < 0)),
        "formula_ci_low": float(ci["ci_low"]),
        "formula_ci_high": float(ci["ci_high"]),
    }


def aggregate_family(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, query), part in frame.groupby(["family", "query_index"], sort=False):
        vector = np.mean(np.stack(part["direction"].to_numpy()), axis=0)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-10:
            continue
        rows.append({
            "family": family,
            "query_index": int(query),
            "query_ik14": str(part.iloc[0]["query_ik14"]),
            "query_formula": str(part.iloc[0]["query_formula"]),
            "direction": (vector / norm).astype(np.float32),
            "member_cells": int(part["cell_id"].nunique()),
            "mean_tangent_magnitude": float(part["tangent_magnitude"].mean()),
            "mean_margin_alignment": float(part["margin_alignment"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    for name in (
        "sensitivity_dir", "scan_dir", "cache", "embedding_cache", "data",
        "official_checkpoint", "architecture_checkpoint", "p3_dir", "output_dir",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E3: {args.output_dir}")
    sensitivity = json.loads((args.sensitivity_dir / "sensitivity.json").read_text(encoding="utf-8"))
    if not sensitivity.get("formal") or sensitivity.get("status") != "noise_final_e2_sensitivity_complete":
        raise RuntimeError("formal E2 sensitivity audit is not passing")
    candidates = pd.read_csv(args.sensitivity_dir / "e3_candidate_cells.csv")
    if candidates.empty or not candidates["pass_to_e3_after_sensitivity"].astype(bool).all():
        raise RuntimeError("E3 candidate cells are empty or invalid")
    selected_ids = set(candidates["cell_id"].astype(str))
    family_by_cell = dict(zip(candidates["cell_id"].astype(str), candidates["family"].astype(str)))
    actions = pd.read_csv(args.scan_dir / "paired_corrective_interventions.csv.gz")
    actions = actions.loc[
        actions["cell_id"].astype(str).isin(selected_ids)
        & actions["exact_control_match"].astype(bool)
    ].copy() if "exact_control_match" in actions.columns else actions.loc[
        actions["cell_id"].astype(str).isin(selected_ids)
    ].copy()
    # The original M1 file predates the derived exact-control column. Recreate it
    # from the frozen M1b candidate query-cell keys when necessary.
    sensitivity_cells = pd.read_csv(args.sensitivity_dir / "cell_sensitivity.csv")
    if "exact_control_match" not in actions.columns:
        role_selectors = {
            "candidate_gradient", "role_confounder",
            "conditional_missingness_x_confounder",
            "conditional_missingness_x_positive_gradient", "role_shared",
        }
        def is_exact(row: pd.Series) -> bool:
            levels = [x for x in str(row["control_match_levels"]).split(",") if x]
            expected = "role_intensity_mz" if row["selector"] in role_selectors else "intensity_mz"
            return len(levels) == int(row["control_count"]) and all(x == expected for x in levels)
        actions = actions.loc[actions.apply(is_exact, axis=1)].copy()
    del sensitivity_cells
    actions["family"] = actions["cell_id"].astype(str).map(family_by_cell)
    actions = actions.sort_values(["cell_id", "query_index"], kind="stable").reset_index(drop=True)
    if args.max_actions:
        actions = actions.head(args.max_actions).copy()
    if actions.empty:
        raise RuntimeError("E3 has no exact-control action rows")

    cache = Cache(args.cache)
    if cache.n_queries != 23876:
        raise RuntimeError("formal E3 expects the 23,876-query graph")
    if set(map(str, cache.query_ik14)) & load_p3_identities(args.p3_dir):
        raise RuntimeError("P3 identity leakage")
    score_column = cache.feature_names.index("dreams_similarity")
    _, embeddings, embedding_index = load_embeddings(args.embedding_cache)

    query_rows = actions["query_row"].astype(int).unique()
    tensor_by_row: dict[int, torch.Tensor] = {}
    with h5py.File(args.data, "r") as handle:
        for position, row in enumerate(query_rows, start=1):
            tensor_by_row[int(row)] = preprocess_spectrum(
                np.asarray(handle["spectrum"][int(row)]), float(handle["precursor_mz"][int(row)]),
                args.n_highest_peaks,
            )
            if position % 1000 == 0 or position == len(query_rows):
                print(f"[E3 spectra] {position:,}/{len(query_rows):,}", flush=True)

    model, checkpoint_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device(args.device), args.n_highest_peaks,
    )
    if checkpoint_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("E3 requires official fine-tuned DreaMS")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    targets = []
    for row in actions.itertuples(index=False):
        attenuation = float(row.dose) if str(row.operator) == "attenuate" else 1.0
        targets.append(attenuate_sequence(
            tensor_by_row[int(row.query_row)], parse_tokens(row.target_tokens), attenuation,
        ))
    action_vectors = np.empty((len(targets), embeddings.shape[1]), dtype=np.float32)
    device = torch.device(args.device)
    with torch.inference_mode():
        for left in range(0, len(targets), args.batch_size):
            right = min(left + args.batch_size, len(targets))
            action_vectors[left:right] = model(
                torch.stack(targets[left:right]).to(device)
            ).detach().float().cpu().numpy()
            if right % 5000 < args.batch_size or right == len(targets):
                print(f"[E3 encode] {right:,}/{len(targets):,}", flush=True)
    del model, targets
    gc.collect()

    directions = []
    magnitudes = np.empty(len(actions), dtype=np.float32)
    alignments = np.empty(len(actions), dtype=np.float32)
    for index, row in enumerate(actions.itertuples(index=False)):
        clean = embeddings[embedding_index[int(row.query_row)]]
        direction, magnitude = tangent_direction(clean, action_vectors[index])
        directions.append(direction)
        magnitudes[index] = magnitude
        scores, candidate_rows, ptr, _ = query_candidate_block(cache, int(row.query_index), score_column)
        representatives = candidate_representatives(scores, candidate_rows, ptr, 1)
        positive = embeddings[embedding_index[int(representatives.positive_row)]]
        negative = embeddings[embedding_index[int(representatives.negative_rows[0])]]
        margin_direction = positive - negative
        margin_direction /= max(float(np.linalg.norm(margin_direction)), 1e-12)
        alignments[index] = float(direction @ margin_direction)
    actions["direction"] = directions
    actions["tangent_magnitude"] = magnitudes
    actions["margin_alignment"] = alignments

    cell_summary = actions.groupby(["cell_id", "family"], as_index=False).agg(
        actions=("query_index", "size"),
        identities=("query_ik14", "nunique"),
        formulas=("query_formula", "nunique"),
        mean_tangent_magnitude=("tangent_magnitude", "mean"),
        median_tangent_magnitude=("tangent_magnitude", "median"),
        mean_margin_alignment=("margin_alignment", "mean"),
        positive_margin_alignment_fraction=("margin_alignment", lambda x: float(np.mean(np.asarray(x) > 0))),
    )

    cell_frames = {cell: part for cell, part in actions.groupby("cell_id", sort=False)}
    cell_pairs = []
    cell_ids = sorted(cell_frames)
    for left_index, left in enumerate(cell_ids):
        for right_index, right in enumerate(cell_ids):
            result = pairwise_compatibility(
                cell_frames[left], cell_frames[right], args.bootstrap_resamples,
                args.seed + left_index * len(cell_ids) + right_index,
            )
            cell_pairs.append({"left": left, "right": right, **result})
    cell_compatibility = pd.DataFrame(cell_pairs)

    family_actions = aggregate_family(actions)
    family_frames = {family: part for family, part in family_actions.groupby("family", sort=False)}
    family_pairs = []
    families = sorted(family_frames)
    for left_index, left in enumerate(families):
        for right_index, right in enumerate(families):
            result = pairwise_compatibility(
                family_frames[left], family_frames[right], args.bootstrap_resamples,
                args.seed + 10000 + left_index * len(families) + right_index,
            )
            compatible = bool(
                result["overlap"] >= args.minimum_overlap
                and np.isfinite(result["formula_ci_low"])
                and result["formula_ci_low"] > 0
            )
            conflicting = bool(
                result["overlap"] >= args.minimum_overlap
                and np.isfinite(result["formula_ci_high"])
                and result["formula_ci_high"] < 0
            )
            family_pairs.append({
                "left": left, "right": right, **result,
                "compatible": compatible, "conflicting": conflicting,
            })
    family_compatibility = pd.DataFrame(family_pairs)

    temporary = Path(tempfile.mkdtemp(prefix="noise_e3_gradient_", dir=args.output_dir.parent))
    try:
        action_export = actions.drop(columns=["direction"])
        action_export.to_csv(temporary / "action_gradient_summary.csv.gz", index=False, compression="gzip")
        cell_summary.to_csv(temporary / "cell_gradient_summary.csv", index=False)
        cell_compatibility.to_csv(temporary / "cell_gradient_compatibility.csv", index=False)
        family_actions.drop(columns=["direction"]).to_csv(
            temporary / "family_query_gradient_summary.csv.gz", index=False, compression="gzip",
        )
        family_compatibility.to_csv(temporary / "family_gradient_compatibility.csv", index=False)
        report = {
            "status": "noise_final_e3_gradient_compatibility_complete",
            "formal": args.max_actions == 0,
            "candidate_cells": int(len(cell_ids)),
            "mechanism_families": int(len(families)),
            "families": families,
            "action_rows": int(len(actions)),
            "identities": int(actions["query_ik14"].nunique()),
            "formulas": int(actions["query_formula"].nunique()),
            "family_pairs_compatible": int(family_compatibility["compatible"].sum()),
            "family_pairs_conflicting": int(family_compatibility["conflicting"].sum()),
            "gradient_definition": (
                "negative gradient of 1-cos(z_clean, stopgrad(z_action)) in the unit-sphere "
                "embedding tangent space"
            ),
            "contracts": {
                "shared_embedding_target": True,
                "candidate_information_used_only_to_construct_training_actions": True,
                "inference_candidate_independent": True,
                "P2b": "forbidden",
                "P3_identity_overlap": 0,
            },
            "provenance": {
                "sensitivity_sha256": sha256_file(args.sensitivity_dir / "sensitivity.json"),
                "candidate_cells_sha256": sha256_file(args.sensitivity_dir / "e3_candidate_cells.csv"),
                "interventions_sha256": sha256_file(args.scan_dir / "paired_corrective_interventions.csv.gz"),
                "graph_sha256": sha256_file(args.cache),
                "embeddings_sha256": sha256_file(args.embedding_cache),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": (
                "E3 establishes compatibility of desired embedding updates; E4 is required to show "
                "that one shared trainable encoder realizes them and improves clean retrieval."
            ),
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
