"""Frozen m/z-constrained annotation audit for priority LCNEC dark features.

This script compares the official DreaMS similarity with the already frozen
P2b rank fusion.  It never fits or tunes a model on the LCNEC phenotypes.
Candidate generation is precursor-m/z constrained and candidate spectra are
aggregated by InChIKey connectivity block (IK14).

The output is an annotation *hypothesis* table.  No row is MSI level 1 without
an authentic standard and retention-time confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_large_observability_residual import symmetric_features  # noqa: E402
from encode_mona_neg_library import parse_mgf  # noqa: E402
from g8r_p2_rank_fusion_core import (  # noqa: E402
    fuse_one_query,
    fusion_configuration_from_mapping,
    normalize_pair_features,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def strict_order(scores: np.ndarray, labels: list[str]) -> np.ndarray:
    """Descending score, deterministic label tie-break (ties remain visible)."""
    return np.lexsort((np.asarray(labels, dtype=str), -np.asarray(scores, dtype=float)))


def score_margin(scores: np.ndarray) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))[::-1]
    return float(scores[0] - scores[1]) if len(scores) > 1 else float("inf")


def competition_qvalues(target: np.ndarray, decoy: np.ndarray) -> np.ndarray:
    """Paired target-decoy competition q-values for target-winning queries.

    Each query contributes exactly one winner.  Scores are sorted from high to
    low, prefix FDR is (decoy + 1)/(target + 1), and target q-values use the
    reverse cumulative minimum.  With only 30 queries these values are
    descriptive confidence evidence, not a formal annotation FDR guarantee.
    """
    target = np.asarray(target, dtype=float)
    decoy = np.asarray(decoy, dtype=float)
    if target.shape != decoy.shape or target.ndim != 1:
        raise ValueError("target and decoy scores must be aligned vectors")
    winner_is_target = target > decoy  # ties conservatively count as decoys
    winner_score = np.maximum(target, decoy)
    order = np.argsort(-winner_score, kind="stable")
    ordered_target = winner_is_target[order]
    cum_target = np.cumsum(ordered_target)
    cum_decoy = np.cumsum(~ordered_target)
    fdr = (cum_decoy + 1.0) / (cum_target + 1.0)
    q_ordered = np.minimum.accumulate(fdr[::-1])[::-1]
    q = np.ones(len(target), dtype=float)
    q[order] = q_ordered
    q[~winner_is_target] = 1.0
    return np.clip(q, 0.0, 1.0)


def ik14_key(value: object, row: int) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text[:14] if len(text) >= 14 else f"NO_IK_{row:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-dir", type=Path,
        default=ROOT / "data/validation/lcnec_hsst3n_priority_ms2",
    )
    parser.add_argument(
        "--library-dir", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb",
    )
    parser.add_argument(
        "--library-mgf", type=Path,
        default=ROOT / "data/models/mona_neg_full.mgf",
    )
    parser.add_argument(
        "--decoy-embeddings", type=Path,
        default=ROOT / "data/models/mona_neg_decoy_emb.npy",
    )
    parser.add_argument(
        "--p2b-artifact", type=Path,
        default=ROOT / "data/validation/g8r_p2b_rank_fusion.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/lcnec_hsst3n_priority_annotation",
    )
    parser.add_argument("--ppm-windows", default="10,20,50")
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ppm_windows = tuple(float(value) for value in args.ppm_windows.split(","))
    if ppm_windows != tuple(sorted(set(ppm_windows))) or 20.0 not in ppm_windows:
        raise RuntimeError("ppm windows must be unique, sorted, and include 20")

    if args.output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    query_mgf = args.query_dir / "priority_dark_modules.mgf"
    query_effects_path = args.query_dir / "priority_dark_modules.csv"
    query_embedding_dir = args.query_dir / "dreams_embeddings"
    query_embedding_path = query_embedding_dir / "embeddings.npy"
    query_manifest_path = query_embedding_dir / "manifest.csv"
    library_embedding_path = args.library_dir / "embeddings.npy"
    library_manifest_path = args.library_dir / "manifest.csv"
    required = [
        query_mgf, query_effects_path, query_embedding_path, query_manifest_path,
        library_embedding_path, library_manifest_path, args.library_mgf,
        args.decoy_embeddings, args.p2b_artifact,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")

    query_records = parse_mgf(query_mgf)
    library_records = parse_mgf(args.library_mgf)
    query_manifest = pd.read_csv(query_manifest_path)
    query_effects = pd.read_csv(query_effects_path)
    library_manifest = pd.read_csv(library_manifest_path)
    query_embeddings = unit_rows(np.load(query_embedding_path))
    library_embeddings = unit_rows(np.load(library_embedding_path, mmap_mode="r"))
    decoy_embeddings = unit_rows(np.load(args.decoy_embeddings, mmap_mode="r"))

    n_query = len(query_records)
    n_library = len(library_records)
    if not (n_query >= 1 and n_query == len(query_manifest) == len(query_effects) == len(query_embeddings)):
        raise RuntimeError("query MGF, manifests and embeddings are not aligned")
    if not (n_library == len(library_manifest) == len(library_embeddings) == len(decoy_embeddings)):
        raise RuntimeError("MoNA MGF, manifest, target and decoy embeddings are not aligned")

    query_mz = np.asarray([record["precursor_mz"] for record in query_records], dtype=float)
    library_mz = np.asarray([record["precursor_mz"] for record in library_records], dtype=float)
    if not np.allclose(query_mz, query_manifest["precursor_mz"].to_numpy(float), atol=1e-6):
        raise RuntimeError("query MGF and embedding manifest order differ")
    if not np.allclose(query_mz, query_effects["target_mz"].to_numpy(float), atol=1e-5):
        raise RuntimeError("query MGF and effect manifest order differ")
    if not np.allclose(library_mz, library_manifest["precursor_mz"].to_numpy(float), atol=1e-6):
        raise RuntimeError("library MGF and embedding manifest order differ")

    with args.p2b_artifact.open(encoding="utf-8") as handle:
        p2b_body = json.load(handle)
    configuration = fusion_configuration_from_mapping(p2b_body["configuration"])
    if configuration.normalization != "absolute" or len(configuration.weights) != 4:
        raise RuntimeError("unexpected frozen P2b configuration")

    summary_rows: list[dict] = []
    candidate_rows: list[dict] = []
    target_top_by_window: dict[float, np.ndarray] = {}
    decoy_top_by_window: dict[float, np.ndarray] = {}

    for ppm in ppm_windows:
        target_top = np.full(n_query, -1.0, dtype=float)
        decoy_top = np.full(n_query, -1.0, dtype=float)
        for query_index in range(n_query):
            mz = query_mz[query_index]
            candidate_index = np.flatnonzero(np.abs(library_mz - mz) / mz * 1e6 <= ppm)
            base = query_effects.iloc[query_index].to_dict()
            if len(candidate_index) == 0:
                summary_rows.append({
                    **base, "query_index": query_index, "ppm_window": ppm,
                    "candidate_spectra": 0, "candidate_molecules": 0,
                    "dreams_top_ik14": "", "dreams_top_name": "",
                    "dreams_top_score": np.nan, "dreams_margin": np.nan,
                    "p2b_top_ik14": "", "p2b_top_name": "",
                    "p2b_top_score": np.nan, "p2b_margin": np.nan,
                    "p2b_used": False, "p2b_support": 0,
                    "dreams_p2b_agree": False,
                    "selected_mass_error_ppm": np.nan,
                    "selected_reference_spectra": 0,
                    "target_top_score": np.nan, "decoy_top_score": np.nan,
                })
                continue

            cosine = np.asarray(query_embeddings[query_index] @ library_embeddings[candidate_index].T)
            decoy_cosine = np.asarray(query_embeddings[query_index] @ decoy_embeddings[candidate_index].T)
            target_top[query_index] = float(cosine.max())
            decoy_top[query_index] = float(decoy_cosine.max())

            pair_rows: list[dict] = []
            for local, ref_index in enumerate(candidate_index):
                features = symmetric_features(
                    query_records[query_index]["peaks"], mz,
                    library_records[int(ref_index)]["peaks"], library_mz[int(ref_index)],
                    args.fragment_tolerance,
                )
                pair_rows.append({
                    "ref_index": int(ref_index),
                    "ik14": ik14_key(library_manifest.iloc[int(ref_index)]["inchikey"], int(ref_index)),
                    "dreams_similarity": float(cosine[local]),
                    "sqrt_cosine": float(features["sqrt_cosine"]),
                    "entropy_similarity": float(features["entropy_similarity"]),
                    "neutral_loss_sqrt_cosine": float(features["neutral_loss_sqrt_cosine"]),
                })
            pair_frame = pd.DataFrame(pair_rows).sort_values(["ik14", "ref_index"], kind="stable").reset_index(drop=True)
            molecule_labels = list(dict.fromkeys(pair_frame["ik14"].tolist()))
            starts = pair_frame.groupby("ik14", sort=False).size().to_numpy(int)
            molecule_ptr = np.r_[0, np.cumsum(starts)]
            feature_matrix = pair_frame[[
                "dreams_similarity", "sqrt_cosine", "entropy_similarity",
                "neutral_loss_sqrt_cosine",
            ]].to_numpy(float)
            normalized = normalize_pair_features(
                feature_matrix, np.array([0, len(feature_matrix)]), configuration.normalization,
            )
            final_molecule, p2b_used, support = fuse_one_query(
                normalized, feature_matrix[:, 0], molecule_ptr,
                np.asarray(configuration.weights), (1, 2, 3),
                configuration.min_support, configuration.min_advantage,
            )
            dreams_molecule = np.maximum.reduceat(feature_matrix[:, 0], molecule_ptr[:-1])
            dreams_order = strict_order(dreams_molecule, molecule_labels)
            p2b_order = strict_order(final_molecule, molecule_labels)
            dreams_top = int(dreams_order[0])
            p2b_top = int(p2b_order[0])

            def representative(molecule_index: int, values: np.ndarray) -> pd.Series:
                left, right = int(molecule_ptr[molecule_index]), int(molecule_ptr[molecule_index + 1])
                local_index = left + int(np.argmax(values[left:right]))
                return pair_frame.iloc[local_index]

            dreams_rep = representative(dreams_top, feature_matrix[:, 0])
            fused_pair_score = normalized @ np.asarray(configuration.weights)
            p2b_rep = representative(p2b_top, fused_pair_score if p2b_used else feature_matrix[:, 0])
            selected_ref = int(p2b_rep["ref_index"])
            selected_library = library_manifest.iloc[selected_ref]
            selected_group = library_manifest.iloc[
                pair_frame.loc[pair_frame["ik14"].eq(molecule_labels[p2b_top]), "ref_index"].astype(int)
            ]
            selected_full_ik = selected_group["inchikey"].dropna().astype(str)
            selected_names = sorted(set(selected_group["name"].dropna().astype(str)))

            summary_rows.append({
                **base, "query_index": query_index, "ppm_window": ppm,
                "candidate_spectra": len(candidate_index),
                "candidate_molecules": len(molecule_labels),
                "dreams_top_ik14": molecule_labels[dreams_top],
                "dreams_top_name": str(library_manifest.iloc[int(dreams_rep["ref_index"])]["name"]),
                "dreams_top_score": float(dreams_molecule[dreams_top]),
                "dreams_margin": score_margin(dreams_molecule),
                "p2b_top_ik14": molecule_labels[p2b_top],
                "p2b_top_name": str(selected_library["name"]),
                "p2b_top_smiles": str(selected_library["smiles"]),
                "p2b_top_inchikey": str(selected_library["inchikey"]),
                "p2b_top_score": float(final_molecule[p2b_top]),
                "p2b_margin": score_margin(final_molecule),
                "p2b_used": bool(p2b_used), "p2b_support": int(support),
                "dreams_p2b_agree": bool(dreams_top == p2b_top),
                "selected_mass_error_ppm": float(abs(library_mz[selected_ref] - mz) / mz * 1e6),
                "selected_reference_spectra": int(starts[p2b_top]),
                "selected_full_inchikey_count": int(selected_full_ik.nunique()),
                "selected_name_count": len(selected_names),
                "selected_names": " | ".join(selected_names),
                "target_top_score": target_top[query_index],
                "decoy_top_score": decoy_top[query_index],
            })

            for rank_position, molecule_index in enumerate(p2b_order[:5], start=1):
                rep = representative(int(molecule_index), fused_pair_score if p2b_used else feature_matrix[:, 0])
                ref_index = int(rep["ref_index"])
                lib = library_manifest.iloc[ref_index]
                candidate_rows.append({
                    "query_index": query_index, "family_id": int(base["family_id"]),
                    "target_mz": mz, "target_rt_sec": float(base["target_rt_sec"]),
                    "ppm_window": ppm, "rank": rank_position,
                    "ik14": molecule_labels[int(molecule_index)],
                    "inchikey": str(lib["inchikey"]), "name": str(lib["name"]),
                    "smiles": str(lib["smiles"]), "reference_index": ref_index,
                    "mass_error_ppm": float(abs(library_mz[ref_index] - mz) / mz * 1e6),
                    "reference_spectra": int(starts[int(molecule_index)]),
                    "dreams_molecule_score": float(dreams_molecule[int(molecule_index)]),
                    "final_molecule_score": float(final_molecule[int(molecule_index)]),
                    "p2b_used": bool(p2b_used), "p2b_support": int(support),
                })
        target_top_by_window[ppm] = target_top
        decoy_top_by_window[ppm] = decoy_top

    summary = pd.DataFrame(summary_rows)
    candidates = pd.DataFrame(candidate_rows)
    for ppm in ppm_windows:
        mask = summary["ppm_window"].eq(ppm)
        qvalues = competition_qvalues(target_top_by_window[ppm], decoy_top_by_window[ppm])
        summary.loc[mask, "paired_tda_q_descriptive"] = qvalues
        summary.loc[mask, "target_beats_decoy"] = target_top_by_window[ppm] > decoy_top_by_window[ppm]

    # Cross-window stability is evaluated only on windows that contain a candidate.
    identity_by_query: dict[int, dict[float, str]] = {}
    for row in summary.itertuples(index=False):
        if row.candidate_molecules:
            identity_by_query.setdefault(int(row.query_index), {})[float(row.ppm_window)] = str(row.p2b_top_ik14)
    stable = {
        query: len(set(values.values())) == 1 and len(values) >= 2
        for query, values in identity_by_query.items()
    }
    summary["p2b_identity_stable_across_available_windows"] = summary["query_index"].map(stable).fillna(False)

    def confidence(row: pd.Series) -> str:
        if row["ppm_window"] != 20:
            return "not_primary_window"
        if row["candidate_molecules"] == 0:
            return "no_library_candidate"
        strong = (
            row["dreams_p2b_agree"]
            and row["selected_mass_error_ppm"] <= 10
            and row["dreams_top_score"] >= 0.70
            and row["dreams_margin"] >= 0.03
            and row["target_beats_decoy"]
            and row["p2b_identity_stable_across_available_windows"]
        )
        moderate = (
            row["dreams_p2b_agree"]
            and row["dreams_top_score"] >= 0.60
            and row["target_beats_decoy"]
            and row["p2b_identity_stable_across_available_windows"]
        )
        if strong:
            return (
                "high_consistency_compound_level2_hypothesis"
                if row["selected_full_inchikey_count"] == 1
                else "high_consistency_connectivity_family_hypothesis"
            )
        if moderate:
            return (
                "moderate_consistency_compound_level2_hypothesis"
                if row["selected_full_inchikey_count"] == 1
                else "moderate_consistency_connectivity_family_hypothesis"
            )
        return "exploratory_only"

    summary["annotation_confidence"] = summary.apply(confidence, axis=1)
    primary = summary[summary["ppm_window"].eq(20)].copy()
    coverage = {}
    for ppm in ppm_windows:
        block = summary[summary["ppm_window"].eq(ppm)]
        coverage[str(int(ppm))] = {
            "queries_with_candidates": int((block["candidate_molecules"] > 0).sum()),
            "median_candidate_spectra": float(block["candidate_spectra"].median()),
            "dreams_p2b_agreement": int(block["dreams_p2b_agree"].sum()),
            "target_beats_decoy": int(block["target_beats_decoy"].sum()),
        }
    report = {
        "status": "lcnec_hsst3n_priority_annotation_complete",
        "formal": True,
        "queries": n_query,
        "library_spectra": n_library,
        "candidate_protocol": "precursor m/z constrained; spectra aggregated by IK14; ties deterministic but margins retained",
        "ppm_coverage": coverage,
        "primary_20ppm": {
            "queries_with_candidates": int((primary["candidate_molecules"] > 0).sum()),
            "high_consistency_compound_level2_hypotheses": int(primary["annotation_confidence"].eq("high_consistency_compound_level2_hypothesis").sum()),
            "high_consistency_connectivity_family_hypotheses": int(primary["annotation_confidence"].eq("high_consistency_connectivity_family_hypothesis").sum()),
            "moderate_consistency_compound_level2_hypotheses": int(primary["annotation_confidence"].eq("moderate_consistency_compound_level2_hypothesis").sum()),
            "moderate_consistency_connectivity_family_hypotheses": int(primary["annotation_confidence"].eq("moderate_consistency_connectivity_family_hypothesis").sum()),
            "exploratory_only": int(primary["annotation_confidence"].eq("exploratory_only").sum()),
            "no_library_candidate": int(primary["annotation_confidence"].eq("no_library_candidate").sum()),
            "stable_across_available_windows": int(primary["p2b_identity_stable_across_available_windows"].sum()),
        },
        "frozen_p2b_configuration": {
            "normalization": configuration.normalization,
            "weights": list(configuration.weights),
            "min_support": configuration.min_support,
            "min_advantage": configuration.min_advantage,
            "negative_mode_calibration_warning": "P2b was frozen outside this LCNEC negative-mode application and is not recalibrated here.",
        },
        "claim_limit": (
            "All names are spectral-library hypotheses. IK14 aggregation cannot distinguish stereoisomers, so multi-InChIKey "
            "groups are connectivity-family hypotheses rather than compound identities. Even compound-level rows remain MSI level 2 "
            "until an authentic standard confirms MS/MS and retention time. Target-decoy evidence is descriptive because only "
            f"{n_query} queries are evaluated."
        ),
        "provenance": {str(path.resolve().relative_to(ROOT)): sha256(path) for path in required},
    }

    summary.to_csv(args.output_dir / "priority_annotation_summary.csv", index=False)
    candidates.to_csv(args.output_dir / "priority_annotation_top5.csv", index=False)
    primary.to_csv(args.output_dir / "priority_annotation_primary20.csv", index=False)
    with (args.output_dir / "annotation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
