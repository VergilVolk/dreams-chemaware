"""Official-DreaMS confirmation of a multi-standard relational coordinate.

This bounded experiment encodes only the frozen reference spectra. It asks
whether a reference identity's response ranking to third-party standards adds
structural-neighbour information to direct official-DreaMS cosine. Query and
candidate identities are removed from their landmark set. Molecular structure
is evaluation truth only; no phenotype or biological sample is read.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from pilot_reference_anchored_multi_probe_a1 import (  # noqa: E402
    fingerprint,
    ndcg_at_k,
    paired_bootstrap,
    rank_concordance,
    sha256,
    spearman,
    tanimoto,
)


SCORE_NAMES = (
    "official_direct",
    "precursor_mass_similarity",
    "official_mass_50_50",
    "official_profile",
    "official_profile_fusion_50_50",
    "official_multi_anchor_augmented",
)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("embedding matrix must be finite and two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("embedding matrix contains a zero vector")
    return values / norms


def identity_table(reference: pd.DataFrame, panel: str) -> pd.DataFrame:
    subset = reference[reference.panel.eq(panel)].copy()
    if subset.empty:
        raise RuntimeError(f"{panel}: no reference spectra")
    # IK14 deliberately collapses stereoisomers. Morgan evaluation below uses
    # the default non-chiral connectivity fingerprint, so multiple stereo
    # SMILES within one IK14 are expected rather than an identity conflict.
    return (
        subset.groupby("ik14", sort=True)
        .agg(
            smiles=("smiles", "first"),
            distinct_stereo_smiles=("smiles", "nunique"),
            representative_name=("name", "first"),
            precursor_mz=("precursor_mz", "median"),
            reference_spectra=("reference_spectrum_id", "size"),
        )
        .reset_index()
    )


def build_identity_cosine(
    reference: pd.DataFrame,
    alignment: pd.DataFrame,
    cache_dir: Path,
    panel: str,
    expected_official_sha256: str,
    expected_mgf_sha256: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    manifest_path = cache_dir / "manifest.csv"
    embedding_path = cache_dir / "embeddings.npy"
    report_path = cache_dir / "report.json"
    for path in (manifest_path, embedding_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_report = json.loads(report_path.read_text(encoding="utf-8"))
    if cache_report.get("status") != "unified_library_p2b_cache_complete":
        raise RuntimeError(f"{panel}: incomplete official embedding cache")
    if cache_report.get("model", {}).get("kind") != "official_dreams":
        raise RuntimeError(f"{panel}: cache model is not official DreaMS")
    observed_official = cache_report.get("provenance", {}).get("official_checkpoint_sha256")
    if observed_official != expected_official_sha256:
        raise RuntimeError(f"{panel}: cache was not encoded by the frozen official checkpoint")
    observed_mgf = cache_report.get("provenance", {}).get("mgf_sha256")
    if observed_mgf != expected_mgf_sha256:
        raise RuntimeError(f"{panel}: cache MGF differs from the frozen reverse-probe MGF")
    panel_alignment = alignment[alignment.panel.eq(panel)].sort_values("subset_row", kind="stable")
    expected = np.arange(len(panel_alignment), dtype=int)
    if not np.array_equal(panel_alignment.subset_row.to_numpy(int), expected):
        raise RuntimeError(f"{panel}: non-contiguous subset rows")
    cache_manifest = pd.read_csv(manifest_path)
    if cache_report.get("provenance", {}).get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError(f"{panel}: cache manifest hash mismatch")
    embeddings = np.load(embedding_path, mmap_mode="r")
    if not (len(panel_alignment) == len(cache_manifest) == len(embeddings)):
        raise RuntimeError(f"{panel}: alignment/cache length mismatch")
    ref_by_id = reference.set_index("reference_spectrum_id", drop=False)
    try:
        ordered = ref_by_id.loc[panel_alignment.reference_spectrum_id.astype(str)].reset_index(drop=True)
    except KeyError as error:
        raise RuntimeError(f"{panel}: alignment references absent from frozen manifest") from error
    if not np.array_equal(
        ordered.ik14.fillna("").astype(str).to_numpy(),
        cache_manifest.ik14.fillna("").astype(str).to_numpy(),
    ):
        raise RuntimeError(f"{panel}: official cache identity order differs from alignment")
    vectors = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    identities = identity_table(reference, panel)
    spectrum_positions: dict[str, list[int]] = {
        value: [] for value in identities.ik14.astype(str)
    }
    for row, ik14 in enumerate(ordered.ik14.astype(str)):
        spectrum_positions[ik14].append(row)
    if any(not rows for rows in spectrum_positions.values()):
        raise RuntimeError(f"{panel}: identity without encoded reference spectrum")
    n = len(identities)
    direct = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        left = vectors[np.asarray(spectrum_positions[str(identities.at[i, "ik14"])], dtype=int)]
        for j in range(i + 1, n):
            right = vectors[np.asarray(spectrum_positions[str(identities.at[j, "ik14"])], dtype=int)]
            value = float(np.max(left @ right.T))
            direct[i, j] = direct[j, i] = value
    return identities, direct, {
        "cache_report_sha256": sha256(report_path),
        "cache_manifest_sha256": sha256(manifest_path),
        "cache_embeddings_sha256": sha256(embedding_path),
        "embedding_dimension": int(vectors.shape[1]),
        "spectra": int(len(vectors)),
    }


def build_profile(direct: np.ndarray, minimum_anchors: int) -> tuple[np.ndarray, int]:
    n = direct.shape[0]
    if direct.shape != (n, n):
        raise RuntimeError("direct matrix must be square")
    if n - 2 < minimum_anchors:
        raise RuntimeError("insufficient third-party landmark identities")
    profile = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        for j in range(i + 1, n):
            anchors = np.asarray([a for a in range(n) if a not in (i, j)], dtype=int)
            value = rank_concordance(direct[i, anchors], direct[j, anchors])
            profile[i, j] = profile[j, i] = value
    return profile, n - 2


def evaluate_panel(
    panel: str,
    identities: pd.DataFrame,
    direct: np.ndarray,
    minimum_anchors: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    identities = identities.reset_index(drop=True)
    n = len(identities)
    if direct.shape != (n, n):
        raise RuntimeError(f"{panel}: identity/direct matrix mismatch")
    fingerprints = [fingerprint(value) for value in identities.smiles]
    structural = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        for j in range(i + 1, n):
            structural[i, j] = structural[j, i] = tanimoto(fingerprints[i], fingerprints[j])
    profile, landmark_count = build_profile(direct, minimum_anchors)
    precursor = identities.precursor_mz.to_numpy(float)
    if not np.isfinite(precursor).all() or np.any(precursor <= 0):
        raise RuntimeError(f"{panel}: invalid precursor mass")
    mass = np.minimum.outer(precursor, precursor) / np.maximum.outer(precursor, precursor)
    score_matrices = {
        "official_direct": direct,
        "precursor_mass_similarity": mass,
        "official_mass_50_50": 0.5 * direct + 0.5 * mass,
        "official_profile": profile,
        "official_profile_fusion_50_50": 0.5 * direct + 0.5 * profile,
        "official_multi_anchor_augmented": (direct + mass + profile) / 3.0,
    }
    rows: list[dict[str, object]] = []
    for query in range(n):
        candidates = np.asarray([value for value in range(n) if value != query], dtype=int)
        truth = structural[query, candidates]
        truth_order = np.argsort(-truth, kind="stable")
        threshold = truth[truth_order[min(2, len(truth_order) - 1)]]
        row: dict[str, object] = {
            "panel": panel,
            "query_ik14": str(identities.at[query, "ik14"]),
            "query_name": str(identities.at[query, "representative_name"]),
            "n_candidate_identities": int(len(candidates)),
            "n_landmarks_per_comparison": int(landmark_count),
            "best_available_structural_similarity": float(np.max(truth)),
        }
        tie_keys = identities.loc[candidates, "ik14"].astype(str).to_numpy()
        for name, matrix in score_matrices.items():
            scores = np.asarray(matrix[query, candidates], dtype=float)
            ordering = np.lexsort((tie_keys, -scores))
            winner_local = int(ordering[0])
            winner = int(candidates[winner_local])
            selected = float(truth[winner_local])
            row[f"{name}__selected_ik14"] = str(identities.at[winner, "ik14"])
            row[f"{name}__selected_structural_similarity"] = selected
            row[f"{name}__structural_regret"] = float(np.max(truth) - selected)
            row[f"{name}__top3_structural_hit"] = bool(selected >= threshold - 1e-12)
            row[f"{name}__ndcg5"] = ndcg_at_k(truth, ordering, 5)
            row[f"{name}__candidate_spearman"] = spearman(scores, truth)
        rows.append(row)
    per_query = pd.DataFrame(rows)
    summary = {
        "panel": panel,
        "identities": int(n),
        "spectra": int(identities.reference_spectra.sum()),
        "landmarks_per_comparison": int(landmark_count),
        "metrics": {
            name: {
                "mean_selected_structural_similarity": float(per_query[f"{name}__selected_structural_similarity"].mean()),
                "mean_structural_regret": float(per_query[f"{name}__structural_regret"].mean()),
                "top3_structural_hit": float(per_query[f"{name}__top3_structural_hit"].mean()),
                "mean_ndcg5": float(per_query[f"{name}__ndcg5"].mean()),
                "mean_candidate_spearman": float(per_query[f"{name}__candidate_spearman"].mean()),
            }
            for name in SCORE_NAMES
        },
    }
    return per_query, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4")
    parser.add_argument("--alignment", type=Path, default=ROOT / "data/mtbls13729/reverse_probe_mgf_v1/alignment.csv")
    parser.add_argument("--mgf-dir", type=Path, default=ROOT / "data/mtbls13729/reverse_probe_mgf_v1")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "data/mtbls13729/reverse_probe_embedding_cache_v1/official")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--strong-baseline-audit", type=Path, default=ROOT / "data/validation/reference_anchored_multi_probe_a1_strong_baseline_20260913.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/reference_anchored_multi_probe_a1b_official_20260913")
    parser.add_argument("--minimum-anchors", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = args.manifest_dir / "reference_spectra.csv.gz"
    required = [reference_path, args.alignment, args.strong_baseline_audit, args.official_checkpoint]
    for panel in ("neg_rp", "pos_rp"):
        required.extend([
            args.mgf_dir / f"{panel}__reverse_probes.mgf",
            args.cache_root / panel / "manifest.csv",
            args.cache_root / panel / "embeddings.npy",
            args.cache_root / panel / "report.json",
        ])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing A1b output: {args.output_dir}")
    audit = json.loads(args.strong_baseline_audit.read_text(encoding="utf-8"))
    if not audit.get("gates", {}).get("worth_one_official_dreams_confirmation"):
        raise RuntimeError("A1 strong-baseline audit did not authorize A1b")
    reference = pd.read_csv(reference_path)
    alignment = pd.read_csv(args.alignment)
    official_sha256 = sha256(args.official_checkpoint)
    if reference.reference_spectrum_id.duplicated().any():
        raise RuntimeError("duplicate reference_spectrum_id")
    reference["reference_spectrum_id"] = reference.reference_spectrum_id.astype(str)
    alignment["reference_spectrum_id"] = alignment.reference_spectrum_id.astype(str)
    if set(reference.reference_spectrum_id) != set(alignment.reference_spectrum_id):
        raise RuntimeError("frozen reference manifest and MGF alignment differ")

    per_panel: list[pd.DataFrame] = []
    panels: dict[str, object] = {}
    cache_provenance: dict[str, object] = {}
    for panel in ("neg_rp", "pos_rp"):
        mgf_sha256 = sha256(args.mgf_dir / f"{panel}__reverse_probes.mgf")
        identities, direct, cache_info = build_identity_cosine(
            reference, alignment, args.cache_root / panel, panel,
            official_sha256, mgf_sha256,
        )
        per_query, summary = evaluate_panel(panel, identities, direct, args.minimum_anchors)
        per_panel.append(per_query)
        panels[panel] = summary
        cache_provenance[panel] = cache_info
    frame = pd.concat(per_panel, ignore_index=True)

    metrics = ("selected_structural_similarity", "top3_structural_hit", "ndcg5", "candidate_spearman")
    comparisons: dict[str, object] = {}
    comparison_specs = (
        ("official_profile", "official_direct"),
        ("official_profile_fusion_50_50", "official_direct"),
        ("official_multi_anchor_augmented", "official_mass_50_50"),
    )
    for candidate, baseline in comparison_specs:
        for metric in metrics:
            key = f"{candidate}_vs_{baseline}__{metric}"
            comparisons[key] = paired_bootstrap(
                frame, candidate, baseline, metric,
                args.bootstrap_resamples, args.seed + len(comparisons),
            )

    primary = comparisons["official_profile_fusion_50_50_vs_official_direct__ndcg5"]
    panel_primary_delta = {
        panel: float(
            summary["metrics"]["official_profile_fusion_50_50"]["mean_ndcg5"]
            - summary["metrics"]["official_direct"]["mean_ndcg5"]
        )
        for panel, summary in panels.items()
    }
    mass_control = comparisons[
        "official_multi_anchor_augmented_vs_official_mass_50_50__ndcg5"
    ]
    report: dict[str, object] = {
        "status": "reference_anchored_multi_probe_a1b_official_complete",
        "formal": False,
        "purpose": "official-DreaMS confirmation of a leave-query-and-candidate-out multi-standard relational coordinate",
        "identities": int(len(frame)),
        "independent_ik14_clusters": int(frame.query_ik14.nunique()),
        "reference_spectra": int(len(reference)),
        "panels": panels,
        "paired_ik14_bootstrap": comparisons,
        "panel_primary_ndcg_delta": panel_primary_delta,
        "gates": {
            "primary_fusion_vs_official_direct_ndcg_ci_low_positive": bool(primary["ci_low"] > 0),
            "primary_fusion_vs_official_direct_nonnegative_each_panel": bool(all(v >= 0 for v in panel_primary_delta.values())),
            "mass_control_incremental_ndcg_ci_low_positive": bool(mass_control["ci_low"] > 0),
            "pass_to_sample_level_peak_operator": bool(
                primary["ci_low"] > 0 and all(v >= 0 for v in panel_primary_delta.values())
            ),
        },
        "contracts": {
            "official_dreams_shared_encoder": True,
            "query_candidate_excluded_from_landmarks": True,
            "fixed_profile_fusion_weights": [0.5, 0.5],
            "weight_search_used": False,
            "identity_labels_used_for_scoring": False,
            "structures_used_for_scoring": False,
            "structures_used_for_evaluation_only": True,
            "phenotype_used": False,
            "sample_abundance_used": False,
            "P2b_used": False,
        },
        "provenance": {
            "reference_spectra_sha256": sha256(reference_path),
            "alignment_sha256": sha256(args.alignment),
            "official_checkpoint_sha256": official_sha256,
            "strong_baseline_audit_sha256": sha256(args.strong_baseline_audit),
            "official_caches": cache_provenance,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A1b tests structural-neighbour headroom among 69 independent IK14 clusters using only public reference spectra. "
            "It is not sample annotation, exact identity confirmation, a biological atlas, or evidence of a new metabolite."
        ),
    }
    parent = args.output_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=parent))
    try:
        per_query_path = temporary / "per_query_official_coordinate.csv.gz"
        frame.to_csv(per_query_path, index=False, compression="gzip")
        report["provenance"]["per_query_sha256"] = sha256(per_query_path)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output_dir.resolve())
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
