"""Low-cost falsification pilot for multi-standard chemical coordinates.

The pilot asks a deliberately narrow question before any sample-level atlas is
built: does the response of one reference spectrum identity to *many other*
reference identities contain structural-neighbour information beyond a direct
pairwise spectrum score?

No phenotype, sample abundance, identity label, or downstream reranker is used
to construct a score.  Molecular structures are opened only after all spectral
scores have been constructed and are used solely as held-out evaluation truth.
The exact query and candidate identities are excluded from the landmark set for
every profile comparison.

This is an inexpensive coordinate-headroom screen.  It is not an annotation,
MSI-level identity, transformation-site, or biological-discovery result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CHANNELS = (
    "sqrt_cosine",
    "entropy_similarity",
    "top10_match_fraction",
    "intensity_coverage_min",
    "matched_peak_fraction_min",
    "neutral_loss_sqrt_cosine",
)
SCORE_NAMES = (
    "sqrt_cosine",
    "entropy_similarity",
    "neutral_loss_sqrt_cosine",
    "precursor_mass_similarity",
    "direct_multichannel",
    "direct_mass_fusion_50_50",
    "multi_anchor_profile",
    "profile_fusion_50_50",
    "multi_anchor_augmented",
)


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=float)
    intensity = np.asarray(spectrum[1], dtype=float)
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz)
    return mz[order], intensity[order]


def greedy_matches(a: np.ndarray, b: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for i, value in enumerate(a):
        lo = int(np.searchsorted(b, value - tolerance, side="left"))
        hi = int(np.searchsorted(b, value + tolerance, side="right"))
        candidates.extend((abs(float(value - b[j])), i, j) for j in range(lo, hi))
    used_a: set[int] = set()
    used_b: set[int] = set()
    output: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            output.append((i, j))
    return output


def matched_metrics(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    if len(mz_a) == 0 or len(mz_b) == 0:
        return {
            "sqrt_cosine": 0.0,
            "entropy_similarity": 0.0,
            "query_intensity_coverage": 0.0,
            "candidate_intensity_coverage": 0.0,
            "matched_peak_fraction_min": 0.0,
            "top10_match_fraction": 0.0,
        }
    matches = greedy_matches(mz_a, mz_b, tolerance)
    ia = int_a / max(float(int_a.sum()), 1e-12)
    ib = int_b / max(float(int_b.sum()), 1e-12)
    sqrt_cosine = float(sum(math.sqrt(ia[i] * ib[j]) for i, j in matches))
    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    query_coverage = float(sum(ia[i] for i in matched_a))
    candidate_coverage = float(sum(ib[j] for j in matched_b))

    pa: list[float] = []
    pb: list[float] = []
    for i, j in matches:
        pa.append(float(ia[i]))
        pb.append(float(ib[j]))
    for i in set(range(len(ia))) - matched_a:
        pa.append(float(ia[i]))
        pb.append(0.0)
    for j in set(range(len(ib))) - matched_b:
        pa.append(0.0)
        pb.append(float(ib[j]))
    pa_array = np.asarray(pa, dtype=float)
    pb_array = np.asarray(pb, dtype=float)
    mean = 0.5 * (pa_array + pb_array)
    nz_a, nz_b = pa_array > 0, pb_array > 0
    js = 0.5 * np.sum(pa_array[nz_a] * np.log(pa_array[nz_a] / mean[nz_a]))
    js += 0.5 * np.sum(pb_array[nz_b] * np.log(pb_array[nz_b] / mean[nz_b]))
    entropy = float(np.clip(1.0 - js / np.log(2.0), 0.0, 1.0))

    top_a = set(np.argsort(int_a)[-min(10, len(int_a)):])
    top_b = set(np.argsort(int_b)[-min(10, len(int_b)):])
    top_matches = sum(i in top_a and j in top_b for i, j in matches)
    return {
        "sqrt_cosine": sqrt_cosine,
        "entropy_similarity": entropy,
        "query_intensity_coverage": query_coverage,
        "candidate_intensity_coverage": candidate_coverage,
        "matched_peak_fraction_min": len(matches) / max(1, min(len(mz_a), len(mz_b))),
        "top10_match_fraction": top_matches / max(1, min(10, len(mz_a), len(mz_b))),
    }


def pair_features(
    spectrum_a: np.ndarray,
    precursor_a: float,
    spectrum_b: np.ndarray,
    precursor_b: float,
    tolerance: float,
) -> dict[str, float]:
    mz_a, int_a = peaks(spectrum_a)
    mz_b, int_b = peaks(spectrum_b)
    fragment = matched_metrics(mz_a, int_a, mz_b, int_b, tolerance)
    loss_a = precursor_a - mz_a
    loss_b = precursor_b - mz_b
    keep_a, keep_b = loss_a > 0, loss_b > 0
    order_a = np.argsort(loss_a[keep_a])
    order_b = np.argsort(loss_b[keep_b])
    neutral = matched_metrics(
        loss_a[keep_a][order_a], int_a[keep_a][order_a],
        loss_b[keep_b][order_b], int_b[keep_b][order_b], tolerance,
    )
    fragment["neutral_loss_sqrt_cosine"] = neutral["sqrt_cosine"]
    fragment["intensity_coverage_min"] = min(
        fragment["query_intensity_coverage"], fragment["candidate_intensity_coverage"]
    )
    return {name: float(fragment[name]) for name in CHANNELS}


def load_mgf(path: Path) -> list[np.ndarray]:
    spectra: list[np.ndarray] = []
    mz: list[float] | None = None
    intensity: list[float] | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                mz, intensity = [], []
            elif line == "END IONS":
                if mz is None or intensity is None:
                    raise RuntimeError(f"malformed MGF block in {path}")
                spectra.append(np.asarray([mz, intensity], dtype=np.float32))
                mz = intensity = None
            elif mz is not None and "=" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mass, height = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    if mass > 0 and height > 0 and np.isfinite(mass) and np.isfinite(height):
                        mz.append(mass)
                        intensity.append(height)
    return spectra


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rank_concordance(left: np.ndarray, right: np.ndarray) -> float:
    """Return [0, 1] concordance of two landmark-response rankings."""
    if len(left) < 2:
        return math.nan
    a = average_ranks(np.asarray(left, dtype=float))
    b = average_ranks(np.asarray(right, dtype=float))
    denom = max(1.0, float(len(a) - 1))
    return float(np.clip(1.0 - np.mean(np.abs(a - b)) / denom, 0.0, 1.0))


def multi_anchor_score(matrices: dict[str, np.ndarray], query: int, candidate: int) -> float:
    """Compare two identities using third-party landmarks only.

    The direct query-candidate entry and both self entries are excluded by
    construction.  This is the non-search kernel that A1 is intended to test.
    """
    n = next(iter(matrices.values())).shape[0]
    anchors = np.asarray([value for value in range(n) if value not in (query, candidate)], dtype=int)
    return float(np.mean([
        rank_concordance(matrices[channel][query, anchors], matrices[channel][candidate, anchors])
        for channel in CHANNELS
    ]))


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return math.nan
    a, b = average_ranks(left), average_ranks(right)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom > 0 else 0.0


def ndcg_at_k(relevance: np.ndarray, ordering: np.ndarray, k: int) -> float:
    take = ordering[: min(k, len(ordering))]
    discounts = 1.0 / np.log2(np.arange(2, len(take) + 2))
    dcg = float(np.sum(relevance[take] * discounts))
    ideal = np.argsort(-relevance, kind="stable")[: len(take)]
    idcg = float(np.sum(relevance[ideal] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def fingerprint(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise RuntimeError(
            "RDKit is required for structure-only evaluation truth; run in the dreams conda environment"
        ) from error
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprint(molecule)


def tanimoto(left, right) -> float:
    from rdkit import DataStructs
    return float(DataStructs.TanimotoSimilarity(left, right))


def build_identity_responses(
    reference: pd.DataFrame,
    spectra_by_reference: dict[str, np.ndarray],
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    identities = (
        reference.groupby(["panel", "ik14"], sort=True)
        .agg(
            smiles=("smiles", "first"),
            representative_name=("name", "first"),
            hypothesis_family=("hypothesis_family", "first"),
            precursor_mz=("precursor_mz", "median"),
            reference_spectra=("reference_spectrum_id", "size"),
        )
        .reset_index()
    )
    responses: dict[str, dict[str, np.ndarray]] = {}
    for panel, panel_identities in identities.groupby("panel", sort=True):
        panel_identities = panel_identities.reset_index()
        n = len(panel_identities)
        matrices = {channel: np.eye(n, dtype=np.float32) for channel in CHANNELS}
        refs = {
            ik14: reference[(reference.panel == panel) & (reference.ik14 == ik14)]
            for ik14 in panel_identities.ik14
        }
        for i in range(n - 1):
            left_ik = str(panel_identities.at[i, "ik14"])
            for j in range(i + 1, n):
                right_ik = str(panel_identities.at[j, "ik14"])
                values: dict[str, list[float]] = defaultdict(list)
                for left in refs[left_ik].itertuples(index=False):
                    for right in refs[right_ik].itertuples(index=False):
                        features = pair_features(
                            spectra_by_reference[str(left.reference_spectrum_id)],
                            float(left.precursor_mz),
                            spectra_by_reference[str(right.reference_spectrum_id)],
                            float(right.precursor_mz),
                            tolerance,
                        )
                        for channel, value in features.items():
                            values[channel].append(value)
                for channel in CHANNELS:
                    # Max across experimental reference conditions is the fixed
                    # direct evidence; profile construction never sees labels.
                    value = max(values[channel]) if values[channel] else 0.0
                    matrices[channel][i, j] = matrices[channel][j, i] = value
        responses[str(panel)] = matrices
    return identities, responses


def evaluate_panel(
    panel: str,
    frame: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    minimum_anchors: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = frame.reset_index(drop=True)
    n = len(frame)
    fingerprints = [fingerprint(value) for value in frame.smiles]
    structural = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        for j in range(i + 1, n):
            value = tanimoto(fingerprints[i], fingerprints[j])
            structural[i, j] = structural[j, i] = value

    direct = np.mean(np.stack([matrices[name] for name in CHANNELS]), axis=0)
    precursor = frame.precursor_mz.to_numpy(float)
    mass_similarity = np.minimum.outer(precursor, precursor) / np.maximum.outer(precursor, precursor)
    profile = np.eye(n, dtype=np.float32)
    anchor_counts = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            anchors = np.asarray([a for a in range(n) if a not in (i, j)], dtype=int)
            anchor_counts.append(len(anchors))
            if len(anchors) < minimum_anchors:
                value = math.nan
            else:
                value = multi_anchor_score(matrices, i, j)
            profile[i, j] = profile[j, i] = value
    fusion = 0.5 * direct + 0.5 * profile
    direct_mass = 0.5 * direct + 0.5 * mass_similarity
    augmented = (direct + mass_similarity + profile) / 3.0
    score_matrices = {
        "sqrt_cosine": matrices["sqrt_cosine"],
        "entropy_similarity": matrices["entropy_similarity"],
        "neutral_loss_sqrt_cosine": matrices["neutral_loss_sqrt_cosine"],
        "precursor_mass_similarity": mass_similarity,
        "direct_multichannel": direct,
        "direct_mass_fusion_50_50": direct_mass,
        "multi_anchor_profile": profile,
        "profile_fusion_50_50": fusion,
        "multi_anchor_augmented": augmented,
    }

    rows: list[dict[str, object]] = []
    for query in range(n):
        candidates = np.asarray([value for value in range(n) if value != query], dtype=int)
        truth = structural[query, candidates]
        truth_order = np.argsort(-truth, kind="stable")
        threshold = truth[truth_order[min(2, len(truth_order) - 1)]]
        row: dict[str, object] = {
            "panel": panel,
            "query_ik14": str(frame.at[query, "ik14"]),
            "query_name": str(frame.at[query, "representative_name"]),
            "n_candidate_identities": int(len(candidates)),
            "n_landmarks_per_comparison": int(max(0, n - 2)),
            "best_available_structural_similarity": float(np.max(truth)),
        }
        for name, score_matrix in score_matrices.items():
            scores = score_matrix[query, candidates]
            ordering = np.lexsort((frame.loc[candidates, "ik14"].astype(str).to_numpy(), -scores))
            winner_local = int(ordering[0])
            winner = int(candidates[winner_local])
            selected_truth = float(truth[winner_local])
            row[f"{name}__selected_ik14"] = str(frame.at[winner, "ik14"])
            row[f"{name}__selected_structural_similarity"] = selected_truth
            row[f"{name}__structural_regret"] = float(np.max(truth) - selected_truth)
            row[f"{name}__top3_structural_hit"] = bool(selected_truth >= threshold - 1e-12)
            row[f"{name}__ndcg5"] = ndcg_at_k(truth, ordering, 5)
            row[f"{name}__candidate_spearman"] = spearman(scores, truth)
        rows.append(row)
    per_query = pd.DataFrame(rows)
    report = {
        "panel": panel,
        "identities": int(n),
        "spectra": int(frame.reference_spectra.sum()),
        "minimum_landmarks": int(min(anchor_counts)) if anchor_counts else 0,
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
    return per_query, report


def paired_bootstrap(
    per_query: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    clustered = per_query.assign(delta=(
        per_query[f"{candidate}__{metric}"].astype(float)
        - per_query[f"{baseline}__{metric}"].astype(float)
    )).groupby("query_ik14", sort=True).delta.mean()
    delta = clustered.to_numpy()
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for position in range(repeats):
        sample = rng.integers(0, len(delta), size=len(delta))
        values[position] = float(np.mean(delta[sample]))
    return {
        "mean_delta": float(np.mean(delta)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "identity_clusters": int(len(delta)),
        "resamples": int(repeats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4",
    )
    parser.add_argument(
        "--mgf-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_mgf_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/reference_anchored_multi_probe_a1_20260913",
    )
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-anchors", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.manifest_dir / "reference_spectra.csv.gz",
        args.manifest_dir / "reference_identities.csv",
        args.manifest_dir / "report.json",
        args.mgf_dir / "alignment.csv",
        args.mgf_dir / "neg_rp__reverse_probes.mgf",
        args.mgf_dir / "pos_rp__reverse_probes.mgf",
        args.mgf_dir / "report.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output_dir}")

    manifest_report = json.loads((args.manifest_dir / "report.json").read_text(encoding="utf-8"))
    if not manifest_report.get("pass_to_reverse_scan"):
        raise RuntimeError("reference-probe manifest did not pass its frozen gate")
    reference = pd.read_csv(args.manifest_dir / "reference_spectra.csv.gz")
    alignment = pd.read_csv(args.mgf_dir / "alignment.csv")
    if reference.reference_spectrum_id.duplicated().any():
        raise RuntimeError("duplicate reference_spectrum_id")

    spectra_by_reference: dict[str, np.ndarray] = {}
    for panel in ("neg_rp", "pos_rp"):
        panel_alignment = alignment[alignment.panel.eq(panel)].sort_values("subset_row", kind="stable")
        spectra = load_mgf(args.mgf_dir / f"{panel}__reverse_probes.mgf")
        if len(spectra) != len(panel_alignment):
            raise RuntimeError(f"{panel}: MGF/alignment length mismatch")
        expected = np.arange(len(panel_alignment))
        if not np.array_equal(panel_alignment.subset_row.to_numpy(int), expected):
            raise RuntimeError(f"{panel}: non-contiguous subset_row")
        spectra_by_reference.update({
            str(reference_id): spectrum
            for reference_id, spectrum in zip(panel_alignment.reference_spectrum_id, spectra)
        })
    if set(spectra_by_reference) != set(reference.reference_spectrum_id.astype(str)):
        raise RuntimeError("reference/alignment identity mismatch")

    identities, responses = build_identity_responses(
        reference, spectra_by_reference, args.fragment_tolerance
    )
    per_panel: list[pd.DataFrame] = []
    panel_reports: dict[str, object] = {}
    for panel, panel_frame in identities.groupby("panel", sort=True):
        if len(panel_frame) < args.minimum_anchors + 2:
            raise RuntimeError(f"{panel}: insufficient identities for excluded-anchor profile")
        per_query, panel_report = evaluate_panel(
            str(panel), panel_frame, responses[str(panel)], args.minimum_anchors
        )
        per_panel.append(per_query)
        panel_reports[str(panel)] = panel_report
    per_query = pd.concat(per_panel, ignore_index=True)

    comparisons = {}
    for candidate in ("multi_anchor_profile", "profile_fusion_50_50", "multi_anchor_augmented"):
        for metric in (
            "selected_structural_similarity", "top3_structural_hit", "ndcg5", "candidate_spearman"
        ):
            key = f"{candidate}_vs_direct_mass_fusion_50_50__{metric}"
            comparisons[key] = paired_bootstrap(
                per_query, candidate, "direct_mass_fusion_50_50", metric,
                args.bootstrap_resamples, args.seed + len(comparisons),
            )

    primary = comparisons["multi_anchor_augmented_vs_direct_mass_fusion_50_50__ndcg5"]
    panel_nonnegative = all(
        panel_reports[panel]["metrics"]["multi_anchor_augmented"]["mean_ndcg5"]
        >= panel_reports[panel]["metrics"]["direct_mass_fusion_50_50"]["mean_ndcg5"]
        for panel in panel_reports
    )
    report: dict[str, object] = {
        "status": "reference_anchored_multi_probe_a1_complete",
        "formal": False,
        "purpose": "low-cost leave-identity-out headroom screen for a multi-standard relational coordinate",
        "identities": int(len(identities)),
        "reference_spectra": int(len(reference)),
        "panels": panel_reports,
        "paired_identity_bootstrap": comparisons,
        "gates": {
            "identities_ge_50": bool(len(identities) >= 50),
            "each_panel_ge_20_identities": bool(all(value["identities"] >= 20 for value in panel_reports.values())),
            "multi_anchor_incremental_ndcg_ci_low_positive": bool(primary["ci_low"] > 0),
            "multi_anchor_incremental_ndcg_nonnegative_in_each_panel": bool(panel_nonnegative),
            "pass_to_official_dreams_confirmation": bool(primary["ci_low"] > 0 and panel_nonnegative),
        },
        "contracts": {
            "query_candidate_excluded_from_landmarks": True,
            "identity_labels_used_for_scoring": False,
            "structures_used_for_scoring": False,
            "structures_used_for_evaluation_only": True,
            "phenotype_used": False,
            "sample_abundance_used": False,
            "P2b_used": False,
            "official_or_experimental_embedding_used": False,
        },
        "provenance": {
            "reference_spectra": sha256(args.manifest_dir / "reference_spectra.csv.gz"),
            "alignment": sha256(args.mgf_dir / "alignment.csv"),
            "negative_mgf": sha256(args.mgf_dir / "neg_rp__reverse_probes.mgf"),
            "positive_mgf": sha256(args.mgf_dir / "pos_rp__reverse_probes.mgf"),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A1 tests whether third-party standard response profiles contain relational structural-neighbour signal. "
            "It is not sample annotation, a complete retained/shifted/lost/gained peak operator, an MSI identity, "
            "or a biological result. Failure stops this branch before expensive sample-level construction."
        ),
    }

    parent = args.output_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=parent))
    try:
        query_path = temporary / "per_query_coordinate_recovery.csv.gz"
        per_query.to_csv(query_path, index=False, compression="gzip")
        report["provenance"]["per_query"] = sha256(query_path)
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
