"""Separate spectrum-limited retrieval failures from DreaMS residual failures.

The audit uses molecule-disjoint discovery and confirmation cohorts.  A shallow
classifier is fitted on discovery spectra only, using hand-auditable raw MS/MS
pair features.  It is not treated as an oracle or a performance competitor;
its sole purpose is to ask whether the raw spectra contain enough conventional
matching evidence to rank the identity positive above same-formula negatives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "sqrt_cosine",
    "linear_cosine",
    "entropy_similarity",
    "query_intensity_coverage",
    "candidate_intensity_coverage",
    "matched_peak_fraction_min",
    "top10_match_fraction",
    "neutral_loss_sqrt_cosine",
    "neutral_loss_query_coverage",
    "neutral_loss_candidate_coverage",
    "peak_count_ratio",
]

CONSENSUS_FEATURES = [
    "entropy_similarity",
    "sqrt_cosine",
    "linear_cosine",
    "top10_match_fraction",
    "query_intensity_coverage",
]


def peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mz, intensity = np.asarray(spectrum[0], float), np.asarray(spectrum[1], float)
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz)
    return mz[order], intensity[order]


def greedy_matches(a: np.ndarray, b: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    candidates = []
    for i, value in enumerate(a):
        lo = int(np.searchsorted(b, value - tolerance, side="left"))
        hi = int(np.searchsorted(b, value + tolerance, side="right"))
        candidates.extend((abs(value - b[j]), i, j) for j in range(lo, hi))
    used_a, used_b, output = set(), set(), []
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
            "linear_cosine": 0.0,
            "entropy_similarity": 0.0,
            "query_intensity_coverage": 0.0,
            "candidate_intensity_coverage": 0.0,
            "matched_peak_fraction_min": 0.0,
            "top10_match_fraction": 0.0,
        }
    matches = greedy_matches(mz_a, mz_b, tolerance)
    ia = int_a / max(float(int_a.sum()), 1e-12)
    ib = int_b / max(float(int_b.sum()), 1e-12)
    sa = np.sqrt(ia)
    sb = np.sqrt(ib)
    linear_a = int_a / max(float(np.linalg.norm(int_a)), 1e-12)
    linear_b = int_b / max(float(np.linalg.norm(int_b)), 1e-12)
    sqrt_cos = float(sum(sa[i] * sb[j] for i, j in matches))
    linear_cos = float(sum(linear_a[i] * linear_b[j] for i, j in matches))
    matched_a = {i for i, _ in matches}
    matched_b = {j for _, j in matches}
    query_coverage = float(sum(ia[i] for i in matched_a))
    candidate_coverage = float(sum(ib[j] for j in matched_b))

    pa, pb = [], []
    for i, j in matches:
        pa.append(ia[i]); pb.append(ib[j])
    for i in set(range(len(ia))) - matched_a:
        pa.append(ia[i]); pb.append(0.0)
    for j in set(range(len(ib))) - matched_b:
        pa.append(0.0); pb.append(ib[j])
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    mean = 0.5 * (pa + pb)
    nz_a, nz_b = pa > 0, pb > 0
    js = 0.5 * np.sum(pa[nz_a] * np.log(pa[nz_a] / mean[nz_a]))
    js += 0.5 * np.sum(pb[nz_b] * np.log(pb[nz_b] / mean[nz_b]))
    entropy_similarity = float(np.clip(1.0 - js / np.log(2.0), 0.0, 1.0))

    top_a = set(np.argsort(int_a)[-min(10, len(int_a)):])
    top_b = set(np.argsort(int_b)[-min(10, len(int_b)):])
    top_matches = sum(i in top_a and j in top_b for i, j in matches)
    return {
        "sqrt_cosine": sqrt_cos,
        "linear_cosine": linear_cos,
        "entropy_similarity": entropy_similarity,
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
    loss_a, keep_a = precursor_a - mz_a, precursor_a - mz_a > 0
    loss_b, keep_b = precursor_b - mz_b, precursor_b - mz_b > 0
    neutral = matched_metrics(
        np.sort(loss_a[keep_a]), int_a[keep_a][np.argsort(loss_a[keep_a])],
        np.sort(loss_b[keep_b]), int_b[keep_b][np.argsort(loss_b[keep_b])], tolerance,
    )
    fragment.update({
        "neutral_loss_sqrt_cosine": neutral["sqrt_cosine"],
        "neutral_loss_query_coverage": neutral["query_intensity_coverage"],
        "neutral_loss_candidate_coverage": neutral["candidate_intensity_coverage"],
        "peak_count_ratio": min(len(mz_a), len(mz_b)) / max(1, max(len(mz_a), len(mz_b))),
    })
    return fragment


def build_pair_table(pilot_dir: Path, split: str, tolerance: float) -> tuple[pd.DataFrame, list[dict]]:
    units = json.loads((pilot_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
    source = np.load(pilot_dir / f"{split}_spectra.npz")
    spectra, precursor = source["spectra"], source["precursor_mz"]
    rows = []
    for unit in units:
        if not unit["is_query_anchor"] or not unit["same_formula_negative_pair_ids"]:
            continue
        pair_id = int(unit["pair_id"])
        for query_view in (0, 1):
            common = {
                "split": split, "pair_id": pair_id, "ik14": unit["ik14"],
                "query_view": query_view, "formula": unit["formula"],
                "ring_class": unit["ring_class"],
            }
            positive = pair_features(
                spectra[pair_id, query_view], float(precursor[pair_id, query_view]),
                spectra[pair_id, 1 - query_view], float(precursor[pair_id, 1 - query_view]), tolerance,
            )
            rows.append(common | {
                "label": 1, "candidate_pair_id": pair_id,
                "candidate_ik14": unit["ik14"], "candidate_view": 1 - query_view,
            } | positive)
            for negative_id in map(int, unit["same_formula_negative_pair_ids"]):
                for candidate_view in (0, 1):
                    negative = pair_features(
                        spectra[pair_id, query_view], float(precursor[pair_id, query_view]),
                        spectra[negative_id, candidate_view], float(precursor[negative_id, candidate_view]), tolerance,
                    )
                    rows.append(common | {
                        "label": 0, "candidate_pair_id": negative_id,
                        "candidate_ik14": units[negative_id]["ik14"],
                        "candidate_view": candidate_view,
                    } | negative)
    return pd.DataFrame(rows), units


def raw_retrieval(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "pair_id", "ik14", "query_view", "formula", "ring_class"]
    for values, group in pair_table.groupby(keys, sort=False):
        positive = float(group.loc[group["label"] == 1, "raw_identity_score"].iloc[0])
        negatives = group.loc[group["label"] == 0]
        per_molecule = negatives.groupby(["candidate_pair_id", "candidate_ik14"], sort=False)["raw_identity_score"].max()
        best_id, best_ik = per_molecule.idxmax()
        best_score = float(per_molecule.max())
        row = dict(zip(keys, values))
        row.update({
            "raw_positive_score": positive,
            "raw_best_negative_score": best_score,
            "raw_best_negative_pair_id": int(best_id),
            "raw_best_negative_ik14": best_ik,
            "raw_margin": positive - best_score,
            "raw_top1_correct": bool(positive > best_score),
        })
        for feature in CONSENSUS_FEATURES:
            feature_positive = float(group.loc[group["label"] == 1, feature].iloc[0])
            feature_negative = negatives.groupby("candidate_pair_id", sort=False)[feature].max()
            feature_best = float(feature_negative.max())
            row[f"{feature}_margin"] = feature_positive - feature_best
            row[f"{feature}_top1_correct"] = bool(feature_positive > feature_best)
        rows.append(row)
    return pd.DataFrame(rows)


def quadrant(dreams_ok: bool, raw_ok: bool) -> str:
    if dreams_ok and raw_ok:
        return "both_correct"
    if not dreams_ok and raw_ok:
        return "model_residual_candidate"
    if not dreams_ok and not raw_ok:
        return "shared_or_spectrum_limited"
    return "dreams_only_correct"


def summarize(frame: pd.DataFrame, seed: int = 20260813) -> dict:
    counts = frame["audit_quadrant"].value_counts().to_dict()
    molecule = frame.groupby("ik14", sort=False).agg(
        dreams_failed_any=("top1_correct", lambda x: bool((~x).any())),
        raw_failed_any=("raw_top1_correct", lambda x: bool((~x).any())),
        model_residual_any=("audit_quadrant", lambda x: bool((x == "model_residual_candidate").any())),
        dreams_margin=("margin", "mean"), raw_margin=("raw_margin", "mean"),
    )
    rng = np.random.default_rng(seed)
    molecule_ids = frame["ik14"].unique()
    grouped = {key: value for key, value in frame.groupby("ik14", sort=False)}
    delta_draws = []
    for _ in range(5000):
        sampled = rng.choice(molecule_ids, len(molecule_ids), replace=True)
        sample = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        delta_draws.append(float(sample["raw_top1_correct"].mean() - sample["top1_correct"].mean()))
    return {
        "query_views": int(len(frame)),
        "query_molecules": int(frame["ik14"].nunique()),
        "dreams_top1": float(frame["top1_correct"].mean()),
        "raw_proxy_top1": float(frame["raw_top1_correct"].mean()),
        "view_quadrants": {str(k): int(v) for k, v in counts.items()},
        "molecules_dreams_failed_any": int(molecule["dreams_failed_any"].sum()),
        "molecules_raw_failed_any": int(molecule["raw_failed_any"].sum()),
        "molecules_with_model_residual_candidate": int(molecule["model_residual_any"].sum()),
        "robust_model_residual_views_consensus_ge_3": int(frame["robust_model_residual_candidate"].sum()),
        "robust_model_residual_molecules_consensus_ge_3": int(
            frame.loc[frame["robust_model_residual_candidate"], "ik14"].nunique()
        ),
        "raw_minus_dreams_top1": float(frame["raw_top1_correct"].mean() - frame["top1_correct"].mean()),
        "raw_minus_dreams_top1_cluster_bootstrap_ci95": np.quantile(delta_draws, [0.025, 0.975]).tolist(),
        "pearson_margin_correlation": float(frame[["margin", "raw_margin"]].corr(method="pearson").iloc[0, 1]),
        "spearman_margin_correlation": float(frame[["margin", "raw_margin"]].corr(method="spearman").iloc[0, 1]),
    }


def plot_audit(frame: pd.DataFrame, output: Path) -> None:
    colors = {
        "both_correct": "#4C78A8",
        "model_residual_candidate": "#E45756",
        "shared_or_spectrum_limited": "#9D755D",
        "dreams_only_correct": "#72B7B2",
    }
    fig, ax = plt.subplots(figsize=(9.2, 7.0), dpi=180)
    for label, subset in frame.groupby("audit_quadrant"):
        ax.scatter(subset["raw_margin"], subset["margin"], s=38, alpha=0.78,
                   color=colors[label], edgecolor="white", linewidth=0.35,
                   label=f"{label} (n={len(subset)})")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Raw-spectrum proxy ranking margin")
    ax.set_ylabel("DreaMS ranking margin")
    ax.set_title("Observability–residual audit: confirmation cohort")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/validation/external_ring_balanced_pilot"))
    parser.add_argument("--e0-results", type=Path, default=Path("data/validation/external_ring_balanced_e0/query_results.csv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/validation/external_e0_failure_taxonomy/failure_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/e0_observability_residual_audit"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    discovery, _ = build_pair_table(args.pilot_dir, "discovery", args.tolerance)
    confirmation, confirmation_units = build_pair_table(args.pilot_dir, "confirmation", args.tolerance)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=20260813),
    )
    model.fit(discovery[FEATURES], discovery["label"])
    discovery["raw_identity_score"] = model.predict_proba(discovery[FEATURES])[:, 1]
    confirmation["raw_identity_score"] = model.predict_proba(confirmation[FEATURES])[:, 1]
    discovery.to_csv(args.output_dir / "discovery_pair_features.csv", index=False)
    confirmation.to_csv(args.output_dir / "confirmation_pair_features.csv", index=False)
    standardized_coefficients = pd.DataFrame({
        "feature": FEATURES,
        "standardized_logistic_coefficient": model.named_steps["logisticregression"].coef_[0],
    }).sort_values("standardized_logistic_coefficient", ascending=False)
    standardized_coefficients.to_csv(args.output_dir / "raw_proxy_coefficients.csv", index=False)

    raw = pd.concat((raw_retrieval(discovery), raw_retrieval(confirmation)), ignore_index=True)
    dreams = pd.read_csv(args.e0_results)
    dreams = dreams.loc[dreams["candidate_protocol"] == "same_formula_negative_pair_ids"].copy()
    keep = [
        "split", "pair_id", "ik14", "query_view", "positive_similarity",
        "best_negative_similarity", "margin", "top1_correct",
        "best_negative_pair_id", "best_negative_ik14",
    ]
    audit = dreams[keep].merge(raw, on=["split", "pair_id", "ik14", "query_view"], validate="one_to_one")
    audit["audit_quadrant"] = [quadrant(bool(d), bool(r)) for d, r in zip(audit["top1_correct"], audit["raw_top1_correct"])]
    consensus_columns = [f"{feature}_top1_correct" for feature in CONSENSUS_FEATURES]
    audit["raw_metric_consensus_votes"] = audit[consensus_columns].sum(axis=1).astype(int)
    audit["robust_model_residual_candidate"] = (
        (~audit["top1_correct"]) & (audit["raw_metric_consensus_votes"] >= 3)
    )

    unit_maps = {}
    for split in ("discovery", "confirmation"):
        units = json.loads((args.pilot_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["units"]
        unit_maps[split] = {int(unit["pair_id"]): unit for unit in units}
    audit["query_smiles"] = [unit_maps[s][int(i)]["smiles"] for s, i in zip(audit["split"], audit["pair_id"])]
    audit["dreams_negative_smiles"] = [unit_maps[s][int(i)]["smiles"] for s, i in zip(audit["split"], audit["best_negative_pair_id"])]
    audit["raw_negative_smiles"] = [unit_maps[s][int(i)]["smiles"] for s, i in zip(audit["split"], audit["raw_best_negative_pair_id"])]

    if args.taxonomy.exists():
        taxonomy = pd.read_csv(args.taxonomy)
        taxonomy = taxonomy.rename(columns={"query_ik14": "ik14", "negative_ik14": "best_negative_ik14"})
        add = ["ik14", "best_negative_ik14", "mces", "morgan_tanimoto", "edit_class_candidate", "functional_group_changes"]
        audit = audit.merge(taxonomy[add], on=["ik14", "best_negative_ik14"], how="left")
    audit.to_csv(args.output_dir / "query_audit.csv", index=False)
    confirmation_audit = audit.loc[audit["split"] == "confirmation"].copy()
    confirmation_audit.sort_values(["audit_quadrant", "margin"]).to_csv(args.output_dir / "confirmation_cases.csv", index=False)

    pair_auc = {
        "discovery": float(roc_auc_score(discovery["label"], discovery["raw_identity_score"])),
        "confirmation": float(roc_auc_score(confirmation["label"], confirmation["raw_identity_score"])),
    }
    report = {
        "status": "e0_observability_residual_audit",
        "protocol": "same-formula, same-adduct, 10-ppm candidate retrieval",
        "raw_proxy": {
            "fit_split": "discovery only",
            "features": FEATURES,
            "consensus_features": CONSENSUS_FEATURES,
            "pair_auc": pair_auc,
            "interpretation": "Conventional raw-spectrum separability proxy, not a Bayes ceiling or causal oracle.",
        },
        "splits": {
            split: summarize(audit.loc[audit["split"] == split])
            for split in ("discovery", "confirmation")
        },
        "quadrants": {
            "model_residual_candidate": "Raw proxy ranks identity correctly while DreaMS fails; priority for peak-level audit.",
            "shared_or_spectrum_limited": "Both methods fail; evidence may be weak, but stronger models are needed before calling it irreducible.",
            "dreams_only_correct": "DreaMS succeeds where the shallow raw proxy fails; useful control and possible learned advantage.",
            "both_correct": "Both methods rank identity correctly.",
        },
        "claim_limit": "Small external annotated01 pilot; provenance is incomplete and overlap with DreaMS SSL pretraining cannot be excluded.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_audit(confirmation_audit, args.output_dir / "confirmation_observability_residual.png")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
