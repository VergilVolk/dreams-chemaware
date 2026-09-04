#!/usr/bin/env python
"""Exploratory full-space paired audit for MTBLS13729 MS1 consensus targets.

This audit is deliberately upstream of metabolite annotation.  It tests every
predefined re-quantification target, keeps the paired Rmu-versus-RN endpoint
primary, checks phenotype-blind PQN sensitivity, and collapses obvious
co-eluting isotope/adduct relations before reporting discovery counts.

The discovery matrices are peak-picker outputs, not targeted EIC
re-extractions.  Therefore this script produces candidates for technical
re-extraction and MS2 review; it cannot establish metabolite identity or an
independent biological replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind


SAMPLE_RE = re.compile(r"^(P\d{2})-(Ltu|Rtu|Rmu|LN|RN)$")
PANELS = ("neg_rp", "pos_rp")
COMMON_MASS_DELTAS = {
    "coordinate_duplicate": 0.0,
    "isotope_13C": 1.003355,
    "NH4_minus_H": 17.026549,
    "H2O": 18.010565,
    "Na_minus_H": 21.981943,
    "K_minus_H": 37.955882,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    p = values[valid]
    if not len(p):
        return out
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    valid_positions = np.flatnonzero(valid)
    out[valid_positions[order]] = np.clip(adjusted, 0.0, 1.0)
    return out


def exact_signflip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return math.nan
    observed = abs(float(values.mean()))
    signs = 1.0 - 2.0 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def sample_pairs(columns: list[str], tumour_suffix: str, normal_suffix: str) -> list[tuple[str, str]]:
    lookup: dict[tuple[str, str], str] = {}
    for sample in columns:
        match = SAMPLE_RE.match(sample)
        if match:
            lookup[(match.group(1), match.group(2))] = sample
    patients = sorted(patient for patient, suffix in lookup if suffix == tumour_suffix)
    return [
        (lookup[(patient, tumour_suffix)], lookup[(patient, normal_suffix)])
        for patient in patients
        if (patient, normal_suffix) in lookup
    ]


def pqn(log_matrix: pd.DataFrame, prevalence: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    background = prevalence.reindex(log_matrix.index).fillna(0.0) >= 0.60
    reference = log_matrix.loc[background].median(axis=1, skipna=True)
    quotients = log_matrix.loc[background].sub(reference, axis=0)
    factors = quotients.median(axis=0, skipna=True)
    return log_matrix.sub(factors, axis=1), factors


def paired_summary(matrix: pd.DataFrame, pairs: list[tuple[str, str]], prefix: str) -> pd.DataFrame:
    tumour = matrix[[pair[0] for pair in pairs]].to_numpy(float)
    normal = matrix[[pair[1] for pair in pairs]].to_numpy(float)
    delta = tumour - normal
    rows: list[dict[str, float | int | bool]] = []
    for values in delta:
        values = values[np.isfinite(values)]
        n = len(values)
        if not n:
            rows.append({f"{prefix}_n": 0})
            continue
        mean = float(values.mean())
        t_p = (
            float(ttest_1samp(values, 0.0).pvalue)
            if n >= 2 and float(np.std(values)) > 0.0
            else 1.0
        )
        loo = np.asarray([np.delete(values, i).mean() for i in range(n)]) if n > 1 else np.asarray([mean])
        rows.append(
            {
                f"{prefix}_n": n,
                f"{prefix}_mean_log2fc": mean,
                f"{prefix}_median_log2fc": float(np.median(values)),
                f"{prefix}_positive_fraction": float(np.mean(values > 0.0)),
                f"{prefix}_ttest_p": t_p,
                f"{prefix}_exact_signflip_p": exact_signflip_p(values),
                f"{prefix}_loo_direction_stable": bool(np.all(np.sign(loo) == np.sign(mean))),
            }
        )
    return pd.DataFrame(rows, index=matrix.index)


def interaction_summary(
    matrix: pd.DataFrame,
    rmu_pairs: list[tuple[str, str]],
    rtu_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    rmu = matrix[[x for pair in rmu_pairs for x in pair]].copy()
    rtu = matrix[[x for pair in rtu_pairs for x in pair]].copy()
    rmu_delta = rmu[[p[0] for p in rmu_pairs]].to_numpy(float) - rmu[[p[1] for p in rmu_pairs]].to_numpy(float)
    rtu_delta = rtu[[p[0] for p in rtu_pairs]].to_numpy(float) - rtu[[p[1] for p in rtu_pairs]].to_numpy(float)
    rows = []
    for a, b in zip(rmu_delta, rtu_delta):
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            rows.append({"interaction_n_rmu": len(a), "interaction_n_rtu": len(b)})
            continue
        rows.append(
            {
                "interaction_n_rmu": len(a),
                "interaction_n_rtu": len(b),
                "interaction_log2fc": float(a.mean() - b.mean()),
                "interaction_p": float(ttest_ind(a, b, equal_var=False).pvalue),
            }
        )
    return pd.DataFrame(rows, index=matrix.index)


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def ion_families(targets: pd.DataFrame, rt_tolerance: float, ppm: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = targets[["feature_id", "mz", "rt_sec"]].dropna().sort_values("rt_sec").reset_index(drop=True)
    ids = work["feature_id"].astype(int).tolist()
    uf = UnionFind(ids)
    edges: list[dict[str, object]] = []
    mz = work["mz"].to_numpy(float)
    rt = work["rt_sec"].to_numpy(float)
    feature = work["feature_id"].to_numpy(int)
    left = 0
    for right in range(len(work)):
        while rt[right] - rt[left] > rt_tolerance:
            left += 1
        for other in range(left, right):
            difference = abs(mz[right] - mz[other])
            for relation, expected in COMMON_MASS_DELTAS.items():
                tolerance = max(0.003, ppm * max(mz[right], mz[other]) / 1e6)
                if abs(difference - expected) <= tolerance:
                    uf.union(int(feature[right]), int(feature[other]))
                    edges.append(
                        {
                            "feature_id_a": int(feature[other]),
                            "feature_id_b": int(feature[right]),
                            "relation": relation,
                            "mz_difference": difference,
                            "rt_difference_sec": float(rt[right] - rt[other]),
                        }
                    )
                    break
    roots = {value: uf.find(value) for value in ids}
    unique_roots = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    mapping = pd.DataFrame(
        {
            "feature_id": ids,
            "ion_family_id": [unique_roots[roots[value]] for value in ids],
        }
    )
    sizes = mapping.groupby("ion_family_id")["feature_id"].transform("size")
    mapping["ion_family_size"] = sizes.astype(int)
    return mapping, pd.DataFrame(edges)


def load_annotations(root: Path, panel: str) -> pd.DataFrame:
    path = root / "full_annotated_feature_audit_v1" / f"{panel}__annotated_feature_audit.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=["feature_id"])
    frame = pd.read_csv(path)
    keep = [
        "feature_id", "best_name", "best_ik14", "best_smiles", "annotation_evidence_tier",
        "max_cosine", "median_cosine", "n_support_spectra", "n_support_samples",
        "structure_agreement_fraction", "n_distinct_ik14_candidates",
    ]
    return frame[[column for column in keep if column in frame]].drop_duplicates("feature_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/mtbls13729"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/full_requantifiable_space_audit_v1"))
    parser.add_argument("--minimum-pairs", type=int, default=8)
    parser.add_argument("--minimum-effect", type=float, default=0.5)
    parser.add_argument("--rt-family-tolerance", type=float, default=8.0)
    parser.add_argument("--ppm", type=float, default=10.0)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "status": "mtbls13729_full_requantifiable_space_audit_complete",
        "formal": False,
        "reason_formal_false": (
            "full-space audit of discovery peak-picker matrices; targeted EIC re-extraction and "
            "independent biological replication are not yet performed"
        ),
        "primary_endpoint": "paired Rmu versus matched RN abundance",
        "secondary_endpoint": "Rmu-RN versus Rtu-RN interaction, not a subtype-specificity claim",
        "panels": {},
    }
    all_candidates = []
    for panel in PANELS:
        consensus = root / "ms1_consensus"
        matrix_path = consensus / f"{panel}__discovery_intensity_matrix.csv.gz"
        target_path = consensus / f"{panel}__requantification_targets.csv.gz"
        matrix = pd.read_csv(matrix_path).set_index("feature_id")
        targets = pd.read_csv(target_path)
        target_ids = targets["feature_id"].astype(int)
        missing = sorted(set(target_ids) - set(matrix.index.astype(int)))
        if missing:
            raise RuntimeError(f"{panel}: {len(missing)} targets absent from discovery matrix")
        matrix = matrix.loc[target_ids].astype(float)
        matrix = matrix.where(matrix > 0.0)
        positives = matrix.stack().to_numpy(float)
        pseudocount = float(np.percentile(positives, 1) / 2.0)
        log_raw = np.log2(matrix + pseudocount)
        prevalence = targets.set_index("feature_id")["global_prevalence"]
        pqn_matrix, pqn_factors = pqn(log_raw, prevalence)
        normalizations = {"log_raw": log_raw, "pqn": pqn_matrix}
        rmu_pairs = sample_pairs(list(matrix.columns), "Rmu", "RN")
        rtu_pairs = sample_pairs(list(matrix.columns), "Rtu", "RN")

        per_normalization = []
        for name, normalized in normalizations.items():
            summary = paired_summary(normalized, rmu_pairs, "rmu_vs_rn")
            interaction = interaction_summary(normalized, rmu_pairs, rtu_pairs)
            summary = summary.join(interaction)
            summary.insert(0, "normalization", name)
            summary.to_csv(output / f"{panel}__{name}__paired_stats.csv.gz")
            per_normalization.append(summary.reset_index())
        combined = pd.concat(per_normalization, ignore_index=True)
        metrics = {}
        for column in [
            "rmu_vs_rn_n", "rmu_vs_rn_mean_log2fc", "rmu_vs_rn_ttest_p",
            "rmu_vs_rn_exact_signflip_p", "rmu_vs_rn_loo_direction_stable",
            "interaction_log2fc", "interaction_p",
        ]:
            metrics[column] = combined.pivot(index="feature_id", columns="normalization", values=column)
        feature = targets.set_index("feature_id").copy()
        feature["min_pairs"] = metrics["rmu_vs_rn_n"].min(axis=1)
        feature["direction_consistent"] = (
            np.sign(metrics["rmu_vs_rn_mean_log2fc"]).nunique(axis=1, dropna=True) == 1
        )
        feature["min_abs_log2fc"] = metrics["rmu_vs_rn_mean_log2fc"].abs().min(axis=1)
        feature["max_exact_p"] = metrics["rmu_vs_rn_exact_signflip_p"].max(axis=1)
        feature["max_ttest_p"] = metrics["rmu_vs_rn_ttest_p"].max(axis=1)
        feature["raw_mean_log2fc"] = metrics["rmu_vs_rn_mean_log2fc"]["log_raw"]
        feature["pqn_mean_log2fc"] = metrics["rmu_vs_rn_mean_log2fc"]["pqn"]
        feature["raw_interaction_log2fc"] = metrics["interaction_log2fc"]["log_raw"]
        feature["pqn_interaction_log2fc"] = metrics["interaction_log2fc"]["pqn"]
        eligible = feature["min_pairs"] >= args.minimum_pairs
        feature["exact_q"] = np.nan
        feature["ttest_q"] = np.nan
        feature.loc[eligible, "exact_q"] = bh_adjust(feature.loc[eligible, "max_exact_p"].to_numpy(float))
        feature.loc[eligible, "ttest_q"] = bh_adjust(feature.loc[eligible, "max_ttest_p"].to_numpy(float))
        feature["effect_and_direction_gate"] = (
            eligible & feature["direction_consistent"] & (feature["min_abs_log2fc"] >= args.minimum_effect)
        )
        feature["nominal_exact_gate"] = feature["effect_and_direction_gate"] & (feature["max_exact_p"] <= 0.05)
        feature["fdr10_exact_gate"] = feature["effect_and_direction_gate"] & (feature["exact_q"] <= 0.10)
        feature["fdr05_exact_gate"] = feature["effect_and_direction_gate"] & (feature["exact_q"] <= 0.05)

        family_map, family_edges = ion_families(targets, args.rt_family_tolerance, args.ppm)
        feature = feature.reset_index().merge(family_map, on="feature_id", how="left", validate="one_to_one")
        annotations = load_annotations(root, panel)
        feature = feature.merge(annotations, on="feature_id", how="left", validate="one_to_one")
        feature.insert(0, "panel", panel)
        feature.to_csv(output / f"{panel}__full_feature_audit.csv.gz", index=False)
        family_edges.insert(0, "panel", panel)
        family_edges.to_csv(output / f"{panel}__ion_family_edges.csv.gz", index=False)

        ranked = feature.sort_values(
            ["fdr10_exact_gate", "exact_q", "max_exact_p", "min_abs_log2fc"],
            ascending=[False, True, True, False],
        )
        ranked.head(250).to_csv(output / f"{panel}__top250_review.csv", index=False)
        selected = feature[feature["nominal_exact_gate"]].copy()
        all_candidates.append(selected)
        family_selected = selected.groupby("ion_family_id", dropna=False).agg(
            member_features=("feature_id", "count"),
            representative_feature=("feature_id", "first"),
            minimum_exact_q=("exact_q", "min"),
            maximum_abs_effect=("min_abs_log2fc", "max"),
            annotated_members=("best_name", lambda x: int(x.notna().sum())),
        ).reset_index()
        family_selected.to_csv(output / f"{panel}__nominal_candidate_families.csv", index=False)
        report["panels"][panel] = {
            "requantifiable_targets": int(len(feature)),
            "rmu_pairs": int(len(rmu_pairs)),
            "rtu_pairs": int(len(rtu_pairs)),
            "pseudocount": pseudocount,
            "effect_and_direction_gate": int(feature["effect_and_direction_gate"].sum()),
            "nominal_exact_gate": int(feature["nominal_exact_gate"].sum()),
            "fdr10_exact_features": int(feature["fdr10_exact_gate"].sum()),
            "fdr05_exact_features": int(feature["fdr05_exact_gate"].sum()),
            "nominal_candidate_ion_families": int(selected["ion_family_id"].nunique()),
            "fdr10_exact_ion_families": int(feature.loc[feature["fdr10_exact_gate"], "ion_family_id"].nunique()),
            "annotated_nominal_candidates": int(selected["best_name"].notna().sum()),
            "pqn_factor_minmax": [float(pqn_factors.min()), float(pqn_factors.max())],
            "provenance": {
                "matrix_sha256": sha256(matrix_path),
                "targets_sha256": sha256(target_path),
            },
        }

    candidates = pd.concat(all_candidates, ignore_index=True)
    candidates.to_csv(output / "all_nominal_candidates.csv", index=False)
    report["total_nominal_candidates"] = int(len(candidates))
    report["total_nominal_candidate_ion_families"] = int(
        candidates.assign(key=candidates["panel"] + ":" + candidates["ion_family_id"].astype(str))["key"].nunique()
    )
    report["claim_limit"] = (
        "This is a paired full-space discovery audit of predefined MS1 targets. FDR is computed on "
        "discovery peak-picker intensities and is not confirmed by targeted EIC re-extraction. Ion-family "
        "relations are coordinate-based redundancy controls, not metabolite identities. Static abundance "
        "does not establish flux, enzyme activity, subtype specificity, or causal mechanism."
    )
    report["script_sha256"] = sha256(Path(__file__).resolve())
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
