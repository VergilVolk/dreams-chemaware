#!/usr/bin/env python
"""ChemProp endpoint-to-time-series cross-design response retrieval.

The endpoint studies and the time-series use independent feature tables. This
script compares precursor-mass matching with mass-constrained raw MS/MS cosine
and WSE. DreaMS/Noise/P2b require checkpoint scoring on the same graph and
remain a separate, explicitly pending comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_chemprop2_response_reproducibility_m1 import (
    cluster_bootstrap_delta,
    median_center,
    parse_samples,
    robust_log2,
    summarize,
)


TIME_TO_CANONICAL = {
    "cilni": "Cilnidipine", "clomi": "Clomifen", "eryth": "Erythromycin",
    "ketoc": "Ketoconazole", "lanso": "Lansoprazole", "lorat": "Loratadine",
    "metro": "Metronidazole", "monte": "Montelukast", "omepr": "Omeprazole",
    "sertr": "Sertraline", "simva": "Simvastatin", "telmi": "Telmisartan",
}
BATCH_DRUGS = {
    "23drug": ["Cilnidipine", "Clomifen", "Erythromycin"],
    "45drug": [
        "Ketoconazole", "Lansoprazole", "Loratadine", "Metronidazole", "Montelukast",
        "Omeprazole", "Sertraline", "Simvastatin", "Telmisartan",
    ],
}
HIGH_SIGNAL_DRUGS = {
    "Cilnidipine", "Clomifen", "Erythromycin", "Ketoconazole", "Lansoprazole",
    "Loratadine", "Metronidazole", "Omeprazole", "Sertraline",
}
LOW_SIGNAL_DRUGS = {"Montelukast", "Simvastatin", "Telmisartan"}


def quant_column_map(quant: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in quant.columns:
        if column.endswith(" Peak area"):
            result[column[:-10]] = column
    return result


def get_column(mapping: dict[str, str], filename: str) -> str:
    if filename not in mapping:
        raise RuntimeError(f"quantification column missing for {filename}")
    return mapping[filename]


def endpoint_profiles(root: Path, batch: str) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    quant_path = next((root / "quantification_table_reformatted").glob("*.csv"))
    quant = pd.read_csv(quant_path)
    metadata = pd.read_csv(root / "qiime2_output/qiime2_metadata.tsv", sep="\t")
    metadata["filename"] = metadata["filename"].astype(str).str.strip()
    mapping = quant_column_map(quant)
    study_metadata = metadata.loc[metadata["filename"].isin(mapping)].copy()
    study_columns = [get_column(mapping, filename) for filename in study_metadata["filename"]]
    values = quant[study_columns].to_numpy(float).T
    log_values = median_center(robust_log2(values))
    index = {filename: idx for idx, filename in enumerate(study_metadata["filename"])}
    blank_names = study_metadata.loc[
        study_metadata.astype(str).apply(lambda col: col.str.contains("Blank", case=False)).any(axis=1),
        "filename",
    ].tolist()
    blank = log_values[[index[name] for name in blank_names]] if blank_names else np.empty((0, len(quant)))
    raw_study = quant[study_columns].to_numpy(float).T
    blank_raw = quant[[get_column(mapping, name) for name in blank_names]].to_numpy(float).T if blank_names else np.empty((0, len(quant)))
    study_q95 = np.quantile(raw_study, 0.95, axis=0)
    blank_q95 = np.quantile(blank_raw, 0.95, axis=0) if len(blank_raw) else np.zeros(len(quant))
    qc_mask = (study_q95 > 0) & (study_q95 >= 3.0 * np.maximum(blank_q95, 1.0))

    profiles: list[np.ndarray] = []
    drugs = BATCH_DRUGS[batch]
    if batch == "23drug":
        for drug in drugs:
            reps: list[np.ndarray] = []
            for replicate in (1.0, 2.0):
                selected: dict[tuple[str, float], str] = {}
                for row in study_metadata.loc[
                    study_metadata["ATTRIBUTE_Drug"].eq(drug)
                    & study_metadata["ATTRIBUTE_Replicate"].eq(replicate)
                ].itertuples(index=False):
                    selected[(row.ATTRIBUTE_Bacteria, float(row.ATTRIBUTE_timepoint_hours))] = row.filename
                keys = [("With_Bacteria", 0.0), ("With_Bacteria", 2.0), ("No_Bacteria", 0.0), ("No_Bacteria", 2.0)]
                if not all(key in selected for key in keys):
                    raise RuntimeError(f"incomplete 23drug cell: {drug} rep {replicate}")
                bac0, bac2, no0, no2 = [log_values[index[selected[key]]] for key in keys]
                reps.append((bac2 - no2) - (bac0 - no0))
            profiles.append(np.mean(reps, axis=0))
    elif batch == "45drug":
        for drug in drugs:
            rows = study_metadata.loc[study_metadata["ATTRIBUTE_drug"].eq(drug)]
            controls = {
                row.ATTRIBUTE_timepoint: row.filename
                for row in rows.loc[rows["ATTRIBUTE_Sample_DrugType"].eq("Control_Drug")].itertuples(index=False)
            }
            if set(controls) != {"t0", "t2"}:
                raise RuntimeError(f"incomplete 45drug negative control: {drug}")
            reps: list[np.ndarray] = []
            for replicate in ("1", "2", "3"):
                sample = {
                    row.ATTRIBUTE_timepoint: row.filename
                    for row in rows.loc[
                        rows["ATTRIBUTE_Sample_DrugType"].eq("Sample_Drug")
                        & rows["ATTRIBUTE_replicates"].astype(str).eq(replicate)
                    ].itertuples(index=False)
                }
                if set(sample) != {"t0", "t2"}:
                    raise RuntimeError(f"incomplete 45drug cell: {drug} rep {replicate}")
                reps.append(
                    (log_values[index[sample["t2"]]] - log_values[index[controls["t2"]]])
                    - (log_values[index[sample["t0"]]] - log_values[index[controls["t0"]]])
                )
            profiles.append(np.mean(reps, axis=0))
    else:
        raise RuntimeError(batch)
    return quant, np.stack(profiles), drugs, qc_mask


def time_endpoint_profiles(quant: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    samples = parse_samples(list(quant.columns))
    mapping = {row.column: idx for idx, row in samples.iterrows()}
    values = quant[samples["column"].tolist()].to_numpy(float).T
    log_values = median_center(robust_log2(values))
    keyed = {
        (row.drug, int(row.time), bool(row.bacteria), int(row.replicate)): mapping[row.column]
        for row in samples.itertuples(index=False)
    }
    rows: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    for code, canonical in TIME_TO_CANONICAL.items():
        for replicate in (1, 2, 3):
            keys = [(code, time, bac, replicate) for time in (0, 2) for bac in (False, True)]
            if not all(key in keyed for key in keys):
                raise RuntimeError(f"incomplete time-series endpoint cell: {code} rep {replicate}")
            bac0 = log_values[keyed[(code, 0, True, replicate)]]
            no0 = log_values[keyed[(code, 0, False, replicate)]]
            bac2 = log_values[keyed[(code, 2, True, replicate)]]
            no2 = log_values[keyed[(code, 2, False, replicate)]]
            rows.append((bac2 - no2) - (bac0 - no0))
            meta.append({"drug": canonical, "replicate": replicate})
    return np.stack(rows), pd.DataFrame(meta)


def robust_feature_scale(profiles: np.ndarray) -> np.ndarray:
    scale = np.nanmedian(np.abs(profiles - np.nanmedian(profiles, axis=0)), axis=0)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def score_retrieval(
    query_profiles: np.ndarray,
    query_meta: pd.DataFrame,
    reference_profiles: np.ndarray,
    reference_drugs: list[str],
    pairs: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    selected = pairs.loc[pairs["selected"]].copy()
    if len(selected) < 10:
        raise RuntimeError(f"too few pairs for {method}: {len(selected)}")
    q_index = selected["query_feature_index"].to_numpy(int)
    r_index = selected["reference_feature_index"].to_numpy(int)
    q_scale = robust_feature_scale(query_profiles)
    r_scale = robust_feature_scale(reference_profiles)
    q = np.clip(query_profiles[:, q_index] / q_scale[q_index], -10, 10)
    r = np.clip(reference_profiles[:, r_index] / r_scale[r_index], -10, 10)
    rows: list[dict[str, object]] = []
    for query_idx, item in query_meta.iterrows():
        if item.drug not in reference_drugs:
            continue
        scores = []
        for candidate_idx in range(len(reference_drugs)):
            denominator = np.linalg.norm(q[query_idx]) * np.linalg.norm(r[candidate_idx])
            scores.append(float(q[query_idx] @ r[candidate_idx] / denominator) if denominator > 0 else -1.0)
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        truth = reference_drugs.index(str(item.drug))
        rank = int(np.flatnonzero(order == truth)[0] + 1)
        rows.append({
            "method": method, "batch": str(selected["batch"].iloc[0]),
            "drug": item.drug, "replicate": int(item.replicate),
            "prespecified_signal_group": "high" if item.drug in HIGH_SIGNAL_DRUGS else "low",
            "features": len(selected), "candidates": len(reference_drugs),
            "rank": rank, "recall1": rank == 1, "reciprocal_rank": 1.0 / rank,
            "truth_similarity": scores[truth],
            "top_wrong_similarity": float(np.max(np.delete(scores, truth))),
            "margin": float(scores[truth] - np.max(np.delete(scores, truth))),
        })
    return pd.DataFrame(rows)


def build_pairs(
    matches: pd.DataFrame,
    edges: pd.DataFrame,
    query_quant: pd.DataFrame,
    reference_quant: pd.DataFrame,
    query_pass_ids: set[int],
    reference_qc: np.ndarray,
    batch: str,
) -> dict[str, pd.DataFrame]:
    reference_ids = reference_quant["row ID"].astype(int).to_numpy()
    reference_lookup = {row_id: idx for idx, row_id in enumerate(reference_ids)}
    reference_mass = reference_quant["row m/z"].to_numpy(float)
    reference_rt = reference_quant["row retention time"].to_numpy(float)
    reference_has_ms2 = set(edges["reference_id"].astype(int))
    reference_order = np.argsort(reference_mass)
    sorted_mass = reference_mass[reference_order]
    rows_mass: list[dict[str, object]] = []
    rows_mass_rt: list[dict[str, object]] = []
    query_lookup = {int(row_id): idx for idx, row_id in enumerate(query_quant["row ID"].astype(int))}
    for row in query_quant.itertuples(index=False):
        query_id = int(getattr(row, "_0")); query_mz = float(getattr(row, "_1")); query_rt = float(getattr(row, "_2"))
        if query_id not in query_pass_ids:
            continue
        lo = np.searchsorted(sorted_mass, query_mz * (1 - 10e-6), side="left")
        hi = np.searchsorted(sorted_mass, query_mz * (1 + 10e-6), side="right")
        candidates = [
            idx for idx in reference_order[lo:hi]
            if reference_qc[idx] and int(reference_ids[idx]) in reference_has_ms2
        ]
        if not candidates:
            continue
        ref_idx = min(candidates, key=lambda idx: (abs(reference_mass[idx] - query_mz), int(reference_ids[idx])))
        rows_mass.append({
            "batch": batch, "query_id": query_id, "reference_id": int(reference_ids[ref_idx]),
            "query_feature_index": query_lookup[query_id], "reference_feature_index": int(ref_idx),
            "selected": True,
        })
        rt_candidates = [idx for idx in candidates if abs(reference_rt[idx] - query_rt) <= 0.5]
        if rt_candidates:
            ref_rt_idx = min(
                rt_candidates,
                key=lambda idx: (abs(reference_mass[idx] - query_mz), abs(reference_rt[idx] - query_rt), int(reference_ids[idx])),
            )
            rows_mass_rt.append({
                "batch": batch, "query_id": query_id, "reference_id": int(reference_ids[ref_rt_idx]),
                "query_feature_index": query_lookup[query_id], "reference_feature_index": int(ref_rt_idx),
                "selected": True,
            })
    mass = pd.DataFrame(rows_mass)
    mass_rt = pd.DataFrame(rows_mass_rt)
    eligible_reference_ids = set(reference_ids[reference_qc])
    eligible_edges = edges.loc[
        edges["query_id"].isin(query_pass_ids)
        & edges["reference_id"].isin(eligible_reference_ids)
    ].copy()
    eligible_edges["query_feature_index"] = eligible_edges["query_id"].map(query_lookup)
    eligible_edges["reference_feature_index"] = eligible_edges["reference_id"].map(reference_lookup)

    def top_by(score: str) -> pd.DataFrame:
        return eligible_edges.sort_values(
            ["query_id", score, "absolute_mass_error_da", "reference_id"],
            ascending=[True, False, True, True],
        ).drop_duplicates("query_id").copy()

    spectral = top_by("cosine")
    spectral["batch"] = batch
    spectral["top_cosine"] = spectral["cosine"]
    reverse = eligible_edges.sort_values(
        ["reference_id", "cosine", "absolute_mass_error_da", "query_id"],
        ascending=[True, False, True, True],
    ).drop_duplicates("reference_id")
    reverse_lookup = dict(zip(reverse["reference_id"], reverse["query_id"]))
    spectral["mutual_top1"] = [
        reverse_lookup.get(reference_id) == query_id
        for query_id, reference_id in zip(spectral["query_id"], spectral["reference_id"])
    ]
    methods = {
        "mass_nearest_10ppm": mass,
        "mass_rt_nearest_10ppm_0.5min": mass_rt,
    }
    for threshold in (0.0, 0.7, 0.8):
        item = spectral.copy()
        item["selected"] = item["top_cosine"].ge(threshold)
        methods[f"mass_cosine_top1_{threshold:.1f}"] = item
    mutual = spectral.copy()
    mutual["selected"] = mutual["top_cosine"].ge(0.7) & mutual["mutual_top1"]
    methods["mass_cosine_mutual_0.7"] = mutual
    wse = top_by("wse")
    wse["batch"] = batch
    wse["selected"] = True
    methods["mass_wse_top1"] = wse
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-root", type=Path, default=Path("data/reference/chemprop2_m1_20261011"))
    parser.add_argument("--endpoint-root", type=Path, default=Path("data/reference/chemprop_cross_design_20261011"))
    parser.add_argument("--overlap-dir", type=Path, default=Path("data/validation/chemprop_cross_design_overlap_audit_v5_20261011"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/chemprop_cross_design_response_v4_20261011"))
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")

    query_quant = pd.read_csv(args.time_root / "quantification.csv")
    query_profiles, query_meta = time_endpoint_profiles(query_quant)
    query_audit = pd.read_csv("data/validation/chemprop2_response_m1_v2_20261011/feature_qc_audit.csv.gz")
    query_pass_ids = set(query_audit.loc[query_audit["analysis_pass"], "row_id"].astype(int))
    matches = pd.read_csv(args.overlap_dir / "top_matches.csv.gz")
    edges = pd.read_csv(args.overlap_dir / "candidate_edges_10ppm.csv.gz")
    all_results: list[pd.DataFrame] = []
    pair_manifests: list[pd.DataFrame] = []
    negative_controls: dict[str, object] = {}
    rng = np.random.default_rng(args.seed)
    for batch in ("23drug", "45drug"):
        root = args.endpoint_root / f"chemprop_{batch}"
        reference_quant, reference_profiles, reference_drugs, reference_qc = endpoint_profiles(root, batch)
        pair_label = f"time_to_{batch}"
        methods = build_pairs(
            matches.loc[matches["pair"].eq(pair_label)],
            edges.loc[edges["pair"].eq(pair_label)], query_quant, reference_quant,
            query_pass_ids, reference_qc, batch,
        )
        for method, pairs in methods.items():
            pairs = pairs.copy(); pairs.insert(0, "method", method); pair_manifests.append(pairs)
            observed = score_retrieval(query_profiles, query_meta, reference_profiles, reference_drugs, pairs, method)
            all_results.append(observed)
            shuffled_alignment: list[tuple[float, float]] = []
            permuted_labels: list[tuple[float, float]] = []
            selected_index = pairs.index[pairs["selected"]].to_numpy()
            for _ in range(200):
                shuffled = pairs.copy()
                shuffled.loc[selected_index, "reference_feature_index"] = rng.permutation(
                    shuffled.loc[selected_index, "reference_feature_index"].to_numpy()
                )
                null_alignment = score_retrieval(
                    query_profiles, query_meta, reference_profiles, reference_drugs, shuffled, method,
                )
                shuffled_alignment.append((null_alignment["recall1"].mean(), null_alignment["reciprocal_rank"].mean()))
                shuffled_drugs = list(rng.permutation(reference_drugs))
                null_labels = score_retrieval(
                    query_profiles, query_meta, reference_profiles, shuffled_drugs, pairs, method,
                )
                permuted_labels.append((null_labels["recall1"].mean(), null_labels["reciprocal_rank"].mean()))
            def null_summary(values: list[tuple[float, float]]) -> dict[str, object]:
                array = np.asarray(values)
                return {
                    "draws": len(array),
                    "recall1_mean": float(array[:, 0].mean()),
                    "recall1_q025": float(np.quantile(array[:, 0], 0.025)),
                    "recall1_q975": float(np.quantile(array[:, 0], 0.975)),
                    "mrr_mean": float(array[:, 1].mean()),
                    "mrr_q025": float(np.quantile(array[:, 1], 0.025)),
                    "mrr_q975": float(np.quantile(array[:, 1], 0.975)),
                }
            negative_controls[f"{method}|{batch}"] = {
                "independent_feature_alignment_permutation": null_summary(shuffled_alignment),
                "reference_drug_label_permutation": null_summary(permuted_labels),
            }
    results = pd.concat(all_results, ignore_index=True)
    summaries: dict[str, dict[str, object]] = {}
    for (method, batch), frame in results.groupby(["method", "batch"]):
        summaries[f"{method}|{batch}"] = {
            **summarize(frame),
            "features": int(frame["features"].iloc[0]),
            "candidates": int(frame["candidates"].iloc[0]),
            "chance_recall1": 1.0 / int(frame["candidates"].iloc[0]),
        }
    for method, frame in results.groupby("method"):
        summaries[f"{method}|pooled"] = {
            **summarize(frame), "features": "batch_specific", "candidates": "3_or_9",
            "stratified_chance_recall1": float(np.mean(1.0 / frame["candidates"])),
        }
        for signal_group, subset in frame.groupby("prespecified_signal_group"):
            summaries[f"{method}|signal_{signal_group}"] = {
                **summarize(subset), "features": "batch_specific", "candidates": "3_or_9",
                "drugs": int(subset["drug"].nunique()),
                "selection_definition": (
                    "paper-prespecified ChemProp1 score >= 1" if signal_group == "high"
                    else "paper-prespecified low-signal challenge"
                ),
            }

    baseline = results.loc[results["method"].eq("mass_nearest_10ppm")]
    deltas: dict[str, object] = {}
    for method in sorted(set(results["method"]) - {"mass_nearest_10ppm"}):
        candidate = results.loc[results["method"].eq(method)]
        deltas[method] = {
            "recall1": cluster_bootstrap_delta(candidate, baseline, "recall1", args.seed + len(deltas)),
            "mrr": cluster_bootstrap_delta(candidate, baseline, "reciprocal_rank", args.seed + 100 + len(deltas)),
        }

    args.output_dir.mkdir(parents=True)
    results.to_csv(args.output_dir / "retrieval_results.csv", index=False)
    pd.concat(pair_manifests, ignore_index=True).to_csv(
        args.output_dir / "feature_pair_manifests.csv.gz", index=False, compression="gzip",
    )
    report = {
        "status": "chemprop_cross_design_response_v4_complete",
        "candidate_graph_contract": "all methods rank within the same 10-ppm, endpoint-QC-passing, nonempty-MS/MS reference candidate graph",
        "formal": False,
        "evidence_role": "development_consumed",
        "datasets": {
            "query": "MSV000094899 12-drug time-series T0/T2 endpoint, 3 replicate queries per drug",
            "reference_23drug": "MSV000096724 endpoint, 3 shared drugs, 2 biological replicates",
            "reference_45drug": "MSV000093571 endpoint, 9 shared drugs, 3 bacterial replicates and one abiotic control",
        },
        "estimand": "(Bac T2 - noBac T2) - (Bac T0 - noBac T0), no vehicle subtraction",
        "natural_feature_id_shared": False,
        "retrieval_denominator": "36 queries: 9 queries in a 3-way batch and 27 queries in a 9-way batch",
        "prespecified_signal_groups": {
            "high": sorted(HIGH_SIGNAL_DRUGS),
            "low": sorted(LOW_SIGNAL_DRUGS),
            "source": "ChemProp2 paper selection rule, fixed before this result was run",
        },
        "summaries": summaries,
        "negative_controls": negative_controls,
        "spectral_minus_mass_nearest_cluster_bootstrap_by_drug": deltas,
        "does_not_establish": [
            "cross-laboratory or cross-instrument transfer",
            "unknown chemical identity",
            "positive increment from DreaMS, Noise, P2b, or ChemAware",
            "general disease or unlabeled-sample utility",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
