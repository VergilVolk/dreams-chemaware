"""Adversarial normalization and redundancy audit for LCNEC dark EIC signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def pqn(matrix: np.ndarray) -> np.ndarray:
    reference = np.median(matrix, axis=0)
    quotients = np.divide(matrix, reference, out=np.full_like(matrix, np.nan), where=reference > 0)
    factors = np.nanmedian(np.where(quotients > 0, quotients, np.nan), axis=1)
    factors[~np.isfinite(factors) | (factors <= 0)] = 1.0
    return matrix / factors[:, None]


def paired_stats(matrix: np.ndarray, pairs: list[tuple[int, int]], quality: np.ndarray) -> pd.DataFrame:
    rows = []
    quality_indices = np.where(quality)[0]
    p_values = []
    pending = []
    for target in quality_indices:
        tu = np.asarray([matrix[a, target] for a, _ in pairs])
        ng = np.asarray([matrix[b, target] for _, b in pairs])
        positive = np.concatenate((tu[tu > 0], ng[ng > 0]))
        pseudo = float(np.min(positive) / 2) if len(positive) else 1.0
        delta = np.log2(tu + pseudo) - np.log2(ng + pseudo)
        p = float(ttest_rel(np.log2(tu + pseudo), np.log2(ng + pseudo)).pvalue)
        try:
            wp = float(wilcoxon(delta).pvalue)
        except ValueError:
            wp = 1.0
        pending.append((target, float(np.mean(delta)), float(max(np.mean(delta > 0), np.mean(delta < 0))), p, wp))
        p_values.append(p)
    q_values = bh(np.asarray(p_values))
    for values, q in zip(pending, q_values, strict=True):
        target, effect, concordance, p, wp = values
        rows.append({
            "target_index": target,
            "mean_log2fc": effect,
            "direction_concordance": concordance,
            "paired_t_p": p,
            "paired_t_q": float(q),
            "wilcoxon_p": wp,
            "robust": q <= 0.10 and wp <= 0.05 and abs(effect) >= 0.50 and concordance >= 0.65,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("data/validation/lcnec_hsst3n_dark_eic_gate/dark_feature_eic_matrix.npz"))
    parser.add_argument("--results", type=Path, default=Path("data/validation/lcnec_hsst3n_dark_eic_gate/dark_feature_paired_results.csv"))
    parser.add_argument("--overview", type=Path, default=Path("data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/lcnec_hsst3n_dark_robustness_gate"))
    args = parser.parse_args()

    cache = np.load(args.matrix)
    area = cache["area"].astype(float)
    sample_ids = cache["sample_id"].astype(str)
    family_ids = cache["family_id"].astype(int)
    results = pd.read_csv(args.results)
    overview = pd.read_csv(args.overview, sep="\t")
    row_by_sample = {sample: index for index, sample in enumerate(sample_ids)}
    study = overview[overview["NOTE"].eq("Study sample")].copy()
    if len(study) != 68:
        raise RuntimeError("expected 68 study samples")
    study_rows = [row_by_sample[value] for value in study["SAMPLE_ID"]]
    study_area = area[study_rows]
    amount = study["AMOUNT"].astype(float).to_numpy()
    per_mg = study_area / amount[:, None]

    qc = overview[overview["NOTE"].eq("QC sample")].copy()
    qc_rows = [row_by_sample[value] for value in qc["SAMPLE_ID"]]
    qc_injection = qc["INJECTION_ID"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    study_injection = study["INJECTION_ID"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    drift_corrected = per_mg.copy()
    for target in range(per_mg.shape[1]):
        values = area[qc_rows, target]
        median = np.median(values)
        if median <= 0:
            continue
        factor = np.interp(study_injection, qc_injection, values / median)
        factor[factor <= 0] = 1.0
        drift_corrected[:, target] /= factor

    pair_map: dict[str, dict[str, int]] = defaultdict(dict)
    for local_index, (_, row) in enumerate(study.iterrows()):
        pair_map[str(row["SAMPLE_CODE"])][str(row["GROUP_CODE"])] = local_index
    pairs = [(value["TU"], value["NG"]) for value in pair_map.values()]
    quality = results.set_index("family_id").loc[family_ids, "quality_pass"].astype(bool).to_numpy()
    target_meta = results.set_index("family_id").loc[family_ids].reset_index()

    matrices = {
        "raw": study_area,
        "per_mg": per_mg,
        "per_mg_pqn": pqn(per_mg),
        "per_mg_drift_pqn": pqn(drift_corrected),
    }
    stats = {name: paired_stats(matrix, pairs, quality) for name, matrix in matrices.items()}
    robust_sets = {name: set(frame.loc[frame["robust"], "target_index"].astype(int)) for name, frame in stats.items()}
    intersection = set.intersection(*robust_sets.values())
    signs_ok = set()
    for target in intersection:
        effects = [float(frame.loc[frame["target_index"].eq(target), "mean_log2fc"].iloc[0]) for frame in stats.values()]
        if all(value > 0 for value in effects) or all(value < 0 for value in effects):
            signs_ok.add(target)
    intersection = signs_ok

    final_matrix = np.log2(matrices["per_mg_drift_pqn"] + 1.0)
    graph: dict[int, set[int]] = defaultdict(set)
    robust_list = sorted(intersection)
    for position, left in enumerate(robust_list):
        for right in robust_list[position + 1:]:
            rt_delta = abs(float(target_meta.iloc[left]["rt_sec"]) - float(target_meta.iloc[right]["rt_sec"]))
            if rt_delta > 5.0:
                continue
            correlation = float(np.corrcoef(final_matrix[:, left], final_matrix[:, right])[0, 1])
            if math.isfinite(correlation) and correlation >= 0.95:
                graph[left].add(right)
                graph[right].add(left)
    seen: set[int] = set()
    modules: list[list[int]] = []
    for start in robust_list:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        modules.append(component)

    audit_rows = []
    for target in range(len(family_ids)):
        row = {"family_id": int(family_ids[target]), "mz": float(target_meta.iloc[target]["mz"]), "rt_sec": float(target_meta.iloc[target]["rt_sec"]), "quality_pass": bool(quality[target]), "cross_normalization_robust": target in intersection}
        for name, frame in stats.items():
            match = frame[frame["target_index"].eq(target)]
            row[f"{name}_log2fc"] = float(match["mean_log2fc"].iloc[0]) if len(match) else math.nan
            row[f"{name}_q"] = float(match["paired_t_q"].iloc[0]) if len(match) else math.nan
            row[f"{name}_robust"] = bool(match["robust"].iloc[0]) if len(match) else False
        audit_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "normalization_robustness_ledger.csv", index=False)
    module_rows = []
    for module_id, component in enumerate(modules):
        for target in component:
            module_rows.append({"module_id": module_id, "module_size": len(component), "family_id": int(family_ids[target]), "mz": float(target_meta.iloc[target]["mz"]), "rt_sec": float(target_meta.iloc[target]["rt_sec"])})
    pd.DataFrame(module_rows).to_csv(args.output_dir / "nonredundant_module_membership.csv", index=False)

    report = {
        "status": "lcnec_hsst3n_dark_robustness_complete",
        "formal": True,
        "quality_targets": int(quality.sum()),
        "robust_by_normalization": {name: len(values) for name, values in robust_sets.items()},
        "cross_normalization_same_direction_robust": len(intersection),
        "nonredundant_modules_rt5sec_corr095": len(modules),
        "largest_module_size": max((len(value) for value in modules), default=0),
        "gates": {
            "cross_normalization_robust_ge_5": len(intersection) >= 5,
            "nonredundant_modules_ge_5": len(modules) >= 5,
        },
        "pass_to_identity_annotation": len(intersection) >= 5 and len(modules) >= 5,
        "claim_limit": "Normalization-robust nonredundant abundance modules; identities, pathways, and mechanisms remain unassigned.",
    }
    (args.output_dir / "robustness_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
