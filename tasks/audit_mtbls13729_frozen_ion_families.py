#!/usr/bin/env python
"""Collapse frozen MTBLS13729 biology discoveries to outcome-blind ion families.

The global peak graph is built without phenotype labels.  This script joins
the already frozen abundance audit to that graph and reports descriptive
family-level redundancy.  It deliberately does *not* recompute or relabel
feature-level FDR as family-level FDR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PANELS = ("neg_rp", "pos_rp")
E6_NAME = "e6_fixed_v2_sw2_name"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().isin({"true", "1", "yes"})


def joined_strings(values: pd.Series) -> str:
    clean = sorted({str(value).strip() for value in values.dropna() if str(value).strip()})
    return " | ".join(clean)


def family_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (panel, family_id), group in frame.groupby(["panel", "ion_family_id"], sort=False):
        priority = group[as_bool(group["discovery_priority"])]
        fdr10 = group[as_bool(group["primary_rmu_fdr10_robust"])]
        q = pd.to_numeric(group["max_rmu_q_across_normalizations"], errors="coerce")
        representative_index = q.idxmin() if q.notna().any() else group.index[0]
        representative = group.loc[representative_index]
        names = joined_strings(group[E6_NAME]) if E6_NAME in group else ""
        rows.append({
            "panel": panel,
            "ion_family_id": int(family_id),
            "ion_family_size_in_global_graph": int(group["ion_family_size"].max()),
            "n_frozen_features_in_family": int(len(group)),
            "n_priority_features_in_family": int(len(priority)),
            "n_primary_fdr10_features_in_family": int(len(fdr10)),
            "representative_feature_id": int(representative["feature_id"]),
            "representative_mz": float(representative["mz"]),
            "representative_rt_sec": float(representative["rt_sec"]),
            "minimum_feature_level_q": float(q.min()) if q.notna().any() else np.nan,
            "member_feature_ids": ";".join(str(int(value)) for value in sorted(group["feature_id"])),
            "priority_feature_ids": ";".join(str(int(value)) for value in sorted(priority["feature_id"])),
            "fdr10_feature_ids": ";".join(str(int(value)) for value in sorted(fdr10["feature_id"])),
            "candidate_names": names,
            "candidate_name_conflict": bool(len({name for name in names.split(" | ") if name}) > 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--biology-audit-dir",
        type=Path,
        default=Path("data/mtbls13729/threeway_biology_audit_v2"),
    )
    parser.add_argument(
        "--global-peak-dir",
        type=Path,
        default=Path("data/mtbls13729/bioaware_global_peak_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/frozen_ion_family_audit_v1"),
    )
    parser.add_argument("--allow-nonformal-graph", action="store_true")
    args = parser.parse_args()

    report_path = args.global_peak_dir / "report.json"
    biology_report_path = args.biology_audit_dir / "report.json"
    for path in (report_path, biology_report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    graph_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not graph_report.get("formal", False) and not args.allow_nonformal_graph:
        raise RuntimeError("global peak graph is non-formal; rebuild from uniform re-quantified EIC matrices")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    joined: list[pd.DataFrame] = []
    edge_evidence: list[pd.DataFrame] = []
    panel_reports: dict[str, dict[str, object]] = {}
    for panel in PANELS:
        audited_path = args.biology_audit_dir / f"{panel}__audited_features.csv.gz"
        nodes_path = args.global_peak_dir / f"{panel}__global_peak_nodes.csv.gz"
        edges_path = args.global_peak_dir / f"{panel}__global_peak_edges.csv.gz"
        for path in (audited_path, nodes_path, edges_path):
            if not path.exists():
                raise FileNotFoundError(path)
        audited = pd.read_csv(audited_path)
        nodes = pd.read_csv(nodes_path)
        if nodes["feature_id"].duplicated().any():
            raise RuntimeError(f"{panel}: duplicate feature_id in global graph nodes")
        merged = audited.merge(
            nodes[["feature_id", "ion_family_id", "ion_family_size"]],
            on="feature_id",
            how="left",
            validate="one_to_one",
        )
        if merged["ion_family_id"].isna().any():
            missing = merged.loc[merged["ion_family_id"].isna(), "feature_id"].tolist()
            raise RuntimeError(f"{panel}: frozen features absent from global graph: {missing[:10]}")
        merged["panel"] = panel
        joined.append(merged)

        edges = pd.read_csv(edges_path)
        discoveries = set(
            merged.loc[as_bool(merged["primary_rmu_fdr10_robust"]), "feature_id"].astype(int)
        )
        relevant = edges[
            edges["feature_id_a"].astype(int).isin(discoveries)
            | edges["feature_id_b"].astype(int).isin(discoveries)
        ].copy()
        relevant.insert(0, "panel", panel)
        edge_evidence.append(relevant)

        fdr10 = merged[as_bool(merged["primary_rmu_fdr10_robust"])]
        panel_reports[panel] = {
            "frozen_features": int(len(merged)),
            "priority_features": int(as_bool(merged["discovery_priority"]).sum()),
            "primary_fdr10_features": int(len(fdr10)),
            "primary_fdr10_descriptive_ion_families": int(fdr10["ion_family_id"].nunique()),
            "primary_fdr10_features_in_multimember_global_families": int(
                (fdr10["ion_family_size"] > 1).sum()
            ),
        }

    all_features = pd.concat(joined, ignore_index=True)
    families = family_rows(all_features)
    relevant_edges = pd.concat(edge_evidence, ignore_index=True)
    all_features.to_csv(output / "frozen_features_with_ion_families.csv.gz", index=False)
    families.to_csv(output / "descriptive_ion_family_summary.csv", index=False)
    relevant_edges.to_csv(output / "discovery_related_peak_edges.csv.gz", index=False)

    duplicated_discoveries = families[
        (families["n_primary_fdr10_features_in_family"] > 1)
        | (
            (families["n_primary_fdr10_features_in_family"] > 0)
            & (families["ion_family_size_in_global_graph"] > 1)
        )
    ].copy()
    duplicated_discoveries.to_csv(output / "fdr10_family_redundancy_review.csv", index=False)
    payload = {
        "status": "mtbls13729_frozen_ion_family_audit_complete",
        "formal_global_peak_graph": bool(graph_report.get("formal", False)),
        "panels": panel_reports,
        "fdr10_families_requiring_redundancy_review": int(len(duplicated_discoveries)),
        "candidate_name_conflicts_in_fdr10_families": int(
            duplicated_discoveries["candidate_name_conflict"].sum()
        ) if len(duplicated_discoveries) else 0,
        "provenance": {
            "biology_report_sha256": sha256(biology_report_path),
            "global_peak_report_sha256": sha256(report_path),
        },
        "claim_limit": (
            "Family counts are a descriptive redundancy audit after a feature-level frozen FDR analysis. "
            "They are not recomputed family-level q-values and cannot promote molecular identity."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
