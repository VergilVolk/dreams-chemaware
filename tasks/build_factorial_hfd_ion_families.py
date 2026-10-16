#!/usr/bin/env python3
"""Conservatively collapse recurrent anchors into phenotype-blind ion families.

Only coeluting anchors separated by a predeclared isotope/adduct mass gap and
showing both strong sample-wise concordance and a stable log-intensity ratio
are joined.  No genotype, diet, effect, identity, or cross-study field is read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_GAPS = {
    "isotope_z1": 1.003354835,
    "isotope_z2": 0.501677418,
    "na_minus_h": 21.981943,
}
POSITIVE_GAPS = {
    "nh4_minus_h": 17.026549,
    "k_minus_h": 37.955882,
}
NEGATIVE_GAPS = {
    "chloride_minus_h": 35.976678,
    "formate_minus_h": 46.005479,
    "acetate_minus_h": 60.021129,
}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def infer_polarity(assay_file: str) -> str:
    upper = assay_file.upper()
    positive = "POS" in upper
    negative = "NEG" in upper
    if positive == negative:
        raise ValueError(f"cannot infer unique polarity from assay filename: {assay_file}")
    return "positive" if positive else "negative"


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    lrank = pd.Series(left).rank(method="average").to_numpy(float)
    rrank = pd.Series(right).rank(method="average").to_numpy(float)
    if np.std(lrank) == 0 or np.std(rrank) == 0:
        return float("nan")
    return float(np.corrcoef(lrank, rrank)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument("--panel-registry", required=True, type=Path)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rt-seconds", type=float, default=2.0)
    parser.add_argument("--mass-ppm", type=float, default=10.0)
    parser.add_argument("--mass-absolute-da", type=float, default=0.005)
    parser.add_argument("--minimum-joint-detections", type=int, default=8)
    parser.add_argument("--minimum-spearman", type=float, default=0.95)
    parser.add_argument("--maximum-log2-ratio-mad", type=float, default=0.35)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    report = json.loads((args.panel_object / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "development_consumed_abundance_matrix_frozen":
        raise RuntimeError("panel abundance matrix is not frozen")
    anchors = pd.read_csv(args.panel_object / "anchor_manifest.tsv", sep="\t")
    with np.load(args.panel_object / "ms1_peak_height.npz", allow_pickle=False) as packed:
        values = np.asarray(packed["values"], dtype=float)
        anchor_ids = np.asarray(packed["anchor_id"], dtype=str)
    if list(anchors.anchor_id.astype(str)) != list(anchor_ids) or values.shape[0] != len(anchors):
        raise RuntimeError("anchor manifest and abundance matrix do not align")
    registry = pd.read_csv(args.panel_registry, sep="\t", dtype=str).set_index("panel_id", verify_integrity=True)
    if args.panel_id not in registry.index:
        raise ValueError(f"panel {args.panel_id} absent from registry")
    polarity = infer_polarity(str(registry.loc[args.panel_id, "assay_file"]))
    gaps = dict(COMMON_GAPS)
    gaps.update(POSITIVE_GAPS if polarity == "positive" else NEGATIVE_GAPS)

    mz = anchors.precursor_mz.to_numpy(float)
    rt = anchors.rt_seconds.to_numpy(float)
    order = np.argsort(mz)
    sorted_mz = mz[order]
    uf = UnionFind(len(anchors))
    edge_rows: list[dict[str, object]] = []
    for left in range(len(anchors)):
        for relation, gap in gaps.items():
            target = mz[left] + gap
            tolerance = max(args.mass_absolute_da, target * args.mass_ppm * 1e-6)
            lo = int(np.searchsorted(sorted_mz, target - tolerance, side="left"))
            hi = int(np.searchsorted(sorted_mz, target + tolerance, side="right"))
            for right in order[lo:hi]:
                right = int(right)
                if right == left or abs(rt[left] - rt[right]) > args.rt_seconds:
                    continue
                joint = (values[left] > 0) & (values[right] > 0)
                if int(joint.sum()) < args.minimum_joint_detections:
                    continue
                log_left = np.log2(values[left, joint])
                log_right = np.log2(values[right, joint])
                correlation = rank_correlation(log_left, log_right)
                ratio = log_left - log_right
                ratio_mad = float(np.median(np.abs(ratio - np.median(ratio))))
                if not math.isfinite(correlation) or correlation < args.minimum_spearman:
                    continue
                if ratio_mad > args.maximum_log2_ratio_mad:
                    continue
                uf.union(left, right)
                edge_rows.append({
                    "anchor_id_a": str(anchor_ids[left]), "anchor_id_b": str(anchor_ids[right]),
                    "relation": relation, "mass_error_da": float(abs((mz[right] - mz[left]) - gap)),
                    "rt_error_seconds": float(abs(rt[left] - rt[right])),
                    "joint_detections": int(joint.sum()), "spearman": correlation,
                    "log2_ratio_mad": ratio_mad,
                })

    roots = np.asarray([uf.find(index) for index in range(len(anchors))])
    unique_roots = {root: index + 1 for index, root in enumerate(sorted(set(roots)))}
    family_id = np.asarray([f"F{unique_roots[root]:07d}" for root in roots])
    prevalence = (values > 0).mean(axis=1)
    members = anchors[["anchor_id", "precursor_mz", "rt_seconds"]].copy()
    members.insert(0, "panel_id", args.panel_id)
    members["family_id"] = family_id
    members["detection_prevalence"] = prevalence
    members["representative"] = False
    for _, block in members.groupby("family_id", sort=True):
        candidates = block.index.to_numpy(int)
        tic = anchors.loc[candidates, "representative_tic"].to_numpy(float)
        best_order = np.lexsort((-tic, -prevalence[candidates]))
        members.loc[candidates[best_order[0]], "representative"] = True
    members["family_size"] = members.family_id.map(members.family_id.value_counts()).astype(int)

    args.output_dir.mkdir(parents=True)
    members.to_csv(args.output_dir / "ion_family_membership.tsv", sep="\t", index=False)
    pd.DataFrame(edge_rows).to_csv(args.output_dir / "admitted_family_edges.tsv", sep="\t", index=False)
    family_sizes = members.groupby("family_id").size()
    summary = {
        "schema": "factorial_hfd_ion_families_v1",
        "status": "phenotype_blind_conservative_ion_families_frozen",
        "panel_id": args.panel_id,
        "polarity": polarity,
        "denominators": {
            "anchors": len(members), "families": int(members.family_id.nunique()),
            "multi_anchor_families": int((family_sizes > 1).sum()),
            "admitted_edges": len(edge_rows),
        },
        "criteria": {
            "allowed_mass_gaps_da": gaps, "maximum_rt_seconds": args.rt_seconds,
            "mass_ppm": args.mass_ppm, "mass_absolute_da": args.mass_absolute_da,
            "minimum_joint_detections": args.minimum_joint_detections,
            "minimum_spearman": args.minimum_spearman,
            "maximum_log2_ratio_mad": args.maximum_log2_ratio_mad,
        },
        "representative_selection": "highest phenotype-blind detection prevalence; representative MS2 TIC tie-break",
        "leakage_status": "no genotype_diet_effect_identity_or_cross_study_truth_read",
        "claim_boundary": [
            "Families are conservative ion-redundancy groups, not confirmed metabolite identities.",
            "Unmerged anchors are not proven distinct metabolites.",
            "Anchor-level results remain a sensitivity analysis; family representatives define the primary entity universe.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
