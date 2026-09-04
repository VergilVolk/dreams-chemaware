#!/usr/bin/env python
"""Build the outcome-blind MTBLS13729 global MS1 peak network.

Formal mode requires uniformly re-quantified EIC matrices.  Discovery matrices
may be used only with ``--allow-discovery-matrix`` and are marked non-formal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from annotation.bioaware_global_peak import (
        UnionFind,
        mass_relation_pairs,
        normalize_intensity_matrix,
        pair_evidence,
        panel_relations,
    )
except ModuleNotFoundError as exc:
    # Cluster deployments commonly sync ``tasks/`` without newly added package
    # modules.  Keep a self-contained, byte-for-byte-equivalent fallback so a
    # missing optional source module cannot waste a batch allocation.
    if exc.name not in {"annotation", "annotation.bioaware_global_peak"}:
        raise

    EPS = 1e-12

    @dataclass(frozen=True)
    class MassRelation:
        name: str
        delta_mz: float
        family: str
        merge_ion_family: bool

    COMMON_LOSSES = (
        ("loss_H2O", 18.010565),
        ("loss_NH3", 17.026549),
        ("loss_CO", 27.994915),
        ("loss_CO2", 43.989829),
        ("loss_HCOOH", 46.005480),
        ("loss_H3PO4", 97.976896),
        ("loss_hexose", 162.052824),
    )

    def panel_relations(panel: str) -> tuple[MassRelation, ...]:
        shared = [
            MassRelation("isotope_z1", 1.003355, "isotope", True),
            MassRelation("isotope_z2", 0.501677, "isotope", True),
            *(MassRelation(name, mass, "in_source_loss", False) for name, mass in COMMON_LOSSES),
        ]
        if panel.startswith("pos"):
            adducts = (
                MassRelation("adduct_NH4_vs_H", 17.026549, "adduct", True),
                MassRelation("adduct_Na_vs_H", 21.981942, "adduct", True),
                MassRelation("adduct_K_vs_H", 37.955882, "adduct", True),
            )
        elif panel.startswith("neg"):
            adducts = (
                MassRelation("adduct_Na2H_vs_H", 21.981942, "adduct", True),
                MassRelation("adduct_Cl_vs_H", 35.976678, "adduct", True),
                MassRelation("adduct_formate_vs_H", 46.005477, "adduct", True),
                MassRelation("adduct_acetate_vs_H", 60.021127, "adduct", True),
            )
        else:
            raise ValueError(f"panel must begin with pos or neg, observed {panel!r}")
        return tuple(shared) + adducts

    def normalize_intensity_matrix(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float).copy()
        matrix[~np.isfinite(matrix) | (matrix < 0)] = 0.0
        medians = np.nanmedian(np.where(matrix > 0, matrix, np.nan), axis=0)
        if np.any(~np.isfinite(medians) | (medians <= 0)):
            bad = np.flatnonzero(~np.isfinite(medians) | (medians <= 0)).tolist()
            raise ValueError(f"samples without positive intensity: {bad}")
        return np.log1p(matrix / medians[None, :])

    def pair_evidence(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError("pair evidence expects equal one-dimensional vectors")

        def pearson(left: np.ndarray, right: np.ndarray) -> float:
            if left.size < 3 or np.std(left) <= EPS or np.std(right) <= EPS:
                return np.nan
            return float(np.corrcoef(left, right)[0, 1])

        joint = (x > 0) & (y > 0)
        union = (x > 0) | (y > 0)
        even = np.arange(x.size) % 2 == 0
        return {
            "n_samples": int(x.size),
            "n_joint_detected": int(joint.sum()),
            "co_detection_jaccard": float(joint.sum() / union.sum()) if union.any() else np.nan,
            "abundance_pearson": pearson(x, y),
            "abundance_spearman": pearson(
                rankdata(x, method="average"), rankdata(y, method="average")
            ),
            "pearson_half_even": pearson(x[even], y[even]),
            "pearson_half_odd": pearson(x[~even], y[~even]),
        }

    def mass_relation_pairs(
        mz: np.ndarray,
        rt_sec: np.ndarray,
        relations: tuple[MassRelation, ...],
        ppm: float,
        absolute_floor_da: float,
        rt_tolerance_sec: float,
    ) -> list[tuple[int, int, MassRelation, float, float]]:
        masses = np.asarray(mz, dtype=float)
        retention = np.asarray(rt_sec, dtype=float)
        if masses.shape != retention.shape or masses.ndim != 1:
            raise ValueError("m/z and RT must be equal one-dimensional arrays")
        order = np.argsort(masses, kind="mergesort")
        sorted_mz = masses[order]
        rows: list[tuple[int, int, MassRelation, float, float]] = []
        for left in range(len(masses)):
            for relation in relations:
                target = masses[left] + relation.delta_mz
                tolerance = max(absolute_floor_da, target * ppm * 1e-6)
                lo = int(np.searchsorted(sorted_mz, target - tolerance, side="left"))
                hi = int(np.searchsorted(sorted_mz, target + tolerance, side="right"))
                for position in range(lo, hi):
                    right = int(order[position])
                    if right == left:
                        continue
                    drt = abs(float(retention[right] - retention[left]))
                    if drt <= rt_tolerance_sec:
                        rows.append((
                            left,
                            right,
                            relation,
                            float((masses[right] - masses[left]) - relation.delta_mz),
                            drt,
                        ))
        return rows

    class UnionFind:
        def __init__(self, n: int):
            self.parent = list(range(n))

        def find(self, item: int) -> int:
            while self.parent[item] != item:
                self.parent[item] = self.parent[self.parent[item]]
                item = self.parent[item]
            return item

        def union(self, left: int, right: int) -> None:
            a, b = self.find(left), self.find(right)
            if a != b:
                self.parent[max(a, b)] = min(a, b)

        def labels(self) -> np.ndarray:
            roots = [self.find(index) for index in range(len(self.parent))]
            mapping = {root: label for label, root in enumerate(sorted(set(roots)))}
            return np.asarray([mapping[root] for root in roots], dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve_matrix(args: argparse.Namespace, panel: str) -> tuple[Path, Path | None, str]:
    formal = args.requant_dir / f"{panel}__eic_auc_matrix.csv.gz"
    if formal.exists():
        detection = args.requant_dir / f"{panel}__eic_detection_matrix.csv.gz"
        if not detection.exists():
            raise RuntimeError(f"{panel}: formal EIC matrix lacks detection mask {detection}")
        report_path = args.requant_dir / "report.json"
        if not report_path.exists():
            raise RuntimeError(f"{panel}: formal matrix lacks requantification report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        panel_report = report.get("panels", {}).get(panel, {})
        if not panel_report.get("resolve_local_peaks", False):
            raise RuntimeError(f"{panel}: formal matrix was not resolved to local chromatographic peaks")
        if not panel_report.get("all_samples_parameter_locked", False):
            raise RuntimeError(f"{panel}: per-sample EIC parameter provenance is incomplete")
        return formal, detection, "uniform_requantified_eic"
    discovery = args.consensus_dir / f"{panel}__discovery_intensity_matrix.csv.gz"
    if args.allow_discovery_matrix and discovery.exists():
        return discovery, None, "discovery_missingness_pilot"
    raise FileNotFoundError(
        f"{panel}: missing formal EIC matrix {formal}. Run targeted EIC "
        "requantification first, or explicitly pass --allow-discovery-matrix for a non-formal pilot."
    )


def build_panel(args: argparse.Namespace, panel: str, output: Path) -> dict:
    target_path = args.consensus_dir / f"{panel}__requantification_targets.csv.gz"
    sample_path = args.consensus_dir / f"{panel}__samples.csv"
    matrix_path, detection_path, matrix_kind = resolve_matrix(args, panel)
    for path in (target_path, sample_path, matrix_path):
        if not path.exists():
            raise FileNotFoundError(path)
    targets = pd.read_csv(target_path).drop_duplicates("feature_id")
    matrix = pd.read_csv(matrix_path)
    samples = pd.read_csv(sample_path)
    forbidden = {"tissue", "histology", "tumor", "normal", "phenotype", "qvalue", "pvalue"}
    suspicious = [column for column in matrix if any(token in column.lower() for token in forbidden)]
    if suspicious:
        raise RuntimeError(f"phenotype-like columns forbidden in graph matrix: {suspicious}")
    required = {"feature_id", "mz", "rt_sec"}
    if not required <= set(targets):
        raise RuntimeError(f"targets missing {sorted(required - set(targets))}")
    if "feature_id" not in matrix or matrix["feature_id"].duplicated().any():
        raise RuntimeError("intensity matrix needs one unique row per feature_id")

    sample_columns = [column for column in matrix if column != "feature_id"]
    declared = set(samples["sample_name"].astype(str))
    unknown = sorted(set(sample_columns) - declared)
    if unknown:
        raise RuntimeError(f"matrix contains undeclared samples: {unknown[:10]}")
    aligned = targets[["feature_id", "mz", "rt_sec"]].merge(
        matrix, on="feature_id", how="inner", validate="one_to_one"
    ).sort_values("feature_id").reset_index(drop=True)
    if len(aligned) != len(targets):
        raise RuntimeError(f"{panel}: matrix covers {len(aligned)}/{len(targets)} target features")
    raw_values = aligned[sample_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if detection_path is not None:
        detection = pd.read_csv(detection_path)
        if "feature_id" not in detection or detection["feature_id"].duplicated().any():
            raise RuntimeError(f"{panel}: detection matrix needs one unique row per feature_id")
        detection_columns = [column for column in detection if column != "feature_id"]
        if detection_columns != sample_columns:
            raise RuntimeError(f"{panel}: EIC AUC/detection sample columns differ")
        aligned_detection = aligned[["feature_id"]].merge(
            detection, on="feature_id", how="left", validate="one_to_one"
        )
        if aligned_detection[sample_columns].isna().any().any():
            raise RuntimeError(f"{panel}: detection matrix does not cover every target/sample")
        detected = aligned_detection[sample_columns].astype(bool).to_numpy()
        raw_values[~detected] = 0.0
    normalized = normalize_intensity_matrix(raw_values)
    candidates = mass_relation_pairs(
        aligned["mz"].to_numpy(float),
        aligned["rt_sec"].to_numpy(float),
        panel_relations(panel),
        ppm=args.ppm,
        absolute_floor_da=args.absolute_floor_da,
        rt_tolerance_sec=args.rt_tolerance_sec,
    )

    edge_rows: list[dict] = []
    for number, (left, right, relation, mass_error, drt) in enumerate(candidates, start=1):
        evidence = pair_evidence(normalized[left], normalized[right])
        pearson = float(evidence["abundance_pearson"])
        even = float(evidence["pearson_half_even"])
        odd = float(evidence["pearson_half_odd"])
        accepted = (
            int(evidence["n_joint_detected"]) >= args.minimum_joint_samples
            and np.isfinite(pearson)
            and pearson >= args.minimum_pearson
        )
        split_replicated = (
            np.isfinite(even)
            and np.isfinite(odd)
            and even >= args.minimum_half_pearson
            and odd >= args.minimum_half_pearson
        )
        edge_rows.append({
            "feature_id_a": int(aligned.iloc[left].feature_id),
            "feature_id_b": int(aligned.iloc[right].feature_id),
            "relation": relation.name,
            "relation_family": relation.family,
            "merge_ion_family": bool(relation.merge_ion_family),
            "mass_error_da": mass_error,
            "rt_difference_sec": drt,
            **evidence,
            "accepted": bool(accepted),
            "split_replicated": bool(split_replicated),
        })
        if number % 100000 == 0:
            print(f"[{panel}] evidence {number:,}/{len(candidates):,}", flush=True)
    edges = pd.DataFrame(edge_rows)
    if edges.empty:
        edges = pd.DataFrame(columns=[
            "feature_id_a", "feature_id_b", "relation", "relation_family",
            "merge_ion_family", "accepted", "split_replicated",
        ])

    feature_position = {int(value): index for index, value in enumerate(aligned["feature_id"])}
    union = UnionFind(len(aligned))
    strong = edges[edges["accepted"].astype(bool) & edges["merge_ion_family"].astype(bool)]
    for row in strong.itertuples(index=False):
        union.union(feature_position[int(row.feature_id_a)], feature_position[int(row.feature_id_b)])
    nodes = aligned[["feature_id", "mz", "rt_sec"]].copy()
    nodes["ion_family_id"] = union.labels()
    family_size = nodes.groupby("ion_family_id")["feature_id"].transform("size")
    nodes["ion_family_size"] = family_size.astype(int)

    prefix = output / panel
    edge_path = prefix.with_name(prefix.name + "__global_peak_edges.csv.gz")
    node_path = prefix.with_name(prefix.name + "__global_peak_nodes.csv.gz")
    edges.to_csv(edge_path, index=False)
    nodes.to_csv(node_path, index=False)
    accepted = edges[edges["accepted"].astype(bool)]
    report = {
        "panel": panel,
        "formal": matrix_kind == "uniform_requantified_eic",
        "matrix_kind": matrix_kind,
        "features": int(len(nodes)),
        "samples": int(len(sample_columns)),
        "mass_rt_candidate_edges": int(len(edges)),
        "accepted_edges": int(len(accepted)),
        "split_replicated_edges": int(accepted["split_replicated"].sum()) if len(accepted) else 0,
        "accepted_by_family": {
            str(key): int(value) for key, value in accepted.groupby("relation_family").size().items()
        },
        "ion_families": int(nodes["ion_family_id"].nunique()),
        "multimember_ion_families": int((nodes.groupby("ion_family_id").size() > 1).sum()),
        "features_in_multimember_families": int((nodes["ion_family_size"] > 1).sum()),
        "provenance": {
            "targets_sha256": sha256(target_path),
            "samples_sha256": sha256(sample_path),
            "matrix_sha256": sha256(matrix_path),
            "detection_sha256": sha256(detection_path) if detection_path is not None else None,
            "edges_sha256": sha256(edge_path),
            "nodes_sha256": sha256(node_path),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--requant-dir", type=Path, default=Path("data/mtbls13729/ms1_eic_requant"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/bioaware_global_peak_v1"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--absolute-floor-da", type=float, default=0.002)
    parser.add_argument("--rt-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--minimum-joint-samples", type=int, default=8)
    parser.add_argument("--minimum-pearson", type=float, default=0.30)
    parser.add_argument("--minimum-half-pearson", type=float, default=0.10)
    parser.add_argument("--allow-discovery-matrix", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reports = [build_panel(args, panel, output) for panel in args.panels]
    payload = {
        "status": "mtbls13729_bioaware_global_peak_network_complete",
        "formal": all(report["formal"] for report in reports),
        "phenotype_labels_used": False,
        "candidate_identities_used": False,
        "panels": reports,
        "parameters": {
            "ppm": args.ppm,
            "absolute_floor_da": args.absolute_floor_da,
            "rt_tolerance_sec": args.rt_tolerance_sec,
            "minimum_joint_samples": args.minimum_joint_samples,
            "minimum_pearson": args.minimum_pearson,
            "minimum_half_pearson": args.minimum_half_pearson,
        },
        "decision_contract": (
            "This graph may group ion forms and support annotation coverage. "
            "It cannot by itself promote a strict molecular identity."
        ),
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
