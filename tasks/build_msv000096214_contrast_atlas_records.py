#!/usr/bin/env python
"""Build gated Contrast Atlas records from the MSV000096214 raw matrices.

This script may run only after the truth-blind table audit and the frozen raw
effect stage both pass.  Genotype labels are opened only at that point.  The
result is a within-study response table, not cross-study validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "WT_Col0"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_to_genotype(manifest: pd.DataFrame, polarity: str) -> dict[str, str]:
    rows = manifest.loc[(manifest["role"] == "study") & (manifest["polarity"] == polarity)]
    mapping = {Path(row.filename).stem: str(row.genotype) for row in rows.itertuples(index=False)}
    if len(mapping) != len(rows):
        raise RuntimeError(f"duplicate sample stem in {polarity} manifest")
    return mapping


def prepare_log2(matrix: pd.DataFrame) -> pd.DataFrame:
    numeric = matrix.apply(pd.to_numeric, errors="raise").where(lambda frame: frame > 0)
    transformed = np.log2(numeric)
    return transformed.subtract(transformed.median(axis=0, skipna=True), axis=1)


def build_records(
    matrix: pd.DataFrame,
    groups: pd.DataFrame,
    genotype_by_sample: dict[str, str],
    polarity: str,
    study_id: str,
    source_family: str,
    min_detected_per_group: int = 4,
) -> pd.DataFrame:
    if matrix.index.name != "group_id":
        matrix.index.name = "group_id"
    missing_samples = set(genotype_by_sample) - set(matrix.columns)
    if missing_samples:
        raise RuntimeError(f"matrix missing registered study samples: {sorted(missing_samples)[:3]}")
    log2 = prepare_log2(matrix[list(genotype_by_sample)])
    reference_columns = [sample for sample, genotype in genotype_by_sample.items() if genotype == REFERENCE]
    if len(reference_columns) < min_detected_per_group:
        raise RuntimeError("insufficient reference biological samples")
    annotations = groups.set_index("group_id")
    if not log2.index.isin(annotations.index).all():
        raise RuntimeError("matrix contains group IDs absent from groups table")

    rows: list[pd.DataFrame] = []
    genotypes = sorted(set(genotype_by_sample.values()) - {REFERENCE})
    reference = log2[reference_columns]
    for genotype in genotypes:
        mutant_columns = [sample for sample, label in genotype_by_sample.items() if label == genotype]
        if len(mutant_columns) < min_detected_per_group:
            raise RuntimeError(f"insufficient biological samples for {genotype}")
        mutant = log2[mutant_columns]
        n_mutant = mutant.notna().sum(axis=1)
        n_reference = reference.notna().sum(axis=1)
        detectable = (n_mutant >= min_detected_per_group) & (n_reference >= min_detected_per_group)
        beta = mutant.mean(axis=1) - reference.mean(axis=1)
        variance = mutant.var(axis=1, ddof=1) / n_mutant + reference.var(axis=1, ddof=1) / n_reference
        se = np.sqrt(variance)
        eligible = detectable & np.isfinite(beta) & np.isfinite(se) & (se > 0)
        selected = pd.DataFrame({
            "study_id": study_id,
            "source_family": source_family,
            "contrast_id": f"{genotype}_vs_{REFERENCE}",
            "entity_id": [f"{polarity}:g{group_id}" for group_id in log2.index],
            "independent_unit_id": [f"{polarity}:g{group_id}" for group_id in log2.index],
            "identity_status": "unresolved_entity",
            "beta": beta.to_numpy(float),
            "se": se.to_numpy(float),
            "qc_pass": eligible.to_numpy(bool),
            "detectable": detectable.to_numpy(bool),
            "data_role": "development/consumed",
            "polarity": polarity,
            "group_id": log2.index.to_numpy(),
            "mz": annotations.loc[log2.index, "mz"].to_numpy(float),
            "rt_sec": annotations.loc[log2.index, "rt_sec"].to_numpy(float),
            "n_mutant": n_mutant.to_numpy(int),
            "n_reference": n_reference.to_numpy(int),
            "effect_model": "sample_median_centered_log2_mean_difference_with_Welch_SE",
        })
        rows.append(selected.loc[selected["qc_pass"]].copy())
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/external/msv000096214/consensus"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/msv000096214_perturbation_readiness_20261012_v2/sample_manifest.csv.gz"))
    parser.add_argument("--table-audit", type=Path, default=Path("data/validation/msv000096214_table_audit/report.json"))
    parser.add_argument("--effect-report", type=Path, default=Path("data/validation/msv000096214_effect_stage/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/msv000096214_contrast_atlas_records"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    audit = load_json(args.table_audit)
    effect = load_json(args.effect_report)
    if not audit.get("pass_to_effect_stage"):
        raise SystemExit("CONTRAST_RECORDS_BLOCKED: truth-blind table audit did not pass")
    if not effect.get("pass_to_posthoc_t4"):
        raise SystemExit("CONTRAST_RECORDS_BLOCKED: frozen raw response-object gate did not pass")

    manifest = pd.read_csv(args.manifest)
    all_records = []
    input_hashes = {
        "manifest": sha256(args.manifest),
        "table_audit": sha256(args.table_audit),
        "effect_report": sha256(args.effect_report),
    }
    polarity_summary = {}
    for polarity in ("positive", "negative"):
        matrix_path = args.consensus_dir / polarity / "matrix_filled.csv.gz"
        groups_path = args.consensus_dir / polarity / "groups.csv"
        report_path = args.consensus_dir / polarity / "report.json"
        for path in (matrix_path, groups_path, report_path):
            if not path.exists():
                raise FileNotFoundError(path)
        input_hashes[f"{polarity}_matrix"] = sha256(matrix_path)
        input_hashes[f"{polarity}_groups"] = sha256(groups_path)
        input_hashes[f"{polarity}_report"] = sha256(report_path)
        matrix = pd.read_csv(matrix_path, index_col="group_id")
        groups = pd.read_csv(groups_path)
        mapping = sample_to_genotype(manifest, polarity)
        records = build_records(
            matrix, groups, mapping, polarity,
            study_id=f"MSV000096214_{polarity}",
            source_family="MassIVE_MSV000096214_SJ_flavonoid_mutants",
        )
        all_records.append(records)
        polarity_summary[polarity] = {
            "consensus_groups": int(len(groups)),
            "study_samples": int(len(mapping)),
            "contrasts": int(records["contrast_id"].nunique()),
            "eligible_records": int(len(records)),
        }

    combined = pd.concat(all_records, ignore_index=True)
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "contrast_records.csv.gz"
    combined.to_csv(output_path, index=False, compression="gzip")
    report = {
        "status": "MSV000096214_CONTRAST_ATLAS_RECORDS_COMPLETE",
        "formal": True,
        "data_role": "development/consumed",
        "sealed": False,
        "structure_labels_used": False,
        "gate_provenance": {
            "truth_blind_table_audit_passed": True,
            "frozen_raw_response_object_passed": True,
        },
        "effect_model": "sample-median-centered log2 mean difference with Welch SE",
        "reference": REFERENCE,
        "source_family": "MassIVE_MSV000096214_SJ_flavonoid_mutants",
        "polarity": polarity_summary,
        "total_records": int(len(combined)),
        "input_sha256": input_hashes,
        "output_sha256": sha256(output_path),
        "claim_limit": "Within-study consumed response records only; no MS/MS, cross-study retrieval, unknown increment, or external validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
