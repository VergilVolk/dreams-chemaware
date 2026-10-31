#!/usr/bin/env python3
"""Prepare the qualified MSV000080918 response objects for Atlas ingestion.

Only response-eligible features with a reciprocal-unique feature--cluster link
are exported.  Ambiguous spectral candidates remain in the source audit but do
not become one-spectrum response entities in this collection adapter.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


STATUS = "MSV000080918_ATLAS_SOURCE_PREPARED"
VALID_SPECTRAL_STATES = {
    "reciprocal_unique_unresolved": "unresolved_entity",
    "reciprocal_unique_library_supported": "library_supported",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mgf_blocks(path: Path) -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    current: list[str] | None = None
    scan: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.strip() == "BEGIN IONS":
                if current is not None:
                    raise RuntimeError("nested MGF block")
                current = ["BEGIN IONS"]
                scan = None
                continue
            if current is None:
                continue
            current.append(line)
            if line.startswith("SCANS="):
                scan = int(line.split("=", 1)[1])
            if line.strip() == "END IONS":
                if scan is None or scan in blocks:
                    raise RuntimeError("missing or duplicate MGF SCANS")
                blocks[scan] = current
                current = None
    if current is not None:
        raise RuntimeError("unterminated MGF block")
    return blocks


def feature_block(feature_id: str, cluster: int, block: list[str]) -> list[str]:
    output = ["BEGIN IONS", f"FEATURE_ID={feature_id}", f"TITLE={feature_id}"]
    for line in block[1:-1]:
        key = line.split("=", 1)[0].upper() if "=" in line else ""
        if key in {"FEATURE_ID", "TITLE"}:
            continue
        output.append(line)
    output.extend([f"ORIGINAL_CLUSTER_INDEX={cluster}", "END IONS"])
    return output


def run(source_dir: Path, mgf_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_report_path = source_dir / "report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    expected_mgf = source_report["inputs"]["mgf_sha256"]
    if sha256(mgf_path) != expected_mgf:
        raise RuntimeError("MGF checksum differs from qualified source report")

    effect_path = source_dir / "contrast_effects.tsv.gz"
    candidate_path = source_dir / "feature_cluster_candidates.tsv.gz"
    effects = pd.read_csv(effect_path, sep="\t", dtype={"feature_id": str})
    candidates = pd.read_csv(candidate_path, sep="\t", dtype={"feature_id": str})
    effects = effects.loc[
        effects["response_eligible"].astype(bool)
        & effects["spectral_evidence_status"].isin(VALID_SPECTRAL_STATES)
    ].copy()
    if effects.empty or effects.duplicated(["contrast_id", "feature_id"]).any():
        raise RuntimeError("invalid qualified effect rows")
    effects["canonical_feature_id"] = effects["feature_id"].str.replace(
        r"\s+", "", regex=True
    )
    if effects.duplicated(["contrast_id", "canonical_feature_id"]).any():
        raise RuntimeError("feature ID normalization creates duplicate response rows")

    entity_status = effects[["canonical_feature_id", "spectral_evidence_status"]].drop_duplicates()
    if entity_status["canonical_feature_id"].duplicated().any():
        raise RuntimeError("identity state differs across contrasts")
    entity_status["identity_status"] = entity_status["spectral_evidence_status"].map(
        VALID_SPECTRAL_STATES
    )
    entity_ids = set(entity_status["canonical_feature_id"])

    candidates["canonical_feature_id"] = candidates["feature_id"].str.replace(
        r"\s+", "", regex=True
    )
    selected_candidates = candidates.loc[candidates["canonical_feature_id"].isin(entity_ids)].copy()
    if selected_candidates["canonical_feature_id"].duplicated().any():
        raise RuntimeError("exported feature has more than one spectral candidate")
    if set(selected_candidates["canonical_feature_id"]) != entity_ids:
        raise RuntimeError("exported feature lacks a spectral candidate")
    if not selected_candidates["cluster_feature_count"].eq(1).all():
        raise RuntimeError("exported candidate is not reciprocal unique")

    standardized = pd.DataFrame({
        "contrast_id": effects["contrast_id"].astype(str),
        "entity_id": effects["canonical_feature_id"].astype(str),
        "beta": pd.to_numeric(effects["beta_log2_treatment_minus_control"]),
        "se": pd.to_numeric(effects["se_cluster_cr1"]),
        "prevalence": pd.to_numeric(effects["pooled_cage_prevalence"]),
        "effect_model": (
            "cage-cell OLS: treatment/genotype + collection-date fixed effects + "
            "centered age; cage-clustered CR1 SE"
        ),
    })
    if standardized[["beta", "se"]].isna().any().any() or (standardized["se"] <= 0).any():
        raise RuntimeError("non-finite Atlas response effect")

    identity = entity_status[["canonical_feature_id", "identity_status"]].rename(
        columns={"canonical_feature_id": "entity_id"}
    ).sort_values("entity_id")
    blocks = read_mgf_blocks(mgf_path)
    selected_candidates = selected_candidates.sort_values("canonical_feature_id")
    missing = set(selected_candidates["cluster_index"].astype(int)) - set(blocks)
    if missing:
        raise RuntimeError(f"selected clusters absent from MGF: {sorted(missing)[:3]}")

    output_dir.mkdir(parents=True)
    effect_out = output_dir / "effects.tsv.gz"
    evidence_out = output_dir / "response_evidence.tsv.gz"
    identity_out = output_dir / "source_native_identity.tsv"
    mgf_out = output_dir / "representative_spectra.mgf"
    standardized.to_csv(
        effect_out, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    evidence_columns = [
        "study_id", "source_family", "contrast_id", "analysis_role", "feature_id",
        "beta_log2_treatment_minus_control", "se_cluster_cr1", "t_cluster_cr1",
        "p_cluster_cr1", "q_cluster_cr1_within_contrast", "bh_family_size",
        "date_loso_estimable_folds", "date_loso_direction_consistent",
        "date_loso_min_abs_beta", "independent_cage_clusters", "cage_arm_cells",
        "pooled_detected_cages", "pooled_cage_prevalence",
        "minimum_required_detected_cages", "spectral_evidence_status",
    ]
    evidence = effects[evidence_columns].rename(
        columns={"feature_id": "source_feature_id"}
    ).copy()
    evidence.insert(
        evidence.columns.get_loc("source_feature_id") + 1,
        "entity_id",
        effects["canonical_feature_id"].to_numpy(),
    )
    evidence.to_csv(
        evidence_out, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    identity.to_csv(identity_out, sep="\t", index=False)
    rendered_blocks: list[str] = []
    for row in selected_candidates.itertuples(index=False):
        cluster = int(row.cluster_index)
        rendered_blocks.append("\n".join(feature_block(
            str(row.canonical_feature_id), cluster, blocks[cluster]
        )))
    mgf_out.write_bytes(("\n\n".join(rendered_blocks) + "\n").encode("utf-8"))

    summary = effects.groupby(["contrast_id", "spectral_evidence_status"]).size()
    report: dict[str, object] = {
        "schema": "msv000080918_contrast_atlas_source_v1",
        "status": STATUS,
        "data_role": "development_consumed",
        "source_family": "MSV000080918_MSV000080558",
        "counts": {
            "response_rows": int(len(standardized)),
            "response_entities_union": int(identity["entity_id"].nunique()),
            "representative_spectra": int(len(selected_candidates)),
            "identity_status": identity["identity_status"].value_counts().to_dict(),
            "contrast_by_spectral_status": {
                f"{contrast}::{state}": int(value)
                for (contrast, state), value in summary.items()
            },
        },
        "selection_contract": (
            "response_eligible and reciprocal-unique in the fixed 10-ppm/12-second/"
            "source-overlap graph; no response value, q-value, or direction used for spectral selection"
        ),
        "contrast_roles": {
            "mup_hfd_10_11m_iga_ko_vs_intact_date_matched": "primary_development_positive_control",
            "mup_hfd_10_11m_cd8_ko_vs_intact_date_matched": "sensitivity_development_positive_control",
        },
        "inputs": {
            "qualified_source_report": str(source_report_path),
            "qualified_source_report_sha256": sha256(source_report_path),
            "contrast_effects_sha256": sha256(effect_path),
            "feature_cluster_candidates_sha256": sha256(candidate_path),
            "mgf_sha256": sha256(mgf_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "outputs_sha256": {
            effect_out.name: sha256(effect_out),
            evidence_out.name: sha256(evidence_out),
            identity_out.name: sha256(identity_out),
            mgf_out.name: sha256(mgf_out),
        },
        "claim_boundary": [
            "Reciprocal uniqueness is graph-local evidence, not chemical identity truth.",
            "The CD8 contrast remains sensitivity-only.",
            "No cross-study retrieval, unknown increment, algorithm increment, or external validation is claimed.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--mgf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.mgf, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
