#!/usr/bin/env python3
"""Test only the IBD end of three frozen AD-skin--VEO-IBD hypotheses.

The three target precursor masses were selected before ST001000 was opened.
Their expected IBD directions are read from the frozen VEO-IBD responses in
the connection-evidence ledger, also before any ST001000 abundance is used.
ST001000 features are selected by positive-mode precursor mass alone.

This analysis can support (or fail to support) replication of the VEO-IBD end
in independent PRISM and Netherlands participants.  It cannot validate the
AD--IBD relation, establish a shared molecular identity, or validate the AD
end.  ST001000 has no sample-linked MS/MS in these abundance matrices.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from validate_contrast_atlas_ibd_identity_hypotheses_prism import (
    bh_adjust,
    design_matrix,
    fit_ols,
    load_mapping,
    load_matrix,
    preprocess_abundance,
    sha256,
    write_tsv,
)


TARGET_MASSES = (295.2266, 197.1285, 158.1176)
TARGET_FREEZE_DATE = "2026-10-30"
AD_STUDY = "MSV000090788"
VEO_STUDY = "MSV000097610"
WSE_METHOD = "weighted_spectral_entropy_complete_axis"

ANALYSES = (
    ("AN001878", "HILIC_positive", 1),
    ("AN001881", "C8_positive", 2),
)


@dataclass(frozen=True)
class FrozenTarget:
    target_mz: float
    veo_entity_id: str
    veo_precursor_mz: float
    veo_beta: float
    veo_se: float
    veo_effect_q: float
    expected_ibd_direction: int
    ad_entity_id: str
    ad_precursor_mz: float
    identity_world: str
    method_edge_id: str
    source_connection_ids: str
    source_ad_contrasts: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--an001878", type=Path, required=True)
    parser.add_argument("--an001881", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    return parser.parse_args()


def _optional_float(value: str) -> float:
    return float(value) if value.strip() else math.nan


def _identity_world(left_status: str, right_status: str) -> str:
    if left_status == "library_supported" and right_status == "library_supported":
        return "known"
    if left_status == "unresolved_entity" and right_status == "unresolved_entity":
        return "unresolved"
    return "mixed_known_unresolved"


def load_frozen_targets(path: Path, ppm: float) -> tuple[list[FrozenTarget], dict]:
    """Resolve pre-fixed masses to VEO entities without using ST001000 data."""
    rows_by_target: dict[float, list[dict[str, str]]] = {
        mass: [] for mass in TARGET_MASSES
    }
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["method_name"] != WSE_METHOD or row["evidence_world"] != "all":
                continue
            if row["left_study_id"] != AD_STUDY or row["right_study_id"] != VEO_STUDY:
                continue
            veo_mz = float(row["right_precursor_mz"])
            for target_mz in TARGET_MASSES:
                error = abs(veo_mz - target_mz) / target_mz * 1_000_000.0
                if error <= ppm:
                    rows_by_target[target_mz].append(row)

    targets: list[FrozenTarget] = []
    audit_rows: list[dict] = []
    for target_mz in TARGET_MASSES:
        rows = rows_by_target[target_mz]
        if not rows:
            raise ValueError(f"no frozen VEO ledger row within {ppm} ppm of {target_mz}")
        entity_keys = {
            (
                row["left_entity_id"],
                row["right_entity_id"],
                row["method_edge_id"],
            )
            for row in rows
        }
        if len(entity_keys) != 1:
            raise ValueError(
                f"target {target_mz} resolves to {len(entity_keys)} entity pairs"
            )
        veo_betas = {float(row["right_beta"]) for row in rows}
        veo_ses = {float(row["right_se"]) for row in rows}
        veo_qs = {float(row["right_effect_q"]) for row in rows}
        veo_masses = {float(row["right_precursor_mz"]) for row in rows}
        identity_worlds = {
            _identity_world(
                row["left_common_identity_status"],
                row["right_common_identity_status"],
            )
            for row in rows
        }
        if not all(
            len(values) == 1
            for values in (veo_betas, veo_ses, veo_qs, veo_masses, identity_worlds)
        ):
            raise ValueError(f"inconsistent frozen VEO evidence for target {target_mz}")
        beta = next(iter(veo_betas))
        if beta == 0:
            raise ValueError(f"zero frozen VEO beta for target {target_mz}")
        first = rows[0]
        target = FrozenTarget(
            target_mz=target_mz,
            veo_entity_id=first["right_entity_id"],
            veo_precursor_mz=next(iter(veo_masses)),
            veo_beta=beta,
            veo_se=next(iter(veo_ses)),
            veo_effect_q=next(iter(veo_qs)),
            expected_ibd_direction=1 if beta > 0 else -1,
            ad_entity_id=first["left_entity_id"],
            ad_precursor_mz=float(first["left_precursor_mz"]),
            identity_world=next(iter(identity_worlds)),
            method_edge_id=first["method_edge_id"],
            source_connection_ids=";".join(
                sorted({row["undirected_connection_id"] for row in rows})
            ),
            source_ad_contrasts=";".join(
                sorted({row["left_contrast_id"] for row in rows})
            ),
        )
        targets.append(target)
        audit_rows.append(
            {
                "target_mz": target_mz,
                "ledger_rows": len(rows),
                "veo_entity_id": target.veo_entity_id,
                "veo_precursor_mz": target.veo_precursor_mz,
                "veo_beta": target.veo_beta,
                "veo_effect_q": target.veo_effect_q,
                "expected_ibd_direction": target.expected_ibd_direction,
                "identity_world": target.identity_world,
                "source_ad_contrasts": target.source_ad_contrasts,
            }
        )
    return targets, {"target_freeze_date": TARGET_FREEZE_DATE, "targets": audit_rows}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets, freeze_audit = load_frozen_targets(args.ledger, args.ppm)
    mapping_rows, mapping = load_mapping(args.mapping)
    matrix_paths = {"AN001878": args.an001878, "AN001881": args.an001881}

    match_rows: list[dict] = []
    effect_rows: list[dict] = []
    analysis_audit: dict[str, dict] = {}
    for analysis_id, mode, metadata_columns in ANALYSES:
        sample_names, feature_ids, masses, raw_matrix = load_matrix(
            matrix_paths[analysis_id], metadata_columns
        )
        if set(sample_names) != set(mapping):
            raise ValueError(f"sample namespace mismatch for {analysis_id}")
        transformed = preprocess_abundance(raw_matrix)

        provisional: list[dict] = []
        for target in targets:
            errors = np.abs(masses - target.target_mz) / target.target_mz * 1_000_000.0
            feature_idx = int(np.argmin(errors))
            if float(errors[feature_idx]) > args.ppm:
                continue
            provisional.append(
                {
                    **target.__dict__,
                    "analysis_id": analysis_id,
                    "mode": mode,
                    "feature_index": feature_idx,
                    "feature_id": feature_ids[feature_idx],
                    "feature_mz": float(masses[feature_idx]),
                    "match_ppm": float(errors[feature_idx]),
                }
            )

        # One measured feature is one validation opportunity.  Resolve any
        # collision using only the precursor-mass error, never the phenotype.
        best_by_feature: dict[str, dict] = {}
        for row in sorted(provisional, key=lambda item: (item["match_ppm"], item["target_mz"])):
            best_by_feature.setdefault(row["feature_id"], row)
        matches = list(best_by_feature.values())
        match_rows.extend(matches)

        cohort_ns: dict[str, int] = {}
        fit_specs = (
            ("PRISM", "complete_case", "complete_case"),
            ("PRISM", "na_as_no_sensitivity", "na_as_no"),
            ("Neth", "not_applicable", "na_as_no"),
        )
        for cohort, policy_label, design_policy in fit_specs:
            selected, design, coefficient_names, _ = design_matrix(
                sample_names, mapping, cohort, design_policy
            )
            cohort_ns[f"{cohort}:{policy_label}"] = int(selected.size)
            for match in matches:
                fit = fit_ols(
                    transformed[int(match["feature_index"]), selected],
                    design,
                    coefficient_names,
                )
                for diagnosis in ("CD", "UC"):
                    estimate = fit[diagnosis]
                    effect_rows.append(
                        {
                            **{key: value for key, value in match.items() if key != "feature_index"},
                            "cohort": cohort,
                            "missing_medication_policy": policy_label,
                            "diagnosis": diagnosis,
                            "n": int(selected.size),
                            "beta": estimate["beta"],
                            "se": estimate["se"],
                            "t": estimate["t"],
                            "p": estimate["p"],
                            "df": estimate["df"],
                            "direction_matches_veo": int(
                                np.sign(estimate["beta"])
                                == int(match["expected_ibd_direction"])
                            ),
                        }
                    )

        analysis_audit[analysis_id] = {
            "mode": mode,
            "features": len(feature_ids),
            "samples": len(sample_names),
            "targets_mass_matched": len(matches),
            "unmatched_target_masses": sorted(
                set(TARGET_MASSES) - {row["target_mz"] for row in matches}
            ),
            "cohort_fit_n": cohort_ns,
        }

    # One multiplicity family per cohort/model: all pre-frozen targets, modes,
    # and CD/UC endpoints actually observed in that fit.
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(effect_rows):
        groups.setdefault((row["cohort"], row["missing_medication_policy"]), []).append(idx)
    for indices in groups.values():
        q_values = bh_adjust(effect_rows[idx]["p"] for idx in indices)
        for idx, q_value in zip(indices, q_values):
            effect_rows[idx]["q"] = q_value

    primary: dict[tuple[str, str], dict] = {}
    sensitivity: dict[tuple[str, str], dict] = {}
    neth: dict[tuple[str, str], dict] = {}
    for row in effect_rows:
        key = (row["feature_id"], row["diagnosis"])
        if row["cohort"] == "PRISM" and row["missing_medication_policy"] == "complete_case":
            primary[key] = row
        elif row["cohort"] == "PRISM":
            sensitivity[key] = row
        else:
            neth[key] = row

    replication_rows: list[dict] = []
    for key, prism in primary.items():
        dutch = neth[key]
        sens = sensitivity[key]
        strict = (
            prism["direction_matches_veo"] == 1
            and dutch["direction_matches_veo"] == 1
            and sens["direction_matches_veo"] == 1
            and prism["q"] <= 0.05
            and dutch["q"] <= 0.05
            and sens["q"] <= 0.05
        )
        replication_rows.append(
            {
                "target_mz": prism["target_mz"],
                "veo_entity_id": prism["veo_entity_id"],
                "veo_beta": prism["veo_beta"],
                "veo_effect_q": prism["veo_effect_q"],
                "expected_ibd_direction": prism["expected_ibd_direction"],
                "identity_world": prism["identity_world"],
                "analysis_id": prism["analysis_id"],
                "mode": prism["mode"],
                "feature_id": prism["feature_id"],
                "feature_mz": prism["feature_mz"],
                "match_ppm": prism["match_ppm"],
                "diagnosis": prism["diagnosis"],
                "prism_beta": prism["beta"],
                "prism_p": prism["p"],
                "prism_q": prism["q"],
                "neth_beta": dutch["beta"],
                "neth_p": dutch["p"],
                "neth_q": dutch["q"],
                "prism_na_as_no_beta": sens["beta"],
                "prism_na_as_no_q": sens["q"],
                "all_three_direction_match": bool(
                    prism["direction_matches_veo"]
                    and dutch["direction_matches_veo"]
                    and sens["direction_matches_veo"]
                ),
                "strict_three_fit_concordance": strict,
                "claim_boundary": (
                    "IBD-end candidate passing three within-fit BH filters at matching "
                    "positive-mode mass only;not replicated-findings FDR control;"
                    "not AD-IBD relation validation;not same-entity confirmation"
                ),
            }
        )
    replication_rows.sort(
        key=lambda row: (
            not row["strict_three_fit_concordance"],
            max(row["prism_q"], row["neth_q"], row["prism_na_as_no_q"]),
            row["target_mz"],
            row["analysis_id"],
            row["diagnosis"],
        )
    )

    match_fields = [key for key in match_rows[0] if key != "feature_index"] if match_rows else []
    effect_fields = [key for key in effect_rows[0] if key != "feature_index"] if effect_rows else []
    replication_fields = list(replication_rows[0]) if replication_rows else []
    write_tsv(
        args.output_dir / "phenotype_blind_mass_matches.tsv",
        [{key: value for key, value in row.items() if key != "feature_index"} for row in match_rows],
        match_fields,
    )
    write_tsv(args.output_dir / "cohort_effects.tsv", effect_rows, effect_fields)
    write_tsv(
        args.output_dir / "ibd_end_concordance_screen.tsv",
        replication_rows,
        replication_fields,
    )
    strict_rows = [row for row in replication_rows if row["strict_three_fit_concordance"]]
    write_tsv(
        args.output_dir / "strict_three_fit_concordant_candidates.tsv",
        strict_rows,
        replication_fields,
    )

    cohort_counts: dict[str, dict[str, int]] = {}
    for cohort in ("PRISM", "Neth"):
        selected = [row for row in mapping_rows if row["Cohort"] == cohort]
        cohort_counts[cohort] = {
            "total": len(selected),
            "CD": sum(row["Diagnosis"] == "CD" for row in selected),
            "UC": sum(row["Diagnosis"] == "UC" for row in selected),
            "Control": sum(row["Diagnosis"] == "Control" for row in selected),
        }

    report = {
        "status": "COMPLETE",
        "data_role": "development_now_consumed_two_participant_cohorts_not_sealed_external",
        "scientific_question": (
            "Do the pre-frozen VEO-IBD directions of three AD-skin--VEO-IBD "
            "mass hypotheses replicate at the IBD end in PRISM and Netherlands?"
        ),
        "target_freeze": freeze_audit,
        "leakage_status": {
            "target_masses": (
                "fixed to 295.2266, 197.1285, and 158.1176 before ST001000 effects"
            ),
            "expected_directions": (
                "read from the VEO-IBD right_beta in the frozen ledger before "
                "ST001000 abundance or phenotype effects"
            ),
            "feature_selection": (
                "nearest positive-mode feature within 10 ppm, selected without phenotype"
            ),
        },
        "claim_boundary": {
            "supported_scope": "independent participant replication test of the IBD end",
            "not_supported": [
                "AD-skin--IBD biological relation",
                "same molecular identity across studies",
                "AD-end replication",
                "unknown-information increment",
                "in-house algorithm increment",
            ],
            "identity_limitation": (
                "ST001000 abundance matrices are full-scan and lack sample-linked MS/MS; "
                "mass matching is not identity confirmation"
            ),
        },
        "source_provenance": {
            "study": "Metabolomics Workbench ST001000 / PR000677",
            "primary_publication": "https://www.nature.com/articles/s41564-018-0306-4",
            "open_full_text": "https://pmc.ncbi.nlm.nih.gov/articles/mid/EMS80310/",
            "cohort_independence": (
                "PRISM and Netherlands contain independent participants but belong to "
                "one published study package; they are not two independent projects"
            ),
        },
        "preprocessing_and_models": {
            "abundance": (
                "within-sample sum normalization to 1e6, half-minimum zero smoothing, log10"
            ),
            "PRISM_primary": (
                "CD and UC vs control with age, antibiotics, immunosuppressants, "
                "mesalamine, and steroids; medication complete cases"
            ),
            "PRISM_sensitivity": "same model with medication NA treated as no",
            "Netherlands": "CD and UC vs control with age",
            "multiplicity": (
                "BH within each cohort/model across all mass-matched target-mode x CD/UC tests"
            ),
        },
        "cohort_counts": cohort_counts,
        "analysis_audit": analysis_audit,
        "result_counts": {
            "frozen_targets": len(targets),
            "target_mode_mass_matches": len(match_rows),
            "target_mode_diagnosis_tests": len(replication_rows),
            "all_three_direction_match": sum(
                row["all_three_direction_match"] for row in replication_rows
            ),
            "three_fit_concordant_candidates": len(strict_rows),
        },
        "result_interpretation": (
            "A three-fit concordance pass requires the frozen VEO direction and "
            "within-fit BH q<=0.05 for the same diagnosis in PRISM primary, "
            "Netherlands, and PRISM sensitivity. This is not formal replicated-findings "
            "FDR control, and PRISM sensitivity is not an independent cohort."
        ),
        "ols_inference_boundary": (
            "Classical homoskedastic OLS standard errors reproduce the intended published "
            "model; heteroskedasticity-robust and outlier sensitivity were not tested."
        ),
        "inputs": {
            "ledger": str(args.ledger),
            "mapping": str(args.mapping),
            "AN001878": str(args.an001878),
            "AN001881": str(args.an001881),
        },
        "input_sha256": {
            "ledger": sha256(args.ledger),
            "mapping": sha256(args.mapping),
            "AN001878": sha256(args.an001878),
            "AN001881": sha256(args.an001881),
        },
        "script_sha256": sha256(Path(__file__)),
        "shared_analysis_module_sha256": sha256(
            Path(__file__).with_name(
                "validate_contrast_atlas_ibd_identity_hypotheses_prism.py"
            )
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    output_files = (
        "phenotype_blind_mass_matches.tsv",
        "cohort_effects.tsv",
        "ibd_end_concordance_screen.tsv",
        "strict_three_fit_concordant_candidates.tsv",
        "report.json",
    )
    checksum_lines = [
        f"{sha256(args.output_dir / name)}  {name}" for name in output_files
    ]
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
