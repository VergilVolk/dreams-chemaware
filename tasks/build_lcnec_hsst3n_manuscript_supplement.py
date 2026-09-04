"""Assemble frozen LCNEC manuscript tables without refitting or reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_manuscript_supplement"),
    )
    args = parser.parse_args()

    sources = {
        "robust_features": Path("data/validation/lcnec_hsst3n_dark_robustness_gate/normalization_robustness_ledger.csv"),
        "modules": Path("data/validation/lcnec_hsst3n_dark_robustness_gate/nonredundant_module_membership.csv"),
        "identities": Path("data/validation/lcnec_hsst3n_annotation_biology/identity_evidence_ledger.csv"),
        "structure": Path("data/validation/lcnec_hsst3n_priority_structure/priority_structure_ledger.csv"),
        "pairs": Path("data/validation/lcnec_hsst3n_priority_pair_consistency/pair_consistency_ledger.csv"),
        "per_patient": Path("data/validation/lcnec_hsst3n_priority_pair_consistency/per_patient_effects.csv"),
        "bioaware": Path("data/validation/lcnec_hsst3n_bioaware_context/bioaware_context_ledger.csv"),
        "adduct": Path("data/validation/lcnec_hsst3n_priority_adduct_audit/priority_adduct_audit.csv"),
        "readiness": Path("data/validation/lcnec_hsst3n_manuscript_readiness/readiness_report.json"),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen manuscript inputs: {missing}")

    robust = pd.read_csv(sources["robust_features"])
    modules = pd.read_csv(sources["modules"])
    identities = pd.read_csv(sources["identities"])
    structure = pd.read_csv(sources["structure"])
    pairs = pd.read_csv(sources["pairs"])
    per_patient = pd.read_csv(sources["per_patient"])
    bioaware = pd.read_csv(sources["bioaware"])
    adduct = pd.read_csv(sources["adduct"])

    module_table = modules.merge(
        robust,
        on=["family_id", "mz", "rt_sec"],
        how="left",
        validate="one_to_one",
    ).sort_values(["module_id", "family_id"])
    cross_platform = identities.loc[
        identities["author_status"].eq("published_atlas_overlap")
    ].copy()
    unreported = identities.loc[
        identities["author_status"].eq("author_unreported_spectral_hypothesis")
    ].copy()
    priority = (
        structure.merge(pairs, on=["family_id", "priority_name"], how="inner", validate="one_to_one")
        .merge(bioaware, on=["priority_name", "ik14"], how="inner", validate="one_to_one")
        .merge(
            adduct.drop(columns=["target_mz", "target_rt_sec"], errors="ignore"),
            on=["family_id", "priority_name"],
            how="inner",
            validate="one_to_one",
        )
    )

    if modules["module_id"].nunique() != 81:
        raise RuntimeError("frozen module table no longer contains 81 modules")
    if module_table["quality_pass"].isna().any():
        raise RuntimeError("one or more frozen module members lack robustness evidence")
    if identities["ik14"].nunique() != 21:
        raise RuntimeError("frozen identity table no longer contains 21 hypotheses")
    if len(cross_platform) != 12:
        raise RuntimeError("cross-platform table no longer contains 12 overlaps")
    if len(unreported) != 9:
        raise RuntimeError("author-unreported table no longer contains 9 hypotheses")
    if len(priority) != 4:
        raise RuntimeError("priority table did not join to exactly four hypotheses")
    if not priority["leave_one_pair_out_sign_stable"].astype(bool).all():
        raise RuntimeError("priority LOPO direction is not uniformly stable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table_s1_81_dark_module_membership.csv": module_table,
        "table_s2_21_identity_hypotheses.csv": identities,
        "table_s3_12_cross_platform_reproductions.csv": cross_platform,
        "table_s4_9_author_unreported_hypotheses.csv": unreported,
        "table_s5_4_priority_evidence_ledger.csv": priority,
        "table_s6_priority_per_patient_effects.csv": per_patient,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False)

    manifest = {
        "status": "lcnec_hsst3n_manuscript_supplement_complete",
        "formal": True,
        "tables": {
            name: {"rows": int(len(table)), "sha256": sha256(args.output_dir / name)}
            for name, table in outputs.items()
        },
        "source_provenance": {
            name: {"path": path.as_posix(), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "contracts": {
            "result_refit": False,
            "candidate_reselection": False,
            "phenotype_used_for_identity": False,
            "level1_identity_claimed": False,
            "flux_or_causality_claimed": False,
        },
    }
    (args.output_dir / "supplement_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
