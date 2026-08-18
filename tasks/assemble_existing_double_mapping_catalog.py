"""Join validated peak factors, structure contexts, rules, and interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factor-catalog", type=Path,
        default=ROOT / "data/validation/spectral_first_fragmentation_factor_pilot/validated_factor_catalog.csv",
    )
    parser.add_argument(
        "--occlusion-report", type=Path,
        default=ROOT / "data/validation/validated_factor_occlusion_v2/report.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/existing_factor_catalog",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factors = pd.read_csv(args.factor_catalog)
    occlusion_payload = json.loads(args.occlusion_report.read_text(encoding="utf-8"))
    occlusion = pd.DataFrame(occlusion_payload["factors"])
    joined = factors.merge(occlusion, on="factor", how="left", validate="one_to_one")
    joined["mapping_embedding_to_structure"] = (
        joined["structure_context_replicated"].fillna(False)
        & joined["structure_bh_q"].fillna(1.0).le(0.05)
    )
    joined["mapping_factor_to_peak"] = (
        joined["peak_localization_bh_q"].fillna(1.0).le(0.05)
        & joined["confirmation_active_molecules"].fillna(0).ge(10)
    )
    joined["peak_to_global_embedding_causal"] = joined.apply(
        lambda row: (
            isinstance(row.get("selective_embedding_shift_molecule_bootstrap_ci95"), list)
            and row["selective_embedding_shift_molecule_bootstrap_ci95"][0] > 0
        ),
        axis=1,
    )
    joined["peak_to_retrieval_causal"] = joined.apply(
        lambda row: (
            isinstance(row.get("selective_margin_drop_molecule_bootstrap_ci95"), list)
            and row["selective_margin_drop_molecule_bootstrap_ci95"][0] > 0
        ),
        axis=1,
    )
    joined["double_mapping_status"] = "spectral_factor_only"
    joined.loc[
        joined["mapping_embedding_to_structure"] & joined["mapping_factor_to_peak"],
        "double_mapping_status",
    ] = "two_mapping_edges_replicated"
    joined.loc[
        joined["mapping_embedding_to_structure"]
        & joined["mapping_factor_to_peak"]
        & joined["peak_to_global_embedding_causal"],
        "double_mapping_status",
    ] = "closed_loop_embedding_candidate"
    joined.to_csv(args.output_dir / "double_mapping_factor_catalog.csv", index=False)
    report = {
        "status": "existing_double_mapping_catalog_complete",
        "validated_spectral_factors": int(len(joined)),
        "embedding_to_structure_replicated": int(joined["mapping_embedding_to_structure"].sum()),
        "factor_to_peak_replicated": int(joined["mapping_factor_to_peak"].sum()),
        "closed_loop_embedding_candidates": int(
            joined["double_mapping_status"].eq("closed_loop_embedding_candidate").sum()
        ),
        "retrieval_causal_factors": int(joined["peak_to_retrieval_causal"].sum()),
        "closed_loop_factors": joined.loc[
            joined["double_mapping_status"].eq("closed_loop_embedding_candidate"),
            ["factor", "spectral_kind", "mass_da", "structure_environment",
             "structure_odds_ratio", "matched_core_rules", "matched_massbank_records"],
        ].to_dict(orient="records"),
        "interpretation": (
            "Closed-loop candidate means a stable local embedding factor maps to a replicated "
            "structure context and peak mass, and deleting that peak changes the global embedding."
        ),
        "claim_limit": (
            "No factor has yet shown a molecule-bootstrap retrieval-margin effect above zero; "
            "mass and environment annotations are not unique mechanisms."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
