#!/usr/bin/env python3
"""Check corrected MSV000095331 effects against published product directions.

The response table is estimated without drug identities.  Only afterwards this
script opens three exact product targets described by Zhao et al. (2025): an
atorvastatin -18.01-Da analog, an abacavir +31.99-Da analog, and the abiotic
omeprazole product at m/z 330.127.  Four microbial communities are reported as
fixed contexts, not treated as four independent studies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TARGETS = (
    {
        "target_id": "atorvastatin_delta_minus_18.01",
        "mixture": 3,
        "name_contains": "atorvastatin (Delta Mass:-18.01)",
        "precursor_mz": 541.249,
        "expected_endpoint_direction": "positive",
        "published_interpretation": "microbiota-associated transformation product",
    },
    {
        "target_id": "abacavir_delta_plus_31.99",
        "mixture": 4,
        "name_contains": "abacavir (Delta Mass:31.99)",
        "precursor_mz": 319.151,
        "expected_endpoint_direction": "positive",
        "published_interpretation": "microbiota-associated transformation product",
    },
    {
        "target_id": "omeprazole_product_mz330.127",
        "mixture": 3,
        "name_contains": "EsomeprazoleNa (Delta Mass:-15.995)",
        "precursor_mz": 330.127,
        "expected_endpoint_direction": "nonpositive",
        "published_interpretation": "abiotic transformation product",
    },
)


def library_targets(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    for column in ("#Scan#", "MQScore", "SharedPeaks", "Precursor_MZ"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.loc[
        table["#Scan#"].notna()
        & table["MQScore"].ge(0.9)
        & table["SharedPeaks"].ge(5)
    ].sort_values(["#Scan#", "MQScore"], ascending=[True, False]).drop_duplicates("#Scan#")
    rows = []
    for target in TARGETS:
        selected = table.loc[
            table["Compound_Name"].astype(str).str.contains(
                target["name_contains"], case=False, regex=False
            )
            & table["Precursor_MZ"].sub(target["precursor_mz"]).abs().le(0.01)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected exactly one library target for {target['target_id']}, found {len(selected)}"
            )
        row = selected.iloc[0]
        rows.append({
            **target,
            "entity_id": str(int(row["#Scan#"])),
            "library_compound_name": str(row["Compound_Name"]),
            "library_mqscore": float(row["MQScore"]),
            "library_shared_peaks": int(row["SharedPeaks"]),
            "observed_precursor_mz": float(row["Precursor_MZ"]),
        })
    return pd.DataFrame(rows)


def extract_effects(path: Path, targets: pd.DataFrame, estimand: str) -> pd.DataFrame:
    effects = pd.read_csv(path, sep="\t", dtype={"entity_id": str})
    selected = effects.merge(targets, on="entity_id", how="inner", validate="many_to_one")
    expected_prefix = selected["mixture"].map(lambda value: f"drug_mix_{int(value)}_")
    selected = selected.loc[
        [contrast.startswith(prefix) for contrast, prefix in zip(
            selected["contrast_id"].astype(str), expected_prefix
        )]
    ].copy()
    selected["estimand"] = estimand
    selected["context"] = selected["contrast_id"].str.replace(
        r"^drug_mix_[1-4]_", "", regex=True
    )
    return selected


def summarize(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_id, block in primary.groupby("target_id", sort=False):
        pooled = block.loc[block["context"].eq("pooled_microbiota")]
        if len(pooled) != 1:
            raise ValueError(f"missing pooled effect for {target_id}")
        pooled = pooled.iloc[0]
        local = block.loc[~block["context"].eq("pooled_microbiota")]
        sensitivity_pooled = sensitivity.loc[
            sensitivity["target_id"].eq(target_id)
            & sensitivity["context"].eq("pooled_microbiota")
        ]
        if len(local) != 4 or len(sensitivity_pooled) != 1:
            raise ValueError(f"incomplete context effects for {target_id}")
        expected = str(pooled["expected_endpoint_direction"])
        direction_recovered = (
            float(pooled["beta"]) > 0 if expected == "positive"
            else float(pooled["beta"]) <= 0
        )
        t_value = float(pooled["beta"] / pooled["se"])
        rows.append({
            "target_id": target_id,
            "entity_id": pooled["entity_id"],
            "library_compound_name": pooled["library_compound_name"],
            "published_interpretation": pooled["published_interpretation"],
            "expected_endpoint_direction": expected,
            "pooled_endpoint_beta": float(pooled["beta"]),
            "pooled_endpoint_se": float(pooled["se"]),
            "descriptive_t_df2_pvalue": float(2.0 * stats.t.sf(abs(t_value), 2)),
            "positive_community_contexts": int(local["beta"].gt(0).sum()),
            "negative_community_contexts": int(local["beta"].lt(0).sum()),
            "direction_recovered": bool(direction_recovered),
            "time_did_sensitivity_beta": float(sensitivity_pooled.iloc[0]["beta"]),
            "primary_sensitivity_same_sign": bool(
                np.sign(float(pooled["beta"]))
                == np.sign(float(sensitivity_pooled.iloc[0]["beta"]))
            ),
            "library_mqscore": float(pooled["library_mqscore"]),
            "library_shared_peaks": int(pooled["library_shared_peaks"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response-dir", type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_drug_endpoint_interaction_responses_v3_20261023"
        ),
    )
    parser.add_argument(
        "--library", type=Path,
        default=Path(
            "data/reference/msv000095331_drug_microbiome_20261023/"
            "library_matches.tsv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    targets = library_targets(args.library)
    primary = extract_effects(args.response_dir / "effects.tsv", targets, "endpoint_interaction")
    sensitivity = extract_effects(
        args.response_dir / "time_did_sensitivity_effects.tsv", targets, "time_did_sensitivity"
    )
    summary = summarize(primary, sensitivity)

    args.output_dir.mkdir(parents=True)
    targets.to_csv(args.output_dir / "prespecified_product_registry.tsv", sep="\t", index=False)
    primary.to_csv(args.output_dir / "product_context_effects.tsv", sep="\t", index=False)
    sensitivity.to_csv(
        args.output_dir / "product_time_did_sensitivity_effects.tsv", sep="\t", index=False
    )
    summary.to_csv(args.output_dir / "product_direction_summary.tsv", sep="\t", index=False)
    report = {
        "status": "MSV000095331_PUBLISHED_PRODUCT_DIRECTION_CHECK_COMPLETE",
        "data_role": "development; published targets opened only after response estimation",
        "dataset": "MSV000095331",
        "denominators": {
            "analysis_samples": 135,
            "prespecified_exact_product_targets_observable": int(len(summary)),
            "microbiota_associated_positive_targets": 2,
            "abiotic_nonpositive_targets": 1,
            "directions_recovered": int(summary["direction_recovered"].sum()),
        },
        "results": summary.to_dict("records"),
        "not_observable_as_exact_product_targets": [
            "ritonavir propagated analog", "quetiapine propagated analog"
        ],
        "claim_boundary": [
            "These are three prepublished product-ion direction checks, not an untargeted discovery count.",
            "Propagated-analog library labels are related-spectrum evidence, not level-1 structures.",
            "The four community designs are fixed biological contexts within one experiment, not independent studies.",
            "The df=2 t p-values are descriptive and are not used as a discovery or FDR claim.",
            "A mixture cannot attribute a product response to one constituent without targeted follow-up.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
