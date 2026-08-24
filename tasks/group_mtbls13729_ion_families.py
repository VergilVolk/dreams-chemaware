"""Group provisional MS1 ions into adduct/charge families with OpenMS.

This prevents isotopes, charge states and adducts from being counted as
independent metabolites.  The family assignments are hypotheses and remain
separate from spectral-library identification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


def meta_dict(item: object) -> dict[str, object]:
    keys: list[bytes] = []
    item.getKeys(keys)
    result: dict[str, object] = {}
    for key in keys:
        name = key.decode() if isinstance(key, bytes) else str(key)
        value = item.getMetaValue(key)
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        elif isinstance(value, np.generic):
            value = value.item()
        result[name] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="pos_rp")
    parser.add_argument(
        "--candidate-table",
        type=Path,
        default=Path("data/mtbls13729/ms1_paired_analysis/pos_rp__discovery_priority_features.csv"),
    )
    parser.add_argument(
        "--eic-matrix",
        type=Path,
        default=Path("data/mtbls13729/ms1_eic_requant/pos_rp__eic_auc_matrix.csv.gz"),
    )
    parser.add_argument("--max-charge", type=int, default=6)
    parser.add_argument("--rt-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--mass-tolerance-ppm", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mtbls13729/ion_families")
    )
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidate_table)
    auc = pd.read_csv(args.eic_matrix).set_index("feature_id")
    fmap = oms.FeatureMap()
    unique_to_feature: dict[int, int] = {}
    for row in candidates.itertuples(index=False):
        feature_id = int(row.feature_id)
        feature = oms.Feature()
        feature.setMZ(float(row.mz))
        feature.setRT(float(row.rt_sec))
        values = auc.loc[feature_id].to_numpy(float) if feature_id in auc.index else np.asarray([])
        values = values[np.isfinite(values) & (values > 0)]
        feature.setIntensity(float(np.median(values)) if len(values) else 1.0)
        feature.setCharge(0)
        feature.setMetaValue("original_feature_id", feature_id)
        feature.ensureUniqueId()
        unique_to_feature[int(feature.getUniqueId())] = feature_id
        fmap.push_back(feature)
    fmap.setUniqueIds()

    algorithm = oms.MetaboliteFeatureDeconvolution()
    params = algorithm.getDefaults()
    params.setValue("charge_min", 1)
    params.setValue("charge_max", int(args.max_charge))
    params.setValue("charge_span_max", int(args.max_charge))
    params.setValue("q_try", "all")
    params.setValue("retention_max_diff", float(args.rt_tolerance_sec))
    params.setValue("retention_max_diff_local", float(args.rt_tolerance_sec))
    params.setValue("mass_max_diff", float(args.mass_tolerance_ppm))
    params.setValue("unit", "ppm")
    if args.panel.startswith("pos"):
        params.setValue("negative_mode", "false")
        params.setValue(
            "potential_adducts",
            ["H:+:0.6", "Na:+:0.2", "NH4:+:0.15", "K:+:0.05", "H-2O-1:0:0.05"],
        )
    else:
        params.setValue("negative_mode", "true")
        params.setValue("potential_adducts", ["H-1:-:0.9", "Cl:-:0.1", "CH2O2:0:0.05"])
    algorithm.setParameters(params)

    decharged = oms.FeatureMap()
    groups = oms.ConsensusMap()
    pairs = oms.ConsensusMap()
    algorithm.compute(fmap, decharged, groups, pairs)

    decharged_rows = []
    for index, feature in enumerate(decharged):
        decharged_rows.append(
            {
                "decharged_index": index,
                "neutral_mz": float(feature.getMZ()),
                "rt_sec": float(feature.getRT()),
                "intensity": float(feature.getIntensity()),
                "charge": int(feature.getCharge()),
                **meta_dict(feature),
            }
        )

    decharged_frame = pd.DataFrame(decharged_rows)
    # OpenMS preserves the input feature id and writes a common ``Group`` meta
    # value for ions assigned to the same neutral analyte hypothesis.  This is
    # more reliable in pyOpenMS than mapping ConsensusFeature handle IDs, which
    # may be regenerated during the ILP step.
    if "Group" not in decharged_frame:
        decharged_frame["Group"] = decharged_frame["original_feature_id"].map(lambda value: f"singleton:{value}")
    group_key = decharged_frame["Group"].astype("string")
    missing_group = group_key.isna()
    group_key.loc[missing_group] = decharged_frame.loc[missing_group, "original_feature_id"].map(
        lambda value: f"singleton:{value}"
    )
    decharged_frame["family_id"] = pd.factorize(group_key, sort=False)[0]
    family_sizes = decharged_frame.groupby("family_id")["original_feature_id"].transform("size")
    family_frame = decharged_frame[
        ["family_id", "original_feature_id", "neutral_mz", "rt_sec", "charge", "dc_charge_adducts", "adducts"]
    ].rename(
        columns={
            "original_feature_id": "feature_id",
            "neutral_mz": "family_mz",
            "rt_sec": "family_rt_sec",
            "charge": "inferred_charge",
        }
    )
    family_frame.insert(2, "family_size", family_sizes.astype(int).to_numpy())
    family_frame = family_frame.sort_values(["family_id", "feature_id"])

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    family_path = out / f"{args.panel}__candidate_ion_families.csv"
    decharged_path = out / f"{args.panel}__candidate_decharged_features.csv"
    family_frame.to_csv(family_path, index=False)
    decharged_frame.to_csv(decharged_path, index=False)
    report = {
        "status": "complete",
        "panel": args.panel,
        "n_input_ions": int(len(candidates)),
        "n_families": int(family_frame.family_id.nunique()),
        "n_multimember_families": int((family_frame.groupby("family_id").size() > 1).sum()),
        "n_decharged_features": int(len(decharged_frame)),
        "parameters": {
            "max_charge": args.max_charge,
            "rt_tolerance_sec": args.rt_tolerance_sec,
            "mass_tolerance_ppm": args.mass_tolerance_ppm,
        },
        "family_table": str(family_path),
        "decharged_table": str(decharged_path),
        "interpretation_limit": "Ion-family grouping is not molecular identification.",
    }
    (out / f"{args.panel}__report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
