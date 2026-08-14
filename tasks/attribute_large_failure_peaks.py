"""Attribute identity and confounder peaks to auditable chemical evidence.

Discovery proposes signatures; confirmation evaluates the frozen list.  Only
the 335 sourced core rules are used.  Peak-level claims are limited to CF, NL,
ISO and HR rules whose numerical definitions can be mapped to a particular
peak.  Molecule-level parity rules are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


M_H = 1.0078250319


def parse(text: object, dtype=float) -> list:
    if pd.isna(text) or str(text).strip() == "":
        return []
    return [dtype(item) for item in str(text).split(";") if item != ""]


def load_rules(path: Path) -> list[dict]:
    package = json.loads(path.read_text(encoding="utf-8"))
    raw_rules = []
    for index, rule in enumerate(package["rules"]):
        if rule["category"] not in {"CF", "NL", "ISO", "HR"}:
            continue
        raw_rules.append({"index": index, **rule})
    # Collapse aliases that encode the same numerical event.  For example,
    # tropylium and amphetamine_frag share m/z 91.0542; signed HR labels are
    # also identical under the current |n_H| matcher.
    grouped: dict[tuple, list[dict]] = {}
    for rule in raw_rules:
        value = rule["value"]
        if isinstance(value, list):
            canonical_value = tuple(round(float(item), 5) for item in value)
        elif rule["category"] == "HR":
            canonical_value = round(abs(float(value)), 5)
        else:
            canonical_value = round(float(value), 5)
        grouped.setdefault(
            (rule["category"], rule["match_type"], canonical_value), []
        ).append(rule)
    concepts = []
    for (category, match_type, canonical_value), aliases in grouped.items():
        value = aliases[0]["value"]
        if category == "HR":
            value = float(canonical_value)
        concepts.append({
            "index": min(item["index"] for item in aliases),
            "name": " | ".join(sorted({item["name"] for item in aliases})),
            "category": category, "match_type": match_type, "value": value,
            "source": " | ".join(sorted({item.get("source", "") for item in aliases})),
            "alias_count": len(aliases), "canonical_value": canonical_value,
        })
    return concepts


def bin_label(value: float, edges: list[float], labels: list[str]) -> str:
    position = int(np.searchsorted(edges, value, side="right"))
    return labels[position]


def peak_features(
    mz: float, intensity: float, precursor: float, all_mz: np.ndarray,
    rules: list[dict], tolerance: float,
) -> list[tuple[str, str, str, str]]:
    output: list[tuple[str, str, str, str]] = []
    neutral_loss = precursor - mz
    matched_categories: set[str] = set()
    for rule in rules:
        category, match_type, value = rule["category"], rule["match_type"], rule["value"]
        matched = False
        if category == "CF" and match_type == "peak_mz":
            matched = abs(mz - float(value)) <= tolerance
        elif category == "NL" and match_type == "mass_diff":
            matched = neutral_loss > 0 and abs(neutral_loss - float(value)) <= tolerance
        elif category == "ISO" and match_type == "mass_range":
            lo, hi = map(float, value)
            differences = np.abs(all_mz - mz)
            matched = bool(np.any((differences >= lo) & (differences <= hi) & (differences > 0)))
        elif category == "HR" and match_type == "hr_shift":
            n_h = float(value)
            differences = np.abs(all_mz - mz)
            if n_h == 0:
                matched = bool(np.any(
                    (differences >= 12.0)
                    & (np.abs(differences - np.rint(differences)) <= tolerance)
                ))
            else:
                matched = bool(np.any(np.abs(differences - abs(n_h) * M_H) <= tolerance))
        if matched:
            output.append((
                f"CONCEPT::{category}::{rule['canonical_value']}", category,
                rule["name"], rule.get("source", ""),
            ))
            matched_categories.add(category)
    for category in sorted(matched_categories):
        output.append((f"CATEGORY::{category}", category, f"Any {category} rule", "core_rule_aggregate"))

    ratio = mz / precursor if precursor > 0 else math.nan
    output.extend([
        (f"BIN::relative_mz::{bin_label(ratio, [0.25, 0.5, 0.75], ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0+'])}",
         "relative_mz", "relative fragment m/z", "predefined_bin"),
        (f"BIN::intensity::{bin_label(intensity, [0.01, 0.05, 0.2], ['<1%', '1-5%', '5-20%', '>=20%'])}",
         "intensity", "relative peak intensity", "predefined_bin"),
    ])
    if neutral_loss > 0:
        output.append((
            f"BIN::neutral_loss::{bin_label(neutral_loss, [20, 50, 100, 200], ['0-20', '20-50', '50-100', '100-200', '>=200'])}",
            "neutral_loss", "neutral loss magnitude", "predefined_bin",
        ))
    return output


def expand_peaks(frame: pd.DataFrame, rules: list[dict], tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    peaks_out, features_out = [], []
    for row in frame.itertuples(index=False):
        all_mz = []
        for category in ("identity_support", "confounder_support", "shared", "unassigned"):
            all_mz.extend(parse(getattr(row, f"fragment_{category}_mz")))
        all_mz_array = np.asarray(all_mz, float)
        for evidence in ("identity", "confounder"):
            prefix = "identity_support" if evidence == "identity" else "confounder_support"
            mz_values = parse(getattr(row, f"fragment_{prefix}_mz"))
            intensities = parse(getattr(row, f"fragment_{prefix}_intensity"))
            if len(mz_values) != len(intensities):
                raise RuntimeError(f"Peak metadata mismatch for {row.split}:{row.query_index}")
            for local_index, (mz, intensity) in enumerate(zip(mz_values, intensities)):
                peak_id = f"{row.split}:{row.query_index}:{evidence}:{local_index}"
                base = {
                    "peak_id": peak_id, "split": row.split, "query_index": int(row.query_index),
                    "ik14": row.ik14, "formula": row.formula, "ring_class": row.ring_class,
                    "audit_quadrant": row.audit_quadrant, "evidence": evidence,
                    "mz": mz, "neutral_loss": float(row.query_precursor_mz) - mz,
                    "intensity": intensity, "relative_mz": mz / float(row.query_precursor_mz),
                }
                peaks_out.append(base)
                for feature_id, family, name, source in peak_features(
                    mz, intensity, float(row.query_precursor_mz), all_mz_array, rules, tolerance,
                ):
                    features_out.append(base | {
                        "feature_id": feature_id, "feature_family": family,
                        "feature_name": name, "feature_source": source,
                    })
    return pd.DataFrame(peaks_out), pd.DataFrame(features_out)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    if not len(p_values):
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.clip(ranked, 0, 1)
    return output


def paired_feature_stats(features: pd.DataFrame, peaks: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = peaks.loc[peaks["split"] == split]
    counts = subset.groupby(["query_index", "evidence"]).size().unstack(fill_value=0)
    eligible_queries = counts.index[(counts.get("identity", 0) > 0) & (counts.get("confounder", 0) > 0)]
    feature_subset = features.loc[
        (features["split"] == split) & features["query_index"].isin(eligible_queries)
    ]
    metadata = feature_subset.drop_duplicates("feature_id").set_index("feature_id")
    hits = feature_subset.drop_duplicates(["query_index", "evidence", "feature_id"])
    hit_set = set(zip(hits["query_index"], hits["evidence"], hits["feature_id"]))
    rows = []
    for feature_id in sorted(feature_subset["feature_id"].unique()):
        identity = np.asarray([(q, "identity", feature_id) in hit_set for q in eligible_queries], bool)
        confounder = np.asarray([(q, "confounder", feature_id) in hit_set for q in eligible_queries], bool)
        identity_only = int(np.sum(identity & ~confounder))
        confounder_only = int(np.sum(~identity & confounder))
        discordant = identity_only + confounder_only
        p_value = float(binomtest(min(identity_only, confounder_only), discordant, 0.5).pvalue) if discordant else 1.0
        info = metadata.loc[feature_id]
        rows.append({
            "split": split, "feature_id": feature_id,
            "feature_family": info["feature_family"], "feature_name": info["feature_name"],
            "feature_source": info["feature_source"], "paired_queries": len(eligible_queries),
            "identity_query_fraction": float(identity.mean()),
            "confounder_query_fraction": float(confounder.mean()),
            "identity_minus_confounder": float(identity.mean() - confounder.mean()),
            "identity_only_queries": identity_only, "confounder_only_queries": confounder_only,
            "discordant_queries": discordant, "mcnemar_exact_p": p_value,
            "total_query_support": int(np.sum(identity | confounder)),
        })
    output = pd.DataFrame(rows)
    output["fdr_bh"] = bh_adjust(output["mcnemar_exact_p"].to_numpy(float))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/validation/large_all_failure_peak_localization/all_peak_evidence.csv"))
    parser.add_argument("--rules", type=Path, default=Path("dreams/models/chem_aware/chem_rules_data.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_failure_peak_chemical_attribution"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-discovery-support", type=int, default=25)
    parser.add_argument("--min-effect", type=float, default=0.03)
    parser.add_argument("--discovery-fdr", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input)
    rules = load_rules(args.rules)
    peaks, features = expand_peaks(source, rules, args.tolerance)
    peaks.to_csv(args.output_dir / "peak_table.csv", index=False)
    features.to_csv(args.output_dir / "peak_feature_hits.csv", index=False)

    discovery = paired_feature_stats(features, peaks, "discovery")
    confirmation = paired_feature_stats(features, peaks, "confirmation")
    discovery.to_csv(args.output_dir / "discovery_feature_stats.csv", index=False)
    confirmation.to_csv(args.output_dir / "confirmation_feature_stats.csv", index=False)
    selected = discovery.loc[
        (discovery["total_query_support"] >= args.min_discovery_support)
        & (discovery["identity_minus_confounder"].abs() >= args.min_effect)
        & (discovery["fdr_bh"] <= args.discovery_fdr)
    ].copy()
    frozen = selected[[
        "feature_id", "feature_family", "feature_name", "feature_source",
        "identity_minus_confounder", "fdr_bh", "total_query_support",
    ]].rename(columns={
        "identity_minus_confounder": "discovery_effect", "fdr_bh": "discovery_fdr",
        "total_query_support": "discovery_support",
    })
    frozen["discovery_direction"] = np.sign(frozen["discovery_effect"]).astype(int)
    validation = frozen.merge(confirmation, on="feature_id", how="left", suffixes=("", "_confirmation"))
    validation["direction_replicated"] = (
        np.sign(validation["identity_minus_confounder"]).fillna(0).astype(int)
        == validation["discovery_direction"]
    )
    validation["confirmation_nominal_p05"] = validation["mcnemar_exact_p"] <= 0.05
    validation["replicated_panel"] = (
        validation["direction_replicated"]
        & validation["confirmation_nominal_p05"]
        & (validation["total_query_support"] >= 10)
    )
    validation.to_csv(args.output_dir / "discovery_to_confirmation_validation.csv", index=False)
    validation.loc[validation["replicated_panel"]].to_csv(args.output_dir / "replicated_evidence_panel.csv", index=False)
    report = {
        "status": "large_failure_peak_chemical_attribution",
        "input_failure_queries": int(source[["split", "query_index"]].drop_duplicates().shape[0]),
        "input_failure_molecules": int(source["ik14"].nunique()),
        "input_failure_formulas": int(source["formula"].nunique()),
        "peak_rows": len(peaks), "feature_hit_rows": len(features),
        "core_rules_loaded": 335,
        "peak_attributable_rule_concepts_after_alias_collapse": len(rules),
        "excluded_from_peak_claims": ["NR parity", "EE molecule-level rule", "3151 MassBank empirical masses"],
        "discovery_features_tested": len(discovery),
        "discovery_features_frozen": len(validation),
        "confirmation_direction_replicated": int(validation["direction_replicated"].sum()),
        "confirmation_nominal_p05": int(validation["confirmation_nominal_p05"].sum()),
        "replicated_panel_size": int(validation["replicated_panel"].sum()),
        "claim_limit": "rule match supplies a mechanistic hypothesis, not structure-level fragment annotation",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["replicated_panel_size"]:
        print(validation.loc[validation["replicated_panel"], [
            "feature_id", "discovery_effect", "identity_minus_confounder", "mcnemar_exact_p"
        ]].sort_values("discovery_effect").to_string(index=False))


if __name__ == "__main__":
    main()
