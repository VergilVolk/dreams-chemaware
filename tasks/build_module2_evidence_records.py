"""Materialize module-2 peak evidence from the frozen causal audits.

The output is an auditable record, not a new prediction model.  Each directed
pair contains the peaks that were actually removed, their matched candidate
factors/rules, and the effect relative to intensity/mass-matched random peak
deletion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from run_residual_pair_peak_occlusion import (
    matched_and_unique_mz,
    select_target_subset,
    target_tokens,
)
from train_e1_identity import preprocess_spectrum


ROOT = Path(__file__).resolve().parent.parent


def load_rules() -> list[dict]:
    rules = []
    for tier, path in (
        ("core", ROOT / "dreams/models/chem_aware/chem_rules_data.json"),
        ("massbank", ROOT / "dreams/models/chem_aware/chem_rules_massbank.json"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rule in payload["rules"]:
            item = dict(rule)
            item["tier"] = tier if tier == "core" else item.get("tier", "extended")
            rules.append(item)
    return rules


def nearby_rules(rules: list[dict], peak: float, loss: float, tolerance: float) -> list[dict]:
    core_matches = []
    massbank_matches = []
    for rule in rules:
        kind = rule.get("match_type")
        value = rule.get("value")
        observed = peak if kind == "peak_mz" else loss if kind == "mass_diff" else None
        if observed is None or not isinstance(value, (int, float)):
            continue
        error = abs(float(value) - observed)
        if error <= tolerance:
            item = {
                "name": rule.get("name"), "category": rule.get("category"),
                "tier": rule.get("tier"), "source": rule.get("source"),
                "mass_error_da": error, "support": rule.get("support"),
                "alias_group": rule.get("alias_group"),
                "scope": rule.get("scope"),
            }
            (core_matches if rule.get("tier") == "core" else massbank_matches).append(item)
    core_matches.sort(key=lambda x: x["mass_error_da"])
    grouped: dict[str, list[dict]] = {}
    for item in massbank_matches:
        key = item.get("alias_group") or f"{item['category']}:{round(peak, 2):.2f}"
        grouped.setdefault(str(key), []).append(item)
    empirical_groups = []
    for key, items in grouped.items():
        items.sort(key=lambda x: x["mass_error_da"])
        empirical_groups.append({
            "name": key,
            "category": items[0]["category"],
            "tier": "massbank_empirical_group",
            "source": "MassBank record-derived cluster",
            "mass_error_da": items[0]["mass_error_da"],
            "record_count": len(items),
            "representative_records": [item["name"] for item in items[:3]],
            "scopes": sorted({str(item["scope"]) for item in items if item.get("scope")})[:3],
        })
    empirical_groups.sort(key=lambda x: (-x["record_count"], x["mass_error_da"]))
    return core_matches[:5] + empirical_groups[:3]


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def sanitize_json(value):
    """Convert pandas/numpy missing scalars into strict JSON null values."""
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def nearby_factors(factors: pd.DataFrame, peak: float, loss: float, tolerance: float) -> list[dict]:
    output = []
    for row in factors.itertuples(index=False):
        observed = peak if row.spectral_kind == "fragment_mz" else loss
        error = abs(float(row.mass_da) - observed)
        if error <= tolerance:
            output.append({
                "factor": int(row.factor), "kind": row.spectral_kind,
                "mass_da": float(row.mass_da), "mass_error_da": error,
                "support_tier": row.support_tier,
                "structure_context_replicated": bool(row.structure_context_replicated),
                "structure_environment": (
                    None if pd.isna(row.structure_environment) else row.structure_environment
                ),
            })
    return output


def build_split(
    split: str, effects_path: Path, manifest_path: Path, data_path: Path,
    factors: pd.DataFrame, rules: list[dict], output_dir: Path,
    fragment_tolerance: float, token_tolerance: float, rule_tolerance: float,
) -> dict:
    effects = pd.read_csv(effects_path)
    manifest = pd.read_csv(manifest_path)
    required_manifest_rows = np.unique(np.concatenate([
        effects["source_index"].to_numpy(np.int64),
        effects["target_index"].to_numpy(np.int64),
    ]))
    required_hdf = manifest.loc[required_manifest_rows, "hdf5_row"].to_numpy(np.int64)
    with h5py.File(data_path, "r") as handle:
        loaded = np.asarray(handle["spectrum"][np.sort(np.unique(required_hdf))])
    sorted_rows = np.sort(np.unique(required_hdf))
    spectra = {int(row): loaded[i] for i, row in enumerate(sorted_rows)}

    records = []
    flat_rows = []
    for row in effects.itertuples(index=False):
        source_meta = manifest.iloc[int(row.source_index)]
        target_meta = manifest.iloc[int(row.target_index)]
        source_hdf, target_hdf = int(source_meta.hdf5_row), int(target_meta.hdf5_row)
        raw_source, raw_target = spectra[source_hdf], spectra[target_hdf]
        shared, unique = matched_and_unique_mz(raw_source, raw_target, fragment_tolerance)
        target_mz = shared if row.target_mode == "shared" else unique
        clean = preprocess_spectrum(raw_source, float(source_meta.precursor_mz), 100)
        all_tokens = target_tokens(clean, target_mz, token_tolerance)
        selected, _ = select_target_subset(clean, all_tokens, 12)
        peaks = []
        for token in selected:
            peak_mz = float(clean[int(token), 0])
            intensity = float(clean[int(token), 1])
            neutral_loss = float(source_meta.precursor_mz) - peak_mz
            evidence = {
                "peak_mz": peak_mz, "relative_intensity": intensity,
                "neutral_loss_da": neutral_loss,
                "candidate_factors": nearby_factors(factors, peak_mz, neutral_loss, rule_tolerance),
                "matched_rules": nearby_rules(rules, peak_mz, neutral_loss, rule_tolerance),
            }
            peaks.append(evidence)
            flat_rows.append({
                "split": split, "pair_key": row.pair_key, "source_side": row.source_side,
                "mechanism": row.mechanism_screen, "source_ik14": source_meta.ik14,
                "target_ik14": target_meta.ik14, "peak_mz": peak_mz,
                "relative_intensity": intensity, "neutral_loss_da": neutral_loss,
                "directional_support": float(row.directional_support),
                "factor_ids": ";".join(str(x["factor"]) for x in evidence["candidate_factors"]),
                "rule_names": ";".join(str(x["name"]) for x in evidence["matched_rules"]),
            })
        record = {
            "split": split, "pair_key": row.pair_key, "pair_type": row.pair_type,
            "mechanism": row.mechanism_screen, "source_side": row.source_side,
            "source": {
                "ik14": source_meta.ik14, "formula": source_meta.formula,
                "smiles": source_meta.smiles, "instrument": source_meta.instrument,
                "collision_energy": json_value(source_meta.collision_energy),
                "precursor_mz": float(source_meta.precursor_mz), "hdf5_row": source_hdf,
            },
            "target": {
                "ik14": target_meta.ik14, "formula": target_meta.formula,
                "smiles": target_meta.smiles, "instrument": target_meta.instrument,
                "collision_energy": json_value(target_meta.collision_energy),
                "precursor_mz": float(target_meta.precursor_mz), "hdf5_row": target_hdf,
            },
            "clean_similarity": float(row.clean_pair_cosine),
            "targeted_similarity_change": float(row.cosine_change),
            "matched_random_similarity_change": float(row.random_cosine_change),
            "directional_support": float(row.directional_support),
            "evidence_interpretation": (
                "shared peaks causally support a misleading cross-molecule similarity"
                if row.target_mode == "shared" else
                "condition-specific peaks causally support an unstable same-molecule separation"
            ),
            "removed_peak_evidence": peaks,
        }
        records.append(record)

    jsonl = output_dir / f"{split}_evidence_records.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(
                sanitize_json(record), ensure_ascii=False, allow_nan=False,
            ) + "\n")
    pd.DataFrame(flat_rows).to_csv(output_dir / f"{split}_peak_evidence.csv", index=False)
    return {
        "directed_pair_records": len(records), "peak_evidence_rows": len(flat_rows),
        "pairs_with_factor_match": int(sum(any(p["candidate_factors"] for p in r["removed_peak_evidence"]) for r in records)),
        "pairs_with_rule_match": int(sum(any(p["matched_rules"] for p in r["removed_peak_evidence"]) for r in records)),
        "jsonl": str(jsonl.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/module2_evidence_records")
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--token-tolerance", type=float, default=0.005)
    parser.add_argument("--annotation-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factors = pd.read_csv(
        ROOT / "data/validation/spectral_first_fragmentation_factor_pilot/validated_factor_catalog.csv"
    )
    factors["structure_context_replicated"] = (
        factors["structure_context_replicated"].astype(str).str.lower().eq("true")
    )
    rules = load_rules()
    configs = {
        "discovery": (
            ROOT / "data/validation/dreams_residual_pair_occlusion_discovery_v2/paired_effects.csv",
            ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
        ),
        "confirmation": (
            ROOT / "data/validation/dreams_residual_pair_occlusion_confirmation_v2/paired_effects.csv",
            ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
        ),
    }
    report = {
        "status": "module2_evidence_materialized",
        "scope": "frozen official DreaMS residual peak interventions; no model retraining",
        "claim_boundary": "factor/rule mass matches annotate evidence; they do not prove a unique fragment structure or bond-breaking mechanism",
        "splits": {},
    }
    for split, (effects, manifest) in configs.items():
        report["splits"][split] = build_split(
            split, effects, manifest, args.data, factors, rules, args.output_dir,
            args.fragment_tolerance, args.token_tolerance, args.annotation_tolerance,
        )
        print(json.dumps({split: report["splits"][split]}, ensure_ascii=False, indent=2))
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
