"""Link decodable molecular environments to decodable spectrum concepts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact

from train_frozen_structure_environment_probe import scaffold_split


ROOT = Path(__file__).resolve().parent.parent


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def table_and_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    odds_ratio, p = fisher_exact([[n11, n10], [n01, n00]], alternative="greater")
    return float(odds_ratio), float(p), n11, int(a.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure-data", type=Path,
        default=ROOT / "data/validation/double_mapping/structure_environment_probe_data.npz",
    )
    parser.add_argument(
        "--structure-probe-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/frozen_structure_probe",
    )
    parser.add_argument(
        "--spectrum-labels", type=Path,
        default=ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz",
    )
    parser.add_argument(
        "--concept-probe-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/frozen_concept_probe",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/structure_spectrum_links",
    )
    parser.add_argument("--minimum-observation-rate", type=float, default=0.5)
    parser.add_argument("--minimum-odds-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structure = np.load(args.structure_data, allow_pickle=False)
    struct_ik = structure["ik14"].astype(str)
    struct_labels = structure["labels"].astype(bool)
    environments = structure["environment"].astype(str)
    scaffolds = structure["scaffold"].astype(str)
    train_idx, val_idx, test_idx = scaffold_split(scaffolds, args.seed)
    discovery_idx = np.concatenate([train_idx, val_idx])

    spectrum = np.load(args.spectrum_labels, allow_pickle=False)
    spectrum_ik = spectrum["ik14"].astype(str)
    spectrum_labels = spectrum["labels"].astype(np.float32)
    accumulator: dict[str, list[np.ndarray]] = defaultdict(list)
    for molecule, row in zip(spectrum_ik.tolist(), spectrum_labels):
        accumulator[molecule].append(row)
    molecule_concepts = np.zeros((len(struct_ik), spectrum_labels.shape[1]), dtype=bool)
    missing = 0
    for row, molecule in enumerate(struct_ik.tolist()):
        values = accumulator.get(molecule)
        if not values:
            missing += 1
            continue
        molecule_concepts[row] = np.mean(values, axis=0) >= args.minimum_observation_rate
    if missing:
        raise RuntimeError(f"{missing} structure molecules lack aligned spectrum concepts")

    struct_metrics = pd.read_csv(args.structure_probe_dir / "per_environment_metrics.csv")
    struct_metrics = struct_metrics.loc[
        struct_metrics["test_positive_molecules"].ge(20)
        & struct_metrics["auprc_lift"].ge(2)
    ]
    concept_metrics = pd.read_csv(args.concept_probe_dir / "per_rule_metrics.csv")
    concept_metrics = concept_metrics.loc[
        concept_metrics["test_auprc"].ge(0.60)
        & concept_metrics["auprc_lift"].ge(2)
    ]

    struct_probe = torch.load(
        args.structure_probe_dir / "structure_probe.pt", map_location="cpu", weights_only=True
    )
    concept_probe = torch.load(
        args.concept_probe_dir / "concept_probe.pt", map_location="cpu", weights_only=True
    )
    struct_weight = struct_probe["state_dict"]["weight"].numpy() / struct_probe["embedding_std"].numpy()[None, :]
    concept_weight = concept_probe["state_dict"]["weight"].numpy() / concept_probe["embedding_std"].numpy()[None, :]
    struct_weight /= np.linalg.norm(struct_weight, axis=1, keepdims=True).clip(min=1e-12)
    concept_weight /= np.linalg.norm(concept_weight, axis=1, keepdims=True).clip(min=1e-12)

    rules = json.loads(
        (ROOT / "dreams/models/chem_aware/chem_rules_data.json").read_text(encoding="utf-8")
    )["rules"]
    rows = []
    for env_row in struct_metrics.itertuples(index=False):
        environment_index = int(env_row.environment_index)
        env_discovery = struct_labels[discovery_idx, environment_index]
        env_confirmation = struct_labels[test_idx, environment_index]
        for concept_row in concept_metrics.itertuples(index=False):
            rule_index = int(concept_row.rule_index)
            rule_discovery = molecule_concepts[discovery_idx, rule_index]
            rule_confirmation = molecule_concepts[test_idx, rule_index]
            d_or, d_p, d_both, d_env = table_and_test(env_discovery, rule_discovery)
            c_or, c_p, c_both, c_env = table_and_test(env_confirmation, rule_confirmation)
            rows.append({
                "environment": environments[environment_index],
                "environment_index": environment_index,
                "environment_probe_auprc": float(env_row.test_auprc),
                "rule_name": concept_row.rule_name,
                "rule_index": rule_index,
                "rule_category": concept_row.category,
                "rule_match_type": rules[rule_index]["match_type"],
                "rule_value": json.dumps(rules[rule_index]["value"], ensure_ascii=False),
                "concept_probe_auprc": float(concept_row.test_auprc),
                "discovery_odds_ratio": d_or,
                "discovery_p": d_p,
                "discovery_both": d_both,
                "discovery_environment_positive": d_env,
                "confirmation_odds_ratio": c_or,
                "confirmation_p": c_p,
                "confirmation_both": c_both,
                "confirmation_environment_positive": c_env,
                "embedding_direction_cosine": float(
                    struct_weight[int(env_row.probe_output_index)]
                    @ concept_weight[int(concept_row.probe_output_index)]
                ),
            })
    frame = pd.DataFrame(rows)
    frame["discovery_bh_q"] = bh_adjust(frame["discovery_p"].to_numpy(float))
    frame["confirmation_bh_q"] = bh_adjust(frame["confirmation_p"].to_numpy(float))
    frame["replicated_link"] = (
        frame["discovery_bh_q"].le(0.05)
        & frame["confirmation_bh_q"].le(0.05)
        & frame["discovery_odds_ratio"].ge(args.minimum_odds_ratio)
        & frame["confirmation_odds_ratio"].ge(args.minimum_odds_ratio)
        & frame["confirmation_both"].ge(5)
    )
    cosine_threshold = float(np.quantile(frame["embedding_direction_cosine"], 0.99))
    frame["embedding_aligned_replicated_link"] = (
        frame["replicated_link"]
        & frame["embedding_direction_cosine"].ge(cosine_threshold)
    )
    frame = frame.sort_values(
        ["replicated_link", "confirmation_bh_q", "confirmation_odds_ratio"],
        ascending=[False, True, False],
    )
    frame.to_csv(args.output_dir / "all_structure_spectrum_links.csv", index=False)
    replicated = frame.loc[frame["replicated_link"]].copy()
    replicated.to_csv(args.output_dir / "replicated_structure_spectrum_links.csv", index=False)
    aligned = frame.loc[frame["embedding_aligned_replicated_link"]].copy()
    aligned.to_csv(args.output_dir / "embedding_aligned_replicated_links.csv", index=False)
    report = {
        "status": "structure_spectrum_linking_complete",
        "molecules": len(struct_ik),
        "spectral_concept_aggregation": (
            f"rule observed in at least {args.minimum_observation_rate:.0%} of a molecule's spectra"
        ),
        "candidate_structure_environments": int(struct_metrics.shape[0]),
        "candidate_spectral_concepts": int(concept_metrics.shape[0]),
        "pairs_tested": int(len(frame)),
        "replicated_links": int(len(replicated)),
        "embedding_direction_cosine_threshold": cosine_threshold,
        "embedding_aligned_replicated_links": int(len(aligned)),
        "replication_rule": (
            "BH q<=0.05 and odds ratio>=2 in discovery and scaffold-isolated confirmation; "
            "at least 5 joint-positive confirmation molecules"
        ),
        "unique_linked_environments": int(replicated["environment"].nunique()),
        "unique_linked_spectral_concepts": int(replicated["rule_name"].nunique()),
        "unique_aligned_environments": int(aligned["environment"].nunique()),
        "unique_aligned_spectral_concepts": int(aligned["rule_name"].nunique()),
        "claim_limit": (
            "Population association and aligned probe directions prioritize mappings; "
            "peak deletion is still required for per-spectrum causal explanation."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(replicated):
        print(replicated.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
