"""Mine cross-domain ChemAware actions without pooling raw spectra.

For each parent predicate and observation channel, contrasts are computed
inside formula and acquisition domain first.  Domain contrasts are averaged
within formula, after which formulas receive equal statistical mass.  Rules
must replicate across multiple domains and a formula-disjoint confirmation
fold.  This recovers power lost by isolated domain cells without allowing
instrument, adduct, or collision-energy composition to create an association.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_direct_action_core import formula_bootstrap
from mine_chemaware_structure_conditioned_rules import (
    bh_qvalues,
    matched_formula_differences,
    sha256_file,
    sign_flip_pvalue,
)

FORMULA_PATTERN = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-rule-dir", type=Path, required=True)
    parser.add_argument("--predicates", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-folds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--confirmation-fold", type=int, default=2)
    parser.add_argument("--minimum-formulas", type=int, default=12)
    parser.add_argument("--minimum-positive-identities", type=int, default=24)
    parser.add_argument("--minimum-domain-formulas", type=int, default=3)
    parser.add_argument("--minimum-replicating-domains", type=int, default=2)
    parser.add_argument("--minimum-domain-nonnegative-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-discovery-effect", type=float, default=0.05)
    parser.add_argument("--minimum-confirmation-effect", type=float, default=0.03)
    parser.add_argument("--minimum-domain-effect", type=float, default=0.0)
    parser.add_argument("--max-channels-per-predicate", type=int, default=2)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def stratified_formula_contrasts(
    observation: np.ndarray,
    predicate: np.ndarray,
    formula: np.ndarray,
    identity: np.ndarray,
    domain: np.ndarray,
    fold: np.ndarray,
    valid: np.ndarray,
    selected_folds: tuple[int, ...],
    eligible_domains: tuple[str, ...],
) -> dict:
    """Build one equal-formula matrix after within-domain comparisons."""
    by_formula: dict[str, list[np.ndarray]] = defaultdict(list)
    domain_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    matched_positions: list[np.ndarray] = []
    for domain_value in eligible_domains:
        positions = np.flatnonzero(
            (domain == domain_value)
            & np.isin(fold, selected_folds)
            & np.asarray(valid, dtype=bool)
        )
        groups, difference, _, _ = matched_formula_differences(
            observation, predicate, formula, positions
        )
        if not len(groups):
            continue
        domain_values[str(domain_value)] = (groups.astype(str), difference)
        matched = positions[np.isin(formula[positions], groups)]
        matched_positions.append(matched)
        for group, value in zip(groups.astype(str), difference):
            by_formula[str(group)].append(value)
    formulas = np.asarray(sorted(by_formula))
    matrix = (
        np.vstack([np.mean(by_formula[value], axis=0) for value in formulas])
        if len(formulas)
        else np.empty((0, observation.shape[1]), dtype=np.float64)
    )
    positions = (
        np.unique(np.concatenate(matched_positions))
        if matched_positions
        else np.empty(0, dtype=np.int64)
    )
    positive = len(np.unique(identity[positions[predicate[positions]]]))
    negative = len(np.unique(identity[positions[~predicate[positions]]]))
    return {
        "formulas": formulas,
        "matrix": matrix,
        "positive_identities": int(positive),
        "negative_identities": int(negative),
        "domain_values": domain_values,
    }


def domain_replication(
    evidence: dict,
    channel: int,
    minimum_domain_formulas: int,
    minimum_domain_effect: float,
) -> dict:
    effects = []
    details = []
    for domain, (formulas, matrix) in sorted(evidence["domain_values"].items()):
        if len(formulas) < minimum_domain_formulas:
            continue
        effect = float(np.mean(matrix[:, channel]))
        effects.append(effect)
        details.append(
            {"domain": domain, "formulas": int(len(formulas)), "effect": effect}
        )
    values = np.asarray(effects, dtype=np.float64)
    return {
        "evaluable_domains": int(len(values)),
        "replicating_domains": int(np.sum(values >= minimum_domain_effect)),
        "positive_domains": int(np.sum(values > 0)),
        "nonnegative_domain_fraction": (
            float(np.mean(values >= minimum_domain_effect)) if len(values) else 0.0
        ),
        "minimum_domain_effect": float(np.min(values)) if len(values) else 0.0,
        "domains": details,
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists() and not args.preflight_only:
        raise FileExistsError(args.output)
    if (
        args.draws < 10_000
        or args.minimum_formulas < 12
        or args.minimum_positive_identities < 24
        or args.minimum_domain_formulas < 3
        or args.minimum_replicating_domains < 2
        or not 0.5 < args.minimum_domain_nonnegative_fraction <= 1
        or args.minimum_discovery_effect < 0.05
        or args.minimum_confirmation_effect < 0.03
        or not 0 < args.fdr <= 0.05
    ):
        raise ValueError("stratified meta-action admission thresholds were weakened")
    required = (
        args.domain_rule_dir / "report.json",
        args.domain_rule_dir / "identity_domain_evidence.npz",
        args.predicates,
        args.observations,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"stratified meta-action inputs missing: {missing}")
    domain_report = json.loads(
        (args.domain_rule_dir / "report.json").read_text(encoding="utf-8")
    )
    predicates = json.loads(args.predicates.read_text(encoding="utf-8"))["predicates"]
    channels = json.loads(args.observations.read_text(encoding="utf-8"))["channels"]
    with np.load(args.domain_rule_dir / "identity_domain_evidence.npz") as source:
        evidence = {key: source[key] for key in source.files}
    if evidence["predicate_presence"].shape != (len(evidence["identity"]), len(predicates)):
        raise RuntimeError("predicate evidence matrix is misaligned")
    if evidence["observation_prevalence"].shape != (
        len(evidence["unit_identity"]),
        len(channels),
    ):
        raise RuntimeError("observation evidence matrix is misaligned")
    eligible_domains = tuple(domain_report["preflight"]["eligible_domains"])
    unit_predicates = evidence["predicate_presence"][evidence["unit_identity"]]
    valid = evidence["valid_structure_unit"].astype(bool)
    observation = evidence["observation_prevalence"].astype(np.float32)
    formula = evidence["unit_formula"].astype(str)
    identity = evidence["unit_identity"].astype(np.int64)
    domain = evidence["unit_domain"].astype(str)
    fold = evidence["unit_formula_fold"].astype(np.int16)
    admissible_channel = np.asarray(
        [
            bool(item.get("observed_species_formulae"))
            and bool(
                FORMULA_PATTERN.fullmatch(
                    str(item["observed_species_formulae"][0])
                )
            )
            for item in channels
        ],
        dtype=bool,
    )
    if not np.any(admissible_channel):
        raise RuntimeError("no formula-specified observation channel is available")
    preflight = {
        "status": "CHEMAWARE_STRATIFIED_META_ACTION_PREFLIGHT_PASS",
        "units": int(len(identity)),
        "identities": int(len(evidence["identity"])),
        "predicates": int(len(predicates)),
        "observations": int(len(channels)),
        "formula_specified_observations": int(np.sum(admissible_channel)),
        "eligible_domains": list(eligible_domains),
        "fold_roles": {
            "discovery": list(args.discovery_folds),
            "confirmation": args.confirmation_fold,
            "embedding_evaluation_untouched": 3,
            "reserve_untouched": 4,
        },
        "contracts": {
            "raw_spectra_pooled_across_domains": False,
            "contrasts_computed_within_formula_and_domain": True,
            "domains_equal_weighted_within_formula": True,
            "formulas_equal_weighted": True,
            "multi_domain_replication_required": True,
            "negative_or_absence_action_authorized": False,
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    discovery_candidates = []
    discovery_cache: dict[int, dict] = {}
    for predicate_index in range(len(predicates)):
        predicate = unit_predicates[:, predicate_index]
        current = stratified_formula_contrasts(
            observation,
            predicate,
            formula,
            identity,
            domain,
            fold,
            valid,
            tuple(args.discovery_folds),
            eligible_domains,
        )
        discovery_cache[predicate_index] = current
        if (
            len(current["formulas"]) < args.minimum_formulas
            or current["positive_identities"] < args.minimum_positive_identities
        ):
            continue
        effect = np.mean(current["matrix"], axis=0)
        candidates = np.flatnonzero(
            admissible_channel & (effect >= args.minimum_discovery_effect)
        )
        candidates = candidates[np.argsort(-effect[candidates], kind="stable")]
        admitted_for_predicate = 0
        for channel_index in candidates:
            replication = domain_replication(
                current,
                int(channel_index),
                args.minimum_domain_formulas,
                args.minimum_domain_effect,
            )
            if (
                replication["replicating_domains"] < args.minimum_replicating_domains
                or replication["positive_domains"] < args.minimum_replicating_domains
                or replication["nonnegative_domain_fraction"]
                < args.minimum_domain_nonnegative_fraction
            ):
                continue
            discovery_candidates.append(
                {
                    "predicate_index": predicate_index,
                    "channel_index": int(channel_index),
                    "discovery_effect": float(effect[channel_index]),
                    "discovery_formulas": int(len(current["formulas"])),
                    "discovery_positive_identities": current["positive_identities"],
                    "discovery_negative_identities": current["negative_identities"],
                    "discovery_domain_replication": replication,
                }
            )
            admitted_for_predicate += 1
            if admitted_for_predicate == args.max_channels_per_predicate:
                break

    confirmation_rows = []
    confirmation_cache: dict[int, dict] = {}
    for candidate_index, candidate in enumerate(discovery_candidates):
        predicate_index = candidate["predicate_index"]
        if predicate_index not in confirmation_cache:
            confirmation_cache[predicate_index] = stratified_formula_contrasts(
                observation,
                unit_predicates[:, predicate_index],
                formula,
                identity,
                domain,
                fold,
                valid,
                (args.confirmation_fold,),
                eligible_domains,
            )
        current = confirmation_cache[predicate_index]
        channel_index = candidate["channel_index"]
        values = current["matrix"][:, channel_index]
        bootstrap = (
            formula_bootstrap(
                values,
                current["formulas"],
                20260935 + candidate_index,
                args.draws,
            )
            if len(values)
            else {
                "formula_macro_mean": 0.0,
                "formula_cluster_bootstrap_95ci": [0.0, 0.0],
                "formula_clusters": 0,
                "draws": args.draws,
            }
        )
        replication = domain_replication(
            current,
            channel_index,
            args.minimum_domain_formulas,
            args.minimum_domain_effect,
        )
        confirmation_rows.append(
            candidate
            | {
                "confirmation_effect": float(np.mean(values)) if len(values) else 0.0,
                "confirmation_formulas": int(len(current["formulas"])),
                "confirmation_positive_identities": current["positive_identities"],
                "confirmation_negative_identities": current["negative_identities"],
                "confirmation_domain_replication": replication,
                "sign_flip_pvalue": sign_flip_pvalue(
                    values, 20261935 + candidate_index, args.draws
                ),
                "bootstrap": bootstrap,
            }
        )

    pvalues = np.asarray(
        [item["sign_flip_pvalue"] for item in confirmation_rows], dtype=np.float64
    )
    qvalues = bh_qvalues(pvalues) if len(pvalues) else pvalues
    admitted = []
    for item, qvalue in zip(confirmation_rows, qvalues):
        item["bh_qvalue"] = float(qvalue)
        replication = item["confirmation_domain_replication"]
        passed = (
            item["confirmation_formulas"] >= args.minimum_formulas
            and item["confirmation_positive_identities"]
            >= args.minimum_positive_identities
            and item["confirmation_effect"] >= args.minimum_confirmation_effect
            and item["bootstrap"]["formula_cluster_bootstrap_95ci"][0] > 0
            and replication["replicating_domains"]
            >= args.minimum_replicating_domains
            and replication["positive_domains"] >= args.minimum_replicating_domains
            and replication["nonnegative_domain_fraction"]
            >= args.minimum_domain_nonnegative_fraction
            and qvalue <= args.fdr
        )
        item["admitted"] = bool(passed)
        if not passed:
            continue
        predicate = predicates[item["predicate_index"]]
        channel = channels[item["channel_index"]]
        supported_domains = [
            value["domain"]
            for value in replication["domains"]
            if value["effect"] > args.minimum_domain_effect
        ]
        admitted.append(
            {
                "rule_id": (
                    f"EMP-META:{predicate['predicate_id']}:{channel['channel_id']}"
                ),
                "claim_type": "stratified_multi_domain_empirical_support",
                "parent_predicate": {"smarts_any": predicate["smarts_any"]},
                "context": {
                    "ion_mode": "positive",
                    "supported_acquisition_domains": supported_domains,
                },
                "observation": {
                    "kind": (
                        "neutral_loss"
                        if channel["match_type"] == "mass_diff"
                        else "diagnostic_fragment"
                    ),
                    "formula": str(channel["observed_species_formulae"][0]),
                    "exact_mass_da": float(channel["value_da"]),
                    "tolerance_da": float(channel["match_tolerance_da"]),
                },
                "suggested_action": {
                    "operation": "support_boost_observed_match",
                    "may_add_new_mz": False,
                    "negative_or_absence_action_authorized": False,
                    "requires_matched_controls": [
                        "within_domain_predicate_present_negative",
                        "official_score_hardness_caliper",
                        "clean_duplicate",
                    ],
                },
                "evidence": {
                    "independent_formulas": item["confirmation_formulas"],
                    "positive_identities": item[
                        "confirmation_positive_identities"
                    ],
                    "confirmation_effect": item["confirmation_effect"],
                    "formula_cluster_bootstrap_95ci": item["bootstrap"][
                        "formula_cluster_bootstrap_95ci"
                    ],
                    "replicating_domains": replication["replicating_domains"],
                    "positive_domains": replication["positive_domains"],
                    "nonnegative_domain_fraction": replication[
                        "nonnegative_domain_fraction"
                    ],
                    "sign_flip_pvalue": item["sign_flip_pvalue"],
                    "bh_qvalue": item["bh_qvalue"],
                    "supported_domain_effects": {
                        value["domain"]: value["effect"]
                        for value in replication["domains"]
                        if value["effect"] > args.minimum_domain_effect
                    },
                },
            }
        )

    report = {
        "status": (
            "CHEMAWARE_STRATIFIED_META_ACTIONS_ADMITTED"
            if admitted
            else "CHEMAWARE_STRATIFIED_META_ACTIONS_NONE_ADMITTED"
        ),
        "formal_training_authorized": False,
        "preflight": preflight,
        "counts": {
            "discovery_candidates": len(discovery_candidates),
            "confirmed_rules": len(admitted),
        },
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if not isinstance(value, Path) and key != "preflight_only"
        },
        "candidates": confirmation_rows,
        "provenance": {
            "domain_evidence_sha256": sha256_file(
                args.domain_rule_dir / "identity_domain_evidence.npz"
            ),
            "predicate_registry_sha256": sha256_file(args.predicates),
            "observation_registry_sha256": sha256_file(args.observations),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "stratified_meta_rules.json").write_text(
        json.dumps(
            {"schema": "chemaware_stratified_meta_action_rules_v1", "rules": admitted},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
