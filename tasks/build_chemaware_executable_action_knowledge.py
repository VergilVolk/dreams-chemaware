"""Build the fail-closed ChemAware executable-rule/action knowledge base.

Legacy mass-difference lists and record-derived MassBank peaks are retained as
observation candidates, never silently promoted to executable chemistry.  A
formal rule must contain a parent-structure predicate, acquisition context, a
chemically specified observation and multi-molecule evidence.  Empirical
formula-disjoint admission is a separate downstream step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_executable_rule(rule: dict) -> list[str]:
    errors: list[str] = []
    required = ("rule_id", "claim_type", "parent_predicate", "context", "observation", "evidence")
    for key in required:
        if key not in rule:
            errors.append(f"missing:{key}")
    if errors:
        return errors
    if rule["claim_type"] not in ("mechanistic", "empirical_structure_conditioned"):
        errors.append("invalid:claim_type")
    predicate = rule["parent_predicate"]
    if not isinstance(predicate, dict) or not predicate.get("smarts_any"):
        errors.append("invalid:parent_predicate.smarts_any")
    context = rule["context"]
    if context.get("ion_mode") not in ("positive", "negative"):
        errors.append("invalid:context.ion_mode")
    if not context.get("adducts"):
        errors.append("invalid:context.adducts")
    observation = rule["observation"]
    if observation.get("kind") not in ("neutral_loss", "diagnostic_fragment", "fragment_relation"):
        errors.append("invalid:observation.kind")
    if not observation.get("formula") or float(observation.get("exact_mass_da", 0)) <= 0:
        errors.append("invalid:observation.formula_or_mass")
    evidence = rule["evidence"]
    if not evidence.get("citation") or not evidence.get("source_id"):
        errors.append("invalid:evidence.provenance")
    if int(evidence.get("independent_molecules", 0)) < 20:
        errors.append("insufficient:evidence.independent_molecules")
    if int(evidence.get("independent_formulas", 0)) < 10:
        errors.append("insufficient:evidence.independent_formulas")
    if evidence.get("formula_disjoint_confirmation_pass") is not True:
        errors.append("unconfirmed:evidence.formula_disjoint")
    if rule["claim_type"] == "empirical_structure_conditioned":
        action = rule.get("suggested_action", {})
        if action.get("operation") not in (
            "support_boost_observed_match", "attenuate_observed_conflict",
        ):
            errors.append("invalid:suggested_action.operation")
        if action.get("may_add_new_mz") is not False:
            errors.append("invalid:suggested_action.may_add_new_mz")
        required_controls = {"predicate_swapped", "matched_random_observed_peak", "clean_duplicate"}
        if not required_controls.issubset(action.get("requires_matched_controls", [])):
            errors.append("invalid:suggested_action.controls")
    return errors


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-core", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--legacy-massbank", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_massbank.json")
    parser.add_argument("--observation-registry", type=Path, default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json")
    parser.add_argument("--parent-predicates", type=Path, default=ROOT / "dreams/models/chem_aware/chem_parent_predicates_rdkit_v1.json")
    parser.add_argument("--curated-rules", type=Path, default=None,
                        help="Optional human-curated executable rules in the new schema.")
    parser.add_argument("--output", type=Path, default=ROOT / "dreams/models/chem_aware/chem_action_knowledge_v3.json")
    parser.add_argument("--report", type=Path, default=ROOT / "data/validation/chemaware_action_knowledge_v3/report.json")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite action knowledge artifacts")
    core = json.loads(args.legacy_core.read_text(encoding="utf-8"))
    massbank = json.loads(args.legacy_massbank.read_text(encoding="utf-8"))
    observations = json.loads(args.observation_registry.read_text(encoding="utf-8"))
    parent_predicates = json.loads(args.parent_predicates.read_text(encoding="utf-8"))
    allowed_smarts = {
        smarts
        for predicate in parent_predicates["predicates"]
        for smarts in predicate["smarts_any"]
    }
    curated = []
    if args.curated_rules is not None:
        curated_body = json.loads(args.curated_rules.read_text(encoding="utf-8"))
        curated = curated_body.get("rules", curated_body if isinstance(curated_body, list) else [])
    admitted = []; rejected = []
    for rule in curated:
        errors = validate_executable_rule(rule)
        if rule.get("claim_type") == "empirical_structure_conditioned":
            submitted_smarts = set(rule.get("parent_predicate", {}).get("smarts_any", []))
            if not submitted_smarts or not submitted_smarts.issubset(allowed_smarts):
                errors.append("unregistered:parent_predicate.smarts_any")
        (rejected if errors else admitted).append({"rule": rule, "errors": errors})
    admitted_rules = [item["rule"] for item in admitted]
    knowledge = {
        "schema": "chemaware_executable_action_knowledge_v3",
        "scientific_type": "structure_conditioned_spectral_action_knowledge",
        "rules": admitted_rules,
        "observation_candidates": [
            {"channel_id": channel["channel_id"], "formal_action_enabled": False,
             "reason": "missing_parent_structure_prerequisite_or_formula_disjoint_admission"}
            for channel in observations["channels"]
        ],
        "action_families": [
            {
                "action_id": "corrective_conflict_attenuation",
                "role": "corrective",
                "selector": "training_only_true_minus_same_formula_counterfactual_evidence",
                "operation": "attenuate_observed_peak_intensity",
                "may_add_new_mz": False,
                "requires_matched_controls": ["candidate_swapped", "peak_evidence_permuted", "clean_duplicate"],
                "formal_enabled": False,
            },
            {
                "action_id": "corrective_support_amplification",
                "role": "corrective",
                "selector": "training_only_true_minus_same_formula_counterfactual_evidence",
                "operation": "amplify_observed_peak_intensity",
                "may_add_new_mz": False,
                "requires_matched_controls": ["candidate_swapped", "peak_evidence_permuted", "clean_duplicate"],
                "formal_enabled": False,
            },
            {
                "action_id": "corrective_signed_reweighting",
                "role": "corrective",
                "selector": "training_only_true_minus_same_formula_counterfactual_evidence",
                "operation": "bidirectionally_reweight_observed_peak_intensity",
                "may_add_new_mz": False,
                "requires_matched_controls": ["candidate_swapped", "peak_evidence_permuted", "clean_duplicate"],
                "formal_enabled": False,
            },
            {
                "action_id": "mechanistic_supported_peak_dropout",
                "role": "robustness",
                "selector": "admitted_parent_conditioned_rule_matches_observed_peak",
                "operation": "attenuate_observed_peak_intensity",
                "may_add_new_mz": False,
                "requires_matched_controls": ["matched_random_peak_dropout", "clean_duplicate"],
                "formal_enabled": False,
            },
        ],
        "contracts": {
            "rule_and_action_are_separate_objects": True,
            "record_specific_mass_difference_is_not_a_rule": True,
            "parent_structure_predicate_required": True,
            "observed_peaks_only": True,
            "new_peak_synthesis_forbidden": True,
            "teacher_structure_training_only": True,
            "inference_input_one_clean_spectrum": True,
            "direct_training_must_not_match_teacher_scores_or_embeddings": True,
            "formula_disjoint_action_admission_required": True,
            "global_action_setting_selected_before_confirmation": True,
            "embedding_evaluation_formulas_action_unseen": True,
            "outer_formulas_action_unseen": True,
            "overlapping_rule_matches_union_without_dose_accumulation": True,
        },
        "provenance": {
            "legacy_core_sha256": sha256_file(args.legacy_core),
            "legacy_massbank_sha256": sha256_file(args.legacy_massbank),
            "observation_registry_sha256": sha256_file(args.observation_registry),
            "parent_predicate_registry_sha256": sha256_file(args.parent_predicates),
            "curated_rules_sha256": sha256_file(args.curated_rules) if args.curated_rules else None,
        },
    }
    report = {
        "status": "CHEMAWARE_ACTION_KNOWLEDGE_SCHEMA_PASS",
        "formal_action_training_authorized": False,
        "counts": {
            "legacy_core_records_quarantined": len(core["rules"]),
            "massbank_record_rules_quarantined": len(massbank["rules"]),
            "observation_candidates": len(observations["channels"]),
            "curated_rules_submitted": len(curated), "curated_rules_admitted": len(admitted_rules),
            "curated_rules_rejected": len(rejected), "formal_action_families_enabled": 0,
        },
        "quarantine": {
            "legacy_core": "no executable parent-structure prerequisite",
            "massbank": "record-derived support=1 observations are not transferable rules",
            "observation_registry": "empirical observation candidates only",
        },
        "rejected_curated_rules": rejected,
        "next_gate": "global action-family headroom on discovery, then formula-disjoint confirmation against matched controls",
        "knowledge_sha256": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")
    report["knowledge_sha256"] = sha256_file(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=False)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
