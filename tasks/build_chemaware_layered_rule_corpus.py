"""Compile the ChemAware chemical evidence corpus without promoting noise.

The legacy 335-entry table, record-level MassBank observations, validated
structure-conditioned associations, and generative physical/fragmentation
constraints have different scientific meanings.  This compiler preserves all
of them, but assigns an explicit confidence tier and a fail-closed training
policy.  In particular, a single observed spectrum is never promoted to a
transferable chemical rule.

The corpus is training supervision metadata.  It never changes the DreaMS
loss, and weights are admission/priority weights rather than loss multipliers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TIER_WEIGHT = {
    "A_physical_or_confirmed": 1.0,
    "B_curated_conditional": 0.75,
    "C_calibration_required": 0.35,
    "Q_quarantined": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-core", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--legacy-massbank", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_massbank.json",
    )
    parser.add_argument(
        "--executable-actions", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_action_knowledge_v3.json",
    )
    parser.add_argument(
        "--msfinder-rules", type=Path,
        default=(
            ROOT / "data/validation/chemaware_msfinder_rule_tables_v1_20260928/"
            "msfinder_rules.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def source_id(text: str) -> str:
    value = text.lower()
    if "tsugawa" in value or "ms-finder" in value:
        return "msfinder_hr_2016"
    if "compms2" in value:
        return "compms2miner_substructure_masses"
    if "massbank" in value:
        return "massbank_record"
    if "esi(+)" in value:
        return "legacy_esi_empirical"
    return "legacy_chemaware"


def legacy_policy(rule: dict[str, Any]) -> tuple[str, str, str]:
    """Return confidence tier, training policy, and scientific role."""
    category = str(rule.get("category", ""))
    source = str(rule.get("source", ""))
    if category in {"ISO", "NR", "EE"}:
        return (
            "A_physical_or_confirmed",
            "candidate_conditioned_constraint_only",
            "physical_or_widely_validated_constraint",
        )
    if category == "HR" and ("Tsugawa" in source or "MS-FINDER" in source):
        return (
            "B_curated_conditional",
            "candidate_conditioned_observed_peak_only",
            "validated_hydrogen_rearrangement_template",
        )
    if "CompMS2Miner" in source and rule.get("formula") and rule.get("ref"):
        return (
            "B_curated_conditional",
            "candidate_composition_and_mode_conditioned_only",
            "curated_diagnostic_observation",
        )
    return (
        "C_calibration_required",
        "disabled_until_formula_disjoint_specificity",
        "unconditional_mass_observation_hypothesis",
    )


def mechanism_records() -> list[dict[str, Any]]:
    """Generative constraints: a small family can create many valid events."""
    rows = [
        {
            "record_id": "MECH:elemental-subformula-conservation:v1",
            "family": "elemental_conservation",
            "description": "Fragment and neutral-loss formulae must be elemental subformulae of the candidate precursor after adduct accounting.",
            "applicability": {"ion_modes": ["positive", "negative"], "adducts": ["singly_charged"]},
            "source_ids": ["sirius_fragmentation_tree", "seven_golden_rules"],
        },
        {
            "record_id": "MECH:mass-defect-and-valence-feasibility:v1",
            "family": "formula_feasibility",
            "description": "Exact-mass matches must have nonnegative atom counts and chemically admissible valence/parity; RDBE is recorded but not used as an absolute exclusion.",
            "applicability": {"ion_modes": ["positive", "negative"], "adducts": ["singly_charged"]},
            "source_ids": ["seven_golden_rules"],
        },
        {
            "record_id": "MECH:nitrogen-rule-neutral:v1",
            "family": "nitrogen_rule",
            "description": "For neutral organic formulae, odd nominal mass and odd nitrogen count have matching parity.",
            "applicability": {"ion_modes": ["positive", "negative"], "neutralization_required": True},
            "source_ids": ["seven_golden_rules"],
        },
        {
            "record_id": "MECH:even-electron-preference-esi:v1",
            "family": "even_electron",
            "description": "Low-energy ESI CID predominantly yields even-electron fragment ions; radical losses are exceptions, not impossible events.",
            "applicability": {"ion_modes": ["positive", "negative"], "dissociation": ["CID", "HCD", "unknown_low_energy"]},
            "source_ids": ["even_electron_esi_2007", "msfinder_hr_2016"],
        },
        {
            "record_id": "MECH:isotope-spacing-and-abundance:v1",
            "family": "isotope_pattern",
            "description": "Candidate isotope envelopes use exact isotope spacing and natural-abundance likelihood, only when MS1 isotope evidence is present.",
            "applicability": {"requires_ms1_isotope_envelope": True, "elements": ["C", "N", "O", "S", "Cl", "Br", "Si"]},
            "source_ids": ["seven_golden_rules", "massbank_record_format"],
        },
        {
            "record_id": "MECH:adduct-neutral-mass-consistency:v1",
            "family": "adduct_consistency",
            "description": "Candidate neutral mass, precursor m/z, polarity and declared adduct must agree within the acquisition tolerance.",
            "applicability": {"ion_modes": ["positive", "negative"], "adducts": ["declared"]},
            "source_ids": ["mzmine_ion_identity", "sirius_io"],
        },
    ]
    for row in rows:
        row.update({
            "record_kind": "generative_constraint",
            "confidence_tier": "A_physical_or_confirmed",
            "admission_weight": TIER_WEIGHT["A_physical_or_confirmed"],
            "training_policy": "candidate_conditioned_observed_evidence_only",
            "quality": {
                "may_synthesize_peaks": False,
                "candidate_specific_proof_required": True,
                "matched_null_required": True,
            },
        })
    return rows


SOURCES = [
    {
        "source_id": "msfinder_hr_2016",
        "title": "Hydrogen Rearrangement Rules: Computational MS/MS Fragmentation and Structure Elucidation Using MS-FINDER Software",
        "doi": "10.1021/acs.analchem.5b04181",
        "role": "curated_mechanism",
        "reuse_scope": "rule semantics and citation; software/data license must be checked separately",
    },
    {
        "source_id": "msfinder_rule_tables_v1",
        "title": "MS-FINDER NeutralLossDB_vs2 and ProductIonLib_vs1",
        "doi": "10.1021/acs.analchem.5b04181",
        "role": "curated_empirical_rule_tables",
        "reuse_scope": (
            "pinned internal research import with attribution; embedded table "
            "redistribution license requires project-level review"
        ),
    },
    {
        "source_id": "seven_golden_rules",
        "title": "Seven Golden Rules for heuristic filtering of molecular formulas obtained by accurate mass spectrometry",
        "doi": "10.1186/1471-2105-8-105",
        "role": "physical_constraint",
        "reuse_scope": "published algorithmic constraints",
    },
    {
        "source_id": "even_electron_esi_2007",
        "title": "The even-electron rule in electrospray mass spectra of pesticides",
        "doi": "10.1002/rcm.3271",
        "role": "curated_mechanism",
        "reuse_scope": "published statistical mechanism; exceptions retained",
    },
    {
        "source_id": "compms2miner_substructure_masses",
        "title": "compMS2Miner Substructure_masses curated observation table",
        "doi": None,
        "role": "curated_observation",
        "reuse_scope": "local package-derived table with per-record citations",
    },
    {
        "source_id": "massbank_record_format",
        "title": "MassBank Record Format and annotated peaks",
        "doi": None,
        "role": "calibration_dataset",
        "license": "per-record Creative Commons license; preserve each record license",
        "reuse_scope": "record-level observations; automated annotations remain tentative",
    },
    {
        "source_id": "sirius_fragmentation_tree",
        "title": "SIRIUS fragmentation trees",
        "doi": "10.1038/s41592-019-0344-8",
        "role": "fragmentation_teacher",
        "reuse_scope": "runtime output subject to installed SIRIUS license/account",
    },
    {
        "source_id": "sirius_csi_fingerid",
        "title": "CSI:FingerID structure scoring",
        "doi": "10.1073/pnas.1509788112",
        "role": "fragmentation_teacher",
        "reuse_scope": "runtime output subject to SIRIUS web-service terms",
    },
    {
        "source_id": "mzmine_ion_identity",
        "title": "MZmine ion identity networking adduct and in-source-fragment relations",
        "doi": "10.1038/s41467-021-23953-9",
        "role": "physical_constraint",
        "reuse_scope": "published algorithm and official documentation",
    },
    {
        "source_id": "sirius_io",
        "title": "SIRIUS 6 input/output and formula computation contract",
        "doi": None,
        "role": "software_contract",
        "reuse_scope": "official documentation",
    },
]


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    core = json.loads(args.legacy_core.read_text(encoding="utf-8"))
    massbank = json.loads(args.legacy_massbank.read_text(encoding="utf-8"))
    executable = json.loads(args.executable_actions.read_text(encoding="utf-8"))
    msfinder = json.loads(args.msfinder_rules.read_text(encoding="utf-8"))
    if msfinder.get("schema") != "chemaware.msfinder-rule-tables.v1":
        raise RuntimeError("MS-FINDER rule-table schema is not recognized")
    records: list[dict[str, Any]] = mechanism_records()

    for index, rule in enumerate(core["rules"]):
        tier, policy, role = legacy_policy(rule)
        records.append({
            "record_id": f"LEGACY:{index:04d}:{rule['name']}",
            "record_kind": "fixed_mass_observation" if rule.get("category") in {"NL", "CF"} else "rule_template",
            "family": rule.get("category"),
            "confidence_tier": tier,
            "admission_weight": TIER_WEIGHT[tier],
            "training_policy": policy,
            "scientific_role": role,
            "source_ids": [source_id(str(rule.get("source", "")))],
            "applicability": {
                "ion_mode": rule.get("mode"),
                "candidate_formula_required": bool(rule.get("formula")),
                "candidate_structure_required": False,
            },
            "payload": rule,
            "quality": {
                "candidate_specific_proof_required": True,
                "matched_null_required": True,
                "may_synthesize_peaks": False,
            },
        })

    for index, rule in enumerate(massbank["rules"]):
        records.append({
            "record_id": f"MASSBANK-RECORD:{index:05d}:{rule['name']}",
            "record_kind": "single_spectrum_observation",
            "family": rule.get("category"),
            "confidence_tier": "Q_quarantined",
            "admission_weight": 0.0,
            "training_policy": "never_direct_training; aggregate_and_reconfirm_first",
            "scientific_role": "record_level_observation_not_transferable_rule",
            "source_ids": ["massbank_record_format"],
            "applicability": {"ion_mode": rule.get("mode"), "record_scope": rule.get("scope")},
            "payload": rule,
            "quality": {
                "support": int(rule.get("support", 0)),
                "candidate_specific_proof_required": True,
                "matched_null_required": True,
                "may_synthesize_peaks": False,
            },
        })

    for index, rule in enumerate(executable.get("rules", [])):
        evidence = rule.get("evidence", {})
        if evidence.get("formula_disjoint_confirmation_pass") is not True:
            raise RuntimeError("unconfirmed executable rule reached layered corpus")
        records.append({
            "record_id": f"CONFIRMED:{index:04d}:{rule['rule_id']}",
            "record_kind": "structure_conditioned_association",
            "family": "confirmed_structure_observation",
            "confidence_tier": "A_physical_or_confirmed",
            "admission_weight": 1.0,
            "training_policy": "crossfit_candidate_selection_only",
            "scientific_role": "formula_disjoint_confirmed_association_not_mechanistic_claim",
            "source_ids": [str(evidence.get("source_id"))],
            "applicability": rule.get("context", {}),
            "payload": rule,
            "quality": {
                "candidate_specific_proof_required": True,
                "matched_null_required": True,
                "may_synthesize_peaks": False,
                "formula_disjoint_confirmation_pass": True,
                "bootstrap_95ci": evidence.get("bootstrap_95ci"),
                "bh_qvalue": evidence.get("bh_qvalue"),
            },
        })

    for rule in msfinder["rules"]:
        tier = str(rule["confidence_tier"])
        if tier not in {"B_curated_conditional", "C_calibration_required"}:
            raise RuntimeError(f"invalid MS-FINDER confidence tier: {tier}")
        records.append({
            "record_id": rule["rule_id"],
            "record_kind": "curated_multi_record_observation",
            "family": f"msfinder_{rule['kind']}",
            "confidence_tier": tier,
            "admission_weight": TIER_WEIGHT[tier],
            "training_policy": rule["training_policy"],
            "scientific_role": "candidate_conditioned_empirical_fragmentation_prior",
            "source_ids": ["msfinder_rule_tables_v1"],
            "applicability": {
                "ion_mode": rule["ion_mode"],
                "candidate_formula_required": True,
                "observed_peak_or_loss_required": True,
            },
            "payload": rule,
            "quality": {
                "candidate_specific_proof_required": True,
                "matched_null_required": True,
                "may_synthesize_peaks": False,
                "empirical_frequency": rule["empirical_frequency"],
                "associated_structure_count": rule["associated_structure_count"],
                "formula_mass_error_da": rule["formula_mass_error_da"],
                "formula_disjoint_confirmation_pass": False,
            },
        })

    identifiers = [row["record_id"] for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("layered corpus record identifiers are not unique")
    tier_counts = Counter(row["confidence_tier"] for row in records)
    policy_counts = Counter(row["training_policy"] for row in records)
    active = [row for row in records if float(row["admission_weight"]) > 0]
    if any(
        row["record_kind"] == "single_spectrum_observation"
        or int(row.get("quality", {}).get("support", 2)) <= 1
        for row in active
    ):
        raise RuntimeError("single-spectrum observation was promoted into active evidence")

    corpus = {
        "schema": "chemaware.layered-chemical-evidence.v4",
        "release": "2026-09-28",
        "purpose": "candidate selection for native DreaMS triplets; never a distillation target",
        "tier_weights": TIER_WEIGHT,
        "sources": SOURCES,
        "records": records,
        "contracts": {
            "rule_count_is_not_evidence_count": True,
            "single_spectrum_records_quarantined": True,
            "candidate_specific_proof_required": True,
            "matched_null_required": True,
            "no_peak_synthesis": True,
            "confidence_weight_is_admission_priority_not_loss_weight": True,
            "native_dreams_loss_unchanged": True,
            "outer_roles_untouched": True,
        },
        "provenance": {
            "legacy_core": {"path": str(args.legacy_core), "sha256": sha256_file(args.legacy_core)},
            "legacy_massbank": {"path": str(args.legacy_massbank), "sha256": sha256_file(args.legacy_massbank)},
            "executable_actions": {"path": str(args.executable_actions), "sha256": sha256_file(args.executable_actions)},
            "msfinder_rules": {"path": str(args.msfinder_rules), "sha256": sha256_file(args.msfinder_rules)},
        },
    }
    report = {
        "status": "CHEMAWARE_LAYERED_RULE_CORPUS_COMPLETE",
        "records": len(records),
        "active_admission_records": len(active),
        "quarantined_records": tier_counts["Q_quarantined"],
        "confidence_tiers": dict(tier_counts),
        "training_policies": dict(policy_counts),
        "legacy_core_records": len(core["rules"]),
        "massbank_single_spectrum_records": len(massbank["rules"]),
        "confirmed_structure_rules": len(executable.get("rules", [])),
        "msfinder_curated_rules": len(msfinder["rules"]),
        "generative_constraint_families": len(mechanism_records()),
        "gates": {
            "more_than_legacy_335": len(records) > 335,
            "single_spectrum_records_have_zero_weight": all(
                row["admission_weight"] == 0
                for row in records if row["record_kind"] == "single_spectrum_observation"
            ),
            "all_active_records_require_candidate_specific_proof": all(
                row["quality"]["candidate_specific_proof_required"] for row in active
            ),
            "all_active_records_require_matched_null": all(
                row["quality"]["matched_null_required"] for row in active
            ),
            "no_rule_may_synthesize_peaks": all(
                row["quality"]["may_synthesize_peaks"] is False for row in records
            ),
        },
        "claim_limit": "Corpus size is not triplet count or embedding improvement; only qualified candidate relations can become triplets.",
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"layered rule corpus gates failed: {report['gates']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_layered_rules_", dir=args.output.parent))
    try:
        (temporary / "rule_corpus.json").write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        report["rule_corpus_sha256"] = sha256_file(temporary / "rule_corpus.json")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
