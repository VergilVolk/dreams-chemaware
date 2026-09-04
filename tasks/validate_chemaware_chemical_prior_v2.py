"""Fail-closed validation for the ChemAware v2 chemical-prior contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
FORMULA = re.compile(r"^(?:[A-Z][a-z]?[0-9]*)+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_errors(library: dict) -> list[str]:
    errors: list[str] = []
    source_ids = [source["source_id"] for source in library["sources"]]
    if len(source_ids) != len(set(source_ids)):
        errors.append("sources contain duplicate source_id values")
    known_sources = set(source_ids)
    rule_ids = [rule["rule_id"] for rule in library["rules"]]
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("rules contain duplicate rule_id values")

    for rule in library["rules"]:
        rid = rule["rule_id"]
        scope = rule["acquisition_scope"]
        structure = rule["structure_scope"]
        transformation = rule["transformation"]
        evidence = rule["evidence"]
        deployment = rule["deployment"]

        missing_sources = sorted(set(evidence["source_ids"]) - known_sources)
        if missing_sources:
            errors.append(f"{rid}: unknown evidence source_ids {missing_sources}")

        energy = scope.get("collision_energy")
        if energy and energy["minimum"] is not None and energy["maximum"] is not None:
            if energy["minimum"] > energy["maximum"]:
                errors.append(f"{rid}: collision_energy minimum exceeds maximum")
        if transformation["hydrogen_shift_min"] > transformation["hydrogen_shift_max"]:
            errors.append(f"{rid}: hydrogen shift minimum exceeds maximum")
        if transformation["minimum_path_depth"] > transformation["maximum_path_depth"]:
            errors.append(f"{rid}: path-depth minimum exceeds maximum")
        if transformation["maximum_broken_bonds"] < len([
            edit for edit in transformation["bond_edits"] if edit["operation"] == "break"
        ]):
            errors.append(f"{rid}: maximum_broken_bonds is smaller than explicit break edits")

        mapped_atoms: set[int] = set()
        for smarts in structure["required_smarts"] + structure["forbidden_smarts"]:
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                errors.append(f"{rid}: invalid SMARTS {smarts!r}")
                continue
            mapped_atoms.update(
                atom.GetAtomMapNum() for atom in query.GetAtoms() if atom.GetAtomMapNum() > 0
            )
        for edit in transformation["bond_edits"]:
            for key in ("atom_map_left", "atom_map_right"):
                if edit[key] not in mapped_atoms:
                    errors.append(f"{rid}: {key}={edit[key]} is absent from structure SMARTS maps")

        product_formula = transformation.get("product_formula")
        if product_formula is not None and FORMULA.fullmatch(product_formula) is None:
            errors.append(f"{rid}: product_formula is not a neutral elemental formula")

        positive = evidence["positive_observations"]
        eligible = evidence["eligible_observations"]
        if positive > eligible:
            errors.append(f"{rid}: positive_observations exceeds eligible_observations")
        posterior = evidence["posterior_mean"]
        ci_low, ci_high = evidence["posterior_ci_low"], evidence["posterior_ci_high"]
        supplied = [posterior is not None, ci_low is not None, ci_high is not None]
        if any(supplied) and not all(supplied):
            errors.append(f"{rid}: posterior mean and both interval bounds must be supplied together")
        if all(supplied) and not (ci_low <= posterior <= ci_high):
            errors.append(f"{rid}: posterior_mean is outside its confidence interval")

        if rule["status"] != "active" and deployment["enabled_by_default"]:
            errors.append(f"{rid}: only active rules may be enabled by default")
        if deployment["role"] == "legacy_observation" and deployment["enabled_by_default"]:
            errors.append(f"{rid}: legacy observations cannot be enabled")

        if rule["status"] == "active":
            if not structure["required_smarts"]:
                errors.append(f"{rid}: active rule has no positive structure applicability SMARTS")
            if evidence["evidence_type"] in {"empirical", "hybrid"}:
                if evidence["unique_molecules"] < 5:
                    errors.append(f"{rid}: active empirical rule has fewer than 5 molecules")
                if evidence["unique_scaffolds"] < 3:
                    errors.append(f"{rid}: active empirical rule has fewer than 3 scaffolds")
                if (evidence["unique_sources_or_labs"] or 0) < 2:
                    errors.append(f"{rid}: active empirical rule has fewer than 2 sources/labs")
                if eligible == 0 or posterior is None or evidence["calibration_split"] is None:
                    errors.append(f"{rid}: active empirical rule lacks calibrated positive/negative evidence")
            if deployment["role"] == "training_teacher":
                if deployment["minimum_teacher_probability"] is None:
                    errors.append(f"{rid}: active training teacher lacks a probability threshold")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument(
        "--schema", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_prior_schema_v2.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    library = json.loads(args.library.read_text(encoding="utf-8"))
    schema_errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(library),
        key=lambda error: list(error.absolute_path),
    )
    errors = [
        f"schema:{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
        for error in schema_errors
    ]
    if not schema_errors:
        errors.extend(f"semantic:{message}" for message in semantic_errors(library))

    report = {
        "status": "pass" if not errors else "fail",
        "library": str(args.library.resolve()),
        "library_sha256": sha256_file(args.library),
        "schema": str(args.schema.resolve()),
        "schema_sha256": sha256_file(args.schema),
        "sources": len(library.get("sources", [])),
        "rules": len(library.get("rules", [])),
        "active_rules": sum(rule.get("status") == "active" for rule in library.get("rules", [])),
        "enabled_rules": sum(
            bool(rule.get("deployment", {}).get("enabled_by_default"))
            for rule in library.get("rules", [])
        ),
        "errors": errors,
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

