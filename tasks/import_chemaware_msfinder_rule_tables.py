"""Import pinned MS-FINDER product-ion and neutral-loss rule tables.

The source tables are curated empirical resources, not automatically valid
triplet labels.  Every row is exact-mass checked and assigned a prior
confidence tier; all rows remain subject to candidate-specific matched-control
qualification before they may select a DreaMS negative.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "78c7ef9697240da857b1fff643d9e53afb7308d0"
ATOMIC_MASS = {
    "C": 12.0, "H": 1.00782503223, "N": 14.00307400443,
    "O": 15.99491461957, "P": 30.97376199842, "S": 31.9720711744,
    "F": 18.99840316273, "Cl": 34.968852682, "Br": 78.9183376,
    "I": 126.904468,
}
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral-loss-table", type=Path, required=True)
    parser.add_argument("--product-ion-table", type=Path, required=True)
    parser.add_argument("--source-commit", default=PINNED_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def formula_mass(formula: str) -> float:
    tokens = FORMULA_TOKEN.findall(formula)
    if not tokens or "".join(element + count for element, count in tokens) != formula:
        raise ValueError(f"unsupported molecular formula: {formula!r}")
    unknown = sorted({element for element, _ in tokens}.difference(ATOMIC_MASS))
    if unknown:
        raise ValueError(f"unsupported formula elements {unknown}: {formula}")
    return sum(
        ATOMIC_MASS[element] * (int(count) if count else 1)
        for element, count in tokens
    )


def read_rules(path: Path, kind: str) -> list[dict[str, object]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "Exact mass", "Formula", "IonMode", "Frequency", "FragmentShortInChIKeys",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid MS-FINDER table schema: {path}")
    output = []
    for index, row in enumerate(rows):
        exact_mass = float(row["Exact mass"])
        frequency = float(row["Frequency"])
        ion_mode = row["IonMode"]
        if ion_mode not in {"Positive", "Negative"}:
            raise RuntimeError(f"invalid MS-FINDER ion mode: {ion_mode}")
        if not (math.isfinite(exact_mass) and exact_mass > 0):
            raise RuntimeError(f"invalid exact mass at row {index}: {exact_mass}")
        if not (math.isfinite(frequency) and frequency >= 0.1):
            raise RuntimeError(f"invalid empirical frequency at row {index}: {frequency}")
        mass_error = abs(exact_mass - formula_mass(row["Formula"]))
        if mass_error > 5e-6:
            raise RuntimeError(
                f"formula/exact-mass mismatch at {path}:{index + 2}: {mass_error}"
            )
        inchikeys = sorted({
            value.strip() for value in row["FragmentShortInChIKeys"].split(";")
            if value.strip()
        })
        if not inchikeys or any(len(value) != 14 for value in inchikeys):
            raise RuntimeError(f"invalid short InChIKey set at {path}:{index + 2}")
        # Tier B is only a prior: it requires both nontrivial empirical
        # frequency and recurrence across at least two associated structures.
        # It still cannot enter training before formula-disjoint qualification.
        tier = (
            "B_curated_conditional"
            if frequency >= 1.0 and len(inchikeys) >= 2
            else "C_calibration_required"
        )
        output.append({
            "rule_id": f"MSFINDER:{kind}:{index:05d}",
            "kind": kind,
            "exact_mass": exact_mass,
            "formula": row["Formula"],
            "ion_mode": ion_mode.lower(),
            "empirical_frequency": frequency,
            "associated_short_inchikeys": inchikeys,
            "associated_structure_count": len(inchikeys),
            "formula_mass_error_da": mass_error,
            "confidence_tier": tier,
            "training_policy": "disabled_until_formula_disjoint_matched_control_qualification",
        })
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.source_commit != PINNED_COMMIT:
        raise RuntimeError("MS-FINDER source commit is not the audited pinned revision")
    neutral = read_rules(args.neutral_loss_table, "neutral_loss")
    product = read_rules(args.product_ion_table, "product_ion")
    if len(neutral) != 1643 or len(product) != 2995:
        raise RuntimeError(
            f"pinned MS-FINDER cardinality drift: neutral={len(neutral)} product={len(product)}"
        )
    rules = neutral + product
    identifiers = [row["rule_id"] for row in rules]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("MS-FINDER rule identifiers are not unique")
    tier_counts = {
        tier: sum(row["confidence_tier"] == tier for row in rules)
        for tier in ("B_curated_conditional", "C_calibration_required")
    }
    body = {
        "schema": "chemaware.msfinder-rule-tables.v1",
        "source": {
            "repository": "https://github.com/RECETOX/recetox-msfinder",
            "commit": args.source_commit,
            "neutral_loss_path": "MsfinderCommon/Resources/NeutralLossDB_vs2.ndb",
            "product_ion_path": "MsfinderCommon/Resources/ProductIonLib_vs1.pid",
            "neutral_loss_sha256": sha256(args.neutral_loss_table),
            "product_ion_sha256": sha256(args.product_ion_table),
            "license_observation": (
                "repository README says assembly CC-BY-4.0; third-party notice says "
                "source code LGPL-3.0; redistribution status of embedded empirical "
                "tables requires project-level review"
            ),
            "usage_boundary": "internal research evidence; preserve source attribution",
        },
        "rules": rules,
        "contracts": {
            "frequency_is_not_support_count": True,
            "confidence_is_prior_not_loss_weight": True,
            "candidate_specific_proof_required": True,
            "matched_controls_required": True,
            "formula_disjoint_qualification_required": True,
            "no_peak_synthesis": True,
        },
    }
    report = {
        "status": "CHEMAWARE_MSFINDER_RULE_IMPORT_COMPLETE",
        "rules": len(rules),
        "neutral_loss_rules": len(neutral),
        "product_ion_rules": len(product),
        "confidence_tiers": tier_counts,
        "maximum_formula_mass_error_da": max(row["formula_mass_error_da"] for row in rules),
        "source_commit": args.source_commit,
        "gates": {
            "pinned_commit": args.source_commit == PINNED_COMMIT,
            "expected_cardinality": len(neutral) == 1643 and len(product) == 2995,
            "all_exact_masses_verified": all(row["formula_mass_error_da"] <= 5e-6 for row in rules),
            "no_rule_directly_authorized": all(
                row["training_policy"].startswith("disabled_until") for row in rules
            ),
        },
    }
    if not all(report["gates"].values()):
        raise RuntimeError(f"MS-FINDER rule import gates failed: {report['gates']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_msfinder_rules_", dir=args.output.parent))
    try:
        (temporary / "msfinder_rules.json").write_text(
            json.dumps(body, indent=2) + "\n", encoding="utf-8",
        )
        report["rules_sha256"] = sha256(temporary / "msfinder_rules.json")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
