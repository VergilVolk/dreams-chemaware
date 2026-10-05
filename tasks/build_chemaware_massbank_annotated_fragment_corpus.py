"""Aggregate recurrent, structure-conditioned fragment rules from MassBank.

The importer reads the official pinned MassBank-data ZIP directly and uses only
positive-mode ESI [M+H]+ records with explicit ``PK$ANNOTATION`` fragment
formulae.  Single spectra never become rules.  Observations are grouped by
fragment/loss formula and generic Murcko scaffold, then admitted only when at
least three distinct molecular identities support the relationship.

Every aggregate preserves source accessions, contributors and record licences.
It remains a candidate-selection hypothesis until a formula-disjoint matched-
control qualification succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from import_chemaware_msfinder_rule_tables import ATOMIC_MASS, FORMULA_TOKEN, formula_mass


ROOT = Path(__file__).resolve().parents[1]
MASSBANK_RELEASE = "2026.03"
MASSBANK_DOI = "10.5281/zenodo.19073053"
MASSBANK_MD5 = "51904e29ee8717756153a8d31726789d"
FORMULA_EXACT = re.compile(r"^(?:[A-Z][a-z]?\d*)+[+-]?$", re.ASCII)


@dataclass(frozen=True)
class ParsedRecord:
    accession: str
    contributor: str
    license: str
    ik14: str
    molecule: Chem.Mol
    parent_formula: str
    precursor_mz: float
    annotations: tuple[tuple[float, str], ...]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the published archive.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-identities", type=int, default=3)
    parser.add_argument("--consensus-fraction", type=float, default=0.8)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--minimum-consensus-bits", type=int, default=3)
    parser.add_argument("--maximum-fragment-error-da", type=float, default=0.015)
    parser.add_argument("--maximum-loss-error-da", type=float, default=0.02)
    return parser.parse_args()


def first_tag(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def parse_formula(value: str) -> dict[str, int] | None:
    value = str(value).strip().rstrip("+-")
    # Isotope labels, brackets and ambiguous formulae are excluded.
    if not value or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", value):
        return None
    tokens = list(FORMULA_TOKEN.finditer(value))
    if not tokens or "".join(match.group(0) for match in tokens) != value:
        return None
    if any(match.group(1) not in ATOMIC_MASS for match in tokens):
        return None
    output: dict[str, int] = {}
    for match in tokens:
        output[match.group(1)] = output.get(match.group(1), 0) + int(match.group(2) or 1)
    return output


def format_formula(values: dict[str, int]) -> str:
    if any(count < 0 for count in values.values()):
        raise ValueError("negative atom count")
    order = []
    if values.get("C", 0):
        order.extend(["C", "H"])
    order.extend(sorted(element for element in values if element not in order))
    return "".join(
        element + (str(values[element]) if values[element] != 1 else "")
        for element in order if values.get(element, 0) > 0
    )


def annotation_rows(lines: list[str]) -> list[tuple[float, str]]:
    output = []
    active = False
    for line in lines:
        if line.startswith("PK$ANNOTATION:"):
            active = True
            continue
        if active and not line.startswith(" "):
            active = False
        if not active or not line.strip():
            continue
        tokens = line.strip().split()
        if not tokens:
            continue
        try:
            mz = float(tokens[0])
        except ValueError:
            continue
        formula = None
        for token in reversed(tokens[1:]):
            cleaned = token.strip("{},;")
            if FORMULA_EXACT.fullmatch(cleaned) and parse_formula(cleaned) is not None:
                formula = cleaned.rstrip("+-")
                break
        if formula is not None and math.isfinite(mz) and mz > 0:
            output.append((mz, formula))
    return sorted(set(output))


def parse_record(text: str, member_name: str) -> ParsedRecord | None:
    lines = text.splitlines()
    accession = first_tag(lines, "ACCESSION:")
    if not accession:
        return None
    ion_mode = first_tag(lines, "AC$MASS_SPECTROMETRY: ION_MODE").upper()
    ionization = first_tag(lines, "AC$MASS_SPECTROMETRY: IONIZATION").upper()
    precursor_type = first_tag(lines, "MS$FOCUSED_ION: PRECURSOR_TYPE")
    if ion_mode != "POSITIVE" or ionization != "ESI" or precursor_type != "[M+H]+":
        return None
    try:
        precursor_mz = float(first_tag(lines, "MS$FOCUSED_ION: PRECURSOR_M/Z"))
    except ValueError:
        return None
    parent_formula = first_tag(lines, "CH$FORMULA:")
    if parse_formula(parent_formula) is None:
        return None
    smiles = first_tag(lines, "CH$SMILES:")
    if smiles.strip().upper() in {"", "N/A", "NA", "UNKNOWN"}:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    inchikey = ""
    for line in lines:
        if line.startswith("CH$LINK: INCHIKEY"):
            inchikey = line.split("INCHIKEY", 1)[1].strip()
            break
    if not inchikey:
        try:
            inchikey = Chem.MolToInchiKey(molecule)
        except Exception:
            return None
    ik14 = inchikey.split("-", 1)[0]
    if len(ik14) != 14:
        return None
    annotations = annotation_rows(lines)
    if not annotations:
        return None
    license_value = first_tag(lines, "LICENSE:") or "record_license_not_declared"
    contributor = Path(member_name).parts[-2] if len(Path(member_name).parts) >= 2 else "unknown"
    return ParsedRecord(
        accession=accession,
        contributor=contributor,
        license=license_value,
        ik14=ik14,
        molecule=molecule,
        parent_formula=parent_formula,
        precursor_mz=precursor_mz,
        annotations=tuple(annotations),
    )


def generic_scaffold(molecule: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return "ACYCLIC"
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic, canonical=True)


def exact_formula_mass(formula: str) -> float:
    return formula_mass(formula.rstrip("+-"))


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.minimum_identities < 3
        or not 0.6 <= args.consensus_fraction <= 1
        or args.minimum_consensus_bits < 2
        or args.maximum_fragment_error_da > 0.02
        or args.maximum_loss_error_da > 0.03
    ):
        raise ValueError("MassBank aggregation thresholds were weakened")
    observed_md5 = md5_file(args.archive)
    if observed_md5 != MASSBANK_MD5:
        raise RuntimeError(
            f"MassBank archive MD5 mismatch: expected={MASSBANK_MD5} observed={observed_md5}"
        )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=args.morgan_radius)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    audit = Counter()
    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        audit["text_members"] = len(members)
        for member in members:
            try:
                text = archive.read(member).decode("utf-8")
            except (UnicodeDecodeError, KeyError):
                audit["unreadable_records"] += 1
                continue
            record = parse_record(text, member)
            if record is None:
                audit["records_outside_contract"] += 1
                continue
            audit["eligible_annotated_records"] += 1
            parent = parse_formula(record.parent_formula)
            assert parent is not None
            ion_parent = dict(parent)
            ion_parent["H"] = ion_parent.get("H", 0) + 1
            scaffold = generic_scaffold(record.molecule)
            fingerprint = frozenset(
                int(bit) for bit in generator.GetSparseFingerprint(record.molecule).GetOnBits()
            )
            for mz, fragment_formula in record.annotations:
                fragment = parse_formula(fragment_formula)
                if fragment is None:
                    audit["unsupported_fragment_formula"] += 1
                    continue
                fragment_error = abs(mz - exact_formula_mass(fragment_formula))
                if fragment_error > args.maximum_fragment_error_da:
                    audit["fragment_mass_mismatch"] += 1
                    continue
                common = {
                    "accession": record.accession,
                    "contributor": record.contributor,
                    "license": record.license,
                    "ik14": record.ik14,
                    "fingerprint": fingerprint,
                    "observed_mass": mz,
                }
                groups[("product_ion", fragment_formula, scaffold)].append(common)
                loss_counts = {
                    element: ion_parent.get(element, 0) - fragment.get(element, 0)
                    for element in set(ion_parent).union(fragment)
                }
                if any(count < 0 for count in loss_counts.values()):
                    continue
                loss_formula = format_formula(loss_counts)
                if not loss_formula:
                    continue
                observed_loss = record.precursor_mz - mz
                try:
                    loss_error = abs(observed_loss - exact_formula_mass(loss_formula))
                except ValueError:
                    continue
                if loss_error <= args.maximum_loss_error_da:
                    groups[("neutral_loss", loss_formula, scaffold)].append({
                        **common, "observed_mass": observed_loss,
                    })
                    audit["validated_neutral_losses"] += 1

    records = []
    excluded = Counter()
    for (kind, formula, scaffold), observations in sorted(groups.items()):
        by_identity = {}
        for observation in observations:
            by_identity.setdefault(observation["ik14"], observation)
        if len(by_identity) < args.minimum_identities:
            excluded["insufficient_identity_support"] += 1
            continue
        identities = [by_identity[key] for key in sorted(by_identity)]
        bit_counts: Counter[int] = Counter()
        for observation in identities:
            bit_counts.update(observation["fingerprint"])
        threshold = math.ceil(args.consensus_fraction * len(identities))
        consensus_bits = sorted(bit for bit, count in bit_counts.items() if count >= threshold)
        if len(consensus_bits) < args.minimum_consensus_bits:
            excluded["insufficient_structure_consensus"] += 1
            continue
        masses = np.asarray([observation["observed_mass"] for observation in observations])
        median = float(np.median(masses))
        mad = float(np.median(np.abs(masses - median)))
        contributors = sorted({observation["contributor"] for observation in observations})
        accessions = sorted({observation["accession"] for observation in observations})
        licenses = sorted({observation["license"] for observation in observations})
        tier = (
            "B_curated_conditional"
            if len(by_identity) >= 5 and len(contributors) >= 2
            else "C_calibration_required"
        )
        records.append({
            "record_id": f"MASSBANK:{kind}:{formula}:{hashlib.sha256(scaffold.encode()).hexdigest()[:12]}",
            "kind": kind,
            "formula": formula,
            "exact_mass": exact_formula_mass(formula),
            "median_observed_mass": median,
            "observed_mass_mad": mad,
            "generic_murcko_scaffold": scaffold,
            "consensus_morgan_bits": consensus_bits,
            "source_identity_count": len(by_identity),
            "source_spectrum_count": len(accessions),
            "source_ik14": sorted(by_identity),
            "source_accessions": accessions,
            "source_contributors": contributors,
            "source_licenses": licenses,
            "confidence_tier": tier,
            "training_policy": "formula_disjoint_matched_control_qualification_only",
        })
    if not records:
        raise RuntimeError("no recurrent MassBank annotated fragment passed the frozen gates")
    corpus = {
        "schema": "chemaware.massbank-annotated-fragment-evidence.v1",
        "source": {
            "release": MASSBANK_RELEASE,
            "doi": MASSBANK_DOI,
            "archive_md5": observed_md5,
            "archive_bytes": args.archive.stat().st_size,
            "record_license_contract": "preserved per aggregate; review before redistribution",
        },
        "aggregation": {
            "ionization": "ESI",
            "ion_mode": "POSITIVE",
            "precursor_type": "[M+H]+",
            "minimum_distinct_identities": args.minimum_identities,
            "structure_group": "generic Murcko scaffold",
            "morgan_radius": args.morgan_radius,
            "consensus_fraction": args.consensus_fraction,
            "minimum_consensus_bits": args.minimum_consensus_bits,
            "maximum_fragment_error_da": args.maximum_fragment_error_da,
            "maximum_loss_error_da": args.maximum_loss_error_da,
        },
        "records": records,
        "contracts": {
            "single_spectrum_never_admitted": True,
            "exact_source_identity_suppressed_downstream": True,
            "candidate_truth_not_used": True,
            "matched_controls_required": True,
            "native_dreams_loss_unchanged": True,
        },
    }
    report = {
        "status": "CHEMAWARE_MASSBANK_ANNOTATED_FRAGMENT_CORPUS_COMPLETE",
        "archive_md5": observed_md5,
        "audit": dict(audit),
        "groups_considered": len(groups),
        "records": len(records),
        "product_ion_records": sum(record["kind"] == "product_ion" for record in records),
        "neutral_loss_records": sum(record["kind"] == "neutral_loss" for record in records),
        "confidence_tiers": dict(Counter(record["confidence_tier"] for record in records)),
        "excluded": dict(excluded),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_massbank_annotated_", dir=args.output.parent))
    try:
        (temporary / "massbank_fragment_corpus.json").write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["corpus_sha256"] = sha256_file(temporary / "massbank_fragment_corpus.json")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
