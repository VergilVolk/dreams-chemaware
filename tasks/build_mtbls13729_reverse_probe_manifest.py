"""Freeze reference-spectrum probes for MTBLS13729 reverse spectral phenotyping.

This stage selects chemistry before looking at phenotype outcomes.  It creates
no disease association, annotation gain, or identity claim.  Public library
spectra remain reference spectra rather than authentic standards from the
MTBLS13729 analytical platform.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PANELS = ("neg_rp", "pos_rp")
FAMILY_ORDER = (
    "long_chain_acylcarnitine_parent",
    "methylated_guanosine_parent",
    "acetylated_polyamine_parent",
    "free_sialic_acid",
)

# These identities were frozen by the existing MTBLS13729 biology ledger before
# this reverse-search experiment.  They are hypotheses, not ground truth.
FROZEN_HYPOTHESIS_IK14 = {
    "XOMRRQXKHMYMOC": "long_chain_acylcarnitine_parent",  # palmitoylcarnitine
    "RBFQHRALHSUPIA": "long_chain_acylcarnitine_parent",  # C20:4-acylcarnitine-like
    "IXOXBSCIXZEQEQ": "methylated_guanosine_parent",
    "NNQCGMWOZJYFTM": "methylated_guanosine_parent",
    "BKCVMAZDKFQPHB": "acetylated_polyamine_parent",
    "SQVRNKJHWKZAKO": "free_sialic_acid",
}
FROZEN_HYPOTHESIS_FEATURE_IDS = {
    "XOMRRQXKHMYMOC": 150,
    "RBFQHRALHSUPIA": 3222,
    "IXOXBSCIXZEQEQ": 1597,
    "NNQCGMWOZJYFTM": 3019,
    "BKCVMAZDKFQPHB": 1717,
    "SQVRNKJHWKZAKO": 703,
}

COMBINATORIAL_MARKERS = (
    "known isomers",
    "isobaric peaks",
    "isobaric peak",
)


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def classify_family(name: str, smiles: str, precursor_mz: float) -> str:
    del smiles  # v2 deliberately does not infer biological scope from a name fragment in a SMILES field.
    text = name.strip().lower()
    # unified_v2 contains many combinatorial synthesis products whose names join
    # multiple reagents with underscores.  They are valid discovery probes in a
    # different experiment, but they are not clean endogenous-parent references.
    if "_" in text or any(marker in text for marker in COMBINATORIAL_MARKERS):
        return ""

    long_chain_tokens = (
        "lauroylcarnitine", "dodecanoylcarnitine", "tetradecanoyl-l-carnitine",
        "myristoylcarnitine", "hexadecanoyl-l-carnitine", "palmitoylcarnitine",
        "octadecanoyl-l-carnitine", "stearoylcarnitine", "oleoylcarnitine",
        "linoleyl carnitine", "arachidonoylcarnitine", "eicosatetraenoylcarnitine",
    )
    if precursor_mz >= 330.0 and any(token in text for token in long_chain_tokens):
        return "long_chain_acylcarnitine_parent"

    if re.search(r"(?:^|\s|[-,(])(?:1-|2-|7-|n\d?-)?(?:di)?methylguanosine\b", text):
        return "methylated_guanosine_parent"

    polyamine_parents = (
        "n1,n8-diacetylspermidine",
        "n1-acetylspermine",
        "n-acetylputrescine",
    )
    if any(text.startswith(parent) for parent in polyamine_parents):
        return "acetylated_polyamine_parent"

    if text.startswith("n-acetylneuraminic acid") or text.startswith("neu5ac"):
        return "free_sialic_acid"
    return ""


def iter_mgf_metadata(path: Path):
    fields: dict[str, str] | None = None
    record = -1
    n_peaks = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                record += 1
                fields, n_peaks = {}, 0
            elif line == "END IONS":
                if fields is not None and "PEPMASS" in fields:
                    try:
                        precursor = float(fields["PEPMASS"].split()[0])
                    except (ValueError, IndexError):
                        fields = None
                        continue
                    yield record, fields, precursor, n_peaks
                fields = None
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()
            elif fields is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        float(parts[0]); float(parts[1])
                    except ValueError:
                        continue
                    n_peaks += 1


def sample_fields(panel: str, path: Path) -> dict[str, object]:
    sample = path.stem
    match = re.fullmatch(r"(P\d+)-([LR])(N|tu|mu)", sample)
    if not match:
        raise RuntimeError(f"unexpected MTBLS13729 sample name: {sample}")
    patient, side, suffix = match.groups()
    tissue = "normal" if suffix == "N" else "tumor"
    histology = "normal" if suffix == "N" else (
        "mucinous" if suffix == "mu" else "tubular"
    )
    try:
        stored_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        stored_path = path.as_posix()
    return {
        "panel": panel,
        "sample_id": sample,
        "patient_id": patient,
        "side": "left" if side == "L" else "right",
        "tissue": tissue,
        "histology": histology,
        "hdf5_path": stored_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-map", type=Path,
        default=ROOT / "data/mtbls13729/annotation_biology_benchmark_v1/author_rplc_to_current_targets.csv",
    )
    parser.add_argument(
        "--reference-neg", type=Path,
        default=ROOT / "data/reference/unified_v2/unified_neg.mgf",
    )
    parser.add_argument(
        "--reference-pos", type=Path,
        default=ROOT / "data/reference/unified_v2/unified_pos.mgf",
    )
    parser.add_argument(
        "--sample-root", type=Path, default=ROOT / "data/mtbls13729/mzml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4",
    )
    parser.add_argument("--maximum-spectra-per-identity-adduct", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.author_map, args.reference_neg, args.reference_pos]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.maximum_spectra_per_identity_adduct < 1:
        raise ValueError("maximum spectra per identity/adduct must be positive")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    author = pd.read_csv(args.author_map)
    required_columns = {
        "panel", "source_name", "source_level", "source_inchikey", "feature_id"
    }
    missing = required_columns - set(author)
    if missing:
        raise RuntimeError(f"author map missing columns: {sorted(missing)}")
    calibration = author[
        author.source_level.astype(str).str.casefold().eq("level 1")
        & author.source_inchikey.fillna("").astype(str).str.len().ge(14)
    ].copy()
    calibration["ik14"] = calibration.source_inchikey.astype(str).str[:14]
    calibration_sets = {
        panel: set(calibration.loc[calibration.panel.eq(panel), "ik14"])
        for panel in PANELS
    }

    records: list[dict[str, object]] = []
    mgf_paths = {"neg_rp": args.reference_neg, "pos_rp": args.reference_pos}
    for panel, path in mgf_paths.items():
        for record, fields, precursor, n_peaks in iter_mgf_metadata(path):
            inchikey = fields.get("INCHIKEY", "")
            ik14 = inchikey[:14]
            if len(ik14) != 14:
                continue
            name = fields.get("NAME", "")
            smiles = fields.get("SMILES", "")
            family = FROZEN_HYPOTHESIS_IK14.get(ik14, "") or classify_family(
                name, smiles, precursor
            )
            is_calibration = ik14 in calibration_sets[panel]
            if not family and not is_calibration:
                continue
            records.append({
                "panel": panel,
                "mgf_record": record,
                "inchikey": inchikey,
                "ik14": ik14,
                "name": name,
                "smiles": smiles,
                "precursor_mz": precursor,
                "adduct": fields.get("ADDUCT", ""),
                "source": fields.get("SOURCE", ""),
                "n_peaks": n_peaks,
                "calibration_panel": bool(is_calibration),
                "hypothesis_family": family,
                "frozen_hypothesis_identity": ik14 in FROZEN_HYPOTHESIS_IK14,
                "frozen_hypothesis_feature_id": FROZEN_HYPOTHESIS_FEATURE_IDS.get(ik14),
                "clean_parent_reference": not (
                    "_" in name or any(marker in name.casefold() for marker in COMBINATORIAL_MARKERS)
                ),
                "chemistry_selection_basis": (
                    "frozen_hypothesis_identity" if ik14 in FROZEN_HYPOTHESIS_IK14
                    else ("strict_clean_parent_family" if family else "source_level1_calibration")
                ),
                "identity_claim_scope": (
                    "formula_or_positional_isomer_family_only"
                    if ik14 in {"IXOXBSCIXZEQEQ", "NNQCGMWOZJYFTM"}
                    else "reference_structure_candidate"
                ),
                "reference_status": "public_library_reference_spectrum",
                "same_platform_authentic_standard": False,
            })
    spectra = pd.DataFrame(records)
    if spectra.empty:
        raise RuntimeError("no reference probe spectrum matched the frozen panels")
    spectra = spectra.sort_values(
        ["panel", "ik14", "adduct", "n_peaks", "source", "mgf_record"],
        ascending=[True, True, True, False, True, True], kind="stable",
    )
    spectra["within_identity_adduct_rank"] = (
        spectra.groupby(["panel", "ik14", "adduct"], sort=False).cumcount() + 1
    )
    spectra = spectra[
        spectra.within_identity_adduct_rank.le(args.maximum_spectra_per_identity_adduct)
    ].reset_index(drop=True)
    spectra.insert(0, "reference_spectrum_id", [f"R{i:06d}" for i in range(len(spectra))])

    author_lookup = calibration.groupby(["panel", "ik14"], sort=False).agg(
        author_level1_names=("source_name", lambda s: ";".join(sorted(set(map(str, s))))),
        author_level1_feature_ids=("feature_id", lambda s: ";".join(map(str, sorted(set(map(int, s))))))
    ).reset_index()
    identities = spectra.groupby(["panel", "ik14"], sort=False).agg(
        inchikey=("inchikey", "first"),
        representative_name=("name", "first"),
        representative_smiles=("smiles", "first"),
        n_reference_spectra=("reference_spectrum_id", "size"),
        n_adducts=("adduct", "nunique"),
        calibration_panel=("calibration_panel", "max"),
        frozen_hypothesis_identity=("frozen_hypothesis_identity", "max"),
        frozen_hypothesis_feature_id=("frozen_hypothesis_feature_id", "first"),
        clean_parent_reference=("clean_parent_reference", "min"),
        hypothesis_family=("hypothesis_family", lambda s: ";".join(sorted(set(x for x in s if x)))),
    ).reset_index().merge(author_lookup, on=["panel", "ik14"], how="left", validate="one_to_one")
    identities["probe_scope"] = [
        "calibration_and_hypothesis" if calibration and family else
        ("calibration" if calibration else "hypothesis")
        for calibration, family in zip(identities.calibration_panel, identities.hypothesis_family)
    ]

    samples = []
    for panel in PANELS:
        panel_dir = args.sample_root / panel
        paths = sorted(panel_dir.glob("*.hdf5"))
        if not paths:
            raise FileNotFoundError(f"no HDF5 samples in {panel_dir}")
        samples.extend(sample_fields(panel, path) for path in paths)
    sample_frame = pd.DataFrame(samples)

    spectra_path = out / "reference_spectra.csv.gz"
    identity_path = out / "reference_identities.csv"
    sample_path = out / "sample_manifest.csv"
    spectra.to_csv(spectra_path, index=False, compression="gzip", quoting=csv.QUOTE_MINIMAL)
    identities.to_csv(identity_path, index=False)
    sample_frame.to_csv(sample_path, index=False)

    family_counts = {
        family: int(identities.hypothesis_family.fillna("").str.split(";").apply(lambda xs: family in xs).sum())
        for family in FAMILY_ORDER
    }
    calibration_requested = {panel: len(values) for panel, values in calibration_sets.items()}
    calibration_found = {
        panel: int(identities.panel.eq(panel).mul(identities.calibration_panel).sum())
        for panel in PANELS
    }
    report = {
        "status": "mtbls13729_reverse_probe_manifest_v4_frozen",
        "formal": True,
        "reference_spectra": int(len(spectra)),
        "reference_identities": int(len(identities)),
        "calibration_level1_identities_requested": calibration_requested,
        "calibration_level1_identities_with_public_reference": calibration_found,
        "hypothesis_family_identities": family_counts,
        "samples": sample_frame.groupby("panel").size().astype(int).to_dict(),
        "selection_used_phenotype_outcomes": False,
        "selection_contract": {
            "calibration": "source-paper Level-1 identity mapped before this analysis",
            "hypothesis": (
                "pre-existing frozen identity list plus strict clean-parent name rules; "
                "no Rmu-vs-RN statistic used"
            ),
            "synthetic_derivatives": (
                "underscore/known-isomer/isobaric combinatorial entries are excluded from the primary panel"
            ),
            "replicates": f"at most {args.maximum_spectra_per_identity_adduct} deterministic spectra per panel/IK14/adduct",
        },
        "identity_boundary": (
            "Unified-library entries are reference spectra, not same-platform authentic standards. "
            "Only later RT+MS/MS agreement with an injected standard can establish Level 1."
        ),
        "next_tensor": "reference spectrum x biological sample x evidence channel",
        "provenance": {
            "author_map_sha256": sha256(args.author_map),
            "negative_mgf_sha256": sha256(args.reference_neg),
            "positive_mgf_sha256": sha256(args.reference_pos),
            "reference_spectra_sha256": sha256(spectra_path),
            "reference_identities_sha256": sha256(identity_path),
            "sample_manifest_sha256": sha256(sample_path),
        },
        "gates": {
            "all_samples_present": len(sample_frame) == 119,
            "calibration_references_present": sum(calibration_found.values()) >= 50,
            "hypothesis_families_nonempty": all(value > 0 for value in family_counts.values()),
            "all_hypothesis_rows_are_clean_parents_or_frozen_identities": bool(
                spectra.loc[
                    spectra.hypothesis_family.fillna("").ne(""),
                    ["clean_parent_reference", "frozen_hypothesis_identity"],
                ].any(axis=1).all()
            ),
            "no_same_platform_standard_overclaim": not bool(spectra.same_platform_authentic_standard.any()),
        },
        "claim_limit": "Manifest only; no reverse-search hit, annotation improvement, abundance association, or mechanism result.",
    }
    report["pass_to_reverse_scan"] = all(report["gates"].values())
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
