#!/usr/bin/env python3
"""Build the fixed IBD Plate3 handoff for algorithm-to-biology evaluation.

The query and candidate files are phenotype blind.  Disease labels are used
only to create a deterministic stratified outer split and the separately
sealed held-out effect table.  BioAware context for one outer fold must be
estimated from the complementary participants; its biological response is
evaluated only on participants in the held-out fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_ibd_contrast_atlas_source_artifacts import (
    MIN_BIO_PREVALENCE,
    MIN_BLANK_FOLD,
    _adult_sample_map,
    parse_mgf,
    robust_clr_by_sample,
    welch_effect,
)


STUDY_ID = "IBD200_GROUP"
MINIMUM_FRAGMENT_PEAKS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_stratified_folds(sample_ids: list[str], labels: list[str]) -> np.ndarray:
    """Assign two balanced folds without using any measured feature value."""
    folds = np.full(len(sample_ids), -1, dtype=np.int8)
    frame = pd.DataFrame({"index": np.arange(len(sample_ids)), "sample_id": sample_ids, "label": labels})
    for _, block in frame.groupby("label", sort=True):
        ordered = block.assign(
            key=block["sample_id"].map(
                lambda value: hashlib.sha256(f"ibd200-plate3-v1|{value}".encode()).hexdigest()
            )
        ).sort_values(["key", "sample_id"])
        folds[ordered["index"].to_numpy(int)] = np.arange(len(ordered)) % 2
    if (folds < 0).any():
        raise RuntimeError("outer fold assignment is incomplete")
    return folds


def wrong_context_map(
    entity_ids: np.ndarray,
    values: np.ndarray,
    detected: np.ndarray,
    train_mask: np.ndarray,
    fold: int,
) -> pd.DataFrame:
    """Permute whole feature profiles within exact detection-degree strata.

    A column permutation preserves every training sample's feature coverage.
    Restricting rotations to exact degree strata also preserves feature degree;
    sorting by variance makes the donor adjacent in marginal variability.
    """
    train_values = values[train_mask]
    train_detected = detected[train_mask]
    degree = train_detected.sum(axis=0).astype(int)
    variance = np.var(train_values, axis=0, ddof=1)
    donor = np.arange(len(entity_ids))
    for observed_degree in np.unique(degree):
        indices = np.flatnonzero(degree == observed_degree)
        indices = indices[np.argsort(variance[indices], kind="mergesort")]
        if len(indices) > 1:
            donor[indices] = np.roll(indices, -1)
    return pd.DataFrame({
        "outer_fold": fold,
        "recipient_entity_id": entity_ids,
        "donor_entity_id": entity_ids[donor],
        "training_detection_degree": degree,
        "recipient_training_variance": variance,
        "donor_training_variance": variance[donor],
        "fixed_point": donor == np.arange(len(entity_ids)),
    })


def candidate_tables(
    queries: pd.DataFrame,
    library_manifest: Path,
    precursor_ppm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "row", "inchikey", "smiles", "formula", "exact_mass", "precursor_mz",
        "quality_label", "n_peaks", "spectrum_id",
    ]
    library = pd.read_csv(library_manifest, usecols=columns)
    library = library.loc[
        library["inchikey"].notna()
        & library["smiles"].notna()
        & np.isfinite(pd.to_numeric(library["precursor_mz"], errors="coerce"))
    ].copy()
    library["precursor_mz"] = library["precursor_mz"].astype(float)
    order = np.argsort(library["precursor_mz"].to_numpy(), kind="mergesort")
    sorted_mz = library["precursor_mz"].to_numpy()[order]
    spectrum_rows: list[pd.DataFrame] = []
    for query in queries.itertuples(index=False):
        precursor = float(query.precursor_mz)
        delta = precursor * precursor_ppm * 1e-6
        lo = np.searchsorted(sorted_mz, precursor - delta, side="left")
        hi = np.searchsorted(sorted_mz, precursor + delta, side="right")
        if lo == hi:
            continue
        selected = library.iloc[order[lo:hi]].copy()
        selected.insert(0, "entity_id", str(query.entity_id))
        selected.insert(1, "query_precursor_mz", precursor)
        selected["precursor_ppm"] = (
            1e6 * (selected["precursor_mz"] - precursor).abs() / precursor
        )
        selected["candidate_id"] = selected["inchikey"].astype(str)
        spectrum_rows.append(selected)
    if spectrum_rows:
        spectra = pd.concat(spectrum_rows, ignore_index=True)
    else:
        spectra = pd.DataFrame(columns=["entity_id", "candidate_id", *columns])
    structures = (
        spectra.sort_values(
            ["entity_id", "candidate_id", "precursor_ppm", "quality_label", "row"],
            ascending=[True, True, True, True, True],
        )
        .groupby(["entity_id", "candidate_id"], sort=False, as_index=False)
        .agg(
            smiles=("smiles", "first"),
            formula=("formula", "first"),
            exact_mass=("exact_mass", "first"),
            query_precursor_mz=("query_precursor_mz", "first"),
            minimum_precursor_ppm=("precursor_ppm", "min"),
            library_spectra=("row", "size"),
        )
    )
    return spectra, structures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repaired-mgf", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--library-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    input_paths = [
        args.quant, args.metadata, args.repaired_mgf, args.library_manifest,
        args.collection / "response_entities.tsv.gz", args.collection / "report.json",
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    quant = pd.read_csv(args.quant)
    metadata_raw = pd.read_csv(args.metadata, sep="\t", dtype=str)
    metadata, biological_columns, blank_columns = _adult_sample_map(quant, metadata_raw)
    spectra = parse_mgf(args.repaired_mgf)
    raw_bio = quant[biological_columns].to_numpy(float)
    raw_blank = (
        quant[blank_columns].to_numpy(float)
        if blank_columns else np.zeros((len(quant), 0), dtype=float)
    )
    feature_ids = quant["row ID"].astype(int).to_numpy()
    quant_mz = quant["row m/z"].astype(float).to_numpy()
    prevalence = np.mean(raw_bio > 0, axis=1)
    mean_bio = np.mean(raw_bio, axis=1)
    mean_blank = np.mean(raw_blank, axis=1) if raw_blank.shape[1] else np.zeros(len(quant))
    qc = (
        np.isfinite(quant_mz)
        & (prevalence >= MIN_BIO_PREVALENCE)
        & (mean_bio > 0)
        & (mean_bio >= MIN_BLANK_FOLD * mean_blank)
        & np.asarray([int(value) in spectra for value in feature_ids])
    )
    qc_ids = feature_ids[qc].astype(str)
    values = robust_clr_by_sample(raw_bio[qc])
    detected = raw_bio[qc] > 0
    id_to_row = {entity: row for row, entity in enumerate(qc_ids)}

    response = pd.read_csv(
        args.collection / "response_entities.tsv.gz", sep="\t", dtype={"raw_entity_id": str},
    )
    response = response.loc[response["study_id"].astype(str).eq(STUDY_ID)].copy()
    if response.empty or response["raw_entity_id"].duplicated().any():
        raise RuntimeError("IBD200 response registry is empty or duplicated")
    if not set(response["raw_entity_id"]).issubset(id_to_row):
        raise RuntimeError("Plate3 response registry is outside the frozen QC universe")
    row_index = np.asarray([id_to_row[value] for value in response["raw_entity_id"]], dtype=int)

    plate3 = metadata["Metabolomics-Plate"].eq("Plate3").to_numpy()
    plate3_metadata = metadata.loc[plate3].reset_index(drop=True)
    sample_ids = plate3_metadata["quant_stem"].astype(str).tolist()
    labels = np.where(
        plate3_metadata["Diagnosis"].isin(["CD", "UC"]), "IBD", "healthy"
    ).tolist()
    if labels.count("IBD") != 36 or labels.count("healthy") != 19:
        raise RuntimeError("Plate3 participant denominator drift")
    folds = deterministic_stratified_folds(sample_ids, labels)
    context_values = values[row_index][:, plate3].T.astype(np.float32)
    context_detected = detected[row_index][:, plate3].T.astype(bool)
    entity_ids = response["entity_id"].astype(str).to_numpy(dtype=str)

    query_rows = []
    for entity, raw_entity, precursor in zip(
        entity_ids, response["raw_entity_id"].astype(str),
        quant_mz[[np.flatnonzero(feature_ids.astype(str) == raw)[0] for raw in response["raw_entity_id"].astype(str)]],
    ):
        spectrum = spectra[int(raw_entity)]
        fragment_peaks = int(spectrum.peaks.shape[1])
        query_rows.append({
            "study_id": STUDY_ID,
            "entity_id": entity,
            "raw_entity_id": raw_entity,
            "precursor_mz": float(precursor),
            "fragment_peaks": fragment_peaks,
            "spectral_eligible": bool(fragment_peaks >= MINIMUM_FRAGMENT_PEAKS),
        })
    queries = pd.DataFrame(query_rows)
    candidate_spectra, candidate_structures = candidate_tables(
        queries, args.library_manifest, args.precursor_ppm,
    )
    structure_counts = candidate_structures.groupby("entity_id").size()
    spectrum_counts = candidate_spectra.groupby("entity_id").size()
    queries["candidate_structures"] = queries["entity_id"].map(structure_counts).fillna(0).astype(int)
    queries["candidate_library_spectra"] = queries["entity_id"].map(spectrum_counts).fillna(0).astype(int)
    queries["algorithm_answerable"] = queries["spectral_eligible"] & queries["candidate_structures"].gt(0)

    sample_folds = pd.DataFrame({
        "sample_id": sample_ids,
        "outer_fold": folds,
    })
    sealed_rows: list[pd.DataFrame] = []
    wrong_rows: list[pd.DataFrame] = []
    label_array = np.asarray(labels)
    for fold in (0, 1):
        held_out = folds == fold
        train = ~held_out
        beta, se = welch_effect(
            context_values.T.astype(float),
            held_out & (label_array == "IBD"),
            held_out & (label_array == "healthy"),
        )
        estimable = np.isfinite(beta) & np.isfinite(se) & (se > 0)
        sealed_rows.append(pd.DataFrame({
            "outer_fold": fold,
            "study_id": STUDY_ID,
            "contrast_id": "adult_all_ibd_vs_healthy",
            "entity_id": entity_ids,
            "beta": beta,
            "se": se,
            "effect_estimable": estimable,
            "n_positive": int(np.sum(held_out & (label_array == "IBD"))),
            "n_negative": int(np.sum(held_out & (label_array == "healthy"))),
        }))
        wrong_rows.append(
            wrong_context_map(entity_ids, context_values, context_detected, train, fold)
        )
    sealed = pd.concat(sealed_rows, ignore_index=True)
    wrong = pd.concat(wrong_rows, ignore_index=True)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    outputs = {
        "queries": args.output_dir / "query_entities.tsv.gz",
        "candidate_spectra": args.output_dir / "candidate_library_spectra.tsv.gz",
        "candidate_structures": args.output_dir / "candidate_structures.tsv.gz",
        "sample_folds": args.output_dir / "phenotype_blind_context_folds.tsv",
        "wrong_context": args.output_dir / "degree_preserving_wrong_context.tsv.gz",
        "sealed_effects": args.output_dir / "sealed_heldout_effects.tsv.gz",
        "context": args.output_dir / "phenotype_blind_sample_context.npz",
    }
    queries.to_csv(outputs["queries"], sep="\t", index=False, compression=compression)
    candidate_spectra.to_csv(
        outputs["candidate_spectra"], sep="\t", index=False, compression=compression,
    )
    candidate_structures.to_csv(
        outputs["candidate_structures"], sep="\t", index=False, compression=compression,
    )
    sample_folds.to_csv(outputs["sample_folds"], sep="\t", index=False)
    wrong.to_csv(outputs["wrong_context"], sep="\t", index=False, compression=compression)
    sealed.to_csv(outputs["sealed_effects"], sep="\t", index=False, compression=compression)
    np.savez_compressed(
        outputs["context"],
        schema=np.asarray("ibd200_plate3_phenotype_blind_sample_context_v1"),
        sample_id=np.asarray(sample_ids, dtype=str),
        entity_id=entity_ids,
        robust_clr=context_values,
        detected=context_detected,
        outer_fold=folds,
    )

    fold_table = (
        pd.DataFrame({"fold": folds, "label": labels})
        .groupby(["fold", "label"]).size().unstack(fill_value=0)
    )
    report = {
        "schema": "ibd200_plate3_algorithm_application_slice_v1",
        "status": "DEVELOPMENT_CONSUMED_ALGORITHM_BIOLOGY_HANDOFF_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "plate3_participants": len(sample_ids),
            "plate3_ibd": labels.count("IBD"),
            "plate3_healthy": labels.count("healthy"),
            "response_entities": len(queries),
            "spectral_eligible": int(queries["spectral_eligible"].sum()),
            "spectral_ineligible": int((~queries["spectral_eligible"]).sum()),
            "entities_with_mass_compatible_candidates": int(queries["candidate_structures"].gt(0).sum()),
            "algorithm_answerable": int(queries["algorithm_answerable"].sum()),
            "candidate_structures": len(candidate_structures),
            "candidate_library_spectra": len(candidate_spectra),
            "fold_label_counts": {
                str(index): {str(key): int(value) for key, value in row.items()}
                for index, row in fold_table.to_dict("index").items()
            },
            "heldout_estimable_by_fold": {
                str(fold): int(block["effect_estimable"].sum())
                for fold, block in sealed.groupby("outer_fold")
            },
            "wrong_context_fixed_points_by_fold": {
                str(fold): int(block["fixed_point"].sum())
                for fold, block in wrong.groupby("outer_fold")
            },
        },
        "fixed_comparison_contract": {
            "candidate_universe": "identical 10-ppm GNPS Gold/Silver structure candidates for WSE, B, integrated spectral and BioAware arms",
            "query_denominator": "all 662 corrected Plate3 response entities; unavailable spectra or empty candidate pools are abstentions, not deletions",
            "bioaware_cross_fit": "context from complementary participants; response evaluated only on held-out participants",
            "wrong_context": "whole entity-profile permutation within exact training detection-degree strata; sample coverage exactly preserved",
            "outcome_isolation": "query, candidate and context artifacts contain no diagnosis or beta/SE; outcomes are confined to sealed_heldout_effects.tsv.gz",
        },
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "claim_boundary": [
            "Plate3 removes perfect plate/healthy confounding but is not an independent cohort.",
            "Mass-compatible candidates are a frozen evaluation universe, not correct structures.",
            "Unresolved entities and algorithm predictions are not metabolite identifications.",
            "This handoff enables a fair incremental test; it does not itself establish BioAware or biological benefit.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
