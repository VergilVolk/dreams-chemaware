#!/usr/bin/env python3
"""Prepare three ChemProp accessions as study-level Contrast Atlas sources.

The response estimand is fixed to the bacteria-versus-abiotic T2-minus-T0
difference already used by the consumed ChemProp study-level experiment.  This
script only standardizes within-study beta/SE objects and phenotype-blind MS/MS
eligibility.  It does not read cross-study matches, retrieval scores, or drug
ranking results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_chemprop2_response_reproducibility_m1 import parse_samples
from analyze_chemprop_cross_design_response_retrieval import (
    BATCH_DRUGS,
    get_column,
    quant_column_map,
)
from analyze_chemprop_study_level_contrast_query import (
    query_components,
    reference_components,
    sha256,
)
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    iter_mgf,
    normalise_entity_id,
    sparse_spectrum,
    spectrum_id,
)


STATUS = "CHEMPROP_CONTRAST_ATLAS_SOURCE_PREPARED"


def nonempty_mgf_ids(path: Path) -> tuple[set[str], int, int]:
    ids: set[str] = set()
    total = 0
    for metadata, peaks in iter_mgf(path):
        total += 1
        entity = spectrum_id(metadata)
        if sparse_spectrum(peaks, 0.1)[0]:
            if entity in ids:
                raise ValueError(f"duplicate nonempty MGF entity {entity} in {path}")
            ids.add(entity)
    if not ids:
        raise ValueError(f"no nonempty spectra in {path}")
    return ids, total, len(ids)


def normalized_quant_ids(quant: pd.DataFrame) -> list[str]:
    if "row ID" not in quant:
        raise ValueError("quantification table lacks row ID")
    values = [normalise_entity_id(value) for value in quant["row ID"]]
    if len(values) != len(set(values)):
        raise ValueError("normalized row IDs are not unique")
    return values


def query_prevalence(quant: pd.DataFrame) -> np.ndarray:
    samples = parse_samples(list(quant.columns))
    columns = samples["column"].astype(str).tolist()
    if not columns:
        raise ValueError("no ChemProp2 query samples parsed")
    return (quant[columns].to_numpy(float) > 0).mean(axis=1)


def reference_prevalence(quant: pd.DataFrame, root: Path, batch: str) -> np.ndarray:
    metadata = pd.read_csv(root / "qiime2_output/qiime2_metadata.tsv", sep="\t")
    metadata["filename"] = metadata["filename"].astype(str).str.strip()
    mapping = quant_column_map(quant)
    metadata = metadata.loc[metadata["filename"].isin(mapping)].copy()
    drug_column = "ATTRIBUTE_Drug" if batch == "23drug" else "ATTRIBUTE_drug"
    metadata = metadata.loc[metadata[drug_column].isin(BATCH_DRUGS[batch])]
    columns = [get_column(mapping, value) for value in metadata["filename"]]
    columns = list(dict.fromkeys(columns))
    if not columns:
        raise ValueError(f"no included reference samples for {batch}")
    return (quant[columns].to_numpy(float) > 0).mean(axis=1)


def standardize_effects(
    study_id: str,
    quant: pd.DataFrame,
    components: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    prevalence: np.ndarray,
    eligible_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    ids = normalized_quant_ids(quant)
    if len(prevalence) != len(ids):
        raise ValueError("prevalence and quantification rows differ")
    positions = np.asarray([index for index, value in enumerate(ids) if value in eligible_ids])
    if not len(positions):
        raise ValueError(f"no nonempty MS/MS-linked features for {study_id}")
    rows = []
    zero_se = 0
    for drug in sorted(components):
        mean, se = components[drug]["response_signed"]
        if len(mean) != len(ids) or len(se) != len(ids):
            raise ValueError(f"component length drift for {study_id}/{drug}")
        selected_se = np.asarray(se, dtype=float)[positions]
        zero_se += int((selected_se <= 0).sum())
        selected_se = np.maximum(selected_se, 1e-8)
        block = pd.DataFrame({
            "contrast_id": "drug_" + drug.lower(),
            "entity_id": np.asarray(ids)[positions],
            "beta": np.asarray(mean, dtype=float)[positions],
            "se": selected_se,
            "prevalence": np.asarray(prevalence, dtype=float)[positions],
            "effect_model": "(bacteria_minus_abiotic)_t2_minus_t0",
        })
        rows.append(block)
    output = pd.concat(rows, ignore_index=True)
    if not np.isfinite(output[["beta", "se", "prevalence"]].to_numpy(float)).all():
        raise ValueError(f"non-finite effects for {study_id}")
    return output, {
        "quantification_features": len(ids),
        "nonempty_mgf_features": len(eligible_ids),
        "response_features": len(positions),
        "contrasts": len(components),
        "response_rows": len(output),
        "zero_se_values_floored": zero_se,
    }


def write_source(
    output_root: Path,
    study_id: str,
    effects: pd.DataFrame,
    mgf_path: Path,
    quant_path: Path,
    denominators: dict[str, int],
    biological_units: int,
) -> dict[str, object]:
    root = output_root / study_id
    root.mkdir(parents=True)
    effects_path = root / "effects.tsv"
    identity_path = root / "source_native_identity.tsv"
    effects.to_csv(effects_path, sep="\t", index=False)
    effects[["entity_id"]].drop_duplicates().assign(
        identity_status="unresolved_entity"
    ).to_csv(identity_path, sep="\t", index=False)
    report = {
        "schema": "chemprop_contrast_atlas_source_v1",
        "status": STATUS,
        "study_id": study_id,
        "data_role": "development_consumed",
        "estimand": "(bacteria T2 - abiotic T2) - (bacteria T0 - abiotic T0)",
        "biological_units": biological_units,
        "denominators": denominators,
        "leakage_status": (
            "within-study sample metadata define frozen beta and SE; MS/MS eligibility is "
            "phenotype blind; no cross-study feature map, drug retrieval result, or relation "
            "rank was read"
        ),
        "provenance": {
            "quantification_sha256": sha256(quant_path),
            "mgf_sha256": sha256(mgf_path),
            "effects_sha256": sha256(effects_path),
            "source_native_identity_sha256": sha256(identity_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "The three accessions come from the consumed ChemProp project and are not external validation.",
            "Source-native unresolved status means structure labels were not used here; the common strict library partition must be rebuilt separately.",
            "Drug identity defines registered intervention relevance only after every cross-study candidate score is frozen.",
        ],
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "study_id": study_id,
        "effects_path": str(effects_path),
        "identity_path": str(identity_path),
        "mgf_path": str(mgf_path),
        "report_path": str(report_path),
        **denominators,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-root", type=Path,
        default=Path("data/reference/chemprop2_m1_20261011"),
    )
    parser.add_argument(
        "--endpoint-root", type=Path,
        default=Path("data/reference/chemprop_cross_design_20261011"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    specifications = []
    query_quant_path = args.time_root / "quantification.csv"
    query_mgf_path = args.time_root / "consensus.mgf"
    query_quant = pd.read_csv(query_quant_path)
    query_mgf_ids, query_mgf_total, query_nonempty = nonempty_mgf_ids(query_mgf_path)
    query_effects, query_denominators = standardize_effects(
        "MSV000094899",
        query_quant,
        query_components(query_quant),
        query_prevalence(query_quant),
        query_mgf_ids,
    )
    query_denominators["mgf_spectra_total"] = query_mgf_total
    query_denominators["mgf_spectra_nonempty"] = query_nonempty
    specifications.append((
        "MSV000094899", query_effects, query_mgf_path, query_quant_path,
        query_denominators, 36,
    ))

    for batch, study_id, biological_units in (
        ("23drug", "MSV000096724", 6),
        ("45drug", "MSV000093571", 27),
    ):
        root = args.endpoint_root / f"chemprop_{batch}"
        quant_path = next((root / "quantification_table_reformatted").glob("*.csv"))
        mgf_path = root / "spectra/specs_ms.mgf"
        quant, components = reference_components(root, batch)
        mgf_ids, mgf_total, mgf_nonempty = nonempty_mgf_ids(mgf_path)
        effects, denominators = standardize_effects(
            study_id,
            quant,
            components,
            reference_prevalence(quant, root, batch),
            mgf_ids,
        )
        denominators["mgf_spectra_total"] = mgf_total
        denominators["mgf_spectra_nonempty"] = mgf_nonempty
        specifications.append((
            study_id, effects, mgf_path, quant_path, denominators, biological_units,
        ))

    args.output_dir.mkdir(parents=True)
    sources = [write_source(args.output_dir, *values) for values in specifications]
    registry = pd.DataFrame(sources)
    registry_path = args.output_dir / "source_registry.tsv"
    registry.to_csv(registry_path, sep="\t", index=False)
    report = {
        "schema": "chemprop_contrast_atlas_sources_v1",
        "status": "CHEMPROP_CONTRAST_ATLAS_SOURCES_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "independent_accessions": len(sources),
            "study_level_contrasts": int(registry["contrasts"].sum()),
            "exact_cross_accession_drug_relations": 12,
            "response_features": int(registry["response_features"].sum()),
            "response_rows": int(registry["response_rows"].sum()),
            "biological_units": int(sum(values[-1] for values in specifications)),
        },
        "leakage_status": (
            "source preparation reads within-study quantification, sample metadata, and MS/MS only; "
            "it does not read prior cross-design pair manifests or retrieval outputs"
        ),
        "source_registry_sha256": sha256(registry_path),
        "implementation_sha256": sha256(Path(__file__)),
        "sources": sources,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
