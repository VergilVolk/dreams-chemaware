#!/usr/bin/env python3
"""Estimate and compare two structure-hidden longitudinal metabolomic responses.

The unit of inference is a biological subject.  Technical files are aggregated
inside subject x timepoint before an endpoint-minus-baseline change is formed.
Each feature is then fit with the same frozen factorial change-score model.  No
compound name, library identity, pathway, or cross-study truth is read.

The cross-study statistic is a signed kernel-mean score in a phenotype-blind
MS/MS space.  The current spectral kernel is a transparent sqrt-intensity,
high-resolution fragment-bin baseline.  It is deliberately labelled a baseline:
frozen DreaMS/Noise embeddings can replace it without changing the biological
effect model or the permutation contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def sample_stem(value: object) -> str:
    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    text = re.sub(r"\s+(?:filtered\s+)?Peak area$", "", text, flags=re.I)
    return Path(text).stem.casefold()


def read_quant(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, low_memory=False)
    normalized = {str(column).strip().casefold(): str(column) for column in frame.columns}
    id_column = next((normalized[name] for name in ("row id", "feature_id", "feature id", "row_id") if name in normalized), None)
    if id_column is None:
        raise ValueError(f"{path}: feature identifier column missing")
    sample_columns = [
        str(column) for column in frame.columns
        if re.search(r"\.mz(?:ml|xml)\s+(?:filtered\s+)?Peak area$", str(column), flags=re.I)
    ]
    if not sample_columns:
        raise ValueError(f"{path}: no quantitative sample columns")
    frame[id_column] = frame[id_column].map(feature_id)
    if frame[id_column].duplicated().any():
        raise ValueError(f"{path}: duplicate feature identifiers")
    values = frame.set_index(id_column)[sample_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or (values < 0).any().any():
        raise ValueError(f"{path}: values must be complete, numeric, and non-negative")
    return values, sample_columns


def read_mgf(path: Path) -> dict[str, dict[str, object]]:
    spectra: dict[str, dict[str, object]] = {}
    metadata: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    inside = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                metadata, peaks, inside = {}, [], True
            elif line == "END IONS" and inside:
                key = feature_id(metadata.get("FEATURE_ID") or metadata.get("SCANS") or "")
                if not key or key in spectra:
                    raise ValueError(f"{path}: missing or duplicate FEATURE_ID {key!r}")
                precursor = float(metadata.get("PEPMASS", "nan").split()[0])
                spectra[key] = {"precursor_mz": precursor, "peaks": peaks}
                inside = False
            elif inside and "=" in line and not line[:1].isdigit():
                key, value = line.split("=", 1)
                metadata[key.upper()] = value
            elif inside:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        mz, intensity = float(fields[0]), float(fields[1])
                    except ValueError:
                        continue
                    if math.isfinite(mz) and math.isfinite(intensity) and mz > 0 and intensity > 0:
                        peaks.append((mz, intensity))
    return spectra


def hc3_effect(y: np.ndarray, design: np.ndarray, coefficient: int) -> tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ y
    residual = y - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    adjusted = residual / np.maximum(1.0 - leverage[:, None], 1e-8)
    bread_row = xtx_inv[coefficient] @ design.T
    variance = np.sum((bread_row[:, None] * adjusted) ** 2, axis=0)
    return coefficients[coefficient], np.sqrt(np.maximum(variance, 1e-12))


def fit_study(config: dict[str, object]) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, object]]:
    quant_path = Path(str(config["quant_path"]))
    design_path = Path(str(config["design_path"]))
    mgf_path = Path(str(config["mgf_path"]))
    quant, sample_columns = read_quant(quant_path)
    design = pd.read_csv(design_path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    design["_sample_stem"] = design[str(config["filename_column"])].map(sample_stem)
    quant_stems = {sample_stem(column) for column in sample_columns}
    design = design.loc[design["_sample_stem"].isin(quant_stems)].copy()
    if design["_sample_stem"].duplicated().any():
        duplicated = design.loc[design["_sample_stem"].duplicated(False), "_sample_stem"].unique()
        raise ValueError(f"{design_path}: duplicate filenames, examples {duplicated[:3]}")
    design = design.set_index("_sample_stem")
    column_map = {column: sample_stem(column) for column in sample_columns}
    joined_columns = [column for column, stem in column_map.items() if stem in design.index]
    if not joined_columns:
        raise ValueError(f"{config['study_id']}: no quantitative column joins design")

    target = dict(config["target"])
    adjustments = [dict(item) for item in config.get("adjustments", [])]
    required = [str(config["unit_column"]), str(config["time_column"]), target["column"]]
    required += [item["column"] for item in adjustments]
    missing = set(required) - set(design.columns)
    if missing:
        raise ValueError(f"{config['study_id']}: design lacks {sorted(missing)}")

    sample_meta = design.loc[[column_map[column] for column in joined_columns]].copy()
    sample_meta["_quant_column"] = joined_columns
    allowed = {target["column"]: {target["reference"], target["exposed"]}}
    for item in adjustments:
        allowed[item["column"]] = {item["reference"], item["exposed"]}
    keep = sample_meta[str(config["time_column"])].isin([config["baseline_level"], config["endpoint_level"]])
    for column, values in allowed.items():
        keep &= sample_meta[column].isin(values)
    sample_meta = sample_meta.loc[keep].copy()

    unit_column = str(config["unit_column"])
    time_column = str(config["time_column"])
    factor_columns = [target["column"]] + [item["column"] for item in adjustments]
    for column in factor_columns:
        if (sample_meta.groupby(unit_column)[column].nunique() > 1).any():
            raise ValueError(f"{config['study_id']}: {column} changes within biological unit")

    # Normalize before any outcome effect is estimated.  The feature-specific
    # pseudocount is determined from all joined biological files, and each file
    # is median-centred to reduce total-ion/batch scale differences.
    selected = quant[sample_meta["_quant_column"].tolist()].to_numpy(np.float64)
    positive = np.where(selected > 0, selected, np.nan)
    active = np.any(selected > 0, axis=1)
    if not active.any():
        raise ValueError(f"{config['study_id']}: every feature is zero in retained files")
    quant = quant.iloc[active].copy()
    selected = selected[active]
    minimum = np.nanmin(positive[active], axis=1)
    transformed = np.log2(selected + minimum[:, None] / 2.0)
    transformed -= np.median(transformed, axis=0, keepdims=True)
    value_frame = pd.DataFrame(transformed, index=quant.index, columns=sample_meta["_quant_column"])

    # Median aggregation is only across technical files from the same
    # biological unit and timepoint; it never increases biological n.
    unit_time: dict[tuple[str, str], np.ndarray] = {}
    for (unit, time), block in sample_meta.groupby([unit_column, time_column], sort=True):
        unit_time[(str(unit), str(time))] = np.median(
            value_frame[block["_quant_column"].tolist()].to_numpy(), axis=1
        )
    units = sorted({unit for unit, _ in unit_time})
    paired = [
        unit for unit in units
        if (unit, str(config["baseline_level"])) in unit_time
        and (unit, str(config["endpoint_level"])) in unit_time
    ]
    if len(paired) < 20:
        raise ValueError(f"{config['study_id']}: only {len(paired)} complete biological units")
    change = np.stack([
        unit_time[(unit, str(config["endpoint_level"]))]
        - unit_time[(unit, str(config["baseline_level"]))]
        for unit in paired
    ])
    unit_meta = sample_meta.drop_duplicates(unit_column).set_index(unit_column).loc[paired]
    columns = [np.ones(len(paired), dtype=float)]
    design_names = ["intercept"]
    for item in [target] + adjustments:
        observed = set(unit_meta[item["column"]])
        expected = {item["reference"], item["exposed"]}
        if observed != expected:
            raise ValueError(f"{config['study_id']}: {item['column']} observed {observed}, expected {expected}")
        columns.append(unit_meta[item["column"]].eq(item["exposed"]).to_numpy(float))
        design_names.append(f"{item['column']}:{item['exposed']}-vs-{item['reference']}")
    model = np.column_stack(columns)
    if np.linalg.matrix_rank(model) != model.shape[1]:
        raise ValueError(f"{config['study_id']}: singular factorial model")
    beta, se = hc3_effect(change, model, coefficient=1)

    prevalence = (selected > 0).mean(axis=1)
    response = beta / np.sqrt(beta * beta + se * se)
    spectra = read_mgf(mgf_path)
    shared = [entity for entity in quant.index.astype(str) if entity in spectra]
    positions = {entity: index for index, entity in enumerate(quant.index.astype(str))}
    effects = pd.DataFrame({
        "study_id": str(config["study_id"]),
        "dataset_accession": str(config["dataset_accession"]),
        "source_family": str(config["source_family"]),
        "contrast_family": str(config.get("contrast_family", "")),
        "entity_id": shared,
        "beta_endpoint_change_exposed_minus_reference": [float(beta[positions[x]]) for x in shared],
        "hc3_se": [float(se[positions[x]]) for x in shared],
        "bounded_response": [float(response[positions[x]]) for x in shared],
        "prevalence": [float(prevalence[positions[x]]) for x in shared],
        "identity_status": "unresolved_entity",
    })
    report = {
        "study_id": config["study_id"],
        "dataset_accession": config["dataset_accession"],
        "quantitative_columns": len(sample_columns),
        "design_joined_biological_files": len(joined_columns),
        "model_files": len(sample_meta),
        "complete_biological_units": len(paired),
        "model_rank": int(np.linalg.matrix_rank(model)),
        "design_columns": design_names,
        "ms2_linked_entities": len(shared),
        "quant_features": len(quant),
        "inputs": {
            "quant_sha256": sha256(quant_path),
            "mgf_sha256": sha256(mgf_path),
            "design_sha256": sha256(design_path),
        },
    }
    return effects, {entity: spectra[entity] for entity in shared}, report


def spectral_matrix(
    entity_ids: list[str], spectra: dict[str, dict[str, object]], bin_width: float, max_mz: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    n_bins = int(math.ceil(max_mz / bin_width))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    mass_rows: list[int] = []
    mass_cols: list[int] = []
    for row, entity in enumerate(entity_ids):
        record = spectra[entity]
        aggregate: dict[int, float] = {}
        for mz, intensity in record["peaks"]:
            column = int(float(mz) / bin_width)
            if 0 <= column < n_bins:
                aggregate[column] = aggregate.get(column, 0.0) + math.sqrt(float(intensity))
        norm = math.sqrt(sum(value * value for value in aggregate.values()))
        if norm > 0:
            for column, value in aggregate.items():
                rows.append(row); cols.append(column); data.append(value / norm)
        precursor = float(record["precursor_mz"])
        column = int(precursor / bin_width)
        if math.isfinite(precursor) and 0 <= column < n_bins:
            mass_rows.append(row); mass_cols.append(column)
    shape = (len(entity_ids), n_bins)
    fragments = sparse.csr_matrix((np.asarray(data, np.float32), (rows, cols)), shape=shape)
    precursor = sparse.csr_matrix((np.ones(len(mass_rows), np.float32), (mass_rows, mass_cols)), shape=shape)
    return fragments, precursor


def sketch(matrix: sparse.csr_matrix, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(response, 0.0)
    negative = np.maximum(-response, 0.0)

    def integrate(weight: np.ndarray) -> np.ndarray:
        total = float(weight.sum())
        if total <= 0:
            return np.zeros(matrix.shape[1], dtype=np.float32)
        return np.asarray(weight @ matrix / total).ravel().astype(np.float32)

    return integrate(positive), integrate(negative)


def signed_score(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    lp, ln = left; rp, rn = right
    pp = float(lp @ rp); nn = float(ln @ rn); pn = float(lp @ rn); np_ = float(ln @ rp)
    return {
        "positive_positive": pp,
        "negative_negative": nn,
        "positive_negative": pn,
        "negative_positive": np_,
        "signed_response_score": 0.5 * ((pp + nn) - (pn + np_)),
        "unsigned_response_score": 0.5 * ((pp + nn) + (pn + np_)),
    }


def permutation_test(
    matrices: list[sparse.csr_matrix], responses: list[np.ndarray], observed: float,
    permutations: int, rng: np.random.Generator,
) -> dict[str, float | int]:
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = [rng.permutation(response) for response in responses]
        null[index] = signed_score(
            sketch(matrices[0], shuffled[0]), sketch(matrices[1], shuffled[1])
        )["signed_response_score"]
    return {
        "permutations": permutations,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "z_vs_shuffle": float((observed - null.mean()) / max(null.std(ddof=1), 1e-12)),
        "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q99": float(np.quantile(null, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--max-mz", type=float, default=1200.0)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if str(config.get("status", "active")) != "active":
        raise ValueError(
            f"configuration is not active: {config.get('status')} - "
            f"{config.get('invalidation_reason', 'no reason recorded')}"
        )
    if len(config.get("studies", [])) != 2:
        raise ValueError("this confirmatory pair implementation requires exactly two studies")
    for study in config["studies"]:
        study["contrast_family"] = config["contrast_family"]

    fitted = [fit_study(study) for study in config["studies"]]
    effects = [item[0] for item in fitted]
    spectra = [item[1] for item in fitted]
    study_reports = [item[2] for item in fitted]
    fragment_matrices: list[sparse.csr_matrix] = []
    precursor_matrices: list[sparse.csr_matrix] = []
    responses: list[np.ndarray] = []
    for effect, spectrum in zip(effects, spectra):
        entities = effect["entity_id"].astype(str).tolist()
        fragment, precursor = spectral_matrix(entities, spectrum, args.bin_width, args.max_mz)
        fragment_matrices.append(fragment)
        precursor_matrices.append(precursor)
        responses.append(effect["bounded_response"].to_numpy(float) * np.sqrt(effect["prevalence"].to_numpy(float)))

    fragment_score = signed_score(
        sketch(fragment_matrices[0], responses[0]), sketch(fragment_matrices[1], responses[1])
    )
    precursor_score = signed_score(
        sketch(precursor_matrices[0], responses[0]), sketch(precursor_matrices[1], responses[1])
    )
    rng = np.random.default_rng(args.seed)
    fragment_null = permutation_test(
        fragment_matrices, responses, fragment_score["signed_response_score"], args.permutations, rng
    )
    precursor_null = permutation_test(
        precursor_matrices, responses, precursor_score["signed_response_score"], args.permutations, rng
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(effects, ignore_index=True).to_csv(args.output_dir / "entity_effects.tsv", sep="\t", index=False)
    pd.DataFrame(study_reports).to_json(args.output_dir / "study_reports.json", orient="records", indent=2)
    score_rows = []
    for representation, score, null in (
        ("sqrt_intensity_fragment_bins", fragment_score, fragment_null),
        ("precursor_mass_bins", precursor_score, precursor_null),
    ):
        score_rows.append({"representation": representation, **score, **null})
    pd.DataFrame(score_rows).to_csv(args.output_dir / "pair_scores.tsv", sep="\t", index=False)
    report = {
        "schema": "factorial_longitudinal_signed_response_pair_v1",
        "status": "development_positive_control_not_external_validation",
        "contrast_family": config["contrast_family"],
        "effect_direction": config["effect_direction"],
        "denominators": {
            "studies": 2,
            "independent_source_families": 2,
            "complete_biological_units": int(sum(row["complete_biological_units"] for row in study_reports)),
            "ms2_linked_unresolved_entities": int(sum(len(frame) for frame in effects)),
            "permutations_per_representation": args.permutations,
        },
        "study_reports": study_reports,
        "fragment_score": {**fragment_score, **fragment_null},
        "precursor_only_score": {**precursor_score, **precursor_null},
        "leakage_status": (
            "feature_spectra_normalization_and_factorial_models_fixed_without_compound_names_pathways_or_cross_study_score_tuning"
        ),
        "claim_boundary": [
            "All entities are treated as structure-unresolved; this run cannot estimate known-only versus unresolved increment.",
            "The spectral representation is a transparent fragment-bin baseline, not frozen Noise V1 or the final unified encoder.",
            "A two-study positive control tests whether a signed response is detectable, not repository-scale generalization.",
            "Biological interpretation requires source-paper adjudication and replication beyond these two source families.",
        ],
        "inputs": {"config": str(args.config.resolve()), "config_sha256": sha256(args.config)},
        "parameters": {
            "bin_width_da": args.bin_width,
            "max_fragment_mz": args.max_mz,
            "permutations": args.permutations,
            "seed": args.seed,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
