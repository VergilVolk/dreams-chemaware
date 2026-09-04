"""Spatially localize frozen MTBLS13729-linked metabolic programs in GSE236697.

The experiment contains one mucinous CRC tumour and its matched adjacent normal
tissue.  Spots are therefore localization units, not independent biological
replicates.  This script reports descriptive effect sizes and partial spatial
correlations without treating spot counts as population-level evidence.

All gene sets and source-study compartment markers are fixed in this file.  The
source paper's Space Ranger QC threshold (>100 detected genes) is reproduced.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.stats import rankdata, spearmanr


AXES = {
    "purine_synthesis_salvage": [
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT",
        "XDH", "ADA", "ADK",
    ],
    "carnitine_long_chain_fao": [
        "CPT1A", "CPT2", "SLC25A20", "ACADVL", "ACADM", "ACADS", "HADHA",
        "HADHB", "ETFA", "ETFB", "ETFDH", "CRAT", "CROT",
    ],
    "polyamine_synthesis": ["ODC1", "AMD1", "SRM", "SMS", "DHPS", "DOHH"],
    "polyamine_acetylation_catabolism": ["SAT1", "PAOX", "SMOX"],
    "acidity_lactate_response": ["CA9", "LDHA", "SLC16A3", "SLC2A1", "PDK1", "HIF1A"],
    "neutrophil_recruitment": ["CXCL1", "CXCL2", "CXCL5", "CXCL8", "CSF3"],
}

# Markers are transcribed from the source paper's spatial-transcriptomics section.
TUMOUR_COMPARTMENTS = {
    "tumour_epithelial": ["KRT20", "CLDN4", "CDH1"],
    "fibroblast": ["GPX2", "PLA2G2A"],
    "goblet": ["TFF3", "FCGBP"],
    "cancer_associated_fibroblast": ["MYL9", "TAGLN", "POSTN"],
    "immune": ["MALAT1", "IGHG1"],
    "monocyte_macrophage": ["IL1B", "CCL3", "S100A9"],
}

SINGLE_GENES = ["MUC1", "SAT1", "PAOX", "SMOX", "MS4A4A", "FGF7", "THBS1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_lines(path: Path) -> list[list[str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t") for line in handle]


def read_positions(path: Path) -> dict[str, tuple[int, int, int, float, float]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].lower() == "barcode":
                continue
            if len(row) < 6:
                raise RuntimeError(f"invalid spatial position row in {path}: {row}")
            result[row[0]] = (
                int(row[1]), int(row[2]), int(row[3]), float(row[4]), float(row[5])
            )
    return result


def load_sample(raw_dir: Path, prefix: str, condition: str) -> dict:
    paths = {
        "barcodes": raw_dir / f"{prefix}.barcodes.tsv.gz",
        "features": raw_dir / f"{prefix}.features.tsv.gz",
        "matrix": raw_dir / f"{prefix}.matrix.mtx.gz",
        "positions": raw_dir / f"{prefix}.tissue_positions_list.csv.gz",
    }
    if prefix == "GSM7573205_p1":
        paths["image"] = raw_dir / "GSM7573205_p1.tissue_lowres_image.png.gz"
        paths["scalefactors"] = raw_dir / "GSM7573205_p1.scalefactors_json.json.gz"
    else:
        paths["image"] = raw_dir / "GSM8286350_p1.tissue_lowres_image.png.gz"
        paths["scalefactors"] = raw_dir / "GSM8286350_p1_N.scalefactors_json.json.gz"
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    features = read_gzip_lines(paths["features"])
    barcodes = [row[0] for row in read_gzip_lines(paths["barcodes"])]
    symbols = [
        (row[1] if len(row) > 1 else row[0]).upper()
        for row in features
    ]
    matrix = mmread(paths["matrix"]).tocsr()
    if matrix.shape != (len(symbols), len(barcodes)):
        raise RuntimeError(
            f"{condition} matrix shape {matrix.shape} != features/barcodes "
            f"{(len(symbols), len(barcodes))}; download may be incomplete"
        )
    positions = read_positions(paths["positions"])
    missing_positions = sorted(set(barcodes) - set(positions))
    if missing_positions:
        raise RuntimeError(f"{condition}: {len(missing_positions)} barcodes lack positions")

    library_size = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    detected = np.asarray(matrix.getnnz(axis=0)).ravel().astype(int)
    in_tissue = np.asarray([positions[barcode][0] == 1 for barcode in barcodes])
    qc = in_tissue & (detected > 100) & (library_size > 0)
    if qc.sum() < 500:
        raise RuntimeError(f"{condition}: only {qc.sum()} in-tissue QC spots")

    matrix = matrix[:, qc]
    kept_barcodes = [barcode for barcode, keep in zip(barcodes, qc) if keep]
    library_size = library_size[qc]
    detected = detected[qc]
    normalizer = sparse.diags(10_000.0 / library_size)
    lognorm = matrix @ normalizer
    lognorm.data = np.log1p(lognorm.data)

    row_map: dict[str, list[int]] = {}
    for row, symbol in enumerate(symbols):
        row_map.setdefault(symbol, []).append(row)
    return {
        "condition": condition,
        "paths": paths,
        "symbols": symbols,
        "row_map": row_map,
        "matrix": matrix,
        "lognorm": lognorm.tocsr(),
        "barcodes": kept_barcodes,
        "positions": positions,
        "library_size": library_size,
        "detected": detected,
        "raw_spots": len(barcodes),
        "in_tissue_spots": int(in_tissue.sum()),
        "qc_spots": int(qc.sum()),
    }


def gene_vector(sample: dict, gene: str) -> np.ndarray:
    rows = sample["row_map"].get(gene, [])
    if not rows:
        return np.full(sample["lognorm"].shape[1], np.nan)
    return np.asarray(sample["lognorm"][rows].mean(axis=0)).ravel()


def mean_score(sample: dict, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    present = [gene for gene in genes if gene in sample["row_map"]]
    if not present:
        return np.full(sample["lognorm"].shape[1], np.nan), []
    vectors = np.vstack([gene_vector(sample, gene) for gene in present])
    return np.nanmean(vectors, axis=0), present


def rank_score(sample: dict, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    present = [gene for gene in genes if gene in sample["row_map"]]
    if not present:
        return np.full(sample["lognorm"].shape[1], np.nan), []
    vectors = []
    for gene in present:
        values = gene_vector(sample, gene)
        vectors.append(rankdata(values, method="average") / len(values))
    return np.mean(vectors, axis=0), present


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), covariates])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    if mask.sum() < 20:
        return float("nan")
    rx = residualize(rankdata(x[mask]), covariates[mask])
    ry = residualize(rankdata(y[mask]), covariates[mask])
    return float(spearmanr(rx, ry).statistic)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Exact rank formulation; positive values mean x tends to exceed y.
    combined = np.concatenate([x, y])
    ranks = rankdata(combined)
    rank_sum_x = float(ranks[: len(x)].sum())
    u = rank_sum_x - len(x) * (len(x) + 1) / 2
    return float(2 * u / (len(x) * len(y)) - 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_spatial(sample: dict, score_map: dict[str, np.ndarray], output: Path) -> None:
    with gzip.open(sample["paths"]["scalefactors"], "rt", encoding="utf-8") as handle:
        scales = json.load(handle)
    scale = float(scales["tissue_lowres_scalef"])
    with gzip.open(sample["paths"]["image"], "rb") as source:
        image = plt.imread(source, format="png")
    xy = np.asarray([
        [sample["positions"][barcode][4] * scale, sample["positions"][barcode][3] * scale]
        for barcode in sample["barcodes"]
    ])
    preferred = [
        "purine_synthesis_salvage", "carnitine_long_chain_fao",
        "polyamine_acetylation_catabolism", "acidity_lactate_response", "MUC1", "SAT1",
    ]
    selected = [name for name in preferred if name in score_map]
    selected.extend(name for name in score_map if "__rank" not in name and name not in selected)
    selected = selected[:6]
    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for axis, name in zip(axes.ravel(), selected):
        values = score_map[name]
        lo, hi = np.nanquantile(values, [0.02, 0.98])
        axis.imshow(image)
        scatter = axis.scatter(
            xy[:, 0], xy[:, 1], c=values, s=8, cmap="viridis", alpha=0.88,
            vmin=lo, vmax=hi, linewidths=0,
        )
        axis.set_title(name)
        axis.set_axis_off()
        figure.colorbar(scatter, ax=axis, fraction=0.036, pad=0.01)
    figure.suptitle(f"GSE236697 {sample['condition']}: descriptive spatial localization")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/GSE236697/raw_files"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/external/GSE236697/spatial_metabolic_axes_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        load_sample(args.raw_dir, "GSM7573205_p1", "tumour"),
        load_sample(args.raw_dir, "GSM8286350_p1_N", "normal"),
    ]
    sample_scores = {}
    spot_rows = []
    sample_summary = []
    for sample in samples:
        scores = {}
        present_genes = {}
        for name, genes in AXES.items():
            scores[name], present = mean_score(sample, genes)
            scores[f"{name}__rank"], _ = rank_score(sample, genes)
            present_genes[name] = present
        for name, genes in TUMOUR_COMPARTMENTS.items():
            scores[f"compartment__{name}"], present = mean_score(sample, genes)
            present_genes[f"compartment__{name}"] = present
        for gene in SINGLE_GENES:
            scores[gene] = gene_vector(sample, gene)
            present_genes[gene] = [gene] if gene in sample["row_map"] else []
        sample_scores[sample["condition"]] = scores
        sample_summary.append({
            "condition": sample["condition"],
            "raw_spots": sample["raw_spots"],
            "in_tissue_spots": sample["in_tissue_spots"],
            "qc_spots": sample["qc_spots"],
            "median_detected_genes": float(np.median(sample["detected"])),
            "median_library_size": float(np.median(sample["library_size"])),
            "present_genes": present_genes,
        })
        for index, barcode in enumerate(sample["barcodes"]):
            position = sample["positions"][barcode]
            row = {
                "condition": sample["condition"], "barcode": barcode,
                "array_row": position[1], "array_col": position[2],
                "pixel_row_fullres": position[3], "pixel_col_fullres": position[4],
                "library_size": sample["library_size"][index],
                "detected_genes": sample["detected"][index],
            }
            for name, values in scores.items():
                row[name] = values[index]
            spot_rows.append(row)
        plot_spatial(sample, scores, args.output_dir / f"{sample['condition']}_spatial_axes.png")

    comparison_rows = []
    tumour = sample_scores["tumour"]
    normal = sample_scores["normal"]
    for name in AXES:
        x, y = tumour[name], normal[name]
        comparison_rows.append({
            "axis": name,
            "tumour_median_mean_score": float(np.nanmedian(x)),
            "normal_median_mean_score": float(np.nanmedian(y)),
            "tumour_minus_normal_median": float(np.nanmedian(x) - np.nanmedian(y)),
            "cliffs_delta_spot_distribution_descriptive": cliffs_delta(x, y),
            "tumour_median_rank_score": float(np.nanmedian(tumour[f"{name}__rank"])),
            "normal_median_rank_score": float(np.nanmedian(normal[f"{name}__rank"])),
        })

    tumour_sample = samples[0]
    covariates = np.column_stack([
        np.log1p(tumour_sample["library_size"]), tumour_sample["detected"],
    ])
    correlation_rows = []
    preferred_key_axes = [
        "purine_synthesis_salvage", "carnitine_long_chain_fao",
        "polyamine_synthesis", "polyamine_acetylation_catabolism",
        "acidity_lactate_response", "neutrophil_recruitment", "SAT1",
    ]
    key_axes = [name for name in preferred_key_axes if name in tumour]
    key_axes.extend(name for name in AXES if name not in key_axes)
    context_scores = {
        name: tumour[f"compartment__{name}"] for name in TUMOUR_COMPARTMENTS
    }
    context_scores["MUC1"] = tumour["MUC1"]
    for axis_name in key_axes:
        for context_name, context_values in context_scores.items():
            x = tumour[axis_name]
            correlation_rows.append({
                "axis": axis_name,
                "context": context_name,
                "spearman_raw": float(spearmanr(x, context_values).statistic),
                "spearman_partial_library_detected": partial_spearman(
                    x, context_values, covariates
                ),
            })

    write_csv(args.output_dir / "spot_scores.csv", spot_rows)
    write_csv(args.output_dir / "tumour_normal_descriptive.csv", comparison_rows)
    write_csv(args.output_dir / "tumour_context_correlations.csv", correlation_rows)
    provenance = {}
    for sample in samples:
        for key, path in sample["paths"].items():
            provenance[f"{sample['condition']}__{key}"] = sha256(path)
    report = {
        "status": "gse236697_spatial_metabolic_axes_complete",
        "formal": True,
        "samples": sample_summary,
        "tumour_normal_descriptive": comparison_rows,
        "tumour_context_correlations": correlation_rows,
        "contracts": {
            "biological_replicates": 1,
            "spot_level_p_values": "forbidden",
            "interpretation": "descriptive spatial localization only",
            "source_qc": "in-tissue, >100 detected genes",
            "compartment_markers": "fixed from source paper spatial section",
        },
        "provenance": provenance,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "tumour_spots": sample_summary[0]["qc_spots"],
        "normal_spots": sample_summary[1]["qc_spots"],
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
