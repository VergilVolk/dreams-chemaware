#!/usr/bin/env python
"""Official-portal normalized-expression sensitivity for GSE178341.

This is deliberately secondary to the preregistered raw-count patient
pseudobulk.  The Broad Single Cell Portal exposes a deterministic 100,000-cell
subsample of the epithelial embedding with log-normalized expression.  We use
it only to test whether the frozen patient-level directions are plausible while
the full 10x count matrix is being acquired.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/GSE178341_mucinous_secretory_audit"
META = BASE / "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
CLUSTERS = BASE / "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
PREFLIGHT = BASE / "metadata_preflight_v1.json"
MATCHES = BASE / "metadata_matches_v1.csv"
DEFAULT_OUT = BASE / "scp_portal_normalized_sensitivity_v1"
API = "https://singlecell.broadinstitute.org/single_cell/api/v1/studies/SCP1162/expression/violin"

PURE_ADENO = "Adenocarcinoma"
PURE_MUC = "Adenocarcinoma;Mucinous"
GOBLET_CODES = ("cE02", "cE06", "cE07", "cE08")
PRIMARY_GENE = "NXPE1"
SECRETORY = ("MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2")
SIALIC = ("GNE", "NANS", "CMAS", "SLC35A1")
OAC = ("CASD1", "SIAE")
GENES = (PRIMARY_GENE,) + SECRETORY + SIALIC + OAC
SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    statistics = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def bootstrap_difference(case: np.ndarray, control: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(20_000, dtype=float)
    for start in range(0, len(values), 2_000):
        stop = min(start + 2_000, len(values))
        n = stop - start
        a = case[rng.integers(0, len(case), size=(n, len(case)))].mean(axis=1)
        b = control[rng.integers(0, len(control), size=(n, len(control)))].mean(axis=1)
        values[start:stop] = a - b
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def permutation_p(case: np.ndarray, control: np.ndarray, seed: int) -> float:
    values = np.concatenate([case, control])
    observed = abs(float(case.mean() - control.mean()))
    rng = np.random.default_rng(seed)
    exceed = 1
    for _ in range(100_000):
        selected = rng.choice(len(values), len(case), replace=False)
        mask = np.zeros(len(values), dtype=bool)
        mask[selected] = True
        statistic = abs(float(values[mask].mean() - values[~mask].mean()))
        exceed += statistic >= observed - 1e-12
    return exceed / 100_001


def bh(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(array) - reverse_rank + 1
        running = min(running, float(array[index]) * len(array) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def fetch_gene(gene: str, cache: Path) -> dict[str, object]:
    path = cache / f"{gene}.json"
    if not path.exists():
        response = requests.get(
            API,
            params={
                "genes": gene,
                "cluster": "Epithelial cells (tSNE)",
                "annotation_name": "_default",
                "annotation_type": "group",
                "annotation_scope": "cluster",
            },
            headers={"Accept": "application/json", "User-Agent": "DreaMS-GSE178341-audit/1.0"},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gene_names") != [gene]:
        raise RuntimeError(f"portal gene mismatch for {gene}: {payload.get('gene_names')}")
    if int(payload.get("rendered_subsample", -1)) != 100_000:
        raise RuntimeError(f"unexpected portal subsample for {gene}: {payload.get('rendered_subsample')}")
    return payload


def flatten(payload: dict[str, object]) -> tuple[list[str], np.ndarray]:
    cells: list[str] = []
    values: list[float] = []
    groups = payload.get("values")
    if not isinstance(groups, dict):
        raise RuntimeError("portal values are not a group dictionary")
    for group in groups.values():
        if not isinstance(group, dict):
            raise RuntimeError("invalid portal group")
        group_cells = [str(value) for value in group.get("cells", [])]
        group_values = [float(value) for value in group.get("y", [])]
        if len(group_cells) != len(group_values):
            raise RuntimeError("portal cell/value length mismatch")
        cells.extend(group_cells)
        values.extend(group_values)
    if len(cells) != 100_000 or len(set(cells)) != 100_000:
        raise RuntimeError(f"portal cell universe is not exactly 100,000 unique cells: {len(cells)} / {len(set(cells))}")
    return cells, np.asarray(values, dtype=np.float32)


def summarize_compartment(cell_frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    subset = cell_frame.loc[mask].copy()
    counts = subset.groupby("PID", observed=True).size().rename("cells")
    means = subset.groupby("PID", observed=True)[list(GENES)].mean()
    first = subset.sort_values("PatientTypeID").drop_duplicates("PID").set_index("PID")
    result = means.join(counts).join(first[["HistologicTypeSimple", "MMRStatus", "TissueSiteSimple"]])
    result = result.rename(columns={"HistologicTypeSimple": "histology", "TissueSiteSimple": "site"})
    result.insert(0, "compartment", label)
    result.index = result.index.astype(str)
    return result.reset_index()


def analyse_gene(frame: pd.DataFrame, gene: str, compartment: str, matches: pd.DataFrame, seed: int) -> dict[str, object]:
    case = frame.loc[frame["histology"].eq(PURE_MUC), gene].to_numpy(float)
    control = frame.loc[frame["histology"].eq(PURE_ADENO), gene].to_numpy(float)
    if len(case) != 6 or len(control) < 45:
        raise RuntimeError(f"unexpected patient support for {compartment}/{gene}: {len(case)} vs {len(control)}")
    lookup = frame.set_index("PID")[gene]
    deltas = []
    usable = True
    for patient, group in matches.groupby("case_PID", sort=True):
        controls = group["control_PID"].astype(str).tolist()
        if str(patient) not in lookup.index or any(control_id not in lookup.index for control_id in controls):
            usable = False
            break
        deltas.append(float(lookup.loc[str(patient)] - lookup.loc[controls].mean()))
    result = {
        "gene": gene,
        "compartment": compartment,
        "n_mucinous": len(case),
        "n_adenocarcinoma": len(control),
        "mucinous_mean": float(case.mean()),
        "adenocarcinoma_mean": float(control.mean()),
        "mean_difference": float(case.mean() - control.mean()),
        "bootstrap_95ci": bootstrap_difference(case, control, seed),
        "permutation_p": permutation_p(case, control, seed + 1),
        "matched_available": usable,
    }
    if usable:
        delta = np.asarray(deltas, dtype=float)
        result.update({
            "matched_mean_difference": float(delta.mean()),
            "matched_exact_sign_flip_p": exact_sign_flip(delta),
            "matched_all_same_direction": bool(np.all(delta > 0) or np.all(delta < 0)),
            "matched_case_differences": deltas,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = BASE / "scp_portal_cache_v1"
    cache.mkdir(parents=True, exist_ok=True)

    for path in (META, CLUSTERS, PREFLIGHT, MATCHES):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "gse178341_nxpe1_mucinous_metadata_preflight_passed":
        raise RuntimeError("metadata preflight did not pass")

    cell_ids: list[str] | None = None
    expression: dict[str, np.ndarray] = {}
    cache_hashes: dict[str, str] = {}
    for position, gene in enumerate(GENES, start=1):
        payload = fetch_gene(gene, cache)
        cells, values = flatten(payload)
        if cell_ids is None:
            cell_ids = cells
        elif cells != cell_ids:
            raise RuntimeError(f"portal subsample/order differs for {gene}")
        expression[gene] = values
        cache_hashes[gene] = sha256(cache / f"{gene}.json")
        print(f"[portal] {position}/{len(GENES)} {gene}", flush=True)
    assert cell_ids is not None

    meta = pd.read_csv(META)
    clusters = pd.read_csv(CLUSTERS)
    if not meta["cellID"].astype(str).equals(clusters["sampleID"].astype(str)):
        raise RuntimeError("metadata and cluster files are not exactly aligned")
    full = pd.concat([meta.reset_index(drop=True), clusters.drop(columns=["sampleID"]).reset_index(drop=True)], axis=1)
    if full["cellID"].duplicated().any():
        raise RuntimeError("duplicated metadata cell IDs")
    index = full.set_index(full["cellID"].astype(str), drop=False)
    exact = np.asarray([cell in index.index for cell in cell_ids], dtype=bool)
    missing = [cell for cell, keep in zip(cell_ids, exact) if not keep]
    missing_patients = sorted({cell.split("_", 1)[0] for cell in missing})
    patient_histology = (
        full.sort_values("PatientTypeID")
        .drop_duplicates("PID")
        .set_index(full.sort_values("PatientTypeID").drop_duplicates("PID")["PID"].astype(str))["HistologicTypeSimple"]
    )
    unknown_missing_patients = [patient for patient in missing_patients if patient not in patient_histology.index]
    mucinous_missing_patients = [
        patient for patient in missing_patients
        if patient in patient_histology.index and patient_histology.loc[patient] == PURE_MUC
    ]
    # SCP1162 currently contains two small epithelial aliquots (C121/C144)
    # absent from the GEO submit matrix.  This secondary audit may use the
    # exact 98%+ intersection only when the mismatch is confined to known,
    # non-mucinous patients.  The primary raw-count analysis never intersects.
    if exact.mean() < 0.98 or mucinous_missing_patients:
        raise RuntimeError(
            "portal/GEO mismatch exceeds the locked secondary-analysis boundary: "
            f"exact={exact.mean():.5f}, unknown={unknown_missing_patients}, mucinous={mucinous_missing_patients}"
        )
    kept_cell_ids = [cell for cell, keep in zip(cell_ids, exact) if keep]
    cells = index.loc[kept_cell_ids].reset_index(drop=True).copy()
    for gene in GENES:
        cells[gene] = expression[gene][exact]
    if not cells["clTopLevel"].eq("Epi").all():
        raise RuntimeError("portal epithelial cluster contains non-epithelial metadata cells")

    pure_tumour = cells["SPECIMEN_TYPE"].eq("T") & cells["HistologicTypeSimple"].isin((PURE_ADENO, PURE_MUC))
    broad = summarize_compartment(cells, pure_tumour, "epithelial_portal_subsample")
    goblet_mask = pure_tumour & cells["cl295v11SubShort"].isin(GOBLET_CODES)
    goblet = summarize_compartment(cells, goblet_mask, "goblet_family_portal_subsample")
    patient = pd.concat([broad, goblet], ignore_index=True)
    patient.to_csv(output / "patient_normalized_expression.csv", index=False)
    matches = pd.read_csv(MATCHES, dtype=str)

    results = []
    for compartment, frame in patient.groupby("compartment", sort=False):
        if frame.loc[frame["histology"].eq(PURE_MUC), "cells"].min() < 10:
            raise RuntimeError(f"a mucinous patient has fewer than 10 portal cells in {compartment}")
        for gene_index, gene in enumerate(GENES):
            results.append(analyse_gene(frame, gene, compartment, matches, SEED + gene_index * 11))
    table = pd.DataFrame(results)
    table["BH_q_within_compartment"] = np.nan
    for _, indices in table.groupby("compartment").groups.items():
        indices = list(indices)
        table.loc[indices, "BH_q_within_compartment"] = bh(table.loc[indices, "permutation_p"].tolist())
    table.to_csv(output / "fixed_panel_patient_results.csv", index=False)

    broad_primary = table[(table["compartment"].eq("epithelial_portal_subsample")) & (table["gene"].eq(PRIMARY_GENE))].iloc[0]
    goblet_primary = table[(table["compartment"].eq("goblet_family_portal_subsample")) & (table["gene"].eq(PRIMARY_GENE))].iloc[0]
    muc_cells = broad.loc[broad["histology"].eq(PURE_MUC), ["PID", "cells"]].set_index("PID")["cells"].to_dict()
    report = {
        "status": "gse178341_scp_portal_normalized_sensitivity_complete",
        "formal": False,
        "role": "secondary official-portal normalized-expression sensitivity; not raw-count pseudobulk",
        "portal_subsample_cells": len(cell_ids),
        "portal_cells_exactly_joined_to_geo": int(exact.sum()),
        "portal_geo_exact_join_fraction": float(exact.mean()),
        "portal_cells_absent_from_geo": len(missing),
        "portal_geo_mismatch_patients": missing_patients,
        "portal_geo_mismatch_patients_absent_entirely_from_geo": unknown_missing_patients,
        "pure_tumour_portal_cells": int(pure_tumour.sum()),
        "pure_tumour_goblet_cells": int(goblet_mask.sum()),
        "patient_support": {
            "mucinous": int(broad.loc[broad["histology"].eq(PURE_MUC), "PID"].nunique()),
            "adenocarcinoma": int(broad.loc[broad["histology"].eq(PURE_ADENO), "PID"].nunique()),
            "mucinous_epithelial_cells": muc_cells,
        },
        "NXPE1": {
            "epithelial": broad_primary.to_dict(),
            "goblet_family": goblet_primary.to_dict(),
        },
        "fixed_panel": table.to_dict(orient="records"),
        "provenance": {
            "study": "SCP1162 / GSE178341",
            "api": API,
            "metadata_sha256": sha256(META),
            "cluster_sha256": sha256(CLUSTERS),
            "matches_sha256": sha256(MATCHES),
            "portal_cache_sha256_by_gene": cache_hashes,
        },
        "contracts": {
            "unit": "patient PID",
            "expression": "Single Cell Portal supplied normalized values on deterministic 100,000-cell epithelial subsample",
            "primary_result_unchanged": "preregistered raw-count patient pseudobulk remains primary",
            "portal_geo_reconciliation": "exact cell-ID intersection only; permitted because >98% matched and no absent cell belongs to a GEO-known mucinous patient; one portal patient absent entirely from GEO is excluded and disclosed",
            "phenotype_used_for_network_or_annotation": False,
            "post_hoc_gene_addition": False,
        },
        "claim_limit": "This subsampled normalized-expression audit can establish directional plausibility only. It cannot replace raw-count patient pseudobulk, establish metabolite source, flux, enzyme activity, or MSI Level-1 metabolite identity.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    plot = table[table["compartment"].eq("epithelial_portal_subsample")].copy()
    plot = plot.set_index("gene").loc[list(GENES)].reset_index()
    fig, axis = plt.subplots(figsize=(10, 4.8))
    colors = ["#a11d21" if gene == PRIMARY_GENE else "#336699" for gene in plot["gene"]]
    axis.bar(plot["gene"], plot["mean_difference"], color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Mucinous - conventional\nmean portal-normalized expression")
    axis.set_title("GSE178341 patient-level epithelial sensitivity (portal 100k subsample)")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output / "fixed_panel_patient_directions.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"status": report["status"], "NXPE1": report["NXPE1"], "output": str(output)}, indent=2, default=str))


if __name__ == "__main__":
    main()
