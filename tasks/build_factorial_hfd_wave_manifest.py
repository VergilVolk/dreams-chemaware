#!/usr/bin/env python3
"""Freeze a design-native HFD response wave from official MetaboLights metadata.

The acquisition tree contains no genotype or diet labels.  Those labels are
written to a sealed design table and are opened only after each assay-local
abundance matrix is frozen.  One biological animal is never replaced by its
multiple LC-MS methods or duplicate repository paths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


API = "https://www.ebi.ac.uk/metabolights/ws/studies"
FTP_HTTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"
STUDIES = {
    "MTBLS135": {
        "assay_pattern": r"SERUM",
        "body_part": "blood serum",
        "reference_genotype": "wild type genotype",
        "modified_genotype": "Hi-MYC",
        "control_diet": "Control Group",
        "hfd_diet": "Diet, High-Fat",
        "expected_samples": 24,
        "source_family": "Dana-Farber_Hi-MYC_wave1",
    },
    "MTBLS3316": {
        "assay_pattern": r"LC-MS_METHOD",
        "body_part": "blood serum",
        "reference_genotype": "wild type (WT)",
        "modified_genotype": "Hi-MYC",
        "control_diet": "control diet (CTD)",
        "hfd_diet": "high-fat diet (HFD)",
        "expected_samples": 24,
        "source_family": "Dana-Farber_Hi-MYC_wave2",
    },
    "MTBLS2085": {
        "assay_pattern": r"Serum",
        "body_part": "blood serum",
        "reference_genotype": "WT",
        "modified_genotype": "NQO1-Tg",
        "control_diet": "standard diet (SD)",
        "hfd_diet": "high-fat diet (HFD)",
        "expected_samples": 19,
        "source_family": "NIA_NQO1",
    },
}


def fetch_json(url: str) -> tuple[dict, bytes]:
    with urlopen(url, timeout=120) as response:  # noqa: S310 - fixed official host
        payload = response.read()
    return json.loads(payload), payload


def stable_id(*parts: str, length: int = 20) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:length]


def remote_size(url: str) -> int:
    error: Exception | None = None
    for attempt in range(5):
        try:
            request = Request(url, method="HEAD", headers={"User-Agent": "DreaMS-Contrast-Atlas/1.0"})
            with urlopen(request, timeout=120) as response:  # noqa: S310 - frozen official host
                value = response.headers.get("Content-Length")
            if value is None or int(value) <= 0:
                raise RuntimeError(f"missing positive Content-Length: {url}")
            return int(value)
        except Exception as caught:  # network retry is bounded and then fails closed
            error = caught
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"remote size probe failed after retries: {url}: {error}")


def choose_repository_path(paths: list[str]) -> str:
    unique = sorted(set(paths))
    if len(unique) == 1:
        return unique[0]
    basenames = {PurePosixPath(path).name for path in unique}
    if len(basenames) != 1:
        raise ValueError(f"multiple non-mirror paths for one sample/assay: {unique}")
    # MTBLS3316 deposits a root copy and a method-directory copy.  The deeper
    # official path is selected deterministically; content identity is checked
    # after acquisition.
    return max(unique, key=lambda value: (len(PurePosixPath(value).parts), value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    metadata_dir = args.output_dir / "official_metadata_snapshots"
    blind_dir = args.output_dir / "phenotype_blind"
    sealed_dir = args.output_dir / "sealed_design"
    metadata_dir.mkdir(parents=True)
    blind_dir.mkdir()
    sealed_dir.mkdir()

    acquisition_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for accession, spec in STUDIES.items():
        study, study_bytes = fetch_json(f"{API}/{accession}")
        study_path = metadata_dir / f"{accession}_study.json"
        study_path.write_bytes(study_bytes)
        record = study["isaInvestigation"]["studies"][0]
        sample_name = record["filename"]
        samples, sample_bytes = fetch_json(f"{API}/{accession}/{quote(sample_name)}")
        sample_path = metadata_dir / f"{accession}_{sample_name}.json"
        sample_path.write_bytes(sample_bytes)
        sample_rows = pd.DataFrame(samples["data"]["rows"])
        sample_rows = sample_rows.loc[
            sample_rows["Characteristics[Organism part]"].astype(str).eq(spec["body_part"])
        ].copy()
        if len(sample_rows) != spec["expected_samples"]:
            raise RuntimeError(f"{accession}: expected {spec['expected_samples']} samples, got {len(sample_rows)}")
        sample_rows = sample_rows[["Sample Name", "Factor Value[Genotype]", "Factor Value[Diet]"]]
        sample_rows.columns = ["sample_name", "genotype", "diet"]
        for column in ("sample_name", "genotype", "diet"):
            sample_rows[column] = sample_rows[column].astype(str).str.strip()
        if sample_rows["sample_name"].duplicated().any():
            raise RuntimeError(f"{accession}: duplicate sample name")
        allowed_g = {spec["reference_genotype"], spec["modified_genotype"]}
        allowed_d = {spec["control_diet"], spec["hfd_diet"]}
        if set(sample_rows.genotype) != allowed_g or set(sample_rows.diet) != allowed_d:
            raise RuntimeError(f"{accession}: unexpected genotype/diet levels")
        sample_lookup = sample_rows.set_index("sample_name")

        selected_assays = [
            assay for assay in record["assays"]
            if re.search(spec["assay_pattern"], assay["filename"], flags=re.IGNORECASE)
        ]
        if not selected_assays:
            raise RuntimeError(f"{accession}: no selected assays")
        for assay in selected_assays:
            assay_name = assay["filename"]
            assay_data, assay_bytes = fetch_json(f"{API}/{accession}/{quote(assay_name)}")
            assay_path = metadata_dir / f"{accession}_{assay_name}.json"
            assay_path.write_bytes(assay_bytes)
            rows = pd.DataFrame(assay_data["data"]["rows"])
            rows = rows.loc[rows["Sample Name"].astype(str).isin(sample_lookup.index)].copy()
            content = set(rows.get("Parameter Value[Data file content]", pd.Series(dtype=str)).dropna())
            content2 = set(rows.get("Parameter Value[Data file content].1", pd.Series(dtype=str)).dropna())
            if "MS1 spectrum" not in content or "MSn spectrum" not in content2:
                raise RuntimeError(f"{accession}/{assay_name}: official metadata does not declare MS1+MSn")
            grouped = rows.groupby("Sample Name", sort=True)["Derived Spectral Data File"].apply(list)
            if set(grouped.index) != set(sample_lookup.index):
                raise RuntimeError(f"{accession}/{assay_name}: assay/sample join is incomplete")
            panel_id = stable_id(accession, assay_name, length=16)
            for sample, paths in grouped.items():
                selected_path = choose_repository_path([str(path) for path in paths if str(path).endswith(".mzML")])
                meta = sample_lookup.loc[sample]
                sample_id = f"S{stable_id(accession, str(sample))}"
                acquisition_rows.append({
                    "panel_id": panel_id, "dataset_accession": accession,
                    "sample_id": sample_id, "biological_unit_id": sample_id,
                    "assay_file": assay_name, "remote_path": selected_path,
                    "download_url": f"{FTP_HTTP}/{accession}/{quote(selected_path, safe='/')}",
                    "filename": PurePosixPath(selected_path).name,
                    "local_path": f"data/external/factorial_hfd_wave_20261016/{accession}/{panel_id}/{PurePosixPath(selected_path).name}",
                })
                design_rows.append({
                    "panel_id": panel_id, "dataset_accession": accession,
                    "sample_id": sample_id, "biological_unit_id": sample_id,
                    "source_family": spec["source_family"],
                    "genotype": str(meta.genotype), "diet": str(meta.diet),
                    "genotype_code": int(meta.genotype == spec["modified_genotype"]),
                    "diet_code": int(meta.diet == spec["hfd_diet"]),
                })
            cells = (
                pd.DataFrame(design_rows).loc[lambda x: x.panel_id.eq(panel_id)]
                .groupby(["genotype_code", "diet_code"])["biological_unit_id"].nunique().to_dict()
            )
            if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)} or min(cells.values()) < 4:
                raise RuntimeError(f"{accession}/{assay_name}: invalid 2x2 cell counts {cells}")
            panel_rows.append({
                "panel_id": panel_id, "dataset_accession": accession,
                "source_family": spec["source_family"], "assay_file": assay_name,
                "biological_units": len(grouped),
                "cell_00": cells[(0, 0)], "cell_01": cells[(0, 1)],
                "cell_10": cells[(1, 0)], "cell_11": cells[(1, 1)],
            })
            provenance.append({
                "accession": accession, "assay_file": assay_name,
                "assay_snapshot_sha256": hashlib.sha256(assay_bytes).hexdigest(),
            })
        provenance.append({
            "accession": accession,
            "study_snapshot_sha256": hashlib.sha256(study_bytes).hexdigest(),
            "sample_snapshot_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        })

    acquisition = pd.DataFrame(acquisition_rows).drop_duplicates()
    design = pd.DataFrame(design_rows).drop_duplicates()
    if acquisition.duplicated(["panel_id", "sample_id"]).any():
        raise RuntimeError("more than one selected file per sample and assay")
    if design.duplicated(["panel_id", "sample_id"]).any():
        raise RuntimeError("duplicate sealed design row")
    with ThreadPoolExecutor(max_workers=4) as executor:
        acquisition["deposited_bytes"] = list(executor.map(remote_size, acquisition["download_url"]))
    acquisition.to_csv(blind_dir / "acquisition_manifest.tsv", sep="\t", index=False)
    design.to_csv(sealed_dir / "factorial_design.tsv", sep="\t", index=False)
    pd.DataFrame(panel_rows).to_csv(args.output_dir / "panel_registry.tsv", sep="\t", index=False)
    report = {
        "schema": "factorial_hfd_wave_v1",
        "status": "official_metadata_frozen_raw_unopened",
        "denominators": {
            "studies": len(STUDIES), "source_families": len(STUDIES),
            "assay_panels": len(panel_rows), "panel_sample_files": len(acquisition),
            "unique_biological_units_across_studies": int(design[["dataset_accession", "biological_unit_id"]].drop_duplicates().shape[0]),
            "download_bytes": int(acquisition["deposited_bytes"].sum()),
            "download_gib": float(acquisition["deposited_bytes"].sum() / (1024 ** 3)),
        },
        "primary_biological_objects": [
            "HFD_minus_control_in_reference_genotype",
            "HFD_minus_control_in_modified_genotype",
            "modified_minus_reference_under_control_diet",
            "modified_minus_reference_under_HFD",
            "genotype_by_HFD_interaction",
        ],
        "leakage_status": "official design metadata opened; abundance_MS2_and_cross_study_scores_unopened",
        "provenance": provenance,
        "claim_boundary": [
            "MTBLS135 and MTBLS3316 are separate Hi-MYC experiments but share model and research lineage.",
            "MTBLS2085 is a distinct NQO1 experiment and is not assumed to have the same interaction direction.",
            "MS1+MSn declarations establish acquisition capacity, not usable linked-entity coverage.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
