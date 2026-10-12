#!/usr/bin/env python
"""Build the frozen ST003325 MS/MS query set without using structures or effects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile


PANELS = {
    "AN005444": {"polarity": "negative", "dda_token": "HILICneg"},
    "AN005445": {"polarity": "positive", "dda_token": "RPpos"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum_digest(mz: np.ndarray, intensity: np.ndarray) -> str:
    values = np.column_stack((mz.astype("<f8"), intensity.astype("<f8")))
    return hashlib.sha256(values.tobytes()).hexdigest()


def peak_bin(n: int) -> str:
    lower = 2 ** int(np.floor(np.log2(max(n, 1))))
    return f"{lower}-{2 * lower - 1}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--dda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-peaks", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    spectra_by_panel: dict[str, list[dict[str, object]]] = {}
    dda_provenance: dict[str, str] = {}
    for analysis, specification in PANELS.items():
        paths = sorted(args.dda.rglob(f"*DDA_{specification['dda_token']}*.mzML"))
        if len(paths) != 3:
            raise FileNotFoundError(f"{analysis}: expected three DDA mzML files")
        files = []
        for file_index, path in enumerate(paths):
            experiment = MSExperiment()
            MzMLFile().load(str(path), experiment)
            native = {spectrum.getNativeID(): spectrum for spectrum in experiment if spectrum.getMSLevel() == 2}
            files.append(native)
            dda_provenance[str(path.relative_to(args.dda))] = sha256(path)

        linkage_path = args.linkage / f"{analysis}_feature_linkage.csv.gz"
        linkage = pd.read_csv(linkage_path)
        eligible = linkage.loc[
            linkage["measurement_qualified"].astype(bool)
            & linkage["linked_replicated"].astype(bool)
        ].copy()
        output = []
        for row in eligible.itertuples(index=False):
            spectrum = files[int(row.best_dda_file)].get(str(row.best_scan_id))
            if spectrum is None:
                raise KeyError(f"Missing {analysis}:{row.best_scan_id}")
            mz, intensity = spectrum.get_peaks()
            if len(mz) < args.min_peaks:
                continue
            output.append({
                "analysis_id": analysis,
                "query_id": row.query_id,
                "polarity": specification["polarity"],
                "feature_mz": float(row.mz),
                "feature_rt_sec": float(row.rt_sec),
                "native_scan_id": str(row.best_scan_id),
                "dda_file_index": int(row.best_dda_file),
                "dda_replicates": int(row.dda_replicates),
                "ppm_error": float(row.best_ppm_error),
                "rt_error_sec": float(row.best_rt_error_sec),
                "n_peaks": int(len(mz)),
                "mz_bin_50da": int(float(row.mz) // 50) * 50,
                "peak_count_bin": peak_bin(len(mz)),
                "technical_stratum": (
                    f"{specification['polarity']}|mz{int(float(row.mz) // 50) * 50}|peaks{peak_bin(len(mz))}"
                ),
                "spectrum_sha256": spectrum_digest(mz, intensity),
                "mz_array": mz,
                "intensity_array": intensity,
            })
        spectra_by_panel[analysis] = output

    records = [row for rows in spectra_by_panel.values() for row in rows]
    metadata = pd.DataFrame([{k: v for k, v in row.items() if not k.endswith("_array")} for row in records])
    metadata["spectral_unit_id"] = "SPU:" + metadata["spectrum_sha256"].str.slice(0, 20)
    metadata.to_csv(args.output / "entity_spectrum_links.csv.gz", index=False, compression="gzip")
    representative_indices = (
        metadata.assign(link_error=metadata["ppm_error"] / 5.0 + metadata["rt_error_sec"] / 10.0)
        .sort_values(["spectral_unit_id", "link_error", "query_id"])
        .drop_duplicates("spectral_unit_id")
        .index
        .tolist()
    )
    unit_records = [records[index] for index in representative_indices]
    units = metadata.loc[representative_indices].sort_values("spectral_unit_id").copy()
    units.to_csv(args.output / "spectral_units.csv.gz", index=False, compression="gzip")

    with (args.output / "queries.mgf").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(unit_records, key=lambda value: value["spectrum_sha256"]):
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE=SPU:{row['spectrum_sha256'][:20]}\n")
            handle.write(f"PEPMASS={row['feature_mz']:.8f}\n")
            handle.write(f"RTINSECONDS={row['feature_rt_sec']:.4f}\n")
            handle.write(f"IONMODE={str(row['polarity']).upper()}\n")
            for mz, intensity in zip(row["mz_array"], row["intensity_array"]):
                handle.write(f"{float(mz):.8f} {float(intensity):.8g}\n")
            handle.write("END IONS\n\n")

    duplicated_spectra = metadata.groupby("spectrum_sha256")["query_id"].nunique()
    report = {
        "status": "ST003325_STRUCTURE_HIDDEN_QUERY_SET_BUILT",
        "formal": True,
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed",
        "sealed": False,
        "structure_labels_used": False,
        "phenotype_or_effect_values_used": False,
        "selection": {
            "measurement_qualified": True,
            "linked_in_at_least_two_of_three_dda_files": True,
            "representative_scan_rule": "minimum normalized ppm-plus-RT distance from the linkage audit",
            "min_fragment_peaks": args.min_peaks,
        },
        "entity_spectrum_links": int(len(metadata)),
        "entity_links_by_panel": {key: len(value) for key, value in spectra_by_panel.items()},
        "independent_spectral_units": int(len(units)),
        "spectrum_hashes_assigned_to_multiple_queries": int((duplicated_spectra > 1).sum()),
        "max_queries_per_spectrum_hash": int(duplicated_spectra.max()),
        "peak_count": {
            "minimum": int(metadata["n_peaks"].min()),
            "median": float(metadata["n_peaks"].median()),
            "maximum": int(metadata["n_peaks"].max()),
        },
        "provenance": {
            "linkage_report_sha256": sha256(args.linkage / "report.json"),
            "dda_files": dda_provenance,
            "entity_spectrum_links_sha256": sha256(args.output / "entity_spectrum_links.csv.gz"),
            "spectral_units_sha256": sha256(args.output / "spectral_units.csv.gz"),
            "mgf_sha256": sha256(args.output / "queries.mgf"),
        },
        "mandatory_public_reference_exclusion": {
            "study_accessions": ["ST003325", "PR001844"],
            "raw_file_hashes": list(dda_provenance.values()),
        },
        "claim_limit": (
            "A structure-hidden MS/MS query universe only. Feature-to-scan linkage remains an m/z/RT "
            "association, not molecular identity, and no perturbation or biological effect is established."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
