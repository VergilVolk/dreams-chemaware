from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image


FIGURE_LEVELS = [
    # Values are manually transcribed from the normalized-level labels printed
    # in Supplementary Figs. 20, 22, 23, 24 and 25 (MOESM1, pages 25, 28-31).
    # They are figure-display levels, not integrated peak areas or concentrations.
    ("Colon1a_tumour", "Patient1", "tumour", "Neu5Ac", 1.42e6, 20),
    ("Colon1a_tumour", "Patient1", "tumour", "AcNeu5Ac", 1.42e6, 23),
    ("Colon1a_tumour", "Patient1", "tumour", "Ac2Neu5Ac", 4.05e5, 24),
    ("Colon1b_tumour", "Patient1", "tumour", "Neu5Ac", 5.79e5, 20),
    ("Colon1b_tumour", "Patient1", "tumour", "AcNeu5Ac", 5.79e5, 23),
    ("Colon1b_tumour", "Patient1", "tumour", "Ac2Neu5Ac", 1.32e6, 24),
    ("Colon2_tumour1", "Patient2", "tumour", "Neu5Ac", 3.38e6, 20),
    ("Colon2_tumour1", "Patient2", "tumour", "AcNeu5Ac", 7.14e4, 23),
    ("Colon2_tumour1", "Patient2", "tumour", "Ac2Neu5Ac", 1.80e4, 24),
    ("Colon2_tumour2", "Patient2", "tumour", "Neu5Ac", 2.58e6, 20),
    ("Colon2_tumour2", "Patient2", "tumour", "AcNeu5Ac", 2.43e4, 23),
    ("Colon2_tumour2", "Patient2", "tumour", "Ac2Neu5Ac", 1.08e4, 24),
    ("Healthy_colon", "HealthyDonor", "healthy", "HexNAc", 1.16e7, 22),
    ("Healthy_colon", "HealthyDonor", "healthy", "Neu5Ac", 1.57e5, 22),
    ("Healthy_colon", "HealthyDonor", "healthy", "AcNeu5Ac", 1.87e6, 22),
    ("Healthy_colon", "HealthyDonor", "healthy", "Ac2Neu5Ac", 6.96e6, 22),
    ("Healthy_colon", "HealthyDonor", "healthy", "Ac3Neu5Ac", 2.86e7, 22),
    ("Colon1a_adjacent", "Patient1", "tumour_adjacent", "HexNAc", 1.64e7, 25),
    ("Colon1a_adjacent", "Patient1", "tumour_adjacent", "Neu5Ac", 3.82e5, 25),
    ("Colon1a_adjacent", "Patient1", "tumour_adjacent", "AcNeu5Ac", 6.75e4, 25),
    ("Colon1a_adjacent", "Patient1", "tumour_adjacent", "Ac2Neu5Ac", 1.67e6, 25),
    ("Colon1b_adjacent", "Patient1", "tumour_adjacent", "HexNAc", 2.81e6, 25),
    ("Colon1b_adjacent", "Patient1", "tumour_adjacent", "Neu5Ac", 1.01e6, 25),
    ("Colon1b_adjacent", "Patient1", "tumour_adjacent", "AcNeu5Ac", 4.70e5, 25),
    ("Colon1b_adjacent", "Patient1", "tumour_adjacent", "Ac2Neu5Ac", 1.22e6, 25),
    ("Colon2_adjacent1", "Patient2", "tumour_adjacent", "HexNAc", 1.80e7, 25),
    ("Colon2_adjacent1", "Patient2", "tumour_adjacent", "Neu5Ac", 1.09e5, 25),
    ("Colon2_adjacent1", "Patient2", "tumour_adjacent", "AcNeu5Ac", 3.08e5, 25),
    ("Colon2_adjacent1", "Patient2", "tumour_adjacent", "Ac2Neu5Ac", 1.30e5, 25),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/PXD055865_2026_MUC2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/PXD055865_2026_MUC2/source_data_audit_v1"
        ),
    )
    args = parser.parse_args()

    supplement_paths = sorted(args.root.glob("41467_2026_72853_MOESM*_ESM.*"))
    expected = {f"41467_2026_72853_MOESM{i}_ESM" for i in range(1, 9)}
    observed = {path.stem for path in supplement_paths}
    if observed != expected:
        raise RuntimeError(
            f"expected all eight supplementary files; missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )

    source_root = args.root / "source_data" / "Source Data MALDI Images"
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    image_rows: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*.png")):
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        image_rows.append(
            {
                "figure": path.relative_to(source_root).parts[0],
                "relative_path": path.relative_to(source_root).as_posix(),
                "bytes": path.stat().st_size,
                "width_px": width,
                "height_px": height,
                "mode": mode,
                "sha256": sha256(path),
            }
        )
    images = pd.DataFrame(image_rows)
    if images.empty:
        raise RuntimeError("no source-data PNG files found")

    non_image = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".png" and path.name != ".DS_Store"
    ]

    levels = pd.DataFrame(
        FIGURE_LEVELS,
        columns=[
            "specimen_region",
            "patient",
            "tissue",
            "fingerprint",
            "normalized_level",
            "supplementary_figure",
        ],
    )
    levels["source_pdf_page"] = levels["supplementary_figure"].map(
        {20: 25, 22: 28, 23: 29, 24: 30, 25: 31}
    )
    levels["measurement_semantics"] = (
        "manually transcribed figure normalized-level label; not integrated area"
    )

    wide = levels.pivot_table(
        index=["specimen_region", "patient", "tissue"],
        columns="fingerprint",
        values="normalized_level",
        aggfunc="first",
    ).reset_index()
    for fingerprint in ("AcNeu5Ac", "Ac2Neu5Ac", "Ac3Neu5Ac"):
        wide[f"{fingerprint}_to_Neu5Ac"] = wide[fingerprint] / wide["Neu5Ac"]
        wide[f"log2_{fingerprint}_to_Neu5Ac"] = wide[
            f"{fingerprint}_to_Neu5Ac"
        ].map(lambda value: math.log2(value) if pd.notna(value) and value > 0 else None)

    ratios = wide[
        [
            "specimen_region",
            "patient",
            "tissue",
            "Neu5Ac",
            "AcNeu5Ac",
            "Ac2Neu5Ac",
            "Ac3Neu5Ac",
            "AcNeu5Ac_to_Neu5Ac",
            "Ac2Neu5Ac_to_Neu5Ac",
            "Ac3Neu5Ac_to_Neu5Ac",
            "log2_AcNeu5Ac_to_Neu5Ac",
            "log2_Ac2Neu5Ac_to_Neu5Ac",
            "log2_Ac3Neu5Ac_to_Neu5Ac",
        ]
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images.to_csv(args.output_dir / "source_image_inventory.csv", index=False)
    levels.to_csv(args.output_dir / "sialic_fingerprint_normalized_levels.csv", index=False)
    ratios.to_csv(args.output_dir / "sialic_fingerprint_ratios.csv", index=False)

    per_figure = (
        images.groupby("figure", as_index=False)
        .agg(pngs=("relative_path", "size"), bytes=("bytes", "sum"))
        .sort_values("figure")
    )
    report = {
        "status": "pxd055865_source_data_audit_complete",
        "supplementary_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in supplement_paths
        ],
        "source_data": {
            "pngs": int(len(images)),
            "non_image_files": len(non_image),
            "numerical_tables": 0,
            "figures": per_figure.to_dict("records"),
            "interpretation": (
                "MOESM8 is image-only source data. It permits image provenance "
                "checking but not re-estimation of patient-level abundance or uncertainty."
            ),
        },
        "figure_level_audit": {
            "rows": int(len(levels)),
            "specimen_regions": int(levels["specimen_region"].nunique()),
            "independent_colorectal_tumour_patients": 2,
            "healthy_donors": 1,
            "healthy_oacetyl_to_neu5ac": {
                "AcNeu5Ac": float(
                    ratios.loc[
                        ratios["specimen_region"].eq("Healthy_colon"),
                        "AcNeu5Ac_to_Neu5Ac",
                    ].iloc[0]
                ),
                "Ac2Neu5Ac": float(
                    ratios.loc[
                        ratios["specimen_region"].eq("Healthy_colon"),
                        "Ac2Neu5Ac_to_Neu5Ac",
                    ].iloc[0]
                ),
                "Ac3Neu5Ac": float(
                    ratios.loc[
                        ratios["specimen_region"].eq("Healthy_colon"),
                        "Ac3Neu5Ac_to_Neu5Ac",
                    ].iloc[0]
                ),
            },
            "tumour_oacetyl_to_neu5ac_ranges": {
                "AcNeu5Ac": [
                    float(ratios.loc[ratios["tissue"].eq("tumour"), "AcNeu5Ac_to_Neu5Ac"].min()),
                    float(ratios.loc[ratios["tissue"].eq("tumour"), "AcNeu5Ac_to_Neu5Ac"].max()),
                ],
                "Ac2Neu5Ac": [
                    float(ratios.loc[ratios["tissue"].eq("tumour"), "Ac2Neu5Ac_to_Neu5Ac"].min()),
                    float(ratios.loc[ratios["tissue"].eq("tumour"), "Ac2Neu5Ac_to_Neu5Ac"].max()),
                ],
            },
            "interpretation": (
                "The labels show high O-acetyl/Neu5Ac display ratios in the single "
                "healthy colon and marked heterogeneity across two tumour patients. "
                "They support carrier/pool decoupling as descriptive context, not an "
                "independent abundance replication or a population comparison."
            ),
        },
        "claim_limits": [
            "normalized level is a figure-display scale, not a calibrated concentration",
            "Colon1a and Colon1b are two specimens from the same patient",
            "the healthy colon is one independent donor",
            "no patient-level variance estimate or hypothesis test is possible",
            "the study does not measure the free Neu5Ac pool used in MTBLS13729",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
