from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/PXD055865_2026_MUC2/pride_raw_audit_v1"
API = (
    "https://www.ebi.ac.uk/pride/ws/archive/v2/projects/"
    "PXD055865/files?pageSize=1000&page=0"
)

TARGETS = {
    "230701_Colon1a_R1_StcEx2.raw": ("Patient1", "Colon1a", "tumour"),
    "230701_Colon1a_R2_StcEx2.raw": ("Patient1", "Colon1a", "tumour_adjacent"),
    "230721_Colon1b_R1_StcEx2.raw": ("Patient1", "Colon1b", "tumour"),
    "230721_Colon1b_R2_StcEx2.raw": ("Patient1", "Colon1b", "tumour_adjacent"),
    "240404_Colon2_Tumor1_StcE.raw": ("Patient2", "Colon2", "tumour_1"),
    "240404_Colon2_Tumor2_StcE.raw": ("Patient2", "Colon2", "tumour_2"),
    "240404_Colon2_Adjacent1_StcE.raw": ("Patient2", "Colon2", "tumour_adjacent_1"),
    "240404_Colon2_Adjacent2_StcE.raw": ("Patient2", "Colon2", "tumour_adjacent_2"),
    "240404_HealthyColon_StcE.raw": ("HealthyDonor", "HealthyColon", "healthy"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    request = Request(API, headers={"User-Agent": "DreaMS-PXD055865-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    files = payload.get("_embedded", {}).get("files") if isinstance(payload, dict) else None
    if files is None:
        files = payload
    if not isinstance(files, list):
        raise RuntimeError("unexpected PRIDE files response")

    rows = []
    for item in files:
        locations = {
            entry.get("name"): entry.get("value")
            for entry in item.get("publicFileLocations", [])
        }
        category = item.get("fileCategory") or {}
        rows.append(
            {
                "file_name": item.get("fileName", ""),
                "bytes": int(item.get("fileSizeBytes") or 0),
                "category": category.get("value", ""),
                "ftp": locations.get("FTP Protocol", ""),
                "aspera": locations.get("Aspera Protocol", ""),
            }
        )

    by_name = {row["file_name"]: row for row in rows}
    missing = sorted(set(TARGETS) - set(by_name))
    if missing:
        raise RuntimeError(f"missing target raw files: {missing}")

    target_rows = []
    for name, (patient, specimen, region) in TARGETS.items():
        row = dict(by_name[name])
        row.update(
            {
                "patient": patient,
                "specimen": specimen,
                "region": region,
                "region_mapping_source": "MOESM1 Supplementary Fig. 9",
            }
        )
        target_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["file_name", "bytes", "category", "ftp", "aspera"]
    with (OUT / "pride_file_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    target_fields = fields + [
        "patient",
        "specimen",
        "region",
        "region_mapping_source",
    ]
    with (OUT / "fingerprint_xic_target_raw_files.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(target_rows)

    report = {
        "status": "pxd055865_pride_raw_file_audit_complete",
        "api": API,
        "project_files": len(rows),
        "raw_files": sum(row["category"] == "RAW" for row in rows),
        "xic_target_files": len(target_rows),
        "xic_target_bytes": sum(row["bytes"] for row in target_rows),
        "independent_tumour_patients": 2,
        "healthy_donors": 1,
        "regions": {
            f"{row['specimen']}__{row['region']}": row["file_name"]
            for row in target_rows
        },
        "provenance": {
            "inventory_sha256": sha256(OUT / "pride_file_inventory.csv"),
            "target_sha256": sha256(OUT / "fingerprint_xic_target_raw_files.csv"),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "This freezes the public files and region mapping for a uniform XIC "
            "re-extraction. It contains no re-integrated abundance result."
        ),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
