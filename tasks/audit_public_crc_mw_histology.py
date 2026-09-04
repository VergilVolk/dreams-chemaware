"""Audit Metabolomics Workbench CRC studies for public histology labels."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "data/external/omicsdi_crc_metabolomics_search_20260830.json"
OUT = ROOT / "data/external/public_crc_mw_histology_audit_v1"
CACHE = OUT / "metadata"

HISTOLOGY_RE = re.compile(r"mucin|histolog|adenocarcinoma|patholog|tumou?r[ _-]?type|morpholog|signet", re.I)
MUCINOUS_RE = re.compile(r"mucinous|mucin-producing|mucin producing", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "DreaMS-public-metadata-audit/1.0"}), timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {error}")


def parse_tsv(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    search = json.loads(SEARCH.read_text(encoding="utf-8-sig"))
    studies = [row for row in search["datasets"] if re.fullmatch(r"ST\d+", str(row["id"]))]
    reports: list[dict] = []
    values: list[dict] = []

    for position, study in enumerate(studies, start=1):
        accession = str(study["id"])
        report = {
            "accession": accession,
            "title": str(study.get("title") or ""),
            "factor_rows": 0,
            "histology_like_fields": "",
            "mucinous_rows": 0,
            "status": "",
        }
        try:
            factor_path = CACHE / f"{accession}__factors.tsv"
            summary_path = CACHE / f"{accession}__summary.tsv"
            if not factor_path.exists():
                factor_path.write_bytes(download(f"https://www.metabolomicsworkbench.org/rest/study/study_id/{accession}/factors/json"))
            if not summary_path.exists():
                summary_path.write_bytes(download(f"https://www.metabolomicsworkbench.org/rest/study/study_id/{accession}/summary/json"))
            rows = parse_tsv(factor_path.read_bytes())
            fields = sorted({field for row in rows for field in row if HISTOLOGY_RE.search(field)})
            mucinous_rows = 0
            for row_index, row in enumerate(rows, start=1):
                hits = []
                for field, value in row.items():
                    value = value or ""
                    if HISTOLOGY_RE.search(field) or MUCINOUS_RE.search(value):
                        if value.strip():
                            hits.append((field, value.strip()))
                if any(MUCINOUS_RE.search(value) for _, value in hits):
                    mucinous_rows += 1
                for field, value in hits:
                    values.append(
                        {
                            "accession": accession,
                            "row": row_index,
                            "field": field,
                            "value": value,
                            "contains_mucinous": bool(MUCINOUS_RE.search(value)),
                        }
                    )
            report.update(
                {
                    "factor_rows": len(rows),
                    "histology_like_fields": " | ".join(fields),
                    "mucinous_rows": mucinous_rows,
                    "status": "audited",
                }
            )
        except Exception as exc:
            report["status"] = f"error: {type(exc).__name__}: {exc}"
        reports.append(report)
        print(f"[MW {position}/{len(studies)}] {accession}: {report['status']}")

    with (OUT / "study_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    with (OUT / "histology_values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["accession", "row", "field", "value", "contains_mucinous"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)

    errors = [row for row in reports if row["status"].startswith("error")]
    mucinous = [row for row in reports if int(row["mucinous_rows"]) > 0]
    summary = {
        "status": "public_crc_mw_histology_audit_complete",
        "formal": False,
        "workbench_studies": len(studies),
        "studies_audited": len(studies) - len(errors),
        "study_errors": len(errors),
        "studies_with_mucinous_rows": [row["accession"] for row in mucinous],
        "mucinous_study_details": mucinous,
        "claim_limit": "Public Metabolomics Workbench factor metadata audit only. Missing histology labels mean the deposited patient-level factors cannot support mucinous replication; unpublished metadata may exist.",
        "provenance": {
            "omicsdi_search_sha256": sha256(SEARCH),
            "study_audit_sha256": sha256(OUT / "study_audit.csv"),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"metadata audit incomplete: {len(errors)} study errors")


if __name__ == "__main__":
    main()
