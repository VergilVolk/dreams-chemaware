"""Audit public CRC metabolomics study metadata for usable histology labels.

This searches the frozen OmicsDI CRC-metabolomics result, downloads only small
MetaboLights sample metadata tables, and records whether patient-level
mucinous/histology fields are actually present.  It never downloads raw MS data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "data/external/omicsdi_crc_metabolomics_search_20260830.json"
OUT = ROOT / "data/external/public_crc_metabolomics_histology_audit_v1"
CACHE = OUT / "sample_metadata"

HISTOLOGY_RE = re.compile(
    r"mucin|histolog|adenocarcinoma|patholog|tumou?r[ _-]?type|morpholog|signet",
    re.IGNORECASE,
)
MUCINOUS_RE = re.compile(r"mucinous|mucin-producing|mucin producing", re.IGNORECASE)
ENCODED_MUCINOUS_RE = re.compile(r"(?:^|[-_])R?mu(?:$|[-_])", re.IGNORECASE)
HUMAN_TISSUE_RE = re.compile(
    r"human|patient|tumou?r|normal tissue|adjacent|colon tissue|colorectal tissue",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def get_bytes(url: str, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "DreaMS-public-metadata-audit/1.0"})
            with urlopen(request, timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {error}")


def parse_table(raw: bytes) -> tuple[list[str], list[list[str]]]:
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    if not rows:
        return [], []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return rows[0], rows[1:]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SEARCH.read_text(encoding="utf-8-sig"))
    studies = [row for row in payload["datasets"] if str(row["id"]).startswith("MTBLS")]
    reports: list[dict] = []
    values_out: list[dict] = []

    for position, study in enumerate(studies, start=1):
        accession = str(study["id"])
        title = str(study.get("title") or "")
        description = re.sub(r"<[^>]+>", " ", str(study.get("description") or ""))
        report = {
            "accession": accession,
            "title": title,
            "sample_file": "",
            "n_sample_rows": 0,
            "histology_like_columns": "",
            "mucinous_rows": 0,
            "explicit_mucinous_rows": 0,
            "encoded_mucinous_rows": 0,
            "human_tissue_text_screen": bool(HUMAN_TISSUE_RE.search(title + " " + description)),
            "status": "",
        }
        try:
            listing_url = f"https://www.ebi.ac.uk/metabolights/ws/studies/{accession}/files"
            listing = json.loads(get_bytes(listing_url).decode("utf-8"))
            files = [entry.get("file", "") for entry in listing.get("study", [])]
            sample_files = sorted({name for name in files if re.match(r"^s_.*\.txt$", name, re.I)})
            if not sample_files:
                report["status"] = "no_sample_metadata"
                reports.append(report)
                continue
            # Some studies expose more than one version; audit every table but
            # report the union without silently choosing a favorable version.
            all_columns: set[str] = set()
            mucinous_rows = 0
            explicit_mucinous_rows = 0
            encoded_mucinous_rows = 0
            total_rows = 0
            used: list[str] = []
            for sample_file in sample_files:
                cached = CACHE / f"{accession}__{Path(sample_file).name}"
                if not cached.exists():
                    url = f"https://www.ebi.ac.uk/metabolights/ws/studies/{accession}/download?file={quote(sample_file)}"
                    cached.write_bytes(get_bytes(url))
                header, rows = parse_table(cached.read_bytes())
                if not header:
                    continue
                used.append(sample_file)
                total_rows += len(rows)
                selected = [idx for idx, name in enumerate(header) if HISTOLOGY_RE.search(name)]
                all_columns.update(header[idx] for idx in selected)
                for row_index, row in enumerate(rows, start=1):
                    selected_values = [row[idx].strip() for idx in selected if idx < len(row) and row[idx].strip()]
                    row_values = [value.strip() for value in row if value.strip()]
                    explicit_hit = any(MUCINOUS_RE.search(value) for value in row_values)
                    encoded_hit = any(ENCODED_MUCINOUS_RE.search(value) for value in row_values)
                    if explicit_hit:
                        explicit_mucinous_rows += 1
                    if encoded_hit:
                        encoded_mucinous_rows += 1
                    if explicit_hit or encoded_hit:
                        mucinous_rows += 1
                    for idx in selected:
                        value = row[idx].strip() if idx < len(row) else ""
                        if not value:
                            continue
                        values_out.append(
                            {
                                "accession": accession,
                                "sample_file": sample_file,
                                "row": row_index,
                                "column": header[idx],
                                "value": value,
                                "contains_mucinous": bool(MUCINOUS_RE.search(value)),
                            }
                        )
            report.update(
                {
                    "sample_file": " | ".join(used),
                    "n_sample_rows": total_rows,
                    "histology_like_columns": " | ".join(sorted(all_columns)),
                    "mucinous_rows": mucinous_rows,
                    "explicit_mucinous_rows": explicit_mucinous_rows,
                    "encoded_mucinous_rows": encoded_mucinous_rows,
                    "status": "audited",
                }
            )
        except Exception as exc:  # fail-open per study, fail-closed in summary
            report["status"] = f"error: {type(exc).__name__}: {exc}"
        reports.append(report)
        print(f"[metadata {position}/{len(studies)}] {accession}: {report['status']}")

    with (OUT / "study_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    with (OUT / "histology_values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["accession", "sample_file", "row", "column", "value", "contains_mucinous"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values_out)

    audited = [row for row in reports if row["status"] == "audited"]
    errors = [row for row in reports if row["status"].startswith("error")]
    mucinous = [row for row in audited if int(row["mucinous_rows"]) > 0]
    summary = {
        "status": "public_crc_metabolomics_histology_audit_complete",
        "formal": False,
        "search_results": len(payload["datasets"]),
        "metabolights_studies": len(studies),
        "sample_metadata_audited": len(audited),
        "study_errors": len(errors),
        "studies_with_histology_like_columns": [
            row["accession"] for row in audited if row["histology_like_columns"]
        ],
        "studies_with_mucinous_rows": [row["accession"] for row in mucinous],
        "mucinous_study_details": mucinous,
        "claim_limit": "Keyword and ISA sample-metadata audit only. Encoded Rmu labels are reported separately from explicit histology fields. Absence of a mucinous field or encoded label does not prove that unpublished clinical metadata do not exist; it means the public patient-level metadata cannot support a mucinous replication.",
        "provenance": {
            "omicsdi_search_sha256": sha256(SEARCH),
            "study_audit_sha256": sha256(OUT / "study_audit.csv"),
            "histology_values_sha256": sha256(OUT / "histology_values.csv"),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"metadata audit incomplete: {len(errors)} study errors; see study_audit.csv")


if __name__ == "__main__":
    main()
