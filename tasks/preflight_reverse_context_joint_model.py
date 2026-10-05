#!/usr/bin/env python
"""Stage A0: build a public chemical-to-context observation ledger.

This is deliberately a feasibility audit, not a context predictor.  It queries the
public StructureMASST/FASSTrecords Datasette instance for representative reference
spectra and their precomputed public-repository matches, while retaining project and
technical metadata needed for later project-held-out and detectability-aware tests.

Absence from the returned match table is *unlabelled*, never a biological negative.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


API = "https://masst-records.gnps2.org/masst_records.json"
USER_AGENT = "DreaMS-ReverseContext-A0/1.0"
SUPERCLASSES = (
    "Lipids and lipid-like molecules",
    "Organic acids and derivatives",
    "Organoheterocyclic compounds",
    "Phenylpropanoids and polyketides",
    "Benzenoids",
    "Alkaloids and derivatives",
    "Organic oxygen compounds",
    "Nucleosides, nucleotides, and analogues",
)
BLANK_TYPES = {
    "blank_analysis",
    "blank_QC",
    "blank_extraction",
    "blank_culturemedia",
    "pool_QC",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv_gz(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp.gz")
    os.close(descriptor)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DatasetteClient:
    def __init__(self, cache_dir: Path, timeout: int = 120, retries: int = 4):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.retries = retries
        cache_dir.mkdir(parents=True, exist_ok=True)

    def query(self, sql: str) -> list[dict[str, Any]]:
        key = _sha256_bytes(sql.encode("utf-8"))
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("sql") != sql or not isinstance(payload.get("rows"), list):
                raise RuntimeError(f"invalid API cache: {cache}")
            return payload["rows"]

        url = API + "?" + urllib.parse.urlencode({"sql": sql, "_shape": "array"})
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    rows = json.load(response)
                if not isinstance(rows, list):
                    raise RuntimeError("Datasette did not return an array")
                _atomic_json(cache, {"sql": sql, "rows": rows})
                return rows
            except Exception as exc:  # network boundary: retry, then fail closed
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Datasette query failed after {self.retries} attempts") from error


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def select_anchors(
    client: DatasetteClient,
    per_superclass: int,
    selection_mode: str,
    selection_seed: int,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for superclass in SUPERCLASSES:
        # Select a concrete representative row first, then recover every field
        # from that exact row.  Aggregating each metadata column independently
        # can silently create an impossible chimera (ID from spectrum A, adduct
        # or precursor mass from spectrum B).
        if selection_mode == "lexicographic":
            selection_order = "InChIKey_smiles_firstBlock"
        elif selection_mode == "seeded":
            # SQLite has no portable seeded random function.  This fixed linear
            # congruential permutation gives a reproducible, outcome-blind order
            # over concrete representative spectrum IDs.
            selection_order = (
                "((min(spectrum_id_int) * 1103515245 + "
                f"{int(selection_seed)}) % 2147483647)"
            )
        else:
            raise ValueError(f"unknown selection mode: {selection_mode}")
        sql = f"""
with chosen as (
  select min(spectrum_id_int) as spectrum_id_int,
         InChIKey_smiles_firstBlock as ik14
  from library_table
  where representative_spectrum_int = spectrum_id_int
    and InChIKey_smiles_firstBlock is not null
    and length(InChIKey_smiles_firstBlock) = 14
    and Compound_Source in ('commercial', 'isolated')
    and GNPS_library_membership = 'GNPS-LIBRARY'
    and classyfire_superclass = {_sql_quote(superclass)}
  group by InChIKey_smiles_firstBlock
  order by {selection_order}, InChIKey_smiles_firstBlock
  limit {int(per_superclass)}
)
select c.spectrum_id_int,
       c.ik14,
       l.Compound_Name as compound_name,
       l.Adduct as adduct,
       l.Ion_Mode as ion_mode,
       l.Precursor_MZ as precursor_mz,
       l.classyfire_superclass
from chosen c
join library_table l on l.spectrum_id_int = c.spectrum_id_int
order by c.ik14
""".strip()
        for row in client.query(sql):
            ik14 = str(row["ik14"])
            if ik14 in seen:
                continue
            seen.add(ik14)
            anchors.append(row)
    return anchors


def query_denominators(client: DatasetteClient) -> list[dict[str, Any]]:
    sql = """
select ATTRIBUTE_DatasetAccession as dataset,
       SampleType as sample_type,
       UBERONBodyPartName as body_part,
       HealthStatus as health_status,
       IonizationSourceAndPolarity as polarity,
       MassSpectrometer as instrument,
       count(*) as n_files,
       sum(cast(MS2spectra_count as integer)) as n_ms2
from redu_table
group by ATTRIBUTE_DatasetAccession, SampleType, UBERONBodyPartName,
         HealthStatus, IonizationSourceAndPolarity, MassSpectrometer
""".strip()
    return client.query(sql)


def query_occurrences(
    client: DatasetteClient,
    spectrum_ids: list[int],
    cosine: float,
    matching_peaks: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(spectrum_ids), batch_size):
        batch = spectrum_ids[start : start + batch_size]
        id_sql = ",".join(str(int(value)) for value in batch)
        sql = f"""
select m.spectrum_id_int,
       r.ATTRIBUTE_DatasetAccession as dataset,
       r.SampleType as sample_type,
       r.UBERONBodyPartName as body_part,
       r.HealthStatus as health_status,
       r.IonizationSourceAndPolarity as polarity,
       r.MassSpectrometer as instrument,
       count(distinct m.mri_id_int) as matched_files,
       count(*) as matched_spectra,
       avg(m.cosine) as mean_cosine,
       max(m.cosine) as max_cosine,
       avg(m.matching_peaks) as mean_matching_peaks
from masst_table m
join redu_table r on r.mri_id_int = m.mri_id_int
where m.spectrum_id_int in ({id_sql})
  and m.cosine >= {float(cosine):.8f}
  and m.matching_peaks >= {int(matching_peaks)}
group by m.spectrum_id_int, r.ATTRIBUTE_DatasetAccession, r.SampleType,
         r.UBERONBodyPartName, r.HealthStatus,
         r.IonizationSourceAndPolarity, r.MassSpectrometer
""".strip()
        rows.extend(client.query(sql))
        print(
            f"[A0 occurrences] {min(start + batch_size, len(spectrum_ids))}/"
            f"{len(spectrum_ids)} anchors; rows={len(rows)}",
            flush=True,
        )
    return rows


def _nonmissing(value: Any) -> bool:
    return value not in (None, "", "missing value", "nan")


def summarize(
    anchors: list[dict[str, Any]],
    denominators: list[dict[str, Any]],
    occurrences: list[dict[str, Any]],
    cosine: float,
    matching_peaks: int,
) -> dict[str, Any]:
    by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        by_anchor[int(row["spectrum_id_int"])].append(row)

    anchor_metrics: list[dict[str, Any]] = []
    for anchor in anchors:
        spectrum_id = int(anchor["spectrum_id_int"])
        group = by_anchor.get(spectrum_id, [])
        projects = {str(row["dataset"]) for row in group if _nonmissing(row["dataset"])}
        sample_types = {
            str(row["sample_type"])
            for row in group
            if _nonmissing(row["sample_type"]) and row["sample_type"] not in BLANK_TYPES
        }
        biological_files = sum(
            int(row["matched_files"])
            for row in group
            if row.get("sample_type") not in BLANK_TYPES
        )
        blank_files = sum(
            int(row["matched_files"])
            for row in group
            if row.get("sample_type") in BLANK_TYPES
        )
        counts = Counter()
        for row in group:
            sample_type = row.get("sample_type")
            if _nonmissing(sample_type) and sample_type not in BLANK_TYPES:
                counts[str(sample_type)] += int(row["matched_files"])
        total = sum(counts.values())
        entropy = 0.0
        if total:
            probabilities = [count / total for count in counts.values()]
            entropy = -sum(prob * math.log(prob) for prob in probabilities)
            if len(probabilities) > 1:
                entropy /= math.log(len(probabilities))
        anchor_metrics.append(
            {
                **anchor,
                "projects_with_matches": len(projects),
                "biological_sample_types": len(sample_types),
                "biological_matched_files": biological_files,
                "blank_matched_files": blank_files,
                "blank_fraction": blank_files / max(1, biological_files + blank_files),
                "sample_type_entropy": entropy,
                "dominant_sample_type": counts.most_common(1)[0][0] if counts else None,
                "dominant_sample_type_fraction": (
                    counts.most_common(1)[0][1] / total if counts else None
                ),
            }
        )

    denominator_projects = {
        str(row["dataset"]) for row in denominators if _nonmissing(row.get("dataset"))
    }
    denominator_files = sum(int(row.get("n_files") or 0) for row in denominators)
    denominator_ms2 = sum(int(row.get("n_ms2") or 0) for row in denominators)
    anchors_with_hits = [row for row in anchor_metrics if row["projects_with_matches"] > 0]
    anchors_multproject = [row for row in anchor_metrics if row["projects_with_matches"] >= 2]
    anchors_multicontext = [row for row in anchor_metrics if row["biological_sample_types"] >= 2]
    anchors_with_blanks = [row for row in anchor_metrics if row["blank_matched_files"] > 0]
    superclasses = Counter(str(row["classyfire_superclass"]) for row in anchors)

    gates = {
        "anchors_ge_40": len(anchors) >= 40,
        "superclasses_ge_6": len(superclasses) >= 6,
        "anchors_with_hits_ge_25": len(anchors_with_hits) >= 25,
        "anchors_in_multiple_projects_ge_20": len(anchors_multproject) >= 20,
        "anchors_in_multiple_contexts_ge_15": len(anchors_multicontext) >= 15,
        "projects_ge_500": len(denominator_projects) >= 500,
        "technical_fields_present": any(
            _nonmissing(row.get("polarity")) and _nonmissing(row.get("instrument"))
            for row in denominators
        ),
    }
    return {
        "status": "reverse_context_joint_model_a0_complete",
        "formal": True,
        "task": "public chemical-to-context observation-ledger feasibility",
        "thresholds": {
            "cosine_minimum": cosine,
            "matching_peaks_minimum": matching_peaks,
        },
        "anchors": {
            "selected": len(anchors),
            "superclasses": dict(sorted(superclasses.items())),
            "with_any_public_match": len(anchors_with_hits),
            "with_matches_in_ge_2_projects": len(anchors_multproject),
            "with_matches_in_ge_2_biological_sample_types": len(anchors_multicontext),
            "with_any_blank_or_qc_match": len(anchors_with_blanks),
        },
        "public_metadata": {
            "strata": len(denominators),
            "projects": len(denominator_projects),
            "files": denominator_files,
            "ms2_spectra_reported": denominator_ms2,
        },
        "occurrence_rows": len(occurrences),
        "gates": gates,
        "pass_to_a1": all(gates.values()),
        "interpretation": (
            "A passing A0 means the public database can support a project-held-out, "
            "detectability-aware existence test. It is not evidence that biological "
            "context is predictable or that a joint model is novel."
        ),
        "label_contract": {
            "match": "positive observation under a fixed spectral threshold",
            "no_match": "unlabelled; never treated as a biological negative",
            "blank_match": "technical-confounding signal retained for audit",
            "disease_labels": "not used for threshold or model selection in A0",
        },
        "next_stage": (
            "A1 freezes projects into development and project-held-out folds, then "
            "compares hit-count B0, technical-only B1, and chemical-plus-technical M1."
        ),
        "anchor_metrics": anchor_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/reverse_context_joint_a0"),
    )
    parser.add_argument("--anchors-per-superclass", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=("seeded", "lexicographic"),
        default="seeded",
    )
    parser.add_argument("--selection-seed", type=int, default=20260912)
    parser.add_argument(
        "--allow-class-shortfall",
        action="store_true",
        help="Retain every available member when a superclass has fewer anchors than requested.",
    )
    parser.add_argument("--cosine", type=float, default=0.85)
    parser.add_argument("--matching-peaks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.anchors_per_superclass < 1:
        raise ValueError("anchors-per-superclass must be positive")
    if not 0.0 < args.cosine <= 1.0:
        raise ValueError("cosine must be in (0, 1]")
    if args.matching_peaks < 3:
        raise ValueError("matching-peaks must be at least 3")
    report_path = args.output_dir / "report.json"
    if report_path.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite completed A0: {report_path}")

    client = DatasetteClient(args.output_dir / "api_cache", timeout=args.timeout)
    anchors = select_anchors(
        client,
        args.anchors_per_superclass,
        args.selection_mode,
        args.selection_seed,
    )
    expected = args.anchors_per_superclass * len(SUPERCLASSES)
    if len(anchors) != expected and not args.allow_class_shortfall:
        raise RuntimeError(f"anchor selection incomplete: expected {expected}, got {len(anchors)}")
    represented = {str(row["classyfire_superclass"]) for row in anchors}
    if args.allow_class_shortfall and represented != set(SUPERCLASSES):
        missing = sorted(set(SUPERCLASSES) - represented)
        raise RuntimeError(f"superclass selection empty for: {missing}")
    denominators = query_denominators(client)
    if not denominators:
        raise RuntimeError("public metadata denominator query returned no rows")
    occurrences = query_occurrences(
        client,
        [int(row["spectrum_id_int"]) for row in anchors],
        args.cosine,
        args.matching_peaks,
        args.batch_size,
    )
    report = summarize(
        anchors, denominators, occurrences, args.cosine, args.matching_peaks
    )
    report["provenance"] = {
        "api": API,
        "api_query_cache_files": len(list((args.output_dir / "api_cache").glob("*.json"))),
        "script_sha256": _sha256_file(Path(__file__)),
    }
    report["parameters"] = {
        "anchors_per_superclass": args.anchors_per_superclass,
        "selection_mode": args.selection_mode,
        "selection_seed": args.selection_seed,
        "allow_class_shortfall": args.allow_class_shortfall,
        "requested_maximum_anchors": expected,
        "selected_anchors": len(anchors),
        "batch_size": args.batch_size,
        "cosine": args.cosine,
        "matching_peaks": args.matching_peaks,
    }

    occurrence_fields = [
        "spectrum_id_int",
        "dataset",
        "sample_type",
        "body_part",
        "health_status",
        "polarity",
        "instrument",
        "matched_files",
        "matched_spectra",
        "mean_cosine",
        "max_cosine",
        "mean_matching_peaks",
    ]
    denominator_fields = [
        "dataset",
        "sample_type",
        "body_part",
        "health_status",
        "polarity",
        "instrument",
        "n_files",
        "n_ms2",
    ]
    _atomic_csv_gz(args.output_dir / "occurrences.csv.gz", occurrences, occurrence_fields)
    _atomic_csv_gz(args.output_dir / "opportunities.csv.gz", denominators, denominator_fields)
    _atomic_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "anchor_metrics"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
