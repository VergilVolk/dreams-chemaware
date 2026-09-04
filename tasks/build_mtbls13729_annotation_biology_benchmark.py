"""Build a denominator-safe MTBLS13729 annotation and biology benchmark.

The source paper, DreaMS, E6 and P2b use different native feature universes.
This audit therefore reports both native denominators and a shared RPLC
requantification-target denominator.  Candidate coverage, evidence tier and
three-way stability are kept separate throughout.

No model is fitted and no phenotype is used to select an annotation.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_XLSX = ROOT / "data/mtbls13729/source_paper_supplements/pr5c01260_si_005.xlsx"
SOURCE_HTML = ROOT / "data/mtbls13729/source_paper_supplements/fulltext.html"
TARGET_DIR = ROOT / "data/mtbls13729/ms1_consensus"
THREEWAY_LOG = ROOT / "mtbls13729_p2b_2326596.out"
BIOLOGY_LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
BIOLOGY_CLAIMS = ROOT / "data/mtbls13729/biology_package_a_release_v1/biology_claim_ledger_v1.csv"
OUTPUT = ROOT / "data/mtbls13729/annotation_biology_benchmark_v1"

PANELS = ("neg_rp", "pos_rp")
METHOD_LABELS = {
    "author_shared_rplc_coordinates": "Author RPLC annotations recovered on current targets",
    "official_dreams": "Official DreaMS",
    "experimental_e6": "E6 shared embedding",
    "frozen_p2b": "Frozen P2b candidate expert",
    "threeway_consensus": "Three-way stable consensus",
    "threeway_union": "Three-way candidate envelope",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_json_objects(text: str) -> list[dict[str, object]]:
    """Recover top-level JSON objects embedded in a mixed Slurm log."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, object]] = []
    position = 0
    while True:
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        position = start + consumed
    return objects


def require_one(objects: list[dict[str, object]], status: str, panel: str | None = None) -> dict[str, object]:
    matches = [item for item in objects if item.get("status") == status]
    if panel is not None:
        matches = [item for item in matches if item.get("panel") == panel]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {status=} {panel=}, found {len(matches)}")
    return matches[0]


def native_source_counts() -> dict[str, object]:
    raw = SOURCE_HTML.read_text(encoding="utf-8")
    plain = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())
    patterns = {
        "pos_detected": r"positive ion mode.*?total of (\d+) molecular features",
        "pos_ms2": r"among which (\d+) features possessed MS/MS",
        "neg_detected": r"negative ion mode.*?(\d+) features were detected",
        "neg_ms2": r"with (\d+) retaining MS\s*2 data",
        "pos_annotated": r"total of (\d+) metabolites \(in ESI \+ \)",
        "neg_annotated": r"and (\d+) metabolites \(in ESI [–-] \)",
    }
    values: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, plain, flags=re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"source-paper denominator not found: {key}")
        values[key] = int(match.group(1))
    values["detected_total"] = values["pos_detected"] + values["neg_detected"]
    values["ms2_total"] = values["pos_ms2"] + values["neg_ms2"]
    values["annotated_total"] = values["pos_annotated"] + values["neg_annotated"]
    values["annotation_rate_all_detected"] = values["annotated_total"] / values["detected_total"]
    values["annotation_rate_ms2_eligible"] = values["annotated_total"] / values["ms2_total"]
    return values


def load_source_annotations() -> pd.DataFrame:
    source = pd.read_excel(SOURCE_XLSX, sheet_name="metabolites", header=1)
    source = source.loc[pd.to_numeric(source["m/z"], errors="coerce").notna()].copy()
    if len(source) != 345:
        raise RuntimeError(f"expected 345 source annotations, found {len(source)}")
    source["polarity"] = np.where(source["Adducts"].astype(str).str.endswith("-"), "neg", "pos")
    source["panel"] = source["polarity"] + "_rp"
    return source


def map_source_rplc_to_targets(source: pd.DataFrame, ppm_tolerance: float = 10.0, rt_tolerance_sec: float = 20.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source = source.loc[source["Type"].astype(str).str.upper().eq("RPLC")].copy()
    for panel in PANELS:
        targets = pd.read_csv(TARGET_DIR / f"{panel}__requantification_targets.csv.gz")
        for source_row, item in source.loc[source.panel.eq(panel)].iterrows():
            ppm = np.abs(targets.mz - float(item["m/z"])) / float(item["m/z"]) * 1e6
            drt = np.abs(targets.rt_sec - float(item["RT [min]"]) * 60.0)
            eligible = (ppm <= ppm_tolerance) & (drt <= rt_tolerance_sec)
            if not bool(eligible.any()):
                continue
            cost = (ppm / ppm_tolerance + drt / rt_tolerance_sec).where(eligible, np.inf)
            target_row = int(cost.idxmin())
            rows.append(
                {
                    "source_row": int(source_row),
                    "panel": panel,
                    "source_name": str(item["metabolites"]),
                    "source_level": str(item["MSI(Metabolomics Standards Initiative)"]),
                    "source_mz": float(item["m/z"]),
                    "source_rt_sec": float(item["RT [min]"]) * 60.0,
                    "source_inchikey": str(item["InChIKey"]),
                    "feature_id": int(targets.loc[target_row, "feature_id"]),
                    "target_mz": float(targets.loc[target_row, "mz"]),
                    "target_rt_sec": float(targets.loc[target_row, "rt_sec"]),
                    "ppm": float(ppm.loc[target_row]),
                    "drt_sec": float(drt.loc[target_row]),
                }
            )
    return pd.DataFrame(rows)


def build_module_ledger() -> pd.DataFrame:
    candidates = pd.read_csv(BIOLOGY_LEDGER)
    selected = candidates.loc[
        candidates.feature_id.astype(str).isin(
            ["703", "1597", "3019", "1717", "3222", "345", "374", "428"]
        )
    ].copy()
    origins = {
        703: ("source Level-1 identity + classical MS2 + DreaMS-enabled cross-panel recovery", "primary free-Neu5Ac pool anchor"),
        1597: ("DreaMS candidate discovery + peak-resolved MS2 + ion-family folding", "modified-guanosine family"),
        3019: ("DreaMS candidate discovery + peak-resolved MS2 + ion-family folding", "dimethylguanosine family"),
        1717: ("DreaMS/raw-MS2 family evidence + cross-chromatography concordance", "acetylated-polyamine axis"),
        3222: ("three-way retained candidate + E6 evidence strengthening + class fragments", "long-chain acylcarnitine axis"),
        345: ("source Level-1 identity + classical MS2 orthogonal recovery", "expanded amino-acid pool"),
        374: ("source Level-1 identity + classical MS2 orthogonal recovery", "expanded amino-acid pool"),
        428: ("cross-panel discordance control", "deliberate identity downgrade"),
    }
    selected["algorithmic_origin"] = selected.feature_id.map(lambda x: origins[int(x)][0])
    selected["biological_role"] = selected.feature_id.map(lambda x: origins[int(x)][1])
    selected["identity_is_not_upgraded_by_abundance"] = True
    columns = [
        "feature_id", "label", "module", "defensible_identity", "manuscript_evidence_tier",
        "algorithmic_origin", "biological_role", "pairs", "mean_log2fc", "positive_pairs",
        "peak_resolved_ms2_spectra", "claim_ceiling", "identity_is_not_upgraded_by_abundance",
    ]
    return selected.loc[:, columns].sort_values(["module", "feature_id"], kind="stable")


def main() -> None:
    required = [SOURCE_XLSX, SOURCE_HTML, THREEWAY_LOG, BIOLOGY_LEDGER, BIOLOGY_CLAIMS]
    required += [TARGET_DIR / f"{panel}__requantification_targets.csv.gz" for panel in PANELS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    source = load_source_annotations()
    source_native = native_source_counts()
    source_rplc = source.loc[source.Type.astype(str).str.upper().eq("RPLC")]
    mapped = map_source_rplc_to_targets(source)
    mapped.to_csv(OUTPUT / "author_rplc_to_current_targets.csv", index=False)

    log_objects = extract_json_objects(THREEWAY_LOG.read_text(encoding="utf-8"))
    threeway = require_one(log_objects, "mtbls13729_threeway_annotation_comparison_complete")
    p2b_reports = {
        panel: require_one(log_objects, "mtbls13729_p2b_vs_dreams_inference_complete", panel)
        for panel in PANELS
    }
    e6_reports = {
        panel: require_one(log_objects, "mtbls13729_embedding_retrieval_complete", panel)
        for panel in PANELS
    }

    comparison_rows: list[dict[str, object]] = []
    panel_details: dict[str, object] = {}
    total_targets = 0
    aggregate = {
        "author_shared_rplc_coordinates": 0,
        "official_dreams": 0,
        "experimental_e6": 0,
        "frozen_p2b": 0,
        "threeway_consensus": 0,
        "threeway_union": 0,
    }
    aggregate_supported = {"official_dreams": 0, "experimental_e6": 0, "frozen_p2b": 0}

    for panel in PANELS:
        targets = pd.read_csv(TARGET_DIR / f"{panel}__requantification_targets.csv.gz")
        denominator = int(len(targets))
        total_targets += denominator
        p2b = p2b_reports[panel]
        e6 = e6_reports[panel]
        panel_threeway = threeway["panels"][panel]
        summary = p2b["systems"]
        author_count = int(mapped.loc[mapped.panel.eq(panel), "feature_id"].nunique())
        counts = {
            "author_shared_rplc_coordinates": author_count,
            "official_dreams": int(summary["dreams"]["annotated_features"]),
            "experimental_e6": int(e6["annotated_features"]),
            "frozen_p2b": int(summary["p2b"]["annotated_features"]),
            "threeway_consensus": int(panel_threeway["feature_threeway_consensus"]),
            "threeway_union": int(panel_threeway["features_union"]),
        }
        supported = {
            "official_dreams": int(summary["dreams"]["level2a_supported"]),
            "frozen_p2b": int(summary["p2b"]["level2a_supported"]),
            "experimental_e6": 31 if panel == "neg_rp" else 245,
        }
        mass_candidates = int(p2b["features_with_candidates"])
        linked_features = int(e6["linked_features"])
        for method, count in counts.items():
            comparison_rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "count": count,
                    "denominator": denominator,
                    "rate": count / denominator,
                    "denominator_definition": "all frozen MS1 requantification targets in this RPLC panel",
                    "evidence_boundary": (
                        "coordinate recovery of source annotation; not a rerun" if method.startswith("author_")
                        else "stable across all three methods; identity still putative" if method == "threeway_consensus"
                        else "candidate envelope; not one deployable decision system" if method == "threeway_union"
                        else "candidate assignment; not automatically MSI Level 2"
                    ),
                }
            )
            aggregate[method] += count
        for method, count in supported.items():
            aggregate_supported[method] += count
        panel_details[panel] = {
            "targets": denominator,
            "ms2_linked_targets": linked_features,
            "targets_with_mass_candidates": mass_candidates,
            "counts": counts,
            "level2a_supported": supported,
            "candidate_graph_rates": {
                "official_dreams": counts["official_dreams"] / mass_candidates,
                "experimental_e6": counts["experimental_e6"] / mass_candidates,
                "frozen_p2b": counts["frozen_p2b"] / mass_candidates,
            },
        }

    for method, count in aggregate.items():
        comparison_rows.append(
            {
                "panel": "combined_rplc",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "count": count,
                "denominator": total_targets,
                "rate": count / total_targets,
                "denominator_definition": "all frozen MS1 requantification targets across neg_rp and pos_rp",
                "evidence_boundary": (
                    "coordinate recovery of source annotation; not a rerun" if method.startswith("author_")
                    else "stable across all three methods; identity still putative" if method == "threeway_consensus"
                    else "candidate envelope; not one deployable decision system" if method == "threeway_union"
                    else "candidate assignment; not automatically MSI Level 2"
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT / "annotation_rate_comparison.csv", index=False)

    module_ledger = build_module_ledger()
    module_ledger.to_csv(OUTPUT / "algorithm_to_biology_module_ledger.csv", index=False)
    claims = pd.read_csv(BIOLOGY_CLAIMS)
    claims.to_csv(OUTPUT / "frozen_biology_claim_ledger.csv", index=False)

    source_rplc_counts = {
        panel: int(source_rplc.panel.eq(panel).sum()) for panel in PANELS
    }
    report = {
        "status": "mtbls13729_annotation_biology_benchmark_v1_complete",
        "formal": True,
        "source_paper_native": source_native,
        "source_paper_rplc_annotations": {
            "total": int(len(source_rplc)),
            "by_panel": source_rplc_counts,
            "mapped_rows_10ppm_20sec": int(len(mapped)),
            "mapped_unique_targets": int(mapped.groupby(["panel", "feature_id"]).ngroups),
            "mapped_unique_level1_targets": int(mapped.loc[mapped.source_level.eq("Level 1")].groupby(["panel", "feature_id"]).ngroups),
            "mapped_unique_level2_targets": int(mapped.loc[mapped.source_level.eq("Level 2")].groupby(["panel", "feature_id"]).ngroups),
        },
        "shared_rplc_target_universe": {
            "targets": total_targets,
            "systems": {
                method: {
                    "count": int(count),
                    "rate": float(count / total_targets),
                    **({"level2a_supported": int(aggregate_supported[method])} if method in aggregate_supported else {}),
                }
                for method, count in aggregate.items()
            },
            "panels": panel_details,
        },
        "increments_vs_official": {
            "e6_candidate_assignments": int(aggregate["experimental_e6"] - aggregate["official_dreams"]),
            "e6_level2a_supported": int(aggregate_supported["experimental_e6"] - aggregate_supported["official_dreams"]),
            "p2b_candidate_assignments": int(aggregate["frozen_p2b"] - aggregate["official_dreams"]),
            "p2b_level2a_supported": int(aggregate_supported["frozen_p2b"] - aggregate_supported["official_dreams"]),
            "threeway_union_candidate_assignments": int(aggregate["threeway_union"] - aggregate["official_dreams"]),
        },
        "biology_integration": {
            "module_ledger_rows": int(len(module_ledger)),
            "claim_status_counts": {str(k): int(v) for k, v in claims.status.value_counts().items()},
            "primary_discovery": "free-Neu5Ac pool expansion with pool-to-donor/destination decoupling",
            "supporting_axes": [
                "modified-guanosine ion families",
                "acetylated-polyamine axis",
                "long-chain acylcarnitine imbalance",
                "expanded amino-acid pool",
            ],
            "bioaware_role": "context, ion-family consolidation and conflict/abstention; no identity-count increment claimed",
        },
        "interpretation": {
            "source_native_rate": "345/9766 all detected features and 345/6054 MS2-bearing features",
            "tool_rate": "must be reported as candidate coverage, Level2a-supported evidence and three-way stability, not as one inflated number",
            "accuracy_limit": "MTBLS13729 lacks large-scale structure truth; changed or added assignments are not corrections",
            "scope_limit": "current same-protocol comparison covers RPLC only; HILIC three-way inference is not yet available",
        },
        "provenance": {
            "source_xlsx_sha256": sha256(SOURCE_XLSX),
            "source_html_sha256": sha256(SOURCE_HTML),
            "threeway_log_sha256": sha256(THREEWAY_LOG),
            "biology_ledger_sha256": sha256(BIOLOGY_LEDGER),
            "biology_claims_sha256": sha256(BIOLOGY_CLAIMS),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    source_all = 100 * source_native["annotation_rate_all_detected"]
    source_ms2 = 100 * source_native["annotation_rate_ms2_eligible"]
    lines = [
        "# MTBLS13729 annotation-to-biology benchmark v1",
        "",
        "## Native source-paper denominator",
        "",
        f"- Author annotations: 345 / 9,766 detected features = **{source_all:.2f}%**.",
        f"- Author annotations: 345 / 6,054 MS2-bearing features = **{source_ms2:.2f}%**.",
        "- These 345 include 157 Level 1 and 188 Level 2 identities across RPLC and HILIC.",
        "",
        "## Shared RPLC requantification-target denominator",
        "",
        "| Method | Count | Rate | Boundary |",
        "|---|---:|---:|---|",
    ]
    for method in aggregate:
        count = aggregate[method]
        boundary = comparison.loc[
            comparison.panel.eq("combined_rplc") & comparison.method.eq(method), "evidence_boundary"
        ].iloc[0]
        lines.append(f"| {METHOD_LABELS[method]} | {count:,} | {100*count/total_targets:.2f}% | {boundary} |")
    lines.extend(
        [
            "",
            "## Evidence-tier increment versus official DreaMS",
            "",
            f"- E6: {aggregate['experimental_e6']-aggregate['official_dreams']:+d} candidate assignments; "
            f"{aggregate_supported['experimental_e6']-aggregate_supported['official_dreams']:+d} Level2a-supported features.",
            f"- P2b: {aggregate['frozen_p2b']-aggregate['official_dreams']:+d} candidate assignments; "
            f"{aggregate_supported['frozen_p2b']-aggregate_supported['official_dreams']:+d} Level2a-supported features.",
            "- BioAware contributes context, family consolidation and conflict handling; it does not add to the annotation numerator.",
            "",
            "## Claim boundary",
            "",
            "MTBLS13729 has no large-scale structure truth. Candidate expansion is annotation coverage, not accuracy. "
            "The biology is tested only after frozen annotation and remains abundance-level discovery unless supported by standards or an independent cohort.",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
