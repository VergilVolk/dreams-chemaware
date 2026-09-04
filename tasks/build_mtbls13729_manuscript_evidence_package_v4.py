"""Extend the v3 manuscript package with the frozen BioAware benchmark boundary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"
V3 = BASE / "manuscript_evidence_package_v3"
BIO = BASE / "bioaware_algorithm_biology_bridge_v1"
OUT = BASE / "manuscript_evidence_package_v4"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    v3_report = load_json(V3 / "report.json")
    bio_report = load_json(BIO / "report.json")
    if v3_report.get("formal") is not True or bio_report.get("formal") is not True:
        raise RuntimeError("v3 manuscript package and BioAware bridge must both be formal")
    if bio_report["algorithm_verdict"]["sota_claim_allowed"] is not False:
        raise RuntimeError("BioAware claim boundary unexpectedly changed")

    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(V3 / "manuscript_result_evidence_table.csv")
    if "R10" in set(results.result_id):
        raise RuntimeError("R10 already exists in source v3 package")
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                [
                    {
                        "result_id": "R10",
                        "result": "BioAware supplies conservative family and mechanism context, not a confirmed identity upgrade",
                        "primary_number": "internal V3 +3.16pp (5/2, CI crosses 0); external V4 -0.13pp (4/5); external V6 +0.37pp (3/1, CI crosses 0)",
                        "algorithmic_contribution": "phenotype-blind structure, RT and metabolic-network evidence with abstention",
                        "evidence_grade": "version-specific frozen benchmarks; external gain not statistically confirmed",
                        "forbidden_claim": "do not call BioAware SOTA, combine model versions, or promote network neighbors to exact identities",
                        "source_artifact": "bioaware_algorithm_biology_bridge_v1/report.json",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    results.to_csv(OUT / "manuscript_result_evidence_table.csv", index=False)

    figures = pd.read_csv(V3 / "figure_manifest.csv")
    figures = pd.concat(
        [
            figures,
            pd.DataFrame(
                [
                    {
                        "figure": "Extended Data",
                        "message": "BioAware version-specific benchmark and biological claim ceiling",
                        "data": "internal V3, external V4/V6, failure decomposition and MTBLS13729 role ledger",
                        "claim_boundary": "family/context evidence only; no statistically confirmed external identity gain",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    figures.to_csv(OUT / "figure_manifest.csv", index=False)

    v3_text = (V3 / "RESULTS.md").read_text(encoding="utf-8")
    bio_text = """

## Result 7 — BioAware is retained as a conservative context expert, not an annotation breakthrough

The earliest one-hop Rhea pilot failed (21 queries; zero corrected, one introduced), showing that reaction adjacency cannot serve as identity evidence. A later frozen V3 router improved an internal 95-query RPLC panel by 3.16 percentage points (five corrected, two introduced), but its formula-cluster confidence interval crossed zero. Version-specific external evidence was mixed: V4 decreased Recall@1 by 0.13 percentage points across seven panels (four corrected, five introduced), whereas the frozen V6 router increased Recall@1 by 0.37 percentage points across five untouched panels (three corrected, one introduced), again with a confidence interval crossing zero.

The failure decomposition explains why weight tuning is insufficient: the true identity was absent from Rhea for 11 of 22 development errors, and raw network-MS2 evidence favored the wrong candidate or tied for five more. BioAware is therefore used in MTBLS13729 only to consolidate ion families, abstain under evidence conflict, and rank biological interpretations after identity evidence is fixed. It contributes zero exact identity promotions. This negative boundary is part of the method, because it prevents biologically attractive network neighbors from being mislabeled as identified metabolites.

## Updated algorithm-to-biology statement

The source paper annotated 345 of 9,766 features (3.53%). Official DreaMS expanded frozen shared-target candidate coverage to 3,417 of 16,953 (20.16%), E6 to 3,426 (20.21%), P2b to 3,588 (21.16%), and the three-way union to 3,599 (21.23%). These are coverage estimates, not accuracy gains. E6 and P2b provide embedding and candidate-expert evidence, respectively; BioAware supplies conservative family/context evidence. Exact biological conclusions remain anchored by source Level-1 identities, raw peak-resolved MS2, targeted EIC abundance and orthogonal cohort/literature evidence.
"""
    (OUT / "RESULTS.md").write_text(v3_text.rstrip() + bio_text, encoding="utf-8")

    for item in V3.iterdir():
        if item.name in {"report.json", "manuscript_result_evidence_table.csv", "figure_manifest.csv", "RESULTS.md"}:
            continue
        destination = OUT / item.name
        if item.is_file():
            shutil.copy2(item, destination)
    for suffix in ("png", "pdf"):
        source = BIO / f"bioaware_algorithm_biology_bridge.{suffix}"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / source.name)

    report = {
        "status": "mtbls13729_manuscript_evidence_package_v4_complete",
        "formal": True,
        "result_claims": int(len(results)),
        "figures": int(len(figures)),
        "biology_package_A_ready": v3_report["biology_package_A_ready"],
        "new_exact_metabolite_claims": 0,
        "bioaware_external_gain_confirmed": False,
        "bioaware_role": bio_report["mtbls13729_role"]["safe_use"],
        "hard_missing_items": list(v3_report["hard_missing_items"])
        + [
            "statistically confirmed BioAware external annotation gain",
            "complete official MetDNA3 MRN assets/workdir for exact reproduction",
        ],
        "claim_limit": v3_report["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
