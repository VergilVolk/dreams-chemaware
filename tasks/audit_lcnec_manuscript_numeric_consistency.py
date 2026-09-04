"""Fail closed when key LCNEC manuscript numbers drift across frozen drafts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_manuscript_numeric_audit_v1"
DOCS = {
    "abstract": ROOT / "docs/LCNEC_MANUSCRIPT_ABSTRACT_DRAFT_20260901.md",
    "results": ROOT / "docs/LCNEC_MANUSCRIPT_RESULTS_DRAFT_20260831.md",
    "methods": ROOT / "docs/LCNEC_MANUSCRIPT_METHODS_OUTLINE_20260901.md",
    "discussion": ROOT / "docs/LCNEC_MANUSCRIPT_DISCUSSION_DRAFT_20260901.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    texts = {name: " ".join(path.read_text(encoding="utf-8").split()).lower() for name, path in DOCS.items()}
    checks = [
        ("abstract", "263 qualified denominator", "263"),
        ("abstract", "42 source overlap", "42"),
        ("abstract", "158 DreaMS coverage", "158/263"),
        ("abstract", "66 full evidence", "66/263"),
        ("abstract", "12 cross-platform controls", "12/12"),
        ("abstract", "four priorities", "four author-unreported priorities"),
        ("abstract", "global source absence", "all 1,054 source-atlas identity rows"),
        ("abstract", "17/19 positive controls", "17/19"),
        ("abstract", "external genomic groups", "22 clean stk11/keap1-altered and 17 clean rb1-altered"),
        ("results", "zero exact claims", "number of new exact metabolite claims remains zero"),
        ("results", "zero patient covariance pairs", "none of the six pairwise"),
        ("results", "three of four formula rivals", "three of the four priorities"),
        ("results", "global source mass audit", "adduct for 1,050 rows"),
        ("results", "zero recorded technical confounding tests", "none passed the joint gate after bh correction"),
        ("results", "technical audit minimum q", "minimum bh q value was 0.378"),
        ("results", "objective cotinine sample split", "11 cotinine-classified smokers and 23 non-smokers"),
        ("results", "zero smoking-sensitive priorities", "none of the four priorities passed"),
        ("results", "external genomic axes all pass", "all three frozen axes passed"),
        ("results", "external genomic r2 range", "13.7% of"),
        ("results", "external redox leave-one-gene", "redox axis passed all eight omissions"),
        ("results", "external anchored axes", "removing parp1 reduced"),
        ("methods", "34 paired patients", "34 patients"),
        ("methods", "68 study injections", "68 study injections"),
        ("methods", "nine pooled QC", "nine pooled-qc"),
        ("methods", "103 independent protein pairs", "103 tumor/nat pairs"),
        ("methods", "80 pure pairs", "80 pure"),
        ("methods", "19 reverse claims", "tests 19"),
        ("methods", "16 technical tests", "16 tests total"),
        ("methods", "cotinine table source", "table s4"),
        ("methods", "cotinine adjusted sensitivity", "ols with hc3"),
        ("methods", "external genomic clean sizes", "22 versus 17 tumors"),
        ("discussion", "same formula errors", "two same-formula isomer errors"),
        ("discussion", "global source identity audit", "all 1,054 reported identity rows"),
        ("discussion", "no patient module", "none of six"),
        ("discussion", "redox exploratory 12/46", "12 of 46"),
        ("discussion", "technical audit q", "bh q value was 0.378"),
        ("discussion", "cotinine audit boundary", "none of the four effects passed"),
        ("discussion", "external tumor-only boundary", "external pathway-context heterogeneity, not replication"),
    ]
    rows = []
    for document, label, token in checks:
        found = token.lower() in texts[document]
        rows.append({"document": document, "check": label, "required_token": token, "found": found})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "numeric_consistency_audit.csv", index=False)

    forbidden = {
        "confirmed_four_metabolites": "four confirmed metabolites",
        "four_independently_replicated": "four metabolites were independently replicated",
        "increased_quinolinate_flux": "quinolinate flux increased",
        "pure_lcnec_ppp_activated": "pentose-phosphate pathway was activated in pure lcnec",
        "parp_discovered_in_lcnec": "parp metabolism was discovered in lcnec",
        "external_metabolite_replication": "metabolite directions were replicated in the external transcriptomic cohort",
    }
    forbidden_hits = []
    for label, phrase in forbidden.items():
        for document, text in texts.items():
            if phrase in text:
                forbidden_hits.append({"claim": label, "document": document, "phrase": phrase})

    report = {
        "status": "lcnec_manuscript_numeric_consistency_audit_complete",
        "formal": True,
        "checks": len(audit),
        "checks_passing": int(audit["found"].sum()),
        "forbidden_hits": forbidden_hits,
        "pass": bool(audit["found"].all() and not forbidden_hits),
        "provenance": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in DOCS.items()},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["pass"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(f"[audit_lcnec_manuscript_numeric_consistency] PASS checks={len(audit)}")


if __name__ == "__main__":
    main()
