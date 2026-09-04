#!/usr/bin/env python3
"""Static fail-closed claim audit for the MTBLS13729 biology manuscript set."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "results": ROOT / "docs/MTBLS13729_BIOLOGY_MANUSCRIPT_RESULTS_V2_20260830.md",
    "decision": ROOT / "docs/MTBLS13729_BIOLOGY_PUBLICATION_DECISION_20260831.md",
    "composition": ROOT / "docs/GSE178341_EPITHELIAL_COMPOSITION_DIAGNOSTIC_RESULT_20260831.md",
    "raw_umi": ROOT / "docs/GSE178341_RAW_UMI_MUCINOUS_SIALIC_AUDIT_20260831.md",
    "proteomics": ROOT / "docs/MTBLS13729_INDEPENDENT_MUCINOUS_PROTEOMICS_AUDIT_20260831.md",
}
REQUIRED = {
    "results": ["结果 5c", "post-result", "AGR2仍为", "25门完成度总账", "不使用 `flux`"],
    "decision": ["上皮组成诊断后的最终生物学措辞", "Package A", "不是因果中介分析"],
    "composition": ["post-result diagnostic", "not a new confirmatory endpoint", "causal mediation or"],
    "raw_umi": ["does **not** support a uniformly activated", "They do not establish Neu5Ac biochemical source"],
    "proteomics": ["does **not** confirm", "cannot support the claims that the pathway is confirmed"],
}
FORBIDDEN_EXACT = [
    "Neu5Ac通量升高已得到证明",
    "NXPE1驱动Neu5Ac",
    "NEU1/NEU3驱动free Neu5Ac",
    "independent Neu5Ac abundance replication was confirmed",
    "therapeutic target established by this study",
]


def main() -> None:
    texts = {}
    for name, path in FILES.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        texts[name] = text
        missing = [phrase for phrase in REQUIRED[name] if phrase not in text]
        if missing:
            raise RuntimeError(f"{name} missing required claim-boundary phrases: {missing}")
    joined = "\n".join(texts.values())
    present = [phrase for phrase in FORBIDDEN_EXACT if phrase in joined]
    if present:
        raise RuntimeError(f"forbidden positive causal claims present: {present}")
    print("[validate_mtbls13729_biology_manuscript_claims_v1] PASS files=5")


if __name__ == "__main__":
    main()
