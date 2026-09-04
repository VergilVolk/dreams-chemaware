"""Freeze manuscript roles and validation priorities for the 18-candidate ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv"
DELTA = ROOT / "data/mtbls13729/original_paper_delta_v2/candidate_original_paper_delta_v2.csv"
OUT = ROOT / "data/mtbls13729/manuscript_evidence_matrix_v2"

PRIMARY = {345, 374, 703, 1597, 3019, 1717, 3222}
VALIDATION_PRIORITY = {
    703: "P0 Neu5Ac same-method standard, spike-in and glycan-linkage readout",
    1717: "P0 N1,N8-diacetylspermidine positional-isomer standard panel",
    1597: "P1 methylguanosine positional-isomer standard panel",
    3019: "P1 dimethylguanosine positional-isomer standard panel",
    3222: "P1 C20:4/C16:0/C18:0/C18:1 acylcarnitine standard panel",
    345: "P2 low incremental value: already source Level-1 plus strong orthogonal bridge",
    374: "P2 low incremental value: already source Level-1 plus strong orthogonal bridge",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not LEDGER.is_file() or not DELTA.is_file():
        raise FileNotFoundError("run integrated ledger v2 and original-paper delta v2 first")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(LEDGER)
    delta = pd.read_csv(DELTA)[["feature_id", "delta_type", "novelty_interpretation"]]
    frame = ledger.merge(delta, on="feature_id", validate="one_to_one")

    roles = []
    external_levels = []
    for item in frame.itertuples(index=False):
        feature = int(item.feature_id)
        if item.manuscript_evidence_tier == "C_downgraded_or_control":
            role = "negative_control_main_or_extended"
        elif feature in PRIMARY:
            role = "primary_figure_anchor"
        elif item.manuscript_evidence_tier.startswith("A_"):
            role = "module_support_extended_data"
        else:
            role = "family_support_extended_data"
        roles.append(role)

        if feature in {345, 374, 703}:
            external = "same-cohort orthogonal identity recovery; not independent replication"
        elif feature in {1597, 3019}:
            external = "independent modified-guanosine context is heterogeneous/adversarial"
        elif feature == 1717:
            external = "independent pathway context; exact metabolite replication absent"
        elif feature == 3222:
            external = "independent long-chain lipid context; exact acylcarnitine replication absent"
        else:
            external = "supporting node; external evidence not used as exact identity replication"
        external_levels.append(external)

    frame.insert(1, "manuscript_role", roles)
    frame["external_evidence_level"] = external_levels
    frame["validation_priority"] = [
        VALIDATION_PRIORITY.get(int(feature), "not prioritized before primary anchors")
        for feature in frame.feature_id
    ]
    frame["identity_claim_in_manuscript"] = frame.apply(
        lambda item: (
            item.source_name
            if item.manuscript_evidence_tier in {"A_source_identity_remap", "A_source_identity_orthogonal_recovery"}
            else item.defensible_identity
        ),
        axis=1,
    )
    frame["forbidden_interpretation"] = (
        "do not infer flux, enzyme activity, cellular source, subtype specificity or causality from static abundance"
    )
    frame.to_csv(OUT / "candidate_manuscript_evidence_matrix_v2.csv", index=False)

    report = {
        "status": "mtbls13729_manuscript_evidence_matrix_v2_complete",
        "formal": False,
        "candidates": int(len(frame)),
        "role_counts": {str(k): int(v) for k, v in frame.manuscript_role.value_counts().items()},
        "primary_figure_features": sorted(int(x) for x in frame.loc[frame.manuscript_role == "primary_figure_anchor", "feature_id"]),
        "negative_control_features": sorted(int(x) for x in frame.loc[frame.manuscript_role == "negative_control_main_or_extended", "feature_id"]),
        "validation_order": [703, 1717, 1597, 3019, 3222, 345, 374],
        "selection_rule": (
            "Primary anchors require either source-anchored orthogonal recovery or a strong source-table-absent "
            "family with raw peak-resolved MS2 and a central biological role. Other nodes support modules."
        ),
        "claim_limit": (
            "Manuscript role is not evidence of correctness. Same-cohort orthogonal recovery is not independent "
            "replication; family candidates remain unresolved without standards."
        ),
        "provenance": {"ledger_sha256": sha256(LEDGER), "delta_sha256": sha256(DELTA)},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
