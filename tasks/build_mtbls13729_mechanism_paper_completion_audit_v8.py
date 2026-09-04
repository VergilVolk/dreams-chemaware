"""Add the phenotype-blind O-acetyl-Neu5Ac-like negative-result gate to v7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v7_final"
OAC = ROOT / "data/mtbls13729/oacetyl_neu5ac_like_v2/report.json"
AUDIT_DOC = ROOT / "docs/MTBLS13729_OACETYL_NEU5AC_LIKE_AUDIT_20260831.md"
FIGURE = ROOT / "data/mtbls13729/oacetyl_neu5ac_like_figure_v1/oacetyl_neu5ac_like_audit.png"
OUT = ROOT / "data/mtbls13729/mechanism_paper_completion_audit_v8_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    required = [
        V7 / "mechanism_paper_completion_audit_v7.csv",
        V7 / "report.json",
        OAC,
        AUDIT_DOC,
        FIGURE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    oac = json.loads(OAC.read_text(encoding="utf-8"))
    if oac.get("status") != "mtbls13729_oacetyl_neu5ac_like_audit_complete":
        raise RuntimeError("O-acetyl-Neu5Ac-like audit is incomplete")
    features = oac["primary_rmu_results"]
    if len(features) != 2:
        raise RuntimeError("expected exactly two frozen phenotype-blind RT features")
    if any(row["complete_exact_sign_flip_bh_q"] < 0.10 for row in features):
        raise RuntimeError("O-acetyl-Neu5Ac-like abundance result is no longer negative")
    if any(abs(row["prevalence"] * 60 - row["samples"]) > 1e-9 for row in features):
        raise RuntimeError("sample-prevalence denominator mismatch")

    ledger = pd.read_csv(V7 / "mechanism_paper_completion_audit_v7.csv")
    if len(ledger) != 21 or set(ledger.gate_id) != {f"G{i:02d}" for i in range(1, 22)}:
        raise RuntimeError("unexpected v7 completion ledger")
    addition = pd.DataFrame([{
        "gate_id": "G22",
        "domain": "phenotype_blind_mono_oacetyl_neu5ac_like_pool",
        "status": "NEGATIVE_RESULT",
        "evidence": (
            "Negative-HILIC exact-mass discovery froze two independent m/z 350.109269 RT features "
            "(50/60 and 54/60 sample support; 47 and 56 RT-resolved MS2 spectra). Both had strong "
            "m/z 87 motifs, but neither showed a reproducible Rmu paired increase (BH q 0.930) or "
            "patient-level coupling to Level-1 free Neu5Ac."
        ),
        "claim_enabled": (
            "The expanded free Neu5Ac pool is not accompanied by a reproducible increase in either "
            "of the two phenotype-blind mono-O-acetyl-Neu5Ac-like exact-mass features."
        ),
        "claim_forbidden": (
            "Exact mass and m/z 87 do not identify an O-acetyl position. This result cannot exclude "
            "glycan-bound or cell-specific O-acetylation, other isomers, or NXPE1/CASD1/SIAE activity."
        ),
        "next_action": (
            "Use 4-O- and 9-O-acetyl-Neu5Ac standards together if identity is pursued; otherwise "
            "prioritise linkage-aware O-glycomics or MUC2 glycopeptides."
        ),
        "priority": "P2",
    }])
    ledger = pd.concat([ledger, addition], ignore_index=True)
    if ledger.gate_id.duplicated().any() or len(ledger) != 22:
        raise RuntimeError("v8 gate construction failed")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_path = OUT / "mechanism_paper_completion_audit_v8.csv"
    ledger.to_csv(ledger_path, index=False)
    report = json.loads((V7 / "report.json").read_text(encoding="utf-8"))
    negative_results = list(report.get("negative_results_with_value", []))
    negative_results.append(
        "two phenotype-blind mono-O-acetyl-Neu5Ac-like exact-mass features do not increase in Rmu or track free Neu5Ac"
    )
    report.update({
        "status": "mtbls13729_mechanism_paper_completion_audit_v8_complete",
        "gates": 22,
        "status_counts": {str(k): int(v) for k, v in ledger.status.value_counts().items()},
        "mono_oacetyl_neu5ac_like_pool": {
            "status": "NEGATIVE_RESULT",
            "phenotype_blind_rt_features": 2,
            "feature_rt_sec": [row["median_rt_sec"] for row in features],
            "feature_sample_support": [row["samples"] for row in features],
            "rt_resolved_ms2_spectra": [row["ms2_spectra"] for row in features],
            "rmu_complete_pair_bh_q": [row["complete_exact_sign_flip_bh_q"] for row in features],
            "positional_isomer_identity_resolved": False,
            "glycan_bound_oacetylation_measured": False,
        },
        "negative_results_with_value": negative_results,
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "claim_limit": (
            "The current cohort supports free-pool/donor decoupling and now rejects a simple bulk "
            "mono-O-acetyl-Neu5Ac-like pool increase. It does not identify the biochemical source, "
            "O-acetyl positional isomer, glycan carrier, spatial destination, or flux."
        ),
    })
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# MTBLS13729 mechanism-paper completion audit v8",
        "",
        f"**Primary phenomenon:** {report['primary_publishable_phenomenon']}",
        "",
        "| Gate | Domain | Status | Evidence | Next action |",
        "|---|---|---|---|---|",
    ]
    for row in ledger.to_dict("records"):
        lines.append(
            f"| {row['gate_id']} | {row['domain']} | {row['status']} | "
            f"{row['evidence']} | {row['next_action']} |"
        )
    lines.extend(["", "## Claim boundary", "", report["claim_limit"], ""])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
