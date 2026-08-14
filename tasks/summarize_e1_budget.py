"""Compare raw, official and two E1 pilot evaluations."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS = {
    "R0 raw SSL": ROOT / "data/validation/e0_baseline/e0_report.json",
    "R0 official FT": ROOT / "data/validation/e1_budget/r0_official/e0_report.json",
    "E1 Pilot A": ROOT / "data/validation/e1_budget/pilot_a/e0_report.json",
    "E1 Pilot B": ROOT / "data/validation/e1_budget/pilot_b/e0_report.json",
}


def primary_metrics(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["summary_table"]
    row = next(item for item in rows if item["Protocol"].startswith("10ppm, [[M+H]+]"))
    return {
        "pooled_auc": float(row["ROC-AUC"]),
        "macro_auc": float(row["Macro-AUC"]),
        "average_precision": float(row["AP"]),
        "separation": float(row["Separation"]),
        "recall_at_1": float(row["Recall@1"]),
        "mrr": float(row["MRR"]),
    }


def gate(child: dict, parent: dict) -> dict:
    keys = ("pooled_auc", "macro_auc", "separation")
    improved = [key for key in keys if child[key] > parent[key]]
    return {"pass": len(improved) >= 2, "improved": improved, "required": 2}


def main() -> None:
    required = ("R0 official FT", "E1 Pilot B")
    missing_required = [str(REPORTS[name]) for name in required if not REPORTS[name].is_file()]
    if missing_required:
        raise SystemExit(
            "Missing required official-continuation reports:\n  "
            + "\n  ".join(missing_required)
        )
    metrics = {
        name: primary_metrics(path)
        for name, path in REPORTS.items()
        if path.is_file()
    }
    gates = {
        "pilot_b_vs_official": gate(metrics["E1 Pilot B"], metrics["R0 official FT"]),
    }
    if "E1 Pilot A" in metrics and "R0 raw SSL" in metrics:
        gates["pilot_a_vs_raw"] = gate(metrics["E1 Pilot A"], metrics["R0 raw SSL"])
    payload = {"metrics": metrics, "gates": gates}
    output_dir = ROOT / "data/validation/e1_budget"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBudget E1 comparison: primary [M+H]+ 10-ppm protocol")
    print(f"{'Run':<18} {'PooledAUC':>10} {'MacroAUC':>10} {'AP':>8} {'Sep':>8}")
    print("-" * 58)
    for name in REPORTS:
        if name not in metrics:
            continue
        value = metrics[name]
        print(
            f"{name:<18} {value['pooled_auc']:>10.4f} {value['macro_auc']:>10.4f} "
            f"{value['average_precision']:>8.4f} {value['separation']:>8.4f}"
        )
    for name, value in gates.items():
        print(f"{name}: {'PASS' if value['pass'] else 'STOP'}; improved={value['improved']}")
    print(f"Saved: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
