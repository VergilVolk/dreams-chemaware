"""CPU contracts for the sealed role-3 confirmation decision."""
from __future__ import annotations

from audit_chemaware_role3_confirmation import decision


def row(name: str, recall1: float, recall3: float, mrr: float, micro: float, macro: float):
    return {
        "name": name,
        "metrics": {
            "recall1": recall1, "recall3": recall3, "mrr": mrr,
            "micro_auc": micro, "macro_auc": macro,
        },
    }


def main() -> None:
    base = row("phaseA_2pp", 0.90, 0.98, 0.94, 0.95, 0.96)
    candidate = row("sirius_selected", 0.92, 0.98, 0.95, 0.951, 0.961)
    candidate["paired_vs_phaseA_2pp"] = {
        "delta_recall1": 0.02, "delta_mrr": 0.01,
        "corrected_at_1": 30, "introduced_at_1": 10,
        "formula_cluster_bootstrap_delta_recall1_ci95": [0.005, 0.035],
    }
    report = {"formula_role": 3, "results": [base, candidate]}
    assert decision(report, "phaseA_2pp", "sirius_selected")["confirmed"] is True
    candidate["paired_vs_phaseA_2pp"]["introduced_at_1"] = 15
    failed = decision(report, "phaseA_2pp", "sirius_selected")
    assert failed["confirmed"] is False
    assert failed["gates"]["corrected_exceeds_twice_introduced"] is False
    print("PASS: ChemAware role-3 confirmation contracts", flush=True)


if __name__ == "__main__":
    main()
