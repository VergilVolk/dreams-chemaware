"""CPU contracts for independent multi-control source qualification."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from qualify_chemaware_candidate_source_ledger import formula_fold


SCRIPT = Path(__file__).with_name("qualify_chemaware_candidate_source_ledger.py")


def main() -> None:
    formulas = []
    index = 1
    while len(formulas) < 20:
        formula = f"C{index}H{2 * index + 2}"
        if formula_fold(formula, 20260928, 5) in {3, 4}:
            formulas.append(formula)
        index += 1
    with tempfile.TemporaryDirectory(prefix="chem_source_qualification_test_") as directory:
        root = Path(directory)
        ledger = root / "ledger"
        ledger.mkdir()
        (ledger / "report.json").write_text(json.dumps({
            "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_UNQUALIFIED",
            "truth_fields_exported": False,
            "source_families": {
                "three_control_family": {
                    "scope": "all_candidates", "larger_is_better": True,
                    "confidence_tier": "A_test",
                    "matched_controls": ["intensity", "mass", "candidate"],
                    "specificity_gate_passed": False,
                },
            },
        }), encoding="utf-8")
        fields = [
            "manifest_query", "local_candidate", "ik14", "formula",
            "source_family", "scope", "source_score", "control_a_score",
            "control_b_score", "control_c_score", "controls_available",
        ]
        with (ledger / "candidate_scores.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for query, formula in enumerate(formulas):
                for candidate in (0, 1):
                    writer.writerow({
                        "manifest_query": query, "local_candidate": candidate,
                        "ik14": f"{query:013d}{candidate}", "formula": formula,
                        "source_family": "three_control_family",
                        "scope": "all_candidates",
                        "source_score": 2.0 if candidate == 0 else 0.0,
                        "control_a_score": 0.0 if candidate == 0 else 1.0,
                        "control_b_score": -1.0 if candidate == 0 else 0.5,
                        "control_c_score": 0.2 if candidate == 0 else 0.8,
                        "controls_available": 1,
                    })
        manifest = root / "manifest.npz"
        np.savez_compressed(
            manifest,
            query_formula=np.asarray(formulas),
            query_ptr=np.arange(0, 2 * len(formulas) + 1, 2),
            molecule_label=np.tile(np.asarray([True, False]), len(formulas)),
        )
        output = root / "qualified"
        subprocess.run([
            sys.executable, str(SCRIPT), "--ledger", str(ledger),
            "--manifest", str(manifest), "--output", str(output),
            "--minimum-applicable-queries", "20", "--bootstrap-draws", "1000",
        ], check=True, capture_output=True, text=True)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        qualification = report["specificity_qualification"]
        assert qualification["admitted_families"] == ["three_control_family"]
        assert qualification["control_columns"] == [
            "control_a_score", "control_b_score", "control_c_score",
        ]
        controls = qualification["family_results"]["three_control_family"]["controls"]
        assert len(controls) == 3 and all(body["passed"] for body in controls.values())
        assert report["formal_triplet_mining_authorized"] is True
        assert report["source_families"]["three_control_family"]["specificity_gate_passed"] is True
        assert report["unqualified_source_families"] == {}
    print("PASS: ChemAware independent multi-control qualification contracts", flush=True)


if __name__ == "__main__":
    main()
