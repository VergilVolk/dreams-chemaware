"""CPU contracts for controlled SIRIUS evidence normalization."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("import_chemaware_sirius_controlled_source_ledger.py")


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(prefix: str, tree: dict[str, float], csi: dict[str, float]):
    formulas = {"f00": "C2H4O", "f01": "C3H6O"}
    structures = {
        "f00": [("AAAAAAAAAAAAAA", csi["a"]), ("CCCCCCCCCCCCCC", csi["c"])],
        "f01": [("BBBBBBBBBBBBBB", csi["b"])],
    }
    rows = []
    for suffix, formula in formulas.items():
        for ik14, score in structures[suffix]:
            rows.append({
                "mappingFeatureId": f"q0_{prefix}_{suffix}",
                "molecularFormula": formula,
                "TreeScore": tree[suffix],
                "InChIkey2D": f"{ik14}-UHFFFAOYSA-N",
                "CSI:FingerIDScore": score,
            })
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="chem_sirius_controlled_test_") as directory:
        root = Path(directory)
        panel = root / "panel"
        panel.mkdir()
        (panel / "report.json").write_text(json.dumps({
            "status": "CHEMAWARE_SIRIUS_SOURCE_PANEL_COMPLETE",
            "gates": {
                "matched_control_cardinality": True,
                "instrument_profile_cardinality": True,
                "all_global_formula_structures_fit_top_k_50_summary": True,
            },
        }), encoding="utf-8")
        write_table(panel / "formula_feature_registry.tsv", [
            {
                "manifest_query": 0, "candidate_formula": "C2H4O",
                "formula_feature_id": "q0_correct_f00",
                "intensity_rank_permuted_feature_id": "q0_intensity_f00",
                "mass_shifted_feature_id": "q0_mass_f00",
            },
            {
                "manifest_query": 0, "candidate_formula": "C3H6O",
                "formula_feature_id": "q0_correct_f01",
                "intensity_rank_permuted_feature_id": "q0_intensity_f01",
                "mass_shifted_feature_id": "q0_mass_f01",
            },
        ])
        write_table(panel / "candidate_ledger.tsv", [
            {"manifest_query": 0, "local_candidate": 0, "formula": "C2H4O", "ik14": "AAAAAAAAAAAAAA"},
            {"manifest_query": 0, "local_candidate": 1, "formula": "C3H6O", "ik14": "BBBBBBBBBBBBBB"},
            {"manifest_query": 0, "local_candidate": 2, "formula": "C2H4O", "ik14": "CCCCCCCCCCCCCC"},
        ])
        write_table(root / "correct.tsv", summary_rows(
            "correct", {"f00": 9.0, "f01": 5.0}, {"a": 8.0, "b": 4.0, "c": 3.0},
        ))
        write_table(root / "intensity.tsv", summary_rows(
            "intensity", {"f00": 4.0, "f01": 3.0}, {"a": 2.0, "b": 2.0, "c": 1.0},
        ))
        write_table(root / "mass.tsv", summary_rows(
            "mass", {"f00": 6.0, "f01": 2.0}, {"a": 5.0, "b": 1.0, "c": 2.0},
        ))
        output = root / "output"
        subprocess.run([
            sys.executable, str(SCRIPT), "--source-panel", str(panel),
            "--correct-summary", str(root / "correct.tsv"),
            "--intensity-summary", str(root / "intensity.tsv"),
            "--mass-summary", str(root / "mass.tsv"), "--output", str(output),
        ], check=True, capture_output=True, text=True)
        with (output / "candidate_scores.tsv").open(
            "r", encoding="utf-8", newline="",
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert len(rows) == 6
        keyed = {(row["source_family"], int(row["local_candidate"])): row for row in rows}
        tree = keyed[("sirius_tree_controlled", 0)]
        assert float(tree["source_score"]) == 9.0
        assert float(tree["control_a_score"]) == 4.0
        assert float(tree["control_b_score"]) == 6.0
        assert float(tree["control_c_score"]) == 5.0
        csi = keyed[("sirius_csi_controlled", 0)]
        assert float(csi["source_score"]) == 8.0
        assert float(csi["control_a_score"]) == 2.0
        assert float(csi["control_b_score"]) == 5.0
        assert float(csi["control_c_score"]) == 3.0
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["formal_triplet_mining_authorized"] is False
        assert not any(report["missing_scores"].values())
    print("PASS: ChemAware controlled SIRIUS source-ledger contracts", flush=True)


if __name__ == "__main__":
    main()
