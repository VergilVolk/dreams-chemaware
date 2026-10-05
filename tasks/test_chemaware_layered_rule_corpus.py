"""CPU contracts for the layered ChemAware chemical evidence corpus."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="chem_layered_rule_test_") as directory:
        output = Path(directory) / "corpus"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tasks/build_chemaware_layered_rule_corpus.py"),
                "--output", str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        corpus = json.loads((output / "rule_corpus.json").read_text(encoding="utf-8"))
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "CHEMAWARE_LAYERED_RULE_CORPUS_COMPLETE"
        assert report["records"] > 8_000
        assert report["legacy_core_records"] == 335
        assert report["massbank_single_spectrum_records"] == 3_151
        assert report["confirmed_structure_rules"] >= 2
        assert report["msfinder_curated_rules"] == 4_638
        assert all(report["gates"].values())
        rows = corpus["records"]
        quarantined = [row for row in rows if row["record_kind"] == "single_spectrum_observation"]
        assert len(quarantined) == 3_151
        assert all(row["admission_weight"] == 0 for row in quarantined)
        msfinder = [row for row in rows if row["source_ids"] == ["msfinder_rule_tables_v1"]]
        assert len(msfinder) == 4_638
        assert all(row["training_policy"].startswith("disabled_until") for row in msfinder)
        assert all("matched_null_required" in row["quality"] for row in rows)
        assert corpus["contracts"]["native_dreams_loss_unchanged"] is True
    print("PASS: ChemAware layered rule-corpus contracts", flush=True)


if __name__ == "__main__":
    main()
