from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd


MODULE_PATH = Path(__file__).with_name("audit_mtbls13729_integrated_candidate_algorithm_value.py")
SPEC = importlib.util.spec_from_file_location("integrated_candidate_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalized_ik14_and_reference_boundaries() -> None:
    assert MODULE.normalized_ik14("ABCDEFGHIJKLMN-REST") == "ABCDEFGHIJKLMN"
    assert MODULE.normalized_ik14("short") == ""
    assert MODULE.normalized_ik14(float("nan")) == ""


def test_end_to_end_reference_separation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        ledger_path = root / "ledger.csv"
        threeway = root / "threeway"
        output = root / "output"
        threeway.mkdir()
        pd.DataFrame(
            [
                {"feature_id": 1, "label": "level1", "discovery_panel": "neg_rp", "published_source_msi": "Level 1", "source_inchikey": "ABCDEFGHIJKLMN-REST"},
                {"feature_id": 2, "label": "level2", "discovery_panel": "pos_rp", "published_source_msi": "Level 2", "source_inchikey": "QRSTUVWXYZABCD-REST"},
            ]
        ).to_csv(ledger_path, index=False)
        columns = {
            "dreams_ik14": "ABCDEFGHIJKLMN",
            "p2b_ik14": "ZZZZZZZZZZZZZZ",
            "e6_fixed_v2_sw2_ik14": "ABCDEFGHIJKLMN",
        }
        pd.DataFrame([{"feature_id": 1, **columns}]).to_csv(threeway / "neg_rp__threeway_features.csv.gz", index=False)
        pd.DataFrame(
            [{"feature_id": 2, "dreams_ik14": "", "p2b_ik14": "QRSTUVWXYZABCD", "e6_fixed_v2_sw2_ik14": "QRSTUVWXYZABCD"}]
        ).to_csv(threeway / "pos_rp__threeway_features.csv.gz", index=False)
        subprocess.run(
            [sys.executable, str(MODULE_PATH), "--ledger", str(ledger_path), "--threeway-dir", str(threeway), "--output-dir", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["methods"]["official_dreams"]["level1_source_concordant"] == 1
        assert report["methods"]["official_dreams"]["level2_abstained"] == 1
        assert report["methods"]["frozen_p2b"]["level1_alternative"] == 1
        assert report["methods"]["frozen_p2b"]["level2_source_concordant"] == 1


if __name__ == "__main__":
    test_normalized_ik14_and_reference_boundaries()
    test_end_to_end_reference_separation()
    print("[test_mtbls13729_integrated_candidate_algorithm_value] PASS")
