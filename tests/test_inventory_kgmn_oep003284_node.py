from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks" / "inventory_kgmn_oep003284_node.py"
SPEC = importlib.util.spec_from_file_location("oep_inventory", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_rows() -> list[dict]:
    rows = []
    index = 0
    for group in (1, 2, 4):
        for polarity in ("pos", "neg"):
            for repeat in range(1, 5):
                index += 1
                rows.append(
                    {
                        "name": f"g{group}_46std_{polarity}_{repeat}.mzXML",
                        "datNo": f"OED{index:08d}",
                        "runNo": MODULE.RUN_NUMBER,
                        "fileSize": 201 * 1024 * 1024,
                        "md5": f"{index:032x}",
                        "security": "Public",
                        "accessible": True,
                    }
                )
    return rows


def test_exact_public_layout_passes() -> None:
    result = MODULE.validate_remote(make_rows())
    assert result["pass"] is True
    assert result["cells"] == {
        "g1_pos": 4,
        "g1_neg": 4,
        "g2_pos": 4,
        "g2_neg": 4,
        "g4_pos": 4,
        "g4_neg": 4,
    }


def test_missing_raw_file_fails() -> None:
    result = MODULE.validate_remote(make_rows()[:-1])
    assert result["pass"] is False
    assert any("expected 24" in value for value in result["problems"])


def test_node_sftp_sharding_is_exact() -> None:
    assert MODULE.sftp_run_path("OER00253320") == (
        "/Public/byRun/OER00/OER0025/OER002533/OER00253320"
    )
