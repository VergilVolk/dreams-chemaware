from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "audit_netid_public_release.py"
SPEC = importlib.util.spec_from_file_location("audit_netid_public_release", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manual_mapping_uses_mass_rt_and_row_order_not_manual_id() -> None:
    raw = pd.DataFrame(
        {
            "groupId": [10, 40],
            "medMz": [100.0, 200.0],
            "medRt": [1.0, 2.0],
        }
    )
    output = pd.DataFrame(
        {
            "peak_id": [1, 2],
            "class": ["Metabolite", "Artifact"],
            "formula": ["C1H2", "C2H4"],
            "annotation": ["first", "second"],
        }
    )
    manual = pd.DataFrame(
        {
            "id": [999, 123],
            "medMz": [200.0, 100.0],
            "medRt": [2.0, 1.0],
            "class": ["Artifact", "Metabolite"],
            "Confidence": [True, True],
            "Ground truth": ["C2H4", "C1H2"],
        }
    )
    mapped = MODULE.map_yeast_manual_to_author_output(manual, raw, output)
    assert mapped["netid_peak_id"].tolist() == [2, 1]
    assert mapped["joint_correct"].tolist() == [True, True]


def test_manual_mapping_rejects_ambiguous_raw_mass_rt() -> None:
    raw = pd.DataFrame({"medMz": [100.0, 100.0], "medRt": [1.0, 1.0]})
    output = pd.DataFrame(
        {
            "peak_id": [1, 2],
            "class": ["Unknown", "Unknown"],
            "formula": ["Unknown", "Unknown"],
            "annotation": ["", ""],
        }
    )
    manual = pd.DataFrame(
        {
            "id": [1],
            "medMz": [100.0],
            "medRt": [1.0],
            "class": ["Unknown"],
            "Confidence": [True],
            "Ground truth": ["Unknown"],
        }
    )
    with pytest.raises(RuntimeError, match="not unique"):
        MODULE.map_yeast_manual_to_author_output(manual, raw, output)


def test_solver_audit_detects_public_cplex_boundary(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "NetID_function.R").write_text(
        "library(cplexAPI)\nRun_cplex <- function(x) x\n", encoding="utf-8"
    )
    (code / "NetID_run_script.R").write_text(
        "CplexSet <- Run_cplex(CplexSet)\n"
        "cyto_nodes <- CplexSet$ilp_nodes %>% filter(ilp_solution > 0.01)\n",
        encoding="utf-8",
    )
    result = MODULE.solver_audit(tmp_path)
    assert result["cplex_api_imported"] is True
    assert result["cplex_solver_called"] is True
    assert result["complete_pre_solution_ilp_state_bundled"] is False
    assert result["cytoscape_files_are_post_solution_only"] is True
    assert result["exact_public_solver_reproduction_ready"] is False
