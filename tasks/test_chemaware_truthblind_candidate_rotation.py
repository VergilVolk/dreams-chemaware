from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_orthogonal_rule_policy_core import (  # noqa: E402
    rotate_candidate_contrast_truthblind,
)


def main() -> None:
    contrast = np.asarray([
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[4.0, 40.0], [9.0, 90.0], [9.0, 90.0]],
    ], dtype=np.float32)
    valid = np.asarray([
        [True, True, True],
        [True, False, False],
    ])
    rotated, source = rotate_candidate_contrast_truthblind(contrast, valid)
    np.testing.assert_array_equal(
        rotated[0], contrast[0, [2, 0, 1]],
    )
    np.testing.assert_array_equal(rotated[1], np.zeros_like(rotated[1]))
    np.testing.assert_array_equal(source[0], np.asarray([2, 0, 1]))
    np.testing.assert_array_equal(source[1], np.asarray([-1, -1, -1]))

    core = (ROOT / "tasks/chemaware_orthogonal_rule_policy_core.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(core)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "rotate_candidate_contrast_truthblind"
    )
    identifiers = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    forbidden = ("baseline_rank", "labels", "truth", "identity", "formula", "outcome")
    assert not (set(forbidden) & identifiers)
    print("PASS: ChemAware truth-blind candidate-rotation contracts")


if __name__ == "__main__":
    main()
