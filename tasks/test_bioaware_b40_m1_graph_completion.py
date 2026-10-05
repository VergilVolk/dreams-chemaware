#!/usr/bin/env python
"""Fast checks for the B40-M1 frozen graph-completion matrix."""
from __future__ import annotations

from freeze_bioaware_b40_m1_graph_completion_manifest import cells


def main() -> None:
    matrix = cells()
    assert len(matrix) == 6
    assert len({item["cell_id"] for item in matrix}) == 6
    assert sum(item["role"] == "core" for item in matrix) == 2
    assert sum(item["role"] == "negative_control" for item in matrix) == 4
    print("[test_bioaware_b40_m1_graph_completion] PASS", flush=True)


if __name__ == "__main__":
    main()
