#!/usr/bin/env python
"""Fast unit checks for the frozen B39-M2 action matrix."""
from __future__ import annotations

from freeze_bioaware_b39_m2_fixed_action_manifest import fixed_cells


def main() -> None:
    cells = fixed_cells()
    assert len(cells) == 12
    assert len({cell["cell_id"] for cell in cells}) == 12
    assert sum(cell["constructible_from_m1"] for cell in cells) == 10
    assert sum(cell["role"] == "core" for cell in cells) == 6
    assert sum(cell["role"] == "negative_control" for cell in cells) == 3
    assert all(cell["require_rhea"] for cell in cells)
    print("[test_bioaware_b39_m2_fixed_action] PASS", flush=True)


if __name__ == "__main__":
    main()
