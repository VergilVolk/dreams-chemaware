from __future__ import annotations

import math


def test_ten_point_requirement_rounds_up() -> None:
    assert math.ceil(0.10 * 117) == 12
