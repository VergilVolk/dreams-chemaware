import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

from audit_g8r_p3_transition_mechanisms import transition_counts  # noqa: E402


def test_fallback_transition_accounting_is_paired():
    rows = [
        {"dreams_top1": False, "p2b_frozen_top1": True},
        {"dreams_top1": True, "p2b_frozen_top1": False},
        {"dreams_top1": True, "p2b_frozen_top1": True},
    ]
    all_fusion = transition_counts(rows, np.asarray([True, True, True]))
    selective = transition_counts(rows, np.asarray([True, False, True]))
    assert all_fusion["corrected"] == 1 and all_fusion["introduced"] == 1
    assert selective["corrected"] == 1 and selective["introduced"] == 0
    assert selective["net"] == 1
