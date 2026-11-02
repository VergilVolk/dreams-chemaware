#!/usr/bin/env python3
"""Contracts for winner-anchored Full23 v1.3.1 direct repairs."""
from dataclasses import asdict
from pathlib import Path

from train_full23_v131_direct_repairs import ARM_ORDER, direct_arm_specs


def main() -> None:
    assert ARM_ORDER == ("013", "014", "015")
    specs = direct_arm_specs()
    assert tuple(specs) == ARM_ORDER
    base = specs["013"].settings

    assert base.include_anchor_context is True
    assert base.pair_dimension == 64
    assert base.include_signed_difference is False

    assert specs["014"].settings.include_anchor_context is False
    assert specs["014"].settings.pair_dimension == 128
    assert specs["014"].settings.include_signed_difference is False

    assert specs["015"].settings.include_anchor_context is False
    assert specs["015"].settings.pair_dimension == 64
    assert specs["015"].settings.include_signed_difference is True

    # All three arms retain the selected 01 representation contract and differ
    # from it by exactly one field.
    baseline = asdict(specs["013"].settings)
    baseline["include_anchor_context"] = False
    for arm in ARM_ORDER:
        settings = asdict(specs[arm].settings)
        changed = {key for key in settings if settings[key] != baseline[key]}
        assert len(changed) == 1, (arm, changed)

    trainer = Path(__file__).with_name("train_full23_v131_direct_repairs.py").read_text(
        encoding="utf-8"
    )
    assert 'bounds["legacy_median"]' in trainer
    assert 'residual_penalty_space="raw"' in trainer
    assert "error_q75" not in trainer
    assert '"warm_start": False' in trainer
    print("PASS: Full23 v1.3.1 direct-repair contracts")


if __name__ == "__main__":
    main()
