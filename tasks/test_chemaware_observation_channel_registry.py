"""Contracts for strict observation-channel grouping and quarantine."""
from __future__ import annotations

from build_chemaware_observation_channel_registry import (
    candidate_for_positive_calibration, group_channels, mass_collision_groups,
)


def rule(name: str, value: float, mode: str, source: str = "CompMS2Miner (x)") -> dict:
    return {"name": name, "category": "NL", "match_type": "mass_diff", "value": value,
            "mode": mode, "source": source, "ref": "citation", "formula": "H2O"}


def main() -> None:
    positive = rule("a", 18.0100, "pos")
    alias = rule("b", 18.0110, "pos+neg")
    negative = rule("c", 19.0, "neg")
    unknown = rule("d", 20.0, "?")
    legacy = rule("e", 21.0, "pos", "baseline")
    assert candidate_for_positive_calibration(positive)
    assert not any(candidate_for_positive_calibration(item) for item in (negative, unknown, legacy))
    channels = group_channels([positive, alias, negative, unknown, legacy], 0.005)
    assert len(channels) == 1
    assert channels[0]["aliases"] == ["a", "b"]
    assert channels[0]["enabled_for_formal_training"] is False
    assert channels[0]["mechanistic_rule"] is False
    assert mass_collision_groups([positive, alias], 0.005) == [["a", "b"]]
    print("observation registry mode, alias and quarantine contracts passed")


if __name__ == "__main__":
    main()
