from __future__ import annotations

from dataclasses import asdict

from full23_tensor_set_readout_core import (
    SCHEMA, Full23TensorSetReadout, model_from_payload,
)
from train_full23_v13_cumulative_repairs import ARM_ORDER, cumulative_arm_specs


def main() -> None:
    specs = cumulative_arm_specs()
    assert tuple(specs) == ARM_ORDER
    for index, arm in enumerate(ARM_ORDER):
        assert len(specs[arm].interventions) == index + 1
        if index:
            previous = specs[ARM_ORDER[index - 1]].interventions
            assert specs[arm].interventions[:-1] == previous
    assert specs["0"].residual_penalty_space == "bounded"
    assert specs["01"].residual_penalty_space == "raw"
    assert specs["012"].bound_rule == "error_q75"
    assert specs["0123"].settings.include_anchor_context
    assert specs["01234"].settings.pair_dimension == 128
    assert specs["012345"].settings.include_signed_difference

    # Every stage changes exactly one declared design choice relative to its
    # predecessor; this prevents a cumulative label from hiding extra edits.
    fields = (
        "residual_penalty_space", "bound_rule", "settings",
    )
    for left, right in zip(ARM_ORDER[:-1], ARM_ORDER[1:]):
        changed = 0
        for field in fields:
            first, second = getattr(specs[left], field), getattr(specs[right], field)
            if field == "settings":
                first, second = asdict(first), asdict(second)
                changed += sum(first[key] != second[key] for key in first)
            else:
                changed += int(first != second)
        assert changed == 1, (left, right, changed)

    # Every architecture variant must round-trip through the same deployment
    # loader used by GNPS and the application pipeline.
    for arm in ARM_ORDER:
        model = Full23TensorSetReadout(specs[arm].settings, residual_bound=0.75)
        payload = {
            "schema": SCHEMA,
            "settings": asdict(specs[arm].settings),
            "residual_bound": 0.75,
            "state_dict": model.state_dict(),
        }
        restored = model_from_payload(payload, "cpu")
        assert restored.settings == specs[arm].settings
        assert restored.trainable_parameters == model.trainable_parameters
    print("PASS: Full23 v1.3 cumulative-repair contracts")


if __name__ == "__main__":
    main()
