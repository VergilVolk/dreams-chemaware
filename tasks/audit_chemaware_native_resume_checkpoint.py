"""Fail-closed audit for a native ChemAware Adam continuation checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chemaware_native_adam import adam_state_steps
from e1_checkpoint_io import checkpoint_kind, torch_load_compat


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-lr", type=float, default=5e-6)
    parser.add_argument("--expected-weight-decay", type=float, default=0.0)
    parser.add_argument("--minimum-step", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(
    checkpoint: Path, expected_lr: float, expected_weight_decay: float,
    minimum_step: int,
) -> dict[str, object]:
    package = torch_load_compat(checkpoint, map_location="cpu")
    states = package.get("optimizer_states")
    if not isinstance(states, list) or len(states) != 1:
        raise RuntimeError(
            "protected Phase-A checkpoint lacks exactly one optimizer state; "
            "a native Adam continuation is not scientifically valid"
        )
    state = states[0]
    groups = state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise RuntimeError("Phase-A optimizer does not have exactly one parameter group")
    group = groups[0]
    lr = float(group.get("lr", float("nan")))
    weight_decay = float(group.get("weight_decay", float("nan")))
    if abs(lr - float(expected_lr)) > 1e-15:
        raise RuntimeError(f"Phase-A Adam LR drift: expected={expected_lr} observed={lr}")
    if abs(weight_decay - float(expected_weight_decay)) > 1e-15:
        raise RuntimeError(
            "Phase-A Adam weight decay drift: "
            f"expected={expected_weight_decay} observed={weight_decay}"
        )
    steps = adam_state_steps(state)
    if min(steps.values()) < int(minimum_step):
        raise RuntimeError(
            f"Phase-A Adam step is below {minimum_step}: min={min(steps.values())}"
        )
    result = {
        "status": "CHEMAWARE_NATIVE_ADAM_RESUME_CHECKPOINT_PASS",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_kind": checkpoint_kind(package),
        "optimizer_states": 1,
        "parameter_states": len(steps),
        "step_min": min(steps.values()),
        "step_max": max(steps.values()),
        "lr": lr,
        "weight_decay": weight_decay,
        "continuation_authorized": True,
    }
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_checkpoint(
        args.checkpoint, args.expected_lr, args.expected_weight_decay,
        args.minimum_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
