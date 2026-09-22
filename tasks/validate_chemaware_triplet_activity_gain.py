"""Require a material native-hinge activity gain before GPU fine-tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-active-event-ratio", type=float, default=1.25)
    parser.add_argument("--min-activation-probability-ratio", type=float, default=1.25)
    parser.add_argument("--min-mean-hinge-ratio", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    ratio = {
        "active_events": float(new["active_events"]) / max(1.0, float(old["active_events"])),
        "mean_activation_probability": float(new["mean_activation_probability"])
        / max(1e-12, float(old["mean_activation_probability"])),
        "mean_native_hinge": float(new["mean_native_hinge"])
        / max(1e-12, float(old["mean_native_hinge"])),
    }
    gates = {
        "active_event_ratio": ratio["active_events"] >= args.min_active_event_ratio,
        "activation_probability_ratio": (
            ratio["mean_activation_probability"] >= args.min_activation_probability_ratio
        ),
        "mean_hinge_ratio": ratio["mean_native_hinge"] >= args.min_mean_hinge_ratio,
    }
    report = {
        "status": (
            "CHEMAWARE_TRIPLET_ACTIVITY_GAIN_PASS"
            if all(gates.values()) else "CHEMAWARE_TRIPLET_ACTIVITY_GAIN_FAIL"
        ),
        "old_audit": str(args.old.resolve()),
        "new_audit": str(args.new.resolve()),
        "ratios": ratio,
        "required": {
            "min_active_event_ratio": args.min_active_event_ratio,
            "min_activation_probability_ratio": args.min_activation_probability_ratio,
            "min_mean_hinge_ratio": args.min_mean_hinge_ratio,
        },
        "gates": gates,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not all(gates.values()):
        raise RuntimeError(f"reference-aligned triplet activity gates failed: {gates}")


if __name__ == "__main__":
    main()
