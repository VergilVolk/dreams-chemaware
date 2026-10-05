"""Read-only scan: which candidate checkpoints still carry native Adam state.

Diagnostic only — never gates, never raises on an individual file.  For each
--checkpoint it reports file size, top-level keys, checkpoint kind,
optimizer_states count, Adam lr / weight decay, and the per-parameter step
range, so a native continuation source can be chosen on facts instead of
assumptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemaware_native_adam import adam_state_steps
from e1_checkpoint_io import checkpoint_kind, torch_load_compat


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scan_one(path: Path) -> dict[str, object]:
    entry: dict[str, object] = {"checkpoint": str(path)}
    if not path.is_file():
        entry["status"] = "MISSING"
        return entry
    entry["size_bytes"] = path.stat().st_size
    try:
        package = torch_load_compat(path, map_location="cpu")
    except Exception as error:  # noqa: BLE001 - diagnostic, report everything
        entry["status"] = "UNREADABLE"
        entry["error"] = repr(error)
        return entry
    entry["status"] = "READ"
    entry["top_level_keys"] = sorted(package.keys())
    entry["checkpoint_kind"] = checkpoint_kind(package)
    states = package.get("optimizer_states")
    if not isinstance(states, list) or not states:
        entry["optimizer_states"] = 0
        entry["adam_bearing"] = False
        return entry
    entry["optimizer_states"] = len(states)
    state = states[0]
    groups = state.get("param_groups")
    if isinstance(groups, list) and groups:
        entry["lr"] = float(groups[0].get("lr", float("nan")))
        entry["weight_decay"] = float(groups[0].get("weight_decay", float("nan")))
    try:
        steps = adam_state_steps(state)
        entry["step_min"] = min(steps.values())
        entry["step_max"] = max(steps.values())
        entry["parameter_states"] = len(steps)
        entry["adam_bearing"] = min(steps.values()) > 0
    except Exception as error:  # noqa: BLE001
        entry["adam_step_read_error"] = repr(error)
        entry["adam_bearing"] = False
    return entry


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    entries = [scan_one(path) for path in args.checkpoint]
    report = {
        "status": "GLM_CHEMAWARE_ADAM_BEARING_CHECKPOINT_SCAN_COMPLETE",
        "scanned": len(entries),
        "adam_bearing": [
            item["checkpoint"] for item in entries if item.get("adam_bearing")
        ],
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
