"""Run the locked ChemAware evaluation battery for one trained checkpoint.

The default path uses only development/validation material.  The independent
confirmation residual atlas stays locked unless ``--unlock-confirmation`` is
explicitly supplied after model and epoch selection have been frozen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--triplets", type=int, default=21163)
    parser.add_argument("--negative-probe-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--panel-seed", type=int, default=20260815)
    parser.add_argument(
        "--panels", nargs="+", default=["random", "hard", "masked", "hard-masked"],
        choices=["random", "hard", "masked", "hard-masked"],
    )
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument(
        "--unlock-confirmation", action="store_true",
        help="Read the molecule-disjoint confirmation residual atlas once, after selection.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool, commands: list[list[str]]) -> None:
    commands.append(command)
    print(f"\n>>> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file() and not args.dry_run:
        raise FileNotFoundError(args.checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []

    if not args.skip_retrieval:
        run([
            PYTHON, "tasks/eval_e0_baseline.py",
            "--ckpt", str(args.checkpoint), "--fold", "val",
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--ppm-tol", "10", "--n-bootstrap", str(args.bootstrap),
            "--output-dir", str(args.output_root / "strict_10ppm_retrieval"),
        ], args.dry_run, commands)

    if not args.skip_panels:
        for panel in args.panels:
            run([
                PYTHON, "tasks/eval_causal_cpu_paired.py",
                "--checkpoint", str(args.checkpoint),
                "--device", args.device, "--batch-size", str(args.batch_size),
                "--cpu-threads", str(args.cpu_threads),
                "--n-bootstrap", str(args.bootstrap),
                "--panel", panel, "--triplets", str(args.triplets),
                "--negative-probe-size", str(args.negative_probe_size),
                "--panel-seed", str(args.panel_seed),
                "--output", str(args.output_root / f"paired_{panel}.json"),
            ], args.dry_run, commands)

    if not args.skip_structure:
        command = [
            PYTHON, "tasks/eval_causal_chemmask_structure.py",
            "--checkpoint", str(args.checkpoint),
            "--output-root", str(args.output_root / "structure_continuity"),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--n-bootstrap", str(args.bootstrap),
        ]
        if args.unlock_confirmation:
            command.append("--include-confirmation")
        run(command, args.dry_run, commands)

    manifest = {
        "status": "planned" if args.dry_run else "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "strict_ppm": 10,
            "triplet_anchors": args.triplets,
            "panel_seed": args.panel_seed,
            "negative_probe_size": args.negative_probe_size,
            "confirmation_unlocked": args.unlock_confirmation,
            "confirmation_policy": (
                "read once after model/epoch selection was frozen"
                if args.unlock_confirmation else "locked; discovery-only structure audit"
            ),
        },
        "commands": commands,
    }
    (args.output_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nEvaluation manifest: {args.output_root / 'evaluation_manifest.json'}")


if __name__ == "__main__":
    main()
