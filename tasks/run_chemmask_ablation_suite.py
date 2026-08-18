"""Launch the three locked strict-10ppm head-only ablations.

Every variant starts from the same official DreaMS embedding head.  This suite
is for causal attribution; it must not warm-start from the earlier
counterfactual checkpoint used by the performance-oriented continuation run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

VARIANTS = {
    "identity": {"hard": 0.0, "mask": 0.0, "probe": 1},
    "identity_hard": {"hard": 0.5, "mask": 0.0, "probe": 32},
    "identity_hard_mask": {"hard": 0.5, "mask": 0.3, "probe": 32},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/e1/strict_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260821, 20260822, 20260823])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every-batches", type=int, default=500)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n>>> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    planned = []
    for seed in args.seeds:
        for name in args.variants:
            values = VARIANTS[name]
            output = args.output_root / name
            command = [
                PYTHON, "tasks/train_causal_chemmask_head.py",
                "--output-dir", str(output), "--seed", str(seed),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--grad-accum", str(args.grad_accum), "--num-workers", str(args.workers),
                "--lr", "3e-5", "--weight-decay", "1e-4",
                "--margin", "0.05", "--lambda-preserve", "5.0",
                "--hard-negative-prob", str(values["hard"]),
                "--negative-probe-size", str(values["probe"]),
                "--identity-mask-prob", str(values["mask"]),
                "--identity-mask-max-fraction", "0.3",
                "--identity-mask-max-peaks", "12",
                "--val-triplets", "21163", "--max-train-batches", "0",
                "--max-val-batches", "0", "--sequential-anchors",
                "--checkpoint-every-batches", str(args.checkpoint_every_batches),
                "--device", args.device,
            ]
            command.append("--amp" if args.device.startswith("cuda") else "--no-amp")
            planned.append({"seed": seed, "variant": name, "command": command})
            run(command, args.dry_run)
            checkpoint = output / f"seed_{seed}" / "best_causal_head.pt"
            if args.evaluate:
                run([
                    PYTHON, "tasks/run_chemmask_locked_evaluation.py",
                    "--checkpoint", str(checkpoint),
                    "--output-root", str(args.output_root / "evaluation" / name / f"seed_{seed}"),
                    "--device", args.device, "--batch-size", str(args.batch_size),
                ], args.dry_run)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "planned" if args.dry_run else "complete",
        "scientific_question": {
            "identity": "gain from strict identity supervision alone",
            "identity_hard": "incremental gain from shared-major-peak hard-negative mining",
            "identity_hard_mask": "incremental gain from condition-specific peak masking",
        },
        "common_initialization": "official DreaMS embedding head",
        "seeds": args.seeds,
        "runs": planned,
    }
    (args.output_root / "ablation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
