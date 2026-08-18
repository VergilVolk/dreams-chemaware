"""Stepwise CPU/GPU launcher for causal ChemMask head fine-tuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=[
            "check", "cpu-check", "cpu-smoke", "cpu-pilot", "cpu-eval", "cpu-full",
            "pilot", "eval", "structure", "formal", "confirm",
        ], required=True
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot-epochs", type=int, default=3)
    parser.add_argument("--pilot-train-batches", type=int, default=1000)
    parser.add_argument("--pilot-val-batches", type=int, default=250)
    parser.add_argument("--formal-epochs", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cpu-pilot-epochs", type=int, default=2)
    parser.add_argument("--cpu-pilot-train-batches", type=int, default=100)
    parser.add_argument("--cpu-pilot-val-batches", type=int, default=25)
    parser.add_argument("--cpu-full-epochs", type=int, default=3)
    parser.add_argument(
        "--cpu-eval-panel", choices=["random", "hard", "masked", "hard-masked"],
        default="random",
    )
    parser.add_argument("--cpu-eval-triplets", type=int, default=100)
    parser.add_argument("--cpu-eval-panel-seed", type=int, default=20260815)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n>>> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def train_command(args: argparse.Namespace, formal: bool) -> tuple[list[str], Path]:
    label = "formal" if formal else "pilot"
    output = ROOT / f"data/e1/causal_chemmask_{label}"
    epochs = args.formal_epochs if formal else args.pilot_epochs
    train_batches = 0 if formal else args.pilot_train_batches
    val_batches = 625 if formal else args.pilot_val_batches
    command = [
        PYTHON, "tasks/train_causal_chemmask_head.py",
        "--output-dir", str(output),
        "--seed", str(args.seed),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", "2",
        "--num-workers", str(args.workers),
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--margin", "0.1",
        "--lambda-preserve", "0.1",
        "--hard-negative-prob", "0.5",
        "--negative-probe-size", "8",
        "--identity-mask-prob", "0.3",
        "--identity-mask-max-fraction", "0.3",
        "--identity-mask-max-peaks", "12",
        "--val-triplets", str(val_batches * args.batch_size),
        "--max-train-batches", str(train_batches),
        "--max-val-batches", str(val_batches),
        "--device", "cuda", "--amp",
    ]
    return command, output / f"seed_{args.seed}/best_causal_head.pt"


def cpu_train_command(args: argparse.Namespace, smoke: bool) -> tuple[list[str], Path]:
    label = "cpu_smoke" if smoke else "cpu_pilot"
    output = ROOT / f"data/e1/causal_chemmask_{label}"
    batch_size = 2 if smoke else min(args.batch_size, 4)
    epochs = 1 if smoke else args.cpu_pilot_epochs
    train_batches = 8 if smoke else args.cpu_pilot_train_batches
    val_batches = 4 if smoke else args.cpu_pilot_val_batches
    command = [
        PYTHON, "tasks/train_causal_chemmask_head.py",
        "--output-dir", str(output),
        "--seed", str(args.seed),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", "1" if smoke else "2",
        "--num-workers", "0",
        "--cpu-threads", str(args.cpu_threads),
        "--lr", "1e-4", "--weight-decay", "1e-4",
        "--margin", "0.1", "--lambda-preserve", "0.1",
        "--hard-negative-prob", "0.5",
        "--negative-probe-size", "4" if smoke else "8",
        "--identity-mask-prob", "0.3",
        "--identity-mask-max-fraction", "0.3",
        "--identity-mask-max-peaks", "12",
        "--val-triplets", str(val_batches * batch_size),
        "--max-train-batches", str(train_batches),
        "--max-val-batches", str(val_batches),
        "--device", "cpu", "--no-amp",
    ]
    return command, output / f"seed_{args.seed}/best_causal_head.pt"


def cpu_full_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    """Continue the validated counterfactual head on every strict-10ppm anchor."""
    output = ROOT / "data/e1/strict_counterfactual_full_cpu"
    previous_best = (
        ROOT / "data/e1/counterfactual_formal/head/seed_20260813/best_counterfactual.pt"
    )
    command = [
        PYTHON, "tasks/train_causal_chemmask_head.py",
        "--output-dir", str(output),
        "--seed", str(args.seed),
        "--epochs", str(args.cpu_full_epochs),
        "--batch-size", str(min(args.batch_size, 4)),
        "--grad-accum", "4",
        "--num-workers", "0",
        "--cpu-threads", str(args.cpu_threads),
        "--initial-head-ckpt", str(previous_best),
        "--lr", "3e-5", "--weight-decay", "1e-4",
        "--margin", "0.05", "--lambda-preserve", "5.0",
        "--hard-negative-prob", "0.5",
        "--negative-probe-size", "32",
        "--identity-mask-prob", "0.3",
        "--identity-mask-max-fraction", "0.3",
        "--identity-mask-max-peaks", "12",
        "--val-triplets", "21163",
        "--max-train-batches", "0", "--max-val-batches", "0",
        "--sequential-anchors",
        "--checkpoint-every-batches", "1000",
        "--device", "cpu", "--no-amp",
    ]
    if args.resume is not None:
        command.extend(["--resume", str(args.resume)])
    return command, output / f"seed_{args.seed}/best_causal_head.pt"


def evaluate(checkpoint: Path, args: argparse.Namespace, label: str) -> None:
    run([
        PYTHON, "tasks/eval_e0_baseline.py",
        "--data", str(ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"),
        "--ckpt", str(checkpoint),
        "--fold", "val", "--device", "cuda",
        "--batch-size", "32", "--n-bootstrap", "1000",
        "--output-dir", str(ROOT / f"data/validation/{label}"),
    ], args.dry_run)


def evaluate_structure(
    checkpoint: Path, args: argparse.Namespace, label: str, include_confirmation: bool = False
) -> None:
    command = [
        PYTHON, "tasks/eval_causal_chemmask_structure.py",
        "--checkpoint", str(checkpoint),
        "--output-root", str(ROOT / f"data/validation/{label}_structure"),
        "--device", "cuda", "--batch-size", "32", "--n-bootstrap", "1000",
    ]
    if include_confirmation:
        command.append("--include-confirmation")
    run(command, args.dry_run)


def main() -> None:
    args = parse_args()
    if args.stage == "check":
        run([PYTHON, "tasks/check_e1_budget.py"], args.dry_run)
        return
    if args.stage == "cpu-check":
        run([PYTHON, "tasks/check_e1_budget.py", "--allow-cpu"], args.dry_run)
        return
    if args.stage in {"cpu-smoke", "cpu-pilot"}:
        command, checkpoint = cpu_train_command(args, smoke=args.stage == "cpu-smoke")
        run(command, args.dry_run)
        if not args.dry_run and not checkpoint.is_file():
            raise SystemExit(f"Expected checkpoint was not created: {checkpoint}")
        if not args.dry_run:
            print("\nCPU run complete. This stage validates execution and direction only; ")
            print("full retrieval/structure evaluation remains a CUDA-stage task.")
            print(f"Checkpoint: {checkpoint}")
        return
    if args.stage == "cpu-full":
        command, checkpoint = cpu_full_command(args)
        run(command, args.dry_run)
        if not args.dry_run:
            print("\nFull strict-10ppm CPU training complete.")
            print(f"Checkpoint: {checkpoint}")
        return
    if args.stage == "cpu-eval":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --stage cpu-eval")
        run([
            PYTHON, "tasks/eval_causal_cpu_paired.py",
            "--checkpoint", str(args.checkpoint),
            "--device", "cpu", "--batch-size", "4",
            "--cpu-threads", str(args.cpu_threads), "--n-bootstrap", "2000",
            "--panel", args.cpu_eval_panel,
            "--triplets", str(args.cpu_eval_triplets),
            "--negative-probe-size", "32",
            "--panel-seed", str(args.cpu_eval_panel_seed),
        ], args.dry_run)
        return
    if args.stage in {"eval", "structure", "confirm"}:
        if args.checkpoint is None:
            raise SystemExit(f"--checkpoint is required for --stage {args.stage}")
        label = f"causal_chemmask_eval_seed_{args.seed}"
        if args.stage == "eval":
            evaluate(args.checkpoint, args, label)
        else:
            evaluate_structure(
                args.checkpoint, args, label, include_confirmation=args.stage == "confirm"
            )
        return
    formal = args.stage == "formal"
    command, checkpoint = train_command(args, formal)
    run(command, args.dry_run)
    if args.dry_run or checkpoint.is_file():
        evaluate(checkpoint, args, f"causal_chemmask_{'formal' if formal else 'pilot'}_seed_{args.seed}")
        evaluate_structure(
            checkpoint, args,
            f"causal_chemmask_{'formal' if formal else 'pilot'}_seed_{args.seed}",
            include_confirmation=False,
        )
    else:
        raise SystemExit(f"Expected checkpoint was not created: {checkpoint}")


if __name__ == "__main__":
    main()
