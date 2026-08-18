"""Anaconda-friendly staged runner for counterfactual DreaMS training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUNS = ROOT / "data/e1/counterfactual_formal"
EVALS = ROOT / "data/validation/counterfactual_formal"


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n>>> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"Stage stopped because the command above returned exit code {error.returncode}. "
                "Read the FAIL item above, fix it, and rerun the same stage."
            ) from None


def checkpoint(stage: str, seed: int) -> Path:
    return RUNS / stage / f"seed_{seed}" / "best_counterfactual.pt"


def train(stage: str, resume: Path | None, args: argparse.Namespace) -> Path:
    command = [
        PYTHON, "tasks/train_counterfactual_dreams.py", "--stage", stage,
        "--output-dir", str(RUNS), "--seed", str(args.seed),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum), "--num-workers", str(args.workers),
        "--head-lr", str(args.head_lr), "--backbone-lr", str(args.backbone_lr),
        "--counterfactual-weight", str(args.counterfactual_weight),
        "--random-consistency-weight", str(args.random_consistency_weight),
        "--preserve-weight", str(args.preserve_weight), "--patience", str(args.patience),
        "--device", args.device,
    ]
    command.append("--amp" if args.amp else "--no-amp")
    if args.smoke:
        command += ["--max-train-batches", "20", "--max-val-batches", "20"]
    if resume is not None:
        command += ["--resume", str(resume)]
    run(command, args.dry_run)
    return checkpoint(stage, args.seed)


def evaluate(stage: str, ckpt: Path, args: argparse.Namespace) -> Path:
    output = EVALS / stage / f"seed_{args.seed}"
    command = [
        PYTHON, "tasks/evaluate_counterfactual_checkpoint.py",
        "--checkpoint", str(ckpt), "--output-dir", str(output),
        "--device", args.device, "--batch-size", str(args.eval_batch_size),
        "--bootstrap", str(1000 if args.smoke else args.bootstrap),
    ]
    command.append("--amp" if args.amp else "--no-amp")
    run(command, args.dry_run)
    return output / "report.json"


def passes(report_path: Path, previous_path: Path | None, min_preservation: float) -> bool:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trained = report["trained"]
    if trained["mean_embedding_cosine_to_official"] < min_preservation:
        print("STOP: embedding preservation fell below the safety threshold.")
        return False
    if trained["mrr_minus_official"] < 0 or trained["top1_minus_official"] < -0.001:
        print("STOP: complete-candidate retrieval did not pass.")
        return False
    if previous_path is not None:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))["trained"]
        if trained["mrr"] + 0.0005 < previous["mrr"] and trained["top1"] <= previous["top1"]:
            print("STOP: deeper unfreezing did not improve over the previous stage.")
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("check", "head", "last1", "last2", "aggressive"), required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--counterfactual-weight", type=float, default=0.7)
    parser.add_argument("--random-consistency-weight", type=float, default=0.2)
    parser.add_argument("--preserve-weight", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-preservation", type=float, default=0.98)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--device", default="auto",
                        help="auto (default) resolves to cuda if available, else cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick CPU smoke test: 1 epoch, 20 train/val batches, 1000 bootstrap")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable; use --device cpu")
    return requested


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    if args.amp and args.device != "cuda":
        print("AMP is not available on CPU; disabling automatic mixed precision.", flush=True)
        args.amp = False
    print(f"Device: {args.device} | AMP: {args.amp}", flush=True)
    if args.smoke:
        args.epochs = 1
        print("Smoke mode: 1 epoch, 20 train/val batches, 1000 bootstrap", flush=True)
    if args.stage == "check":
        check_cmd = [PYTHON, "tasks/check_counterfactual_training.py"]
        if args.device == "cpu":
            check_cmd.append("--allow-cpu")
        run(check_cmd, args.dry_run)
        return
    stages = [args.stage] if args.stage != "aggressive" else ["head", "last1", "last2"]
    previous_checkpoint = None
    previous_report = None
    if stages[0] == "last1":
        candidate = checkpoint("head", args.seed)
        previous_checkpoint = candidate
        candidate_report = EVALS / "head" / f"seed_{args.seed}" / "report.json"
        previous_report = candidate_report if candidate_report.is_file() else None
    elif stages[0] == "last2":
        candidate = checkpoint("last1", args.seed)
        previous_checkpoint = candidate
        candidate_report = EVALS / "last1" / f"seed_{args.seed}" / "report.json"
        previous_report = candidate_report if candidate_report.is_file() else None
    for stage in stages:
        if previous_checkpoint is not None and not previous_checkpoint.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Previous stage checkpoint missing: {previous_checkpoint}")
        current = train(stage, previous_checkpoint, args)
        report = evaluate(stage, current, args)
        if not args.dry_run and not passes(report, previous_report, args.min_preservation):
            raise SystemExit(f"Stopped after {stage}; see {report}")
        previous_checkpoint, previous_report = current, report
    print(f"\nWorkflow complete. Best completed stage: {stages[-1]}")
    print(f"Checkpoint: {previous_checkpoint}")
    print("Confirmation/test were not used.")


if __name__ == "__main__":
    main()
