"""Cross-platform, stepwise runner for the low-budget E1 experiment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
OFFICIAL = ROOT / "dreams/models/pretrained/embedding_model.ckpt"
OFFICIAL_SLIM = ROOT / "data/e1/official_embedding_slim.pt"
DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
TRAIN_POOL = ROOT / "data/e1/e1_train_triplet_pool.npz"
VAL_POOL = ROOT / "data/e1/e1_val_triplet_pool.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run budgeted E1 stages with the active conda Python")
    parser.add_argument(
        "--stage",
        choices=("pools", "prepare", "check", "raw", "official", "pilot-a", "pilot-b", "summary", "lean", "all"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batches", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()


def run(command: list[str], dry_run: bool = False) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n>>> {printable}", flush=True)
    if not dry_run:
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"Stage stopped because the command above returned exit code "
                f"{error.returncode}. Fix the reported FAIL item and rerun this stage."
            ) from None


def build_pools(args: argparse.Namespace, only_if_missing: bool = False) -> None:
    for fold, output in (("train", TRAIN_POOL), ("val", VAL_POOL)):
        if only_if_missing and output.is_file():
            print(f"Pool already exists, keeping it: {output}")
            continue
        run([
            PYTHON, "tasks/build_e1_triplet_pool.py", "--data", str(DATA),
            "--fold", fold, "--adduct", "[M+H]+", "--mass-window-da", "0.05",
            "--output", str(output),
        ], args.dry_run)


def evaluate(ckpt: Path, output: Path, args: argparse.Namespace) -> None:
    run([
        PYTHON, "tasks/eval_e0_baseline.py",
        "--data", str(DATA), "--ckpt", str(ckpt), "--fold", "val",
        "--device", "cuda", "--batch-size", str(args.eval_batch_size),
        "--n-bootstrap", str(args.bootstrap), "--output-dir", str(output),
    ], args.dry_run)


def prepare_official(args: argparse.Namespace) -> None:
    if OFFICIAL_SLIM.is_file():
        print(f"Slim official checkpoint already exists, keeping it: {OFFICIAL_SLIM}")
        return
    run([
        PYTHON, "tasks/prepare_official_embedding_checkpoint.py",
        "--source", str(OFFICIAL), "--output", str(OFFICIAL_SLIM),
    ], args.dry_run)


def train_pilot(label: str, base: Path, lr: str, args: argparse.Namespace) -> Path:
    output_root = ROOT / f"data/e1/budget/{label}"
    run([
        PYTHON, "tasks/train_e1_identity.py",
        "--data", str(DATA), "--train-pool", str(TRAIN_POOL), "--val-pool", str(VAL_POOL),
        "--base-ckpt", str(base), "--architecture-ckpt", str(RAW),
        "--output-dir", str(output_root), "--seed", str(args.seed),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--grad-accum", "2", "--lr", lr, "--margin", "0.1",
        "--n-highest-peaks", "100", "--num-workers", str(args.workers),
        "--val-triplets", str(args.val_batches * args.batch_size),
        "--max-train-batches", str(args.train_batches),
        "--max-val-batches", str(args.val_batches), "--patience", "2",
        "--device", "cuda", "--amp",
    ], args.dry_run)
    return output_root / f"seed_{args.seed}/best_e1.pt"


def main() -> None:
    args = parse_args()
    stage = args.stage

    if stage == "pools":
        build_pools(args)
        return
    if stage == "prepare":
        prepare_official(args)
        return
    if stage == "check":
        run([PYTHON, "tasks/check_e1_budget.py"], args.dry_run)
        return
    if stage == "raw":
        evaluate(RAW, ROOT / "data/validation/e0_baseline", args)
        return
    if stage == "official":
        prepare_official(args)
        evaluate(OFFICIAL_SLIM, ROOT / "data/validation/e1_budget/r0_official", args)
        return
    if stage == "pilot-a":
        checkpoint = train_pilot("pilot_a", RAW, "5e-6", args)
        if args.dry_run or checkpoint.is_file():
            evaluate(checkpoint, ROOT / "data/validation/e1_budget/pilot_a", args)
        return
    if stage == "pilot-b":
        prepare_official(args)
        checkpoint = train_pilot("pilot_b", OFFICIAL_SLIM, "1e-6", args)
        if args.dry_run or checkpoint.is_file():
            evaluate(checkpoint, ROOT / "data/validation/e1_budget/pilot_b", args)
        return
    if stage == "summary":
        run([PYTHON, "tasks/summarize_e1_budget.py"], args.dry_run)
        return

    if stage == "lean":
        build_pools(args, only_if_missing=True)
        prepare_official(args)
        run([PYTHON, "tasks/check_e1_budget.py"], args.dry_run)
        evaluate(OFFICIAL_SLIM, ROOT / "data/validation/e1_budget/r0_official", args)
        checkpoint_b = train_pilot("pilot_b", OFFICIAL_SLIM, "1e-6", args)
        evaluate(checkpoint_b, ROOT / "data/validation/e1_budget/pilot_b", args)
        run([PYTHON, "tasks/summarize_e1_budget.py"], args.dry_run)
        return

    build_pools(args, only_if_missing=True)
    prepare_official(args)
    run([PYTHON, "tasks/check_e1_budget.py"], args.dry_run)
    raw_report = ROOT / "data/validation/e0_baseline/e0_report.json"
    if not raw_report.is_file():
        evaluate(RAW, ROOT / "data/validation/e0_baseline", args)
    else:
        print(f"Raw E0 report already exists, keeping it: {raw_report}")
    evaluate(OFFICIAL_SLIM, ROOT / "data/validation/e1_budget/r0_official", args)
    checkpoint_a = train_pilot("pilot_a", RAW, "5e-6", args)
    evaluate(checkpoint_a, ROOT / "data/validation/e1_budget/pilot_a", args)
    checkpoint_b = train_pilot("pilot_b", OFFICIAL_SLIM, "1e-6", args)
    evaluate(checkpoint_b, ROOT / "data/validation/e1_budget/pilot_b", args)
    run([PYTHON, "tasks/summarize_e1_budget.py"], args.dry_run)


if __name__ == "__main__":
    main()
