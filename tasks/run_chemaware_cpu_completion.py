"""Finish every currently possible ChemAware stage on the CPU-only machine.

Order matters: finish/resume MCES preparation first, then resume the interrupted
P1 run through epoch 2, then perform a fixed 500-anchor hard-panel safety audit.
GPU-only formal evaluation and multitask training remain explicit downstream
gates and are never reported as complete by this launcher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
P1_DIR = ROOT / "data/e1/strict_counterfactual_full_cpu/seed_20260815"


def run(command: list[str], dry_run: bool, commands: list[list[str]]) -> None:
    commands.append(command)
    print(f"\n>>> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mces-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    resume = P1_DIR / "latest_resume.pt"
    best = P1_DIR / "best_causal_head.pt"
    if not resume.is_file() and not args.dry_run:
        raise FileNotFoundError(resume)
    commands: list[list[str]] = []

    run([
        PYTHON, "tasks/run_chemaware_pipeline.py", "--mode", "cpu",
        "--mces-workers", str(args.mces_workers),
    ], args.dry_run, commands)
    run([
        PYTHON, "tasks/train_causal_chemmask_full_cpu.py",
        "--resume", str(resume), "--epochs", "2",
        "--cpu-threads", str(args.cpu_threads),
    ], args.dry_run, commands)
    if not best.is_file() and not args.dry_run:
        raise FileNotFoundError(best)
    run([
        PYTHON, "tasks/eval_causal_cpu_paired.py",
        "--checkpoint", str(best), "--device", "cpu", "--batch-size", "4",
        "--cpu-threads", str(args.cpu_threads), "--n-bootstrap", "2000",
        "--panel", "hard", "--triplets", "500", "--negative-probe-size", "32",
        "--panel-seed", "20260815",
        "--output", str(ROOT / "data/pipeline/p1_postresume_hard_n500.json"),
    ], args.dry_run, commands)
    run([
        PYTHON, "tasks/run_chemaware_pipeline.py", "--mode", "status",
        "--p1-checkpoint", str(best),
    ], args.dry_run, commands)

    status = "planned" if args.dry_run else "cpu_completion_finished"
    report = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "p1_checkpoint": str(best.resolve()),
        "p1_epochs_requested": 2,
        "commands": commands,
        "remaining_boundary": "formal locked evaluation and multitask training require CUDA",
    }
    output = ROOT / "data/pipeline/cpu_completion_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
