"""Evaluate a causal ChemMask head on the frozen large structure-pair atlas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
BASELINE = ROOT / "data/validation/dreams_structure_residual_atlas_large_v2/summary.json"


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n>>> {subprocess.list2cmdline(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def metric_block(summary: dict) -> dict:
    metrics = summary["confirmation_metrics"]["official_finetuned"]
    result = {}
    for name in (
        "all_pairs", "different_identity_pairs", "same_formula_different_identity_pairs"
    ):
        values = metrics.get(name, {})
        result[name] = {
            key: values.get(key) for key in ("n", "pearson_r", "spearman_rho")
        }
    for key in ("residual_mae", "residual_rmse", "identity_cosine_mean"):
        result[key] = metrics.get(key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--include-confirmation", action="store_true",
        help="Read the locked confirmation cohort only after model selection is frozen.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    discovery = args.output_root / "embeddings_discovery"
    confirmation = args.output_root / "embeddings_confirmation"
    atlas = args.output_root / "residual_atlas"
    split_outputs = [("discovery", discovery)]
    if args.include_confirmation:
        split_outputs.append(("confirmation", confirmation))
    for split, output in split_outputs:
        run([
            PYTHON, "tasks/encode_large_observability_cohort.py",
            "--output-dir", str(output), "--splits", split,
            "--official-checkpoint", str(args.checkpoint),
            "--device", args.device, "--batch-size", str(args.batch_size),
            "--save-precursor-tokens",
        ], args.dry_run)
    if args.include_confirmation:
        run([
            PYTHON, "tasks/audit_dreams_structure_residuals.py",
            "--discovery-dir", str(discovery),
            "--confirmation-dir", str(confirmation),
            "--output-dir", str(atlas),
            "--pairs-per-bin-discovery", "8000",
            "--pairs-per-bin-confirmation", "3000",
            "--identity-pairs-discovery", "8000",
            "--identity-pairs-confirmation", "3000",
            "--n-bootstrap", str(args.n_bootstrap), "--skip-raw",
        ], args.dry_run)
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.include_confirmation:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        candidate = json.loads((atlas / "summary.json").read_text(encoding="utf-8"))
        before = metric_block(baseline)
        after = metric_block(candidate)
        comparison = {
            "checkpoint": str(args.checkpoint.resolve()),
            "protocol": "locked molecule-disjoint large-v2 confirmation",
            "baseline_official": before,
            "candidate_causal_head": after,
            "delta": {},
        }
        for family in (
            "all_pairs", "different_identity_pairs", "same_formula_different_identity_pairs"
        ):
            comparison["delta"][family] = {}
            for metric in ("pearson_r", "spearman_rho"):
                a, b = before[family].get(metric), after[family].get(metric)
                comparison["delta"][family][metric] = None if a is None or b is None else b - a
        (args.output_root / "official_vs_causal_structure.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        print(json.dumps(comparison, indent=2), flush=True)
    sweep_command = [
        PYTHON, "tasks/eval_causal_heads_on_fixed_pairs.py",
        "--run-dir", str(args.checkpoint.parent),
        "--discovery-dir", str(discovery),
        "--output", str(args.output_root / "epoch_head_comparison"),
        "--device", args.device,
    ]
    if args.include_confirmation:
        sweep_command.extend(["--confirmation-dir", str(confirmation)])
    run(sweep_command, False)


if __name__ == "__main__":
    main()
