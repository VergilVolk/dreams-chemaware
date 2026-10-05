"""Summarize the Injector-V1 plus historical-champion action canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from summarize_noise_corrected_v10_safe_exact_canary import summarize


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-action-view-dir", type=Path, required=True)
    parser.add_argument("--scalar-transfer-only-dir", type=Path, required=True)
    parser.add_argument("--matched-shuffled-dir", type=Path, required=True)
    parser.add_argument("--clean-control-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    paths = {
        "full_action_view": args.full_action_view_dir,
        "scalar_transfer_only": args.scalar_transfer_only_dir,
        "matched_shuffled": args.matched_shuffled_dir,
        "clean_control": args.clean_control_dir,
    }
    report = summarize(
        paths,
        repeats=args.bootstrap_resamples,
        seed=args.seed,
        expected_direct_contract="best_action_v11_historical_best",
        expected_action_bank_contract="historical_best_v1",
        status="noise_corrected_v11_historical_best_canary_complete",
        result_name="V11",
        interpretation=(
            "V11 changes only the upstream corrective action supplier from all "
            "strict views to the deterministic per-query historical champion. "
            "The direct E4/E8 objective and versioned Injector V1 remain frozen."
        ),
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
