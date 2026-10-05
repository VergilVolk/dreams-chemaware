"""Freeze one causally selected ChemAware checkpoint before outer evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-summary", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--development-ledgers", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--outer-selection-seed", type=int, default=20260934)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_freeze(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite release freeze: {args.output}")
    summary = read(args.causal_summary)
    report = read(args.candidate_report)
    if summary.get("status") != "CANDIDATE_RESIDUAL_CAUSAL_DEVELOPMENT_PASS":
        raise RuntimeError("release freeze blocked: matched causal development gate did not pass")
    selected = summary.get("selected_for_outer_confirmation")
    if selected is None:
        raise RuntimeError("release freeze blocked: no uniquely selected arm")
    if report.get("status") != "ARM_COMPLETE":
        raise RuntimeError("release freeze blocked: selected arm is incomplete")
    optimization = report.get("optimization", {})
    expected_arm = {"rule": "rule_response"}.get(selected)
    if expected_arm is None or optimization.get("teacher_arm") != expected_arm:
        raise RuntimeError("release freeze blocked: candidate does not match selected arm")
    contracts = report.get("preflight", {}).get("contracts", {})
    if contracts.get("outer_fold_evaluated") is not False:
        raise RuntimeError("release freeze blocked: candidate report does not attest sealed outer outcomes")
    if int(optimization.get("outer_fold", -1)) != args.outer_fold:
        raise RuntimeError("release freeze blocked: outer fold differs from development contract")
    arm_provenance = summary.get("provenance", {}).get("arms", {}).get(selected, {})
    if arm_provenance.get("report_sha256") != sha256_file(args.candidate_report):
        raise RuntimeError("release freeze blocked: selected report is not the summarized artifact")
    if arm_provenance.get("checkpoint_sha256") != sha256_file(args.candidate_checkpoint):
        raise RuntimeError("release freeze blocked: selected checkpoint is not the summarized artifact")
    if args.bootstrap_draws < 10_000:
        raise RuntimeError("release freeze blocked: outer bootstrap requires at least 10,000 draws")
    ledgers = {}
    for path in args.development_ledgers:
        if not path.is_file():
            raise FileNotFoundError(path)
        ledgers[str(path)] = sha256_file(path)
    return {
        "status": "CHEMAWARE_RELEASE_CANDIDATE_FROZEN",
        "release_eligible": False,
        "outer_evaluated": False,
        "candidate": {
            "selected_arm": selected,
            "report_path": str(args.candidate_report),
            "report_sha256": sha256_file(args.candidate_report),
            "checkpoint_path": str(args.candidate_checkpoint),
            "checkpoint_sha256": sha256_file(args.candidate_checkpoint),
        },
        "development_freeze": {
            "causal_summary_path": str(args.causal_summary),
            "causal_summary_sha256": sha256_file(args.causal_summary),
            "ledgers": ledgers,
            "no_further_model_or_threshold_selection_allowed": True,
        },
        "outer_protocol": {
            "fold": args.outer_fold, "identity_balanced": True,
            "selection_seed": args.outer_selection_seed, "bootstrap_draws": args.bootstrap_draws,
            "single_consumption": True,
            "pass_gate": "strict-positive formula-cluster CI versus official, corrected>introduced, MRR nonnegative, preservation>=0.995",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    args = arguments(); body = build_freeze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
