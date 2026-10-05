"""Static fail-closed audit of every ChemAware GPU training entrypoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
ACTIVE = {
    "run_chemaware_candidate_residual_arm.sbatch": (
        "#SBATCH --gpus=1", "#SBATCH --time=02:00:00", "#SBATCH --array=0-4%1",
        "CHEMAWARE_GPU_AUTHORIZATION", "--epochs 2", "--max-train-identities 512",
        "verify_chemaware_gpu_authorization.py",
        "--gpu-authorization \"$CHEMAWARE_GPU_AUTHORIZATION\"",
    ),
    "run_chemaware_direct_action_views_pilot.sbatch": (
        "#SBATCH --gpus=1", "#SBATCH --time=08:00:00",
        "CHEMAWARE_DIRECT_ACTION_BANK_PASS", "--training-objective direct_projected_guarded",
        "--epochs 2", "--max-action-identities 512", "--lambda-consistency 0",
        "--lambda-margin-floor 2", "--lambda-preserve 20", "--lambda-peak-contrast 0",
        "--maximum-action-gradient-ratio 0.25", "--preflight-only", "COMPLETE.json",
        "run_arm clean_duplicate", "run_arm correct_synthetic", "--stage primary",
        "pass_to_matched_controls", "run_arm candidate_swapped", "run_arm peak_permuted",
    ),
}
LEGACY = {
    "run_chemaware_full_candidate_direct_official.sbatch",
    "run_chemaware_full_candidate_official_pilot.sbatch",
    "run_chemaware_mass_kernel_direct_arm.sbatch",
    "run_chemaware_rule_margin_transfer_arm.sbatch",
    "run_chemaware_shared_v2_g1.sbatch", "run_chemaware_shared_v2_g2.sbatch",
    "run_chemaware_shared_v3_g1.sbatch", "run_chemaware_shared_v3_g2.sbatch",
    "run_chemaware_shared_v3_g2_pilot.sbatch", "run_chemaware_shared_v3_g2b.sbatch",
    "run_chemaware_shared_v3_g2b_pilot.sbatch",
}


def active_directive(text: str, option: str) -> bool:
    return any(line.startswith(f"#SBATCH --{option}") for line in text.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    failures = []
    discovered = {}
    for path in sorted(TASKS.glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        if "train_chemaware" not in text and "pilot_chemaware" not in text:
            continue
        discovered[path.name] = {
            "active_gpu": active_directive(text, "gpus="),
            "active_array": active_directive(text, "array="),
        }
        if path.name in ACTIVE:
            required = ACTIVE[path.name]
            if missing := [value for value in required if value not in text]:
                failures.append(f"active entrypoint missing constraints: {missing}")
            if path.name == "run_chemaware_direct_action_views_pilot.sbatch" and active_directive(text, "array="):
                failures.append("direct-action staged pilot must not use a parallel array")
        elif path.name in LEGACY:
            prefix = text.split("set -euo", 1)[0]
            if ("BLOCKED:" not in prefix or "exit 64" not in prefix
                    or active_directive(text, "gpus=") or active_directive(text, "array=")
                    or "#SBATCH --time=00:01:00" not in text):
                failures.append(f"legacy entrypoint not safely quarantined: {path.name}")
        else:
            failures.append(f"unclassified ChemAware training entrypoint: {path.name}")
    if set(discovered) != LEGACY | set(ACTIVE):
        failures.append("classified/discovered training entrypoint sets differ")
    launchers = {}
    active_launcher = "submit_chemaware_candidate_residual_pilot.sh"
    for path in sorted(TASKS.glob("submit_chemaware*.sh")):
        text = path.read_text(encoding="utf-8")
        if path.name == active_launcher:
            authorizer = text.find("authorize_chemaware_gpu_submission.py")
            submit = text.find("sbatch")
            safe = authorizer >= 0 and submit > authorizer and "--max-total-gpu-hours 10" in text
        else:
            lines = text.splitlines()
            submit_line = next((i for i, line in enumerate(lines) if line.strip().startswith("sbatch ")), len(lines))
            prefix = "\n".join(lines[:submit_line])
            safe = "BLOCKED:" in prefix and ("exit 64" in prefix or "exit 2" in prefix)
        launchers[path.name] = {"safe": safe, "active": path.name == active_launcher}
        if not safe:
            failures.append(f"unsafe ChemAware submit launcher: {path.name}")
    report = {
        "status": "CHEMAWARE_GPU_ENTRYPOINTS_SAFE" if not failures else "CHEMAWARE_GPU_ENTRYPOINTS_UNSAFE",
        "active_training_entrypoints": sorted(ACTIVE),
        "active_controls": {
            "candidate_residual": "24-hour CPU authorization plus evidence rehash",
            "direct_action_views": (
                "passed action bank with payload/provenance rehash, exact-argument CPU preflight, "
                "sequential stage-1 cost gate, atomic completion marker, and an eight-GPU-hour job cap"
            ),
        },
        "legacy_training_entrypoints_quarantined": sorted(LEGACY),
        "discovered": discovered, "submit_launchers": launchers, "failures": failures,
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=False)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
