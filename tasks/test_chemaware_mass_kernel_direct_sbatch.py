"""Static server contracts for paired global/routed chemical-rule transfer."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    arm = (ROOT / "tasks/run_chemaware_mass_kernel_direct_arm.sbatch").read_text()
    summary = (ROOT / "tasks/run_chemaware_mass_kernel_direct_summary.sbatch").read_text()
    submit = (ROOT / "tasks/submit_chemaware_mass_kernel_direct_phase_a.sh").read_text()
    for body in (arm, summary):
        assert "#SBATCH --mem" not in body
        assert "set -euo pipefail" in body
        assert "cd /data02/run01/scv7tsl/DreaMS" in body
    prefix = arm.split("set -euo", 1)[0]
    assert "BLOCKED:" in prefix and "exit 64" in prefix
    assert not any(line.startswith("#SBATCH --gpus=") for line in arm.splitlines())
    assert not any(line.startswith("#SBATCH --array=") for line in arm.splitlines())
    assert "#SBATCH --time=00:01:00" in arm
    assert not any(line.startswith("#SBATCH --gpus=") for line in summary.splitlines())
    assert "LABELS=(none mass rule_response rule_mass rule_mass_shifted mass_error rule_mass_error rule_mass_shifted_error)" in arm
    assert "TEACHERS=(none mass rule_response rule_mass rule_mass_shifted mass rule_mass rule_mass_shifted)" in arm
    assert "BETAS=(0.00 0.10 0.20 0.20 0.20 0.10 0.20 0.20)" in arm
    assert "SCOPES=(all all all all all official_error official_error official_error)" in arm
    assert "run_${SLURM_ARRAY_JOB_ID}" in arm
    assert "--teacher-arm \"$ARM\"" in arm
    assert '--teacher-beta "$BETA"' in arm
    assert '--teacher-scope "$SCOPE" --teacher-boundary-margin 0.02' in arm
    assert "--training-mass formula_identity" in arm
    assert "--references-per-molecule 2" in arm
    assert "--device cuda --no-amp" in arm
    assert "--inner-fold 3 --outer-fold 4" in arm
    assert "BLOCKED:" in submit.split("[[ -f", 1)[0] and "exit 64" in submit.split("[[ -f", 1)[0]
    assert "summarize_chemaware_mass_kernel_direct_arms.py" in summary
    print("PASS: rejected chemical-rule launchers are quarantined")


if __name__ == "__main__":
    main()
