"""Static contracts for the official full-candidate server pilot."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cache = (ROOT / "tasks/run_chemaware_full_candidate_official_cache.sbatch").read_text()
    pilot = (ROOT / "tasks/run_chemaware_full_candidate_official_pilot.sbatch").read_text()
    direct = (ROOT / "tasks/run_chemaware_full_candidate_direct_official.sbatch").read_text()
    submit = (ROOT / "tasks/submit_chemaware_full_candidate_official_pilot.sh").read_text()
    for body in (cache, pilot, direct):
        assert "#SBATCH --mem" not in body
        assert "set -euo pipefail" in body
        assert "cd /data02/run01/scv7tsl/DreaMS" in body
        assert "official_embedding_slim.pt" in body
    assert "#SBATCH --gpus=1" in cache
    for body in (pilot, direct):
        assert "BLOCKED:" in body.split("set -euo", 1)[0]
        assert "exit 64" in body.split("set -euo", 1)[0]
        assert not any(line.startswith("#SBATCH --gpus=") for line in body.splitlines())
        assert "#SBATCH --time=00:01:00" in body
    assert "--device cuda" in cache
    assert "--arm spectrum_only" in pilot
    assert "--max-steps 500" in pilot and "--warmup-steps 50" in pilot
    assert "--error-identity-fraction 0.50" in pilot
    assert "--inner-fold 3 --outer-fold 4" in pilot
    assert "validate_chemaware_full_candidate_alignment.py" in pilot
    assert "${ADAPTER_RUN:?" in direct
    assert "--unfreeze-blocks 1" in direct
    assert "--backbone-lr 2e-6 --head-lr 1e-5" in direct
    assert "--device cuda --no-amp" in direct
    assert "validate_chemaware_full_candidate_direct.py" in direct
    assert "BLOCKED:" in submit.split("[[ -f", 1)[0] and "exit 64" in submit.split("[[ -f", 1)[0]
    print("PASS: historical official full-candidate launchers are quarantined")


if __name__ == "__main__":
    main()
