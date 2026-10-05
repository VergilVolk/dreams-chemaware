#!/usr/bin/env python
"""Static launch-contract test for conditional-null stage 1."""
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("run_conditional_null_energy_stage1_1gpu.sbatch")
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text and "#SBATCH --mem-per-cpu" not in text
    assert "\nsbatch " not in text
    assert "EVIDENCE_DIR" in text and "candidate_graph.npz" in text
    assert "--graph \"$GRAPH\"" in text
    assert "joint no_interaction spectral_only chem_only" in text
    assert "adjudicate_conditional_null_energy.py" in text
    assert "candidate_rotated_truthblind" not in text
    print("[test_conditional_null_energy_sbatch] PASS")


if __name__ == "__main__":
    main()
