"""Static safety contracts for the hard-triplet union continuation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tasks/run_chemaware_fragment_massbank_union_continue.sbatch"


def main() -> None:
    text = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text
    assert "step-000750.ckpt" in text
    assert "run_2346306/qualified_source" in text
    assert "chemaware_massbank_candidate_source_all_qualified_v1_20260928" in text
    assert text.count("--source-ledger") == 2
    assert "--matched-control-policy family_qualified" in text
    assert "--minimum-chemical-events 1000 --minimum-chemical-queries 0" in text
    assert '--official-checkpoint "$START"' in text
    assert '--paired-reference fragment_graph_step750 --formula-role 2' in text
    assert "teacher" not in text.lower()
    assert "distill" not in text.lower()
    print("PASS: ChemAware fragment/MassBank union continuation contracts")


if __name__ == "__main__":
    main()
