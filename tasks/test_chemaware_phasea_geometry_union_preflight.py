"""Static safety contract for the Phase-A geometry-only union audit."""
from pathlib import Path


def main() -> None:
    text = Path(__file__).with_name(
        "run_chemaware_phasea_geometry_union_preflight.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text and "#SBATCH --mem" not in text
    assert '--geometry-checkpoint "$PHASEA"' in text
    assert '--checkpoint "$PHASEA"' in text
    assert "--matched-control-policy pairwise_dominance" in text
    assert "--maximum-positive-references-per-relation 1" in text
    assert "--maximum-negative-references-per-event 1" in text
    assert '"weights_updated": False' in text
    assert "train_chemaware_dreams_native.py" not in text
    assert "--restore-adam-from" not in text
    print("PASS: ChemAware Phase-A geometry union preflight contracts", flush=True)


if __name__ == "__main__":
    main()
