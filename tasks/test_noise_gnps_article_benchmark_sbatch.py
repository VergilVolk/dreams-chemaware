#!/usr/bin/env python
from pathlib import Path


def main() -> None:
    text = Path("tasks/run_noise_gnps_article_benchmark_1gpu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "--partition" not in text
    assert "#SBATCH --mem" not in text
    assert "build_noise_gnps_article_score_bundle.py" in text
    assert "evaluate_noise_gnps_article_benchmark.py" in text
    assert "plot_noise_gnps_article_benchmark.py" in text
    assert "test_gnps_pair_score_benchmark.py" in text
    assert "requires Python 3.11" in text
    assert "gnps_absolute_baselines_run_2347472" in text
    assert "ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64" in text
    assert "train_" not in "\n".join(
        line for line in text.splitlines() if line.strip().startswith("python")
    )
    print("[test_noise_gnps_article_benchmark_sbatch] PASS")


if __name__ == "__main__":
    main()
