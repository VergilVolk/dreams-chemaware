#!/bin/bash
#SBATCH --job-name=gnps10ppm_validate
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/gnps10ppm_validate_%j.out
#SBATCH --error=data/validation/gnps10ppm_validate_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
test -d "$BENCHMARK"

(
  cd "$BENCHMARK"
  sha256sum -c checksums.sha256
)

python -u tasks/test_build_gnps_gold_silver_10ppm_benchmark.py
python -u tasks/test_evaluate_gnps_gold_silver_10ppm_embeddings.py
python -u tasks/validate_gnps_gold_silver_10ppm_benchmark.py

echo "GNPS_GOLD_SILVER_10PPM_VALIDATION_PASS"
