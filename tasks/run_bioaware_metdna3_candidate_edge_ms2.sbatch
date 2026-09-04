#!/usr/bin/env bash
#SBATCH --job-name=metdna3_f2b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=metdna3_f2b_%j.out
#SBATCH --error=metdna3_f2b_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

echo "=== BioAware MetDNA3 F2-B started: $(date) ==="
if [[ ! -s data/validation/bioaware_metdna3_feature_ms2_cache_v1/report.json ]]; then
  python -u tasks/build_bioaware_metdna3_feature_ms2_cache.py
fi
python -u tasks/audit_bioaware_metdna3_candidate_edge_ms2.py
python -u tasks/summarize_bioaware_metdna3_candidate_edge_decision.py
python -m pytest tests/test_bioaware_metdna3_candidate_edge_ms2.py tests/test_bioaware_metdna3_candidate_edge_decision.py tests/test_metdna3_similarity.py tests/test_metdna3_mzml.py -q
echo "=== BioAware MetDNA3 F2-B completed: $(date) ==="
