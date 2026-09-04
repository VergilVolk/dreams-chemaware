#!/bin/bash
#SBATCH --job-name=bio_ctx_dev
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_context_development_%j.out
#SBATCH --error=bioaware_context_development_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/test_bioaware_context_evidence_tensor.py
python -u tasks/test_bioaware_factor_graph.py
python -u tasks/test_bioaware_context_adapter.py
python -u tasks/build_bioaware_context_evidence_tensor.py
python -u tasks/evaluate_bioaware_typed_factor_graph.py
python -u tasks/build_mtbls1905_context_adapter_dataset.py
python -u tasks/train_mtbls1905_context_adapter_oof.py --device cuda
