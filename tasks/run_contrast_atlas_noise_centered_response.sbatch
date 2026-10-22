#!/bin/bash
#SBATCH --job-name=ca_noise_centered
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/contrast_atlas_noise_centered_%j.out
#SBATCH --error=data/validation/contrast_atlas_noise_centered_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CONTRACT="tasks/contrast_atlas_open_collection_v1.json"
COLLECTION="data/validation/contrast_atlas_open_response_collection_v2_20261019"
NOISE_CHECKPOINT="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
EMBEDDINGS="data/validation/contrast_atlas_open_response_embeddings_noise_v1_20261022"
KERNEL="data/validation/contrast_atlas_centered_embedding_kernel_noise_v1_20261022"
HFD="data/validation/contrast_atlas_hfd_source_conditioned_axis_noise_v1_20261022"
OBESITY="data/validation/contrast_atlas_obesity_source_conditioned_axis_noise_v1_20261022"

for required in "$CONTRACT" "$COLLECTION/report.json" "$NOISE_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "required input missing: $required" >&2
    exit 2
  fi
done
for output in "$EMBEDDINGS" "$KERNEL" "$HFD" "$OBESITY"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite frozen output: $output" >&2
    exit 3
  fi
done

python -u tasks/encode_contrast_atlas_open_response_entities.py \
  --contract "$CONTRACT" \
  --collection "$COLLECTION" \
  --model-kind noise_v1 \
  --encoder-checkpoint "$NOISE_CHECKPOINT" \
  --output-dir "$EMBEDDINGS" \
  --device cuda \
  --batch-size 256

python -u tasks/score_contrast_atlas_centered_embedding_kernel.py \
  --collection "$COLLECTION" \
  --embedding-manifest "$EMBEDDINGS/embedding_manifest.tsv" \
  --output-dir "$KERNEL" \
  --null-replicates 31 \
  --null-seed 20261022

python -u tasks/audit_contrast_atlas_hfd_source_conditioned_axis.py \
  --kernel-dir "$KERNEL" \
  --output-dir "$HFD"

python -u tasks/audit_contrast_atlas_obesity_source_conditioned_axis.py \
  --kernel-dir "$KERNEL" \
  --output-dir "$OBESITY"

python - <<'PY'
import json
from pathlib import Path

paths = {
    "embeddings": Path("data/validation/contrast_atlas_open_response_embeddings_noise_v1_20261022/report.json"),
    "kernel": Path("data/validation/contrast_atlas_centered_embedding_kernel_noise_v1_20261022/report.json"),
    "hfd": Path("data/validation/contrast_atlas_hfd_source_conditioned_axis_noise_v1_20261022/report.json"),
    "obesity": Path("data/validation/contrast_atlas_obesity_source_conditioned_axis_noise_v1_20261022/report.json"),
}
reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
assert reports["embeddings"]["model_kind"] == "noise_v1"
assert reports["kernel"]["method_id"] == "noise_v1"
assert reports["hfd"]["denominators"]["target_contrasts"] == 4
assert reports["obesity"]["denominators"]["obesity_queries"] == 4
print(json.dumps({
    "status": "CONTRAST_ATLAS_NOISE_CENTERED_RESPONSE_COMPLETE",
    "hfd_observed": reports["hfd"]["observed"],
    "hfd_sensitivity": reports["hfd"]["sensitivity"],
    "obesity_summary": reports["obesity"]["summary"],
}, indent=2))
PY
