#!/bin/bash
#SBATCH --job-name=chem_evt_peft
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_evt_peft_%j.out
#SBATCH --error=data/validation/chem_evt_peft_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

required_files=(
  tasks/train_chemaware_crossview_event_peft.py
  tasks/chemaware_candidate_residual_distillation_core.py
  tasks/chemaware_crossview_consensus_core.py
  tasks/chemaware_direct_training_core.py
  tasks/chemaware_frozen_prefix_cache.py
  tasks/chemaware_iceberg_direct_core.py
  tasks/noise_final_core.py
  tasks/noise_v3_core.py
  tasks/train_e1_identity.py
  tasks/e1_checkpoint_io.py
  tasks/train_noise_final_r2_shared_encoder.py
  dreams/models/chem_aware/peft_v3.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required runtime file: $required_file" >&2
    exit 2
  }
done

expected_peft_sha="fde14aee280ed13bd0dbc793d5739cdc989bedf377ce0000b5a5663d6ce715ba"
observed_peft_sha="$(sha256sum dreams/models/chem_aware/peft_v3.py | awk '{print $1}')"
[[ "$observed_peft_sha" == "$expected_peft_sha" ]] || {
  echo "Stale PEFT runtime: dreams/models/chem_aware/peft_v3.py" >&2
  echo "Expected SHA256: $expected_peft_sha" >&2
  echo "Observed SHA256: $observed_peft_sha" >&2
  exit 2
}

OUT="data/validation/chemaware_crossview_event_peft/run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

python -u tasks/train_chemaware_crossview_event_peft.py \
  --arm all --output "$OUT" \
  --epochs 3 --batch-queries 8 --event-batch-size 16 \
  --references-per-candidate 4 --reference-temperature 0.002 \
  --last-blocks 1 --rank 4 --alpha 4 --lr 2e-4 \
  --lambda-event 1.0 --maximum-event-gradient-ratio 0.25 \
  --lambda-safety 2.0 --lambda-preserve 5.0 \
  --target-margin 0.005 --protection-slack 0.002 --event-huber 0.02 \
  --frozen-prefix-cache --prefix-cache-batch-size 8 \
  --eval-batch-size 96 --bootstrap-draws 10000 \
  --device cuda --no-amp --torch-threads 8

test -f "$OUT/report.json"
test -f "$OUT/held_predictions.npz"
test -f "$OUT/peft_checkpoints.pt"
