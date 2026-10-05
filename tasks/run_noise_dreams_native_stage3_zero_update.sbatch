#!/bin/bash
#SBATCH --job-name=noise_s3_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_s3_audit_%j.out
#SBATCH --error=data/validation/noise_s3_audit_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 1 )); then
  echo "Expected exactly one allocated GPU, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

STAGE3_RUN="${NOISE_STAGE3_RUN:-data/validation/noise_dreams_native_multidifficulty_stage3_fold_0_run_2345317}"
TRIPLETS="$STAGE3_RUN/triplets"
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
WARM_START="$STAGE1/checkpoint/targeted_final.ckpt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OUTPUT="data/validation/noise_dreams_native_stage3_zero_update_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_s3_audit_${SLURM_JOB_ID}"
SOURCE_SNAPSHOT="$LOCAL_ROOT/source_snapshot"

[[ ! -e "$OUTPUT" ]] || { echo "Output collision: $OUTPUT" >&2; exit 4; }
required_files=(
  "$TRIPLETS/report.json"
  "$TRIPLETS/train_pool.npz"
  "$TRIPLETS/action_spectra.npz"
  "$TRIPLETS/selected_actions.csv.gz"
  "$WARM_START" "$ARCHITECTURE" "$DATA"
  tasks/audit_noise_dreams_native_stage3_zero_update.py
  tasks/test_noise_dreams_native_stage3_zero_update.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing Stage-3 audit input: $path" >&2; exit 5; }
done

cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT
mkdir -p "$SOURCE_SNAPSHOT"
cp -a dreams "$SOURCE_SNAPSHOT/dreams"
cp -a tasks "$SOURCE_SNAPSHOT/tasks"
export PYTHONPATH="$SOURCE_SNAPSHOT:$SOURCE_SNAPSHOT/tasks:${PYTHONPATH:-}"
AUDIT_TASKS="$SOURCE_SNAPSHOT/tasks"

python -m py_compile \
  "$AUDIT_TASKS/audit_noise_dreams_native_stage3_zero_update.py" \
  "$AUDIT_TASKS/test_noise_dreams_native_stage3_zero_update.py"
python -u "$AUDIT_TASKS/test_noise_dreams_native_stage3_zero_update.py"

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  "$AUDIT_TASKS/audit_noise_dreams_native_stage3_zero_update.py" \
  --triplet-dir "$TRIPLETS" \
  --warm-start-checkpoint "$WARM_START" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --data "$DATA" --output "$OUTPUT" \
  --score-batch-size 16 --gradient-events-per-stratum 8 \
  --seed 20260927

echo "=== Noise Stage-3 zero-update audit result: $OUTPUT ==="
