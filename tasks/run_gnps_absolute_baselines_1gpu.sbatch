#!/bin/bash
#SBATCH --job-name=gnps_abs_baseline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/gnps_abs_baseline_%j.out
#SBATCH --error=data/validation/gnps_abs_baseline_%j.err

# GNPS absolute baselines (Step 0 of the GNPS-certified plan).
#
# Before any further change can be called "beneficial", the sealed GNPS
# Gold/Silver 10ppm benchmark must produce absolute numbers for the three
# reference checkpoints: official DreaMS, Noise Stage-1, and the T1/T3 V1
# champion (94.2399% held fold-0).  This job encodes all three on the
# 52,871-spectrum benchmark union and runs two paired evaluations against
# the official baseline.  It performs NO training and never touches
# MassSpecGym held fold-0.  GNPS stays evaluation-only by charter; no GNPS
# spectrum ever enters training, or the benchmark loses its model-blind
# external status.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
OFFICIAL="data/e1/official_embedding_slim.pt"
STAGE1_CHECKPOINT="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820/checkpoint/targeted_final.ckpt"
V1_RUN="${V1_RUN:-data/validation/noise_relation_t1_t3_run_2347055}"
V1_SLIM="$V1_RUN/checkpoint/primary_seed_3407_slim.pt"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
RUN_ROOT="data/validation/gnps_absolute_baselines_run_${SLURM_JOB_ID}"
GNPS="$RUN_ROOT/gnps"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/gnps_absolute_baselines_${SLURM_JOB_ID}"
STAGE1_SLIM="$LOCAL_ROOT/stage1_targeted_slim.pt"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$GNPS" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$ARCHITECTURE" "$OFFICIAL" "$STAGE1_CHECKPOINT" "$V1_SLIM"
  "$GNPS_BENCHMARK/spectra.mgf"
  "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/test_gnps_absolute_baselines_sbatch.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing GNPS baseline input: $path" >&2; exit 5; }
done

python -u tasks/test_gnps_absolute_baselines_sbatch.py

# Format-only slim conversion of the Stage-1 Lightning checkpoint (the
# proven pattern); the encoder never sees the 1.2 GB package directly.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1_CHECKPOINT" --output "$STAGE1_SLIM"

encode() {
  local label="$1"
  local checkpoint="$2"
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$GNPS/${label}_embeddings.npz" --batch-size 64 --label "$label" \
    >"$RUN_ROOT/gnps_encode_${label}.out" 2>"$RUN_ROOT/gnps_encode_${label}.err"
}

encode official "$OFFICIAL"
encode stage1 "$STAGE1_SLIM"
encode v1 "$V1_SLIM"

# The evaluator's --output is a DIRECTORY; report.json lands inside it.
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings "$GNPS/official_embeddings.npz" \
  --candidate-embeddings "$GNPS/stage1_embeddings.npz" \
  --output "$GNPS/paired_official_vs_stage1" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261001
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings "$GNPS/official_embeddings.npz" \
  --candidate-embeddings "$GNPS/v1_embeddings.npz" \
  --output "$GNPS/paired_official_vs_v1" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261001

python - "$RUN_ROOT" <<'PYEOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("=== GNPS absolute Recall@1 ruler ===")
for name in ("paired_official_vs_stage1", "paired_official_vs_v1"):
    report = json.loads(
        (root / "gnps" / name / "report.json").read_text(encoding="utf-8"),
    )
    for panel, block in report["panels"].items():
        for side in ("baseline", "candidate"):
            metrics = block[side]
            reciprocal = {
                key: value for key, value in metrics.items()
                if "reciprocal" in key
            }
            print(
                f"{name} | {panel} | {side} | "
                f"recall@1={metrics['recall@1']:.6f} | {reciprocal}",
            )
PYEOF

echo "=== GNPS absolute baselines complete: $RUN_ROOT ==="
