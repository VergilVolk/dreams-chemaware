#!/bin/bash
#SBATCH --job-name=noise_pn_cap
#SBATCH --partition=gpu
#SBATCH --array=0-5
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_pn_cap_%A_%a.out
#SBATCH --error=data/validation/noise_pn_cap_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

HEADROOM="data/validation/g8r_noise_final_pn_fivepoint_headroom.json"
SUMMARY="data/validation/g8r_noise_final_pn_weight_scan_summary.json"
[[ -f "$HEADROOM" && -f "$SUMMARY" ]] || {
  echo "Missing five-point headroom or P-weight selection artifact" >&2; exit 5;
}
read -r PASS WEIGHT < <(python - "$HEADROOM" "$SUMMARY" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
s = json.load(open(sys.argv[2]))
selected = s.get("selected")
print(int(bool(h.get("pass_to_fivepoint_capacity_scan"))), "NONE" if selected is None else selected["positive_stream_weight"])
PY
)
[[ "$PASS" == "1" && "$WEIGHT" != "NONE" ]] || {
  echo "Five-point supervision gate or P-weight selection failed; refusing capacity scan" >&2; exit 6;
}

BLOCKS=(1 1 2 2 2 3)
ERROR_VIEWS=(2 4 0 2 4 2)
BACKBONE_LR=(2e-06 2e-06 1e-06 1e-06 1e-06 5e-07)
HEAD_LR=(1e-05 1e-05 1e-05 1e-05 1e-05 1e-05)
I="$SLURM_ARRAY_TASK_ID"
BLOCK="${BLOCKS[$I]}"
EVIEW="${ERROR_VIEWS[$I]}"
BLR="${BACKBONE_LR[$I]}"
HLR="${HEAD_LR[$I]}"
SEED=20260828
FOLD=0
SUFFIX="pn_capacity_c${I}"
TAG="curriculum_all_views4_blocks${BLOCK}_blr_${BLR}_hlr_${HLR}_pnw_${WEIGHT}_pv2"
if [[ "$EVIEW" != "0" ]]; then TAG="${TAG}_ev${EVIEW}"; fi
TAG="${TAG}_${SUFFIX}"
OUTPUT="data/validation/g8r_noise_final_e4a_direct/${TAG}/seed_${SEED}/fold_${FOLD}"
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite P/N/S capacity result: $OUTPUT" >&2; exit 4;
}

echo "=== P/N/S capacity c=$I block=$BLOCK errors=$EVIEW weight=$WEIGHT started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_pn_shared_encoder.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --policy curriculum --action-scope all \
  --outer-fold "$FOLD" --seed "$SEED" --epochs 4 \
  --batch-actions 4 --views-per-identity 4 --error-views-per-identity "$EVIEW" \
  --positive-spectra 4 --negative-molecules 20 \
  --positive-stream-weight "$WEIGHT" --positive-ratio 1.0 \
  --positive-views-per-identity 2 \
  --lambda-positive-rank 1.0 --lambda-positive-margin-floor 2.0 \
  --unfreeze-blocks "$BLOCK" --backbone-lr "$BLR" --head-lr "$HLR" \
  --grad-clip 1.0 --run-suffix "$SUFFIX" \
  --rank-margin 0.05 --temperature 0.10 \
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0 \
  --lambda-consistency 0.25 --lambda-margin-floor 2.0 \
  --lambda-preserve 5.0 --margin-floor-slack 0.005 \
  --safety-ratio 1.0 --device cuda --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== P/N/S capacity c=$I completed: $(date) ==="
