#!/bin/bash
#SBATCH --job-name=noise_e14_summary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_e14_summary_%j.out
#SBATCH --error=data/validation/noise_e14_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ROOT_OUT="data/validation/g8r_noise_final_e14_selected_transfer"
SEED=20260906
FOLD=4
CONTROL="$ROOT_OUT/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_e14_continuation_control/seed_${SEED}/fold_${FOLD}"
NO_MARGIN="$ROOT_OUT/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_gpn_selected_gw1_gv2_gtm_symmetric_grp0.67_gmax5_gscope_positive_deficit_errors_gtmargin0_gttargetabsolute_gtdf0.5_grisk0_gbal0_e14_selected_no_fixed_margin/seed_${SEED}/fold_${FOLD}"
DELTA="$ROOT_OUT/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_gpn_selected_gw1_gv2_gtm_symmetric_grp0.67_gmax5_gscope_positive_deficit_errors_gtmargin1_gttargetdelta_gtdf0.5_grisk0_gbal0_e14_selected_conservative_delta/seed_${SEED}/fold_${FOLD}"
DELTA_RISK="$ROOT_OUT/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_gpn_selected_gw1_gv2_gtm_symmetric_grp0.67_gmax5_gscope_positive_deficit_errors_gtmargin1_gttargetdelta_gtdf0.5_grisk1_gbal1_e14_selected_conservative_delta_risk/seed_${SEED}/fold_${FOLD}"
OUTPUT="$ROOT_OUT/summary.json"

[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E14 summary" >&2; exit 3; }
echo "=== E14 selected-transfer summary started: $(date) ==="
python -u tasks/summarize_noise_final_e14_selected_transfer.py \
  --control "$CONTROL" \
  --selected-no-margin "$NO_MARGIN" \
  --selected-delta "$DELTA" \
  --selected-delta-risk "$DELTA_RISK" \
  --outer-fold "$FOLD" \
  --output "$OUTPUT"
echo "=== E14 selected-transfer summary completed: $(date) ==="
