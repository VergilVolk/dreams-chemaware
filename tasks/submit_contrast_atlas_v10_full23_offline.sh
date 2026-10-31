#!/usr/bin/env bash
# Submit the frozen Atlas Full23 run from a git-free, isolated bundle.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="${ATLAS_SERVER_ASSET_ROOT:-/data02/run01/scv7tsl/DreaMS}"
RUNNER="$HERE/tasks/run_contrast_atlas_v10_complete_axis_full23_1gpu.sbatch"

ARCH="$BASE/dreams/models/pretrained/ssl_model_server.pt"
OFFICIAL="$BASE/data/e1/official_embedding_slim.pt"
NOISE="$BASE/data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
STAGE1="$BASE/data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
PHASEA="$BASE/data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
FULLMSG="$BASE/data/validation/noise_chemaware_full_massspecgym_native/run_2401890/checkpoint/noise_chemaware_fullmsg_slim.pt"
MSG_ROOT="$BASE/data/validation/noise_msg_fusion_run_2349961/fusion_model"

required=(
  "$RUNNER"
  "$ARCH"
  "$OFFICIAL"
  "$NOISE"
  "$STAGE1"
  "$PHASEA"
  "$FULLMSG"
  "$MSG_ROOT/fusion_model.joblib"
  "$MSG_ROOT/report.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || {
    echo "ATLAS_V10_OFFLINE_REQUIRED_ASSET_MISSING: $path" >&2
    exit 4
  }
done

export ATLAS_V10_REPO_ROOT="$HERE"
export DREAMS_ARCHITECTURE_CHECKPOINT="$ARCH"
export OFFICIAL_DREAMS_CHECKPOINT="$OFFICIAL"
export NOISE_V1_CHECKPOINT="$NOISE"
export CHEMAWARE_STAGE1_CHECKPOINT="$STAGE1"
export CHEMAWARE_PHASEA_CHECKPOINT="$PHASEA"
export NOISE_CHEMAWARE_FULLMSG_CHECKPOINT="$FULLMSG"
export NOISE_MSG_FUSION_MODEL="$MSG_ROOT"

cd "$HERE"
job_id="$(sbatch --parsable --export=ALL "$RUNNER")"
echo "ATLAS_V10_FULL23_OFFLINE_SUBMITTED: $job_id"
echo "ATLAS_V10_FULL23_OFFLINE_ROOT: $HERE"
