#!/bin/bash
#SBATCH --job-name=chem_action_view
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/chem_action_view_%j.out
#SBATCH --error=data/validation/chem_action_view_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACTION_BANK="data/validation/chemaware_direct_action_bank_v1"
test -f "$ACTION_BANK/report.json"
test -f "$ACTION_BANK/action_bank.npz"
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("data/validation/chemaware_direct_action_bank_v1/report.json").read_text())
if report.get("status") != "CHEMAWARE_DIRECT_ACTION_BANK_PASS" or not report.get("pass_to_direct_action_pilot"):
    raise SystemExit("BLOCKED: direct action bank did not pass its formula-disjoint causal gate")
PY

ROOT="data/validation/chemaware_direct_action_views_pilot/run_${SLURM_JOB_ID}"
[[ ! -e "$ROOT" ]] || { echo "Output collision: $ROOT" >&2; exit 3; }
mkdir -p "$ROOT"
trap 'echo "FAILED: ChemAware direct-action pilot stopped before completion" >&2' ERR

bash -n "$0"
python -m py_compile \
  tasks/train_chemaware_iceberg_direct_shared.py \
  tasks/chemaware_direct_training_core.py \
  tasks/chemaware_frozen_prefix_cache.py \
  tasks/chemaware_retrieval_graph.py \
  tasks/chemaware_shared_v2_core.py \
  tasks/summarize_chemaware_direct_action_views.py
python tasks/test_chemaware_direct_runtime_contracts.py
python tasks/test_chemaware_subset_paired_evaluation.py
python tasks/test_chemaware_direct_action_pilot_sbatch.py

COMMON_ARGS=(
  --training-objective direct_projected_guarded
  --action-bank "$ACTION_BANK" --action-scope bank_qualified
  --action-kind differential
  --inner-fold 3 --outer-fold 4 --fold-seed 20260935
  --action-generation-seed 20260935
  --epochs 2 --max-action-identities 512 --max-safety-identities 512 --batch-queries 4
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5
  --temperature 0.10 --lambda-clean-rank 1.0 --lambda-action-rank 1.0
  --safety-stream-weight 1.0 --lambda-consistency 0
  --lambda-margin-floor 2 --lambda-preserve 20 --lambda-peak-contrast 0
  --grad-clip 5 --maximum-action-gradient-ratio 0.25
  --eval-batch-size 32 --bootstrap-draws 10000
  --device cuda --no-amp --torch-threads 8
)

# Exercise the exact production arguments and all graph/teacher/action-bank/
# HDF5/embedding-cache contracts before constructing the 117M model.
python -u tasks/train_chemaware_iceberg_direct_shared.py \
  "${COMMON_ARGS[@]}" --arm correct_synthetic \
  --output "$ROOT/cpu_preflight" --preflight-only
test -f "$ROOT/cpu_preflight/preflight.json"

run_arm() {
  local arm="$1"
  local out="$ROOT/$arm"
  python -u tasks/train_chemaware_iceberg_direct_shared.py \
    "${COMMON_ARGS[@]}" --arm "$arm" --output "$out"
  test -f "$out/COMPLETE.json"
  test -f "$out/report.json"
  test -f "$out/inner_per_query.csv.gz"
  test -f "$out/final_shared_encoder.pt"
}

# Stage 1 is the cost gate: do not spend two additional model runs unless the
# correct action safely beats equal-budget clean continuation.
run_arm clean_duplicate
run_arm correct_synthetic
python -u tasks/summarize_chemaware_direct_action_views.py \
  --root "$ROOT" --output "$ROOT/stage1_decision.json" --stage primary
if ! python - "$ROOT/stage1_decision.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report.get("pass_to_matched_controls") is not True:
    raise SystemExit(10)
PY
then
  echo "STOPPED_EARLY: correct action did not beat clean continuation; controls not run"
  exit 0
fi

run_arm candidate_swapped
run_arm peak_permuted
python -u tasks/summarize_chemaware_direct_action_views.py \
  --root "$ROOT" --output "$ROOT/final_causal_summary.json" --stage full
