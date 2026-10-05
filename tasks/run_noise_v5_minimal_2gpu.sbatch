#!/bin/bash
#SBATCH --job-name=noise_v5_min
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_v5_minimal_%j.out
#SBATCH --error=data/validation/noise_v5_minimal_%j.err

# Pre-registered V5 minimal residual round, GNPS-certified.
#
# One training unit per atlas-selected V1-current residual query
# (selected_relations.npz): anchor = targeted action view (treatment arm) or
# the matched control view (control arm), positive = exact positive row,
# negatives = exact negative row + current molecule-max top-rival winner row.
# No clean stream exists, so the shared-clean drift that killed v2/v3 cannot
# occur by construction.  Promotion is decided ONLY by the pre-registered
# gates in summarize_noise_v5_minimal.py: (G1) both arms' development recall
# must not fall below the atlas V1 baseline, (G2) targeted vs V1 recall@1
# delta >= 0 on BOTH GNPS panels, (G3) targeted vs control recall@1 delta
# >= 0 on BOTH GNPS panels.  No MassSpecGym held fold-0 evaluation runs in
# this job, and GNPS spectra never enter training.
#
# Submit with the atlas run directory exported:
#   ATLAS_RUN=data/validation/noise_v1_current_residual_atlas_run_<jobid> \
#     sbatch tasks/run_noise_v5_minimal_2gpu.sbatch
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
V1_SLIM="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
ACTION_BANK="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820/triplets/action_spectra.npz"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
BASELINE_RUN="${BASELINE_RUN:-data/validation/gnps_absolute_baselines_run_2347472}"
: "${ATLAS_RUN:?export ATLAS_RUN=data/validation/noise_v1_current_residual_atlas_run_<jobid>}"
RUN_ROOT="data/validation/noise_v5_minimal_run_${SLURM_JOB_ID}"
GNPS="$RUN_ROOT/gnps"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_v5_minimal_${SLURM_JOB_ID}"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$GNPS" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$ARCHITECTURE" "$V1_SLIM" "$GRAPH" "$DATA" "$ACTION_BANK"
  "$GNPS_BENCHMARK/spectra.mgf"
  "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  "$ATLAS_RUN/atlas/selected_relations.npz"
  "$ATLAS_RUN/atlas/report.json"
  "$BASELINE_RUN/gnps/v1_embeddings.npz"
  tasks/train_noise_v5_minimal.py
  tasks/score_noise_dev_graph_recall.py
  tasks/summarize_noise_v5_minimal.py
  tasks/encode_noise_corrected_graph_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/noise_v1_current_residual_atlas_core.py
  tasks/build_noise_v1_current_residual_atlas.py
  tasks/test_noise_v5_minimal.py
  tasks/test_noise_v5_minimal_sbatch.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing V5 minimal input: $path" >&2; exit 5; }
done

python -u tasks/test_noise_v5_minimal.py
python -u tasks/test_noise_v5_minimal_sbatch.py

python - "$ATLAS_RUN" "$GRAPH" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

atlas_run, graph_path = sys.argv[1], sys.argv[2]
report = json.loads(
    (Path(atlas_run) / "atlas" / "report.json").read_text(encoding="utf-8"),
)
digest = hashlib.sha256()
with open(graph_path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
if report.get("provenance", {}).get("graph_sha256") != digest.hexdigest():
    raise SystemExit("atlas provenance names a different candidate graph")
if int(report.get("action_replay", {}).get("selected_current_residual_events", 0)) <= 0:
    raise SystemExit("atlas selected ledger is empty")
print("[atlas-provenance] OK")
PYEOF

run_arm() {
  local label="$1"
  local gpu="$2"
  local arm_dir="$LOCAL_ROOT/arm_${label}"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_v5_minimal.py \
    --data "$DATA" \
    --selected-relations "$ATLAS_RUN/atlas/selected_relations.npz" \
    --action-bank "$ACTION_BANK" \
    --warm-start-checkpoint "$V1_SLIM" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --arm "$label" \
    --output "$arm_dir" \
    >"$RUN_ROOT/train_${label}.out" 2>"$RUN_ROOT/train_${label}.err"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_noise_corrected_graph_checkpoint.py \
    --graph "$GRAPH" --data "$DATA" \
    --checkpoint "$arm_dir/final_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$LOCAL_ROOT/graph_${label}.npz" --batch-size 64 \
    >"$RUN_ROOT/graph_encode_${label}.out" 2>"$RUN_ROOT/graph_encode_${label}.err"
  python -u tasks/score_noise_dev_graph_recall.py \
    --graph "$GRAPH" \
    --embedding-cache "$LOCAL_ROOT/graph_${label}.npz" \
    --output "$RUN_ROOT/dev_recall_${label}.json" \
    >"$RUN_ROOT/dev_score_${label}.out" 2>"$RUN_ROOT/dev_score_${label}.err"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$arm_dir/final_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$GNPS/${label}_v5_embeddings.npz" --batch-size 64 \
    --label "v5_${label}" \
    >"$RUN_ROOT/gnps_encode_${label}.out" 2>"$RUN_ROOT/gnps_encode_${label}.err"
}

run_arm targeted 0 >"$RUN_ROOT/arm_targeted.log" 2>&1 &
TARGET_PID=$!
run_arm control 1 >"$RUN_ROOT/arm_control.log" 2>&1 &
CONTROL_PID=$!
wait "$TARGET_PID"
wait "$CONTROL_PID"

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings "$BASELINE_RUN/gnps/v1_embeddings.npz" \
  --candidate-embeddings "$GNPS/targeted_v5_embeddings.npz" \
  --output "$GNPS/paired_v1_vs_targeted_v5" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261002 \
  >"$RUN_ROOT/eval_targeted_vs_v1.out" 2>"$RUN_ROOT/eval_targeted_vs_v1.err"
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings "$BASELINE_RUN/gnps/v1_embeddings.npz" \
  --candidate-embeddings "$GNPS/control_v5_embeddings.npz" \
  --output "$GNPS/paired_v1_vs_control_v5" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261002 \
  >"$RUN_ROOT/eval_control_vs_v1.out" 2>"$RUN_ROOT/eval_control_vs_v1.err"
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings "$GNPS/targeted_v5_embeddings.npz" \
  --candidate-embeddings "$GNPS/control_v5_embeddings.npz" \
  --output "$GNPS/paired_targeted_v5_vs_control_v5" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261002 \
  >"$RUN_ROOT/eval_targeted_vs_control.out" 2>"$RUN_ROOT/eval_targeted_vs_control.err"

python -u tasks/summarize_noise_v5_minimal.py \
  --targeted-report "$LOCAL_ROOT/arm_targeted/report.json" \
  --control-report "$LOCAL_ROOT/arm_control/report.json" \
  --atlas-report "$ATLAS_RUN/atlas/report.json" \
  --dev-targeted "$RUN_ROOT/dev_recall_targeted.json" \
  --dev-control "$RUN_ROOT/dev_recall_control.json" \
  --gnps-targeted-vs-v1 "$GNPS/paired_v1_vs_targeted_v5/report.json" \
  --gnps-control-vs-v1 "$GNPS/paired_v1_vs_control_v5/report.json" \
  --gnps-targeted-vs-control "$GNPS/paired_targeted_v5_vs_control_v5/report.json" \
  --output "$RUN_ROOT/verdict.json"

mkdir -p "$RUN_ROOT/targeted" "$RUN_ROOT/control"
cp "$LOCAL_ROOT/arm_targeted/final_slim.pt" "$LOCAL_ROOT/arm_targeted/final_optimizer.pt" \
  "$LOCAL_ROOT/arm_targeted/report.json" "$RUN_ROOT/targeted/"
cp "$LOCAL_ROOT/arm_control/final_slim.pt" "$LOCAL_ROOT/arm_control/final_optimizer.pt" \
  "$LOCAL_ROOT/arm_control/report.json" "$RUN_ROOT/control/"

echo "=== V5 minimal round complete: $RUN_ROOT (verdict: $RUN_ROOT/verdict.json) ==="
