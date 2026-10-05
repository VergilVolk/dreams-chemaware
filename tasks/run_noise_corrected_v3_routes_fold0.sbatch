#!/bin/bash
#SBATCH --job-name=noise_v3route_f0
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_v3route_f0_%j.out
#SBATCH --error=data/validation/noise_v3route_f0_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "Expected two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 2
fi
N_A4_GPU="${ALLOCATED_GPUS[0]}"
P_GPU="${ALLOCATED_GPUS[1]}"

FOLD=0
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
A4_SCAN="data/validation/g8r_noise_v3_a4_exact_peak_scan"

FINAL_ROOT="data/validation/noise_corrected_npa4_formal_fold_${FOLD}_run_${SLURM_JOB_ID}_v3routes"
RUN_ROOT="data/validation/.noise_corrected_npa4_formal_fold_${FOLD}_run_${SLURM_JOB_ID}_v3routes.staging"
N_BANK="$RUN_ROOT/routes/n_bank"
N_ROUTE="$RUN_ROOT/routes/n_route"
P_ROUTE="$RUN_ROOT/routes/p_route"
A4_ROUTE="$RUN_ROOT/routes/a4_route"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json" "$DATA" "$OFFICIAL" \
  "$ARCHITECTURE" "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$A4_SCAN/exact_peak_scan.h5" "$A4_SCAN/scan_queries.csv.gz" \
  "$A4_SCAN/decision.json"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen v3 route input: $REQUIRED" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique atomic route run" >&2
  exit 4
}
mkdir -p "$RUN_ROOT/routes"

python -m py_compile \
  tasks/noise_corrected_action_panel.py \
  tasks/noise_corrected_action_routing.py \
  tasks/noise_corrected_action_routing_v3.py \
  tasks/noise_corrected_full_action_registry.py \
  tasks/audit_noise_final_e10_positive_residual_matrix.py \
  tasks/build_noise_corrected_full_action_bank.py \
  tasks/audit_noise_corrected_n_router.py \
  tasks/audit_noise_corrected_full_p_router.py \
  tasks/audit_noise_corrected_a4_full_router.py \
  tasks/build_noise_corrected_routed_action_ledger.py \
  tasks/test_noise_corrected_n_router.py \
  tasks/test_noise_corrected_full_action_bank.py \
  tasks/test_noise_corrected_direct_v3_bundle.py
python -u tasks/test_noise_corrected_direct_v3_bundle.py
python -u tasks/test_noise_corrected_action_routing.py
python -u tasks/test_noise_corrected_action_routing_v3.py
python -u tasks/test_noise_corrected_action_panel.py
python -u tasks/test_noise_corrected_routed_ledger_streaming.py
python -u tasks/test_noise_corrected_full_action_bank.py
python -u tasks/test_noise_final_e10_positive_residual_matrix.py
python -u tasks/test_noise_corrected_full_action_registry.py
python -u tasks/test_noise_corrected_n_router.py
python -u tasks/test_noise_corrected_full_p_router.py
python -u tasks/test_noise_corrected_a4_full_router.py

echo "=== GPU0: mature N construction/routing, then chunked A4 ==="
(
  export CUDA_VISIBLE_DEVICES="$N_A4_GPU" OMP_NUM_THREADS=8
  python -u tasks/build_noise_corrected_full_action_bank.py \
    --graph-dir "$GRAPH" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --formula-fold-seed 20260825 --top-k-negatives 5 \
    --softmax-temperature 0.10 --fragment-tolerance 0.02 \
    --control-repeats 2 --seed 20260906 --n-highest-peaks 100 \
    --selection-batch-size 32 --encode-batch-size 256 \
    --max-queries 0 --device cuda --no-amp --output-dir "$N_BANK"
  python -u tasks/audit_noise_corrected_n_router.py \
    --graph-dir "$GRAPH" --action-bank-dir "$N_BANK" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --paired-advantage-threshold 0.01 --harm-margin-threshold 0.01 \
    --robustness-slack 0.005 --n-highest-peaks 100 \
    --query-chunk-size 128 --maximum-corrective-frontier 16 \
    --maximum-harmful-frontier 8 --maximum-robust-frontier 8 \
    --batch-size 256 --device cuda --no-amp --output-dir "$N_ROUTE"
  python -u tasks/audit_noise_corrected_a4_full_router.py \
    --graph-dir "$GRAPH" --a4-scan-dir "$A4_SCAN" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --formula-fold-seed 20260825 --query-scope initial_error_boundary \
    --max-queries 0 --formal --sample-seed 20260906 \
    --maximum-gradient-rank 50 --control-repeats 2 \
    --paired-advantage-threshold 0.01 --harm-margin-threshold 0.01 \
    --robustness-slack 0.005 --n-highest-peaks 100 \
    --query-chunk-size 16 --boundary-correct-multiplier 2 \
    --minimum-boundary-correct 2048 --maximum-boundary-correct 4096 \
    --maximum-corrective-frontier 16 --maximum-harmful-frontier 8 \
    --maximum-robust-frontier 8 \
    --batch-size 256 --device cuda --no-amp --output-dir "$A4_ROUTE"
) &
N_A4_PID=$!

echo "=== GPU1: chunked complete P66 ==="
(
  export CUDA_VISIBLE_DEVICES="$P_GPU" OMP_NUM_THREADS=8
  python -u tasks/audit_noise_corrected_full_p_router.py \
    --graph-dir "$GRAPH" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --formula-fold-seed 20260825 --query-scope initial_error_boundary \
    --max-queries 0 --formal --sample-seed 20260906 \
    --fragment-tolerance 0.02 --paired-advantage-threshold 0.01 \
    --harm-margin-threshold 0.01 --robustness-slack 0.005 \
    --n-highest-peaks 100 --query-chunk-size 64 \
    --boundary-correct-multiplier 2 --minimum-boundary-correct 4096 \
    --maximum-boundary-correct 8192 --maximum-corrective-frontier 16 \
    --maximum-harmful-frontier 8 --maximum-robust-frontier 8 \
    --batch-size 256 --device cuda --no-amp --output-dir "$P_ROUTE"
) &
P_PID=$!

N_A4_STATUS=0
P_STATUS=0
wait "$N_A4_PID" || N_A4_STATUS=$?
wait "$P_PID" || P_STATUS=$?
if (( N_A4_STATUS != 0 || P_STATUS != 0 )); then
  echo "v3 route pipelines failed: N+A4=$N_A4_STATUS P=$P_STATUS" >&2
  exit 5
fi

python - "$N_ROUTE/report.json" "$P_ROUTE/report.json" "$A4_ROUTE/report.json" <<'PY'
import json, sys
reports = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
expected = {
    "noise_corrected_n_router_audit_complete",
    "noise_corrected_full_p_router_audit_complete",
    "noise_corrected_a4_router_audit_complete",
}
assert {report["status"] for report in reports} == expected
assert len({
    report["provenance"]["action_routing_v3_sha256"] for report in reports
}) == 1
assert all(report["formal_training_authorized"] is True for report in reports)
for report in reports:
    assert report["bounded_memory"]["monolithic_all_action_embedding_allocation"] is False
    assert report["contracts"]["selector_frontier_is_lossless_for_global_caps"] is True
    assert report["contracts"]["registered_formal_route_configuration_verified"] is True
    assert report["contracts"]["exact_action_control_candidate_switch_rows_recorded"] is True
    if report["status"] == "noise_corrected_full_p_router_audit_complete":
        assert report["recipes"] == 66
        assert report["recipe_source_counts"] == {
            "P_guided_original": 6, "E10B": 19, "E11": 16, "E12B": 25
        }
        assert report["contracts"]["all_66_unique_mature_P_cells_materialized"] is True
        assert report["contracts"][
            "original_12_cell_P_intensity_matrix_complete_without_duplicates"
        ] is True
        assert report["contracts"][
            "reference_profiles_cached_per_query_policy_direction"
        ] is True
        assert report["contracts"][
            "missing_peaks_cached_per_query_policy_direction_parameters"
        ] is True
        assert report["contracts"][
            "exact_action_tensor_deduplication_before_encoder_is_lossless"
        ] is True
    assert report["contracts"]["control_semantics_explicit"] is True
    assert report["selector_frontier"] == {
        "maximum_corrective_per_query": 16,
        "maximum_harmful_per_query": 8,
        "maximum_robust_per_query": 8,
    }
for report in reports[1:]:
    assert report["action_panel"]["selection"].startswith("all_initial_E8_errors")
print("[v3 formal route gate] PASS")
PY

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== formal v3 routes complete: $FINAL_ROOT ==="
if [[ "${AUTO_SUBMIT_TRAINING:-1}" == "1" ]]; then
  NEXT_JOB=$(sbatch --parsable --export=ALL,BASE_RUN="$FINAL_ROOT" \
    tasks/run_noise_corrected_direct_v3_from_formal_routes.sbatch)
  echo "=== submitted direct-v3 training job: $NEXT_JOB ==="
else
  echo "BASE_RUN=$FINAL_ROOT sbatch tasks/run_noise_corrected_direct_v3_from_formal_routes.sbatch"
fi
