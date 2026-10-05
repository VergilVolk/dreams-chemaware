#!/bin/bash
#SBATCH --job-name=noise_npa4_f0
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_npa4_f0_%j.out
#SBATCH --error=data/validation/noise_npa4_f0_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Preserve the two physical GPU identifiers assigned by Slurm.  They are not
# necessarily 0 and 1 on a busy node, so never overwrite the allocation blindly.
ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} < 2 )); then
  echo "Expected two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 2
fi
ROUTED_GPU="${ALLOCATED_GPUS[0]}"
CONTROL_GPU="${ALLOCATED_GPUS[1]}"
echo "Using allocated GPUs: routed/N=$ROUTED_GPU control/P+A4=$CONTROL_GPU"

FOLD=0
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
# This manifest supplies frozen query-adduct metadata for evaluation only.  No
# chem-aware supervision, target or loss is consumed by the noise trainer.
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
A4_SCAN="data/validation/g8r_noise_v3_a4_exact_peak_scan"

FINAL_ROOT="data/validation/noise_corrected_npa4_formal_fold_${FOLD}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_corrected_npa4_formal_fold_${FOLD}_run_${SLURM_JOB_ID}.staging"
N_BANK="$RUN_ROOT/routes/n_bank"
N_ROUTE="$RUN_ROOT/routes/n_route"
P_ROUTE="$RUN_ROOT/routes/p_route"
A4_ROUTE="$RUN_ROOT/routes/a4_route"
LEDGER="$RUN_ROOT/ledger"
ARMS="$RUN_ROOT/arms"
SUMMARIES="$RUN_ROOT/summaries"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/official_embeddings.npz" "$GRAPH/report.json" \
  "$EVAL_METADATA/manifest.npz" "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$A4_SCAN/exact_peak_scan.h5" "$A4_SCAN/scan_queries.csv.gz" "$A4_SCAN/decision.json"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen N+P+A4 input: $REQUIRED" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique atomic run: $RUN_ROOT or $FINAL_ROOT" >&2
  exit 4
}
mkdir -p "$RUN_ROOT/routes" "$ARMS" "$SUMMARIES"

python -m py_compile \
  tasks/noise_corrected_fullgraph_evaluation.py \
  tasks/noise_corrected_full_action_registry.py \
  tasks/noise_corrected_action_routing.py \
  tasks/noise_final_direct_boundary_v2_core.py \
  tasks/build_noise_corrected_full_action_bank.py \
  tasks/audit_noise_corrected_n_router.py \
  tasks/audit_noise_corrected_full_p_router.py \
  tasks/audit_noise_corrected_a4_full_router.py \
  tasks/build_noise_corrected_routed_action_ledger.py \
  tasks/train_noise_corrected_routed_direct.py \
  tasks/summarize_noise_corrected_routed_direct_arms.py

python -u tasks/test_noise_corrected_fullgraph_evaluation.py
python -u tasks/test_noise_corrected_full_action_registry.py
python -u tasks/test_noise_corrected_full_action_bank.py
python -u tasks/test_noise_corrected_n_router.py
python -u tasks/test_noise_corrected_full_p_router.py
python -u tasks/test_noise_corrected_a4_full_router.py
python -u tasks/test_noise_corrected_action_routing.py
python -u tasks/test_noise_final_direct_boundary_v2_core.py
python -u tasks/test_summarize_noise_corrected_routed_direct_arms.py

echo "=== parallel stages 1-4/6: GPU0=N; GPU1=P60+A4 ==="
(
  export CUDA_VISIBLE_DEVICES="$ROUTED_GPU" OMP_NUM_THREADS=8
  echo "=== GPU0 stage 1/6: full mature N action construction ==="
  python -u tasks/build_noise_corrected_full_action_bank.py \
    --graph-dir "$GRAPH" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --formula-fold-seed 20260825 --selection-batch-size 32 --encode-batch-size 256 \
    --max-queries 0 --device cuda --no-amp --output-dir "$N_BANK"

  echo "=== GPU0 stage 2/6: current-E8 routing of every mature N path ==="
  python -u tasks/audit_noise_corrected_n_router.py \
    --graph-dir "$GRAPH" --action-bank-dir "$N_BANK" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --batch-size 256 --device cuda --no-amp --output-dir "$N_ROUTE"
) &
N_PIPELINE_PID=$!

(
  export CUDA_VISIBLE_DEVICES="$CONTROL_GPU" OMP_NUM_THREADS=8
  echo "=== GPU1 stage 3/6: all 60 mature P cells in current E8 geometry ==="
  python -u tasks/audit_noise_corrected_full_p_router.py \
    --graph-dir "$GRAPH" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --formula-fold-seed 20260825 --query-scope all --max-queries 0 --formal \
    --batch-size 256 --device cuda --no-amp --output-dir "$P_ROUTE"

  echo "=== GPU1 stage 4/6: every outcome-free A4 Top-50 proposal x four doses ==="
  python -u tasks/audit_noise_corrected_a4_full_router.py \
    --graph-dir "$GRAPH" --a4-scan-dir "$A4_SCAN" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD" \
    --formula-fold-seed 20260825 --query-scope all --max-queries 0 --formal \
    --maximum-gradient-rank 50 --control-repeats 2 \
    --batch-size 256 --device cuda --no-amp --output-dir "$A4_ROUTE"
) &
PA4_PIPELINE_PID=$!

N_PIPELINE_STATUS=0
PA4_PIPELINE_STATUS=0
wait "$N_PIPELINE_PID" || N_PIPELINE_STATUS=$?
wait "$PA4_PIPELINE_PID" || PA4_PIPELINE_STATUS=$?
if (( N_PIPELINE_STATUS != 0 || PA4_PIPELINE_STATUS != 0 )); then
  echo "Parallel routing failed: N=$N_PIPELINE_STATUS P+A4=$PA4_PIPELINE_STATUS" >&2
  exit 5
fi

echo "=== stage 5/6: source/family-diverse query-complete training ledger ==="
python -u tasks/build_noise_corrected_routed_action_ledger.py \
  --graph-dir "$GRAPH" --route-dirs "$N_ROUTE" "$P_ROUTE" "$A4_ROUTE" \
  --maximum-corrective-per-query 16 --maximum-risk-per-query 8 \
  --output-dir "$LEDGER"

python - "$LEDGER/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["status"] == "noise_corrected_routed_action_ledger_complete"
assert report["formal_training_authorized"] is True
assert report["contracts"]["multiple_corrective_actions_preserved"] is True
assert report["contracts"]["teacher_embedding_target_used"] is False
assert report["contracts"]["P3_consumed"] is False
assert {key.split("|")[0] for key in report["source_kind_rows"]} >= {
    "N_mature", "E10B", "E11", "E12B", "A4_exact",
}
PY

echo "=== stage 6/6: paired full-graph direct and clean-control training ==="
COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$EVAL_METADATA"
  --routed-ledger-dir "$LEDGER" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --corrective-objective-mode paired_margin_only
  --epochs 4 --batch-queries 4 --positive-references 3
  --negative-molecules 8 --references-per-negative 2
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4
  --rank-margin 0.05 --temperature 0.10 --advantage-temperature 0.05
  --topk-negatives 8 --action-safety-slack 0.005
  --lambda-margin-transfer 1.0 --lambda-clean-continuation 1.0
  --lambda-risk-floor 1.0 --lambda-preserve 1.0 --risk-margin-slack 0.005
  --calibration-batches 32 --target-corrective-to-risk-ratio 1.0
  --corrective-scale-cap 256 --minimum-corrective-scale 0.5
  --target-preclip-gradient-norm 0.05 --minimum-action-retention-p10 0.50
  --maximum-clip-event-fraction 0.10 --grad-clip 2.0
  --eval-batch-size 256 --bootstrap-resamples 10000 --device cuda --no-amp
)

SEEDS=(20260906 20260907 20260908)
for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="$ARMS/seed_${SEED}"
  echo "=== seed $SEED: GPU0 routed-direct and GPU1 clean-control in parallel ==="
  (
    export CUDA_VISIBLE_DEVICES="$ROUTED_GPU" OMP_NUM_THREADS=8
    python -u tasks/train_noise_corrected_routed_direct.py \
      "${COMMON[@]}" --seed "$SEED" --arm routed_direct \
      --output-dir "$SEED_ROOT/routed_direct"
  ) &
  ROUTED_PID=$!
  (
    export CUDA_VISIBLE_DEVICES="$CONTROL_GPU" OMP_NUM_THREADS=8
    python -u tasks/train_noise_corrected_routed_direct.py \
      "${COMMON[@]}" --seed "$SEED" --arm clean_control \
      --output-dir "$SEED_ROOT/clean_control"
  ) &
  CONTROL_PID=$!

  ROUTED_STATUS=0
  CONTROL_STATUS=0
  wait "$ROUTED_PID" || ROUTED_STATUS=$?
  wait "$CONTROL_PID" || CONTROL_STATUS=$?
  if (( ROUTED_STATUS != 0 || CONTROL_STATUS != 0 )); then
    echo "Seed $SEED paired training failed: routed=$ROUTED_STATUS control=$CONTROL_STATUS" >&2
    exit 6
  fi
  python -u tasks/summarize_noise_corrected_routed_direct_arms.py \
    --graph-dir "$GRAPH" --routed-dir "$SEED_ROOT/routed_direct" \
    --control-dir "$SEED_ROOT/clean_control" --minimum-causal-queries 32 \
    --bootstrap-resamples 10000 --seed "$SEED" \
    --output-dir "$SUMMARIES/seed_${SEED}"
done

python - "$ARMS" "$SUMMARIES" "$RUN_ROOT/promotion_gate.json" <<'PY'
import json, pathlib, sys

arms = pathlib.Path(sys.argv[1])
summaries = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
seeds = (20260906, 20260907, 20260908)
records = []
for seed in seeds:
    routed = json.loads((arms / f"seed_{seed}" / "routed_direct" / "decision.json").read_text())
    control = json.loads((arms / f"seed_{seed}" / "clean_control" / "decision.json").read_text())
    paired = json.loads((summaries / f"seed_{seed}" / "report.json").read_text())
    held = routed["evaluation"]["formal_held_graph"]
    candidate = held["candidate"]
    initial = held["initial_E8"]
    outcome = held["candidate_vs_initial_E8"]
    retrieval = candidate["retrieval"]
    initial_retrieval = initial["retrieval"]
    outer_control = paired["panel_comparisons"]["outer_held"]
    r1_delta_pp = 100.0 * (retrieval["recall@1"] - initial_retrieval["recall@1"])
    nonregression_higher = [
        retrieval[f"recall@{k}"] >= initial_retrieval[f"recall@{k}"]
        for k in (2, 3, 5, 10, 20)
    ] + [
        retrieval["mrr"] >= initial_retrieval["mrr"],
        retrieval["macro_query_auroc"] >= initial_retrieval["macro_query_auroc"],
        retrieval["macro_query_auprc"] >= initial_retrieval["macro_query_auprc"],
        candidate["micro_candidate"]["auroc"] >= initial["micro_candidate"]["auroc"],
        candidate["micro_candidate"]["auprc"] >= initial["micro_candidate"]["auprc"],
        candidate["massspecgym_mh_10ppm_pooled_pairwise"]["auroc"]
        >= initial["massspecgym_mh_10ppm_pooled_pairwise"]["auroc"],
    ]
    gates = {
        "formal": routed["formal"] is True and control["formal"] is True,
        "signal": routed["signal_transmission"]["passed"] is True,
        "clip": routed["signal_transmission"]["clip_gate_passed"] is True,
        "e8_recall1_delta_ge4pp": r1_delta_pp >= 4.0,
        "e8_formula_ci_positive": outcome["formula_cluster_delta_recall1"]["ci_low_pp"] > 0,
        "e8_near_formula_ci_positive": outcome["near_formula_cluster_delta_recall1"]["ci_low_pp"] > 0,
        "control_formula_ci_positive": outer_control["formula_cluster_delta_recall1"]["ci_low_pp"] > 0,
        "control_near_formula_ci_positive": (
            outer_control["near_formula_cluster_delta_recall1"] is not None
            and outer_control["near_formula_cluster_delta_recall1"]["ci_low_pp"] > 0
        ),
        "broad_metrics_nonregressive": all(nonregression_higher),
        "risk_net_positive": outcome["risk_net_lambda2"] > 0,
    }
    records.append({
        "seed": seed, "r1_delta_vs_E8_pp": r1_delta_pp,
        "gates": gates, "passed": all(gates.values()),
    })
report = {
    "status": "noise_corrected_npa4_formal_promotion_gate_complete",
    "seeds": records,
    "all_three_seeds_passed": all(item["passed"] for item in records),
    "promotion_authorized": all(item["passed"] for item in records),
    "claim_limit": "Fold-0 promotion gate only; multifold is not authorized unless true.",
}
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== formal N+P+A4 fold-0 result preserved at: $FINAL_ROOT ==="
