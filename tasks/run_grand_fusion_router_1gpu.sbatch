#!/bin/bash
#SBATCH --job-name=grand_fusion_router
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/grand_fusion_router_%j.out
#SBATCH --error=data/validation/grand_fusion_router_%j.err

# Frozen train-side routing followed by a single sealed GNPS application.
# It consumes frozen score caches and does not perform neural inference, but
# the E6 cluster submission policy requires every batch job to request a GPU.
# Required: EVIDENCE_DIR. Optional: SCORE_BUNDLE and GNPS_BENCHMARK.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

: "${EVIDENCE_DIR:?Set EVIDENCE_DIR to frozen MassSpecGym pair evidence}"
SCORE_BUNDLE="${SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
GNPS_BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
REQUIRE_FULL_EXPERTS="${REQUIRE_FULL_EXPERTS:-1}"
RUN_ROOT="data/validation/grand_fusion_router_run_${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT"

required=(
  "$EVIDENCE_DIR/evidence.npz" "$EVIDENCE_DIR/report.json" "$SCORE_BUNDLE"
  "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  tasks/grand_fusion_components_v1.json
  tasks/grand_fusion_router_core.py
  tasks/train_grand_fusion_router.py
  tasks/apply_grand_fusion_router_to_gnps.py
  tasks/extract_pair_score_cache_from_bundle.py
  tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py
  tasks/adjudicate_grand_fusion_external.py
  tasks/test_grand_fusion.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing grand-fusion input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/grand_fusion_router_core.py \
  tasks/train_grand_fusion_router.py \
  tasks/apply_grand_fusion_router_to_gnps.py \
  tasks/extract_pair_score_cache_from_bundle.py \
  tasks/adjudicate_grand_fusion_external.py
python -u tasks/test_grand_fusion.py

METHOD_ARGS=(
  --method noise_v1=v1_cosine
  --method weighted_spectral_entropy=weighted_entropy
  --method noise_p2b_v1=p2b_fused
  --method neutral_loss=neutral_loss_sqrt_cosine
  --method sqrt_cosine=sqrt_cosine
  --method spectral_entropy=entropy_similarity
  --method official_dreams=official_cosine
)

# ChemAware joins only when both train-side and frozen GNPS scores exist.
if python - "$EVIDENCE_DIR/evidence.npz" "$SCORE_BUNDLE" <<'PY'
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as e:
    train_ok = "chemaware_v2_reranker" in e.files
with np.load(sys.argv[2], allow_pickle=False) as b:
    external_ok = "chemaware_v2_reranker" in set(map(str, b["method_names"]))
raise SystemExit(0 if train_ok and external_ok else 1)
PY
then
  METHOD_ARGS+=(--method chemaware_v2_reranker=chemaware_v2_reranker)
  echo "ChemAware V2 entered the preregistered router."
else
  if [[ "$REQUIRE_FULL_EXPERTS" == "1" ]]; then
    echo "ChemAware V2 is missing from one aligned side; refusing incomplete grand fusion." >&2
    exit 8
  fi
  echo "ChemAware V2 not aligned on both sides; running an explicitly partial router."
fi

# The candidate-level MSG stack enters only through formula-OOF train scores
# and its once-frozen GNPS cache.  Missing either side disables the expert.
if python - "$EVIDENCE_DIR/evidence.npz" "$SCORE_BUNDLE" <<'PY'
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=False) as e:
    train_ok = "noise_msg_fusion_stack_v1" in e.files
with np.load(sys.argv[2], allow_pickle=False) as b:
    external_ok = "noise_msg_fusion_stack_v1" in set(map(str, b["method_names"]))
raise SystemExit(0 if train_ok and external_ok else 1)
PY
then
  METHOD_ARGS+=(--method noise_msg_fusion_stack_v1=noise_msg_fusion_stack_v1)
  echo "MSG candidate stack entered through formula-OOF predictions."
else
  if [[ "$REQUIRE_FULL_EXPERTS" == "1" ]]; then
    echo "MSG candidate stack is missing from one aligned side; refusing incomplete grand fusion." >&2
    exit 9
  fi
  echo "MSG candidate stack missing an aligned side; running an explicitly partial router."
fi

python -u tasks/train_grand_fusion_router.py \
  --evidence "$EVIDENCE_DIR" \
  --output "$RUN_ROOT/router" \
  "${METHOD_ARGS[@]}"

python -u tasks/apply_grand_fusion_router_to_gnps.py \
  --benchmark "$GNPS_BENCHMARK" \
  --score-bundle "$SCORE_BUNDLE" \
  --router "$RUN_ROOT/router" \
  --output "$RUN_ROOT/gnps_router_scores"

for method in weighted_spectral_entropy p2b_noise_v1_frozen noise_v1; do
  python -u tasks/extract_pair_score_cache_from_bundle.py \
    --benchmark "$GNPS_BENCHMARK" --score-bundle "$SCORE_BUNDLE" \
    --method "$method" --output "$RUN_ROOT/${method}_cache"
  python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
    --benchmark "$GNPS_BENCHMARK" \
    --baseline-score-cache "$RUN_ROOT/${method}_cache" \
    --candidate-score-cache "$RUN_ROOT/gnps_router_scores" \
    --output "$RUN_ROOT/router_vs_${method}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261004
done

python -u tasks/adjudicate_grand_fusion_external.py \
  --mode router \
  --wse "$RUN_ROOT/router_vs_weighted_spectral_entropy/report.json" \
  --p2b "$RUN_ROOT/router_vs_p2b_noise_v1_frozen/report.json" \
  --noise "$RUN_ROOT/router_vs_noise_v1/report.json" \
  --output "$RUN_ROOT/adjudication.json"

sha256sum \
  "$RUN_ROOT/router/router.joblib" \
  "$RUN_ROOT/router/report.json" \
  "$RUN_ROOT/gnps_router_scores/report.json" \
  "$RUN_ROOT/gnps_router_scores/routing_ledger_identity_disjoint.csv" \
  "$RUN_ROOT/gnps_router_scores/routing_ledger_formula_disjoint.csv" \
  "$RUN_ROOT/gnps_router_scores/routing_ledger_manifest.json" \
  "$RUN_ROOT/adjudication.json" > "$RUN_ROOT/SHA256SUMS.txt"

echo "Grand-fusion router complete: $RUN_ROOT"
