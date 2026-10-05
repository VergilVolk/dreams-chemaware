#!/bin/bash
#SBATCH --job-name=noise_fusion_5pp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_fusion_5pp_%j.out
#SBATCH --error=data/validation/noise_fusion_5pp_%j.err

# The 5pp shot: formula-isolated pair-evidence fusion ranker, trained on
# MassSpecGym only, applied once to the sealed GNPS panels.
#
# Chain: encode official+V1 MassSpecGym embedding shards -> build the
# corrected-graph pair-evidence table (7 deployment-visible scores per edge,
# pinned ms_entropy) -> train the HistGradientBoosting fusion with
# formula-disjoint 5-fold OOF and freeze it with joblib -> assemble the same
# features on GNPS from frozen artifacts (article bundle + embedding caches)
# -> score every sealed edge once -> evaluate the fused pair-score cache
# against official DreaMS and Noise V1 with the standard evaluator.
# GNPS never enters training, freezing, or any selection decision.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_DEPS="$LOCAL_BASE/noise_fusion_deps_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DEPS"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"pinned ms_entropy wheel requires Python 3.11, got {sys.version.split()[0]}"
    )
PY
python -m pip install --no-deps --target "$LOCAL_DEPS" \
  third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"

GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
OFFICIAL_MSG="data/e1/official_embedding_slim.pt"
V1="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
GNPS_BASELINE_RUN="${GNPS_BASELINE_RUN:-data/validation/gnps_absolute_baselines_run_2347472}"
OFFICIAL_GNPS="$GNPS_BASELINE_RUN/gnps/official_embeddings.npz"
V1_GNPS="$GNPS_BASELINE_RUN/gnps/v1_embeddings.npz"
ARTICLE_RUN="${ARTICLE_RUN:-data/validation/noise_gnps_article_benchmark_run_2349091}"
SCORE_BUNDLE="$ARTICLE_RUN/method_scores.npz"
RUN_ROOT="data/validation/noise_msg_fusion_run_${SLURM_JOB_ID}"
LOCAL_ROOT="$LOCAL_BASE/noise_msg_fusion_${SLURM_JOB_ID}"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required=(
  "$GRAPH" "$DATA" "$ARCHITECTURE" "$OFFICIAL_MSG" "$V1"
  "$GNPS_BENCHMARK/spectra.mgf" "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL_GNPS" "$V1_GNPS" "$SCORE_BUNDLE"
  tasks/noise_msg_pair_evidence_core.py
  tasks/encode_noise_massspecgym_checkpoint.py
  tasks/build_noise_massspecgym_pair_evidence.py
  tasks/train_noise_msg_fusion_stack.py
  tasks/apply_noise_msg_fusion_to_gnps.py
  tasks/extract_pair_score_cache_from_bundle.py
  tasks/summarize_noise_msg_fusion.py
  tasks/test_noise_msg_fusion.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing fusion input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/noise_msg_pair_evidence_core.py \
  tasks/build_noise_massspecgym_pair_evidence.py \
  tasks/train_noise_msg_fusion_stack.py \
  tasks/apply_noise_msg_fusion_to_gnps.py \
  tasks/extract_pair_score_cache_from_bundle.py \
  tasks/summarize_noise_msg_fusion.py \
  tasks/test_noise_msg_fusion.py
python -u tasks/test_noise_msg_fusion.py

echo "=== [stage 1/5] encoding official + V1 MassSpecGym shards (progress below) ==="
encode_shard() {
  local gpu="$1"; local shard="$2"; local checkpoint="$3"; local tag="$4"
  echo "[encode ${tag} shard ${shard}] starting on gpu ${gpu}"
  if CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_noise_massspecgym_checkpoint.py \
    --data "$DATA" --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$LOCAL_ROOT/${tag}_shard_${shard}.npz" \
    --shard-index "$shard" --shard-count 2 --batch-size 64; then
    echo "[encode ${tag} shard ${shard}] OK"
  else
    echo "[encode ${tag} shard ${shard}] FAILED" >&2
    return 1
  fi
}
encode_shard "${ALLOCATED_GPUS[0]}" 0 "$OFFICIAL_MSG" official & P0=$!
encode_shard "${ALLOCATED_GPUS[1]}" 1 "$OFFICIAL_MSG" official & P1=$!
set +e
wait "$P0"; S0=$?
wait "$P1"; S1=$?
set -e
if (( S0 != 0 || S1 != 0 )); then
  echo "Official MassSpecGym encoding failed: ${S0}/${S1}" >&2
  exit 6
fi
encode_shard "${ALLOCATED_GPUS[0]}" 0 "$V1" v1 & P2=$!
encode_shard "${ALLOCATED_GPUS[1]}" 1 "$V1" v1 & P3=$!
set +e
wait "$P2"; S2=$?
wait "$P3"; S3=$?
set -e
if (( S2 != 0 || S3 != 0 )); then
  echo "Noise V1 MassSpecGym encoding failed: ${S2}/${S3}" >&2
  exit 7
fi
echo "=== [stage 1/5] embedding shards complete ==="

echo "=== [stage 2/5] building 6.2M-edge pair evidence (16 workers) ==="
python -u tasks/build_noise_massspecgym_pair_evidence.py \
  --graph "$GRAPH" --data "$DATA" \
  --v1-embedding-shard "$LOCAL_ROOT/v1_shard_0.npz" \
  --v1-embedding-shard "$LOCAL_ROOT/v1_shard_1.npz" \
  --official-embedding-shard "$LOCAL_ROOT/official_shard_0.npz" \
  --official-embedding-shard "$LOCAL_ROOT/official_shard_1.npz" \
  --output "$LOCAL_ROOT/evidence" \
  --workers 16 --fragment-tolerance 0.02 --n-highest-peaks 100
mkdir -p "$RUN_ROOT/evidence"
cp "$LOCAL_ROOT/evidence/evidence.npz" "$RUN_ROOT/evidence/evidence.npz"
cp "$LOCAL_ROOT/evidence/report.json" "$RUN_ROOT/evidence/report.json"
echo "=== [stage 2/5] pair evidence complete ==="

echo "=== [stage 3/5] formula-OOF fusion training + freeze ==="
python -u tasks/train_noise_msg_fusion_stack.py \
  --evidence "$LOCAL_ROOT/evidence" \
  --output "$RUN_ROOT/fusion_model"
cp "$LOCAL_ROOT/evidence/report.json" "$RUN_ROOT/evidence_report.json"
echo "=== [stage 3/5] fusion model frozen ==="

echo "=== [stage 4/5] applying frozen fusion to sealed GNPS panels ==="
python -u tasks/apply_noise_msg_fusion_to_gnps.py \
  --benchmark "$GNPS_BENCHMARK" \
  --score-bundle "$SCORE_BUNDLE" \
  --official-embeddings "$OFFICIAL_GNPS" \
  --v1-embeddings "$V1_GNPS" \
  --model "$RUN_ROOT/fusion_model" \
  --output "$RUN_ROOT/gnps_fusion_scores"
echo "=== [stage 4/5] GNPS fusion scores complete ==="

echo "=== [stage 5/5] sealed evaluations: fusion vs official / V1 / WSE ==="

python -u tasks/extract_pair_score_cache_from_bundle.py \
  --score-bundle "$SCORE_BUNDLE" --method official_dreams \
  --output "$RUN_ROOT/official_pair_cache"
python -u tasks/extract_pair_score_cache_from_bundle.py \
  --score-bundle "$SCORE_BUNDLE" --method noise_v1 \
  --output "$RUN_ROOT/v1_pair_cache"

python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-score-cache "$RUN_ROOT/official_pair_cache" \
  --candidate-score-cache "$RUN_ROOT/gnps_fusion_scores" \
  --output "$RUN_ROOT/gnps_official_vs_fusion" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261003 | tail -n 40
python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-score-cache "$RUN_ROOT/v1_pair_cache" \
  --candidate-score-cache "$RUN_ROOT/gnps_fusion_scores" \
  --output "$RUN_ROOT/gnps_v1_vs_fusion" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261003 | tail -n 40
python -u tasks/extract_pair_score_cache_from_bundle.py \
  --score-bundle "$SCORE_BUNDLE" --method weighted_spectral_entropy \
  --output "$RUN_ROOT/wse_pair_cache"
python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-score-cache "$RUN_ROOT/wse_pair_cache" \
  --candidate-score-cache "$RUN_ROOT/gnps_fusion_scores" \
  --output "$RUN_ROOT/gnps_wse_vs_fusion" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261003 | tail -n 40

python -u tasks/summarize_noise_msg_fusion.py \
  --model-report "$RUN_ROOT/fusion_model/report.json" \
  --official "$RUN_ROOT/gnps_official_vs_fusion/report.json" \
  --v1 "$RUN_ROOT/gnps_v1_vs_fusion/report.json" \
  --wse "$RUN_ROOT/gnps_wse_vs_fusion/report.json" \
  --output "$RUN_ROOT/summary/report.json"

echo "=== fusion 5pp shot complete: $RUN_ROOT ==="
echo "verdicts: gnps_official_vs_fusion, gnps_v1_vs_fusion, gnps_wse_vs_fusion (each report.json)"
