#!/bin/bash
# Two real encoder continuations on MassSpecGym train, full 18,333-query held
# development evaluation, then one gated GNPS evaluation. GNPS never selects
# a checkpoint, triplet, feature, threshold or training decision.
#SBATCH --job-name=noise_chem_native
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/noise_chem_native_%j.out
#SBATCH --error=data/validation/noise_chem_native_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=10

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
TRAIN_POOL="data/e1/e1_train_triplet_pool_10ppm.npz"
VAL_POOL="data/e1/e1_val_triplet_pool_10ppm.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
OFFICIAL_CACHE="data/validation/noise_corrected_candidate_graph_v1_20260906/official_embeddings.npz"
OFFICIAL="data/e1/official_embedding_slim.pt"
NOISE="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
NOISE_HELD="data/validation/noise_relation_t1_t3_run_2347055/evaluation/seed_3407/held_per_query.csv.gz"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
MASSBANK_CORPUS="data/validation/chemaware_massbank_annotated_fragment_corpus_v1_20260928/massbank_fragment_corpus.json"
GNPS="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OUT="data/validation/noise_chemaware_full_massspecgym_native/run_${SLURM_JOB_ID}"
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_BASE/noise_chemaware_fullmsg_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }
mkdir -p "$OUT" "$SCRATCH"
cleanup_scratch() { rm -rf -- "$SCRATCH"; }
trap cleanup_scratch EXIT

required=(
  "$DATA" "$TRAIN_POOL" "$VAL_POOL" "$MANIFEST" "$GRAPH" "$OFFICIAL_CACHE"
  "$OFFICIAL" "$NOISE" "$NOISE_HELD" "$ARCH" "$MASSBANK_CORPUS"
  tasks/encode_chemaware_training_manifest_rows.py
  tasks/export_chemaware_full_massspecgym_candidate_registry.py
  tasks/build_chemaware_massbank_candidate_source_ledger.py
  tasks/qualify_chemaware_candidate_source_ledger.py
  tasks/build_chemaware_full_massspecgym_native_triplets.py
  tasks/train_chemaware_specific_replay_native.py
  tasks/evaluate_noise_dreams_native.py
  tasks/compare_massspecgym_encoder_continuations.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "NOISE_CHEMAWARE_FULL_MSG_MISSING: $path" >&2; exit 2; }
done

python -m py_compile \
  tasks/encode_chemaware_training_manifest_rows.py \
  tasks/build_chemaware_full_massspecgym_native_triplets.py \
  tasks/train_chemaware_specific_replay_native.py \
  tasks/evaluate_noise_dreams_native.py \
  tasks/compare_massspecgym_encoder_continuations.py \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
python -u tasks/test_chemaware_full_massspecgym_native_triplets.py
python -u tasks/test_chemaware_full_massspecgym_native_sbatch.py

# Frozen Noise V1 geometry over TRAIN anchors only. Held MassSpecGym rows and
# all GNPS rows remain invisible to curriculum construction.
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u \
  tasks/encode_chemaware_training_manifest_rows.py \
  --checkpoint "$NOISE" --anchor-pool "$TRAIN_POOL" \
  --manifest "$MANIFEST" --data "$DATA" \
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/noise_train_geometry" --batch-size 64 --device cuda

# Candidate metadata is restricted to the official MassSpecGym TRAIN anchors.
python -u tasks/export_chemaware_full_massspecgym_candidate_registry.py \
  --manifest "$MANIFEST" --official-train-pool "$TRAIN_POOL" --data "$DATA" \
  --output "$SCRATCH/full_candidate_registry"
python -u tasks/build_chemaware_massbank_candidate_source_ledger.py \
  --corpus "$MASSBANK_CORPUS" \
  --candidate-ledger "$SCRATCH/full_candidate_registry/candidate_ledger.tsv" \
  --query-registry "$SCRATCH/full_candidate_registry/query_registry.tsv" \
  --data "$DATA" --scope all_candidates \
  --output "$SCRATCH/massbank_full_unqualified"
python -u tasks/qualify_chemaware_candidate_source_ledger.py \
  --ledger "$SCRATCH/massbank_full_unqualified" --manifest "$MANIFEST" \
  --output "$SCRATCH/massbank_full_qualified" \
  --fold-seed 20260928 --folds 5 --confirmation-folds 3 4 \
  --bootstrap-draws 5000 --minimum-applicable-queries 100

# Exactly 112,601 train anchors: chemistry replaces a qualified hard negative,
# never adds event weight. The 21,163-event validation pool stays validation.
python -u tasks/build_chemaware_full_massspecgym_native_triplets.py \
  --official-train-pool "$TRAIN_POOL" --validation-pool "$VAL_POOL" \
  --source-ledger "$SCRATCH/massbank_full_qualified" \
  --manifest "$MANIFEST" \
  --embedding-rows "$SCRATCH/noise_train_geometry/rows.npy" \
  --embeddings "$SCRATCH/noise_train_geometry/embeddings_f32.npy" \
  --embedding-report "$SCRATCH/noise_train_geometry/report.json" \
  --geometry-checkpoint "$NOISE" --data "$DATA" \
  --output "$OUT/triplets" --margin 0.1 --minimum-supporting-sources 1

train_arm() {
  local gpu="$1"
  local train_pool="$2"
  local output="$3"
  local log_name="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_chemaware_specific_replay_native.py \
    --data "$DATA" --train-pool "$train_pool" --val-pool "$VAL_POOL" \
    --official-checkpoint "$NOISE" --architecture-checkpoint "$ARCH" \
    --output "$output" --seed 3407 --lr 5e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
    --max-epochs 2 --max-steps 45000 --checkpoint-mode fixed_steps \
    --save-every-n-steps 45000 --n-highest-peaks 100 --device cuda \
    >"$OUT/train_${log_name}.out" 2>"$OUT/train_${log_name}.err"
}

# Both GPUs perform real 116M-parameter updates, with identical initialization,
# seed, event count, optimizer, exposure and checkpoint step.
train_arm "${GPUS[0]}" "$TRAIN_POOL" "$OUT/control_training" control & PID_CONTROL=$!
train_arm "${GPUS[1]}" "$OUT/triplets/train_pool.npz" "$OUT/chemistry_training" chemistry & PID_CHEMISTRY=$!
set +e
wait "$PID_CONTROL"; STATUS_CONTROL=$?
wait "$PID_CHEMISTRY"; STATUS_CHEMISTRY=$?
set -e
if (( STATUS_CONTROL != 0 || STATUS_CHEMISTRY != 0 )); then
  tail -n 120 "$OUT/train_control.err" >&2 || true
  tail -n 120 "$OUT/train_chemistry.err" >&2 || true
  echo "Encoder training failed: control=$STATUS_CONTROL chemistry=$STATUS_CHEMISTRY" >&2
  exit 6
fi

CONTROL="$OUT/control_training/step-045000.ckpt"
CHEMISTRY="$OUT/chemistry_training/step-045000.ckpt"
[[ -f "$CONTROL" && -f "$CHEMISTRY" ]] || {
  echo "Fixed-step checkpoints are absent" >&2; exit 7;
}

evaluate_msg() {
  local gpu="$1"
  local checkpoint="$2"
  local output="$3"
  local log_name="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" --embedding-cache "$OFFICIAL_CACHE" \
    --official-slim-checkpoint "$OFFICIAL" --source-manifest "$MANIFEST" \
    --data "$DATA" --checkpoint "$checkpoint" --architecture-checkpoint "$ARCH" \
    --output "$output" --arm targeted --outer-fold 0 \
    --formula-fold-seed 20260825 --batch-size 64 \
    --bootstrap-resamples 10000 --bootstrap-seed 20261023 \
    --no-write-held-metric-evidence \
    >"$OUT/evaluate_${log_name}.out" 2>"$OUT/evaluate_${log_name}.err"
}

evaluate_msg "${GPUS[0]}" "$CONTROL" "$OUT/massspecgym/control" control & PID_EVAL_CONTROL=$!
evaluate_msg "${GPUS[1]}" "$CHEMISTRY" "$OUT/massspecgym/chemistry" chemistry & PID_EVAL_CHEMISTRY=$!
set +e
wait "$PID_EVAL_CONTROL"; STATUS_EVAL_CONTROL=$?
wait "$PID_EVAL_CHEMISTRY"; STATUS_EVAL_CHEMISTRY=$?
set -e
if (( STATUS_EVAL_CONTROL != 0 || STATUS_EVAL_CHEMISTRY != 0 )); then
  tail -n 120 "$OUT/evaluate_control.err" >&2 || true
  tail -n 120 "$OUT/evaluate_chemistry.err" >&2 || true
  echo "MassSpecGym held evaluation failed" >&2
  exit 8
fi

python -u tasks/compare_massspecgym_encoder_continuations.py \
  --noise "$NOISE_HELD" \
  --control "$OUT/massspecgym/control/held_per_query.csv.gz" \
  --chemistry "$OUT/massspecgym/chemistry/held_per_query.csv.gz" \
  --output "$OUT/massspecgym/comparison" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261023

ADVANCE="$(python -c "import json; print(int(json.load(open('$OUT/massspecgym/comparison/report.json'))['advance_to_gnps']))")"
if [[ "$ADVANCE" != "1" ]]; then
  echo "STOP_BEFORE_GNPS: chemistry did not add held-MassSpecGym value beyond Noise and matched continuation"
  exit 0
fi

gnps_required=(
  "$GNPS/spectra.mgf" "$GNPS/manifest.csv.gz"
  "$GNPS/panel_identity_disjoint.npz" "$GNPS/panel_formula_disjoint.npz"
)
for path in "${gnps_required[@]}"; do
  [[ -f "$path" ]] || { echo "GNPS_EVALUATION_INPUT_MISSING_AFTER_MSG_PASS: $path" >&2; exit 9; }
done

mkdir -p "$OUT/checkpoint"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$CHEMISTRY" --output "$OUT/checkpoint/noise_chemaware_fullmsg_slim.pt"
SLIM="$OUT/checkpoint/noise_chemaware_fullmsg_slim.pt"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS" --checkpoint "$NOISE" --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/noise_gnps_embeddings.npz" --batch-size 64 --label noise-v1 & PID_GNPS_NOISE=$!
CUDA_VISIBLE_DEVICES="${GPUS[1]}" python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS" --checkpoint "$SLIM" --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/noise_chemaware_gnps_embeddings.npz" --batch-size 64 \
  --label noise-chemaware-fullmsg & PID_GNPS_CHEM=$!
wait "$PID_GNPS_NOISE"
wait "$PID_GNPS_CHEM"

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS" \
  --baseline-embeddings "$SCRATCH/noise_gnps_embeddings.npz" \
  --candidate-embeddings "$SCRATCH/noise_chemaware_gnps_embeddings.npz" \
  --output "$OUT/gnps_evaluation" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261024

cp "$SCRATCH/noise_gnps_embeddings.json" "$OUT/noise_gnps_encoding.json"
cp "$SCRATCH/noise_chemaware_gnps_embeddings.json" "$OUT/noise_chemaware_gnps_encoding.json"
sha256sum "$SLIM" "$OUT/triplets/train_pool.npz" \
  "$OUT/massspecgym/comparison/report.json" "$OUT/gnps_evaluation/report.json" \
  > "$OUT/SHA256SUMS.txt"
echo "COMPLETE: Noise-V1 initialized full-MassSpecGym ChemAware encoder -> $OUT"
