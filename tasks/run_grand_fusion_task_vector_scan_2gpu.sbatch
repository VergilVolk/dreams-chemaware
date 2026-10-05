#!/bin/bash
#SBATCH --job-name=grand_taskvec
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/grand_taskvec_%j.out
#SBATCH --error=data/validation/grand_taskvec_%j.err

# Low-cost Noise x ChemAware parameter-composition screen.
# CHEM_CHECKPOINT may be a native Lightning or an official_embedding_slim_v1
# checkpoint; it is normalized once inside this immutable run directory.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

: "${CHEM_CHECKPOINT:?Set CHEM_CHECKPOINT to the frozen ChemAware checkpoint}"
BASE_SLIM="${BASE_SLIM:-data/e1/official_embedding_slim.pt}"
NOISE_SLIM="${NOISE_SLIM:-data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt}"
GRAPH="${GRAPH:-data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz}"
DATA="${DATA:-data/models/MassSpecGym_MurckoHist_split.hdf5}"
ARCHITECTURE="${ARCHITECTURE:-dreams/models/pretrained/ssl_model_server.pt}"
OFFICIAL_EMBEDDINGS="${OFFICIAL_EMBEDDINGS:-data/validation/noise_corrected_candidate_graph_v1_20260906/official_embeddings.npz}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz}"
RUN_ROOT="data/validation/grand_fusion_task_vector_scan_${SLURM_JOB_ID}"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/evaluation"
CHEM_SLIM="$RUN_ROOT/chem_normalized_slim.pt"

required=(
  "$BASE_SLIM" "$NOISE_SLIM" "$CHEM_CHECKPOINT" "$GRAPH" "$DATA" "$ARCHITECTURE"
  "$OFFICIAL_EMBEDDINGS" "$SOURCE_MANIFEST"
  tasks/merge_dreams_task_vectors.py
  tasks/normalize_dreams_slim_checkpoint.py
  tasks/analyze_dreams_task_vector_interference.py
  tasks/evaluate_noise_dreams_native.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing task-vector input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/normalize_dreams_slim_checkpoint.py \
  tasks/merge_dreams_task_vectors.py \
  tasks/analyze_dreams_task_vector_interference.py

python -u tasks/normalize_dreams_slim_checkpoint.py \
  --input "$CHEM_CHECKPOINT" --output "$CHEM_SLIM"

python -u tasks/analyze_dreams_task_vector_interference.py \
  --base "$BASE_SLIM" --noise "$NOISE_SLIM" --chem "$CHEM_SLIM" \
  --output "$RUN_ROOT/interference.json"

make_linear() {
  local name="$1"; local noise="$2"; local chem="$3"
  python -u tasks/merge_dreams_task_vectors.py \
    --base "$BASE_SLIM" --noise "$NOISE_SLIM" --chem "$CHEM_SLIM" \
    --mode linear --alpha-noise "$noise" --alpha-chem "$chem" \
    --output "$RUN_ROOT/checkpoints/${name}.pt"
}
make_ties() {
  local name="$1"; local density="$2"; local scale="$3"
  python -u tasks/merge_dreams_task_vectors.py \
    --base "$BASE_SLIM" --noise "$NOISE_SLIM" --chem "$CHEM_SLIM" \
    --mode ties --ties-density "$density" --ties-scale "$scale" \
    --output "$RUN_ROOT/checkpoints/${name}.pt"
}

# Fixed before outcomes are opened. Seven arms keep multiplicity bounded.
make_linear linear_n050_c050 0.50 0.50
make_linear linear_n075_c025 0.75 0.25
make_linear linear_n025_c075 0.25 0.75
make_ties ties_d010_s050 0.10 0.50
make_ties ties_d020_s050 0.20 0.50
make_ties ties_d020_s100 0.20 1.00
make_ties ties_d050_s050 0.50 0.50

evaluate_arm() {
  local gpu="$1"; local name="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" --embedding-cache "$OFFICIAL_EMBEDDINGS" \
    --official-slim-checkpoint "$BASE_SLIM" --source-manifest "$SOURCE_MANIFEST" \
    --data "$DATA" --checkpoint "$RUN_ROOT/checkpoints/${name}.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$RUN_ROOT/evaluation/${name}" --arm targeted \
    --outer-fold 0 --formula-fold-seed 20260825 --batch-size 64 \
    --bootstrap-resamples 10000 --bootstrap-seed 20261004 \
    --no-write-held-metric-evidence \
    >"$RUN_ROOT/evaluate_${name}.out" 2>"$RUN_ROOT/evaluate_${name}.err"
}

arms=(
  linear_n050_c050 linear_n075_c025 linear_n025_c075
  ties_d010_s050 ties_d020_s050 ties_d020_s100 ties_d050_s050
)
for ((index=0; index<${#arms[@]}; index+=2)); do
  first="${arms[$index]}"
  evaluate_arm "${ALLOCATED_GPUS[0]}" "$first" & first_pid=$!
  second_pid=""
  if (( index + 1 < ${#arms[@]} )); then
    second="${arms[$((index + 1))]}"
    evaluate_arm "${ALLOCATED_GPUS[1]}" "$second" & second_pid=$!
  fi
  set +e
  wait "$first_pid"; first_status=$?
  second_status=0
  if [[ -n "$second_pid" ]]; then wait "$second_pid"; second_status=$?; fi
  set -e
  if (( first_status != 0 || second_status != 0 )); then
    echo "Task-vector evaluation batch failed: $first_status/$second_status" >&2
    exit 6
  fi
done

cat >"$RUN_ROOT/README.txt" <<'EOF'
DEVELOPMENT SCREEN ONLY. Seven arms were inspected together on the reused
MassSpecGym fold-0 graph. Select at most one arm using the preregistered
protection rule, then perform one untouched GNPS evaluation. Do not report the
best of seven as confirmatory evidence.
EOF
echo "Task-vector scan complete: $RUN_ROOT"
