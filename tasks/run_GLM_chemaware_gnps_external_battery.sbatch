#!/bin/bash
# ChemAware GNPS external battery: measure transfer of every frozen encoder
# asset onto the sealed GNPS gold/silver 10 ppm benchmark (10,995 + 5,261
# queries, identity/formula overlap with MassSpecGym+MoNA is zero).
# Encoding and paired evaluation only -- no training, no curriculum mining,
# no gates.  Descriptive evidence for the plateau analysis of 2026-09-30.
#SBATCH --job-name=chem_gnps_batt
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_gnps_battery_%j.out
#SBATCH --error=data/validation/chem_gnps_battery_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
PHASEA="data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
STEP750="data/validation/chemaware_fragment_graph_native/run_2346408_resume_run_2346306/training/step-000750.ckpt"
OUT="data/validation/GLM_chemaware_gnps_external_battery/run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/chem_gnps_battery_${SLURM_JOB_ID}"

STAGE1_SLIM="$LOCAL_ROOT/chemaware_stage1_slim.pt"
PHASEA_SLIM="$LOCAL_ROOT/chemaware_phaseA_slim.pt"
STEP750_SLIM="$LOCAL_ROOT/chemaware_step750_slim.pt"
OFFICIAL_EMB="$LOCAL_ROOT/official_embeddings.npz"
STAGE1_EMB="$LOCAL_ROOT/chemaware_stage1_embeddings.npz"
PHASEA_EMB="$LOCAL_ROOT/chemaware_phaseA_embeddings.npz"
STEP750_EMB="$LOCAL_ROOT/chemaware_step750_embeddings.npz"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL" "$ARCHITECTURE" "$STAGE1" "$PHASEA" "$STEP750"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "GLM_GNPS_BATTERY_MISSING_INPUT: $path" >&2; exit 2; }
done

LOCAL_FREE_KB=$(df -Pk "$LOCAL_BASE" | awk 'NR==2 {print $4}')
if [[ -z "$LOCAL_FREE_KB" ]] || (( LOCAL_FREE_KB < 6291456 )); then
  echo "Node-local space below 6 GiB: base=$LOCAL_BASE free_kb=${LOCAL_FREE_KB:-unknown}" >&2
  exit 5
fi

python -m py_compile \
  tasks/prepare_official_embedding_checkpoint.py \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py

# Format conversions only; no optimization anywhere in this battery.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1" --output "$STAGE1_SLIM"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$PHASEA_SLIM"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STEP750" --output "$STEP750_SLIM"

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$OFFICIAL_EMB" \
  --batch-size 64 --label official
python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$STAGE1_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$STAGE1_EMB" \
  --batch-size 64 --label chemaware-stage1
python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$PHASEA_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$PHASEA_EMB" \
  --batch-size 64 --label chemaware-phaseA
python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$STEP750_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$STEP750_EMB" \
  --batch-size 64 --label chemaware-step750-exploratory

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" --baseline-embeddings "$OFFICIAL_EMB" \
  --candidate-embeddings "$STAGE1_EMB" \
  --output "$OUT/chemaware_stage1_vs_official" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260930
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" --baseline-embeddings "$OFFICIAL_EMB" \
  --candidate-embeddings "$PHASEA_EMB" \
  --output "$OUT/chemaware_phaseA_vs_official" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260930
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" --baseline-embeddings "$OFFICIAL_EMB" \
  --candidate-embeddings "$STEP750_EMB" \
  --output "$OUT/chemaware_step750_vs_official" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260930
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" --baseline-embeddings "$STAGE1_EMB" \
  --candidate-embeddings "$PHASEA_EMB" \
  --output "$OUT/chemaware_phaseA_vs_stage1" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260930

cp "${OFFICIAL_EMB%.npz}.json" "$OUT/official_encoding.json"
cp "${STAGE1_EMB%.npz}.json" "$OUT/chemaware_stage1_encoding.json"
cp "${PHASEA_EMB%.npz}.json" "$OUT/chemaware_phaseA_encoding.json"
cp "${STEP750_EMB%.npz}.json" "$OUT/chemaware_step750_encoding.json"

sha256sum "$STAGE1" "$PHASEA" "$STEP750" \
  "$OUT"/chemaware_*_vs_*/report.json > "$OUT/SHA256SUMS.txt"

cat > "$OUT/provenance.txt" <<EOF
benchmark_sha256_panel_identity=3d53b8a17fcc8391eb5800c11db6e424577aeeffb3836da242934de50b96c007
benchmark_sha256_panel_formula=efbd24ca71c9570785bcc9c50c5cb2b79d15b6125b94bc3348ce331ee74a7fa1
role_of_official=training-distribution baseline
role_of_stage1=released ChemAware native stage-1 (role-3 confirmed +1.8144 pp)
role_of_phaseA=protected dev champion (+2.1266 pp role-2, never role-3)
role_of_step750=exploratory continuation (forbidden as released, diagnostic only)
claim_limit=GNPS shares the public-data ecosystem with DreaMS pretraining
EOF

echo "GLM_GNPS_BATTERY_COMPLETE: $OUT"
