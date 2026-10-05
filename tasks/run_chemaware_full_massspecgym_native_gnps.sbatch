#!/bin/bash
# Full MassSpecGym ChemAware curriculum -> one fixed DreaMS-native fine-tune ->
# complete frozen GNPS gold/silver evaluation.  No checkpoint selection is
# performed on GNPS and no GNPS label or score enters triplet construction.
#SBATCH --job-name=chem_fullmsg
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/chem_fullmsg_%j.out
#SBATCH --error=data/validation/chem_fullmsg_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
TRAIN_POOL="data/e1/e1_train_triplet_pool_10ppm.npz"
VAL_POOL="data/e1/e1_val_triplet_pool_10ppm.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
CACHE="data/validation/chemaware_corrected_manifest_tokens_v1"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
MASSBANK_CORPUS="data/validation/chemaware_massbank_annotated_fragment_corpus_v1_20260928/massbank_fragment_corpus.json"
FRAGMENT_LEDGER="data/validation/chemaware_fragment_graph_native/run_2346306/qualified_source"
GNPS="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OUT="data/validation/chemaware_full_massspecgym_native/run_${SLURM_JOB_ID}"
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_BASE/chemaware_full_massspecgym_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH"
cleanup_scratch() { rm -rf -- "$SCRATCH"; }
trap cleanup_scratch EXIT

required=(
  "$DATA" "$TRAIN_POOL" "$VAL_POOL" "$MANIFEST"
  "$CACHE/rows.npy" "$CACHE/official_embeddings_f32.npy" "$CACHE/report.json"
  "$OFFICIAL" "$ARCH" "$MASSBANK_CORPUS"
  "$FRAGMENT_LEDGER/report.json" "$FRAGMENT_LEDGER/candidate_scores.tsv"
  "$GNPS/spectra.mgf" "$GNPS/manifest.csv.gz"
  "$GNPS/panel_identity_disjoint.npz" "$GNPS/panel_formula_disjoint.npz"
  tasks/export_chemaware_full_massspecgym_candidate_registry.py
  tasks/build_chemaware_massbank_candidate_source_ledger.py
  tasks/qualify_chemaware_candidate_source_ledger.py
  tasks/build_chemaware_full_massspecgym_native_triplets.py
  tasks/train_chemaware_dreams_native.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "CHEMAWARE_FULL_MSG_MISSING_INPUT: $path" >&2; exit 2; }
done

python -m py_compile \
  tasks/export_chemaware_full_massspecgym_candidate_registry.py \
  tasks/build_chemaware_full_massspecgym_native_triplets.py \
  tasks/train_chemaware_dreams_native.py \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
python -u tasks/test_chemaware_full_massspecgym_native_triplets.py
python -u tasks/test_evaluate_gnps_gold_silver_10ppm_embeddings.py

# Export every candidate-manifest query that belongs to the complete official
# 112,601-anchor train pool.  No labels are exported.
python -u tasks/export_chemaware_full_massspecgym_candidate_registry.py \
  --manifest "$MANIFEST" --official-train-pool "$TRAIN_POOL" --data "$DATA" \
  --output "$SCRATCH/full_candidate_registry"

# Recompute the qualified MassBank rule evidence over the full available
# MassSpecGym candidate graph instead of reusing the old 4,032-query ledger.
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

# One event per eligible anchor across BOTH official pools (train 112,601 +
# validation 21,163 = 133,764 events; overlap fails closed in the builder).
# Chemistry may replace the reference pair but cannot add duplicate events,
# broadcast identities, change weights or remove any official anchor.
python -u tasks/build_chemaware_full_massspecgym_native_triplets.py \
  --official-train-pool "$TRAIN_POOL" \
  --official-train-pool "$VAL_POOL" \
  --validation-pool "$VAL_POOL" \
  --source-ledger "$SCRATCH/massbank_full_qualified" \
  --source-ledger "$FRAGMENT_LEDGER" \
  --manifest "$MANIFEST" \
  --embedding-rows "$CACHE/rows.npy" \
  --embeddings "$CACHE/official_embeddings_f32.npy" \
  --embedding-report "$CACHE/report.json" \
  --geometry-checkpoint "$OFFICIAL" --data "$DATA" \
  --output "$OUT/triplets" --margin 0.1 --minimum-supporting-sources 1

# Match the successful Phase-A exposure: about 1.35 passes over the event pool.
# 133,764 events / batch 4 = 33,441 steps per pass; fixed 45,000 steps.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" \
  --train-pool "$OUT/triplets/train_pool.npz" \
  --val-pool "$OUT/triplets/val_pool.npz" \
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCH" \
  --output "$OUT/training" --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 2 --max-steps 45000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 45000 --n-highest-peaks 100 --device cuda

TRAINED="$OUT/training/step-045000.ckpt"
[[ -f "$TRAINED" ]] || { echo "Missing fixed final checkpoint: $TRAINED" >&2; exit 7; }

# GNPS is evaluation-only.  Both frozen panels are evaluated by the evaluator:
# identity-disjoint 10,995 queries and formula-disjoint 5,261 queries.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$TRAINED" --output "$SCRATCH/chemaware_fullmsg_slim.pt"
srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS" --checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/official_gnps_embeddings.npz" --batch-size 64 --label official
srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS" --checkpoint "$SCRATCH/chemaware_fullmsg_slim.pt" \
  --architecture-checkpoint "$ARCH" \
  --output "$SCRATCH/chemaware_fullmsg_gnps_embeddings.npz" \
  --batch-size 64 --label chemaware-full-massspecgym
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS" \
  --baseline-embeddings "$SCRATCH/official_gnps_embeddings.npz" \
  --candidate-embeddings "$SCRATCH/chemaware_fullmsg_gnps_embeddings.npz" \
  --output "$OUT/gnps_full_evaluation" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261001

cp "$SCRATCH/full_candidate_registry/report.json" "$OUT/full_candidate_registry_report.json"
cp "$SCRATCH/massbank_full_qualified/qualification_report.json" "$OUT/massbank_full_qualification_report.json"
cp "$SCRATCH/official_gnps_embeddings.json" "$OUT/official_gnps_encoding.json"
cp "$SCRATCH/chemaware_fullmsg_gnps_embeddings.json" "$OUT/chemaware_fullmsg_gnps_encoding.json"
sha256sum "$TRAINED" "$OUT/triplets/train_pool.npz" \
  "$OUT/triplets/chemical_override_ledger.tsv" \
  "$OUT/gnps_full_evaluation/report.json" > "$OUT/SHA256SUMS.txt"

# Lightning's last.ckpt is redundant with the sole fixed-step artifact.  Keep
# the named step checkpoint and remove only this duplicate inside the new run.
rm -f -- "$OUT/training/last.ckpt"
echo "COMPLETE: full MassSpecGym ChemAware embedding + complete GNPS evaluation -> $OUT"
