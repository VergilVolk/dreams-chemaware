#!/bin/bash
#SBATCH --job-name=chem_rolebudget
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_rolebudget_%j.out
#SBATCH --error=data/validation/chem_rolebudget_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
STAGE1="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
ROWS="data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy"
OFFICIAL_EMBEDDINGS="data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy"
REPLAY="data/e1/e1_train_triplet_pool_10ppm.npz"
OUT="data/validation/chemaware_online_role_budget_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL"
  "$ARCHITECTURE"
  "$DATA"
  "$STAGE1"
  "$BASE/triplets/report.json"
  "$BASE/triplets/train_pool.npz"
  "$BASE/triplets/val_pool.npz"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  "$MANIFEST"
  "$ROWS"
  "$OFFICIAL_EMBEDDINGS"
  "$REPLAY"
  tasks/chemaware_online_packet_core.py
  tasks/audit_chemaware_online_packet_preflight.py
  tasks/train_chemaware_online_packet_native.py
  tasks/test_chemaware_online_packet_core.py
  tasks/test_chemaware_online_packet_runtime_contracts.py
  tasks/build_chemaware_max_boundary_native_triplets.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/freeze_chemaware_phasea_2pp_artifact.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required online-packet file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_online_packet_core.py
python -u tasks/test_chemaware_online_packet_runtime_contracts.py
python -u tasks/test_chemaware_max_boundary_native_triplets.py

python -u tasks/audit_chemaware_online_packet_preflight.py \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest "$MANIFEST" \
  --data "$DATA" \
  --dreams-replay-pool "$REPLAY" \
  --embedding-rows "$ROWS" \
  --embeddings "$OFFICIAL_EMBEDDINGS" \
  --output "$OUT/official_geometry_preflight" \
  --margin 0.1 --packet-width 3 --dreams-replay-events 1024 --seed 3407

# Rebuild the exact protected Phase-A empirical curriculum.  This is not a
# residual start: the subsequent process begins from official DreaMS and keeps
# one Adam object alive across Phase A and every online-remine interval.
python -u tasks/build_chemaware_max_boundary_native_triplets.py \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest "$MANIFEST" \
  --embedding-rows "$ROWS" \
  --official-embeddings "$OFFICIAL_EMBEDDINGS" \
  --dreams-replay-pool "$REPLAY" \
  --data "$DATA" \
  --output "$OUT/phasea_triplets" \
  --margin 0.1 --negative-references-per-error 2 \
  --chemical-candidates-per-error 2 --dreams-replay-events 1024 \
  --seed 3407 --min-error-event-fraction 0.20

srun --export=ALL --preserve-env python -u tasks/train_chemaware_online_packet_native.py \
  --data "$DATA" \
  --phasea-train-pool "$OUT/phasea_triplets/train_pool.npz" \
  --phasea-val-pool "$OUT/phasea_triplets/val_pool.npz" \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest "$MANIFEST" \
  --dreams-replay-pool "$REPLAY" \
  --official-checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/training" \
  --seed 3407 --phasea-lr 5e-6 --online-lr 2e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --phasea-steps 2000 --online-steps 1000 --remine-every-steps 250 \
  --packet-width 3 --online-schedule phasea_role_budget \
  --dreams-replay-events 1024 \
  --encode-batch-size 64 --n-highest-peaks 100 --device cuda

PHASEA="$OUT/training/checkpoints/step-002000.ckpt"
test -f "$PHASEA"
checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_step-002000=$PHASEA"
)
for step in 002250 002500 002750 003000; do
  checkpoint="$OUT/training/checkpoints/step-${step}.ckpt"
  test -f "$checkpoint"
  checkpoint_args+=(--checkpoint "online_step-${step}=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference phaseA_step-002000 --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

# The online comparison is valid only if the in-run step-2,000 anchor exactly
# reproduces the protected 54/12 and +2.1266 pp Phase-A result.
python -u tasks/freeze_chemaware_phasea_2pp_artifact.py \
  --checkpoint "$PHASEA" \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --triplet-report "$OUT/phasea_triplets/report.json" \
  --output "$OUT/phasea_reproduction_gate" \
  --model-name phaseA_step-002000 --official-name official

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name phaseA_step-002000 --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_ONLINE_PACKET_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: role-2 online-packet audit -> $OUT"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "released_stage1=$STAGE1" \
  --checkpoint "phaseA_step-002000=$PHASEA" \
  --checkpoint "online_selected=$SELECTED_REAL" \
  --paired-reference phaseA_step-002000 --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

mkdir -p "$OUT/protected_candidate"
ln "$SELECTED_REAL" "$OUT/protected_candidate/chemaware_online_packet_selected.ckpt"
cp "$OUT/checkpoint_selection.json" "$OUT/protected_candidate/"
cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/training/training_report.json" "$OUT/protected_candidate/"
(
  cd "$OUT/protected_candidate"
  sha256sum -- * > SHA256SUMS
)
echo "COMPLETE: protected ChemAware online-packet embedding -> $OUT/protected_candidate"
