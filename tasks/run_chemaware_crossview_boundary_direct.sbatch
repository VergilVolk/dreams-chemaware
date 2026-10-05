#!/bin/bash
#SBATCH --job-name=chem_xview_direct
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_xview_direct_%j.out
#SBATCH --error=data/validation/chem_xview_direct_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_crossview_boundary_direct/run_${SLURM_JOB_ID}"
TRIPLETS="$OUT/triplets"

# The account has a hard Lustre quota. A five-checkpoint native run plus
# Lightning's last checkpoint needs several GiB, so fail before creating any
# output unless at least 12 GiB remain. This is storage quota, not RAM.
quota_row="$(quota -w 2>/dev/null | awk '$1 == "/data02/run01" {print $2, $4; exit}')"
if [[ -n "$quota_row" ]]; then
  read -r quota_used_kb quota_hard_kb <<< "$quota_row"
  required_kb=$((12 * 1024 * 1024))
  available_kb=$((quota_hard_kb - quota_used_kb))
  echo "Quota headroom: ${available_kb} KiB; required: ${required_kb} KiB"
  if (( available_kb < required_kb )); then
    echo "CHEMAWARE_STORAGE_PREFLIGHT_STOP: free at least 12 GiB before this run" >&2
    exit 0
  fi
else
  echo "CHEMAWARE_STORAGE_PREFLIGHT_STOP: unable to read /data02/run01 quota" >&2
  exit 0
fi

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

python -u tasks/test_chemaware_crossview_boundary_core.py
python -u tasks/test_chemaware_crossview_boundary_direct_sbatch.py
python -u tasks/test_chemaware_dreams_native_triplets.py

mapfile -t phasea_candidates < <(
  find data/validation/chemaware_phasea_2pp_protected -type f \
    -path '*/protected_artifact/chemaware_phasea_max_boundary_step2000.ckpt' \
    -print 2>/dev/null | sort
)
[[ ${#phasea_candidates[@]} -gt 0 ]] || {
  echo "Missing protected +2.1266 pp Phase-A artifact" >&2
  exit 2
}
PHASEA="${phasea_candidates[-1]}"
PHASEA_SHA="$(sha256sum "$PHASEA" | awk '{print $1}')"
echo "Protected Phase-A: $PHASEA"
echo "Protected Phase-A SHA256: $PHASEA_SHA"

# Reuse only the exact checkpoint-matched role-0/1 embedding cache.
PHASEA_CACHE=""
mapfile -t cache_reports < <(
  find data/validation -type f -path '*/phasea_role01_cache/report.json' \
    -print 2>/dev/null | sort -r
)
for report in "${cache_reports[@]}"; do
  if python - "$report" "$PHASEA_SHA" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
b = json.loads(p.read_text(encoding="utf-8"))
ok = (
    b.get("status") == "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE"
    and b.get("checkpoint_sha256") == sys.argv[2]
    and b.get("formula_roles") == [0, 1]
    and b.get("outer_roles_2_3_4_accessed") is False
    and (p.parent / "rows.npy").is_file()
    and (p.parent / "embeddings_f32.npy").is_file()
)
raise SystemExit(0 if ok else 1)
PY
  then
    PHASEA_CACHE="$(dirname "$report")"
    break
  fi
done

if [[ -z "$PHASEA_CACHE" ]]; then
  PHASEA_CACHE="$OUT/phasea_role01_cache"
  srun --export=ALL --preserve-env python -u tasks/encode_chemaware_formula_role_checkpoint_rows.py \
    --checkpoint "$PHASEA" \
    --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --official-checkpoint "$OFFICIAL" \
    --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
    --formula-role 0 1 --output "$PHASEA_CACHE" \
    --batch-size 64 --n-highest-peaks 100 --device cuda
else
  echo "Reusing checksum-matched Phase-A cache: $PHASEA_CACHE"
fi

python -u tasks/build_chemaware_crossview_boundary_native_triplets.py \
  --phasea-cache "$PHASEA_CACHE" \
  --validation-pool "$BASE/triplets/val_pool.npz" \
  --frozen-report data/validation/chemaware_orthogonal_rule_residual_policy_v2/report.json \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --token-dir data/validation/chemaware_corrected_manifest_tokens_v1 \
  --rule-library dreams/models/chem_aware/chem_rules_data.json \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$TRIPLETS" \
  --spectra-per-identity 5 --aggregation q25 --min-context 1 \
  --absolute-threshold-multiplier 0.5 --dominance-threshold 0 \
  --margin 0.1 --negative-references-per-correction 3 \
  --safety-events-per-correction 3 --dreams-replay-events 512 \
  --minimum-correction-events 100 --minimum-correction-queries 50 \
  --minimum-correction-formulas 40 --seed 3407

TRIPLET_STATUS="$(python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"
if [[ "$TRIPLET_STATUS" == "CHEMAWARE_CROSSVIEW_BOUNDARY_COVERAGE_STOP" ]]; then
  python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
b = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print("CROSSVIEW_COVERAGE_STOP:", b["coverage"], flush=True)
PY
  echo "COMPLETE: no direct fine-tuning started and no weights were updated"
  exit 0
fi

python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
b = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert b["status"] == "CHEMAWARE_CROSSVIEW_BOUNDARY_TRIPLETS_COMPLETE"
assert b["distillation"] is False
assert b["inference_time_chemistry"] is False
assert b["training_object"] == "ordinary DreaMS spectrum triplets only"
assert all(b["gates"].values())
print("DIRECT_TRIPLET_GATE_PASS:", b["coverage"], flush=True)
PY

# Direct continuation only. No teacher model, teacher target, auxiliary loss,
# reranker, chemical input, or custom sampling weight reaches this command.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$TRIPLETS/train_pool.npz" \
  --val-pool "$TRIPLETS/val_pool.npz" \
  --official-checkpoint "$PHASEA" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 1e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 4 \
  --max-epochs 4 --max-steps 500 --checkpoint-mode fixed_steps \
  --save-every-n-steps 100 --n-highest-peaks 100 --device cuda

checkpoint_args=(--checkpoint "official=$OFFICIAL" --checkpoint "phaseA_base=$PHASEA")
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 5 ]] || {
  echo "Expected five checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "crossview_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference phaseA_base --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name phaseA_base --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"
if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_CROSSVIEW_ROLE2_STOP: no checkpoint safely beat protected Phase A"
  exit 0
fi

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "crossview_direct=$(realpath "$SELECTED")" \
  --paired-reference phaseA_base --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

echo "COMPLETE: direct DreaMS triplet fine-tuning -> $OUT"
