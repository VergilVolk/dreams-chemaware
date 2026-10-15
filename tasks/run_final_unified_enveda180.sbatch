#!/bin/bash
# Freeze the best preplanned GNPS ablation, generate Enveda-180 component
# evidence without labels, and open the two primary frozen panels exactly once.
# No encoder or reranker is trained in this job.
#SBATCH --job-name=unified_e180
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/unified_e180_%j.out
#SBATCH --error=data/validation/unified_e180_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GNPS_RUN="${UNIFIED_GNPS_RUN:-data/validation/unified_candidate_energy_run_2364137}"
BENCHMARK="${ENVEDA180_BENCHMARK:-data/validation/enveda180_cross_condition_benchmark_v3_20261012}"
LAYER_OUT="data/validation/enveda180_layered_scoreblind_run_${SLURM_JOB_ID}"
OUT="data/validation/unified_candidate_energy_enveda_final_run_${SLURM_JOB_ID}"
REGISTRY="tasks/unified_candidate_energy_final_registry_v1.json"
V2_POLICY="${CHEMAWARE_V2_POLICY_DIR:-data/validation/gnps_noise_chemaware_v2_transfer_run_2358497/recovered_frozen_v2_policy}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
if [[ ! -s "$V2_POLICY/truthblind_policy.joblib" ]]; then
  V2_POLICY="data/validation/chemaware_multinull_deployment_safe_full_20260919"
fi

[[ ! -e "$OUT" ]] || { echo "FINAL_UNIFIED_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
[[ ! -e "$LAYER_OUT" ]] || { echo "FINAL_UNIFIED_LAYER_OUTPUT_COLLISION: $LAYER_OUT" >&2; exit 3; }

required=(
  "$GNPS_RUN/evidence_identity_disjoint.npz"
  "$GNPS_RUN/crossfit/report.json"
  "$BENCHMARK/report.json"
  "$BENCHMARK/checksums.sha256"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$V2_POLICY/truthblind_policy.joblib"
  "$V2_POLICY/report.json"
  "$STAGE1"
  "$PHASEA"
  "$ARCH"
  "$REGISTRY"
  tasks/freeze_unified_candidate_energy.py
  tasks/run_enveda180_layered_scoreblind.sbatch
  tasks/build_gnps_chemaware_kernel_cache.py
  tasks/normalize_dreams_slim_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/extend_gnps_article_score_bundle.py
  tasks/export_chemaware_v2_grand_fusion_scores.py
  tasks/build_chemaware_v2_sparse_action_evidence.py
  tasks/build_grand_unified_evidence_bundle.py
  tasks/seal_grand_unified_external_bundle.py
  tasks/evaluate_frozen_candidate_energy_external.py
  tasks/reserve_candidate_energy_enveda_opening.py
  tasks/test_unified_candidate_energy.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "FINAL_UNIFIED_MISSING_INPUT: $path" >&2; exit 4; }
done

python -m py_compile \
  tasks/freeze_unified_candidate_energy.py \
  tasks/build_gnps_chemaware_kernel_cache.py \
  tasks/evaluate_frozen_candidate_energy_external.py \
  tasks/reserve_candidate_energy_enveda_opening.py
python -m pytest tasks/test_unified_candidate_energy.py -q

# This is the only model fitting in the job: four global scalar reliabilities
# on consumed GNPS development, using the already selected preplanned arm.
python -u tasks/freeze_unified_candidate_energy.py \
  --identity-bundle "$GNPS_RUN/evidence_identity_disjoint.npz" \
  --crossfit-report "$GNPS_RUN/crossfit/report.json" \
  --module-registry "$REGISTRY" \
  --output "$OUT/frozen_model" \
  --device cuda

# Generate all spectral component scores while Enveda labels remain unopened.
# The existing v2 embeddings are reused by immutable source_row wherever valid.
ENVEDA180_SCORE_PANELS="identity_disjoint formula_disjoint" \
  bash tasks/run_enveda180_layered_scoreblind.sbatch

# Preserve both historical ChemAware embedding assets as frozen external
# comparators.  They do not enter the selected four-module decision function.
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}/unified_e180_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_ROOT"
cleanup() { rm -rf -- "$SCRATCH_ROOT"; }
trap cleanup EXIT
for item in "stage1:$STAGE1" "phase_a:$PHASEA"; do
  name="${item%%:*}"
  checkpoint="${item#*:}"
  slim="$SCRATCH_ROOT/chemaware_${name}_slim.pt"
  embedding="$SCRATCH_ROOT/chemaware_${name}_embeddings.npz"
  python -u tasks/normalize_dreams_slim_checkpoint.py \
    --input "$checkpoint" --output "$slim"
  cp "${slim}.json" "$OUT/chemaware_${name}_normalization.json"
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$BENCHMARK" \
    --checkpoint "$slim" \
    --architecture-checkpoint "$ARCH" \
    --output "$embedding" \
    --panels identity_disjoint formula_disjoint \
    --expected-selected-rows 0 --expected-records 0 \
    --batch-size 64 --label "chemaware-${name}-enveda180"
  cp "${embedding%.npz}.json" "$OUT/chemaware_${name}_encoding.json"
done

python -u tasks/extract_pair_score_cache_from_bundle.py \
  --benchmark "$BENCHMARK" \
  --score-bundle "$LAYER_OUT/method_scores_unopened.npz" \
  --method noise_v1 \
  --output "$OUT/noise_v1_cache"

python -u tasks/build_gnps_chemaware_kernel_cache.py \
  --benchmark "$BENCHMARK" \
  --official-embeddings "$LAYER_OUT/official_embeddings.npz" \
  --output "$OUT/enveda_official_geometry_kernel_cache" \
  --expected-selected-rows 0 \
  --expected-records 0

python -u tasks/export_chemaware_v2_grand_fusion_scores.py \
  --policy-dir "$V2_POLICY" \
  --token-dir "$OUT/enveda_official_geometry_kernel_cache" \
  --benchmark "$BENCHMARK" \
  --deployment-base-score-cache "$OUT/noise_v1_cache" \
  --output "$OUT/noise_v1_chemaware_v2_cache" \
  --batch-queries 256

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --base-bundle "$LAYER_OUT/method_scores_unopened.npz" \
  --embedding-method "chemaware_stage1_encoder=$SCRATCH_ROOT/chemaware_stage1_embeddings.npz" \
  --embedding-method "chemaware_phase_a_encoder=$SCRATCH_ROOT/chemaware_phase_a_embeddings.npz" \
  --pair-score-method "noise_v1_chemaware_v2_hard_promotion=$OUT/noise_v1_chemaware_v2_cache" \
  --output "$OUT/reference_scores_unopened.npz"

# From this point onward the final truth is considered opened once.  All
# component scores and the model already exist; both panels are reserved
# together so the first result cannot influence the second.
python -u tasks/reserve_candidate_energy_enveda_opening.py \
  --model "$OUT/frozen_model/model.pt" \
  --freeze-report "$OUT/frozen_model/freeze_report.json" \
  --benchmark-report "$BENCHMARK/report.json" \
  --reference-score-bundle "$OUT/reference_scores_unopened.npz" \
  --v2-selection-report "$OUT/noise_v1_chemaware_v2_cache/selection_report.json" \
  --output "$OUT/opening_global.json"

for panel in identity_disjoint formula_disjoint; do
  PANEL_PATH="$BENCHMARK/panel_${panel}.npz"
  python -u tasks/build_chemaware_v2_sparse_action_evidence.py \
    --panel "$PANEL_PATH" \
    --action-ledger "$OUT/noise_v1_chemaware_v2_cache/action_ledger_${panel}.npz" \
    --output "$OUT/chemaware_v2_action_${panel}.npz"

  python -u tasks/build_grand_unified_evidence_bundle.py \
    --panel "$panel" \
    --panel-npz "$PANEL_PATH" \
    --base-scores "$OUT/reference_scores_unopened.npz" \
    --module noise_v1 \
    --module weighted_spectral_entropy \
    --module p2b_noise_v1_frozen \
    --module chemaware_v2_action \
    --extra "chemaware_v2_action=$OUT/chemaware_v2_action_${panel}.npz" \
    --out "$OUT/component_${panel}.npz"

  python -u tasks/seal_grand_unified_external_bundle.py \
    --component-score-bundle "$OUT/component_${panel}.npz" \
    --panel-npz "$PANEL_PATH" \
    --panel "$panel" \
    --benchmark-report "$BENCHMARK/report.json" \
    --benchmark-checksums "$BENCHMARK/checksums.sha256" \
    --model "$OUT/frozen_model/model.pt" \
    --freeze-report "$OUT/frozen_model/freeze_report.json" \
    --module-registry "$REGISTRY" \
    --out "$OUT/sealed_${panel}.npz"
done

# Both panels were predeclared before either result is read.  They are opened
# consecutively with immutable per-panel ledgers; no adaptation occurs between.
python -u tasks/evaluate_frozen_candidate_energy_external.py \
  --model "$OUT/frozen_model/model.pt" \
  --freeze-report "$OUT/frozen_model/freeze_report.json" \
  --test-bundle "$OUT/sealed_identity_disjoint.npz" \
  --component-score-bundle "$OUT/component_identity_disjoint.npz" \
  --reference-score-bundle "$OUT/reference_scores_unopened.npz" \
  --module-registry "$REGISTRY" \
  --benchmark-report "$BENCHMARK/report.json" \
  --benchmark-checksums "$BENCHMARK/checksums.sha256" \
  --benchmark-panel "$BENCHMARK/panel_identity_disjoint.npz" \
  --opening-ledger "$OUT/opening_identity_disjoint.json" \
  --global-opening-ledger "$OUT/opening_global.json" \
  --out-report "$OUT/result_identity_disjoint.json" \
  --seed 20261015 \
  --device cuda

python -u tasks/evaluate_frozen_candidate_energy_external.py \
  --model "$OUT/frozen_model/model.pt" \
  --freeze-report "$OUT/frozen_model/freeze_report.json" \
  --test-bundle "$OUT/sealed_formula_disjoint.npz" \
  --component-score-bundle "$OUT/component_formula_disjoint.npz" \
  --reference-score-bundle "$OUT/reference_scores_unopened.npz" \
  --module-registry "$REGISTRY" \
  --benchmark-report "$BENCHMARK/report.json" \
  --benchmark-checksums "$BENCHMARK/checksums.sha256" \
  --benchmark-panel "$BENCHMARK/panel_formula_disjoint.npz" \
  --opening-ledger "$OUT/opening_formula_disjoint.json" \
  --global-opening-ledger "$OUT/opening_global.json" \
  --out-report "$OUT/result_formula_disjoint.json" \
  --seed 20261016 \
  --device cuda

python - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
summary={
  'status':'FINAL_UNIFIED_ENVEDA180_COMPLETE',
  'identity':json.loads((root/'result_identity_disjoint.json').read_text()),
  'formula':json.loads((root/'result_formula_disjoint.json').read_text()),
  'frozen_model_sha256':sha(root/'frozen_model/model.pt'),
  'no_post_opening_adaptation':True,
}
(root/'final_report.json').write_text(json.dumps(summary,indent=2))
opening=json.loads((root/'opening_global.json').read_text())
opening.update({
  'status':'OPENED_ONCE_BOTH_PREDECLARED_PANELS_COMPLETE_NO_RETRY',
  'identity_result_sha256':sha(root/'result_identity_disjoint.json'),
  'formula_result_sha256':sha(root/'result_formula_disjoint.json'),
})
(root/'opening_global.json').write_text(json.dumps(opening,indent=2))
print(json.dumps({'status':summary['status'],'output':str(root)},indent=2))
PY

echo "FINAL_UNIFIED_ENVEDA180_COMPLETE: $OUT"
