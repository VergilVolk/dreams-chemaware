#!/bin/bash
#SBATCH --job-name=cond_null_s1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=data/validation/cond_null_s1_%j.out
#SBATCH --error=data/validation/cond_null_s1_%j.err

# One allocation, no child submissions.  This is a development-only selector:
# all publishable generalization claims remain locked until a frozen method is
# evaluated on a genuinely unconsumed external panel.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

: "${EVIDENCE_DIR:?Submit with --export=ALL,EVIDENCE_DIR=<frozen Noise pair-evidence directory>}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
(( ${#ALLOCATED_GPUS[@]} == 1 )) || {
  echo "CONDITIONAL_NULL_GPU_CONTRACT_FAIL: expected exactly one visible GPU" >&2
  exit 3
}

OUT="${OUT_DIR:-data/validation/conditional_null_energy_stage1_run_${SLURM_JOB_ID}}"
POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-data/validation/chemaware_multinull_deployment_safe_full_20260919}"
CHEM_MANIFEST="${CHEMAWARE_MANIFEST:-data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz}"
CHEM_TOKENS="${CHEMAWARE_TOKEN_DIR:-data/validation/chemaware_corrected_manifest_tokens_v1}"
GRAPH="${CANDIDATE_GRAPH:-data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz}"
RULE_LIBRARY="${CHEMAWARE_RULE_LIBRARY:-dreams/models/chem_aware/chem_rules_data.json}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }
mkdir -p "$OUT"
trap 'echo "CONDITIONAL_NULL_STAGE1_FAILED: inspect Slurm stderr and partial output" >&2' ERR

required=(
  "$EVIDENCE_DIR/evidence.npz"
  "$EVIDENCE_DIR/report.json"
  "$GRAPH"
  "$CHEM_MANIFEST"
  "$CHEM_TOKENS/rows.npy"
  "$CHEM_TOKENS/official_embeddings_f32.npy"
  "$CHEM_TOKENS/mz_f32.npy"
  "$CHEM_TOKENS/intensity_f32.npy"
  "$CHEM_TOKENS/valid.npy"
  "$CHEM_TOKENS/precursor_mz_f32.npy"
  "$RULE_LIBRARY"
  tasks/conditional_null_energy_core.py
  tasks/export_chemaware_conditional_null_evidence.py
  tasks/train_conditional_null_energy.py
  tasks/adjudicate_conditional_null_energy.py
  tasks/test_conditional_null_energy.py
  tasks/test_conditional_null_energy_pipeline.py
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  tasks/audit_chemaware_mass_kernel_embedding.py
  tasks/chemaware_truthblind_candidate_core.py
  tasks/export_chemaware_v2_grand_fusion_scores.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "CONDITIONAL_NULL_PREFLIGHT_MISSING: $path" >&2; exit 5; }
done

if [[ ! -s "$POLICY_DIR/truthblind_policy.joblib" || ! -s "$POLICY_DIR/report.json" ]]; then
  POLICY_DIR="$OUT/rebuilt_multinull_v2_policy"
  python -u tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
    --rule-control-variants \
      rule_response_content_permuted \
      rule_response_content_permuted_b \
      rule_response_content_permuted_c \
    --contrast-representation symmetric_summary \
    --selection-control-mode deployment_safe \
    --output "$POLICY_DIR"
fi

python - "$POLICY_DIR/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
method = report.get("method", {})
assert method.get("rule_control_variants") == [
    "rule_response_content_permuted",
    "rule_response_content_permuted_b",
    "rule_response_content_permuted_c",
]
assert method.get("contrast_representation") == "symmetric_summary"
assert report.get("selection_control_mode") == "deployment_safe"
assert report.get("candidate_conditioned") is True
assert report.get("shared_embedding_result") is False
selection = report.get("selection", {})
held = report.get("held_inner", {}).get("correct", {})
assert abs(float(selection.get("dose")) - 2.0) < 1e-12
assert abs(float(selection.get("threshold")) - 0.3565609484787261) < 1e-12
assert selection.get("correct", {}).get("corrected_at_1") == 78
assert selection.get("correct", {}).get("introduced_at_1") == 17
assert held.get("corrected_at_1") == 93
assert held.get("introduced_at_1") == 17
PY

python -m py_compile \
  tasks/conditional_null_energy_core.py \
  tasks/export_chemaware_conditional_null_evidence.py \
  tasks/train_conditional_null_energy.py \
  tasks/adjudicate_conditional_null_energy.py
python -u tasks/test_conditional_null_energy.py
python -u tasks/test_conditional_null_energy_pipeline.py

python -u tasks/export_chemaware_conditional_null_evidence.py \
  --policy-dir "$POLICY_DIR" \
  --token-dir "$CHEM_TOKENS" \
  --manifest "$CHEM_MANIFEST" \
  --graph "$GRAPH" \
  --evidence "$EVIDENCE_DIR" \
  --rule-library "$RULE_LIBRARY" \
  --output "$OUT/conditional_null_chem_evidence.npz" \
  --batch-queries "${BATCH_QUERIES:-256}"

for arm in joint no_interaction spectral_only chem_only; do
  python -u tasks/train_conditional_null_energy.py \
    --evidence "$EVIDENCE_DIR" \
    --chem-evidence "$OUT/conditional_null_chem_evidence.npz" \
    --output "$OUT/arms/$arm" \
    --arm "$arm" \
    --device cuda \
    --epochs "${EPOCHS:-50}" \
    --hidden "${HIDDEN:-16}" \
    --learning-rate "${LEARNING_RATE:-0.01}" \
    --residual-bound "${RESIDUAL_BOUND:-0.25}" \
    --null-margin "${NULL_MARGIN:-0.02}" \
    --null-weight "${NULL_WEIGHT:-0.5}" \
    --trust-weight "${TRUST_WEIGHT:-0.05}"
done

python -u tasks/adjudicate_conditional_null_energy.py \
  --joint "$OUT/arms/joint" \
  --no-interaction "$OUT/arms/no_interaction" \
  --spectral-only "$OUT/arms/spectral_only" \
  --chem-only "$OUT/arms/chem_only" \
  --output "$OUT/adjudication" \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20261004

sha256sum \
  "$POLICY_DIR/truthblind_policy.joblib" \
  "$EVIDENCE_DIR/evidence.npz" \
  "$GRAPH" \
  "$CHEM_MANIFEST" \
  "$OUT/conditional_null_chem_evidence.npz" \
  "$OUT/arms/joint/model.pt" \
  "$OUT/adjudication/report.json" > "$OUT/SHA256SUMS.txt"

echo "CONDITIONAL_NULL_STAGE1_COMPLETE: $OUT"
echo "Development-only verdict: $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$OUT/adjudication/report.json")"
