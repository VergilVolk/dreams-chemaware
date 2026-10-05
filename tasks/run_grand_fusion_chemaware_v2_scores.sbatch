#!/bin/bash
# One-shot external transfer and aligned train-side export of frozen ChemAware V2.
# No fitting, threshold selection or label access occurs before the evaluator.
#SBATCH --job-name=grand_chem_v2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/grand_chem_v2_%j.out
#SBATCH --error=data/validation/grand_chem_v2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

: "${EVIDENCE_DIR:?Set EVIDENCE_DIR to frozen MassSpecGym pair evidence}"
: "${MSG_FUSION_OOF:?Set MSG_FUSION_OOF to formula-OOF pair scores}"
: "${MSG_FUSION_GNPS:?Set MSG_FUSION_GNPS to frozen GNPS pair cache}"
POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-data/validation/chemaware_multinull_deployment_safe_full_20260919}"
CHEM_MANIFEST="${CHEMAWARE_MANIFEST:-data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz}"
CHEM_TOKENS="${CHEMAWARE_TOKEN_DIR:-data/validation/chemaware_corrected_manifest_tokens_v1}"
BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
BASE_BUNDLE="${SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
OFFICIAL_GNPS="${GNPS_OFFICIAL_EMBEDDINGS:-data/validation/gnps_absolute_baselines_run_2347472/gnps/official_embeddings.npz}"
OUT="data/validation/grand_fusion_chemaware_v2_run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }
mkdir -p "$OUT"

required=(
  "$EVIDENCE_DIR/evidence.npz" "$EVIDENCE_DIR/report.json"
  "$MSG_FUSION_OOF" "$MSG_FUSION_GNPS/report.json"
  "$CHEM_MANIFEST" "$CHEM_TOKENS/rows.npy" "$CHEM_TOKENS/official_embeddings_f32.npy"
  "$CHEM_TOKENS/mz_f32.npy" "$CHEM_TOKENS/intensity_f32.npy"
  "$CHEM_TOKENS/valid.npy" "$CHEM_TOKENS/precursor_mz_f32.npy"
  "$BENCHMARK/spectra.mgf" "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz" "$OFFICIAL_GNPS" "$BASE_BUNDLE"
  dreams/models/chem_aware/chem_rules_data.json
  tasks/audit_chemaware_candidate_evidence_policy.py
  tasks/audit_chemaware_mass_kernel_embedding.py
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  tasks/build_gnps_chemaware_kernel_cache.py
  tasks/export_chemaware_v2_grand_fusion_scores.py
  tasks/extend_grand_fusion_pair_evidence.py
  tasks/extend_gnps_article_score_bundle.py
  tasks/extract_pair_score_cache_from_bundle.py
  tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py
  tasks/adjudicate_grand_fusion_external.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "GRAND_CHEM_V2_MISSING_INPUT: $path" >&2; exit 5; }
done

# The V2 deployment bundle was intentionally not committed with the lightweight
# validation ledgers.  Reconstruct exactly the preregistered multi-null policy
# when it is absent on a compute node; never substitute the older single-null
# candidate policy under a V2 label.
if [[ ! -s "$POLICY_DIR/truthblind_policy.joblib" \
   || ! -s "$POLICY_DIR/report.json" ]]; then
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

test -s "$POLICY_DIR/truthblind_policy.joblib"
test -s "$POLICY_DIR/report.json"
python - "$POLICY_DIR/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
method = report.get("method", {})
expected_nulls = [
    "rule_response_content_permuted",
    "rule_response_content_permuted_b",
    "rule_response_content_permuted_c",
]
assert method.get("rule_control_variants") == expected_nulls
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
  tasks/build_gnps_chemaware_kernel_cache.py \
  tasks/export_chemaware_v2_grand_fusion_scores.py \
  tasks/extend_grand_fusion_pair_evidence.py \
  tasks/extract_pair_score_cache_from_bundle.py \
  tasks/adjudicate_grand_fusion_external.py

# Export the frozen V2 action onto the exact 6.2M-edge MassSpecGym ledger.
python -u tasks/export_chemaware_v2_grand_fusion_scores.py \
  --policy-dir "$POLICY_DIR" --token-dir "$CHEM_TOKENS" \
  --manifest "$CHEM_MANIFEST" --evidence "$EVIDENCE_DIR" \
  --output "$OUT/chemaware_v2_massspecgym_scores.npz" --batch-queries 256

python -u tasks/extend_grand_fusion_pair_evidence.py \
  --evidence "$EVIDENCE_DIR" \
  --expert "noise_msg_fusion_stack_v1=$MSG_FUSION_OOF" \
  --expert "chemaware_v2_reranker=$OUT/chemaware_v2_massspecgym_scores.npz" \
  --output "$OUT/evidence_with_chemaware"

# Build GNPS peak-rule inputs without opening labels, then freeze V2 actions.
python -u tasks/build_gnps_chemaware_kernel_cache.py \
  --benchmark "$BENCHMARK" --official-embeddings "$OFFICIAL_GNPS" \
  --output "$OUT/gnps_kernel_cache"

python -u tasks/export_chemaware_v2_grand_fusion_scores.py \
  --policy-dir "$POLICY_DIR" --token-dir "$OUT/gnps_kernel_cache" \
  --benchmark "$BENCHMARK" --output "$OUT/gnps_pair_score_cache" \
  --batch-queries 256

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" --base-bundle "$BASE_BUNDLE" \
  --pair-score-method "noise_msg_fusion_stack_v1=$MSG_FUSION_GNPS" \
  --pair-score-method "chemaware_v2_reranker=$OUT/gnps_pair_score_cache" \
  --output "$OUT/method_scores_with_chemaware.npz"

for method in weighted_spectral_entropy p2b_noise_v1_frozen noise_v1; do
  python -u tasks/extract_pair_score_cache_from_bundle.py \
    --benchmark "$BENCHMARK" --score-bundle "$BASE_BUNDLE" \
    --method "$method" --output "$OUT/${method}_cache"
  python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
    --benchmark "$BENCHMARK" --baseline-score-cache "$OUT/${method}_cache" \
    --candidate-score-cache "$OUT/gnps_pair_score_cache" \
    --output "$OUT/chemaware_vs_${method}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261004
done

python -u tasks/adjudicate_grand_fusion_external.py \
  --mode reranker \
  --wse "$OUT/chemaware_vs_weighted_spectral_entropy/report.json" \
  --p2b "$OUT/chemaware_vs_p2b_noise_v1_frozen/report.json" \
  --noise "$OUT/chemaware_vs_noise_v1/report.json" \
  --output "$OUT/adjudication.json"

sha256sum \
  "$POLICY_DIR/truthblind_policy.joblib" \
  "$OUT/chemaware_v2_massspecgym_scores.npz" \
  "$OUT/method_scores_with_chemaware.npz" \
  "$OUT/adjudication.json" > "$OUT/SHA256SUMS.txt"

echo "Grand-fusion ChemAware V2 export complete: $OUT"
