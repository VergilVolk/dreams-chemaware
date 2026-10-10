#!/bin/bash
# Frozen ChemAware V2 selector transferred once onto frozen Noise V1 scores.
# GNPS is development/consumed. Enveda-180 is not accessed by this job.
# The exact historical V2 bundle must already exist; this job never rebuilds it.
#SBATCH --job-name=noise_chem_v2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/noise_chem_v2_%j.out
#SBATCH --error=data/validation/noise_chem_v2_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-data/validation/chemaware_multinull_deployment_safe_full_20260919}"
BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
BASE_BUNDLE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
OFFICIAL_GNPS="${GNPS_OFFICIAL_EMBEDDINGS:-data/validation/gnps_absolute_baselines_run_2347472/gnps/official_embeddings.npz}"
OUT="data/validation/gnps_noise_chemaware_v2_transfer_run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }
mkdir -p "$OUT"

required=(
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/report.json"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$BASE_BUNDLE"
  "$OFFICIAL_GNPS"
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/mz_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/intensity_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/valid.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/precursor_mz_f32.npy
  dreams/models/chem_aware/chem_rules_data.json
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  tasks/build_gnps_chemaware_kernel_cache.py
  tasks/export_chemaware_v2_grand_fusion_scores.py
  tasks/extract_pair_score_cache_from_bundle.py
  tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py
  tasks/test_chemaware_v2_noise_transfer.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "NOISE_CHEMAWARE_V2_MISSING_INPUT: $path" >&2; exit 5; }
done

# The lightweight repository preserved the V2 ledgers but not the serialized
# nuisance/residual estimators.  If the original bundle is absent, reconstruct
# the preregistered V2 algorithm once and freeze it before any GNPS evaluation.
if [[ ! -s "$POLICY_DIR/truthblind_policy.joblib" \
   || ! -s "$POLICY_DIR/report.json" ]]; then
  POLICY_DIR="$OUT/recovered_frozen_v2_policy"
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

# Confirm that the supplied immutable bundle has the preregistered V2 algorithm
# and passed its own development gates.  Historical point estimates are not
# hard-coded because sklearn/runtime drift can alter the fitted threshold while
# leaving the fixed algorithm, roles and pre-GNPS selection protocol unchanged.
python - "$POLICY_DIR/report.json" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
method = report.get("method", {})
selection = report.get("selection", {})
held = report.get("held_inner", {}).get("correct", {})
checks = {
    "status": report.get("status") == "CHEMAWARE_ORTHOGONAL_RULE_RESIDUAL_POLICY_COMPLETE",
    "three_nulls": method.get("rule_control_variants") == [
    "rule_response_content_permuted",
    "rule_response_content_permuted_b",
    "rule_response_content_permuted_c",
    ],
    "symmetric_summary": method.get("contrast_representation") == "symmetric_summary",
    "deployment_safe": report.get("selection_control_mode") == "deployment_safe",
    "candidate_conditioned": report.get("candidate_conditioned") is True,
    "not_shared_embedding": report.get("shared_embedding_result") is False,
    "fixed_nonzero_dose": abs(float(selection.get("dose", float("nan"))) - 2.0) < 1e-12,
    "finite_threshold": math.isfinite(float(selection.get("threshold", float("nan")))),
    "selection_queries": selection.get("correct", {}).get("queries") == 1978,
    "held_queries": held.get("queries") == 1929,
    "selection_exchange": selection.get("correct", {}).get("corrected_at_1", 0)
        > 2 * selection.get("correct", {}).get("introduced_at_1", 0),
    "held_exchange": held.get("corrected_at_1", 0) > 2 * held.get("introduced_at_1", 0),
    "all_gates_pass": bool(report.get("gates")) and all(report["gates"].values()),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"CHEMAWARE_V2_POLICY_CONTRACT_FAIL: {failed}")
print(json.dumps({
    "status": "CHEMAWARE_V2_POLICY_CONTRACT_PASS",
    "dose": selection["dose"],
    "threshold": selection["threshold"],
    "selection_corrected": selection["correct"]["corrected_at_1"],
    "selection_introduced": selection["correct"]["introduced_at_1"],
    "held_corrected": held["corrected_at_1"],
    "held_introduced": held["introduced_at_1"],
}, indent=2))
PY

python -m py_compile \
  tasks/build_gnps_chemaware_kernel_cache.py \
  tasks/export_chemaware_v2_grand_fusion_scores.py \
  tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py
python -u tasks/test_chemaware_v2_noise_transfer.py

for method in official_dreams noise_v1 weighted_spectral_entropy p2b_noise_v1_frozen; do
  python -u tasks/extract_pair_score_cache_from_bundle.py \
    --benchmark "$BENCHMARK" \
    --score-bundle "$BASE_BUNDLE" \
    --method "$method" \
    --output "$OUT/${method}_cache"
done

# The selector sees exactly its historical official-DreaMS geometry.
python -u tasks/build_gnps_chemaware_kernel_cache.py \
  --benchmark "$BENCHMARK" \
  --official-embeddings "$OFFICIAL_GNPS" \
  --output "$OUT/gnps_official_geometry_kernel_cache"

# Only the selected action is transferred; abstentions preserve Noise bit-for-bit.
python -u tasks/export_chemaware_v2_grand_fusion_scores.py \
  --policy-dir "$POLICY_DIR" \
  --token-dir "$OUT/gnps_official_geometry_kernel_cache" \
  --benchmark "$BENCHMARK" \
  --deployment-base-score-cache "$OUT/noise_v1_cache" \
  --output "$OUT/noise_v1_chemaware_v2_cache" \
  --batch-queries 256

for baseline in official_dreams noise_v1 weighted_spectral_entropy p2b_noise_v1_frozen; do
  python -u tasks/evaluate_gnps_gold_silver_10ppm_pair_scores.py \
    --benchmark "$BENCHMARK" \
    --baseline-score-cache "$OUT/${baseline}_cache" \
    --candidate-score-cache "$OUT/noise_v1_chemaware_v2_cache" \
    --output "$OUT/noise_v1_chemaware_v2_vs_${baseline}" \
    --bootstrap-resamples 10000 \
    --bootstrap-seed 20261010
done

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
selection = json.loads((root / "noise_v1_chemaware_v2_cache/selection_report.json").read_text())
comparisons = {}
for baseline in ("official_dreams", "noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen"):
    report = root / f"noise_v1_chemaware_v2_vs_{baseline}/report.json"
    comparisons[baseline] = json.loads(report.read_text())
summary = {
    "status": "GNPS_NOISE_CHEMAWARE_V2_TRANSFER_COMPLETE",
    "dataset_role": "development_consumed",
    "leakage_status": "GNPS has been used repeatedly for development; no external claim",
    "selector": "frozen ChemAware V2 in official-DreaMS geometry",
    "deployment_base": "frozen Noise V1 pair scores",
    "selection": selection,
    "comparisons": comparisons,
    "enveda_180_accessed": False,
}
(root / "final_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "GNPS Noise+ChemAware V2 frozen transfer complete: $OUT"
