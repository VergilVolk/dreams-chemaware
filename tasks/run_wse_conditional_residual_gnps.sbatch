#!/bin/bash
# One GNPS-development experiment with exactly two constructions:
# A) WSE + Noise/P2b spectral residual
# B) A + continuous conditional ChemAware correct-minus-null residual
# Enveda-180 is consumed and is neither read nor scored by this job.
#SBATCH --job-name=wse_cond_resid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/wse_cond_resid_%j.out
#SBATCH --error=data/validation/wse_cond_resid_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE_RUN="${UNIFIED_GNPS_SOURCE_RUN:-data/validation/unified_candidate_energy_run_2364137}"
V2_RUN="${CHEMAWARE_V2_TRANSFER_RUN:-data/validation/gnps_noise_chemaware_v2_transfer_run_2358497}"
IDENTITY_PANEL="${GNPS_IDENTITY_PANEL:-data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz}"
FORMULA_PANEL="${GNPS_FORMULA_PANEL:-data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz}"
TOKEN_DIR="${GNPS_CHEMAWARE_TOKEN_DIR:-$V2_RUN/gnps_official_geometry_kernel_cache}"
RULE_LIBRARY="${CHEMAWARE_RULE_LIBRARY:-dreams/models/chem_aware/chem_rules_data.json}"
OUT="data/validation/wse_conditional_residual_run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "WSE_CONDITIONAL_RESIDUAL_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

# Reuse an exact serialized V2 policy.  These candidates are ordered from the
# pre-Enveda GNPS transfer to the later copy; no Enveda result is inspected.
POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-}"
if [[ -z "$POLICY_DIR" ]]; then
  for candidate in \
    "$V2_RUN/recovered_frozen_v2_policy" \
    data/validation/chemaware_multinull_deployment_safe_full_20260919 \
    data/validation/unified_candidate_energy_enveda_final_run_2372638/recovered_frozen_v2_policy
  do
    if [[ -s "$candidate/truthblind_policy.joblib" && -s "$candidate/report.json" ]]; then
      POLICY_DIR="$candidate"
      break
    fi
  done
fi
[[ -n "$POLICY_DIR" ]] || {
  echo "WSE_CONDITIONAL_RESIDUAL_MISSING: frozen V2 policy directory" >&2
  exit 4
}

required=(
  "$SOURCE_RUN/evidence_identity_disjoint.npz"
  "$SOURCE_RUN/evidence_formula_disjoint.npz"
  "$SOURCE_RUN/crossfit/report.json"
  "$IDENTITY_PANEL"
  "$FORMULA_PANEL"
  "$TOKEN_DIR/rows.npy"
  "$TOKEN_DIR/official_embeddings_f32.npy"
  "$TOKEN_DIR/mz_f32.npy"
  "$TOKEN_DIR/intensity_f32.npy"
  "$TOKEN_DIR/valid.npy"
  "$TOKEN_DIR/precursor_mz_f32.npy"
  "$POLICY_DIR/truthblind_policy.joblib"
  "$POLICY_DIR/report.json"
  "$RULE_LIBRARY"
  tasks/conditional_null_energy_core.py
  tasks/export_gnps_chemaware_continuous_null_evidence.py
  tasks/train_wse_conditional_residual.py
  tasks/test_wse_conditional_residual.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "WSE_CONDITIONAL_RESIDUAL_MISSING: $path" >&2; exit 4; }
done

# A compact location table only.  Missing local paths are UNKNOWN, not
# retroactively declared unrun.  No Enveda report or score file is opened.
python - "$OUT/reuse_inventory.json" "$SOURCE_RUN" "${CHEMAWARE_CONTINUOUS_CACHE:-}" <<'PY'
import glob, json, pathlib, sys
destination, source_run, supplied_cache = sys.argv[1:]
def locate(pattern, reports):
    paths=sorted((pathlib.Path(p) for p in glob.glob(pattern)), key=lambda p:p.stat().st_mtime, reverse=True)
    if not paths:
        return {"status":"UNKNOWN_NOT_LOCATED","path":None,"reusable":False}
    root=paths[0]
    for rel in reports:
        report=root/rel
        if report.is_file():
            try: status=json.loads(report.read_text()).get("status","REPORT_FOUND")
            except Exception: status="REPORT_UNREADABLE"
            return {"status":status,"path":str(root),"report":str(report),"reusable":"COMPLETE" in status or "PASS" in status}
    return {"status":"LOCATED_INCOMPLETE_OR_UNKNOWN","path":str(root),"reusable":False}
inventory={
 "conditional_null_energy":locate("data/validation/conditional_null_energy_stage1_run_*",("adjudication/report.json","report.json")),
 "chemaware_soft_residual":locate("data/validation/chemaware_soft_residual_fusion_policy_*",("report.json",)),
 "old_unified_candidate_energy":locate(source_run,("crossfit/report.json","report.json")),
 "continuous_chem_cache":({"status":"SUPPLIED_FOR_REUSE","path":supplied_cache,"reusable":True} if supplied_cache else {"status":"NOT_SUPPLIED_WILL_COMPUTE_ONLY_MISSING_EVIDENCE","path":None,"reusable":False}),
 "enveda180": {"status":"CONSUMED_NOT_ACCESSED_BY_THIS_JOB","path":None,"reusable":False},
}
pathlib.Path(destination).write_text(json.dumps(inventory,indent=2))
print(json.dumps(inventory,indent=2))
PY

python -m py_compile \
  tasks/export_gnps_chemaware_continuous_null_evidence.py \
  tasks/train_wse_conditional_residual.py
python -u tasks/test_wse_conditional_residual.py

if [[ -n "${CHEMAWARE_CONTINUOUS_CACHE:-}" ]]; then
  CHEM_CACHE="$CHEMAWARE_CONTINUOUS_CACHE"
  for path in \
    "$CHEM_CACHE/report.json" \
    "$CHEM_CACHE/continuous_chem_identity_disjoint.npz" \
    "$CHEM_CACHE/continuous_chem_formula_disjoint.npz"
  do
    [[ -s "$path" ]] || { echo "WSE_CONDITIONAL_RESIDUAL_BAD_CACHE: $path" >&2; exit 5; }
  done
else
  CHEM_CACHE="$OUT/continuous_chem_evidence"
  python -u tasks/export_gnps_chemaware_continuous_null_evidence.py \
    --policy-dir "$POLICY_DIR" \
    --token-dir "$TOKEN_DIR" \
    --rule-library "$RULE_LIBRARY" \
    --panel "identity_disjoint=$IDENTITY_PANEL" \
    --panel "formula_disjoint=$FORMULA_PANEL" \
    --output "$CHEM_CACHE" \
    --batch-queries 256
fi

python -u tasks/train_wse_conditional_residual.py \
  --identity-bundle "$SOURCE_RUN/evidence_identity_disjoint.npz" \
  --formula-bundle "$SOURCE_RUN/evidence_formula_disjoint.npz" \
  --identity-chem "$CHEM_CACHE/continuous_chem_identity_disjoint.npz" \
  --formula-chem "$CHEM_CACHE/continuous_chem_formula_disjoint.npz" \
  --old-report "$SOURCE_RUN/crossfit/report.json" \
  --output "$OUT/modeling" \
  --device cuda

echo "WSE_CONDITIONAL_RESIDUAL_COMPLETE_NO_ENVEDA_ACCESS: $OUT"
