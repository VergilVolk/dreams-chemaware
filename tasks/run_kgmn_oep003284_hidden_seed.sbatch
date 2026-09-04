#!/bin/bash
#SBATCH --job-name=kgmn_oep_hidden
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=kgmn_oep_hidden_%j.out
#SBATCH --error=kgmn_oep_hidden_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

INPUT_ROOT=data/reference/OEP003284
PREFLIGHT=data/validation/kgmn_oep003284_input_preflight_${SLURM_JOB_ID}.json
CONTRACT=data/validation/kgmn_external_validation_contract_20260831
IDENTIFIER_CONTRACT=data/validation/kgmn_oep003284_author_identifier_contract_20260831
CALIBRATION=data/validation/kgmn_dreams_edge_calibration_official_20260831/calibration_artifact.json
CALIBRATION_REPORT=data/validation/kgmn_dreams_edge_calibration_official_20260831/report.json
ROOT=data/validation/kgmn_oep003284_hidden_seed_20260831
AUTHOR_SOURCE=third_party/MetDNA2
OVERLAY="$ROOT/overlay_source"
AUTHOR_RLIB="$ROOT/author_rlib"
OVERLAY_RLIB="$ROOT/overlay_rlib"
ORIGINAL_R_LIBS_USER="${R_LIBS_USER:-}"

for required in "$CONTRACT/report.json" "$CONTRACT/hidden_seed_splits.csv.gz" \
  "$CONTRACT/level1_seed_universe.csv.gz" "$IDENTIFIER_CONTRACT/report.json" \
  "$CALIBRATION" "$CALIBRATION_REPORT"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing frozen prerequisite: $required" >&2
    exit 1
  fi
done
if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite formal output: $ROOT" >&2
  exit 1
fi

python -u tasks/preflight_kgmn_oep003284_inputs.py \
  --input-root "$INPUT_ROOT" \
  --output "$PREFLIGHT" \
  --require-ready \
  --require-dreams-edge \
  --allow-author-mapped-mzxml

python - "$CALIBRATION" "$CALIBRATION_REPORT" "$CONTRACT/report.json" \
  "$IDENTIFIER_CONTRACT/report.json" <<'PY'
import hashlib, json, pathlib, sys
artifact, calibration_report_path, contract_path, identifier_path = map(pathlib.Path, sys.argv[1:])
calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
if calibration_report.get("status") != "kgmn_dreams_edge_calibration_complete":
    raise SystemExit("invalid DreaMS edge calibration report")
if hashlib.sha256(artifact.read_bytes()).hexdigest() != calibration_report.get("provenance", {}).get("calibration_artifact_sha256"):
    raise SystemExit("DreaMS edge calibration artifact hash mismatch")
if calibration_report.get("gates", {}).get("official_dreams_eligible_for_dynamic_propagation_test") is not True:
    raise SystemExit("official DreaMS edge calibration did not pass the frozen dynamic-propagation gate")
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract.get("status") != "kgmn_external_validation_contract_frozen":
    raise SystemExit("external validation contract status mismatch")
if contract.get("primary_protocol", {}).get("repeats") != 10:
    raise SystemExit("formal hidden-seed protocol must contain exactly 10 repeats")
identifier = json.loads(identifier_path.read_text(encoding="utf-8"))
if identifier.get("status") != "kgmn_oep003284_author_identifier_contract_frozen":
    raise SystemExit("author identifier contract status mismatch")
if identifier.get("level1", {}).get("peak_rows") != 80 or identifier.get("level1", {}).get("identities") != 42:
    raise SystemExit("author identifier contract Level-1 universe drift")
if identifier.get("level1", {}).get("all_author_peak_names_found") is not True:
    raise SystemExit("author identifier contract did not reconcile every Level-1 peak")
PY

mkdir -p "$ROOT" "$AUTHOR_RLIB" "$OVERLAY_RLIB"
cp "$PREFLIGHT" "$ROOT/input_preflight.json"
python -u tasks/build_kgmn_dreams_edge_overlay.py --source "$AUTHOR_SOURCE" --output "$OVERLAY"
R CMD INSTALL --library="$AUTHOR_RLIB" "$AUTHOR_SOURCE"
R CMD INSTALL --library="$OVERLAY_RLIB" "$OVERLAY"

panel_paths () {
  local polarity="$1"
  python - "$PREFLIGHT" "$polarity" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
panel = report["panels"][sys.argv[2]]
if panel.get("dreams_edge_mode") not in {
    "direct_identifier_preserving_msp_or_mgf",
    "metdna2_author_mapped_mzxml_cache",
}:
    raise SystemExit("unsupported DreaMS feature-edge bridge mode")
print(panel["directory"])
print(panel["dreams_edge_mode"])
PY
}

for polarity in positive negative; do
  mapfile -t paths < <(panel_paths "$polarity")
  if [[ "${#paths[@]}" -ne 2 ]]; then
    echo "Failed to resolve $polarity input directory and bridge mode" >&2
    exit 1
  fi
  panel_dir="${paths[0]}"
  bridge_mode="${paths[1]}"
  initial_dir="$ROOT/full_initial/$polarity"
  mapped_dir="$ROOT/mapped_spectra/$polarity"
  mapped_msp="$mapped_dir/author_mapped.msp"
  embedding_dir="$ROOT/embeddings/$polarity"
  R_LIBS_USER="$AUTHOR_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/run_kgmn_metdna2_46std_initial_seed.R "$AUTHOR_SOURCE" "$panel_dir" "$initial_dir"
  if [[ "$bridge_mode" != "metdna2_author_mapped_mzxml_cache" && \
        "$bridge_mode" != "direct_identifier_preserving_msp_or_mgf" ]]; then
    echo "Unsupported DreaMS bridge mode: $bridge_mode" >&2
    exit 1
  fi
  Rscript tasks/export_kgmn_metdna2_mapped_ms2.R \
    "$initial_dir" "$mapped_msp" "$mapped_dir/report.json"
  python -u tasks/export_kgmn_feature_dreams_embeddings.py \
    --spectra "$mapped_msp" \
    --ms1-table "$initial_dir/data.csv" \
    --output-dir "$embedding_dir" \
    --device cuda \
    --batch-size 128
done

run_author () {
  local repeat="$1"
  local polarity="$2"
  local state="$ROOT/runs/author/repeat_$(printf '%02d' "$repeat")_$polarity"
  R_LIBS_USER="$AUTHOR_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/prepare_kgmn_hidden_seed_state.R \
      "$ROOT/full_initial/$polarity" "$CONTRACT" "$repeat" "$polarity" "$state"
  R_LIBS_USER="$AUTHOR_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/run_kgmn_metdna2_46std_hidden_seed_author.R "$AUTHOR_SOURCE" "$state"
  R_LIBS_USER="$AUTHOR_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/export_kgmn_hidden_seed_predictions.R \
      "$state" "$CONTRACT" "$repeat" "$polarity" \
      "$ROOT/prediction_shards/author/repeat_$(printf '%02d' "$repeat")_$polarity.csv"
}

run_edge_arm () {
  local arm="$1"
  local repeat="$2"
  local polarity="$3"
  local state="$ROOT/runs/$arm/repeat_$(printf '%02d' "$repeat")_$polarity"
  R_LIBS_USER="$OVERLAY_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/prepare_kgmn_hidden_seed_state.R \
      "$ROOT/full_initial/$polarity" "$CONTRACT" "$repeat" "$polarity" "$state"
  R_LIBS_USER="$OVERLAY_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/run_kgmn_metdna2_46std_hidden_seed_arm.R \
      "$OVERLAY" "$state" "$ROOT/embeddings/$polarity/official_feature_embeddings.csv.gz" \
      "$CALIBRATION" "$arm"
  R_LIBS_USER="$OVERLAY_RLIB${ORIGINAL_R_LIBS_USER:+:$ORIGINAL_R_LIBS_USER}" \
    Rscript tasks/export_kgmn_hidden_seed_predictions.R \
      "$state" "$CONTRACT" "$repeat" "$polarity" \
      "$ROOT/prediction_shards/$arm/repeat_$(printf '%02d' "$repeat")_$polarity.csv"
}

# Two external no-op runs are a technical identity gate.  The consumed 200STD
# workflow already requires full author/no-op equality; here we additionally
# test both OEP polarities before any experimental arm is allowed to run.
for polarity in positive negative; do
  run_author 0 "$polarity"
  run_edge_arm noop_author 0 "$polarity"
  python -u tasks/audit_kgmn_hidden_seed_noop.py \
    --author-run "$ROOT/runs/author/repeat_00_$polarity" \
    --noop-run "$ROOT/runs/noop_author/repeat_00_$polarity" \
    --output "$ROOT/noop_audit_$polarity.json"
done

for repeat in $(seq 0 9); do
  for polarity in positive negative; do
    if [[ "$repeat" -ne 0 ]]; then
      run_author "$repeat" "$polarity"
    fi
    run_edge_arm official_dreams "$repeat" "$polarity"
    run_edge_arm author_official_intersection "$repeat" "$polarity"
  done
done

for arm in author official_dreams author_official_intersection; do
  python -u tasks/combine_kgmn_hidden_seed_predictions.py \
    --shard-dir "$ROOT/prediction_shards/$arm" \
    --output "$ROOT/${arm}_predictions.csv.gz" \
    --report "$ROOT/${arm}_predictions.report.json"
done

python -u tasks/evaluate_kgmn_hidden_seed_recovery.py \
  --contract-dir "$CONTRACT" \
  --author-predictions "$ROOT/author_predictions.csv.gz" \
  --candidate-predictions "$ROOT/official_dreams_predictions.csv.gz" \
  --candidate-arm official_dreams \
  --output-dir "$ROOT/evaluation_official_dreams"
python -u tasks/evaluate_kgmn_hidden_seed_recovery.py \
  --contract-dir "$CONTRACT" \
  --author-predictions "$ROOT/author_predictions.csv.gz" \
  --candidate-predictions "$ROOT/author_official_intersection_predictions.csv.gz" \
  --candidate-arm author_official_intersection \
  --output-dir "$ROOT/evaluation_author_official_intersection"

python -u tasks/summarize_kgmn_oep003284_hidden_seed.py \
  --root "$ROOT" \
  --output "$ROOT/final_decision.json"

echo "=== KGMN OEP003284 hidden-seed experiment completed: $(date) ==="
