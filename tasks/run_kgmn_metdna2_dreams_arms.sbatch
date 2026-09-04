#!/bin/bash
#SBATCH --job-name=kgmn_dreams_arms
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=kgmn_dreams_arms_%j.out
#SBATCH --error=kgmn_dreams_arms_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

BASELINE=data/validation/kgmn_metdna2_200std_author_baseline/frozen_author_baseline.json
CALIBRATION=data/validation/kgmn_dreams_edge_calibration_official_20260831/calibration_artifact.json
CALIBRATION_REPORT=data/validation/kgmn_dreams_edge_calibration_official_20260831/report.json
OVERLAY=data/validation/kgmn_metdna2_dreams_overlay_20260831
RLIB=data/validation/kgmn_metdna2_dreams_overlay_rlib_20260831
EMBEDDINGS=data/validation/kgmn_metdna2_200std_official_embeddings_20260831
NOOP=data/validation/kgmn_metdna2_200std_noop_overlay_20260831
OFFICIAL=data/validation/kgmn_metdna2_200std_official_dreams_20260831
INTERSECTION=data/validation/kgmn_metdna2_200std_author_official_intersection_20260831
DECISION=data/validation/kgmn_metdna2_200std_dreams_arm_decision_20260831.json

for required in "$BASELINE" "$CALIBRATION" "$CALIBRATION_REPORT"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required frozen input: $required" >&2
    exit 1
  fi
done

python - "$CALIBRATION" "$CALIBRATION_REPORT" <<'PY'
import hashlib, json, pathlib, sys
artifact, report_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "kgmn_dreams_edge_calibration_complete":
    raise SystemExit("invalid DreaMS edge calibration report")
digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
if report.get("provenance", {}).get("calibration_artifact_sha256") != digest:
    raise SystemExit("DreaMS edge calibration artifact hash mismatch")
if report.get("gates", {}).get("official_dreams_eligible_for_dynamic_propagation_test") is not True:
    raise SystemExit(
        "official DreaMS failed the preregistered component/formula-isolated edge gate; "
        "formal KGMN propagation arms are forbidden"
    )
PY
for output in "$OVERLAY" "$RLIB" "$EMBEDDINGS" "$NOOP" "$OFFICIAL" "$INTERSECTION" "$DECISION"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite formal output: $output" >&2
    exit 1
  fi
done

python -u tasks/build_kgmn_dreams_edge_overlay.py \
  --source third_party/MetDNA2 \
  --output "$OVERLAY"

mkdir -p "$RLIB"
R CMD INSTALL --library="$RLIB" "$OVERLAY"
export R_LIBS_USER="$(pwd)/$RLIB${R_LIBS_USER:+:$R_LIBS_USER}"
Rscript -e 'stopifnot(as.character(packageVersion("MetDNA2")) == "1.2.10")'
Rscript tasks/preflight_kgmn_metdna2_dp.R

python -u tasks/export_kgmn_200std_dreams_embeddings.py \
  --msp third_party/MetDNA2/inst/extdata/spectra_200STD_neg_200805.msp \
  --output-dir "$EMBEDDINGS" \
  --device cuda \
  --batch-size 128

run_arm () {
  local arm="$1"
  local output="$2"
  local temporary_log
  temporary_log="$(mktemp)"
  Rscript tasks/run_kgmn_metdna2_200std_dreams_arm.R \
    "$OVERLAY" "$output" \
    "$EMBEDDINGS/official_200std_embeddings.csv.gz" \
    "$CALIBRATION" "$arm" 2>&1 | tee "$temporary_log"
  mv "$temporary_log" "$output/run.log.txt"
  python -u tasks/freeze_kgmn_metdna2_dreams_arm.py \
    --run-dir "$output" \
    --author-baseline "$BASELINE" \
    --arm "$arm" \
    --calibration "$CALIBRATION" \
    --calibration-report "$CALIBRATION_REPORT" \
    --embedding-file "$EMBEDDINGS/official_200std_embeddings.csv.gz" \
    --embedding-report "$EMBEDDINGS/report.json" \
    --overlay-manifest "$OVERLAY/DREAMS_EDGE_OVERLAY_MANIFEST.json"
}

# This arm must reproduce every author result table before either experimental
# arm is allowed to execute.  It catches overlay, parallel-worker and scoring
# interface changes independently of DreaMS performance.
run_arm noop_author "$NOOP"
run_arm official_dreams "$OFFICIAL"
run_arm author_official_intersection "$INTERSECTION"

python -u tasks/summarize_kgmn_metdna2_dreams_arms.py \
  --noop "$NOOP/frozen_arm.json" \
  --official "$OFFICIAL/frozen_arm.json" \
  --intersection "$INTERSECTION/frozen_arm.json" \
  --output "$DECISION"

echo "=== KGMN/MetDNA2 matched-protocol DreaMS arms completed ==="
