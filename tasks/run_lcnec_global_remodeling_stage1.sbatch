#!/bin/bash
# LCNEC global remodeling Stage-1 (falsification run).
# Orchestration-layer guards added by GLM review 2026-10-06:
#   (a) no explicit partition (cluster policy; keep --gpus=1 only);
#   (b) post-builder universe-count gate (must be exactly 263 spectra);
#   (c) post-encoder manifest-name gate (263 rows matching _family_<id>).
# The frozen analysis pre-registration in analyze_lcnec_global_remodeling_stage1.py
# is unchanged.
#SBATCH --job-name=lcnec_remodel_s1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/lcnec_remodel_s1_%j.out
#SBATCH --error=data/validation/lcnec_remodel_s1_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_DIR="data/validation/lcnec_global_remodeling_stage1/run_${SLURM_JOB_ID}"
RAW_ZIP="${LCNEC_RAW_ZIP:-data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_HSST3n_mzML_public.zip}"
OVERVIEW="${LCNEC_OVERVIEW:-data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt}"

required=(
  "$RAW_ZIP"
  "$OVERVIEW"
  dreams/models/pretrained/ssl_model_server.pt
  data/e1/official_embedding_slim.pt
  tasks/audit_lcnec_hsst3n_qc_headroom.py
  tasks/build_lcnec_hsst3n_priority_ms2.py
  tasks/encode_mona_neg_library.py
  tasks/quantify_lcnec_hsst3n_all_qc_eic.py
  tasks/analyze_lcnec_global_remodeling_stage1.py
  tasks/test_analyze_lcnec_global_remodeling_stage1.py
)
missing=0
for path in "${required[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "LCNEC_GLOBAL_REMODELING_PREFLIGHT_MISSING: $path" >&2
    missing=1
  fi
done
(( missing == 0 )) || exit 5

[[ ! -e "$RUN_DIR" ]] || { echo "Output collision: $RUN_DIR" >&2; exit 4; }
mkdir -p "$RUN_DIR"
trap 'echo "LCNEC_GLOBAL_REMODELING_STAGE1_FAILED: $RUN_DIR" >&2' ERR

python -m py_compile \
  tasks/quantify_lcnec_hsst3n_all_qc_eic.py \
  tasks/analyze_lcnec_global_remodeling_stage1.py \
  tasks/test_analyze_lcnec_global_remodeling_stage1.py
python tasks/test_analyze_lcnec_global_remodeling_stage1.py

python -u tasks/audit_lcnec_hsst3n_qc_headroom.py \
  --zip "$RAW_ZIP" \
  --overview "$OVERVIEW" \
  --output-dir "$RUN_DIR/qc_headroom"

LEDGER_FAMILIES="$(python -c \
  'import json,sys; r=json.load(open(sys.argv[1])); print(r["all_three_qualified_families"])' \
  "$RUN_DIR/qc_headroom/report.json")"
if [[ "$LEDGER_FAMILIES" != "263" ]]; then
  echo "LCNEC_REMODELING_LEDGER_COUNT_MISMATCH: $LEDGER_FAMILIES of 263 QC-qualified families" >&2
  exit 6
fi

python -u tasks/build_lcnec_hsst3n_priority_ms2.py \
  --target-mode all_qc_qualified \
  --top-n 0 \
  --qc-ledger "$RUN_DIR/qc_headroom/precursor_family_ledger.csv" \
  --zip "$RAW_ZIP" \
  --overview "$OVERVIEW" \
  --output-dir "$RUN_DIR/query"

SPECTRUM_COUNT="$(grep -c "BEGIN IONS" "$RUN_DIR/query/priority_dark_modules.mgf")"
if [[ "$SPECTRUM_COUNT" != "263" ]]; then
  echo "LCNEC_REMODELING_UNIVERSE_COUNT_MISMATCH: builder emitted $SPECTRUM_COUNT of 263 family spectra" >&2
  exit 6
fi

python -u tasks/encode_mona_neg_library.py \
  --mgf "$RUN_DIR/query/priority_dark_modules.mgf" \
  --out "$RUN_DIR/query/dreams_embeddings" \
  --device cuda

EMBED_MANIFEST="$RUN_DIR/query/dreams_embeddings/manifest.csv"
if [[ ! -s "$EMBED_MANIFEST" ]]; then
  echo "LCNEC_REMODELING_EMBEDDING_MANIFEST_MISSING: $EMBED_MANIFEST" >&2
  exit 6
fi
NAME_COLUMN="$(head -1 "$EMBED_MANIFEST" | tr ',' '\n' | grep -n '^name$' | cut -d: -f1)"
if [[ -z "$NAME_COLUMN" ]]; then
  echo "LCNEC_REMODELING_EMBEDDING_MANIFEST_NO_NAME_COLUMN: $EMBED_MANIFEST" >&2
  exit 6
fi
NAMED_ROWS="$(awk -F',' -v col="$NAME_COLUMN" 'NR>1 && $col ~ /_family_[0-9]+$/ {n++} END {print n+0}' "$EMBED_MANIFEST")"
if [[ "$NAMED_ROWS" != "263" ]]; then
  echo "LCNEC_REMODELING_EMBEDDING_NAME_MISMATCH: $NAMED_ROWS of 263 rows match _family_<id> pattern" >&2
  exit 6
fi

python -u tasks/quantify_lcnec_hsst3n_all_qc_eic.py \
  --zip "$RAW_ZIP" \
  --overview "$OVERVIEW" \
  --targets "$RUN_DIR/qc_headroom/precursor_family_ledger.csv" \
  --output-dir "$RUN_DIR/eic"

python -u tasks/analyze_lcnec_global_remodeling_stage1.py \
  --matrix "$RUN_DIR/eic/all_qc_family_eic_matrix.npz" \
  --embedding-dir "$RUN_DIR/query/dreams_embeddings" \
  --output-dir "$RUN_DIR/analysis" \
  --k-values 2,3,5,8 \
  --primary-k 3 \
  --sign-flips 10000 \
  --seed 20261006

echo "LCNEC_GLOBAL_REMODELING_STAGE1_COMPLETE: $RUN_DIR"
