#!/bin/bash
#SBATCH --job-name=mtbls_bio_close
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=mtbls_bio_close_%j.out
#SBATCH --error=mtbls_bio_close_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

echo "=== MTBLS13729 biology closure started: $(date) ==="

GRAPH_DIR=data/mtbls13729/bioaware_global_peak_formal_20260829
FAMILY_DIR=data/mtbls13729/frozen_ion_family_audit_20260829
ANCHOR_DIR=data/mtbls13729/c20_4_anchor_ms2_audit_20260829
CLASS_DIR=data/mtbls13729/acylcarnitine_panel_20260829
CLOSURE_REPORT=data/mtbls13729/biology_closure_20260829/report.json

# Reuse a completed uniformly re-quantified matrix.  If it is absent, build it
# with the same fixed peak-resolved settings used by the frozen biology stage.
if [[ ! -f data/mtbls13729/ms1_eic_requant/report.json ]] || \
   [[ ! -f data/mtbls13729/ms1_eic_requant/pos_rp__eic_auc_matrix.csv.gz ]] || \
   [[ ! -f data/mtbls13729/ms1_eic_requant/neg_rp__eic_auc_matrix.csv.gz ]]; then
  python -u tasks/requantify_mtbls13729_targeted_eic.py \
    --panels neg_rp pos_rp \
    --resolve-local-peaks \
    --max-apex-delta-sec 12
fi

# Rebuild the global peak graph from uniformly re-quantified EICs.  The graph
# builder refuses discovery matrices unless explicitly allowed.
if [[ ! -f "$GRAPH_DIR/report.json" ]]; then
  python -u tasks/build_mtbls13729_global_peak_network.py \
    --output-dir "$GRAPH_DIR"
else
  echo "[resume] reusing $GRAPH_DIR/report.json"
fi

if [[ ! -f "$FAMILY_DIR/report.json" ]]; then
  python -u tasks/audit_mtbls13729_frozen_ion_families.py \
    --global-peak-dir "$GRAPH_DIR" \
    --output-dir "$FAMILY_DIR"
else
  echo "[resume] reusing $FAMILY_DIR/report.json"
fi

if [[ ! -f "$ANCHOR_DIR/report.json" ]]; then
  python -u tasks/audit_mtbls13729_c20_4_ms2.py \
    --output-dir "$ANCHOR_DIR"
else
  echo "[resume] reusing $ANCHOR_DIR/report.json"
fi

# The old class-score artifact was not consistently archived.  Reconstruct it
# from raw MS2 motifs and the current uniform EIC matrix when absent.
if [[ ! -f "$CLASS_DIR/class_score_report.json" ]]; then
  if [[ -d "$CLASS_DIR" ]] && [[ -n "$(ls -A "$CLASS_DIR")" ]]; then
    echo "Incomplete non-empty class directory: $CLASS_DIR" >&2
    exit 1
  fi
  python -u tasks/screen_mtbls13729_acylcarnitine_panel.py \
    --paired-dir data/mtbls13729/threeway_biology_v1 \
    --output-dir "$CLASS_DIR"
  python -u tasks/analyze_mtbls13729_acylcarnitine_class_score.py \
    --species "$CLASS_DIR/acylcarnitine_rt_species.csv" \
    --auc data/mtbls13729/ms1_eic_requant/pos_rp__eic_auc_matrix.csv.gz \
    --detected data/mtbls13729/ms1_eic_requant/pos_rp__eic_detection_matrix.csv.gz \
    --output-dir "$CLASS_DIR"
else
  echo "[resume] reusing $CLASS_DIR/class_score_report.json"
fi

if [[ ! -f "$CLOSURE_REPORT" ]]; then
  python -u tasks/summarize_mtbls13729_biology_closure.py \
    --family-report "$FAMILY_DIR/report.json" \
    --anchor-report "$ANCHOR_DIR/report.json" \
    --class-report "$CLASS_DIR/class_score_report.json" \
    --output "$CLOSURE_REPORT"
else
  echo "[resume] reusing $CLOSURE_REPORT"
fi

echo "=== MTBLS13729 biology closure completed: $(date) ==="
