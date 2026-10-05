#!/bin/bash
# Audit the hardened fragment-graph/MassBank relation funnel under the frozen
# protected Phase-A geometry.  This job never updates model weights.
#SBATCH --job-name=chem_pa_union_u0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/chem_pa_union_u0_%j.out
#SBATCH --error=data/validation/chem_pa_union_u0_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
PHASEA_RUN="data/validation/chemaware_online_role_budget_native/run_2345481"
PHASEA="$PHASEA_RUN/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
PHASEA_POOL="$PHASEA_RUN/phasea_triplets/train_pool.npz"
PHASEA_VAL="$PHASEA_RUN/phasea_triplets/val_pool.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
GRAPH_LEDGER="data/validation/chemaware_fragment_graph_native/run_2346306/qualified_source"
MASSBANK_LEDGER="data/validation/chemaware_massbank_candidate_source_all_qualified_v1_20260928"
REUSABLE_CACHE="data/validation/chemaware_fragment_graph_native/run_2346306/phasea_embeddings"
OUT="data/validation/chemaware_phasea_geometry_union_preflight/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"
required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$MANIFEST" "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  "$GRAPH_LEDGER/report.json" "$GRAPH_LEDGER/candidate_scores.tsv"
  "$MASSBANK_LEDGER/report.json" "$MASSBANK_LEDGER/candidate_scores.tsv"
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/build_chemaware_dynamic_reference_native_triplets.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || { echo "Missing Phase-A union preflight input: $required_file" >&2; exit 2; }
done

if [[ -f "$REUSABLE_CACHE/report.json" && -f "$REUSABLE_CACHE/rows.npy" && -f "$REUSABLE_CACHE/embeddings_f32.npy" ]]; then
  EMBEDDING_DIR="$REUSABLE_CACHE"
  printf '%s\n' "$REUSABLE_CACHE" > "$OUT/reused_embedding_cache.txt"
else
  EMBEDDING_DIR="$OUT/phasea_embeddings"
  srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
    --checkpoint "$PHASEA" --evidence-dir "$BASE/evidence" --manifest "$MANIFEST" \
    --data "$DATA" --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EMBEDDING_DIR" --batch-size 64 --n-highest-peaks 100 --device cuda
fi

python -u tasks/build_chemaware_dynamic_reference_native_triplets.py \
  --phasea-pool "$PHASEA_POOL" --validation-pool "$PHASEA_VAL" \
  --source-ledger "$GRAPH_LEDGER" --source-ledger "$MASSBANK_LEDGER" \
  --manifest "$MANIFEST" --embedding-rows "$EMBEDDING_DIR/rows.npy" \
  --phasea-embeddings "$EMBEDDING_DIR/embeddings_f32.npy" \
  --embedding-report "$EMBEDDING_DIR/report.json" --geometry-checkpoint "$PHASEA" \
  --data "$DATA" --output "$OUT/phasea_geometry_triplets" --margin 0.1 \
  --maximum-candidates-per-query 6 --maximum-positive-references-per-relation 1 \
  --maximum-negative-references-per-event 1 --maximum-events-per-formula 8 \
  --maximum-events-per-identity 3 --minimum-supporting-sources 1 \
  --matched-control-policy pairwise_dominance \
  --geometry-label "frozen protected Phase-A +2.1266 pp geometry"

python - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
report = json.loads((out / "phasea_geometry_triplets/report.json").read_text())
events = int(report["dynamic_chemical_events_added"])
queries = int(report["dynamic_chemical_queries"])
decision = {
    "status": "CHEMAWARE_PHASEA_GEOMETRY_UNION_PREFLIGHT_COMPLETE",
    "weights_updated": False,
    "events": events,
    "queries": queries,
    "formulas": int(report["dynamic_chemical_formulas"]),
    "source_correctable_current_winners": int(report["source_correctable_current_winners"]),
    "coverage_gates": {
        "minimum_1000_distinct_candidate_events": events >= 1000,
        "minimum_500_distinct_queries": queries >= 500,
    },
}
decision["continuous_training_authorized"] = all(decision["coverage_gates"].values())
(out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision, indent=2), flush=True)
PY

echo "COMPLETE: Phase-A geometry union preflight only; no model weights were updated -> $OUT"
