#!/bin/bash
#SBATCH --job-name=entity_effect_transport
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/entity_effect_transport_%j.out
#SBATCH --error=data/validation/entity_effect_transport_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${DISCOVERY_MANIFEST:?required}"
: "${VALIDATION_MANIFEST:?required}"
: "${LINKS:?required}"
: "${LINK_FREEZE:?required}"
: "${DISCOVERY_EFFECTS:?required}"
: "${VALIDATION_EFFECTS:?required}"
: "${EFFECT_SCALE:?required}"
: "${MIN_ABS_EFFECT:?required}"
: "${ANALYSIS_ROLE:?required}"

required=(
  "$DISCOVERY_MANIFEST" "$VALIDATION_MANIFEST" "$LINKS" "$LINK_FREEZE"
  "$DISCOVERY_EFFECTS" "$VALIDATION_EFFECTS"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MISSING_OR_EMPTY: $path" >&2; exit 5; }
done

python -u tasks/test_entity_effect_transport.py

OUT="data/validation/entity_effect_transport/run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "REFUSING_EXISTING_OUTPUT: $OUT" >&2; exit 6; }

python -u tasks/analyze_entity_effect_transport.py \
  --discovery-manifest "$DISCOVERY_MANIFEST" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --links "$LINKS" \
  --link-freeze "$LINK_FREEZE" \
  --discovery-effects "$DISCOVERY_EFFECTS" \
  --validation-effects "$VALIDATION_EFFECTS" \
  --effect-scale "$EFFECT_SCALE" \
  --min-abs-effect "$MIN_ABS_EFFECT" \
  --analysis-role "$ANALYSIS_ROLE" \
  --output-dir "$OUT"

echo "ENTITY_EFFECT_TRANSPORT_COMPLETE: $OUT"
