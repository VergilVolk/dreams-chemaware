#!/bin/bash
#SBATCH --job-name=bio_b0_signal
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b0_signal_%j.out
#SBATCH --error=bioaware_b0_signal_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

CACHE=data/validation/g8r_p2_official_embeddings.npz
PARTICIPANTS=data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz
OUTPUT=data/validation/bioaware_b0_reaction_embedding_signal_formal_20260901

for path in "$CACHE" "$PARTICIPANTS" data/models/MassSpecGym_MurckoHist_split.hdf5; do
  test -s "$path" || { echo "Missing required input: $path" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 3; }

# The cluster's `dreams` environment intentionally has no pytest dependency.
# Execute the dependency-free unit functions directly, so an infrastructure
# package cannot block the scientific preflight.
python -c "import runpy; ns = runpy.run_path('tests/test_audit_bioaware_b0_reaction_embedding_signal.py'); [ns[n]() for n in sorted(ns) if n.startswith('test_')]; print('[BioAware B0 unit checks] PASS')"
python -u tasks/audit_bioaware_b0_reaction_embedding_signal.py \
  --hdf5 data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --embedding-cache "$CACHE" \
  --participants "$PARTICIPANTS" \
  --output-dir "$OUTPUT" \
  --folds 5 \
  --community-resolution 2.0 \
  --controls-per-edge 3 \
  --maximum-standardised-imbalance 0.10 \
  --permutations 200 \
  --bootstrap-resamples 5000 \
  --seed 20260901 \
  --formal

python -m json.tool "$OUTPUT/report.json" >/dev/null
echo "[BioAware B0] PASS execution; scientific pass_to_b1 is reported in $OUTPUT/report.json"
