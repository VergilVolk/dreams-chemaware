#!/usr/bin/env bash
#SBATCH --job-name=glm_mona_15m
#SBATCH --gpus=1
#SBATCH --output=data/validation/GLM_mona_15method_bench_%j.out
#SBATCH --error=data/validation/GLM_mona_15method_bench_%j.err

# GLM: 15-method model-blind bench on the SEALED MoNA polarity transfer
# panels (external validation of the GNPS ladder + confidence gate).
#
# PREREQUISITES (asset sync from local -> server, one-time):
#   data/models/glm_public/                          <- public model files
#     ms2deepscore_model.pt, ms2deepscore_settings.json
#     spec2vec_AllPositive_ratio05_filtered_iter_15.model{,.wv.vectors.npy,.trainables.syn1neg.npy}
#     150225_Spec2Vec_pos_CleanedLibraries.model{,.wv.vectors.npy,.syn1neg.npy}
#   (already on server) data/models/mona_pos_full.mgf, mona_neg_full.mgf
#   (already on server) frozen encoder checkpoints referenced by
#     GLM_score_all15_on_mona.py --encoder-scores (see below)
#
# The panels are the sealed preflight artifacts:
#   data/validation/chemaware_mona_polarity_transfer_local_preflight/{positive,negative}/panel.npz
# (upload them if absent on server; they are the frozen ground truth).
#
# Stage 1 (this sbatch): classical + public learned scores (CPU-heavy).
# Stage 2: our encoder scores via the existing embedding pipeline
#   (noise_v1 / official checkpoints; produce per-row embeddings then
#    GLM_score_all15_on_mona.py --encoder-embeddings <npz>).
# Stage 3: assemble bundle -> adapted frozen evaluation -> ladder.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PANELS_DIR=data/validation/chemaware_mona_polarity_transfer_local_preflight
OUT=data/validation/GLM_mona_15method_bench/run_${SLURM_JOB_ID}
MODELS=data/models/glm_public
mkdir -p "$OUT/staging"

python -X utf8 tasks/GLM_score_all15_on_mona.py \
  --panels "$PANELS_DIR" \
  --positive-mgf data/models/mona_pos_full.mgf \
  --negative-mgf data/models/mona_neg_full.mgf \
  --staging "$OUT/staging" \
  --methods cosine_greedy modified_cosine weighted_spectral_entropy

echo "STAGE1 COMPLETE: classical scores staged in $OUT/staging"
echo "Next (separate submission): encoder scores + bundle + evaluation,"
echo "see docs/GLM_ENSEMBLE_COMPLEMENTARITY_RETRACTION_AND_TRUTHBLIND_RESULT_20261007.md section 9."
