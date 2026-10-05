#!/bin/bash
#SBATCH --job-name=bioaware_b44_full
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b44_full_%j.out
#SBATCH --error=bioaware_b44_full_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
B42_DIR="data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2"
PANEL_DIR="data/validation/bioaware_b44_massbank_panel_${SLURM_JOB_ID}"
ARTIFACT_DIR="data/validation/bioaware_b44_catalogue_v1_${SLURM_JOB_ID}"
RESULT_DIR="data/validation/bioaware_b44_massbank_once_${SLURM_JOB_ID}"

echo "=== BioAware B44 full frozen pipeline started: $(date) ==="
for path in \
  tasks/build_bioaware_b44_massbank_blind_panel.py \
  tasks/validate_bioaware_b44_massbank_blind_panel.py \
  tasks/test_bioaware_b44_massbank_blind_panel.py \
  tasks/freeze_bioaware_b44_catalogue_v1.py \
  tasks/validate_bioaware_b44_catalogue_v1.py \
  tasks/test_bioaware_b44_catalogue_v1.py \
  tasks/evaluate_bioaware_b44_massbank_once.py \
  tasks/validate_bioaware_b44_massbank_once.py \
  tasks/test_bioaware_b44_massbank_once.py \
  tasks/audit_bioaware_b11_catalog_interaction_action.py \
  tasks/audit_bioaware_b12_multicohort_catalog_action.py \
  tasks/audit_bioaware_b16_pairwise_nonlinear_action.py \
  tasks/audit_bioaware_b36_reaction_specificity_ablation.py \
  tasks/audit_bioaware_b41_cross_catalog_topology.py \
  tasks/audit_bioaware_b42_independent_catalog_topology.py \
  tasks/evaluate_bioaware_b39_m2_fixed_action.py \
  tasks/audit_bioaware_b5_graph_prior_external_transfer.py \
  tasks/develop_bioaware_b1_multisource_action.py \
  tasks/develop_bioaware_metdna3_negative_loso_ranker.py \
  tasks/pilot_paired_layer_cka.py \
  tasks/shared_dreams_inference.py \
  tasks/e1_checkpoint_io.py \
  tasks/train_e1_identity.py \
  dreams/models/chem_aware/shared_embedding_v2.py \
  dreams/models/chem_aware/peft_v3.py \
  dreams/models/chem_aware/global_embedding_adapter.py \
  data/massbank/massbank_full.hdf5 \
  data/massbank/massbank_full.mgf \
  data/massbank/massbank_202406_msms.csv \
  data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz \
  data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz \
  data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz \
  data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz \
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz \
  data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz \
  data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz \
  data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz \
  data/e1/official_embedding_slim.pt \
  dreams/models/pretrained/ssl_model_server.pt \
  "${B42_DIR}/report.json" \
  "${B42_DIR}/candidate_catalog_features.csv.gz" \
  "${B42_DIR}/cross_catalog_transitions.csv.gz"; do
  test -s "${path}" || { echo "Missing dependency: ${path}" >&2; exit 2; }
done

"${PYTHON_BIN}" - <<'PY'
import h5py
import joblib
import numpy
import pandas
import scipy
import sklearn
import torch
print("[dependencies] PASS")
PY

echo "=== stage 1/3: metadata-only external panel lock ==="
"${PYTHON_BIN}" tasks/test_bioaware_b44_massbank_blind_panel.py
"${PYTHON_BIN}" tasks/build_bioaware_b44_massbank_blind_panel.py \
  --b42-dir "${B42_DIR}" \
  --output-dir "${PANEL_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b44_massbank_blind_panel.py --input-dir "${PANEL_DIR}"

echo "=== stage 2/3: freeze development-only catalogue expert ==="
"${PYTHON_BIN}" tasks/test_bioaware_b44_catalogue_v1.py
"${PYTHON_BIN}" tasks/freeze_bioaware_b44_catalogue_v1.py \
  --b42-dir "${B42_DIR}" \
  --output-dir "${ARTIFACT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b44_catalogue_v1.py --input-dir "${ARTIFACT_DIR}"

echo "=== stage 3/3: one-time official-DreaMS external evaluation ==="
"${PYTHON_BIN}" tasks/test_bioaware_b44_massbank_once.py
"${PYTHON_BIN}" tasks/evaluate_bioaware_b44_massbank_once.py \
  --panel-dir "${PANEL_DIR}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --output-dir "${RESULT_DIR}" \
  --device cuda
"${PYTHON_BIN}" tasks/validate_bioaware_b44_massbank_once.py --input-dir "${RESULT_DIR}"

echo "=== BioAware B44 full frozen pipeline completed: $(date) ==="
echo "Panel: ${PANEL_DIR}"
echo "Artifact: ${ARTIFACT_DIR}"
echo "Result: ${RESULT_DIR}"
