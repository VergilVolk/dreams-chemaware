#!/bin/bash
#SBATCH --job-name=grand_submit
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=data/validation/grand_submit_%j.out
#SBATCH --error=data/validation/grand_submit_%j.err

# Single Slurm entry point for the bounded stage-1 grand-fusion programme.
# This short CPU dispatch job performs a fail-fast server-side asset audit and
# submits the real GPU/CPU child jobs with their preregistered dependencies.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS

export CHEM_CHECKPOINT="${CHEM_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
export CHEMAWARE_V2_POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-data/validation/chemaware_multinull_deployment_safe_full_20260919}"
export RUN_FIXED_CONSENSUS="${RUN_FIXED_CONSENSUS:-0}"

command -v sbatch >/dev/null 2>&1 || {
  echo "GRAND_FUSION_PREFLIGHT_FAIL: sbatch is unavailable inside the dispatch allocation" >&2
  exit 2
}

required=(
  "$CHEM_CHECKPOINT"
  data/models/MassSpecGym_MurckoHist_split.hdf5
  dreams/models/pretrained/ssl_model_server.pt
  data/e1/official_embedding_slim.pt
  data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt
  data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz
  data/validation/noise_corrected_candidate_graph_v1_20260906/official_embeddings.npz
  data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf
  data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz
  data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_identity_disjoint.npz
  data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz
  data/validation/gnps_absolute_baselines_run_2347472/gnps/official_embeddings.npz
  data/validation/gnps_absolute_baselines_run_2347472/gnps/v1_embeddings.npz
  data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/mz_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/intensity_f32.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/valid.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/precursor_mz_f32.npy
  dreams/models/chem_aware/chem_rules_data.json
  dreams/models/chem_aware/global_embedding_adapter.py
  tasks/audit_chemaware_candidate_evidence_policy.py
  tasks/audit_chemaware_mass_kernel_embedding.py
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
  tasks/submit_grand_fusion_program.sh
  tasks/run_noise_msg_fusion_5pp_2gpu.sbatch
  tasks/run_grand_fusion_task_vector_scan_2gpu.sbatch
  tasks/run_grand_fusion_chemaware_v2_scores.sbatch
  tasks/run_grand_fusion_router_1gpu.sbatch
)

missing=0
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "GRAND_FUSION_PREFLIGHT_MISSING: $path" >&2
    missing=1
  fi
done
(( missing == 0 )) || exit 5

echo "GRAND_FUSION_PREFLIGHT_OK"
echo "CHEM_CHECKPOINT=$CHEM_CHECKPOINT"
echo "CHEMAWARE_V2_POLICY_DIR=$CHEMAWARE_V2_POLICY_DIR"
if [[ -s "$CHEMAWARE_V2_POLICY_DIR/truthblind_policy.joblib" \
   && -s "$CHEMAWARE_V2_POLICY_DIR/report.json" ]]; then
  echo "CHEMAWARE_V2_POLICY_SOURCE=frozen"
else
  echo "CHEMAWARE_V2_POLICY_SOURCE=rebuild_from_frozen_recipe"
fi
echo "RUN_FIXED_CONSENSUS=$RUN_FIXED_CONSENSUS"
bash tasks/submit_grand_fusion_program.sh
