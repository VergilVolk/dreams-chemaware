#!/bin/bash
# Submit the bounded grand-fusion programme with explicit dependencies.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
: "${CHEM_CHECKPOINT:?Set CHEM_CHECKPOINT before submission}"
[[ -f "$CHEM_CHECKPOINT" ]] || { echo "CHEM_CHECKPOINT does not exist: $CHEM_CHECKPOINT" >&2; exit 5; }

msg_submit=$(sbatch --parsable tasks/run_noise_msg_fusion_5pp_2gpu.sbatch)
msg_job="${msg_submit%%;*}"
taskvec_submit=$(sbatch --parsable --export=ALL,CHEM_CHECKPOINT="$CHEM_CHECKPOINT" \
  tasks/run_grand_fusion_task_vector_scan_2gpu.sbatch)
taskvec_job="${taskvec_submit%%;*}"
consensus_job=""
if [[ "${RUN_FIXED_CONSENSUS:-0}" == "1" ]]; then
  consensus_submit=$(sbatch --parsable tasks/run_grand_fusion_fixed_consensus_1gpu.sbatch)
  consensus_job="${consensus_submit%%;*}"
fi

evidence_dir="data/validation/noise_msg_fusion_run_${msg_job}/evidence"
msg_oof="data/validation/noise_msg_fusion_run_${msg_job}/fusion_model/oof_pair_scores.npz"
msg_gnps="data/validation/noise_msg_fusion_run_${msg_job}/gnps_fusion_scores"
chem_submit=$(sbatch --parsable --dependency="afterok:${msg_job}" \
  --export=ALL,EVIDENCE_DIR="$evidence_dir",MSG_FUSION_OOF="$msg_oof",MSG_FUSION_GNPS="$msg_gnps" \
  tasks/run_grand_fusion_chemaware_v2_scores.sbatch)
chem_job="${chem_submit%%;*}"

integrated_evidence="data/validation/grand_fusion_chemaware_v2_run_${chem_job}/evidence_with_chemaware"
integrated_bundle="data/validation/grand_fusion_chemaware_v2_run_${chem_job}/method_scores_with_chemaware.npz"
router_submit=$(sbatch --parsable --dependency="afterok:${chem_job}" \
  --export=ALL,EVIDENCE_DIR="$integrated_evidence",SCORE_BUNDLE="$integrated_bundle" \
  tasks/run_grand_fusion_router_1gpu.sbatch)
router_job="${router_submit%%;*}"

printf 'MSG/evidence job: %s\n' "$msg_job"
printf 'Task-vector job: %s\n' "$taskvec_job"
if [[ -n "$consensus_job" ]]; then
  printf 'Optional fixed-consensus diagnostic job: %s\n' "$consensus_job"
else
  printf '%s\n' 'Fixed-consensus diagnostic skipped (default; no GPU spent).'
fi
printf 'ChemAware V2 aligned export job (after MSG): %s\n' "$chem_job"
printf 'Risk-router job (after MSG + ChemAware): %s\n' "$router_job"
printf '%s\n' 'After the task-vector job finishes, select exactly one protected arm and submit:'
printf '%s\n' '  sbatch --export=ALL,SELECTED_CHECKPOINT=<path> tasks/run_grand_fusion_selected_encoder_gnps_1gpu.sbatch'

if [[ -n "${B47_U3_DIR:-}" ]]; then
  [[ -d "$B47_U3_DIR" ]] || { echo "B47_U3_DIR does not exist: $B47_U3_DIR" >&2; exit 5; }
  bio_submit=$(sbatch --parsable --export=ALL,B47_U3_DIR="$B47_U3_DIR" \
    tasks/run_bioaware_b47_gate_remediation.sbatch)
  bio_job="${bio_submit%%;*}"
  printf 'B47 remediation job: %s\n' "$bio_job"
  repair_dir="data/validation/bioaware_b47_gate_remediation_${bio_job}_v1"
  b47_required_env=(
    B47_CANDIDATE_MANIFEST B47_UNARY_SCORES B47_CATALOGUE_SCORES
    B47_NETWORK_SCORES B47_DEGREE_REWIRED_SCORES B47_SEED_PERMUTED_SCORES
    B47_MATCHED_NONNEIGHBOR_SCORES
  )
  b47_ready=1
  for name in "${b47_required_env[@]}"; do
    [[ -n "${!name:-}" ]] || b47_ready=0
  done
  if (( b47_ready )); then
    confirm_submit=$(sbatch --parsable --dependency="afterok:${bio_job}" \
      --export=ALL,B47_REPAIR_DIR="$repair_dir" \
      tasks/run_bioaware_b47_confirmatory_once.sbatch)
    confirm_job="${confirm_submit%%;*}"
    printf 'B47 one-time confirmatory job (after repair): %s\n' "$confirm_job"
  else
    printf '%s\n' 'B47 confirmatory job not submitted: one or more canonical Track-C score paths are unset.'
  fi
else
  printf 'B47 remediation not submitted: B47_U3_DIR is unset.\n'
fi
