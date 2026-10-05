#!/bin/bash
# Resume the completed run_2346441 from its saved role-2 evaluation.  Training
# and triplet construction are deliberately not repeated.
#SBATCH --job-name=chem_union_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/chem_union_eval_%j.out
#SBATCH --error=data/validation/chem_union_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RUN="${1:-data/validation/chemaware_fragment_massbank_union_continue/run_2346441}"
OFFICIAL="data/e1/official_embedding_slim.pt"
PHASEA="data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
START="data/validation/chemaware_fragment_graph_native/run_2346408_resume_run_2346306/training/step-000750.ckpt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"

RUN="$(realpath "$RUN")"
ROOT_REAL="$(realpath data/validation/chemaware_fragment_massbank_union_continue)"
case "$RUN" in
  "$ROOT_REAL"/run_*) ;;
  *) echo "Refusing union resume outside $ROOT_REAL: $RUN" >&2; exit 2 ;;
esac
required_files=(
  "$RUN/input_sha256sums.txt" "$OFFICIAL" "$PHASEA" "$START"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  tasks/select_chemaware_residual_checkpoint.py tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/audit_chemaware_role3_confirmation.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || { echo "Missing ChemAware union resume input: $required_file" >&2; exit 2; }
done

# The resume is allowed to operate only on the exact frozen inputs recorded by
# the parent run.  This turns the provenance record into an enforced contract.
(cd /data02/run01/scv7tsl/DreaMS && sha256sum --check --strict "$RUN/input_sha256sums.txt")
for frozen_input in \
  "$OFFICIAL" "$PHASEA" "$START" \
  "$BASE/evidence/selection_triplet_evidence.npz" \
  "$BASE/evidence/confirmation_triplet_evidence.npz"; do
  grep -Fq "  $frozen_input" "$RUN/input_sha256sums.txt" || {
    echo "Resume provenance ledger lacks required input: $frozen_input" >&2; exit 3;
  }
done

VERIFIED_ROLE2="$RUN/role2_checkpoint_evaluation_panel_verified.json"
if [[ ! -f "$VERIFIED_ROLE2" ]]; then
  checkpoint_args=(--checkpoint "official=$OFFICIAL" --checkpoint "phaseA_2pp=$PHASEA" --checkpoint "fragment_graph_step750=$START")
  mapfile -t trained_checkpoints < <(find "$RUN/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
  [[ ${#trained_checkpoints[@]} -eq 4 ]] || {
    echo "Expected four fixed-step checkpoints, found ${#trained_checkpoints[@]}" >&2; exit 6;
  }
  for checkpoint in "${trained_checkpoints[@]}"; do
    checkpoint_args+=(--checkpoint "union_$(basename "$checkpoint" .ckpt)=$checkpoint")
  done
  srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
    --policy "$BASE/evidence/selection_triplet_evidence.npz" \
    --role-contract-dir "$BASE/evidence" "${checkpoint_args[@]}" \
    --paired-reference fragment_graph_step750 --formula-role 2 \
    --output "$VERIFIED_ROLE2" --batch-size 64 --bootstrap-draws 10000 --device cuda
fi

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$VERIFIED_ROLE2" \
  --base-name fragment_graph_step750 --protected-baseline-name phaseA_2pp \
  --exclude-name phaseA_2pp \
  --require-positive-formula-ci --output "$RUN/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$RUN/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$RUN/checkpoint_selection.step.txt")"
if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_UNION_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: union role-2 selection -> $RUN"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  --checkpoint "official=$OFFICIAL" --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "fragment_graph_step750=$START" --checkpoint "union_selected=$SELECTED_REAL" \
  --paired-reference fragment_graph_step750 --formula-role 3 \
  --output "$RUN/role3_evaluation.json" --batch-size 64 --bootstrap-draws 10000 --device cuda
python -u tasks/audit_chemaware_role3_confirmation.py \
  --evaluation "$RUN/role3_evaluation.json" --base-name fragment_graph_step750 \
  --protected-baseline-name phaseA_2pp --candidate-name union_selected \
  --output "$RUN/role3_confirmation_decision.json"

CONFIRMED="$(tr -d '\r\n' < "$RUN/role3_confirmation_decision.confirmed.txt")"
DESTINATION="$RUN/protected_negative_result"
if [[ "$CONFIRMED" == "1" ]]; then DESTINATION="$RUN/protected_candidate"; fi
mkdir -p "$DESTINATION"
CHECKPOINT_DEST="$DESTINATION/chemaware_union_selected.ckpt"
if [[ -e "$CHECKPOINT_DEST" ]]; then
  [[ "$SELECTED_REAL" -ef "$CHECKPOINT_DEST" ]] || {
    echo "Resume destination collision: $CHECKPOINT_DEST" >&2; exit 7;
  }
else
  ln "$SELECTED_REAL" "$CHECKPOINT_DEST"
fi
cp "$RUN/checkpoint_selection.json" "$VERIFIED_ROLE2" \
  "$RUN/role3_evaluation.json" "$RUN/role3_confirmation_decision.json" "$DESTINATION/"
if [[ -f "$RUN/protected_corpus/report.json" ]]; then
  cp "$RUN/protected_corpus/report.json" "$DESTINATION/corpus_report.json"
fi
(
  cd "$DESTINATION"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "COMPLETE: ChemAware union selection/confirmation -> $DESTINATION"
