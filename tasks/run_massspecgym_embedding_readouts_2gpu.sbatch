#!/bin/bash
# Unified ranker mainline: encode four existing checkpoints, combine WSE,
# Noise, frozen P2b, ChemAware V2 and complete embedding geometry, then select
# one capacity-controlled candidate ranker on MassSpecGym alone.
#SBATCH --job-name=unified_ranker
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/unified_ranker_%j.out
#SBATCH --error=data/validation/unified_ranker_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=10

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#GPUS[@]} != 2 )); then
  echo "MASSSPECGYM_READOUT_NEEDS_EXACTLY_TWO_GPUS: ${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
NOISE="${NOISE_CHECKPOINT:-data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
FULLMSG_RUN="${FULLMSG_RUN:-data/validation/noise_chemaware_full_massspecgym_native/run_2401890}"
FULLMSG="${FULLMSG_CHECKPOINT:-$FULLMSG_RUN/checkpoint/noise_chemaware_fullmsg_slim.pt}"
EVIDENCE="${MSG_PAIR_EVIDENCE:-data/validation/noise_msg_fusion_run_2349961/evidence}"
CHEM_EVIDENCE="${CHEMAWARE_V2_MSG_EVIDENCE:-}"
OUT="data/validation/massspecgym_unified_ranker_run_${SLURM_JOB_ID}"
SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
SCRATCH="$SCRATCH_BASE/msg_embedding_readouts_${SLURM_JOB_ID}"
POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-data/validation/chemaware_multinull_deployment_safe_full_20260919}"
CHEM_MANIFEST="${CHEMAWARE_MANIFEST:-data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz}"
CHEM_TOKENS="${CHEMAWARE_TOKEN_DIR:-data/validation/chemaware_corrected_manifest_tokens_v1}"
RULE_LIBRARY="${CHEMAWARE_RULE_LIBRARY:-dreams/models/chem_aware/chem_rules_data.json}"
GNPS="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
GNPS_BASE_SCORES="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
GNPS_CHEM_DIR="${GNPS_CONTINUOUS_CHEM_DIR:-data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence}"

# Historical evidence is large and expensive, but already computed.  If the
# canonical path moved, select the newest file-level artifact that declares the
# correct status and is bound to this exact graph; never match its child files
# as separate candidates.
if [[ ! -f "$EVIDENCE/evidence.npz" ]]; then
  EVIDENCE="$(python - "$GRAPH" <<'PY'
import hashlib, json, sys
from pathlib import Path
graph = Path(sys.argv[1])
h = hashlib.sha256(graph.read_bytes()).hexdigest()
matches = []
for report in Path('data/validation').glob('noise_msg_fusion_run_*/evidence/report.json'):
    try:
        body = json.loads(report.read_text())
    except Exception:
        continue
    root = report.parent
    if (body.get('status') == 'NOISE_MSG_PAIR_EVIDENCE_COMPLETE'
            and body.get('provenance', {}).get('graph_sha256') == h
            and (root / 'evidence.npz').is_file()):
        matches.append(root)
if not matches:
    raise SystemExit('MASSSPECGYM_READOUT_MISSING_REUSABLE_PAIR_EVIDENCE')
matches.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)))
print(matches[-1])
PY
)"
  echo "Reusing relocated scalar evidence: $EVIDENCE"
fi

BUILD_CHEM_EVIDENCE=0
if [[ -z "$CHEM_EVIDENCE" || ! -f "$CHEM_EVIDENCE" ]]; then
  set +e
  DISCOVERED_CHEM="$(python - "$EVIDENCE/evidence.npz" "$POLICY_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
source = Path(sys.argv[1])
h = hashlib.sha256(source.read_bytes()).hexdigest()
policy_dir = Path(sys.argv[2])
policy = policy_dir / 'truthblind_policy.joblib'
report = policy_dir / 'report.json'
if policy.is_file():
    policy_h = hashlib.sha256(policy.read_bytes()).hexdigest()
elif report.is_file():
    body = json.loads(report.read_text(encoding='utf-8'))
    policy_h = body.get('frozen_truthblind_policy', {}).get('sha256', '')
    if not policy_h:
        raise SystemExit(1)
else:
    raise SystemExit(1)
matches = []
for path in Path('data/validation').glob('**/conditional_null_chem_evidence.npz'):
    try:
        with np.load(path, allow_pickle=False) as body:
            aligned = (str(body['evidence_sha256'].item()) == h
                       and str(body['policy_sha256'].item()) == policy_h)
    except Exception:
        continue
    if aligned:
        matches.append(path)
if not matches:
    raise SystemExit(1)
matches.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)))
print(matches[-1])
PY
)"
  DISCOVERY_STATUS=$?
  set -e
  if (( DISCOVERY_STATUS == 0 )) && [[ -f "$DISCOVERED_CHEM" ]]; then
    CHEM_EVIDENCE="$DISCOVERED_CHEM"
    echo "Reusing aligned ChemAware V2 evidence: $CHEM_EVIDENCE"
  else
    BUILD_CHEM_EVIDENCE=1
    CHEM_EVIDENCE="$SCRATCH/conditional_null_chem_evidence.npz"
    echo "Aligned ChemAware V2 cache absent; it will be generated inside this job"
  fi
fi

required=(
  "$DATA" "$GRAPH" "$ARCH" "$NOISE" "$STAGE1" "$PHASEA" "$FULLMSG"
  "$EVIDENCE/evidence.npz" "$EVIDENCE/report.json"
  "$GNPS/spectra.mgf" "$GNPS/manifest.csv.gz"
  "$GNPS/panel_identity_disjoint.npz" "$GNPS/panel_formula_disjoint.npz"
  "$GNPS_BASE_SCORES"
  "$GNPS_CHEM_DIR/continuous_chem_identity_disjoint.npz"
  "$GNPS_CHEM_DIR/continuous_chem_formula_disjoint.npz"
  tasks/encode_noise_massspecgym_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_unified_candidate_ranker.py
  tasks/massspecgym_embedding_readout_core.py
  tasks/train_massspecgym_embedding_readouts.py
  tasks/test_massspecgym_embedding_readouts.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "MASSSPECGYM_READOUT_MISSING: $path" >&2; exit 2; }
done
if (( BUILD_CHEM_EVIDENCE == 0 )); then
  [[ -f "$CHEM_EVIDENCE" ]] || { echo "UNIFIED_RANKER_MISSING_CHEMAWARE_V2_EVIDENCE" >&2; exit 2; }
else
  chem_required=(
    "$CHEM_MANIFEST" "$CHEM_TOKENS/rows.npy"
    "$CHEM_TOKENS/official_embeddings_f32.npy" "$CHEM_TOKENS/mz_f32.npy"
    "$CHEM_TOKENS/intensity_f32.npy" "$CHEM_TOKENS/valid.npy"
    "$CHEM_TOKENS/precursor_mz_f32.npy" "$RULE_LIBRARY"
    tasks/export_chemaware_conditional_null_evidence.py
    tasks/audit_chemaware_orthogonal_rule_residual_policy.py
    tasks/chemaware_truthblind_candidate_core.py
  )
  for path in "${chem_required[@]}"; do
    [[ -f "$path" ]] || { echo "UNIFIED_RANKER_CHEMAWARE_V2_BUILD_MISSING: $path" >&2; exit 2; }
  done
fi
[[ ! -e "$OUT" ]] || { echo "MASSSPECGYM_READOUT_OUTPUT_EXISTS: $OUT" >&2; exit 4; }
mkdir -p "$SCRATCH"
cleanup_scratch() { rm -rf -- "$SCRATCH"; }
trap cleanup_scratch EXIT

python -m py_compile \
  tasks/massspecgym_embedding_readout_core.py \
  tasks/train_massspecgym_embedding_readouts.py \
  tasks/evaluate_gnps_unified_candidate_ranker.py \
  tasks/encode_noise_massspecgym_checkpoint.py
python -u tasks/test_massspecgym_embedding_readouts.py

if (( BUILD_CHEM_EVIDENCE == 1 )); then
  # Missing files are invalid.  Only the explicit report/hash contract below
  # may set this to zero.  The previous zero initialization accidentally
  # skipped the registered in-job rebuild when truthblind_policy.joblib was
  # absent, then sent the missing path to the exporter.
  POLICY_VALID=1
  if [[ -s "$POLICY_DIR/truthblind_policy.joblib" && -s "$POLICY_DIR/report.json" ]]; then
    set +e
    python - "$POLICY_DIR/report.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1], encoding='utf-8'))
method = body.get('method', {})
assert method.get('rule_control_variants') == [
    'rule_response_content_permuted',
    'rule_response_content_permuted_b',
    'rule_response_content_permuted_c',
]
assert method.get('contrast_representation') == 'symmetric_summary'
assert body.get('selection_control_mode') == 'deployment_safe'
assert body.get('candidate_conditioned') is True
PY
    POLICY_VALID=$?
    set -e
  fi
  if (( POLICY_VALID != 0 )); then
    POLICY_DIR="$SCRATCH/rebuilt_multinull_v2_policy"
    python -u tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
      --rule-control-variants \
        rule_response_content_permuted \
        rule_response_content_permuted_b \
        rule_response_content_permuted_c \
      --contrast-representation symmetric_summary \
      --selection-control-mode deployment_safe \
      --output "$POLICY_DIR"
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/export_chemaware_conditional_null_evidence.py \
    --policy-dir "$POLICY_DIR" --token-dir "$CHEM_TOKENS" \
    --manifest "$CHEM_MANIFEST" --graph "$GRAPH" \
    --evidence "$EVIDENCE" --rule-library "$RULE_LIBRARY" \
    --output "$CHEM_EVIDENCE" --batch-queries 256
fi

encode_checkpoint() {
  local name="$1" checkpoint="$2"
  echo "=== encoding $name on two GPUs ==="
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/encode_noise_massspecgym_checkpoint.py \
    --data "$DATA" --checkpoint "$checkpoint" --architecture-checkpoint "$ARCH" \
    --output "$SCRATCH/${name}_0.npz" --shard-index 0 --shard-count 2 --batch-size 64 & p0=$!
  CUDA_VISIBLE_DEVICES="${GPUS[1]}" python -u tasks/encode_noise_massspecgym_checkpoint.py \
    --data "$DATA" --checkpoint "$checkpoint" --architecture-checkpoint "$ARCH" \
    --output "$SCRATCH/${name}_1.npz" --shard-index 1 --shard-count 2 --batch-size 64 & p1=$!
  set +e
  wait "$p0"; s0=$?
  wait "$p1"; s1=$?
  set -e
  if (( s0 != 0 || s1 != 0 )); then
    echo "MASSSPECGYM_READOUT_ENCODING_FAILED: $name status=$s0/$s1" >&2
    exit 6
  fi
}

encode_checkpoint noise_v1 "$NOISE"
encode_checkpoint chemaware_stage1 "$STAGE1"
encode_checkpoint chemaware_phasea "$PHASEA"
encode_checkpoint noise_chemaware_fullmsg "$FULLMSG"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/train_massspecgym_embedding_readouts.py \
  --graph "$GRAPH" --data "$DATA" --evidence "$EVIDENCE" \
  --chem-evidence "$CHEM_EVIDENCE" \
  --embedding "noise_v1=$SCRATCH/noise_v1_0.npz" \
  --embedding "noise_v1=$SCRATCH/noise_v1_1.npz" \
  --embedding "chemaware_stage1=$SCRATCH/chemaware_stage1_0.npz" \
  --embedding "chemaware_stage1=$SCRATCH/chemaware_stage1_1.npz" \
  --embedding "chemaware_phasea=$SCRATCH/chemaware_phasea_0.npz" \
  --embedding "chemaware_phasea=$SCRATCH/chemaware_phasea_1.npz" \
  --embedding "noise_chemaware_fullmsg=$SCRATCH/noise_chemaware_fullmsg_0.npz" \
  --embedding "noise_chemaware_fullmsg=$SCRATCH/noise_chemaware_fullmsg_1.npz" \
  --noise-name noise_v1 --output "$OUT/modeling" \
  --epochs 3 --batch-queries 256 --learning-rate 0.0003 \
  --weight-decay 0.0001 --projection-dimension 16 \
  --maximum-negatives 6 --bootstrap-resamples 5000
if (( BUILD_CHEM_EVIDENCE == 1 )); then
  cp "$CHEM_EVIDENCE" "$OUT/conditional_null_chem_evidence.npz"
fi

# Fail before four GNPS encodings if the freshly trained artifact ever drifts
# back to the scalar control or loses its full-integration contract.
python - "$OUT/modeling/report.json" "$OUT/modeling/selected_readout.pt" <<'PY'
import json, sys
from pathlib import Path
import torch
report_path, model_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
try:
    payload = torch.load(model_path, map_location='cpu', weights_only=False)
except TypeError:
    payload = torch.load(model_path, map_location='cpu')
allowed = {'projected_cosine', 'embedding_pair'}
if (report.get('selection_class') != 'full_embedding_integration'
        or report.get('selection') not in allowed
        or payload.get('selection_class') != 'full_embedding_integration'
        or payload.get('arm') != report.get('selection')):
    raise SystemExit('UNIFIED_RANKER_FULL_INTEGRATION_PREFLIGHT_FAIL')
print('UNIFIED_RANKER_FULL_INTEGRATION_PREFLIGHT_PASS')
PY

# A full-embedding arm (never the scalar control) and its full-development
# refit now exist. GNPS becomes readable
# only below this point; no result can flow back into fitting or arm selection.
encode_gnps() {
  local name="$1" checkpoint="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS" --checkpoint "$checkpoint" --architecture-checkpoint "$ARCH" \
    --output "$SCRATCH/gnps_${name}.npz" --batch-size 64 --label "$name"
}

encode_gnps noise_v1 "$NOISE" "${GPUS[0]}" & g0=$!
encode_gnps chemaware_stage1 "$STAGE1" "${GPUS[1]}" & g1=$!
set +e
wait "$g0"; gs0=$?
wait "$g1"; gs1=$?
set -e
if (( gs0 != 0 || gs1 != 0 )); then
  echo "UNIFIED_RANKER_GNPS_ENCODING_FAILED: noise/stage1=$gs0/$gs1" >&2
  exit 7
fi
encode_gnps chemaware_phasea "$PHASEA" "${GPUS[0]}" & g0=$!
encode_gnps noise_chemaware_fullmsg "$FULLMSG" "${GPUS[1]}" & g1=$!
set +e
wait "$g0"; gs0=$?
wait "$g1"; gs1=$?
set -e
if (( gs0 != 0 || gs1 != 0 )); then
  echo "UNIFIED_RANKER_GNPS_ENCODING_FAILED: phasea/fullmsg=$gs0/$gs1" >&2
  exit 7
fi

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/evaluate_gnps_unified_candidate_ranker.py \
  --benchmark "$GNPS" --model "$OUT/modeling/selected_readout.pt" \
  --base-scores "$GNPS_BASE_SCORES" --chem-dir "$GNPS_CHEM_DIR" \
  --embedding "noise_v1=$SCRATCH/gnps_noise_v1.npz" \
  --embedding "chemaware_stage1=$SCRATCH/gnps_chemaware_stage1.npz" \
  --embedding "chemaware_phasea=$SCRATCH/gnps_chemaware_phasea.npz" \
  --embedding "noise_chemaware_fullmsg=$SCRATCH/gnps_noise_chemaware_fullmsg.npz" \
  --output "$OUT/gnps_fixed_evaluation" --device cuda --bootstrap-resamples 5000

cp "$SCRATCH"/*.json "$OUT/"
sha256sum "$NOISE" "$STAGE1" "$PHASEA" "$FULLMSG" \
  "$EVIDENCE/evidence.npz" "$CHEM_EVIDENCE" "$OUT/modeling/selected_readout.pt" \
  "$OUT/gnps_fixed_evaluation/report.json" \
  > "$OUT/SHA256SUMS.txt"
echo "FULL_EMBEDDING_UNIFIED_CANDIDATE_RANKER_AND_FIXED_GNPS_EVALUATION_COMPLETE: $OUT"
