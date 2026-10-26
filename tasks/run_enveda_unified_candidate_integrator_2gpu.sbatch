#!/bin/bash
# Fixed unified architecture:
#   four frozen encoders + WSE + one P2b composite + one MSG composite
#   + continuous ChemAware V2, trained as a candidate-level reliability model.
# Enveda and GNPS are both consumed development resources.  Enveda fits the
# head with global formula OOF; this job reads GNPS only afterwards and does
# not refit or select the already fixed model from that run.
#SBATCH --job-name=e180_unified_fit
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/e180_unified_fit_%j.out
#SBATCH --error=data/validation/e180_unified_fit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=10

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#GPUS[@]} != 2 )); then
  echo "ENVEDA_UNIFIED_NEEDS_EXACTLY_TWO_GPUS: ${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

ENVEDA="${ENVEDA_BENCHMARK:-data/validation/enveda180_cross_condition_benchmark_v3_20261012}"
ENVEDA_LAYER="${ENVEDA_LAYER_RUN:-data/validation/enveda180_layered_scoreblind_run_2372550}"
GNPS="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
GNPS_BASE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
GNPS_BASE_SHA256="${GNPS_SCORE_BUNDLE_SHA256:-cfc3e64a4ff05ae420546183d2e230bc8cd093d614d31ba80aaea313a14bd16b}"
GNPS_CHEM="${GNPS_CONTINUOUS_CHEM_DIR:-data/validation/wse_conditional_residual_run_2376648/continuous_chem_evidence}"
B_PREDICTIONS="${B_GNPS_OOF_PREDICTIONS:-data/validation/B_dual_encoder_multireranker_run_2400157/modeling_nested_oof/candidate_predictions.npz}"

ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
PHASEA="${CHEMAWARE_PHASEA_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
FULLMSG="${NOISE_CHEMAWARE_FULLMSG_CHECKPOINT:-data/validation/noise_chemaware_full_massspecgym_native/run_2401890/checkpoint/noise_chemaware_fullmsg_slim.pt}"
RULE_LIBRARY="${CHEMAWARE_RULE_LIBRARY:-dreams/models/chem_aware/chem_rules_data.json}"
POLICY_DIR="${CHEMAWARE_V2_POLICY_DIR:-data/validation/massspecgym_unified_ranker_run_2409612/rebuilt_multinull_v2_policy}"
V2_POLICY_SHA256="${CHEMAWARE_V2_POLICY_SHA256:-dc9d76dc439fca40db88e3a95ca33e6b59438a92e6a80080d8df77e4615e5a4a}"
MSG_MODEL_DIR="${NOISE_MSG_FUSION_MODEL:-data/validation/noise_msg_fusion_run_2349961/fusion_model}"
GNPS_MSG="${NOISE_MSG_GNPS_CACHE:-data/validation/noise_msg_fusion_run_2349961/gnps_fusion_scores}"

OUT="data/validation/enveda_unified_candidate_integrator_run_${SLURM_JOB_ID}"
SCRATCH="${SLURM_TMPDIR:-/tmp}/enveda_unified_${SLURM_JOB_ID}"
REUSE_RUN="${ENVEDA_UNIFIED_REUSE_RUN:-}"

ENVEDA_STAGE1_EMBEDDINGS="$OUT/inputs/enveda_chemaware_stage1_embeddings.npz"
ENVEDA_PHASEA_EMBEDDINGS="$OUT/inputs/enveda_chemaware_phasea_embeddings.npz"
ENVEDA_FULLMSG_EMBEDDINGS="$OUT/inputs/enveda_noise_chemaware_fullmsg_embeddings.npz"
ENVEDA_MSG_CACHE="$OUT/inputs/enveda_msg_cache"
if [[ -n "$REUSE_RUN" ]]; then
  ENVEDA_STAGE1_EMBEDDINGS="$REUSE_RUN/inputs/enveda_chemaware_stage1_embeddings.npz"
  ENVEDA_PHASEA_EMBEDDINGS="$REUSE_RUN/inputs/enveda_chemaware_phasea_embeddings.npz"
  ENVEDA_FULLMSG_EMBEDDINGS="$REUSE_RUN/inputs/enveda_noise_chemaware_fullmsg_embeddings.npz"
  ENVEDA_MSG_CACHE="$REUSE_RUN/inputs/enveda_msg_cache"
fi

# Locate complete artifact roots only.  Child score files are never treated as
# independent caches.
if [[ ! -s "$ENVEDA_LAYER/official_embeddings.npz" \
      || ! -s "$ENVEDA_LAYER/noise_v1_embeddings.npz" \
      || ! -s "$ENVEDA_LAYER/method_scores_unopened.npz" ]]; then
  ENVEDA_LAYER="$(python - <<'PY'
from pathlib import Path
matches=[]
for root in Path('data/validation').glob('enveda180_layered_scoreblind_run_*'):
    if all((root/name).is_file() for name in (
        'official_embeddings.npz','noise_v1_embeddings.npz','method_scores_unopened.npz')):
        matches.append(root)
if len(matches) != 1:
    detail='\n'.join(map(str,matches)) or '<none>'
    raise SystemExit(f'ENVEDA_UNIFIED_LAYER_RUN_AMBIGUOUS: expected one complete fallback, found {len(matches)}\n{detail}')
print(matches[0])
PY
)"
fi

if [[ ! -s "$POLICY_DIR/truthblind_policy.joblib" || ! -s "$POLICY_DIR/report.json" ]]; then
  POLICY_DIR="$(python - "$V2_POLICY_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import joblib

expected_hash = sys.argv[1]
expected_controls = (
    'rule_response_content_permuted',
    'rule_response_content_permuted_b',
    'rule_response_content_permuted_c',
)
matches=[]
for policy in Path('data/validation').glob('**/truthblind_policy.joblib'):
    report_path = policy.parent / 'report.json'
    if not report_path.is_file():
        continue
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if digest != expected_hash or report.get('frozen_truthblind_policy', {}).get('sha256') != digest:
        continue
    payload = joblib.load(policy)
    if payload.get('schema') != 'chemaware_truthblind_candidate_policy_v1':
        continue
    if tuple(payload.get('control_rule_keys', ())) != expected_controls:
        continue
    matches.append(policy.parent)
if not matches:
    raise SystemExit(f'ENVEDA_UNIFIED_EXACT_V2_POLICY_NOT_FOUND: sha256={expected_hash}')
# Multiple recovered copies with the same byte hash are the same frozen model,
# not competing policies.  Choose a deterministic path without inspecting data.
print(sorted(matches, key=str)[0])
PY
)"
fi

if [[ ! -s "$MSG_MODEL_DIR/fusion_model.joblib" || ! -s "$MSG_MODEL_DIR/report.json" ]]; then
  MSG_MODEL_DIR="$(python - <<'PY'
from pathlib import Path
matches=[]
for report in Path('data/validation').glob('noise_msg_fusion_run_*/fusion_model/report.json'):
    if (report.parent/'fusion_model.joblib').is_file(): matches.append(report.parent)
if len(matches) != 1:
    detail='\n'.join(map(str,matches)) or '<none>'
    raise SystemExit(f'ENVEDA_UNIFIED_MSG_MODEL_AMBIGUOUS: expected one fallback, found {len(matches)}\n{detail}')
print(matches[0])
PY
)"
fi

if [[ ! -s "$GNPS_MSG/report.json" || ! -s "$GNPS_MSG/pair_scores_identity_disjoint.npy" || ! -s "$GNPS_MSG/pair_scores_formula_disjoint.npy" ]]; then
  GNPS_MSG="$(python - <<'PY'
from pathlib import Path
matches=[]
for report in Path('data/validation').glob('noise_msg_fusion_run_*/gnps_fusion_scores/report.json'):
    if (report.parent/'pair_scores_identity_disjoint.npy').is_file() and \
       (report.parent/'pair_scores_formula_disjoint.npy').is_file():
        matches.append(report.parent)
if len(matches) != 1:
    detail='\n'.join(map(str,matches)) or '<none>'
    raise SystemExit(f'ENVEDA_UNIFIED_GNPS_MSG_CACHE_AMBIGUOUS: expected one fallback, found {len(matches)}\n{detail}')
print(matches[0])
PY
)"
fi

required=(
  "$ENVEDA/report.json" "$ENVEDA/checksums.sha256"
  "$ENVEDA/spectra.mgf" "$ENVEDA/manifest.csv.gz"
  "$ENVEDA/panel_identity_disjoint.npz" "$ENVEDA/panel_formula_disjoint.npz"
  "$ENVEDA_LAYER/official_embeddings.npz" "$ENVEDA_LAYER/noise_v1_embeddings.npz"
  "$ENVEDA_LAYER/method_scores_unopened.npz" "$ENVEDA_LAYER/method_scores_unopened.npz.json"
  "$ENVEDA_LAYER/scoreblind_report.json" "$ENVEDA_LAYER/freeze_contract.json"
  "$GNPS/report.json" "$GNPS/spectra.mgf" "$GNPS/manifest.csv.gz"
  "$GNPS/panel_identity_disjoint.npz" "$GNPS/panel_formula_disjoint.npz"
  "$GNPS_BASE" "${GNPS_BASE}.json" "$GNPS_MSG/report.json"
  "$GNPS_MSG/pair_scores_identity_disjoint.npy"
  "$GNPS_MSG/pair_scores_formula_disjoint.npy"
  "$GNPS_CHEM/continuous_chem_identity_disjoint.npz"
  "$GNPS_CHEM/continuous_chem_formula_disjoint.npz"
  "$B_PREDICTIONS" "$ARCH" "$STAGE1" "$PHASEA" "$FULLMSG"
  "$MSG_MODEL_DIR/fusion_model.joblib" "$MSG_MODEL_DIR/report.json"
  "$POLICY_DIR/truthblind_policy.joblib" "$POLICY_DIR/report.json" "$RULE_LIBRARY"
  tasks/enveda_unified_candidate_integrator_core.py
  tasks/train_enveda_unified_candidate_integrator.py
  tasks/evaluate_enveda_integrator_on_gnps.py
  tasks/test_enveda_unified_candidate_integrator.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/build_gnps_chemaware_kernel_cache.py
  tasks/export_gnps_chemaware_continuous_null_evidence.py
  tasks/apply_noise_msg_fusion_to_gnps.py
)
if [[ -n "$REUSE_RUN" ]]; then
  required+=(
    "$ENVEDA_STAGE1_EMBEDDINGS" "${ENVEDA_STAGE1_EMBEDDINGS%.npz}.json"
    "$ENVEDA_PHASEA_EMBEDDINGS" "${ENVEDA_PHASEA_EMBEDDINGS%.npz}.json"
    "$ENVEDA_FULLMSG_EMBEDDINGS" "${ENVEDA_FULLMSG_EMBEDDINGS%.npz}.json"
    "$ENVEDA_MSG_CACHE/report.json"
    "$ENVEDA_MSG_CACHE/pair_scores_identity_disjoint.npy"
    "$ENVEDA_MSG_CACHE/pair_scores_formula_disjoint.npy"
  )
fi
missing=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'ENVEDA_UNIFIED_MISSING_INPUT: %s\n' "${missing[@]}" >&2
  exit 4
fi

# Fail before any GPU encoding if a reusable policy, model, score bundle or
# candidate graph is not the exact frozen family declared above.
python - "$ENVEDA" "$ENVEDA_LAYER" "$GNPS" "$GNPS_BASE" \
  "$GNPS_BASE_SHA256" "$GNPS_MSG" "$MSG_MODEL_DIR" "$POLICY_DIR" \
  "$V2_POLICY_SHA256" "$RULE_LIBRARY" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

from export_chemaware_v2_grand_fusion_scores import _load_policy
from gnps_pair_score_cache import load_pair_score_cache
from noise_msg_pair_evidence_core import MOLECULE_FEATURES

enveda, layer, gnps, gnps_base, expected_gnps_sha, gnps_msg, msg_model, policy_dir, expected_policy_sha, rules = (
    Path(value) if index not in (4, 8) else value
    for index, value in enumerate(sys.argv[1:])
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def edge_counts(benchmark: Path) -> dict[str, int]:
    output = {}
    for panel in ('identity_disjoint', 'formula_disjoint'):
        with np.load(benchmark / f'panel_{panel}.npz', allow_pickle=False) as body:
            output[panel] = int(len(body['candidate_row']))
    return output

layer_report = json.loads((layer / 'scoreblind_report.json').read_text(encoding='utf-8'))
layer_freeze = json.loads((layer / 'freeze_contract.json').read_text(encoding='utf-8'))
layer_meta = json.loads((layer / 'method_scores_unopened.npz.json').read_text(encoding='utf-8'))
layer_bundle = layer / 'method_scores_unopened.npz'
if layer_report.get('status') != 'ENVEDA180_LAYERED_COMPONENT_SCORES_READY_UNOPENED':
    raise SystemExit('ENVEDA_UNIFIED_LAYER_STATUS_INVALID')
if layer_report.get('score_bundle_sha256') != sha256(layer_bundle):
    raise SystemExit('ENVEDA_UNIFIED_LAYER_SCORE_HASH_MISMATCH')
if layer_freeze.get('benchmark_directory') != str(enveda):
    raise SystemExit('ENVEDA_UNIFIED_LAYER_BENCHMARK_MISMATCH')
if layer_meta.get('benchmark') not in (None, str(enveda)):
    raise SystemExit('ENVEDA_UNIFIED_LAYER_METADATA_BENCHMARK_MISMATCH')
expected_enveda_rows = edge_counts(enveda)
declared_enveda_rows = layer_meta.get('score_rows', {})
if any(declared_enveda_rows.get(key) != value for key, value in expected_enveda_rows.items()):
    raise SystemExit('ENVEDA_UNIFIED_LAYER_EDGE_AXIS_MISMATCH')

if sha256(gnps_base) != expected_gnps_sha:
    raise SystemExit('ENVEDA_UNIFIED_GNPS_BASE_HASH_MISMATCH')
gnps_meta = json.loads(Path(f'{gnps_base}.json').read_text(encoding='utf-8'))
if gnps_meta.get('benchmark') not in (None, str(gnps)):
    raise SystemExit('ENVEDA_UNIFIED_GNPS_BASE_BENCHMARK_MISMATCH')
expected_gnps_rows = edge_counts(gnps)
declared_gnps_rows = gnps_meta.get('score_rows', {})
if any(declared_gnps_rows.get(key) != value for key, value in expected_gnps_rows.items()):
    raise SystemExit('ENVEDA_UNIFIED_GNPS_BASE_EDGE_AXIS_MISMATCH')
load_pair_score_cache(gnps_msg, gnps)

msg_report = json.loads((msg_model / 'report.json').read_text(encoding='utf-8'))
msg_payload = joblib.load(msg_model / 'fusion_model.joblib')
if msg_report.get('status') != 'NOISE_MSG_FUSION_FROZEN':
    raise SystemExit('ENVEDA_UNIFIED_MSG_MODEL_NOT_FROZEN')
if tuple(msg_payload.get('feature_names', ())) != tuple(MOLECULE_FEATURES):
    raise SystemExit('ENVEDA_UNIFIED_MSG_FEATURE_CONTRACT_MISMATCH')

policy_args = argparse.Namespace(
    policy_dir=policy_dir, rule_library=rules,
    expected_policy_sha256=expected_policy_sha,
)
policy, _, _ = _load_policy(policy_args)
expected_controls = (
    'rule_response_content_permuted',
    'rule_response_content_permuted_b',
    'rule_response_content_permuted_c',
)
if tuple(policy.get('control_rule_keys', ())) != expected_controls:
    raise SystemExit('ENVEDA_UNIFIED_V2_THREE_NULL_CONTRACT_MISMATCH')
print('ENVEDA_UNIFIED_FROZEN_INPUT_PREFLIGHT_PASS')
PY

if [[ -n "$REUSE_RUN" ]]; then
  # A failed later stage must not force three completed encoder passes and the
  # frozen MSG application to run again.  Reuse is accepted only when every
  # cache matches the current checkpoint bytes and exact Enveda candidate axis.
  python - "$ENVEDA" "$ENVEDA_LAYER" "$MSG_MODEL_DIR" "$ENVEDA_MSG_CACHE" \
    "$ENVEDA_STAGE1_EMBEDDINGS" chemaware_stage1 "$STAGE1" \
    "$ENVEDA_PHASEA_EMBEDDINGS" chemaware_phasea "$PHASEA" \
    "$ENVEDA_FULLMSG_EMBEDDINGS" noise_chemaware_fullmsg "$FULLMSG" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

from encode_gnps_gold_silver_10ppm_checkpoint import required_rows
from gnps_pair_score_cache import load_pair_score_cache, sha256_file

benchmark, layer, msg_model, msg_cache = map(Path, sys.argv[1:5])
specifications = sys.argv[5:]
if len(specifications) != 9:
    raise SystemExit('ENVEDA_UNIFIED_REUSE_ARGUMENT_CONTRACT_BROKEN')
panels = ('identity_disjoint', 'formula_disjoint')
expected_rows = required_rows(benchmark, 0, panels)
expected_preprocessing = {
    'n_highest_peaks': 100,
    'precursor_intensity': 1.1,
    'precision': 32,
    'augmentation': False,
}

for offset in range(0, len(specifications), 3):
    cache = Path(specifications[offset])
    label = specifications[offset + 1]
    checkpoint = Path(specifications[offset + 2])
    sidecar = cache.with_suffix('.json')
    report = json.loads(sidecar.read_text(encoding='utf-8'))
    if report.get('status') != 'gnps_gold_silver_10ppm_checkpoint_encoding_complete':
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_INCOMPLETE: {cache}')
    if report.get('label') != label:
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_LABEL_MISMATCH: {cache}')
    if report.get('checkpoint_sha256') != sha256_file(checkpoint):
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_CHECKPOINT_MISMATCH: {cache}')
    expected_panel_hashes = {
        name: sha256_file(benchmark / f'panel_{name}.npz') for name in panels
    }
    if report.get('panel_sha256') != expected_panel_hashes:
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_PANEL_MISMATCH: {cache}')
    if report.get('native_preprocessing') != expected_preprocessing:
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_PREPROCESSING_MISMATCH: {cache}')
    with np.load(cache, allow_pickle=False) as body:
        rows = np.asarray(body['rows'], dtype=np.int64)
        embeddings = np.asarray(body['embeddings'], dtype=np.float32)
    if not np.array_equal(rows, expected_rows):
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_AXIS_MISMATCH: {cache}')
    if embeddings.shape != (len(expected_rows), 1024):
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_SHAPE_MISMATCH: {cache}')
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 2e-3:
        raise SystemExit(f'ENVEDA_UNIFIED_REUSE_EMBEDDING_VALUES_INVALID: {cache}')

msg_report, _ = load_pair_score_cache(msg_cache, benchmark)
sources = msg_report.get('method', {}).get('sources', {})
expected_sources = {
    'score_bundle_sha256': sha256_file(layer / 'method_scores_unopened.npz'),
    'official_embeddings_sha256': sha256_file(layer / 'official_embeddings.npz'),
    'v1_embeddings_sha256': sha256_file(layer / 'noise_v1_embeddings.npz'),
    'model_report_sha256': sha256_file(msg_model / 'report.json'),
}
if sources != expected_sources:
    raise SystemExit('ENVEDA_UNIFIED_REUSE_MSG_PROVENANCE_MISMATCH')
print('ENVEDA_UNIFIED_REUSE_INPUTS_VALIDATED')
PY
fi

[[ ! -e "$OUT" ]] || { echo "ENVEDA_UNIFIED_OUTPUT_EXISTS: $OUT" >&2; exit 4; }
mkdir -p "$OUT/inputs" "$OUT/gnps_inputs" "$SCRATCH"
trap 'rm -rf -- "$SCRATCH"' EXIT
if [[ -n "$REUSE_RUN" ]]; then
  printf '%s\n' "$REUSE_RUN" > "$OUT/REUSED_ENVEDA_INPUTS.txt"
  sha256sum \
    "$ENVEDA_STAGE1_EMBEDDINGS" "${ENVEDA_STAGE1_EMBEDDINGS%.npz}.json" \
    "$ENVEDA_PHASEA_EMBEDDINGS" "${ENVEDA_PHASEA_EMBEDDINGS%.npz}.json" \
    "$ENVEDA_FULLMSG_EMBEDDINGS" "${ENVEDA_FULLMSG_EMBEDDINGS%.npz}.json" \
    "$ENVEDA_MSG_CACHE/report.json" \
    "$ENVEDA_MSG_CACHE/pair_scores_identity_disjoint.npy" \
    "$ENVEDA_MSG_CACHE/pair_scores_formula_disjoint.npy" \
    "$MSG_MODEL_DIR/fusion_model.joblib" "$MSG_MODEL_DIR/report.json" \
    > "$OUT/REUSED_ENVEDA_INPUTS.sha256"
fi

python -m py_compile \
  tasks/enveda_unified_candidate_integrator_core.py \
  tasks/train_enveda_unified_candidate_integrator.py \
  tasks/evaluate_enveda_integrator_on_gnps.py \
  tasks/build_gnps_chemaware_kernel_cache.py \
  tasks/export_gnps_chemaware_continuous_null_evidence.py \
  tasks/apply_noise_msg_fusion_to_gnps.py
python -u tasks/test_enveda_unified_candidate_integrator.py

encode_one() {
  local benchmark="$1" name="$2" checkpoint="$3" output="$4" gpu="$5"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$benchmark" --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCH" --output "$output" --label "$name" \
    --batch-size 64 --expected-selected-rows 0 --expected-records 0 \
    --panels identity_disjoint formula_disjoint
}

# The official fine-tuned DreaMS embedding baseline is already the
# `official_dreams` channel in both frozen score bundles.  The SSL checkpoint
# above supplies architecture only and is never reported as that baseline.
if [[ -z "$REUSE_RUN" ]]; then
  encode_one "$ENVEDA" chemaware_stage1 "$STAGE1" \
    "$ENVEDA_STAGE1_EMBEDDINGS" "${GPUS[0]}" & p0=$!
  encode_one "$ENVEDA" chemaware_phasea "$PHASEA" \
    "$ENVEDA_PHASEA_EMBEDDINGS" "${GPUS[1]}" & p1=$!
  set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
  (( s0 == 0 && s1 == 0 )) || { echo "ENVEDA_UNIFIED_ENCODING_FAILED: stage1/phasea=$s0/$s1" >&2; exit 6; }

  encode_one "$ENVEDA" noise_chemaware_fullmsg "$FULLMSG" \
    "$ENVEDA_FULLMSG_EMBEDDINGS" "${GPUS[0]}"

  python -u tasks/apply_noise_msg_fusion_to_gnps.py \
    --benchmark "$ENVEDA" \
    --score-bundle "$ENVEDA_LAYER/method_scores_unopened.npz" \
    --official-embeddings "$ENVEDA_LAYER/official_embeddings.npz" \
    --v1-embeddings "$ENVEDA_LAYER/noise_v1_embeddings.npz" \
    --model "$MSG_MODEL_DIR" --output "$ENVEDA_MSG_CACHE"
else
  echo "ENVEDA_UNIFIED_REUSING_VALIDATED_INPUTS: $REUSE_RUN"
fi

python -u tasks/build_gnps_chemaware_kernel_cache.py \
  --benchmark "$ENVEDA" \
  --official-embeddings "$ENVEDA_LAYER/official_embeddings.npz" \
  --output "$OUT/inputs/enveda_chem_tokens" --n-highest-peaks 100 \
  --staging-dir "$SCRATCH/enveda_chem_tokens_build" \
  --panels identity_disjoint formula_disjoint \
  --expected-selected-rows 0 --expected-records 0

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/export_gnps_chemaware_continuous_null_evidence.py \
  --policy-dir "$POLICY_DIR" --token-dir "$OUT/inputs/enveda_chem_tokens" \
  --rule-library "$RULE_LIBRARY" \
  --panel "identity_disjoint=$ENVEDA/panel_identity_disjoint.npz" \
  --panel "formula_disjoint=$ENVEDA/panel_formula_disjoint.npz" \
  --output "$OUT/inputs/enveda_continuous_chem" --batch-queries 256

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/train_enveda_unified_candidate_integrator.py \
  --benchmark "$ENVEDA" \
  --score-bundle "$ENVEDA_LAYER/method_scores_unopened.npz" \
  --msg-cache "$ENVEDA_MSG_CACHE" \
  --stage1-embeddings "$ENVEDA_STAGE1_EMBEDDINGS" \
  --phasea-embeddings "$ENVEDA_PHASEA_EMBEDDINGS" \
  --fullmsg-embeddings "$ENVEDA_FULLMSG_EMBEDDINGS" \
  --chem-dir "$OUT/inputs/enveda_continuous_chem" \
  --output "$OUT/modeling" --folds 5 --epochs 20 --batch-queries 256 \
  --hidden 16 --learning-rate 0.001 --weight-decay 0.0001 \
  --anchor-strength 0.01 --bootstrap-resamples 5000 --device cuda

# GNPS is not read until the complete Enveda OOF report and all-data refit exist.
encode_one "$GNPS" chemaware_stage1 "$STAGE1" \
  "$OUT/gnps_inputs/chemaware_stage1_embeddings.npz" "${GPUS[0]}" & p0=$!
encode_one "$GNPS" chemaware_phasea "$PHASEA" \
  "$OUT/gnps_inputs/chemaware_phasea_embeddings.npz" "${GPUS[1]}" & p1=$!
set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
(( s0 == 0 && s1 == 0 )) || { echo "ENVEDA_UNIFIED_GNPS_ENCODING_FAILED: stage1/phasea=$s0/$s1" >&2; exit 7; }

encode_one "$GNPS" noise_chemaware_fullmsg "$FULLMSG" \
  "$OUT/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz" "${GPUS[0]}"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -u tasks/evaluate_enveda_integrator_on_gnps.py \
  --benchmark "$GNPS" --checkpoint "$OUT/modeling/unified_candidate_integrator.pt" \
  --score-bundle "$GNPS_BASE" --msg-cache "$GNPS_MSG" \
  --stage1-embeddings "$OUT/gnps_inputs/chemaware_stage1_embeddings.npz" \
  --phasea-embeddings "$OUT/gnps_inputs/chemaware_phasea_embeddings.npz" \
  --fullmsg-embeddings "$OUT/gnps_inputs/noise_chemaware_fullmsg_embeddings.npz" \
  --chem-dir "$GNPS_CHEM" \
  --comparator-score "B_gnps_nested_oof=$B_PREDICTIONS" \
  --output "$OUT/gnps_fixed_evaluation" --device cuda --bootstrap-resamples 5000

sha256sum \
  "$OUT/modeling/unified_candidate_integrator.pt" \
  "$OUT/modeling/candidate_predictions_oof.npz" \
  "$OUT/modeling/report.json" \
  "$OUT/gnps_fixed_evaluation/report.json" \
  > "$OUT/SHA256SUMS.txt"
echo "ENVEDA_UNIFIED_CANDIDATE_INTEGRATOR_AND_FIXED_GNPS_COMPLETE: $OUT"
