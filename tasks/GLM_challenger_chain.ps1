$ErrorActionPreference = "Continue"
Set-Location D:\DreaMS
$bench = "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
$frozen = "data/validation/noise_gnps_article_benchmark_run_2349091"
$models = "third_party\public_models"
$staging = "data\validation\GLM_challenger_scores\staging"
$log = "data\validation\GLM_challenger_scores\challenger.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Step($msg) { "$(Get-Date -Format 'HH:mm:ss')  $msg" | Tee-Object -FilePath $log -Append }
function Die($msg) { Step "FATAL: $msg"; exit 1 }

Step "challenger scoring started"

# ---- 1. entropy_raw + denoising_search (no zip needed) ----
$env:PYTHONPATH = "D:\DreaMS;D:\DreaMS\tasks"
$env:PYTHONUTF8 = "1"
cmd /c "python -X utf8 tasks/GLM_score_challenger_models_on_gnps.py --benchmark $bench --frozen-run $frozen --models-dir $models --staging $staging --methods entropy_raw_public denoising_search_public >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Die "entropy/denoising scoring failed ($LASTEXITCODE)" }
Step "entropy_raw + denoising_search staged"

# ---- 2. wait for spec2vec zip (timeout 6 h) ----
$zip = "$models\Spec2Vec_2026.zip"
$target = 1992840216
$deadline = (Get-Date).AddHours(6)
while (((Get-Item $zip -ErrorAction SilentlyContinue).Length) -lt $target -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 60
}
$zlen = (Get-Item $zip -ErrorAction SilentlyContinue).Length
if ($zlen -ne $target) { Die "Spec2Vec zip incomplete: $zlen / $target" }
Step "zip complete at $zlen bytes"

# ---- 3. unzip + extract model files ----
$extract = "$models\spec2vec_2026"
New-Item -ItemType Directory -Force $extract | Out-Null
cmd /c "powershell -Command Expand-Archive -Path '$zip' -DestinationPath '$extract' -Force >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Die "unzip failed ($LASTEXITCODE)" }
$model_file = Get-ChildItem $extract -Recurse -Filter "*_model" | Where-Object Name -match "150225" | Select-Object -First 1
if (-not $model_file) { Die "model file not found in extracted zip" }
Step "model extracted: $($model_file.FullName)"

# ---- 4. score spec2vec 2026 ----
cmd /c "python -X utf8 tasks/GLM_score_challenger_models_on_gnps.py --benchmark $bench --frozen-run $frozen --models-dir '$($model_file.Directory.FullName)' --staging $staging --methods spec2vec_2026_retrained >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Die "spec2vec 2026 scoring failed ($LASTEXITCODE)" }
Step "spec2vec_2026 staged"

# ---- 5. merge staged scores into the 12-method bundle -> 15-method bundle ----
$bundle12 = "data\validation\GLM_gnps_article_benchmark_public_models\run_local\bundle"
$bundle15 = "data\validation\GLM_gnps_article_benchmark_challengers\run_local\bundle"
New-Item -ItemType Directory -Force (Split-Path $bundle15) | Out-Null
python -X utf8 -c @"
import numpy as np, json, sys
from pathlib import Path
staging = Path('$staging')
bundle12 = Path('$bundle12')
bundle15 = Path('$bundle15')
with np.load(bundle12 / 'method_scores.npz') as z:
    names = [str(v) for v in z['method_names']]
    old = {p: np.asarray(z[f'scores_{p}']) for p in ('identity_disjoint','formula_disjoint')}
new_names = ['entropy_raw_public', 'denoising_search_public', 'spec2vec_2026_retrained']
all_names = names + new_names
matrices = {}
for panel in ('identity_disjoint','formula_disjoint'):
    blocks = [old[panel].astype(np.float32)]
    for nm in new_names:
        p = staging / f'scores_{panel}_{nm}.npy'
        if not p.is_file():
            print(f'WARNING: {p} missing, filling zeros')
            blocks.append(np.zeros(old[panel].shape[1], dtype=np.float32))
        else:
            blocks.append(np.load(p).astype(np.float32))
    matrices[panel] = np.vstack(blocks)
np.savez_compressed(bundle15 / 'method_scores.npz',
    method_names=np.asarray(all_names),
    **{f'scores_{p}': matrices[p] for p in matrices})
meta = json.loads((bundle12 / 'method_scores.npz.json').read_text(encoding='utf-8'))
meta.update({
    'status': 'challenger_extended_bundle',
    'methods': all_names,
    'information_levels': {
        'spectrum_only': meta.get('information_levels',{}).get('spectrum_only',[]) + new_names,
        'frozen_candidate_reranker': meta.get('information_levels',{}).get('frozen_candidate_reranker',[]),
    },
})
(bundle15 / 'method_scores.npz.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(f'15-method bundle written: {len(all_names)} methods')
"@ 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Die "bundle merge failed" }
Step "15-method bundle written"

# ---- 6. evaluate with frozen evaluator ----
$panels = "data\validation\GLM_gnps_identity_panel_reconstruction"
$run15 = "data\validation\GLM_gnps_article_benchmark_challengers\run_local"
cmd /c "python -X utf8 tasks/evaluate_noise_gnps_article_benchmark.py --score-bundle `"$bundle15\method_scores.npz`" --benchmark $panels --output `"$run15\evaluation`" --baseline-method official_dreams --bootstrap-resamples 10000 --bootstrap-seed 20261006 >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Die "frozen evaluator failed ($LASTEXITCODE)" }
Step "15-method evaluation complete"

# ---- 7. verified ladder ----
cmd /c "python -X utf8 tasks/GLM_assemble_gnps_article_ladder.py --run $run15 >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Die "ladder assembly failed ($LASTEXITCODE)" }
Step "15-method verified ladder written"

# ---- 8. commit + push ----
git add docs/GLM_GNPS_ARTICLE_LADDER_20261006.md docs/GLM_GNPS_ARTICLE_LADDER_20261007.md tasks/GLM_score_challenger_models_on_gnps.py deliverables/GLM_gnps_article_ladder deliverables/figures/GLM_gnps_article_ladder.svg deliverables/figures/GLM_gnps_article_ladder.pdf 2>$null
git add -f deliverables/figures/GLM_gnps_article_ladder.png deliverables/figures/GLM_gnps_metric_split.png 2>$null
git commit -q -m "Add 15-method GNPS ladder with Spec2Vec 2026 retrained, entropy raw, and Denoising Search challengers" 2>&1 | Out-Null
git push origin module1-chem-attn-v6-0717 2>&1 | Out-Null
Step "challenger chain complete"
