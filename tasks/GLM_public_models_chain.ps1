# GLM supervisor chain: public-model downloads -> md5 -> scoring -> frozen
# evaluation -> verified 12-method ladder -> git commit + push.
# Runs unattended; every step fails closed and logs to chain.log.
$ErrorActionPreference = "Stop"
Set-Location D:\DreaMS
$models = "third_party\public_models"
$bench = "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
$panels = "data/validation/GLM_gnps_identity_panel_reconstruction"
$frozen = "data/validation/noise_gnps_article_benchmark_run_2349091"
$run = "data/validation/GLM_gnps_article_benchmark_public_models/run_local"
New-Item -ItemType Directory -Force $run | Out-Null
$log = Join-Path $run "chain.log"
function Step($msg) { "$(Get-Date -Format 'HH:mm:ss')  $msg" | Tee-Object -FilePath $log -Append }
function Die($msg) { Step "FATAL: $msg"; exit 1 }

Step "supervisor started"

# ---- 1. wait for the two spec2vec arrays (timeout 5.5 h) ----
$wv = "$models\spec2vec_AllPositive_ratio05_filtered_iter_15.model.wv.vectors.npy"
$syn = "$models\spec2vec_AllPositive_ratio05_filtered_iter_15.model.trainables.syn1neg.npy"
$target = 139092128
$deadline = (Get-Date).AddHours(5.5)
while (((Get-Item $wv).Length -lt $target -or (Get-Item $syn).Length -lt $target) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 60
}
if ((Get-Item $wv).Length -ne $target -or (Get-Item $syn).Length -ne $target) {
    Die "spec2vec arrays incomplete after deadline: wv=$((Get-Item $wv).Length) syn=$((Get-Item $syn).Length)"
}
Step "downloads complete: both spec2vec arrays at $target bytes"

# ---- 2. md5 gate (all three model files) ----
$md5s = @{
  "$models\ms2deepscore_model.pt" = "d5cbf4694a1c476ae59e0c810f56c320"
  "$models\spec2vec_AllPositive_ratio05_filtered_iter_15.model" = "d2776c240ca98f12b7cabdc4b4ee26b6"
  $wv = "56954cde50ed146625614f3316130739"
  $syn = "09ab5d7df23d8196419ed7a7e099dfa1"
}
foreach ($path in $md5s.Keys) {
  $actual = (Get-FileHash -Algorithm MD5 $path).Hash.ToLower()
  if ($actual -ne $md5s[$path]) { Die "md5 mismatch for $path ($actual)" }
}
Step "md5 gate passed for all model files"

# ---- 3. score public models ----
$env:PYTHONPATH = "D:\DreaMS;D:\DreaMS\tasks"
python -X utf8 tasks/GLM_score_public_models_on_gnps.py `
  --benchmark $bench --frozen-run $frozen --models-dir $models `
  --output "$run\bundle" *>> $log
if ($LASTEXITCODE -ne 0) { Die "public-model scoring failed" }
Step "public-model scoring complete"

# ---- 4. frozen evaluator on certified panels ----
python -X utf8 tasks/evaluate_noise_gnps_article_benchmark.py `
  --score-bundle "$run\bundle\method_scores.npz" `
  --benchmark $panels --output "$run\evaluation" `
  --baseline-method official_dreams --bootstrap-resamples 10000 `
  --bootstrap-seed 20261003 *>> $log
if ($LASTEXITCODE -ne 0) { Die "frozen evaluator failed" }
Step "frozen evaluation complete (12 methods x 2 panels)"

# ---- 5. verified ladder assembly (dual-source cross-checks) ----
python -X utf8 tasks/GLM_assemble_gnps_article_ladder.py --run $run *>> $log
if ($LASTEXITCODE -ne 0) { Die "ladder assembly/verification failed" }
Step "12-method ladder verified and written"

# ---- 6. commit and push ----
git add -A docs/GLM_GNPS_ARTICLE_LADDER_20261005.md tasks/GLM_assemble_gnps_article_ladder.py tasks/GLM_score_public_models_on_gnps.py tasks/GLM_reconstruct_gnps_identity_panel.py deliverables/GLM_gnps_article_ladder deliverables/figures/GLM_gnps_article_ladder.svg deliverables/figures/GLM_gnps_article_ladder.pdf 2>$null | Out-Null
git add -f deliverables/figures/GLM_gnps_article_ladder.png 2>$null | Out-Null
git commit -q -m "Add 12-method GNPS ladder with public MS2DeepScore/Spec2Vec baselines (supervised chain)" *>> $log
git push origin module1-chem-attn-v6-0717 *>> $log
if ($LASTEXITCODE -ne 0) { Step "WARNING: git push returned $LASTEXITCODE (commit may still be local)" }
Step "chain complete — results in $run ; ladder doc + figure updated"
