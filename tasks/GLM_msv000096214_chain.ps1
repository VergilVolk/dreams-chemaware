$ErrorActionPreference = "Continue"
Set-Location D:\DreaMS
$env:PYTHONUTF8 = "1"
$root = "data/external/msv000096214"
$val = "data/validation"
$log = "$root/chain.log"
function Step($msg) { "$(Get-Date -Format 'HH:mm:ss')  $msg" | Tee-Object -FilePath $log -Append }

Step "MSV000096214 truth-blind chain started"

# 0. wait for calibration (frozen threshold file)
while (-not (Test-Path "$root/calibration.json")) { Start-Sleep -Seconds 60 }
Step "calibration.json present"

# 1. wait for download to complete (report.json written by download script)
while (-not (Test-Path "$root/report.json")) { Start-Sleep -Seconds 60 }
$rep = Get-Content "$root/report.json" | ConvertFrom-Json
Step "download complete: files_ok=$($rep.files_ok)/$($rep.files_expected)"
if ($rep.files_ok -lt $rep.files_expected) { Step "WARNING: incomplete download, extracting available files" }

# 2. extraction
Step "extraction starting"
python -X utf8 tasks/extract_msv000096214_ms1_features.py 2>&1 | Select-Object -Last 5 | Tee-Object -FilePath $log -Append
Step "extraction done"

# 3. alignment + gap filling
Step "alignment starting"
python -X utf8 tasks/align_msv000096214_consensus.py 2>&1 | Select-Object -Last 5 | Tee-Object -FilePath $log -Append
Step "alignment done"

# 4. truth-blind table audit
Step "table audit starting"
python -X utf8 tasks/audit_msv000096214_feature_table.py --output "$val/msv000096214_table_audit" 2>&1 | Tee-Object -FilePath $log -Append
$audit = Get-Content "$val/msv000096214_table_audit/report.json" | ConvertFrom-Json
Step "audit pass_to_effect_stage=$($audit.pass_to_effect_stage)"

# 5. effect stage (only if audit passed)
if ($audit.pass_to_effect_stage) {
  Step "effect stage starting (genotype labels opened per contract)"
  python -X utf8 tasks/analyze_msv000096214_effect_stage.py 2>&1 | Tee-Object -FilePath $log -Append
  Step "effect stage done"
} else {
  Step "CHAIN STOPPED: feature-table audit failed - no effect stage"
}
Step "chain finished"
