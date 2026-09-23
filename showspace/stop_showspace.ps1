$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path (Join-Path $ShowspaceDir 'run') 'showspace.pid'

if (-not (Test-Path $PidFile)) {
    Write-Host 'showspace is not running (PID file not found).'
    exit 0
}

$showspacePid = [int](Get-Content $PidFile | Select-Object -First 1)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$showspacePid" -ErrorAction SilentlyContinue
if ($process -and $process.Name -match '^python(w)?\.exe$' -and $process.CommandLine -match 'showspace[\\/]+app\.py') {
    Stop-Process -Id $showspacePid -Force
    Write-Host "Stopped showspace PID $showspacePid"
} else {
    Write-Host "Stale PID file; PID $showspacePid is not the Showspace process. Nothing was killed."
}
Remove-Item $PidFile -Force
