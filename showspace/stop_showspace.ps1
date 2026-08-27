$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path (Join-Path $ShowspaceDir 'run') 'showspace.pid'

if (-not (Test-Path $PidFile)) {
    Write-Host 'showspace is not running (PID file not found).'
    exit 0
}

$showspacePid = [int](Get-Content $PidFile | Select-Object -First 1)
$process = Get-Process -Id $showspacePid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $showspacePid -Force
    Write-Host "Stopped showspace PID $showspacePid"
} else {
    Write-Host "No process found for PID $showspacePid"
}
Remove-Item $PidFile -Force
