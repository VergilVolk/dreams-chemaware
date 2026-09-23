$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path (Join-Path $ShowspaceDir 'run') 'cloudflared.pid'
if (-not (Test-Path -LiteralPath $PidFile)) { Write-Host 'cloudflared is not running (PID file not found).'; exit 0 }
$tunnelPid = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$tunnelPid" -ErrorAction SilentlyContinue
if ($process -and $process.Name -eq 'cloudflared.exe') {
    Stop-Process -Id $tunnelPid -Force
    Write-Host "Stopped cloudflared PID $tunnelPid"
} else {
    Write-Host "Stale PID file; PID $tunnelPid is not cloudflared. Nothing was killed."
}
Remove-Item -LiteralPath $PidFile -Force
