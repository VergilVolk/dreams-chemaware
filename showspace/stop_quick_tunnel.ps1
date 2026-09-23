$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $ShowspaceDir 'run'
$PidFile = Join-Path $RunDir 'cloudflared-quick.pid'
$UrlFile = Join-Path $RunDir 'cloudflared-quick.url'

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host 'Quick tunnel is not running (PID file not found).'
    exit 0
}
$tunnelPid = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$tunnelPid" -ErrorAction SilentlyContinue
if ($process -and $process.Name -eq 'cloudflared.exe' -and $process.CommandLine -match 'tunnel.+--url') {
    Stop-Process -Id $tunnelPid -Force
    Write-Host "Stopped quick tunnel PID $tunnelPid"
} else {
    Write-Host "Stale PID file; PID $tunnelPid is not the quick tunnel. Nothing was killed."
}
Remove-Item -LiteralPath $PidFile, $UrlFile -Force -ErrorAction SilentlyContinue
