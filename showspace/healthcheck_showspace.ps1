$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ShowspaceDir '.env'
$port = 7860
$hostName = '127.0.0.1'

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $name, $value = $line.Split('=', 2)
            if ($name.Trim() -eq 'GRADIO_SERVER_PORT') { $port = [int]$value.Trim() }
            if ($name.Trim() -eq 'GRADIO_SERVER_NAME') { $hostName = $value.Trim() }
        }
    }
}

$url = "http://$hostName`:$port/"
$response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15
if ($response.StatusCode -ne 200) { throw "Unexpected HTTP status: $($response.StatusCode)" }
Write-Host "showspace healthy: $url (HTTP $($response.StatusCode))"
