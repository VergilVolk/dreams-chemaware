$ErrorActionPreference = 'Stop'
$ShowspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ShowspaceDir '.env'
$port = 7860
$hostName = '127.0.0.1'
$username = ''
$password = ''

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $name, $value = $line.Split('=', 2)
            if ($name.Trim() -eq 'GRADIO_SERVER_PORT') { $port = [int]$value.Trim() }
            if ($name.Trim() -eq 'GRADIO_SERVER_NAME') { $hostName = $value.Trim() }
            if ($name.Trim() -eq 'SHOWSPACE_USERNAME') { $username = $value.Trim() }
            if ($name.Trim() -eq 'SHOWSPACE_PASSWORD') { $password = $value.Trim() }
        }
    }
}

$url = "http://$hostName`:$port/"
$headers = @{}
if ($username -and $password) {
    $bytes = [Text.Encoding]::ASCII.GetBytes("$username`:$password")
    $headers.Authorization = 'Basic ' + [Convert]::ToBase64String($bytes)
}
$response = Invoke-WebRequest -Uri $url -Headers $headers -UseBasicParsing -TimeoutSec 15
if ($response.StatusCode -ne 200) { throw "Unexpected HTTP status: $($response.StatusCode)" }
Write-Host "showspace healthy: $url (HTTP $($response.StatusCode))"
