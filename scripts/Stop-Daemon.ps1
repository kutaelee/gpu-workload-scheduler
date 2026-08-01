[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PidPath = Join-Path $RepoRoot '.runtime\server.pid'
if (-not (Test-Path -LiteralPath $PidPath -PathType Leaf)) {
    Write-Host 'GPU queue daemon is not running.'
    exit 0
}

$DaemonPid = [int](Get-Content -Raw -LiteralPath $PidPath)
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $DaemonPid"
if (-not $process) { throw 'Stale daemon PID file found; no process is running.' }
$expected = [Regex]::Escape($RepoRoot)
if ($process.CommandLine -notmatch 'gpuq\.server' -or $process.CommandLine -notmatch $expected) {
    throw "PID $DaemonPid does not belong to this GPU queue daemon."
}
$config = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot '.runtime\config.json') | ConvertFrom-Json
$headers = @{ 'X-GPUQ-Token' = $config.api_token }
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8790/api/shutdown' -Headers $headers -ContentType 'application/json' -Body '{}' | Out-Null
Wait-Process -Id $DaemonPid -Timeout 10 -ErrorAction SilentlyContinue
if (Get-Process -Id $DaemonPid -ErrorAction SilentlyContinue) {
    throw "GPU queue daemon PID $DaemonPid did not stop within 10 seconds."
}
Write-Host "Gracefully stopped GPU queue daemon PID $DaemonPid"
