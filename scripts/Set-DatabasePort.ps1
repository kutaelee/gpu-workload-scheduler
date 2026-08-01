[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 55670
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path $RepoRoot '.env'
$ConfigPath = Join-Path $RepoRoot '.runtime\config.json'

if (-not (Test-Path -LiteralPath $EnvPath) -or -not (Test-Path -LiteralPath $ConfigPath)) {
    throw 'Run scripts/Install.ps1 before changing the scheduler database port.'
}

$envLines = Get-Content -LiteralPath $EnvPath
$portLineFound = $false
$updatedEnv = foreach ($line in $envLines) {
    if ($line -match '^GPUQ_POSTGRES_PORT=') {
        $portLineFound = $true
        "GPUQ_POSTGRES_PORT=$Port"
    }
    else {
        $line
    }
}
if (-not $portLineFound) {
    $updatedEnv += "GPUQ_POSTGRES_PORT=$Port"
}

$config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
$databaseUri = [UriBuilder]$config.database_url
$databaseUri.Port = $Port
$config.database_url = $databaseUri.Uri.AbsoluteUri

$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllLines($EnvPath, [string[]]$updatedEnv, $utf8)
[IO.File]::WriteAllText($ConfigPath, ($config | ConvertTo-Json), $utf8)
Write-Host "Configured GPU scheduler PostgreSQL host port $Port. Restart the scheduler task to apply it."
