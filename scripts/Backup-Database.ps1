[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$TargetRoot = 'E:\Data\DB\Dumps\gpu-workload-scheduler'
$Target = Join-Path $TargetRoot "gpuq-$Stamp.dump"
$ContainerTarget = "/tmp/gpuq-$Stamp.dump"
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

Push-Location $RepoRoot
try {
    $container = (docker compose ps -q postgres).Trim()
    if (-not $container) { throw 'GPU queue PostgreSQL container is not running.' }
    docker exec $container pg_dump -U gpuq -d gpuq -Fc -f $ContainerTarget
    if ($LASTEXITCODE -ne 0) { throw 'pg_dump failed.' }
    docker cp "${container}:${ContainerTarget}" $Target
    if ($LASTEXITCODE -ne 0) { throw 'docker cp failed.' }
    docker exec $container rm -f -- $ContainerTarget
}
finally {
    Pop-Location
}
Write-Host "Created logical dump $Target"
