[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$DumpPath
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedDump = (Resolve-Path -LiteralPath $DumpPath).Path
$TestDatabase = 'gpuq_restore_test_' + (Get-Date -Format 'yyyyMMddHHmmss')
$ContainerTarget = "/tmp/$TestDatabase.dump"
$container = $null

Push-Location $RepoRoot
try {
    $container = (docker compose ps -q postgres).Trim()
    if (-not $container) { throw 'GPU queue PostgreSQL container is not running.' }
    docker cp $ResolvedDump "${container}:${ContainerTarget}"
    if ($LASTEXITCODE -ne 0) { throw 'Could not copy the dump into PostgreSQL.' }
    docker exec $container createdb -U gpuq $TestDatabase
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the restore-test database.' }
    docker exec $container pg_restore -U gpuq -d $TestDatabase $ContainerTarget
    if ($LASTEXITCODE -ne 0) { throw 'Restore test failed.' }
    $count = docker exec $container psql -U gpuq -d $TestDatabase -Atc 'SELECT count(*) FROM jobs'
    Write-Host "Restore test passed in $TestDatabase with $count job rows."
}
finally {
    if ($container) {
        docker exec $container dropdb -U gpuq --if-exists --force $TestDatabase 2>$null
        docker exec $container rm -f -- $ContainerTarget 2>$null
    }
    Pop-Location
}
