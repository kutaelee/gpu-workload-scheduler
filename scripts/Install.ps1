[CmdletBinding()]
param(
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $RepoRoot '.runtime'
if (-not $Python) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $Python = $pythonCommand.Source }
}

if (-not $Python -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python 3.12 or newer was not found. Pass -Python with an absolute python.exe path.'
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI is not available.'
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
New-Item -ItemType Directory -Force -Path 'E:\Data\GpuScheduler\Logs' | Out-Null
New-Item -ItemType Directory -Force -Path 'E:\Data\DB\Dumps\gpu-workload-scheduler' | Out-Null

function New-HexSecret([int]$Bytes = 32) {
    $buffer = [byte[]]::new($Bytes)
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($buffer) }
    finally { $generator.Dispose() }
    return -join ($buffer | ForEach-Object { $_.ToString('x2') })
}

function Write-Utf8NoBom([string]$Path, [string]$Content) {
    [IO.File]::WriteAllText($Path, $Content, [Text.UTF8Encoding]::new($false))
}

$EnvPath = Join-Path $RepoRoot '.env'
$ConfigPath = Join-Path $RuntimeRoot 'config.json'
if (-not (Test-Path -LiteralPath $EnvPath)) {
    $dbPassword = New-HexSecret 24
    $envContent = @(
        'POSTGRES_DB=gpuq'
        'POSTGRES_USER=gpuq'
        "POSTGRES_PASSWORD=$dbPassword"
        'GPUQ_POSTGRES_PORT=55670'
    ) -join [Environment]::NewLine
    Write-Utf8NoBom $EnvPath ($envContent + [Environment]::NewLine)
}

$settings = @{}
Get-Content -LiteralPath $EnvPath | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') { $settings[$matches[1]] = $matches[2] }
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    $configContent = @{
        database_url = "postgresql://$($settings.POSTGRES_USER):$($settings.POSTGRES_PASSWORD)@127.0.0.1:$($settings.GPUQ_POSTGRES_PORT)/$($settings.POSTGRES_DB)"
        api_token = New-HexSecret 32
        api_host = '127.0.0.1'
        api_port = 8790
        log_root = 'E:\Data\GpuScheduler\Logs'
        safety_vram_mb = 2048
        fairness_window_minutes = 60
        max_parallel_jobs = 2
        poll_seconds = 2.0
        cancel_grace_seconds = 30.0
        terminate_grace_seconds = 10.0
        post_job_cooldown_seconds = 2.0
        wsl_force_terminate = $false
    } | ConvertTo-Json
    Write-Utf8NoBom $ConfigPath $configContent
}

$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv (Join-Path $RepoRoot '.venv')
}
& $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $RepoRoot 'requirements.lock')

Push-Location $RepoRoot
try {
    docker compose up -d postgres
    $healthy = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $state = docker inspect --format '{{.State.Health.Status}}' gpu-workload-scheduler-postgres-1 2>$null
        if ($state -eq 'healthy') { $healthy = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $healthy) { throw 'GPU queue PostgreSQL did not become healthy.' }
}
finally {
    Pop-Location
}

Write-Host "Installed GPU queue runtime at $RepoRoot"
