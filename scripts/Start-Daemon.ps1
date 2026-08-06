[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$StartupLog = Join-Path $RepoRoot '.runtime\startup.log'
$ServerLog = Join-Path $RepoRoot '.runtime\server.log'
$RuntimeConfig = Get-Content -Raw (Join-Path $RepoRoot '.runtime\config.json') | ConvertFrom-Json
$DatabasePort = ([Uri]$RuntimeConfig.database_url).Port
$env:PYTHONPATH = $RepoRoot
Set-Location -LiteralPath $RepoRoot

function Write-StartupLog([string]$Message) {
    $line = '{0:o} {1}' -f (Get-Date), $Message
    Add-Content -LiteralPath $StartupLog -Value $line -Encoding UTF8
}

function Test-LocalPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connect = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne([TimeSpan]::FromSeconds(3))) {
            return $false
        }
        $client.EndConnect($connect)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Wait-ForDocker {
    $attempt = 0
    while ($true) {
        $attempt++
        & $env:ComSpec /d /c 'docker info >nul 2>&1'
        if ($LASTEXITCODE -eq 0) {
            if ($attempt -gt 1) {
                Write-StartupLog "Docker Desktop became ready after $attempt attempts."
            }
            return $true
        }
        if ($attempt -eq 1 -or $attempt % 6 -eq 0) {
            Write-StartupLog "Waiting for Docker Desktop (attempt $attempt; no startup deadline)."
        }
        Start-Sleep -Seconds 10
    }
}

function Ensure-DatabaseReachable([int]$Port) {
    $composeCommand = 'docker compose up -d --wait --wait-timeout 300 postgres >> "' +
        $StartupLog + '" 2>&1'
    & $env:ComSpec /d /c $composeCommand
    if ($LASTEXITCODE -ne 0) {
        Write-StartupLog "PostgreSQL startup failed with exit code $LASTEXITCODE."
        return $false
    }

    if (Test-LocalPort $Port) {
        return $true
    }

    # Docker can report a healthy container while its Windows loopback publish rule
    # was lost during host reboot. Recreate only this disposable container; its named
    # PostgreSQL volume is retained and is never copied or removed here.
    Write-StartupLog "PostgreSQL container is healthy but port $Port is unreachable; recreating its host publish rule."
    $recreateCommand = 'docker compose up -d --force-recreate --wait --wait-timeout 300 postgres >> "' +
        $StartupLog + '" 2>&1'
    & $env:ComSpec /d /c $recreateCommand
    if ($LASTEXITCODE -ne 0) {
        Write-StartupLog "PostgreSQL recreate failed with exit code $LASTEXITCODE."
        return $false
    }

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        if (Test-LocalPort $Port) {
            Write-StartupLog "PostgreSQL host port $Port is reachable after recreate."
            return $true
        }
        Start-Sleep -Seconds 3
    }
    Write-StartupLog "PostgreSQL is healthy but host port $Port remains unreachable."
    return $false
}

[void](Wait-ForDocker)

# The scheduled task owns this supervisor. It keeps the API recoverable after
# transient DB/Docker failures instead of consuming its finite Task Scheduler
# restart budget and leaving the GPU queue offline for the rest of the session.
$delaySeconds = 5
while ($true) {
    if (-not (Ensure-DatabaseReachable $DatabasePort)) {
        Write-StartupLog "Database is not reachable; retrying supervisor startup in $delaySeconds seconds."
        Start-Sleep -Seconds $delaySeconds
        $delaySeconds = [Math]::Min($delaySeconds * 2, 300)
        continue
    }

    $delaySeconds = 5
    Write-StartupLog 'PostgreSQL is reachable; starting GPU queue API.'
    & $Python -m gpuq.server *>> $ServerLog
    $code = $LASTEXITCODE
    Write-StartupLog "GPU queue API exited with code $code; restarting in $delaySeconds seconds."
    Start-Sleep -Seconds $delaySeconds
    $delaySeconds = [Math]::Min($delaySeconds * 2, 300)
}
