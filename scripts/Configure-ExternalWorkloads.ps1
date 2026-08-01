[CmdletBinding()]
param(
    [string]$OllamaExecutable
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $RepoRoot '.runtime\config.json'
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "GPU scheduler runtime config not found: $ConfigPath"
}

$config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
if (-not $OllamaExecutable) {
    $ollamaCommand = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($ollamaCommand) { $OllamaExecutable = $ollamaCommand.Source }
}
if (-not $OllamaExecutable -or -not (Test-Path -LiteralPath $OllamaExecutable -PathType Leaf)) {
    throw 'Ollama was not found. Pass -OllamaExecutable with an absolute ollama.exe path.'
}
$workloads = [ordered]@{
    'windows-ollama-generation' = [ordered]@{
        kind = 'ollama_host'
        label = 'Workstation Ollama'
        executable = $OllamaExecutable
    }
}

$timestamp = Get-Date -Format 'yyyyMMddTHHmmss'
$backupPath = "$ConfigPath.$timestamp.bak"
Copy-Item -LiteralPath $ConfigPath -Destination $backupPath
$config | Add-Member -NotePropertyName external_workloads -NotePropertyValue $workloads -Force
$temporaryPath = "$ConfigPath.tmp"
[IO.File]::WriteAllText(
    $temporaryPath,
    (($config | ConvertTo-Json -Depth 10) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)
Move-Item -LiteralPath $temporaryPath -Destination $ConfigPath -Force
Write-Host "Configured the workstation Ollama allowlist; backup: $backupPath"
