[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $RepoRoot '.runtime\config.json'
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw 'GPUQ runtime config is missing; run scripts/Install.ps1 first.'
}

$config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
$settings = [ordered]@{
    max_parallel_jobs = 1
    gpu_telemetry_enabled = $true
    probe_gpu_on_startup = $false
    active_gpu_probe_interval_seconds = 10.0
    process_scan_interval_seconds = 30.0
    post_job_no_touch_seconds = 15.0
    post_high_load_no_touch_seconds = 20.0
    post_high_load_probe_interval_seconds = 5.0
    post_high_load_stable_samples = 3
    post_high_load_process_stable_scans = 2
    post_high_load_vram_tolerance_mb = 4096
    post_high_load_max_idle_utilization_percent = 5
    high_load_min_peak_used_mb = 24576
}
foreach ($entry in $settings.GetEnumerator()) {
    $config | Add-Member -MemberType NoteProperty -Name $entry.Key -Value $entry.Value -Force
}
foreach ($deprecated in @('post_high_load_cooldown_seconds', 'high_load_min_runtime_seconds')) {
    $config.PSObject.Properties.Remove($deprecated)
}

$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText(
    $ConfigPath,
    (($config | ConvertTo-Json -Depth 20) + "`n"),
    $utf8
)
Write-Host 'Configured GPUQ incident safety mode (serial admission and state-based no-touch transitions).'
