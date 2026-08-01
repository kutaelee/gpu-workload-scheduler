# GPU Workload Scheduler

Single-GPU reservation queue for the RTX 5090 workstation. Agents submit a
command with required VRAM, expected duration, and priority. The host daemon
starts only jobs that fit the guarded VRAM budget, ages waiting jobs to prevent
starvation, and penalizes recent per-agent VRAM-seconds. The incident-diagnostic
default is strict serialization (`max_parallel_jobs=1`); parallel backfill is
not enabled by the installer.

The Local Knowledge Portal can request a managed job stop and reorder queued
jobs. A stop is cooperative first (Ctrl+Break). Native jobs then use bounded
terminate/kill escalation. WSL jobs are not force-terminated unless an
operator configured an exact Linux cleanup command or explicitly enabled
`wsl_force_terminate`; killing only `wsl.exe` can orphan the real Linux/CUDA
worker. A drag-and-drop queue order is explicit but does not
override the VRAM safety reserve, parallelism limit, or safe short-job
backfilling when the selected head cannot fit. Reorder requests must contain
the complete current queue, so a stale browser view is rejected instead of
reordering the wrong work.

Detached helpers such as Docker-hosted Ollama use the operator-owned
`cleanup_commands` mapping in `.runtime/config.json`. Each workload key maps to
one exact argv allowlist entry. API clients cannot provide cleanup commands;
the scheduler runs the configured argv without a shell after cancel or timeout.

After a job reaches a terminal state, the scheduler stops all NVIDIA queries
for 15 seconds. A measured peak of at least 24 GiB extends only this no-touch
window to 20 seconds; runtime duration and temperature do not classify a job
as high load. After the no-touch window, admission requires three idle
telemetry samples and two unchanged process censuses at five-second intervals.
Healthy handoff therefore normally completes in about 25-30 seconds instead of
waiting a fixed 180 seconds. Any failed query, renewed utilization, excess
resident VRAM, or changed process census resets the evidence. Failed and
canceled jobs still run their allowlisted cleanup once.

Workloads listed in `reboot_boundary_workloads` use a stricter diagnostic
boundary. When one reaches any terminal state, GPUQ persists the job and boot
epoch in `.runtime/gpu-workload-reboot-boundary.json`, stops all further
NVIDIA queries, and blocks admission for the rest of that Windows boot. The
latch clears only by observing a later boot; daemon restart cannot bypass it.
The boundary is first persisted as `armed` immediately before process launch;
a daemon that restarts while the process is running treats that state as an
active same-boot boundary. Process exit is checked before every NVIDIA query,
so the terminal transition starts without one final telemetry touch.
GPUQ deliberately does not run `wsl --terminate` automatically because a
shared distro can contain unrelated services. Use this allowlist only while a
driver/GPU-PV handoff fault is under diagnosis, or for workloads that require
an explicit reboot boundary by policy.

`reboot_boundary_workload_patterns` provides the same boundary for an explicit
set of lowercase shell-style workload-key patterns (`*` and `?`). This prevents
versioned diagnostic jobs from silently escaping the boundary when only their
run number or subject suffix changes. Patterns apply to the declared GPUQ
workload key, never to argv or environment data. Keep them narrow; for example,
`wedding-*klein9b*qfloat8*` covers the currently isolated Klein 9B qfloat8
training family without blocking unrelated ComfyUI prompts.

`reboot_boundary_exempt_workloads` is an exact-key diagnostic exception list.
It overrides a matching boundary pattern only for explicitly reviewed workload
keys whose trigger path has been removed and whose submission wrapper validates
the executable configuration and source hashes. Do not use wildcard exemptions;
keep the legacy failing workload keys protected while same-boot load/unload/reload
validation is in progress.

GPU telemetry is an admission circuit breaker. Transient query failures back
off 5, 15, then 60 seconds while process lifecycle reaping continues without
touching NVIDIA. Exit code 6, `GPU is lost`, or `reboot required` is different:
it atomically latches `reboot_required` in `.runtime/gpu-fault-state.json` and
the current boot performs no further `nvidia-smi` calls, even after daemon
restart. A later boot remains `rearm_required` until an authenticated manual
rearm confirms a changed boot epoch, no current-boot nvlddmkm 14/153 events,
and one successful GPU query. State changes are appended without commands or
environment data to `E:\Data\GpuScheduler\Logs\gpu-health.jsonl`.
An NVIDIA query timeout while a configured reboot-boundary workload is still
running is also fatal: GPUQ does not perform a second query in that boot.

`GET /livez` reports only API-process liveness. `GET /readyz` and the legacy
`GET /api/health` report admission readiness and return HTTP 503 while the GPU
is blocked, in no-touch transition, or awaiting manual rearm. Runtime status
also records the Git commit/branch, dirty-worktree flag, config SHA-256, service
start time, and Windows boot epoch.

For post-incident analysis, a 10-second rolling sample of temperature, power
draw/limit, graphics and memory clocks, PCIe generation/width, VRAM,
utilization, and driver version is appended to daily
`gpu-telemetry-YYYYMMDD.jsonl` files in the same log directory. These records
contain no process command lines or environment values.

ComfyUI's standard `E:\AI\Apps\ComfyUI\run-comfyui.ps1` first requires GPUQ
admission readiness and refuses startup while another managed job is active.
Its local GPUQ bridge admits each `/prompt` and
`/api/prompt` request as an individual `comfyui-prompt` queue job. The retired
`run-comfyui-gpuq.ps1` server-wide reservation must not be restored: it hides
individual prompts and lets an idle UI block unrelated GPU work.

Long-lived Ollama runtimes can be registered in the `external_workloads`
allowlist. The workstation uses one Windows Ollama daemon for Hermes, portal
embedding, and portal knowledge editing. Its loaded-model state is sampled at
the same bounded interval as the process census and exposed in `/api/status`.
An authenticated `POST /api/external-workloads/<key>/stop` unloads only model
names returned by `ollama ps` from the exact configured executable. Clients
cannot provide a PID, executable, model, or command. Apply the workstation
default with:

```powershell
.\scripts\Configure-ExternalWorkloads.ps1
```

The scheduler also exposes a bounded NVIDIA process census for attribution.
Processes which are not launched as scheduler children are shown as **observed
external GPU use**, never converted into a fake queue job and never terminated
by a queue stop request. On Windows WDDM the driver can report per-process
memory as unavailable; the dashboard therefore shows the GPU-wide usage and
process identity separately rather than assigning an invented VRAM value. The
process census is sampled every 15 seconds by default and is not on the
scheduling hot path.

## Runtime layout

- Source: `C:\Dev\Repos\gpu-workload-scheduler`
- Host API/dashboard: `http://127.0.0.1:8790`
- PostgreSQL: Docker Desktop, `127.0.0.1:55670` by default (`GPUQ_POSTGRES_PORT` is configurable)
- Docker volume: `gpu-workload-scheduler_postgres-data`
- Job logs: `E:\Data\GpuScheduler\Logs`
- Logical dumps: `E:\Data\DB\Dumps\gpu-workload-scheduler`
- Local secrets: `.env` and `.runtime\config.json` (ignored)

The scheduler runs on Windows because it must launch both Windows-native and
WSL commands. PostgreSQL is durable queue state; no Redis, Kafka, or second
Docker Engine is required.

## Install and operate

```powershell
.\scripts\Install.ps1
.\scripts\Register-Task.ps1
Invoke-RestMethod http://127.0.0.1:8790/api/health
```

The scheduled task is `\Codex\GPU Workload Scheduler`, runs only in the signed-in
user session, and does not open a visible window. At logon, its supervisor waits
up to 15 minutes for Docker Desktop, starts the queue PostgreSQL service with
Compose's health gate, and verifies the Windows loopback publication at
the configured loopback PostgreSQL port (default `127.0.0.1:55670`) before
starting the host API. If Docker reports PostgreSQL
healthy but the host publication is missing after a reboot, the supervisor
recreates only the PostgreSQL *container* and retains the named data volume.
The supervisor logs API stderr, uses bounded retry backoff, and restarts an API
that exits unexpectedly; it does not leave the scheduler down after Task
Scheduler's finite retry budget is exhausted.

If a Windows reserved port prevents Docker from publishing PostgreSQL after a
driver, Hyper-V, or Docker Desktop change, select an unreserved loopback port
and apply it consistently before restarting the task:

```powershell
.\scripts\Set-DatabasePort.ps1 -Port 55670
```

```powershell
.\scripts\Stop-Daemon.ps1
Start-ScheduledTask -TaskPath '\Codex\' -TaskName 'GPU Workload Scheduler'
docker compose stop postgres
```

The graceful stop endpoint refuses shutdown while managed work is running. On
restart, database rows unexpectedly left as `running` are marked `orphaned`;
unknown OS processes are not killed.

## Agent CLI

`gpuq` prints the queued job UUID. Commands are argument arrays and are launched
without a shell.

```powershell
gpuq run --vram 12288 --eta 600 --priority 60 --workload qwen-edit -- python generate.py
gpuq status
gpuq wait <job-id>
gpuq cancel <job-id>
gpuq estimate qwen-edit
```

Use priority 80–100 for an explicitly urgent interactive task, 40–70 for normal
work, and 0–30 for opportunistic batch work. Include stable workload keys so
the service can learn p50/p90 duration estimates. The default hard timeout is
the larger of three times the estimate and ten minutes.

The dashboard is read-only. Mutation APIs require the token from
`.runtime\config.json`. CORS is restricted to the Local Knowledge Portal at
`localhost:3010`; secrets are never exposed to that UI.

## Test

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pytest
```

Integration verification should submit two bounded test jobs with known VRAM
and duration, observe serialized execution, then inspect the exit codes and
per-job logs. An active transition gate is persisted, so restarting the daemon
cannot bypass either the no-touch deadline or the required health samples.

## Backup and restore

Create an append-only logical dump:

```powershell
.\scripts\Backup-Database.ps1
.\scripts\Test-Restore.ps1 -DumpPath E:\Data\DB\Dumps\gpu-workload-scheduler\<dump>.dump
```

The test script restores into a uniquely named temporary database, queries the
jobs table, and removes only that test database. For a manual restore, create a
new target database first:

```powershell
docker compose exec -T postgres createdb -U gpuq gpuq_test
docker cp E:\Data\DB\Dumps\gpu-workload-scheduler\<dump>.dump gpu-workload-scheduler-postgres-1:/tmp/restore.dump
docker compose exec -T postgres pg_restore -U gpuq -d gpuq_test /tmp/restore.dump
```

Never copy the live Docker volume or VHDX. Queue history can be rebuilt or
archived, but a restore must not overwrite the active database as a test.
