# GPU Workload Scheduler repository rules

- This is a Windows-native workstation service under `C:\Dev\Repos\gpu-workload-scheduler`.
- Bind the API and dashboard to `127.0.0.1` only.
- Keep PostgreSQL in the repository Compose file and its writable data in the explicit named volume `gpu-workload-scheduler_postgres-data`.
- Treat the Docker volume as opaque. Logical dumps belong under `E:\Data\DB\Dumps\gpu-workload-scheduler`.
- Runtime logs belong under `E:\Data\GpuScheduler\Logs`; generated local credentials belong under `.runtime` and must stay ignored.
- Never log commands' environment variables or secrets.
- The daemon may launch only argv arrays without a shell. Do not add `shell=True`.
- Do not terminate arbitrary unmanaged GPU processes. They affect admission
  control only. Explicitly configured `external_workloads` may expose
  application-native graceful model unload only when the server selects the
  exact allowlisted container and discovers the loaded model itself; API
  clients must never choose a PID, command, container, or model name.
- Update README operational instructions when ports, paths, schema, scheduling policy, or service behavior change.
