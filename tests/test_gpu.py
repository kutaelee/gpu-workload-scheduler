from types import SimpleNamespace

from gpuq.gpu import read_gpu_processes


def test_process_census_preserves_unknown_wddm_memory(monkeypatch):
    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout="""25360, C:\\Python\\python.exe, [N/A]\n4242, worker.exe, 512\ninvalid\n"""
        )

    monkeypatch.setattr("gpuq.gpu.subprocess.run", fake_run)

    processes = read_gpu_processes()

    assert [process.pid for process in processes] == [25360, 4242]
    assert processes[0].used_mb is None
    assert processes[1].used_mb == 512
    assert "command" not in processes[0].to_dict()
