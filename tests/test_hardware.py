"""The hardware record. NO GPU HAS BEEN INVOLVED in any of these tests: the nvidia-smi text below is written from the
tool's documented formats, and the Docker call is a recording fake. They show that ReproFix asks the right question in the
right place and reads the answer carefully; they do not show what a real H100/L40S host prints or that --gpus all works."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import BUG_LINE, FIX_LINE, hypothesis_reply, repair_reply, scripted
from reprofix.config import Settings
from reprofix.core.orchestrator import Orchestrator
from reprofix.models import MetricSpec, RunRequest
from reprofix.sandbox import hardware as hw
from reprofix.sandbox.base import Sandbox, SandboxPolicy
from reprofix.sandbox.docker_backend import DockerSandbox
from reprofix.sandbox.local_backend import LocalSandbox

H100 = "0, NVIDIA H100 80GB HBM3, 81559, 550.54.15\n"
TWO = "0, NVIDIA L40S, 46068, 535.161.08\n1, NVIDIA L40S, 46068, 535.161.08\n"
HEADER = ("+-----------------------------------------------------------------------------------------+\n"
          "| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |\n"
          "+-----------------------------------------+------------------------+----------------------+\n")


def test_csv_is_parsed_and_junk_is_skipped_not_guessed():
    assert hw.parse_gpu_csv(H100) == [{"index": 0, "name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "driver": "550.54.15"}]
    assert [g["index"] for g in hw.parse_gpu_csv(TWO)] == [0, 1]
    assert hw.parse_gpu_csv("0, NVIDIA A10, [N/A], 535.1\n")[0]["memory_mib"] is None
    for junk in ["", "No devices were found", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
                 "0, only, three\n", "x, name, 1, 2.0\n", "0, , 1, 2.0\n"]:
        assert hw.parse_gpu_csv(junk) == [], junk
    assert hw.parse_gpu_csv("0, A, 10, not-a-version\n")[0]["driver"] is None


def test_cuda_version_comes_from_the_header_line():
    assert hw.parse_cuda_version(HEADER) == "12.4" and hw.parse_cuda_version("nothing") is None and hw.parse_cuda_version("") is None


def test_the_summary_names_the_gpu_count_memory_driver_and_cuda():
    rec = hw.from_smi(requested=True, probed_in="x", query_stdout=TWO, query_code=0, header_stdout=HEADER)
    assert rec["status"] == "detected" and rec["summary"] == "2× NVIDIA L40S, 46068 MiB each, driver 535.161.08, CUDA 12.4 (driver's maximum)"
    assert rec["raw"].startswith("0, NVIDIA L40S") and "does not show that the repository's code ran on it" in rec["note"]


def test_failures_say_what_happened():
    none = hw.from_smi(requested=True, probed_in="x", query_stdout="", query_code=0)
    assert none["status"] == "none_visible" and none["summary"] == "no GPU visible"
    err = hw.from_smi(requested=True, probed_in="x", query_stdout="", query_code=125,
                      query_stderr='docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].')
    assert err["status"] == "probe_failed" and "could not select device driver" in err["detail"] and err["gpus"] == []
    assert hw.record(requested=False, status="not_requested", probed_in=None)["summary"] == "CPU only (GPU not enabled)"


# --------------------------------------------------------------------------- Docker: the same flags as the run phase
class FakeDockerRun:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, cmd, **kw):
        if cmd[0] != "docker":                          # the chmod of the workspace that DockerSandbox.prepare() does as root
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        self.calls.append(cmd)
        code, out, err = self.replies.pop(0)
        return subprocess.CompletedProcess(cmd, code, out.encode(), err.encode())


def docker(gpu: bool) -> DockerSandbox:
    return DockerSandbox(Settings(sandbox="docker", sandbox_gpu=gpu), check=False)


def test_docker_probe_runs_nvidia_smi_in_a_container_with_the_run_phase_flags(monkeypatch, tmp_path):
    fake = FakeDockerRun([(0, H100, ""), (0, HEADER, "")])
    monkeypatch.setattr(subprocess, "run", fake)
    rec = docker(True).probe_hardware(tmp_path)
    q, h = (c for c in fake.calls if c[:2] == ["docker", "run"])
    for cmd in (q, h):
        assert cmd[cmd.index("--network") + 1] == "none" and cmd[cmd.index("--gpus") + 1] == "all"
        assert "--read-only" in cmd and cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert q[-3:] == ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    assert h[-1] == "nvidia-smi"
    assert rec["status"] == "detected" and rec["requested"] and rec["probed_in"].startswith("inside the sandbox container")
    assert rec["gpus"][0]["name"] == "NVIDIA H100 80GB HBM3" and rec["cuda_driver"] == "12.4"


def test_docker_without_the_gpu_flag_starts_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no container may be started")))
    rec = docker(False).probe_hardware(tmp_path)
    assert rec["status"] == "not_requested" and rec["summary"] == "CPU only (GPU not enabled)" and "REPROFIX_SANDBOX_GPU" in rec["detail"]


def test_docker_reports_a_missing_container_toolkit(monkeypatch, tmp_path):
    fake = FakeDockerRun([(125, "", 'docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].')])
    monkeypatch.setattr(subprocess, "run", fake)
    rec = docker(True).probe_hardware(tmp_path)
    assert rec["status"] == "probe_failed" and "could not select device driver" in rec["detail"] and len(fake.calls) == 1   # the failed query; no header call after it


def test_a_crashing_probe_never_raises(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("docker vanished")
    monkeypatch.setattr(subprocess, "run", boom)
    rec = docker(True).probe_hardware(tmp_path)
    assert rec["status"] == "probe_failed" and "docker vanished" in rec["detail"]


# --------------------------------------------------------------------------- local development backend
def fake_smi(tmp_path: Path, monkeypatch, *, query: str, code: int = 0) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        py_script = bin_dir / "_nvidia_smi.py"
        py_script.write_text(f"""import sys
header = {repr(HEADER)}
query = {repr(query)}
if len(sys.argv) <= 1:
    sys.stdout.write(header)
    sys.exit(0)
else:
    sys.stdout.write(query)
    sys.exit({code})
""")
        cmd_file = bin_dir / "nvidia-smi.cmd"
        cmd_file.write_text(f'@"{sys.executable}" "{py_script}" %*\n')
    else:
        exe = bin_dir / "nvidia-smi"
        exe.write_text(f"#!/bin/sh\nif [ \"$1\" = \"\" ]; then cat <<'EOF'\n{HEADER}EOF\nelse cat <<'EOF'\n{query}EOF\nexit {code}\nfi\n")
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_local_backend_reports_the_hosts_gpu_when_nvidia_smi_exists(settings, tmp_path, monkeypatch):
    fake_smi(tmp_path, monkeypatch, query=H100)
    rec = LocalSandbox(settings).probe_hardware(tmp_path)
    assert rec["status"] == "detected" and rec["gpus"][0]["memory_mib"] == 81559 and rec["cuda_driver"] == "12.4"
    assert "no device isolation" in rec["probed_in"]


def test_local_backend_without_nvidia_smi_says_cpu_only(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    rec = LocalSandbox(settings, python="/usr/bin/python3").probe_hardware(tmp_path)
    assert rec["status"] == "none_visible" and "not installed" in rec["detail"]


# --------------------------------------------------------------------------- in the run and the report
def run_it(settings, tiny_repo, tmp_path, sandbox, **req):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    request = RunRequest(local_path=str(tiny_repo), command="python train.py", metric=MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02), max_attempts=3, **req)
    events: list = []
    orch = Orchestrator(settings=settings, request=request, run_dir=tmp_path / "run", sandbox=sandbox, backend=b, emit=lambda k, d: events.append((k, d)))
    return orch.run(), events


class GpuSandbox(LocalSandbox):
    def __init__(self, settings, rec):
        super().__init__(settings)
        self.rec = rec

    def probe_hardware(self, workdir):
        return self.rec


def test_the_report_carries_the_hardware_record_and_the_claims_card_says_where_it_was_measured(settings, tiny_repo, tmp_path):
    rec = hw.from_smi(requested=True, probed_in="inside the sandbox container (docker run --gpus all, no network)", query_stdout=H100, query_code=0, header_stdout=HEADER)
    rep, events = run_it(settings, tiny_repo, tmp_path, GpuSandbox(settings, rec), paper_text="Our model reaches a validation accuracy of 90.0%.")
    assert rep["hardware"] == rec and rep["claims"]["measured_on"] == rec["summary"]
    assert any(k == "hardware" and d["status"] == "detected" for k, d in events)
    assert not any("GPU was requested" in c for c in rep["caveats"])


def test_a_requested_gpu_that_was_not_found_is_a_caveat(settings, tiny_repo, tmp_path):
    rec = hw.from_smi(requested=True, probed_in="x", query_stdout="", query_code=125, query_stderr="could not select device driver")
    rep, _ = run_it(settings, tiny_repo, tmp_path, GpuSandbox(settings, rec))
    assert any("GPU was requested" in c and "could not select device driver" in c for c in rep["caveats"])


def test_a_probe_that_raises_does_not_stop_the_run(settings, tiny_repo, tmp_path):
    class Boom(LocalSandbox):
        def probe_hardware(self, workdir):
            raise RuntimeError("probe exploded")
    rep, _ = run_it(settings, tiny_repo, tmp_path, Boom(settings))
    assert rep["status"] == "verified" and rep["hardware"]["status"] == "probe_failed" and "probe exploded" in rep["hardware"]["detail"]


def test_a_cpu_only_run_says_so(settings, tiny_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")        # whatever this machine has, a fake-free PATH finds no nvidia-smi in CI containers
    rep, _ = run_it(settings, tiny_repo, tmp_path, LocalSandbox(settings))
    assert rep["hardware"]["status"] in ("none_visible", "detected") and rep["hardware"]["note"]


def test_mixed_gpus_are_not_described_as_if_they_were_all_the_first_one():
    mixed = "0, NVIDIA A100-SXM4-40GB, 40960, 535.1\n1, Tesla T4, 15360, 535.1\n"
    rec = hw.from_smi(requested=True, probed_in="x", query_stdout=mixed, query_code=0, header_stdout=HEADER)
    assert "40960 MiB each" not in rec["summary"] and "15360 MiB each" not in rec["summary"]
    assert "NVIDIA A100-SXM4-40GB" in rec["summary"] and "Tesla T4" in rec["summary"] and "40960/15360 MiB" in rec["summary"]
    differ = "0, A, 100, 1.0\n1, A, 100, 2.0\n"
    assert "driver" not in hw.from_smi(requested=True, probed_in="x", query_stdout=differ, query_code=0)["summary"]


def test_a_gpu_without_a_real_name_is_not_a_detected_gpu_and_unreadable_output_is_a_failed_probe_not_an_empty_one():
    for name in ("[N/A]", "N/A", "Unknown", "[Not Supported]"):
        assert hw.parse_gpu_csv(f"0, {name}, 1000, 535.1\n") == [], name
    assert hw.parse_gpu_csv("٠, NVIDIA A10, 1000, 535.1\n") == []                       # a non-ASCII digit is not an index
    odd = hw.from_smi(requested=True, probed_in="x", query_stdout="GPU 0: something new ReproFix has never seen\n", query_code=0)
    assert odd["status"] == "probe_failed" and "could not be read" in odd["detail"] and "something new" in odd["raw"]
    assert hw.from_smi(requested=True, probed_in="x", query_stdout="", query_code=0)["status"] == "none_visible"
