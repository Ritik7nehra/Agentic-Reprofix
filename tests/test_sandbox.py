import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from reprofix.config import Settings
from reprofix.sandbox import make_sandbox
from reprofix.sandbox.base import SandboxUnavailable, pip_install_argv, rewrite_argv
from reprofix.sandbox.docker_backend import DockerSandbox, build_docker_argv
from reprofix.sandbox.local_backend import LocalSandbox


# --------------------------------------------------------------------------- argv helpers
def test_rewrite_argv_drops_site_packages_by_default():
    assert rewrite_argv(["python", "train.py", "--x"], "/py") == ["/py", "-S", "train.py", "--x"]
    assert rewrite_argv(["python3", "-m", "pkg"], "/py") == ["/py", "-S", "-m", "pkg"]
    assert rewrite_argv(["pytest", "-q"], "/py") == ["/py", "-S", "-m", "pytest", "-q"]
    assert rewrite_argv(["python", "a.py"], "/py", isolated_site=False) == ["/py", "a.py"]
    with pytest.raises(ValueError):
        rewrite_argv(["bash", "x"], "/py")


def test_pip_install_is_wheels_only_unless_sdists_are_allowed():
    safe = pip_install_argv("python", ["requirements.txt", "dev/requirements.txt"], allow_sdist=False)
    assert "--only-binary=:all:" in safe and "--target" in safe and safe[safe.index("--target") + 1] == ".deps"
    assert safe.count("-r") == 2
    assert "--only-binary=:all:" not in pip_install_argv("python", ["r.txt"], allow_sdist=True)


# --------------------------------------------------------------------------- docker command line (no daemon needed)
def docker_cmd(**over):
    base = dict(image="reprofix-sandbox:latest", name="n1", workdir=Path("/runs/a/work"), argv=["python", "train.py"],
                network=False, memory_mb=4096, cpus=2.0, pids=512, gpu=False, user="1000:1000", env={"B": "2", "A": "1"})
    base.update(over)
    return build_docker_argv(**base)


def pairs(cmd):
    return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}


def test_docker_run_phase_is_locked_down():
    cmd = docker_cmd()
    p = pairs(cmd)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert p["--network"] == "none" and p["--cap-drop"] == "ALL" and p["--security-opt"] == "no-new-privileges"
    assert "--read-only" in cmd and p["--user"] == "1000:1000" and p["--pids-limit"] == "512"
    assert p["--memory"] == "4096m" and p["--memory-swap"] == "4096m" and p["--cpus"] == "2.0"
    assert "--privileged" not in cmd and "--gpus" not in cmd and "--network=host" not in cmd
    assert cmd[cmd.index("-v") + 1] == "/runs/a/work:/work:rw" and cmd.count("-v") == 1      # only the workspace is mounted
    assert cmd[-3:] == ["reprofix-sandbox:latest", "python", "train.py"]                      # image, then the command


def test_docker_install_phase_only_differs_by_network():
    assert pairs(docker_cmd(network=True))["--network"] == "bridge"
    hardened = {k: v for k, v in pairs(docker_cmd(network=True)).items() if k != "--network"}
    assert hardened == {k: v for k, v in pairs(docker_cmd()).items() if k != "--network"}


def test_docker_gpu_is_only_ever_attached_to_the_no_network_phase():
    assert pairs(docker_cmd(gpu=True))["--gpus"] == "all"
    assert "--gpus" not in docker_cmd(gpu=True, network=True)


def test_docker_env_is_exactly_what_was_given_and_never_inherits_secrets(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "sekret")
    cmd = docker_cmd()
    assert [cmd[i + 1] for i, c in enumerate(cmd) if c == "-e"] == ["A=1", "B=2"]
    assert "sekret" not in " ".join(cmd)
    env = DockerSandbox(Settings(), check=False)._env()
    assert not any("KEY" in k or "TOKEN" in k for k in env)


def test_docker_policy_and_unavailable_daemon(monkeypatch):
    pol = DockerSandbox(Settings(), check=False).policy()
    assert pol.isolated and not pol.network_run and pol.network_install
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("docker")))
    with pytest.raises(SandboxUnavailable, match="docker CLI not usable"):
        DockerSandbox(Settings())


def test_make_sandbox_refuses_unsafe_local_unless_opted_in(tmp_path):
    with pytest.raises(SandboxUnavailable, match="REPROFIX_ALLOW_UNSAFE_LOCAL"):
        make_sandbox(Settings(sandbox="local", data_dir=tmp_path))
    assert make_sandbox(Settings(sandbox="local", allow_unsafe_local=True, data_dir=tmp_path)).policy().isolated is False


# --------------------------------------------------------------------------- local backend (really executes code)
@pytest.fixture
def sbx(settings):
    return LocalSandbox(settings)


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


def run(sbx, ws, code, timeout=20):
    (ws / "t.py").write_text(code)
    return sbx.run(ws, ["python", "t.py"], timeout)


def test_runs_code_and_captures_streams_and_exit_code(sbx, ws):
    r = run(sbx, ws, "import sys\nprint('out')\nprint('err', file=sys.stderr)\nsys.exit(3)\n")
    assert (r.exit_code, r.stdout.strip(), r.stderr.strip(), r.timed_out, r.network) == (3, "out", "err", False, False)


def test_api_keys_never_reach_repository_code(sbx, ws, monkeypatch):
    for k in ("NEBIUS_API_KEY", "TAVILY_API_KEY", "REPROFIX_API_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "leaked-" + k)
    r = run(sbx, ws, "import os\nprint(sorted(os.environ))\nprint(any('leaked-' in v for v in os.environ.values()))\n")
    assert r.stdout.strip().endswith("False") and "NEBIUS" not in r.stdout and "TAVILY" not in r.stdout


@pytest.mark.skipif(os.name == "nt", reason="pgrep and POSIX process-group kill are POSIX only")
def test_timeout_kills_the_process_tree(sbx, ws):
    r = run(sbx, ws, "import subprocess, time\nsubprocess.Popen(['sleep', '61.5'])\ntime.sleep(60)\n", timeout=2)
    assert r.timed_out and r.exit_code is None and r.duration_s < 15
    for _ in range(30):                                                 # the group kill needs a moment to reap
        left = subprocess.run(["pgrep", "-f", "sleep 61.5"], capture_output=True, text=True).stdout.split()
        if not left:
            break
        time.sleep(0.2)
    assert left == []                                                   # the grandchild died with the process group


def test_output_is_capped_but_keeps_the_end_of_the_log(sbx, ws):
    r = run(sbx, ws, "print('x' * 5_000_000)\nprint('TRACEBACK-AT-THE-END')\n")
    assert r.truncated and len(r.stdout) < 100_000 and "TRACEBACK-AT-THE-END" in r.stdout


@pytest.mark.skipif(not LocalSandbox(Settings(allow_unsafe_local=True, data_dir=Path("/tmp/rf-probe"))).netns,
                    reason="network namespaces are not available in this environment")
def test_run_phase_has_no_network(sbx, ws):
    r = run(sbx, ws, "import socket\ns = socket.socket()\ns.settimeout(3)\n"
                     "try:\n    s.connect(('1.1.1.1', 80)); print('CONNECTED')\nexcept OSError as e:\n    print('blocked', type(e).__name__)\n")
    assert r.exit_code == 0 and "blocked" in r.stdout and "CONNECTED" not in r.stdout


def test_host_site_packages_are_hidden_so_missing_requirements_fail_like_a_clean_machine(sbx, ws):
    # pytest is installed on this machine (it runs this test) but is not a repo requirement
    r = run(sbx, ws, "import pytest\n")
    assert r.exit_code != 0 and "ModuleNotFoundError" in r.stderr


def test_deps_directory_is_importable_even_with_a_relative_workdir(sbx, tmp_path, monkeypatch):
    # regression: a relative run dir produced a relative PYTHONPATH that broke once the child's cwd was the workdir
    ws = tmp_path / "rel" / "work"
    (ws / ".deps").mkdir(parents=True)
    (ws / ".deps" / "fakedep.py").write_text("VALUE = 42\n")
    (ws / "t.py").write_text("import fakedep\nprint(fakedep.VALUE)\n")
    monkeypatch.chdir(tmp_path)
    r = sbx.run(Path("rel/work"), ["python", "t.py"], 20)
    assert r.exit_code == 0 and r.stdout.strip() == "42", r.stderr


def test_pytest_is_launched_as_python_dash_S_dash_m_pytest_from_deps(sbx, ws):
    # A stub `pytest` package under .deps stands in for a real install: this checks the launch path
    # (sandbox interpreter, -S, -m pytest, cwd=workdir, .deps importable), not pytest itself.
    pkg = ws / ".deps" / "pytest"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text("import os, sys\nprint('stub-pytest', sys.argv[1:], 'S' if sys.flags.no_site else 'site', os.getcwd())\n")
    r = sbx.run(ws, ["pytest", "-q", "-p", "no:cacheprovider"], 60)
    assert r.exit_code == 0, r.stderr
    assert "['-q', '-p', 'no:cacheprovider']" in r.stdout and " S " in r.stdout and str(ws.resolve()) in r.stdout


def test_without_pytest_in_deps_the_sandbox_reports_it_clearly(sbx, ws):
    (ws / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    r = sbx.run(ws, ["pytest", "-q"], 60)
    assert r.exit_code != 0 and "No module named pytest" in r.stderr
