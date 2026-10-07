"""Docker command lines and fail-closed behaviour of the install-phase egress setup. No daemon needed: the `docker` CLI is
replaced by a recording fake, so these tests check what ReproFix *asks* Docker to do, not what Docker then does."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reprofix.config import Settings
from reprofix.sandbox import egress_ctl as E
from reprofix.sandbox.base import SandboxUnavailable
from reprofix.sandbox.docker_backend import DockerSandbox, build_docker_argv


class FakeDocker:
    """Answers docker CLI calls from a tiny model of the world and records every call."""

    def __init__(self, *, network: str | None = None, container: str | None = None, networks=("bridge",), allow="pypi.org,files.pythonhosted.org",
                 selftest_out="PASS a\nPASS b\nPASS c\n", selftest_code=0, health_ok=True, fail: dict | None = None):
        self.network = network                    # None | "internal" | "open"
        self.container = container                # None | "running" | "stopped"
        self.networks = list(networks)
        self.allow = allow
        self.selftest_out, self.selftest_code, self.health_ok = selftest_out, selftest_code, health_ok
        self.fail = fail or {}
        self.calls: list[list[str]] = []

    def __call__(self, argv, timeout=60):
        self.calls.append(list(argv))
        cp = lambda code=0, out="", err="": subprocess.CompletedProcess(argv, code, out, err)  # noqa: E731
        key = " ".join(argv[1:3])
        if key in self.fail:
            return cp(1, "", self.fail[key])
        if argv[1:3] == ["network", "inspect"]:
            return cp(0, "true\n" if self.network == "internal" else "false\n") if self.network else cp(1, "", "no such network")
        if argv[1:3] == ["network", "create"]:
            self.network = "internal" if "--internal" in argv else "open"
            return cp()
        if argv[1:3] == ["network", "connect"]:
            self.networks.append(argv[3])
            return cp()
        if argv[1:3] == ["network", "rm"]:
            self.network = None
            return cp()
        if argv[1] == "inspect":
            if not self.container:
                return cp(1, "", "no such container")
            return cp(0, f"{'true' if self.container == 'running' else 'false'}|{self.allow}|{' '.join(self.networks)} \n")
        if argv[1] == "rm":
            self.container = None
            return cp()
        if argv[1:2] == ["exec"]:
            return cp(0 if self.health_ok else 1)
        if argv[1:2] == ["run"] and "-d" in argv:
            self.container, self.networks = "running", ["bridge"]
            self.allow = next(a.split("=", 1)[1] for a in argv if a.startswith("reprofix.egress.allow="))
            return cp(0, "cid\n")
        if argv[1:2] == ["run"] and "--selftest" in argv:
            return cp(self.selftest_code, self.selftest_out)
        raise AssertionError(f"unexpected docker call: {argv}")


S = Settings(install_network="proxy")


# --------------------------------------------------------------------------- command lines
def test_the_network_is_created_internal():
    assert E.network_create_argv(S) == ["docker", "network", "create", "--internal", "reprofix-install"]


def test_the_proxy_container_is_hardened_and_starts_on_the_normal_bridge():
    cmd = E.proxy_run_argv(S)
    p = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
    assert cmd[:4] == ["docker", "run", "-d", "--name"] and p["--name"] == "reprofix-egress"
    assert p["--network"] == "bridge" and p["--cap-drop"] == "ALL" and p["--security-opt"] == "no-new-privileges"
    assert "--read-only" in cmd and p["--user"] == "1000:1000" and p["--memory"] == "256m"
    assert "--privileged" not in cmd and "-v" not in cmd and "--publish" not in cmd and "-p" not in cmd     # no mounts, no published ports
    assert cmd[-5:] == ["reprofix-sandbox:latest", "python", "/opt/reprofix/egress.py", "--listen", "0.0.0.0:3128"]
    assert "REPROFIX_EGRESS_ALLOW=pypi.org,files.pythonhosted.org" in cmd


def test_a_custom_allow_list_and_upstream_reach_the_container():
    cmd = E.proxy_run_argv(Settings(egress_allow=("pypi.org", "download.pytorch.org"), egress_upstream="http://corp:3128"))
    assert "REPROFIX_EGRESS_ALLOW=pypi.org,download.pytorch.org" in cmd and "REPROFIX_EGRESS_UPSTREAM=http://corp:3128" in cmd


def test_the_selftest_container_sits_where_an_install_container_sits():
    cmd = E.selftest_argv(S)
    assert cmd[cmd.index("--network") + 1] == "reprofix-install" and "--selftest" in cmd
    assert cmd[cmd.index("--selftest") + 1] == "reprofix-egress:3128" and cmd[cmd.index("--allowed-target") + 1] == "pypi.org:443"
    assert "-v" not in cmd and "--privileged" not in cmd


def test_proxy_env_points_https_at_the_proxy_and_never_http():
    env = E.proxy_env(S)
    assert env == {"HTTPS_PROXY": "http://reprofix-egress:3128", "https_proxy": "http://reprofix-egress:3128"}
    assert not any(k.upper() in {"HTTP_PROXY", "ALL_PROXY"} for k in env)


# --------------------------------------------------------------------------- the install container
def install_cmd(**over):
    base = dict(image="reprofix-sandbox:latest", name="n1", workdir=Path("/runs/a/work"), argv=["python", "-m", "pip", "install"],
                network=True, memory_mb=4096, cpus=2.0, pids=512, gpu=False, user="1000:1000", env={})
    base.update(over)
    return build_docker_argv(**base)


def test_install_container_joins_the_internal_network_only_when_asked_and_the_run_phase_never_does():
    cmd = install_cmd(network_name="reprofix-install")
    assert cmd[cmd.index("--network") + 1] == "reprofix-install"
    assert install_cmd()[install_cmd().index("--network") + 1] == "bridge"                         # unchanged default
    run = install_cmd(network=False, network_name="reprofix-install")
    assert run[run.index("--network") + 1] == "none"                                               # the run phase has no network, ever


def test_docker_sandbox_in_proxy_mode_installs_on_the_internal_network_with_the_proxy_env(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: seen.append(list(argv)) or subprocess.CompletedProcess(argv, 0, b"", b""))
    monkeypatch.setattr(E, "check_ready", lambda s, run=None: None)
    sb = DockerSandbox(S, check=False)
    sb.install(tmp_path, ["requirements.txt"], 30)
    cmd = next(c for c in seen if c[:2] == ["docker", "run"])
    assert cmd[cmd.index("--network") + 1] == "reprofix-install"
    envs = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-e"]
    assert "HTTPS_PROXY=http://reprofix-egress:3128" in envs and "https_proxy=http://reprofix-egress:3128" in envs
    assert not any(e.startswith(("NEBIUS", "TAVILY")) for e in envs)
    seen.clear()
    sb.run(tmp_path, ["python", "train.py"], 30)
    run = next(c for c in seen if c[:2] == ["docker", "run"])
    assert run[run.index("--network") + 1] == "none" and not any(c.startswith("HTTPS_PROXY") for c in run)


def test_open_mode_keeps_the_existing_behaviour(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: seen.append(list(argv)) or subprocess.CompletedProcess(argv, 0, b"", b""))
    DockerSandbox(Settings(), check=False).install(tmp_path, ["requirements.txt"], 30)
    cmd = next(c for c in seen if c[:2] == ["docker", "run"])
    assert cmd[cmd.index("--network") + 1] == "bridge" and not any("PROXY" in c.upper() for c in cmd)


def test_the_policy_says_which_install_network_is_in_force_and_which_hosts_it_allows():
    open_pol = DockerSandbox(Settings(), check=False).policy().to_dict()
    assert open_pol["install_egress"] == {"mode": "open", "hosts": []} and any("OPEN" in n for n in open_pol["notes"])
    pol = DockerSandbox(S, check=False).policy().to_dict()
    assert pol["install_egress"] == {"mode": "allow-list", "hosts": ["pypi.org", "files.pythonhosted.org"]}
    assert any("--internal" in n for n in pol["notes"]) and not any("OPEN" in n for n in pol["notes"])
    assert pol["network_during_run"] is False


def test_an_unknown_install_network_value_is_refused():
    with pytest.raises(SandboxUnavailable, match="must be 'open' or 'proxy'"):
        DockerSandbox(Settings(install_network="bridge"), check=False)


# --------------------------------------------------------------------------- fail closed
def test_check_ready_accepts_only_an_internal_network_and_a_running_attached_proxy():
    E.check_ready(S, FakeDocker(network="internal", container="running", networks=["bridge", "reprofix-install"]))
    with pytest.raises(E.EgressError, match="does not exist"):
        E.check_ready(S, FakeDocker(network=None))
    with pytest.raises(E.EgressError, match="NOT internal"):
        E.check_ready(S, FakeDocker(network="open", container="running", networks=["reprofix-install"]))
    with pytest.raises(E.EgressError, match="not running"):
        E.check_ready(S, FakeDocker(network="internal", container="stopped"))
    with pytest.raises(E.EgressError, match="not running"):
        E.check_ready(S, FakeDocker(network="internal", container=None))
    with pytest.raises(E.EgressError, match="not attached"):
        E.check_ready(S, FakeDocker(network="internal", container="running", networks=["bridge"]))


def test_a_replaced_non_internal_network_with_the_same_name_blocks_installs(monkeypatch, tmp_path):
    """The enforcement is the network being internal. If someone recreates it as a normal one, installs must stop."""
    monkeypatch.setattr(E, "_run", FakeDocker(network="open", container="running", networks=["reprofix-install"]))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no container may be started")))
    sb = DockerSandbox(S, check=False)
    with pytest.raises(SandboxUnavailable, match="refusing to install.*NOT internal"):
        sb.install(tmp_path, ["requirements.txt"], 30)


def test_constructing_the_sandbox_in_proxy_mode_reports_a_missing_proxy(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, "26.0", ""))      # docker info works
    monkeypatch.setattr(E, "_run", FakeDocker(network=None))
    with pytest.raises(SandboxUnavailable, match="not ready.*reprofix egress up"):
        DockerSandbox(S)


# --------------------------------------------------------------------------- up / down / status / test
def test_up_creates_everything_in_order_on_a_clean_host():
    d = FakeDocker()
    done = E.up(S, d)
    verbs = [" ".join(c[1:3]) for c in d.calls]
    assert verbs.index("network create") < verbs.index("run -d") < verbs.index("network connect")
    assert any("created internal network" in x for x in done) and any("attached" in x for x in done) and done[-1] == "proxy is answering"
    E.check_ready(S, d)                                           # and afterwards the world passes the readiness check


def test_check_ready_also_checks_the_running_proxys_allow_list():
    ready = dict(network="internal", container="running", networks=["bridge", "reprofix-install"])
    E.check_ready(S, FakeDocker(**ready, allow=" Files.PythonHosted.org , pypi.org "))          # order, case and spacing do not matter
    with pytest.raises(E.EgressError, match="allows 'pypi.org,files.pythonhosted.org,example.com'"):
        E.check_ready(S, FakeDocker(**ready, allow="pypi.org,files.pythonhosted.org,example.com"))   # broader than configured
    with pytest.raises(E.EgressError, match="allows 'pypi.org'"):
        E.check_ready(S, FakeDocker(**ready, allow="pypi.org"))                                       # narrower: also not what was asked
    with pytest.raises(E.EgressError, match="no allow-list label"):
        E.check_ready(S, FakeDocker(**ready, allow=""))                                               # started by hand: cannot be verified
    st = E.status(S, FakeDocker(**ready, allow="pypi.org,files.pythonhosted.org,example.com"))
    assert st["ready"] is False and st["allow_matches_running"] is False
    assert E.status(S, FakeDocker(**ready, allow="files.pythonhosted.org,pypi.org"))["allow_matches_running"] is True


def test_up_is_idempotent_when_everything_is_already_in_place():
    d = FakeDocker(network="internal", container="running", networks=["bridge", "reprofix-install"])
    done = E.up(S, d)
    assert done == ["proxy is answering"]
    assert not any(c[1:3] in (["network", "create"], ["run", "-d"], ["network", "connect"]) and c[1] != "exec" for c in d.calls)


def test_up_replaces_a_proxy_that_has_a_different_allow_list():
    d = FakeDocker(network="internal", container="running", networks=["bridge", "reprofix-install"], allow="pypi.org")
    done = E.up(S, d)
    assert any("different allow-list" in x for x in done) and d.allow == "pypi.org,files.pythonhosted.org"


def test_up_refuses_to_reuse_a_network_that_is_not_internal():
    with pytest.raises(E.EgressError, match="not internal"):
        E.up(S, FakeDocker(network="open"))


def test_up_reports_why_the_proxy_could_not_start_and_hints_at_the_image():
    with pytest.raises(E.EgressError, match="could not start the proxy.*docker/sandbox.Dockerfile"):
        E.up(S, FakeDocker(network="internal", fail={"run -d": "executable file not found"}))


def test_up_notices_a_proxy_that_never_answers(monkeypatch):
    monkeypatch.setattr(E.time, "time", iter([0, 0, 100, 100, 100, 100]).__next__)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    with pytest.raises(E.EgressError, match="did not answer its health check"):
        E.up(S, FakeDocker(health_ok=False))


def test_down_and_status():
    d = FakeDocker(network="internal", container="running", networks=["bridge", "reprofix-install"])
    st = E.status(S, d)
    assert st["ready"] and st["problem"] is None and st["allow_matches_running"] is True
    assert E.down(S, d) == ["removed reprofix-egress", "removed network reprofix-install"]
    assert not E.status(S, d)["ready"]


def test_selftest_runs_inside_the_network_and_reports_what_the_container_printed():
    ok, lines = E.selftest(S, FakeDocker(network="internal", container="running", networks=["reprofix-install"]))
    assert ok and len(lines) == 3
    ok, lines = E.selftest(S, FakeDocker(network="internal", container="running", networks=["reprofix-install"],
                                         selftest_out="FAIL direct connection succeeded\nPASS x\n", selftest_code=1))
    assert not ok and lines[0].startswith("FAIL")
    ok, lines = E.selftest(S, FakeDocker(network="internal", container="running", networks=["reprofix-install"], selftest_out="", selftest_code=125))
    assert not ok and "no output" in lines[0]
    with pytest.raises(E.EgressError):
        E.selftest(S, FakeDocker())                               # the self-test refuses to run against a missing setup


# --------------------------------------------------------------------------- configuration and CLI
def test_settings_read_the_install_network_from_the_environment(monkeypatch):
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")
    for k in ("REPROFIX_INSTALL_NETWORK", "REPROFIX_EGRESS_ALLOW", "REPROFIX_EGRESS_UPSTREAM"):
        monkeypatch.delenv(k, raising=False)
    d = Settings.from_env()
    assert d.install_network == "open" and d.egress_allow == ("pypi.org", "files.pythonhosted.org") and d.egress_upstream == ""
    monkeypatch.setenv("REPROFIX_INSTALL_NETWORK", " Proxy ")
    monkeypatch.setenv("REPROFIX_EGRESS_ALLOW", "PyPI.org, download.pytorch.org")
    s = Settings.from_env()
    assert s.install_network == "proxy" and s.egress_allow == ("pypi.org", "download.pytorch.org")


def test_cli_egress_commands(monkeypatch, capsys):
    from reprofix import cli
    d = FakeDocker()
    monkeypatch.setattr(E, "_run", d)
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")
    assert cli.main(["egress", "status"]) == 1
    assert "NOT READY" in capsys.readouterr().out
    assert cli.main(["egress", "up"]) == 0
    assert "proxy is answering" in capsys.readouterr().out
    assert cli.main(["egress", "status"]) == 0 and "internal (no route out)" in capsys.readouterr().out
    assert cli.main(["egress", "test"]) == 0 and "egress confinement verified" in capsys.readouterr().out
    assert cli.main(["egress", "down"]) == 0


def _doctor(monkeypatch, capsys, *, mode: str, docker: FakeDocker):
    """`reprofix doctor` with the Docker backend selected but no daemon: the sandbox object and the image check are faked."""
    from reprofix import cli
    import reprofix.sandbox as sandbox_pkg
    from reprofix.sandbox.base import SandboxPolicy

    class _SB:
        def policy(self):
            return SandboxPolicy(kind="docker", isolated=True)

    monkeypatch.setenv("REPROFIX_SANDBOX", "docker")
    monkeypatch.setenv("REPROFIX_INSTALL_NETWORK", mode)
    monkeypatch.setattr(sandbox_pkg, "make_sandbox", lambda settings: _SB())
    monkeypatch.setattr(E, "_run", docker)
    real_run = subprocess.run
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, *a, **k: subprocess.CompletedProcess(argv, 0, b"", b"") if argv[:3] == ["docker", "image", "inspect"] else real_run(argv, *a, **k))
    cli.main(["doctor"])
    return capsys.readouterr().out


def test_doctor_warns_when_the_install_network_is_open(monkeypatch, capsys):
    out = _doctor(monkeypatch, capsys, mode="open", docker=FakeDocker())
    assert "install network is OPEN" in out and "REPROFIX_INSTALL_NETWORK=proxy" in out


def test_doctor_runs_the_confinement_self_test_in_proxy_mode(monkeypatch, capsys):
    d = FakeDocker(network="internal", container="running", networks=["bridge", "reprofix-install"])
    out = _doctor(monkeypatch, capsys, mode="proxy", docker=d)
    assert out.count("install network:") == 3 and "[FAIL] install network" not in out
    assert any("--selftest" in c for c in d.calls)


def test_doctor_fails_in_proxy_mode_when_the_proxy_is_missing(monkeypatch, capsys):
    out = _doctor(monkeypatch, capsys, mode="proxy", docker=FakeDocker())
    assert "[FAIL] install network is 'proxy' but not ready" in out and "reprofix egress up" in out


def test_local_backend_reports_an_open_install_network():
    from reprofix.sandbox.local_backend import LocalSandbox
    pol = LocalSandbox(Settings(sandbox="local", allow_unsafe_local=True)).policy()
    assert pol.install_egress == "open" and pol.to_dict()["install_egress"] == {"mode": "open", "hosts": []}


def test_doctor_reports_the_gpu_probe_when_a_gpu_is_enabled(monkeypatch, capsys):
    from reprofix import cli
    import reprofix.sandbox as sandbox_pkg
    from reprofix.sandbox.base import SandboxPolicy

    class _SB:
        def policy(self):
            return SandboxPolicy(kind="docker", isolated=True)

    calls = []

    def fake_run(argv, *a, **k):
        calls.append(argv)
        if "--gpus" in argv:
            return subprocess.CompletedProcess(argv, 0, "0, NVIDIA L40S, 46068, 535.161.08\n", "")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setenv("REPROFIX_SANDBOX", "docker")
    monkeypatch.setenv("REPROFIX_SANDBOX_GPU", "1")
    monkeypatch.setenv("REPROFIX_INSTALL_NETWORK", "open")
    monkeypatch.setattr(sandbox_pkg, "make_sandbox", lambda settings: _SB())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "[ok] GPU in the sandbox: NVIDIA L40S, 46068 MiB each, driver 535.161.08" in out
    probe = next(c for c in calls if "--gpus" in c)
    assert probe[probe.index("--network") + 1] == "none" and probe[-3] == "nvidia-smi"


def test_doctor_reports_a_failed_gpu_probe(monkeypatch, capsys):
    from reprofix import cli
    import reprofix.sandbox as sandbox_pkg
    from reprofix.sandbox.base import SandboxPolicy

    class _SB:
        def policy(self):
            return SandboxPolicy(kind="docker", isolated=True)

    def fake_run(argv, *a, **k):
        if "--gpus" in argv:
            return subprocess.CompletedProcess(argv, 125, "", 'docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].')
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setenv("REPROFIX_SANDBOX", "docker")
    monkeypatch.setenv("REPROFIX_SANDBOX_GPU", "1")
    monkeypatch.setenv("REPROFIX_INSTALL_NETWORK", "open")
    monkeypatch.setattr(sandbox_pkg, "make_sandbox", lambda settings: _SB())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] GPU in the sandbox: GPU probe failed" in out and "could not select device driver" in out and "docs/gpu.md" in out


def test_doctor_says_when_the_gpu_is_not_enabled(monkeypatch, capsys):
    monkeypatch.delenv("REPROFIX_SANDBOX_GPU", raising=False)
    out = _doctor(monkeypatch, capsys, mode="open", docker=FakeDocker())
    assert "GPU: not enabled (REPROFIX_SANDBOX_GPU=0)" in out and "docs/gpu.md" in out
