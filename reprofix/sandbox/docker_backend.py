"""Docker sandbox: the default, isolated backend.

Run phase: --network none, read-only root, all capabilities dropped, no-new-privileges, pid /
memory / cpu limits, non-root user, only the run's workspace mounted (read-write).
Install phase: same hardening, but it needs a network so pip can reach an index. Two modes
(REPROFIX_INSTALL_NETWORK):
  open   the default bridge: the container can reach any host. Wheels-only installs (the default) mean no package code
         executes during that window.
  proxy  a Docker network created with --internal (no route out) whose only neighbour is an allow-list proxy that
         tunnels HTTPS to PyPI and nothing else (sandbox/egress.py, `reprofix egress up`). It fails closed: if the proxy
         or the internal network is missing, installs are refused rather than run with an open network.
See docs/security.md.
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path

from ..config import Settings
from ..util import truncate_middle
from . import egress_ctl, hardware
from .base import (DEPS_DIR, ExecResult, Sandbox, SandboxPolicy, SandboxUnavailable, pip_install_argv, refuse_unsafe_requirements,
                   rewrite_argv)


def container_user() -> str:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        uid, gid = os.getuid(), os.getgid()
        return f"{uid}:{gid}" if uid != 0 else "1000:1000"
    return "1000:1000"


def build_docker_argv(
    *,
    image: str,
    name: str,
    workdir: Path,
    argv: list[str],
    network: bool,
    memory_mb: int,
    cpus: float,
    pids: int,
    gpu: bool,
    user: str,
    env: dict[str, str],
    network_name: str | None = None,
) -> list[str]:
    """`network_name` replaces the default bridge for the install phase (the --internal network of proxy mode)."""
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", (network_name or "bridge") if network else "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(pids),
        "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
        "--cpus", str(cpus),
        "--read-only",
        "--tmpfs", "/tmp:rw,size=512m,mode=1777",
        "-v", f"{workdir.as_posix()}:/work:rw",
        "-w", "/work",
        "--user", user,
    ]
    if gpu and not network:
        cmd += ["--gpus", "all"]
    for k, v in sorted(env.items()):
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    cmd += argv
    return cmd


class DockerSandbox(Sandbox):
    kind = "docker"

    def __init__(self, settings: Settings, check: bool = True):
        self.settings = settings
        if settings.install_network not in ("open", "proxy"):
            raise SandboxUnavailable(f"REPROFIX_INSTALL_NETWORK must be 'open' or 'proxy', not {settings.install_network!r}")
        if check:
            self._check_daemon()
            if self._proxy_mode:
                try:
                    egress_ctl.check_ready(settings)
                except egress_ctl.EgressError as exc:
                    raise SandboxUnavailable(f"install network mode is 'proxy' but it is not ready: {exc}") from exc

    @property
    def _proxy_mode(self) -> bool:
        return self.settings.install_network == "proxy"

    def _check_daemon(self) -> None:
        try:
            r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxUnavailable(f"docker CLI not usable: {exc}") from exc
        if r.returncode != 0:
            raise SandboxUnavailable(
                "docker daemon not reachable: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1]
            )

    def policy(self) -> SandboxPolicy:
        s = self.settings
        if self._proxy_mode:
            install_note = (f"Install phase: --internal Docker network '{s.egress_network}' with no route out; the only way "
                            f"through is the allow-list proxy ({', '.join(s.egress_allow)}, HTTPS only).")
        else:
            install_note = "Install phase has an OPEN bridge network (any host reachable); wheels-only by default. Set REPROFIX_INSTALL_NETWORK=proxy to restrict it."
        return SandboxPolicy(
            kind="docker", isolated=True, network_install=True, network_run=False,
            install_egress="allow-list" if self._proxy_mode else "open",
            install_hosts=list(s.egress_allow) if self._proxy_mode else [],
            notes=[
                "Run phase: --network none, read-only root, cap-drop ALL, no-new-privileges, pid/memory/cpu limits.",
                install_note,
                "API keys are never passed into the container.",
            ],
        )

    def _env(self) -> dict[str, str]:
        return {
            "HOME": "/tmp", "PYTHONPATH": f"/work/{DEPS_DIR}", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_CACHE_DIR": "/tmp/pip-cache",
        }

    def _exec(self, workdir: Path, argv: list[str], timeout: int, network: bool) -> ExecResult:
        name = f"reprofix-{uuid.uuid4().hex[:12]}"
        s = self.settings
        env = self._env()
        network_name = None
        if network and self._proxy_mode:
            env.update(egress_ctl.proxy_env(s))
            network_name = s.egress_network
        cmd = build_docker_argv(
            image=s.sandbox_image, name=name, workdir=workdir.resolve(), argv=argv, network=network,
            memory_mb=s.sandbox_memory_mb, cpus=s.sandbox_cpus, pids=s.sandbox_pids, gpu=s.sandbox_gpu,
            user=container_user(), env=env, network_name=network_name,
        )
        t0 = time.time()
        timed_out = False
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
            out, err, code = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
            out, err, code = exc.stdout or b"", exc.stderr or b"", None
        so, t1 = truncate_middle(out.decode("utf-8", "replace"))
        se, t2 = truncate_middle(err.decode("utf-8", "replace"))
        return ExecResult(argv=argv, exit_code=code, stdout=so, stderr=se, duration_s=time.time() - t0,
                          timed_out=timed_out, network=network, truncated=t1 or t2)

    def prepare(self, workdir: Path) -> None:
        # Container user differs from host user when the host is root; make the workspace writable.
        if getattr(os, "getuid", lambda: -1)() == 0:
            subprocess.run(["chmod", "-R", "a+rwX", str(workdir)], check=False)

    def probe_hardware(self, workdir: Path) -> dict:
        """Run nvidia-smi the way the run phase runs the repository: same container flags (--gpus all, no network)."""
        if not self.settings.sandbox_gpu:
            return hardware.record(requested=False, status="not_requested", probed_in=None,
                                   detail="REPROFIX_SANDBOX_GPU is not set, so the sandbox containers get no GPU: CPU only")
        where = "inside the sandbox container (docker run --gpus all, no network)"
        try:
            self.prepare(workdir)
            q = self._exec(workdir, hardware.SMI_QUERY_ARGV, 60, network=False)
            h = self._exec(workdir, hardware.SMI_HEADER_ARGV, 60, network=False) if q.exit_code == 0 else None
        except Exception as exc:        # a probe must never stop a run
            return hardware.record(requested=True, status="probe_failed", probed_in=where, detail=f"{type(exc).__name__}: {str(exc)[:160]}")
        return hardware.from_smi(requested=True, probed_in=where, query_stdout=q.stdout, query_code=q.exit_code,
                                 query_stderr=q.stderr, header_stdout=h.stdout if h else "")

    def install(self, workdir: Path, requirements: list[str], timeout: int) -> ExecResult:
        if self._proxy_mode:        # re-checked for every install: the proxy can die or the network be replaced mid-session
            try:
                egress_ctl.check_ready(self.settings)
            except egress_ctl.EgressError as exc:
                raise SandboxUnavailable(f"refusing to install: install network mode is 'proxy' but {exc}") from exc
        refused = refuse_unsafe_requirements(workdir, requirements, self.settings.allow_sdist)
        if refused is not None:
            return refused
        self.prepare(workdir)
        argv = pip_install_argv("python", requirements, self.settings.allow_sdist)
        return self._exec(workdir, argv, timeout, network=True)

    def run(self, workdir: Path, argv: list[str], timeout: int) -> ExecResult:
        self.prepare(workdir)
        real = rewrite_argv(argv, "python")
        res = self._exec(workdir, real, timeout, network=False)
        res.argv = argv
        return res
