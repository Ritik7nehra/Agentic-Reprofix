"""DEV-ONLY sandbox: subprocess with resource limits and (when available) a network namespace.

This does NOT isolate the filesystem. It exists so the pipeline can be developed and tested on
machines without Docker. It refuses to start unless REPROFIX_ALLOW_UNSAFE_LOCAL=1.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

from ..config import Settings
from ..util import truncate_middle
from .base import (DEPS_DIR, ExecResult, Sandbox, SandboxPolicy, SandboxUnavailable, pip_install_argv, refuse_unsafe_requirements,
                   rewrite_argv)

MAX_CAPTURE_BYTES = 8 * 1024 * 1024


def _netns_available() -> bool:
    try:
        r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _drain(stream, sink: list[bytes], cap: int) -> None:
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if total < cap:
            sink.append(chunk[: cap - total])
        total += len(chunk)


class LocalSandbox(Sandbox):
    kind = "local"

    def __init__(self, settings: Settings, python: str | None = None):
        if not settings.allow_unsafe_local:
            raise SandboxUnavailable(
                "The local backend does not isolate the filesystem. "
                "Set REPROFIX_ALLOW_UNSAFE_LOCAL=1 to use it on a trusted development machine."
            )
        self.settings = settings
        self.python = python or sys.executable
        self.netns = _netns_available()
        cache = settings.data_dir / "pip-cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.pip_cache = cache

    def policy(self) -> SandboxPolicy:
        notes = ["DEV ONLY: repository code can read and write anywhere this process can."]
        if self.netns:
            notes.append("Run phase executes inside a new network namespace (no network).")
        else:
            notes.append("Network namespaces unavailable: run phase has NO network isolation.")
        return SandboxPolicy(kind="local", isolated=False, network_run=not self.netns, notes=notes)

    def probe_hardware(self, workdir: Path) -> dict:
        """The development backend has no device isolation: whatever nvidia-smi shows on this machine is what runs see."""
        from . import hardware
        where = "this machine (development backend: no device isolation)"
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return hardware.record(requested=False, status="none_visible", probed_in=where, detail="nvidia-smi is not installed here: CPU only")
        try:
            q = subprocess.run([exe, *hardware.SMI_QUERY_ARGV[1:]], capture_output=True, text=True, timeout=30)
            h = subprocess.run([exe], capture_output=True, text=True, timeout=30) if q.returncode == 0 else None
        except (OSError, subprocess.SubprocessError) as exc:
            return hardware.record(requested=False, status="probe_failed", probed_in=where, detail=f"{type(exc).__name__}: {str(exc)[:160]}")
        return hardware.from_smi(requested=False, probed_in=where, query_stdout=q.stdout, query_code=q.returncode,
                                 query_stderr=q.stderr, header_stdout=h.stdout if h else "")

    # -- environment ---------------------------------------------------------
    def _env(self, workdir: Path) -> dict[str, str]:
        # Allow-list only. NEBIUS_API_KEY / TAVILY_API_KEY must never reach repository code.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(workdir / ".home"),
            "USERPROFILE": str(workdir / ".home"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONPATH": str(workdir / DEPS_DIR),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CACHE_DIR": str(self.pip_cache),
            "TMPDIR": str(workdir / ".home"),
            "TEMP": str(workdir / ".home"),
            "TMP": str(workdir / ".home"),
        }
        for k in ("SYSTEMROOT", "SystemRoot", "WINDIR", "windir", "COMSPEC", "PATHEXT"):
            if k in os.environ:
                env[k] = os.environ[k]
        return env

    def _limits(self, timeout: int):
        if resource is None or os.name == "nt":
            return None
        mem = self.settings.sandbox_memory_mb * 1024 * 1024

        def apply() -> None:  # runs in the child between fork and exec
            if hasattr(os, "setsid"):
                os.setsid()
            if resource is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))
                resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except (ValueError, OSError):  # pragma: no cover
                    pass

        return apply

    def _exec(self, argv: list[str], cwd: Path, timeout: int, network: bool, wrap_netns: bool) -> ExecResult:
        cwd = cwd.resolve()  # PYTHONPATH must be absolute: the child's cwd is the workdir itself
        (cwd / ".home").mkdir(exist_ok=True)
        full = (["unshare", "-rn"] + argv) if (wrap_netns and self.netns) else argv
        t0 = time.time()
        limits_fn = self._limits(timeout)
        proc = subprocess.Popen(
            full, cwd=cwd, env=self._env(cwd),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=limits_fn if os.name != "nt" else None,
        )
        out: list[bytes] = []
        err: list[bytes] = []
        threads = [
            threading.Thread(target=_drain, args=(proc.stdout, out, MAX_CAPTURE_BYTES), daemon=True),
            threading.Thread(target=_drain, args=(proc.stderr, err, MAX_CAPTURE_BYTES), daemon=True),
        ]
        for t in threads:
            t.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            proc.wait()
        for t in threads:
            t.join(timeout=5)
        so, t1 = truncate_middle(b"".join(out).decode("utf-8", "replace"))
        se, t2 = truncate_middle(b"".join(err).decode("utf-8", "replace"))
        return ExecResult(
            argv=argv, exit_code=None if timed_out else proc.returncode, stdout=so, stderr=se,
            duration_s=time.time() - t0, timed_out=timed_out, network=network, truncated=t1 or t2,
        )

    # -- public API ----------------------------------------------------------
    def install(self, workdir: Path, requirements: list[str], timeout: int) -> ExecResult:
        refused = refuse_unsafe_requirements(workdir, requirements, self.settings.allow_sdist)
        if refused is not None:
            return refused
        argv = pip_install_argv(self.python, requirements, self.settings.allow_sdist)
        return self._exec(argv, workdir, timeout, network=True, wrap_netns=False)

    def run(self, workdir: Path, argv: list[str], timeout: int) -> ExecResult:
        real = rewrite_argv(argv, self.python)
        res = self._exec(real, workdir, timeout, network=False, wrap_netns=True)
        res.argv = argv  # report what the user asked for, not the wrapped form
        return res
