"""Sandbox interface.

Two phases, deliberately different:
  * install: network ON (needed to reach a package index). Wheels only by default so that
    untrusted `setup.py` files do not execute while the network is reachable.
  * run:     network OFF. The repository's own code never gets network access.

The agent's API keys are never placed in a sandbox environment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

DEPS_DIR = ".deps"


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class ExecResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    network: bool = False
    truncated: bool = False


@dataclass
class SandboxPolicy:
    kind: str
    isolated: bool  # True only when the backend isolates the filesystem too
    network_install: bool = True
    network_run: bool = False
    # What the install phase can reach: "open" (any host) or "allow-list" (only `install_hosts`, enforced by an
    # --internal Docker network plus the proxy in egress.py).
    install_egress: str = "open"
    install_hosts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "filesystem_isolated": self.isolated,
            "network_during_install": self.network_install,
            "network_during_run": self.network_run,
            "install_egress": {"mode": self.install_egress, "hosts": self.install_hosts},
            "notes": self.notes,
        }


class Sandbox(ABC):
    kind = "abstract"

    @abstractmethod
    def policy(self) -> SandboxPolicy: ...

    @abstractmethod
    def install(self, workdir: Path, requirements: list[str], timeout: int) -> ExecResult:
        """Install requirement files (paths relative to workdir) into workdir/.deps."""

    @abstractmethod
    def run(self, workdir: Path, argv: list[str], timeout: int) -> ExecResult:
        """Run argv with no network. argv[0] may be python/python3/pytest."""

    def probe_hardware(self, workdir: Path) -> dict:
        """Which GPU (if any) the run phase can see, as a report record (see sandbox/hardware.py). Never raises."""
        from . import hardware
        return hardware.record(requested=False, status="not_requested", probed_in=None, detail="this backend does not probe hardware")

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


def rewrite_argv(argv: list[str], python: str, isolated_site: bool = True) -> list[str]:
    """Map `python x.py` / `pytest -q` onto the sandbox interpreter.

    `-S` drops site-packages so that only `.deps` (installed from the repo's own
    requirements) is importable: a missing requirement then fails exactly as it would on a
    clean machine instead of being masked by whatever happens to be installed on the host.
    """
    head = [python] + (["-S"] if isolated_site else [])
    if argv[0] in {"python", "python3"}:
        return head + argv[1:]
    if argv[0] == "pytest":
        return head + ["-m", "pytest"] + argv[1:]
    raise ValueError(f"unsupported interpreter {argv[0]!r}")


def refuse_unsafe_requirements(workdir: Path, requirements: list[str], allow_sdist: bool) -> "ExecResult | None":
    """In a wheels-only install, an install result that says why the repository's requirements were refused (or None).
    Nothing is run: see sandbox/reqcheck.py for what is refused and why."""
    from .reqcheck import check_requirement_files, refusal
    if allow_sdist:
        return None
    problems = check_requirement_files(workdir, requirements)
    if not problems:
        return None
    return ExecResult(argv=["pip", "install", *[a for r in requirements for a in ("-r", r)]], exit_code=2, stdout="",
                      stderr=refusal(problems), duration_s=0.0, network=False)


def pip_install_argv(python: str, requirements: list[str], allow_sdist: bool) -> list[str]:
    argv = [
        python, "-m", "pip", "install",
        "--target", DEPS_DIR, "--upgrade",
        "--no-input", "--disable-pip-version-check", "--progress-bar", "off",
    ]
    if not allow_sdist:
        argv.append("--only-binary=:all:")
    for req in requirements:
        argv += ["-r", req]
    return argv
