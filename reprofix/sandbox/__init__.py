from __future__ import annotations

from ..config import Settings
from .base import ExecResult, Sandbox, SandboxPolicy, SandboxUnavailable
from .docker_backend import DockerSandbox
from .local_backend import LocalSandbox


def make_sandbox(settings: Settings) -> Sandbox:
    if settings.sandbox == "local":
        return LocalSandbox(settings)
    if settings.sandbox == "docker":
        return DockerSandbox(settings)
    raise SandboxUnavailable(f"unknown sandbox backend {settings.sandbox!r} (use 'docker' or 'local')")


__all__ = ["Sandbox", "SandboxPolicy", "SandboxUnavailable", "ExecResult", "DockerSandbox", "LocalSandbox", "make_sandbox"]
