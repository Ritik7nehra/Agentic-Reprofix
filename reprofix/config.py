"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Nebius's quickstart and "switch" docs give this global endpoint. Its cookbook examples use
# https://api.tokenfactory.us-central1.nebius.com/v1/ instead; `reprofix doctor` probes both with your key.
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Model IDs as published by Nebius. Their own pages spell the Nano ID differently (the cookbook uses
# "nvidia/nvidia-nemotron-3-nano-30b-a3b", the model catalog "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"), so the
# client resolves a configured ID against GET /models ignoring case and punctuation. `reprofix doctor` shows
# which spelling your key is served.
DEFAULT_MODELS = {
    "nano": "nvidia/nvidia-nemotron-3-nano-30b-a3b",
    "super": "nvidia/nemotron-3-super-120b-a12b",
    "ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}

# USD per 1M tokens (input, output), from Nebius's published model catalog (tokenfactory.nebius.com/model-catalog.md).
# Prices change: override with REPROFIX_PRICE_<TIER>, or set it empty to show cost as n/a.
DEFAULT_PRICES: dict[str, tuple[float, float] | None] = {
    "nano": (0.06, 0.24),
    "super": (0.30, 0.90),
    "ultra": (1.00, 3.00),
}

TIERS = ("nano", "super", "ultra")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _price(name: str, default: tuple[float, float] | None) -> tuple[float, float] | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return None
    try:
        a, b = (float(x) for x in raw.split(","))
        return (a, b)
    except ValueError as exc:  # pragma: no cover - config error path
        raise ValueError(f"{name} must look like '0.30,0.90'") from exc


def load_env_file(path: str | None = None) -> list[str]:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Variables that are already set in the real environment always win. The file is `./.env` unless
    REPROFIX_ENV_FILE points elsewhere; REPROFIX_ENV_FILE="" disables loading. Returns the keys it set.
    """
    raw = os.environ.get("REPROFIX_ENV_FILE", ".env") if path is None else path
    if not raw:
        return []
    p = Path(raw)
    if not p.is_file():
        return []
    loaded: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = (x.strip() for x in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:                      # inline comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


@dataclass
class Settings:
    nebius_api_key: str = ""
    nebius_base_url: str = DEFAULT_BASE_URL
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    prices: dict[str, tuple[float, float] | None] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    extra_body: dict = field(default_factory=dict)
    tavily_api_key: str = ""

    sandbox: str = "docker"
    sandbox_image: str = "reprofix-sandbox:latest"
    allow_unsafe_local: bool = False
    sandbox_gpu: bool = False
    allow_sdist: bool = False
    sandbox_memory_mb: int = 4096
    sandbox_cpus: float = 2.0
    sandbox_pids: int = 512
    # Network of the Docker *install* phase. "open": the default bridge, any host reachable. "proxy": a Docker network
    # created with --internal (no route out) plus an allow-list proxy container that only tunnels HTTPS to `egress_allow`
    # (see reprofix/sandbox/egress.py and `reprofix egress up`). Fails closed: if the proxy is not running, installs refuse.
    install_network: str = "open"
    egress_allow: tuple[str, ...] = ("pypi.org", "files.pythonhosted.org")
    egress_upstream: str = ""                  # optional: reach the Internet through this HTTP proxy
    egress_network: str = "reprofix-install"
    egress_container: str = "reprofix-egress"
    egress_port: int = 3128

    data_dir: Path = Path("./data")
    api_token: str = ""
    max_concurrent_runs: int = 2
    allowed_git_hosts: tuple[str, ...] = ("github.com",)
    allow_local_paths: bool = False
    allow_pull_requests: bool = True   # lets API callers open a GitHub pull request with a token they supply
    known_issues: bool = True          # search GitHub issues / the web for the observed failure (sends a sanitised query out)

    # Hard server-side ceilings: API requests are clamped to these.
    max_attempts_ceiling: int = 10
    max_run_seconds_ceiling: int = 3600
    max_command_timeout_ceiling: int = 1800
    max_repo_mb: int = 200

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()
        models = {
            "nano": os.environ.get("NEBIUS_MODEL_NANO", DEFAULT_MODELS["nano"]),
            "super": os.environ.get("NEBIUS_MODEL_SUPER", DEFAULT_MODELS["super"]),
            "ultra": os.environ.get("NEBIUS_MODEL_ULTRA", DEFAULT_MODELS["ultra"]),
        }
        prices = {
            "nano": _price("REPROFIX_PRICE_NANO", DEFAULT_PRICES["nano"]),
            "super": _price("REPROFIX_PRICE_SUPER", DEFAULT_PRICES["super"]),
            "ultra": _price("REPROFIX_PRICE_ULTRA", DEFAULT_PRICES["ultra"]),
        }
        egress_allow = tuple(
            h.strip().lower() for h in re.split(r"[,\s]+", os.environ.get("REPROFIX_EGRESS_ALLOW", "")) if h.strip()
        ) or ("pypi.org", "files.pythonhosted.org")
        extra = os.environ.get("NEBIUS_EXTRA_BODY", "").strip()
        extra_body = json.loads(extra) if extra else {}
        hosts = tuple(
            h.strip().lower()
            for h in os.environ.get("REPROFIX_ALLOWED_GIT_HOSTS", "github.com").split(",")
            if h.strip()
        )
        return cls(
            nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
            nebius_base_url=os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            models=models,
            prices=prices,
            extra_body=extra_body,
            tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
            sandbox=os.environ.get("REPROFIX_SANDBOX", "docker").strip().lower(),
            sandbox_image=os.environ.get("REPROFIX_SANDBOX_IMAGE", "reprofix-sandbox:latest"),
            allow_unsafe_local=_flag("REPROFIX_ALLOW_UNSAFE_LOCAL"),
            sandbox_gpu=_flag("REPROFIX_SANDBOX_GPU"),
            allow_sdist=_flag("REPROFIX_ALLOW_SDIST"),
            data_dir=Path(os.environ.get("REPROFIX_DATA_DIR", "./data")).resolve(),
            api_token=os.environ.get("REPROFIX_API_TOKEN", ""),
            max_concurrent_runs=int(os.environ.get("REPROFIX_MAX_CONCURRENT_RUNS", "2")),
            allowed_git_hosts=hosts or ("github.com",),
            allow_local_paths=_flag("REPROFIX_ALLOW_LOCAL_PATHS"),
            allow_pull_requests=_flag("REPROFIX_ALLOW_PULL_REQUESTS", True),
            known_issues=_flag("REPROFIX_KNOWN_ISSUES", True),
            install_network=os.environ.get("REPROFIX_INSTALL_NETWORK", "open").strip().lower() or "open",
            egress_allow=egress_allow,
            egress_upstream=os.environ.get("REPROFIX_EGRESS_UPSTREAM", "").strip(),
        )
