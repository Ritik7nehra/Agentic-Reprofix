"""Set up, inspect and test the install-phase egress proxy with the Docker CLI.

    reprofix egress up      create the --internal network and start the proxy container
    reprofix egress status  what is running, and whether the network really is internal
    reprofix egress test    from inside the network: no direct Internet, a stranger host refused, PyPI reachable
    reprofix egress down    remove the container and the network

Everything here is built from `Settings` and runs the `docker` CLI; the CLI call is injectable so the command lines can
be tested without a Docker daemon. None of this has been run against a real daemon by the author: see docs/security.md.
"""
from __future__ import annotations

import re

import subprocess
import time
from typing import Callable

from ..config import Settings

PROXY_SCRIPT = "/opt/reprofix/egress.py"        # copied into the sandbox image by docker/sandbox.Dockerfile
ALLOW_LABEL = "reprofix.egress.allow"


class EgressError(RuntimeError):
    pass


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _run(argv: list[str], timeout: int = 60) -> "subprocess.CompletedProcess[str]":
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise EgressError(f"docker CLI not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EgressError(f"`{' '.join(argv[:3])} ...` timed out") from exc


def proxy_url(s: Settings) -> str:
    return f"http://{s.egress_container}:{s.egress_port}"


def proxy_env(s: Settings) -> dict[str, str]:
    """Environment that points pip (and anything else that honours it) at the proxy. Only HTTPS is proxied: plain HTTP has
    no route and no proxy, so it fails instead of being allowed."""
    url = proxy_url(s)
    return {"HTTPS_PROXY": url, "https_proxy": url}


def allow_string(s: Settings) -> str:
    return ",".join(s.egress_allow)


def _allow_set(text: str | None) -> frozenset[str]:
    return frozenset(h.strip().lower() for h in re.split(r"[,\s]+", text or "") if h.strip())


# --------------------------------------------------------------------------- command lines
def network_create_argv(s: Settings) -> list[str]:
    return ["docker", "network", "create", "--internal", s.egress_network]


def proxy_run_argv(s: Settings) -> list[str]:
    cmd = [
        "docker", "run", "-d", "--name", s.egress_container, "--restart", "unless-stopped",
        "--network", "bridge",                                   # the proxy is the only thing with a way out
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
        "--tmpfs", "/tmp:rw,size=16m,mode=1777",
        "--pids-limit", "256", "--memory", "256m", "--memory-swap", "256m", "--cpus", "1",
        "--user", "1000:1000",
        "--label", f"{ALLOW_LABEL}={allow_string(s)}",
        "-e", f"REPROFIX_EGRESS_ALLOW={allow_string(s)}",
    ]
    if s.egress_upstream:
        cmd += ["-e", f"REPROFIX_EGRESS_UPSTREAM={s.egress_upstream}"]
    cmd += [s.sandbox_image, "python", PROXY_SCRIPT, "--listen", f"0.0.0.0:{s.egress_port}"]
    return cmd


def connect_argv(s: Settings) -> list[str]:
    return ["docker", "network", "connect", s.egress_network, s.egress_container]


def selftest_argv(s: Settings) -> list[str]:
    """A throwaway container placed exactly where an install container is placed."""
    target = next((h for h in s.egress_allow if not h.startswith("*")), "pypi.org")
    return [
        "docker", "run", "--rm", "--network", s.egress_network, "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--memory", "128m", "--user", "1000:1000",
        s.sandbox_image, "python", PROXY_SCRIPT, "--selftest", f"{s.egress_container}:{s.egress_port}",
        "--allowed-target", f"{target}:443",
    ]


def _health_argv(s: Settings) -> list[str]:
    code = f"import urllib.request as u; u.urlopen('http://127.0.0.1:{s.egress_port}/__health', timeout=3).read()"
    return ["docker", "exec", s.egress_container, "python", "-c", code]


# --------------------------------------------------------------------------- inspection
def inspect_network(s: Settings, run: Runner | None = None) -> dict:
    run = run or _run
    r = run(["docker", "network", "inspect", "-f", "{{.Internal}}", s.egress_network])
    return {"exists": r.returncode == 0, "internal": r.returncode == 0 and r.stdout.strip() == "true"}


def inspect_container(s: Settings, run: Runner | None = None) -> dict:
    run = run or _run
    r = run(["docker", "inspect", "-f", "{{.State.Running}}|{{index .Config.Labels \"" + ALLOW_LABEL + "\"}}|{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
             s.egress_container])
    if r.returncode != 0:
        return {"exists": False, "running": False, "allow": None, "networks": []}
    running, _, rest = r.stdout.strip().partition("|")
    allow, _, nets = rest.partition("|")
    return {"exists": True, "running": running == "true", "allow": allow or None, "networks": nets.split()}


def check_ready(s: Settings, run: Runner | None = None) -> None:
    """Raise unless installs through the proxy are actually confined. Used before every install in proxy mode, so a
    missing or misconfigured proxy refuses installs instead of silently running them with an open network."""
    run = run or _run
    net = inspect_network(s, run)
    if not net["exists"]:
        raise EgressError(f"Docker network {s.egress_network!r} does not exist: run `reprofix egress up` (docker compose up creates it)")
    if not net["internal"]:
        raise EgressError(f"Docker network {s.egress_network!r} is NOT internal, so it has a route to the Internet and the allow-list "
                          f"would not be enforced. Remove it (`docker network rm {s.egress_network}`) and run `reprofix egress up`")
    c = inspect_container(s, run)
    if not c["running"]:
        raise EgressError(f"the egress proxy container {s.egress_container!r} is not running: run `reprofix egress up`")
    if s.egress_network not in c["networks"]:
        raise EgressError(f"the egress proxy is not attached to {s.egress_network!r}: run `reprofix egress up`")
    if not c["allow"]:
        raise EgressError(f"the egress proxy {s.egress_container!r} carries no allow-list label, so its allow-list cannot be verified "
                          f"(`reprofix egress up` and docker-compose.yml both set it): run `reprofix egress up`")
    if _allow_set(c["allow"]) != _allow_set(allow_string(s)):
        raise EgressError(f"the running egress proxy allows {c['allow']!r} but REPROFIX_EGRESS_ALLOW is {allow_string(s)!r}: "
                          "run `reprofix egress up` to recreate it")


def status(s: Settings, run: Runner | None = None) -> dict:
    run = run or _run
    net, c = inspect_network(s, run), inspect_container(s, run)
    try:
        check_ready(s, run)
        problem = None
    except EgressError as exc:
        problem = str(exc)
    return {"network": net, "container": c, "ready": problem is None, "problem": problem, "mode": s.install_network,
            "allow": list(s.egress_allow), "allow_matches_running": (_allow_set(c["allow"]) == _allow_set(allow_string(s))) if c["exists"] else None}


# --------------------------------------------------------------------------- actions
def up(s: Settings, run: Runner | None = None) -> list[str]:
    """Idempotent: creates what is missing, replaces a proxy whose allow-list differs, returns what it did."""
    run = run or _run
    done: list[str] = []
    net = inspect_network(s, run)
    if not net["exists"]:
        r = run(network_create_argv(s))
        if r.returncode != 0:
            raise EgressError("could not create the network: " + (r.stderr.strip() or r.stdout.strip()))
        done.append(f"created internal network {s.egress_network}")
    elif not net["internal"]:
        raise EgressError(f"network {s.egress_network!r} exists but is not internal; remove it with `docker network rm {s.egress_network}` and retry")
    c = inspect_container(s, run)
    if c["exists"] and (not c["running"] or c["allow"] != allow_string(s)):
        run(["docker", "rm", "-f", s.egress_container])
        done.append(f"removed the old proxy container ({'stopped' if not c['running'] else 'different allow-list'})")
        c = {"exists": False, "running": False, "allow": None, "networks": []}
    if not c["exists"]:
        r = run(proxy_run_argv(s), 120)
        if r.returncode != 0:
            raise EgressError("could not start the proxy: " + (r.stderr.strip() or r.stdout.strip())
                              + f" (is {s.sandbox_image} built from the current docker/sandbox.Dockerfile, which contains {PROXY_SCRIPT}?)")
        done.append(f"started {s.egress_container} (allow: {allow_string(s)})")
        c["networks"] = ["bridge"]
    if s.egress_network not in c["networks"]:
        r = run(connect_argv(s))
        if r.returncode != 0:
            raise EgressError("could not attach the proxy to the internal network: " + (r.stderr.strip() or r.stdout.strip()))
        done.append(f"attached the proxy to {s.egress_network}")
    deadline = time.time() + 20
    while True:
        if run(_health_argv(s), 15).returncode == 0:
            break
        if time.time() > deadline:
            raise EgressError("the proxy container started but did not answer its health check within 20 s: "
                              f"see `docker logs {s.egress_container}`")
        time.sleep(0.5)
    done.append("proxy is answering")
    return done


def down(s: Settings, run: Runner | None = None) -> list[str]:
    run = run or _run
    done = []
    if run(["docker", "rm", "-f", s.egress_container]).returncode == 0:
        done.append(f"removed {s.egress_container}")
    if run(["docker", "network", "rm", s.egress_network]).returncode == 0:
        done.append(f"removed network {s.egress_network}")
    return done


def selftest(s: Settings, run: Runner | None = None) -> tuple[bool, list[str]]:
    """Run the three checks from inside the install network. Needs the Internet for the PyPI check."""
    run = run or _run
    check_ready(s, run)
    r = run(selftest_argv(s), 90)
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    if not lines:
        lines = ["FAIL the self-test produced no output: " + (r.stderr.strip()[-200:] or f"exit {r.returncode}")]
    return r.returncode == 0 and all(ln.startswith("PASS") for ln in lines), lines
