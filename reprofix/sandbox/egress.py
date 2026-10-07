"""Egress allow-list proxy for the install phase of the Docker sandbox.

Standalone on purpose (standard library only, no imports from the rest of ReproFix): the same file is copied into the
sandbox image and run there as `python /opt/reprofix/egress.py`.

Why it exists. During `pip install` a sandbox container needs the network, and plain Docker cannot say "only PyPI".
The enforcement here comes from Docker networking, not from this program being polite:

    install container ──(--internal network: no gateway, no route out)──> this proxy ──> pypi.org / files.pythonhosted.org

The install container sits on a Docker network created with `--internal`, so the only thing it can reach is the proxy
container that is attached to that network (and, separately, to a normal bridge). The proxy then decides what is allowed:

  * only `CONNECT host:443` tunnels; plain HTTP requests are refused (pip talks HTTPS to PyPI);
  * the host must be on the allow-list (exact names or `*.suffix`); an IP literal is only allowed if it is listed;
  * (direct mode; in upstream mode the upstream proxy resolves names, so only the allow-list applies) the proxy resolves the
    name itself and refuses to connect to a non-public address (loopback, private ranges,
    link-local such as cloud metadata at 169.254.169.254, carrier-grade NAT, ...), so an allowed name that resolves to an
    internal address does not become a way into the host's network; it connects to the address it checked, not to a
    second lookup;
  * connection count, header size, idle time and tunnel lifetime are bounded.

What this does not do: it does not look inside the TLS tunnel (a package index could in principle be asked for anything
that host serves), it does not filter by URL path, and it does not make a wheel's contents trustworthy. It closes the
"install container can reach any host on the Internet" gap; it is not a general sandbox.
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import re
import select
import socket
import socketserver
import sys
import threading
import time
from urllib.parse import urlsplit

DEFAULT_ALLOW = ("pypi.org", "files.pythonhosted.org")
ALLOW_ENV = "REPROFIX_EGRESS_ALLOW"
DEFAULT_PORT = 3128

MAX_HEAD_BYTES = 8192        # request line + headers
HEAD_TIMEOUT_S = 10.0        # time to receive them
CONNECT_TIMEOUT_S = 10.0     # time to reach the destination
IDLE_TIMEOUT_S = 120.0       # no bytes in either direction
MAX_TUNNEL_S = 1800.0        # absolute lifetime of one tunnel
MAX_CONNECTIONS = 64

_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9\-.]*[a-z0-9])?$")


class PolicyError(Exception):
    """The request is refused by policy (as opposed to failing because the network failed)."""


# --------------------------------------------------------------------------- policy
def parse_authority(text: str) -> tuple[str, int]:
    """`host:port` or `[v6]:port` -> (host, port). Raises ValueError on anything else."""
    text = text.strip()
    if text.startswith("["):
        host, sep, rest = text[1:].partition("]")
        if not sep or not rest.startswith(":"):
            raise ValueError("malformed authority")
        port_s = rest[1:]
    else:
        host, sep, port_s = text.rpartition(":")
        if not sep:
            raise ValueError("a port is required")
    if not port_s.isdigit() or not 0 < int(port_s) < 65536:
        raise ValueError("bad port")
    host = host.lower().rstrip(".")
    if not host:
        raise ValueError("empty host")
    if not (_HOST_RE.match(host) or _is_ip(host)):
        raise ValueError("bad host")
    return host, int(port_s)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def is_public_address(ip: str) -> bool:
    """True only for globally routable addresses."""
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    if addr.version == 6:
        # IPv6 forms that carry an IPv4 address: the carried address decides. Python's is_global does not look inside them.
        n = int(addr)
        embedded = None
        if (n >> 32) in (0, int(ipaddress.IPv6Address("64:ff9b::")) >> 32, int(ipaddress.IPv6Address("::ffff:0:0:0")) >> 32):
            embedded = n & 0xFFFFFFFF                            # ::/96 (IPv4-compatible), 64:ff9b::/96 (NAT64), ::ffff:0:0:0/96 (SIIT)
        elif (n >> 112) == 0x2002:
            embedded = (n >> 80) & 0xFFFFFFFF                    # 6to4
        elif (n >> 96) == 0x20010000:
            embedded = (~n) & 0xFFFFFFFF                         # Teredo stores the client's address inverted
        if embedded is not None and not ipaddress.IPv4Address(embedded).is_global:
            return False
    return addr.is_global and not addr.is_multicast


class AllowList:
    """Exact host names and `*.suffix` patterns, on a fixed set of ports (443 by default)."""

    def __init__(self, hosts, ports=(443,)):
        self.exact: set[str] = set()
        self.suffixes: list[str] = []
        for h in hosts:
            h = str(h).strip().lower().rstrip(".")
            if not h:
                continue
            if h.startswith("*."):
                self.suffixes.append(h[1:])           # ".example.com": matches a.example.com, not example.com
            else:
                self.exact.add(h)
        self.ports = frozenset(ports)

    def allows(self, host: str, port: int) -> bool:
        host = host.lower().rstrip(".")
        if port not in self.ports:
            return False
        return host in self.exact or any(host.endswith(s) for s in self.suffixes)

    def describe(self) -> list[str]:
        return sorted(self.exact) + ["*" + s for s in sorted(self.suffixes)]


def parse_allow(value: str | None) -> list[str]:
    return [h for h in re.split(r"[,\s]+", value or "") if h] or list(DEFAULT_ALLOW)


# --------------------------------------------------------------------------- stats
class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.allowed: dict[str, int] = {}
        self.denied: dict[str, int] = {}
        self.started = time.time()

    def add(self, kind: str, host: str) -> None:
        with self._lock:
            d = self.allowed if kind == "allowed" else self.denied
            key = host[:100]
            if key in d or len(d) < 200:      # bounded: a hostile client cannot grow this without limit
                d[key] = d.get(key, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"allowed": dict(self.allowed), "denied": dict(self.denied),
                    "n_allowed": sum(self.allowed.values()), "n_denied": sum(self.denied.values()),
                    "uptime_s": round(time.time() - self.started, 1)}


# --------------------------------------------------------------------------- connecting
def connect_direct(host: str, port: int, allow_private: bool = False) -> socket.socket:
    """Resolve `host` ourselves and connect to an address we have checked."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    refused: str | None = None
    last: Exception | None = None
    for _fam, _typ, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if not allow_private and not is_public_address(ip):
            refused = ip
            continue
        try:
            return socket.create_connection((ip, port), timeout=CONNECT_TIMEOUT_S)
        except OSError as exc:
            last = exc
    if last is not None:
        raise last
    if refused is not None:
        raise PolicyError(f"{host} resolves to a non-public address ({refused})")
    raise OSError(f"{host} did not resolve")


def connect_via_upstream(upstream: str, host: str, port: int) -> socket.socket:
    """Open a tunnel to host:port through another HTTP proxy (for networks that only reach the Internet through one)."""
    u = urlsplit(upstream if "//" in upstream else "http://" + upstream)
    if not u.hostname:
        raise OSError("bad upstream proxy address")
    sock = socket.create_connection((u.hostname, u.port or 3128), timeout=CONNECT_TIMEOUT_S)
    try:
        lines = [f"CONNECT {host}:{port} HTTP/1.1", f"Host: {host}:{port}"]
        if u.username:
            cred = base64.b64encode(f"{u.username}:{u.password or ''}".encode()).decode()
            lines.append(f"Proxy-Authorization: Basic {cred}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        sock.settimeout(CONNECT_TIMEOUT_S)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(4096)
            if not chunk or len(head) > MAX_HEAD_BYTES:
                raise OSError("upstream proxy closed the connection")
            head += chunk
        status = head.split(b"\r\n", 1)[0].decode("latin-1")
        if not re.match(r"HTTP/1\.[01] 2\d\d", status):
            raise OSError(f"upstream proxy refused the tunnel: {status[:80]}")
        return sock
    except BaseException:
        sock.close()
        raise


# --------------------------------------------------------------------------- the proxy
def _respond(conn: socket.socket, code: int, reason: str, body: str = "", ctype: str = "text/plain; charset=utf-8") -> None:
    data = body.encode()
    head = (f"HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n"
            "Connection: close\r\n\r\n").encode()
    try:
        conn.sendall(head + data)
    except OSError:
        pass


def _read_head(conn: socket.socket) -> tuple[bytes, bytes]:
    """Read up to the blank line that ends the request head. Returns (head, any bytes that followed it)."""
    buf = b""
    deadline = time.time() + HEAD_TIMEOUT_S
    while b"\r\n\r\n" not in buf:
        if len(buf) > MAX_HEAD_BYTES or time.time() > deadline:
            raise ValueError("request head too large or too slow")
        chunk = conn.recv(2048)
        if not chunk:
            raise ValueError("connection closed")
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def _tunnel(a: socket.socket, b: socket.socket) -> None:
    start = last = time.time()
    socks = [a, b]
    for s in socks:
        s.setblocking(False)
    while True:
        now = time.time()
        if now - start > MAX_TUNNEL_S or now - last > IDLE_TIMEOUT_S:
            return
        try:
            readable, _, _ = select.select(socks, [], [], max(0.05, min(1.0, IDLE_TIMEOUT_S - (now - last))))
        except (OSError, ValueError):
            return
        for src in readable:
            dst = b if src is a else a
            try:
                data = src.recv(65536)
            except BlockingIOError:
                continue
            except OSError:
                return
            if not data:
                return
            last = time.time()
            view = memoryview(data)
            while view:
                try:
                    sent = dst.send(view)
                    view = view[sent:]
                except BlockingIOError:
                    select.select([], [dst], [], 5.0)
                except OSError:
                    return


class _Handler(socketserver.BaseRequestHandler):
    server: "_Server"

    def handle(self) -> None:
        conn: socket.socket = self.request
        conn.settimeout(HEAD_TIMEOUT_S)
        try:
            head, rest = _read_head(conn)
        except (ValueError, OSError):
            return
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split()
        if len(parts) != 3 or not parts[2].startswith("HTTP/"):
            _respond(conn, 400, "Bad Request", "malformed request line\n")
            return
        method, target = parts[0].upper(), parts[1]
        if method == "CONNECT":
            self._connect(conn, target, rest)
        elif method in ("GET", "HEAD") and target.startswith("/"):
            self._own_page(conn, target)
        else:
            try:
                host = (urlsplit(target).hostname or "?") if "://" in target else "?"
            except ValueError:                                    # e.g. "http://[::1": not a URL at all
                host = "?"
            self.server.log_event("DENY", f"{host} plain HTTP request ({method})")
            self.server.stats.add("denied", host)
            _respond(conn, 403, "Forbidden", "ReproFix egress policy: only HTTPS tunnels (CONNECT) to allow-listed hosts "
                                             "are permitted; plain HTTP requests are refused.\n")

    def _own_page(self, conn: socket.socket, target: str) -> None:
        if target.split("?")[0] == "/__health":
            body = json.dumps({"status": "ok", "allow": self.server.allow.describe(), "ports": sorted(self.server.allow.ports),
                               **self.server.stats.snapshot()})
            _respond(conn, 200, "OK", body, "application/json")
        else:
            _respond(conn, 404, "Not Found", "this is an egress proxy, not a web server\n")

    def _connect(self, conn: socket.socket, target: str, rest: bytes) -> None:
        srv = self.server
        try:
            host, port = parse_authority(target)
        except ValueError:
            _respond(conn, 400, "Bad Request", "bad CONNECT target\n")
            return
        if not srv.allow.allows(host, port):
            srv.log_event("DENY", f"{host}:{port} not on the allow-list")
            srv.stats.add("denied", f"{host}:{port}")
            _respond(conn, 403, "Forbidden", f"ReproFix egress policy: {host}:{port} is not on the allow-list "
                                             f"({', '.join(srv.allow.describe())}).\n")
            return
        try:
            if srv.upstream:
                remote = connect_via_upstream(srv.upstream, host, port)
            else:
                remote = connect_direct(host, port, srv.allow_private)
        except PolicyError as exc:
            srv.log_event("DENY", f"{host}:{port} {exc}")
            srv.stats.add("denied", f"{host}:{port}")
            _respond(conn, 403, "Forbidden", f"ReproFix egress policy: {exc}\n")
            return
        except OSError as exc:
            srv.log_event("FAIL", f"{host}:{port} {type(exc).__name__}: {exc}")
            _respond(conn, 502, "Bad Gateway", f"could not reach {host}:{port}\n")
            return
        srv.log_event("ALLOW", f"{host}:{port}")
        srv.stats.add("allowed", f"{host}:{port}")
        try:
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            if rest:
                remote.sendall(rest)
            _tunnel(conn, remote)
        except OSError:
            pass
        finally:
            remote.close()


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, addr, allow: AllowList, upstream: str | None, allow_private: bool, quiet: bool):
        super().__init__(addr, _Handler)
        self.allow = allow
        self.upstream = upstream
        self.allow_private = allow_private
        self.quiet = quiet
        self.stats = Stats()
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def log_event(self, what: str, detail: str) -> None:
        if not self.quiet:
            print(f"{time.strftime('%H:%M:%S')} {what:5} {detail}", file=sys.stderr, flush=True)

    def process_request(self, request, client_address):  # noqa: D102 - bounded concurrency
        if not self._slots.acquire(blocking=False):
            _respond(request, 503, "Service Unavailable", "too many connections\n")
            socketserver.TCPServer.shutdown_request(self, request)      # refused: no slot was taken, none is released
            return
        super().process_request(request, client_address)

    def shutdown_request(self, request):  # noqa: D102 - called when a handler thread finishes: give its slot back
        try:
            super().shutdown_request(request)
        finally:
            self._slots.release()


class EgressProxy:
    """The proxy as an object, for tests and for embedding. `start()` returns once it is accepting connections."""

    def __init__(self, allow=DEFAULT_ALLOW, host: str = "127.0.0.1", port: int = 0, upstream: str | None = None,
                 allow_private: bool = False, ports=(443,), quiet: bool = True):
        self._server = _Server((host, port), AllowList(allow, ports), upstream, allow_private, quiet)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address[:2]

    @property
    def stats(self) -> Stats:
        return self._server.stats

    def start(self) -> "EgressProxy":
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "EgressProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# --------------------------------------------------------------------------- self test (runs inside the install network)
def _code(status_line: str) -> int | None:
    m = re.match(r"HTTP/1\.[01] (\d{3})", status_line)
    return int(m.group(1)) if m else None


def selftest(proxy: str, allowed_target: str = "pypi.org:443", direct_probe: str = "1.1.1.1:443") -> tuple[bool, list[str]]:
    """Check, from where an install container sits, that (1) the Internet cannot be reached directly, (2) a host that is
    not on the list is refused by the proxy, (3) an allow-listed host is reachable through it."""
    lines: list[str] = []
    ok = True

    def record(passed: bool, text: str) -> None:
        nonlocal ok
        ok = ok and passed
        lines.append(("PASS " if passed else "FAIL ") + text)

    h, p = parse_authority(direct_probe)
    try:
        socket.create_connection((h, p), timeout=5).close()
        record(False, f"direct connection to {direct_probe} SUCCEEDED: this container is not isolated from the Internet")
    except OSError as exc:
        record(True, f"direct connection to {direct_probe} is impossible ({type(exc).__name__})")

    ph, pp = parse_authority(proxy)

    def connect(target: str) -> str:
        with socket.create_connection((ph, pp), timeout=10) as s:
            s.settimeout(15)
            s.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
            data = b""
            while b"\r\n" not in data:
                chunk = s.recv(512)
                if not chunk:
                    break
                data += chunk
            return data.split(b"\r\n", 1)[0].decode("latin-1")

    try:
        status = connect("example.invalid:443")
        record(_code(status) == 403, f"CONNECT to a host that is not allow-listed -> {status or 'no answer'}")
    except OSError as exc:
        record(False, f"could not talk to the proxy at {proxy}: {exc}")
    try:
        status = connect(allowed_target)
        record(_code(status) == 200, f"CONNECT to {allowed_target} -> {status or 'no answer'}")
    except OSError as exc:
        record(False, f"CONNECT to {allowed_target} failed: {exc}")
    return ok, lines


# --------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ReproFix egress allow-list proxy (HTTPS CONNECT only)")
    ap.add_argument("--listen", default=f"0.0.0.0:{DEFAULT_PORT}", help="host:port to listen on")
    ap.add_argument("--allow", default=None, help=f"comma-separated hosts; default ${ALLOW_ENV} or {','.join(DEFAULT_ALLOW)}")
    ap.add_argument("--upstream", default=os.environ.get("REPROFIX_EGRESS_UPSTREAM"),
                    help="reach the Internet through this HTTP proxy (optional)")
    ap.add_argument("--allow-private", action="store_true", help="TESTING ONLY: allow destinations that resolve to private addresses")
    ap.add_argument("--selftest", metavar="HOST:PORT", help="run the isolation check against a proxy and exit")
    ap.add_argument("--allowed-target", default="pypi.org:443", help="with --selftest: an allow-listed HOST:PORT that must be reachable")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        ok, lines = selftest(a.selftest, a.allowed_target)
        print("\n".join(lines))
        return 0 if ok else 1
    host, port = parse_authority(a.listen)
    allow = parse_allow(a.allow if a.allow is not None else os.environ.get(ALLOW_ENV))
    proxy = EgressProxy(allow=allow, host=host, port=port, upstream=a.upstream, allow_private=a.allow_private, quiet=a.quiet)
    print(f"egress proxy listening on {host}:{proxy.address[1]}; allowed: {', '.join(AllowList(allow).describe())}"
          + (f"; upstream {urlsplit(a.upstream).hostname}" if a.upstream else ""), file=sys.stderr, flush=True)
    try:
        proxy._server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
