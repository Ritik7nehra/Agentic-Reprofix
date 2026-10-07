"""The egress allow-list proxy, exercised over real sockets on the loopback interface (no Internet, no Docker)."""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from reprofix.sandbox import egress
from reprofix.sandbox.egress import AllowList, EgressProxy, is_public_address, parse_authority


# --------------------------------------------------------------------------- helpers
class Echo:
    """A tiny TCP echo server standing in for a package index."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    @staticmethod
    def _serve(c):
        try:
            while True:
                d = c.recv(4096)
                if not d:
                    return
                c.sendall(d)
        except OSError:
            pass
        finally:
            c.close()

    def close(self):
        self.alive = False
        self.sock.close()


@pytest.fixture
def echo():
    e = Echo()
    yield e
    e.close()


def talk(proxy: EgressProxy, request: bytes, read_all: bool = True) -> bytes:
    with socket.create_connection(proxy.address, timeout=5) as s:
        s.sendall(request)
        s.settimeout(5)
        data = b""
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
                if not read_all and b"\r\n\r\n" in data:
                    break
        except socket.timeout:
            pass
        return data


def status_of(data: bytes) -> int:
    return int(data.split(b" ", 2)[1])


def test_the_module_is_standalone():
    """It is copied alone into the sandbox image, so it must not import anything from the rest of ReproFix."""
    src = open(egress.__file__).read()
    assert "from reprofix" not in src and "import reprofix" not in src and "from .." not in src


# --------------------------------------------------------------------------- policy pieces
def test_allow_list_matches_exact_names_wildcards_and_ports():
    a = AllowList(["PyPI.org", "files.pythonhosted.org.", "*.example.com"])
    assert a.allows("pypi.org", 443) and a.allows("FILES.pythonhosted.org", 443)
    assert a.allows("cdn.example.com", 443) and a.allows("a.b.example.com", 443)
    assert not a.allows("example.com", 443)                      # *.example.com does not include the bare domain
    assert not a.allows("evilpypi.org", 443) and not a.allows("pypi.org.evil.com", 443) and not a.allows("xexample.com", 443)
    assert not a.allows("pypi.org", 80) and not a.allows("pypi.org", 8443)
    assert not a.allows("127.0.0.1", 443) and not a.allows("93.184.216.34", 443)   # an IP literal must be listed explicitly
    assert AllowList(["93.184.216.34"]).allows("93.184.216.34", 443)


@pytest.mark.parametrize("text,expected", [
    ("pypi.org:443", ("pypi.org", 443)), ("PyPI.org.:443", ("pypi.org", 443)), ("[2606:4700::1111]:443", ("2606:4700::1111", 443)),
    ("127.0.0.1:8080", ("127.0.0.1", 8080)),
])
def test_parse_authority_accepts_well_formed_targets(text, expected):
    assert parse_authority(text) == expected


@pytest.mark.parametrize("text", ["pypi.org", "pypi.org:", "pypi.org:0", "pypi.org:65536", "pypi.org:abc", ":443", "[::1", "a b:443",
                                   "pypi.org/path:443", "pypi.org:443\r\nHost: x", "-bad.example:443", "http://pypi.org:443"])
def test_parse_authority_rejects_everything_else(text):
    with pytest.raises(ValueError):
        parse_authority(text)


@pytest.mark.parametrize("ip,public", [
    ("8.8.8.8", True), ("93.184.216.34", True), ("2606:4700:4700::1111", True),
    ("127.0.0.1", False), ("10.1.2.3", False), ("172.16.0.5", False), ("192.168.1.1", False),
    ("169.254.169.254", False),                                   # cloud metadata
    ("100.64.0.1", False),                                       # carrier-grade NAT
    ("0.0.0.0", False), ("224.0.0.1", False), ("::1", False), ("fe80::1", False), ("fc00::1", False),
    ("::ffff:127.0.0.1", False), ("::ffff:10.0.0.1", False), ("not-an-ip", False),
    # IPv6 forms that carry an IPv4 address: the carried address decides
    ("64:ff9b::7f00:1", False), ("64:ff9b::a9fe:a9fe", False), ("64:ff9b::808:808", True), ("::10.0.0.1", False),
    ("::ffff:0:a00:1", False), ("2002:a00:1::1", False), ("2001:0:4136:e378:8000:63bf:f5ff:fffe", False),
])
def test_only_globally_routable_addresses_count_as_public(ip, public):
    assert is_public_address(ip) is public


# --------------------------------------------------------------------------- behaviour over real sockets
def test_an_allowed_host_gets_a_working_tunnel_and_is_counted(echo):
    with EgressProxy(allow=["127.0.0.1"], ports=(echo.port,), allow_private=True) as proxy:
        with socket.create_connection(proxy.address, timeout=5) as s:
            s.sendall(f"CONNECT 127.0.0.1:{echo.port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            s.settimeout(5)
            head = b""
            while b"\r\n\r\n" not in head:
                head += s.recv(1024)
            assert head.startswith(b"HTTP/1.1 200")
            s.sendall(b"hello through the tunnel")
            assert s.recv(1024) == b"hello through the tunnel"
        assert proxy.stats.snapshot()["allowed"] == {f"127.0.0.1:{echo.port}": 1}


def test_bytes_sent_right_after_the_connect_head_are_not_lost(echo):
    with EgressProxy(allow=["127.0.0.1"], ports=(echo.port,), allow_private=True) as proxy:
        with socket.create_connection(proxy.address, timeout=5) as s:
            s.sendall(f"CONNECT 127.0.0.1:{echo.port} HTTP/1.1\r\n\r\nEARLY".encode())    # pipelined, before the 200
            s.settimeout(5)
            data = b""
            while b"EARLY" not in data:
                data += s.recv(1024)
            assert b"200" in data.split(b"\r\n", 1)[0]


def test_a_host_that_is_not_on_the_list_is_refused_and_named_in_the_stats():
    with EgressProxy(allow=["pypi.org"]) as proxy:
        resp = talk(proxy, b"CONNECT evil.example:443 HTTP/1.1\r\nHost: evil.example:443\r\n\r\n")
        assert status_of(resp) == 403 and b"not on the allow-list" in resp and b"pypi.org" in resp
        resp = talk(proxy, b"CONNECT pypi.org:22 HTTP/1.1\r\n\r\n")                       # right host, wrong port
        assert status_of(resp) == 403
        snap = proxy.stats.snapshot()
        assert snap["denied"] == {"evil.example:443": 1, "pypi.org:22": 1} and snap["n_allowed"] == 0


def test_an_allowed_name_that_resolves_to_a_private_address_is_refused(echo):
    """The DNS-based way in: an allow-listed name pointing at the host's own network must not be reachable."""
    with EgressProxy(allow=["localhost"], ports=(echo.port,)) as proxy:                   # allow_private stays False
        resp = talk(proxy, f"CONNECT localhost:{echo.port} HTTP/1.1\r\n\r\n".encode())
        assert status_of(resp) == 403 and b"non-public" in resp
        assert proxy.stats.snapshot()["n_allowed"] == 0


def test_plain_http_requests_are_refused_even_for_an_allowed_host():
    with EgressProxy(allow=["pypi.org"], ports=(80, 443)) as proxy:
        resp = talk(proxy, b"GET http://pypi.org/simple/ HTTP/1.1\r\nHost: pypi.org\r\n\r\n")
        assert status_of(resp) == 403 and b"plain HTTP" in resp
        assert proxy.stats.snapshot()["denied"] == {"pypi.org": 1}


@pytest.mark.parametrize("target", ["http://[::1", "http://[", "http://[::1]:99999/x", "http://a b/", "ftp://[x"])
def test_an_absolute_uri_that_is_not_a_url_gets_a_refusal_not_a_crashed_handler(target, capfd):
    with EgressProxy(allow=["pypi.org"], quiet=True) as proxy:
        resp = talk(proxy, f"GET {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        assert status_of(resp) in (400, 403), resp
        assert proxy.stats.snapshot()["n_allowed"] == 0
    assert "Traceback" not in capfd.readouterr().err


def test_health_page_reports_the_policy_and_other_paths_are_404():
    with EgressProxy(allow=["pypi.org", "*.pythonhosted.org"]) as proxy:
        body = json.loads(talk(proxy, b"GET /__health HTTP/1.1\r\n\r\n").split(b"\r\n\r\n", 1)[1])
        assert body["status"] == "ok" and body["allow"] == ["pypi.org", "*.pythonhosted.org"] and body["ports"] == [443]
        assert status_of(talk(proxy, b"GET /anything HTTP/1.1\r\n\r\n")) == 404


@pytest.mark.parametrize("junk", [b"hello\r\n\r\n", b"CONNECT\r\n\r\n", b"CONNECT a:443 NOTHTTP\r\n\r\n", b"CONNECT bad host:443 HTTP/1.1\r\n\r\n"])
def test_malformed_requests_get_a_400_and_do_not_hurt_the_server(junk):
    with EgressProxy(allow=["pypi.org"]) as proxy:
        assert status_of(talk(proxy, junk)) == 400
        assert status_of(talk(proxy, b"GET /__health HTTP/1.1\r\n\r\n")) == 200          # still serving


def test_an_oversized_request_head_is_dropped_without_a_reply_or_a_crash():
    with EgressProxy(allow=["pypi.org"]) as proxy:
        assert talk(proxy, b"CONNECT pypi.org:443 HTTP/1.1\r\n" + b"X-Pad: " + b"a" * 20000) == b""
        assert status_of(talk(proxy, b"GET /__health HTTP/1.1\r\n\r\n")) == 200


def test_an_unreachable_destination_is_a_502_not_a_hang():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]                             # free now, and nothing listens on it
    with EgressProxy(allow=["127.0.0.1"], ports=(port,), allow_private=True) as proxy:
        assert status_of(talk(proxy, f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())) == 502


def test_connection_cap_refuses_extra_clients_and_gives_slots_back(monkeypatch):
    monkeypatch.setattr(egress, "MAX_CONNECTIONS", 2)
    with EgressProxy(allow=["pypi.org"]) as proxy:
        held = [socket.create_connection(proxy.address, timeout=5) for _ in range(2)]    # two idle clients occupy both slots
        time.sleep(0.3)
        assert status_of(talk(proxy, b"GET /__health HTTP/1.1\r\n\r\n")) == 503
        for s in held:
            s.close()
        deadline = time.time() + 5
        while time.time() < deadline:                              # slots come back as the handlers notice the close
            if status_of(talk(proxy, b"GET /__health HTTP/1.1\r\n\r\n")) == 200:
                break
            time.sleep(0.1)
        else:
            pytest.fail("slots were not released after the clients disconnected")
        # and the refusal path did not mint an extra slot: the cap is still 2
        held = [socket.create_connection(proxy.address, timeout=5) for _ in range(2)]
        time.sleep(0.3)
        assert status_of(talk(proxy, b"GET /__health HTTP/1.1\r\n\r\n")) == 503
        for s in held:
            s.close()


def test_an_idle_tunnel_is_closed(monkeypatch, echo):
    monkeypatch.setattr(egress, "IDLE_TIMEOUT_S", 0.5)
    with EgressProxy(allow=["127.0.0.1"], ports=(echo.port,), allow_private=True) as proxy:
        with socket.create_connection(proxy.address, timeout=5) as s:
            s.sendall(f"CONNECT 127.0.0.1:{echo.port} HTTP/1.1\r\n\r\n".encode())
            s.settimeout(5)
            assert b"200" in s.recv(1024)
            t0 = time.time()
            assert s.recv(1024) == b""                             # the proxy hung up on us
            assert time.time() - t0 < 4


# --------------------------------------------------------------------------- chaining through another proxy
class FakeUpstream:
    """A minimal HTTP proxy that tunnels every CONNECT to one local port and records what it was asked for."""

    def __init__(self, target_port: int, status: str = "200 Connection Established"):
        self.requests: list[str] = []
        self.target_port, self.status = target_port, status
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        data = b""
        while b"\r\n\r\n" not in data:
            data += c.recv(1024)
        self.requests.append(data.decode("latin-1"))
        c.sendall(f"HTTP/1.1 {self.status}\r\n\r\n".encode())
        if not self.status.startswith("200"):
            c.close()
            return
        up = socket.create_connection(("127.0.0.1", self.target_port))
        egress._tunnel(c, up)
        c.close()
        up.close()


def test_the_proxy_can_chain_through_an_upstream_proxy_and_still_applies_its_own_list(echo):
    up = FakeUpstream(echo.port)
    with EgressProxy(allow=["index.example"], ports=(443,), upstream=f"http://user:secret@127.0.0.1:{up.port}") as proxy:
        denied = talk(proxy, b"CONNECT other.example:443 HTTP/1.1\r\n\r\n")
        assert status_of(denied) == 403 and up.requests == []     # never reached the upstream
        with socket.create_connection(proxy.address, timeout=5) as s:
            s.sendall(b"CONNECT index.example:443 HTTP/1.1\r\n\r\n")
            s.settimeout(5)
            assert b"200" in s.recv(1024)
            s.sendall(b"ping")
            assert s.recv(1024) == b"ping"
    assert up.requests[0].startswith("CONNECT index.example:443 HTTP/1.1")
    assert "Proxy-Authorization: Basic dXNlcjpzZWNyZXQ=" in up.requests[0]


def test_an_upstream_that_refuses_the_tunnel_gives_a_502(echo):
    up = FakeUpstream(echo.port, status="403 Forbidden")
    with EgressProxy(allow=["index.example"], upstream=f"127.0.0.1:{up.port}") as proxy:
        assert status_of(talk(proxy, b"CONNECT index.example:443 HTTP/1.1\r\n\r\n")) == 502


# --------------------------------------------------------------------------- the isolation self-test
def test_selftest_passes_when_the_world_is_as_it_should_be(echo):
    with EgressProxy(allow=["127.0.0.1"], ports=(echo.port,), allow_private=True) as proxy:
        ok, lines = egress.selftest(f"127.0.0.1:{proxy.address[1]}", allowed_target=f"127.0.0.1:{echo.port}", direct_probe="127.0.0.1:1")
    assert ok, lines
    assert [ln.split()[0] for ln in lines] == ["PASS", "PASS", "PASS"]


def test_selftest_fails_loudly_when_the_internet_is_directly_reachable(echo):
    with EgressProxy(allow=["127.0.0.1"], ports=(echo.port,), allow_private=True) as proxy:
        ok, lines = egress.selftest(f"127.0.0.1:{proxy.address[1]}", allowed_target=f"127.0.0.1:{echo.port}",
                                    direct_probe=f"127.0.0.1:{echo.port}")           # "direct" connection works -> not isolated
    assert not ok and lines[0].startswith("FAIL") and "not isolated" in lines[0]


def test_selftest_fails_when_the_allowed_host_cannot_be_reached(echo):
    with EgressProxy(allow=["pypi.org"]) as proxy:                  # 127.0.0.1 is not on this list
        ok, lines = egress.selftest(f"127.0.0.1:{proxy.address[1]}", allowed_target=f"127.0.0.1:{echo.port}", direct_probe="127.0.0.1:1")
    assert not ok and lines[2].startswith("FAIL")


def test_selftest_reports_an_unreachable_proxy():
    ok, lines = egress.selftest("127.0.0.1:1", allowed_target="pypi.org:443", direct_probe="127.0.0.1:1")
    assert not ok and sum(ln.startswith("FAIL") for ln in lines) == 2


def test_cli_entry_runs_the_selftest(echo, capsys):
    with EgressProxy(allow=["127.0.0.1"], ports=(echo.port,), allow_private=True) as proxy:
        # the default targets (pypi.org, 1.1.1.1) are not reachable from here, so the exit code must be non-zero
        assert egress.main(["--selftest", f"127.0.0.1:{proxy.address[1]}"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_default_allow_list_is_pypi_only_and_can_come_from_the_environment():
    assert egress.parse_allow(None) == ["pypi.org", "files.pythonhosted.org"]
    assert egress.parse_allow("pypi.org, download.pytorch.org") == ["pypi.org", "download.pytorch.org"]
    assert egress.parse_allow("  ") == ["pypi.org", "files.pythonhosted.org"]
