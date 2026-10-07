# Deployment

How to run ReproFix on a Linux VM so that other people can use it. Two routes: **Docker Compose** (everything in
containers) and **systemd** (the app on the host, only the sandboxes in containers).

## What has and has not been checked

| | State |
|---|---|
| `docker-compose.yml` renders and its required variables are enforced | Checked with `docker compose config` (Compose v5.5.1), including a bug that check found (see below) |
| Static properties: data directory mounted at the same path inside and out, port on loopback only, token required, no secret copied into the image, non-root user, the egress service hardened and the install network `internal: true` | `tests/test_deploy.py`, 17 tests |
| The egress proxy's logic (allow-list, refusing non-public addresses, tunnelling, fail-closed readiness check, command lines) | `tests/test_egress.py` (real local sockets) and `tests/test_egress_ctl.py` (a recording fake of `docker`) |
| **The egress proxy and the `--internal` network confine a real `pip install` under a Docker daemon** | **Not tested.** `reprofix egress test` is the first thing to run on your VM. |
| `Dockerfile` builds | **Not tested.** No image was built. |
| The stack starts, the health check passes, a run works end to end from the container | **Not tested.** |
| `deploy/reprofix.service` | **Not tested.** Never installed. |
| `deploy/Caddyfile`: certificate issuance, streaming through the proxy | **Not tested.** |
| A Nebius VM was created, or the stack was run on Nebius AI Cloud | **No.** |

So treat this as a careful first draft. The first-start checklist below is how to find out quickly whether it works
on your VM. If something fails, the likeliest places are the Dockerfile (package names, the `docker:cli` image tag) and
the Docker socket permissions.

The check with `docker compose config` found one real defect: Compose interpolates variables for *every* service,
including ones in a profile that is not enabled, so a "required" variable on the optional TLS service would have
stopped a plain `docker compose up`. That variable now has a default.

## What you need

- A Linux x86-64 VM with Docker Engine and, for Route A, the Compose plugin
  ([install instructions](https://docs.docker.com/engine/install/)). Python 3.11 or newer for Route B (Ubuntu 24.04
  has 3.12; Ubuntu 22.04 has 3.10, which is too old).
- Outbound internet: `api.tokenfactory.nebius.com` (the model), `github.com` (cloning), PyPI (`pypi.org` and
  `files.pythonhosted.org`, for installs inside the sandbox), `api.tavily.com` if you set a Tavily key, and
  `api.github.com` if you use the pull-request button or the known-issues search (`REPROFIX_KNOWN_ISSUES=0` turns the
  search off). Building the images additionally needs Docker Hub and the Debian package mirrors (and the PyPI hosts, for the
  app image's `pip install`). Only the egress proxy container (not the sandboxes) needs to reach PyPI for installs; add hosts
  to `REPROFIX_EGRESS_ALLOW` only if a repository needs them (for example `download.pytorch.org` for CPU-only torch wheels;
  not tried).
- For Route A, a recent Docker Compose v2: the compose file uses `env_file` with `required: false`, which older releases
  may reject. It was rendered with Compose v5.5.1 only; I did not try an older version.
- A Nebius Token Factory API key. Without one only the offline demo works.
- Size: derived from the limits in `config.py`, **not measured**. Each sandboxed command may use 4 GB of memory and
  two runs may overlap by default, so 8 GB of RAM is a floor for `REPROFIX_MAX_CONCURRENT_RUNS=1` and 16 GB is
  comfortable for 2. Plan 20 GB of disk plus about 5 GB for every run of a PyTorch repository (measured here: the
  PyPI `torch` wheel unpacks to 5.3 GB, and a run keeps its own copy).

The hackathon requires the model calls to run on Nebius Token Factory or AI Cloud. ReproFix does that through the
Token Factory API wherever the *app* is hosted; hosting the app itself on Nebius AI Cloud is optional.

### If you use a Nebius AI Cloud VM

Nebius's pages for creating a VM ([manage VMs](https://docs.nebius.com/compute/virtual-machines/manage),
[quickstart](https://docs.nebius.com/compute/quickstart)) describe a console flow (Compute, Virtual machines, Create),
an `nebius compute instance create` CLI, an automatically assigned public IP and SSH access. The quickstart is written
around GPU VMs with the `ubuntu24.04-cuda13.0` image family. It says nothing I could find about CPU-only presets or
about firewall or security-group rules, and I did not create a VM. Before relying on a public address, confirm from
outside the VM which ports are actually reachable.

[Nebius Tunnels](https://docs.nebius.com/tunnels/quickstart) can expose a local HTTP service through a public URL while
the VM has no public address. The documentation says it is free during preview and does not say whether HTTPS or
Server-Sent Events (the live run stream) work through it. Untested.

## Route A: Docker Compose

```bash
git clone <your repository url> reprofix && cd reprofix
cp .env.example .env
```

Edit `.env`:

| Variable | Value |
|---|---|
| `NEBIUS_API_KEY` | your key |
| `REPROFIX_API_TOKEN` | a long random secret: `openssl rand -hex 24`. Compose refuses to start without one; `reprofix serve` itself only warns, so with Route B you must set it yourself. |
| `DOCKER_GID` | the VM's docker group id: `getent group docker \| cut -d: -f3` |
| `REPROFIX_HOST_DATA_DIR` | optional, default `/srv/reprofix` |

```bash
sudo install -d -o 1000 -g 1000 /srv/reprofix      # the app runs as uid 1000
docker compose up -d --build                       # builds the sandbox image and the app image, then starts the egress proxy and the app
docker compose ps                                  # reprofix, reprofix-egress (and sandbox-image, which exits when the build is done)
curl -s http://127.0.0.1:8000/api/health           # sandbox.install_egress should read {"mode": "allow-list", "hosts": ["pypi.org", ...]}
docker compose exec reprofix reprofix egress test          # the proxy's self-test, from a container placed like an install container
```

**The egress service.** Compose creates a Docker network named `reprofix-install` with `internal: true` (no route out) and a
`reprofix-egress` container, built from the sandbox image, that sits on that network and on the default one. The app starts
its install containers on the internal network, so during `pip install` they can reach only what the proxy tunnels to:
`REPROFIX_EGRESS_ALLOW` (default `pypi.org,files.pythonhosted.org`), HTTPS only. The app refuses to install if the network is
not internal or the proxy is not running. To go back to the open bridge network, set `REPROFIX_INSTALL_NETWORK=open` in
`.env`. `docker compose exec reprofix reprofix egress status` shows what the app sees. Design and limits:
[security.md](security.md#install-phase-egress-proxy). Not run under Docker by the author.

The app listens on `127.0.0.1:8000` only. From your laptop: `ssh -L 8000:127.0.0.1:8000 <user>@<vm>` and open
`http://127.0.0.1:8000`.

**Why the data directory is mounted at the same path inside and outside.** The app starts sandbox containers as
siblings through the host's Docker daemon, and the daemon resolves `-v <workspace>:/work` on the *host*. If the app saw
`/data` while the host had `/srv/reprofix`, every run would mount a path that does not exist. `tests/test_deploy.py`
asserts that the two paths are the same.

### Public HTTPS (optional)

Point a DNS name at the VM, open ports 80 and 443, then:

```bash
echo 'REPROFIX_DOMAIN=reprofix.example.com' >> .env
docker compose --profile tls up -d
```

Caddy obtains the certificate and proxies to the app with response buffering off (`flush_interval -1` in
`deploy/Caddyfile`), which the run stream needs. Without a domain, a name such as `<ip-with-dashes>.sslip.io` resolves to
the IP, but I have not tried it.

## Route B: systemd on the host

This keeps the Docker socket out of an application container: the app runs as an ordinary user and calls the
host's `docker`.

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin reprofix
sudo usermod -aG docker reprofix                   # root-equivalent, see below
sudo install -d -o reprofix -g reprofix /opt/reprofix /var/lib/reprofix
sudo -u reprofix git clone <your repository url> /opt/reprofix
cd /opt/reprofix
sudo -u reprofix python3 -m venv .venv && sudo -u reprofix .venv/bin/pip install -e .
docker build -f docker/sandbox.Dockerfile -t reprofix-sandbox:latest .

sudo tee /etc/reprofix.env >/dev/null <<EOF
NEBIUS_API_KEY=...
REPROFIX_API_TOKEN=$(openssl rand -hex 24)
REPROFIX_DATA_DIR=/var/lib/reprofix
EOF
sudo chmod 600 /etc/reprofix.env

sudo cp deploy/reprofix.service /etc/systemd/system/    # sets REPROFIX_INSTALL_NETWORK=proxy and runs `reprofix egress up` before start
sudo systemctl daemon-reload && sudo systemctl enable --now reprofix
journalctl -u reprofix -f
```

For TLS in front of it, install Caddy and use a `Caddyfile` like `deploy/Caddyfile`, with your real domain in place of
`{$REPROFIX_DOMAIN}` (a host-installed Caddy will not have that variable set) and `reverse_proxy 127.0.0.1:8000` in place
of `reverse_proxy reprofix:8000`.

## First-start checklist

Do these in order; each one narrows down what is wrong.

1. `curl -s http://127.0.0.1:8000/api/health`. `sandbox.available` should be `true` and `llm.configured` `true`. If the
   sandbox is unavailable, the error says whether the Docker CLI or the daemon is the problem (usually the socket
   group: check `DOCKER_GID`).
2. `reprofix egress test` (inside the app container for Route A: `docker compose exec reprofix reprofix egress test`). It must print
   three PASS lines (a direct connection to 1.1.1.1 is impossible, a host that is not on the list is refused, PyPI is
   reachable through the proxy) and "egress confinement verified". If it fails, installs will be refused (proxy mode) or
   unconfined (open mode); fix it before sharing the instance.
3. In the UI, enter the API token and press the **offline demo** button. It exercises the real sandbox (Docker, a real
   `pip` install through the egress proxy, the run with networking off) using a scripted stand-in for the model. If this
   verifies, the stack is sound apart from the model calls.
4. Start a real run on a small repository you control. This is the first thing that spends Nebius credit.

## Operating it

- **Logs:** `docker compose logs -f reprofix` or `journalctl -u reprofix -f`.
- **Upgrade:** `git pull`, then `docker compose up -d --build` (Route A) or `.venv/bin/pip install -e . && sudo systemctl restart reprofix` (Route B).
- **State:** runs and the SQLite database are under the data directory (`runs/<id>/…` and `reprofix.db`). Back up with
  `sqlite3 reprofix.db ".backup 'copy.db'"` rather than copying the file while the server runs. A run's `orig/` snapshot
  is what the pull-request button rebuilds a change from, so deleting `runs/<id>/` also ends that run's ability to open one.
- **Disk:** old runs are not removed automatically. Remove `runs/<id>/work` for runs you no longer need; PyTorch
  repositories leave about 5 GB each.
- **Limits:** `REPROFIX_MAX_CONCURRENT_RUNS` and `REPROFIX_ALLOWED_GIT_HOSTS` (default `github.com` only) are environment
  variables (see `.env.example`). The ceilings on attempts (10), run time (1 hour), per-command timeout (30 minutes) and
  the sandbox's memory, CPU and process limits are fixed in `reprofix/config.py` (and the 2M-token cap in
  `reprofix/api/app.py`); changing them means editing the code. They clamp whatever a client asks for.

## Risks to know before you share a link

- **The Docker socket is root on the VM.** Route A mounts it into the app container; Route B puts the service user in
  the `docker` group. Either way, a flaw in ReproFix or the sandbox configuration can become root on that machine. Use
  a VM that holds nothing else and that you can throw away. Container hardening is applied to the *sandbox* containers
  (no network while running, read-only root, all capabilities dropped, memory and process limits); it does not protect
  the host from whoever controls the Docker API.
- **Install-time network.** In proxy mode (the Compose and systemd default) the install container can reach only the
  hosts in `REPROFIX_EGRESS_ALLOW`, through a proxy that has not been run under a Docker daemon by the author. In `open`
  mode (or if you set it so) the container is on the default Docker bridge and can reach any host. Either way wheels only
  are accepted by default and the requirements files are checked for lines that would build code (`-e .`, local paths,
  direct URLs, `--no-binary`), so no package code runs during that window (`REPROFIX_ALLOW_SDIST=1` removes that protection).
  See [security.md](security.md).
- **GPU VMs.** `REPROFIX_SANDBOX_GPU=1` needs a GPU, the NVIDIA driver and the NVIDIA Container Toolkit on the VM; setup
  steps (copied from Nebius's and NVIDIA's documentation, not run) are in [gpu.md](gpu.md).
- **One shared token, no accounts.** Anyone who has the API token can start runs that spend your Nebius credit and
  execute (in the sandbox) any repository on an allowed host. There is no per-user quota or rate limit. Cap spend on the
  Nebius side if the account offers a limit (I did not check that it does), keep `REPROFIX_MAX_CONCURRENT_RUNS` low, and
  share the token only with people you would trust with the credit.
- **Plain HTTP exposes tokens.** The API token and, in the pull-request form, a visitor's GitHub token are sent from the
  browser to the server. Over `http://` on anything but loopback they cross the network unencrypted. The pull-request
  form refuses to submit from an insecure page; the API token has no such guard, so serve a public instance over HTTPS only.
- **Ephemeral VMs.** If the VM is preemptible or short-lived, the data directory disappears with it unless it is on a
  persistent disk.
