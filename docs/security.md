# Security

ReproFix executes code from repositories it did not write, on the instruction of a language model that has read text
from those same repositories. Treat it accordingly. This page states what is protected, what is not, and what you
must do before exposing it to anyone else.

**Short version:** run the default Docker backend, keep the server on `127.0.0.1` or behind your own authenticated
proxy, set `REPROFIX_API_TOKEN` if it listens anywhere else, and do not point it at repositories you would not run on a
throw-away machine. The Docker backend has been checked at the command-line level only; it has not been executed in
this repository's test environment: no sandbox command was ever run through a Docker daemon.

## Threat model

| Actor | What they can do | Assumed defence |
|---|---|---|
| A repository author (untrusted) | Put arbitrary Python in the repo, add malicious requirements, write text meant to steer the model | Sandbox, wheels-only installs with a check of the requirements files, untrusted-text fencing, patch and command guards |
| Web content and issue-tracker text returned by search (untrusted) | Contain instructions aimed at the model | Fencing; URLs only count as evidence if the search returned them |
| The text of a paper or README pasted by the user (untrusted) | Make the claim reader see a wrong number | The text is read by fixed patterns, never sent to a model, and the report shows what was read |
| A caller of the HTTP API | Start runs, read results | Optional bearer token; URL host allow-list; server-side ceilings on every budget |
| The model itself (fallible, possibly manipulated) | Propose commands, edits, evidence | None of its output is executed or trusted without a check by code |
| A person opening a pull request | Hand the server a GitHub token so it can act as them on GitHub | Used for one request, never stored or logged; explicit confirmation; see "Pull requests" |

Out of scope: a hostile operator, a compromised host or Docker daemon, multi-tenant isolation between mutually
distrusting users.

## Executing untrusted code

**Two phases, deliberately different** (`reprofix/sandbox/base.py`):

- *Install:* network on, because pip must reach an index. **Wheels only by default** (`--only-binary=:all:`), so no
  package's `setup.py` runs while the network is reachable. The flag alone is not enough: **a requirements file can defeat
  it.** Run against pip 24.0, each of `--no-binary :all:`, `-e ./pkg`, `./pkg` and `pkg @ <url of a source archive>` in a
  `requirements.txt` made pip build the package and execute its `setup.py` despite `--only-binary=:all:`. So ReproFix reads
  the requirements files first (`reprofix/sandbox/reqcheck.py`, including `-r`/`-c` includes) and refuses to start pip if they
  contain anything beyond plain named requirements, `--hash`, an https `--index-url`/`--extra-index-url`, `--prefer-binary`,
  `--pre`, `--require-hashes` and `--only-binary :all:`. The refusal is an ordinary install failure that names the file, line
  and reason, so the diagnoser can propose editing it. This reads the *text* of the files; pip's parser stays the authority, so the
  check errs toward refusing. A repository that really needs `-e .` fails to install until the line is changed (or
  `REPROFIX_ALLOW_SDIST=1`, below). `REPROFIX_ALLOW_SDIST=1` lifts all of this and should only be set for repositories you
  trust. With `REPROFIX_INSTALL_NETWORK=open` (the default of a plain `reprofix serve`) that network
  is Docker's default bridge: any host. With `REPROFIX_INSTALL_NETWORK=proxy` it is an egress-confined network; see
  "Install-phase egress proxy" below.
- *Run:* **no network.** The repository's own code, including the training script and its tests, never reaches the
  network.

**Docker backend (default).** Each command is a fresh `docker run --rm` with: `--cap-drop ALL`,
`--security-opt no-new-privileges`, a read-only root filesystem, a 512 MB `/tmp` tmpfs, pid, memory (swap equal to
memory) and CPU limits, a non-root user, and only the run's workspace mounted read-write. Run phase uses
`--network none`.

Known gaps:

1. **The install phase has an open bridge network unless the egress proxy is on.** Plain Docker cannot allow-list
   hostnames, so in `open` mode the container can reach any host during install. Wheels-only installs mean no package code
   executes in that window, but a wheel's contents are still unpacked, and anything that does run (pip itself) can reach
   out. `proxy` mode closes that gap by design (below) but **has only been tested with real sockets on one machine and
   with command-line fakes; it has never been run under a Docker daemon**, so whether Docker's `--internal` network and
   the proxy container confine a real install as intended is unverified. `reprofix egress test` is how you find out.
2. **No disk quota on the workspace mount.** `/tmp` is capped; `/work` is limited only by the host's disk.
3. **No seccomp/AppArmor profile beyond Docker's defaults,** and no gVisor/Firecracker-class isolation. A
   container escape is a host compromise.
4. **GPU runs** (`REPROFIX_SANDBOX_GPU=1`) add `--gpus all` to the run phase only (never the install phase), which also
   adds the NVIDIA device nodes and driver libraries to the container: a larger surface than the CPU-only sandbox, still
   with no network. The command line is unit-tested; **no GPU, driver or container toolkit was available, so it has never
   run.** See [gpu.md](gpu.md).
5. **The Docker socket is root on the host.** The sandbox needs the Docker API, so a deployed ReproFix always holds
   it: `docker-compose.yml` mounts the socket into the app container (non-root uid with the docker group added,
   `cap_drop: ALL`, `no-new-privileges`, port published on loopback only), and the systemd route puts the service user
   in the `docker` group. Those measures limit the app container itself; they do not stop whoever controls the Docker
   API from starting a privileged container. ReproFix builds every `docker run` argument list in code from its settings
   and never from request data, but a bug in ReproFix, or a container escape from a sandbox, would still end as root on
   that VM. Run it on a machine that holds nothing else. See [deployment.md](deployment.md).

### Install-phase egress proxy

`REPROFIX_INSTALL_NETWORK=proxy` (what `docker-compose.yml` and the systemd unit set) confines the install phase to a list of
hosts, `REPROFIX_EGRESS_ALLOW` (default `pypi.org` and `files.pythonhosted.org`):

- Install containers start on a Docker network created with `--internal`, which has no route to the Internet. They get
  `HTTPS_PROXY` set to the proxy container, which sits on that network **and** the normal one. Plain HTTP has no proxy and no
  route, so it fails instead of being allowed.
- The proxy (`reprofix/sandbox/egress.py`, one standard-library file with no third-party imports) accepts only
  `CONNECT host:443`, checks the host against the list (exact names, or `*.suffix` entries you add), resolves it itself and
  **refuses addresses that are not publicly routable** (so an allowed name pointing at `127.0.0.1`, a private range, the
  cloud metadata address, or an IPv6 form that carries one of those, such as `64:ff9b::7f00:1`, gets nothing; in `upstream`
  mode the upstream proxy does the resolving, so only the allow-list applies), and then copies bytes without looking inside: the TLS session is end to end between
  pip and PyPI, and nothing is decrypted. Plain-HTTP requests, other ports and malformed requests are refused. It caps
  connections and closes idle tunnels.
- **It fails closed.** Before every install `check_ready` verifies that the network exists and is internal, that the proxy
  is running and attached to it, and that the allow-list it carries (a container label that `reprofix egress up` and
  docker-compose.yml both set) is the same set as `REPROFIX_EGRESS_ALLOW`; a proxy with no label, or a different list, is
  refused. Otherwise the install is refused with the reason, not run with an open network. The label is what the proxy was
  *started with*; ReproFix cannot see inside the container to confirm it kept to it. `reprofix egress test` runs a throw-away container placed exactly like an install container
  and checks three things: it cannot connect directly to the Internet, the proxy refuses a host not on the list, and the
  proxy reaches an allowed host.
- **What it does not do.** It restricts *which hosts* are reachable, not *what is fetched from them*: PyPI is a public
  upload site, so a package an attacker published there can still be installed (wheels only, by default, so it does not
  run). A name on the list is trusted completely, including everything it serves; adding hosts adds exactly that much trust.
  It does not apply to the run phase (which has no network at all) or to cloning (done by the server, not in the sandbox).
  A bug in the proxy is a bug in code that sits between untrusted containers and the Internet; it is small on purpose, and
  tested over real sockets, but it has only had one independent read-through and has not been tried against a hostile client.
  Known limits: one container can use all 64 connection slots (there is no per-client limit) and a tunnel can stay open for
  up to 30 minutes, so a hostile install can delay other runs' installs (it cannot reach anything new); and the self-test only
  checks that a direct TCP connection to 1.1.1.1:443 fails. DNS from the internal network (Docker's embedded resolver),
  ICMP and services on the Docker host's gateway address are not tested.

**Local backend (development only).** It refuses to start unless `REPROFIX_ALLOW_UNSAFE_LOCAL=1`. It runs the
repository's code as your user with: an environment allow-list (so `NEBIUS_API_KEY` and `TAVILY_API_KEY` never reach
it), resource limits (CPU time, memory, file size, open files), a process group that is killed on timeout, and, where
the kernel allows, a new network namespace (`unshare -rn`) for the run phase. **It does not isolate the filesystem.**
The UI and the report carry a "development sandbox" banner whenever it is in use.

**Secrets.** API keys live only in the server process. They are never written into a sandbox environment, the
`/api/health` response, events, or reports. Search queries sent to Tavily are sanitised first (absolute paths reduced to
base names, long token-like strings and `Bearer ...` removed).

**What leaves the machine, besides model calls.**

- *Known-issue searches* (`REPROFIX_KNOWN_ISSUES`, on by default; `0` sends nothing). When a run is not already passing, a
  short query goes to `api.github.com/search/issues`, scoped to the repository that was given, and, if a Tavily key is set,
  to Tavily restricted to `github.com`, `stackoverflow.com`, `discuss.pytorch.org`, `discuss.huggingface.co` and
  `forums.developer.nvidia.com`. The query is built only from identifiers (`[A-Za-z0-9_]` words: the exception name, names
  from the error message, the metric name) with paths, quotes, long tokens and search operators removed, so text in a log
  cannot add a `repo:` qualifier or change what is searched. The GitHub request is unauthenticated: it carries no token, does
  not follow redirects, and stops for the rest of the run when GitHub answers 403 or 429. A failure you consider private
  still leaks the words of its exception to those services; turn the feature off for private code (private repositories
  are not supported anyway).
- *Paper text and claims.* The text you paste and the quotes read from it stay in the server's database and the report. They are
  never sent to a model or to a search service. The headline claim's metric name and number go to the models the way an
  `--expected` value does.

**What the workspace audit covers.** The report's "protected files unchanged" check looks at the files the *agent*
edited. A repository's own code could rewrite files in its workspace while it runs; that is not audited. In the
benchmark, the hidden check compares protected files byte for byte against the originals.

## Controlling what the model can do

- **Commands:** only `python`, `python3` or `pytest`, split with `shlex`, rejecting shell metacharacters; never passed
  to a shell. An explicit command from the API is validated the same way.
- **Edits:** exact, unique search/replace or new-file creation; paths are confined to the workspace (no `..`, absolute
  paths, or symlinks); `.deps/`, `.git/`, `.home/` and caches are off limits; `tests/**`, `test_*.py`, `README*` and
  `LICENSE*` are protected; at most 5 files per patch; applied atomically and reverted on failure.
- **Hardcoding:** a patch that writes the expected metric into code that prints or computes a metric is refused, and the
  report repeats the check on the final diff. This is a heuristic, not a proof; the reviewer model and the human
  reading the diff are the backstop.
- **Read-only tools:** the diagnoser's `read_file`, `grep` and `list_dir` are confined to the workspace and capped in
  output size.
- **Budgets:** attempts, wall-clock time, per-command timeout and total tokens are limited per run, and the server
  clamps whatever the API request asks for to its own ceilings.

## Pull requests (a GitHub token passes through the server)

The "Open pull request" button and `reprofix pr` create a branch on GitHub (in a fork when the token cannot push to the
repository) and open a pull request, as the owner of the token. That is a write action in someone else's name, so it is
fenced:

- **The token is the caller's own and lives for one request.** The UI sends it in the request body; the server holds it
  in memory for that call and never writes it to the database, events, reports or logs. The GitHub client redacts it from
  error text. `reprofix pr` reads it from `GITHUB_TOKEN`, never from the command line (which other users can see in a
  process list). A test asserts that the token is not in the database after a successful call.
- **The server operator can see it.** Because the server makes the GitHub calls, a token entered in a web UI passes
  through the operator's machine. If you do not trust the operator of an instance, do not paste a token into it: run
  `reprofix pr` on your own machine from the run directory instead.
- **HTTPS or loopback only, in the UI.** The page refuses to enable the form when it was loaded over plain `http://` from
  anything other than `localhost`, `127.0.0.1` or `[::1]`. The server cannot enforce this itself (it may sit behind a TLS
  proxy), so a direct API client can still send a token over HTTP: serve any shared instance over HTTPS. After a failed
  attempt the token stays in the page's memory (so it can be retried); it is cleared only when the pull request is
  opened, or when the page is closed or reloaded.
- **Explicit and bounded.** The request must carry `confirm: true`. The server opens at most one pull request per run
  (a per-run lock and a database record make a double submit safe); `reprofix pr` does not track that, so running it
  twice opens two. `REPROFIX_ALLOW_PULL_REQUESTS=0` removes the feature from the server and the UI; it does not affect
  `reprofix pr`, which runs as you on your own machine. `GET .../pull-request` previews the target, the files, the title and
  the description, and makes no network call.
- **What it will and will not write.** The changed files are rebuilt from the stored diff applied to a snapshot of the
  original repository, at the commit that was cloned; the run workspace is not read. It refuses deletions, paths outside the
  repository, and anything under `.github/` (workflow files can run code with the repository's secrets). Before
  overwriting an existing file it checks that GitHub's copy is the one the diff was made against, and otherwise stops.
  The client does not follow redirects (a test asserts it), so a token is not sent to another host. Runs that did not end `verified`, `executes` or
  `partial`, or that have no recorded 40-character commit, offer no pull request.
- **What the pull request says.** The description is drafted from the report's computed checks; the diagnosis lines are
  the model's words and are labelled as such. The person can edit it before sending. It is public once opened.
- **Token scopes.** The interface tells people a fine-grained token needs *Contents* and *Pull requests* write access on
  the repository, and that proposing a change to a repository they cannot push to needs a classic token with the
  `public_repo` scope because a fork is created. That is taken from GitHub's documentation. **The flow has only been run
  against an in-memory stand-in for the GitHub API, never against github.com**, so scope requirements, rate limits and
  fork timing on the real service are unverified.

## Prompt injection

Anything that came from a repository, a program's output, the web or an issue tracker is wrapped in
`<untrusted source="...">` tags (with any literal closing tag in the content neutralised), and every system prompt says
that such text is data and never an instruction. This lowers the risk; it does not remove it, which is why the controls
above sit in code and not in the prompt. A successful injection could at worst steer the model toward a bad edit
inside the guards, which still has to *improve the measured result* to be kept and which the report shows in full. The
test suite checks that an injected instruction in a README ends up inside an `untrusted` fence in the prompt, and that known-issue
text is fenced the same way and cited as a URL only because a search returned it. Issue titles and bodies are written by
anyone with a GitHub account, so they are as untrusted as the repository. A search result is shown to the person as
"possibly related, not verified"; ReproFix does not decide that an issue describes the failure, and it never follows its
links or runs anything it suggests. It cannot tell you whether a real Nemotron model resists such text; that has not been tested.

## The HTTP server

- **Authentication** is a single shared bearer token (`REPROFIX_API_TOKEN`), compared in constant time. With it unset,
  anyone who can reach the server can make it execute repositories; `reprofix serve` only warns when it is bound to a
  non-loopback address without one. Only `docker-compose.yml` enforces it (Compose refuses to start without the
  variable); with the systemd route you have to set it yourself. `?token=` is accepted because browsers cannot set headers on `EventSource`;
  query strings can end up in access logs, so keep this behind TLS and treat the token as rotated if logs are shared.
  `/api/health` and `/api/benchmark` are open and contain no secrets.
- **Not provided:** per-user accounts, authorisation between users, rate limiting, quotas. Anyone with the token sees
  every run. Put a reverse proxy with TLS in front, and rate-limit there.
- **Repository URLs:** `https` only, no credentials, host on an allow-list (`github.com` by default), path shaped like
  `/<owner>/<repo>`. The clone runs with hooks disabled, `file` protocol disabled, and symlink creation off; any
  remaining symlinks are deleted; there are caps on file count and size. Private repositories are not supported.
- **Local paths** in API requests are off unless `REPROFIX_ALLOW_LOCAL_PATHS=1` (trusted development machines only).
- **The UI** builds all run-derived content with `textContent`; nothing from a run is assigned to `innerHTML`, and
  `javascript:` URLs in evidence are never rendered as links. A jsdom test feeds it hostile log text and graph labels
  and asserts that nothing executes. The page makes no third-party requests (no external fonts or scripts).
- **Data directory:** `REPROFIX_DATA_DIR` holds the SQLite database and every run's workspace, logs, and report,
  including the contents of the repositories that were investigated. It grows without limit; nothing prunes it. Treat it as
  sensitive and clean it on a schedule.
- **Single process:** runs execute on a thread pool inside the server process; a crash interrupts them (they are
  marked `interrupted` on the next start).

## Before you expose it

1. Use the Docker backend. Build the image and run `reprofix doctor`.
2. Set `REPROFIX_API_TOKEN`, bind to loopback, and put TLS and rate limiting in front. Serve over HTTPS only: the API
   token (and, for pull requests, a GitHub token) travels in requests.
3. Keep `REPROFIX_ALLOWED_GIT_HOSTS` as narrow as you can, and leave `REPROFIX_ALLOW_LOCAL_PATHS` and
   `REPROFIX_ALLOW_SDIST` off.
4. Use `REPROFIX_INSTALL_NETWORK=proxy` (the Compose and systemd default) and run `reprofix egress test` on your own
   machine, or decide that the open install network is acceptable for your users. The proxy has not been run under a Docker
   daemon by the author, so treat a passing self-test as the first evidence, not a guarantee.
5. Monitor disk use and prune the data directory.

For a public demo, prefer the bundled offline demo and a curated list of repositories over arbitrary URLs.
