# ReproFix AI

ReproFix takes a broken machine-learning repository, runs it in a sandbox, works out why it fails or why it does
not reproduce its documented result, repairs it, re-runs it to check, and hands back a patch plus a report in which
every claim points at evidence.

It was built for the **Nebius x NVIDIA Global AI Hackathon** (Coding & Agentic Engineering track). The language
model calls go to **NVIDIA Nemotron 3 (Nano / Super / Ultra) on Nebius Token Factory**; web evidence comes from
**Tavily**.

```
reproduce -> diagnose -> repair -> verify -> evidence-backed report
```

The design rule behind everything here: **a language model proposes, code decides.** Whether something is fixed is
decided by running the project and comparing a number parsed from its output with the documented value. No model
judges success, and no model-stated confidence ever changes the status of anything.

Give it the text of a paper (or a README) and it also acts as a **paper-claim checker**: it reads the headline numbers the
text states with fixed patterns (no model), runs the repository, and prints a report card of *claimed vs measured*, before
and after the repair, with the hardware the sandbox could see. See "Check a paper's headline result" below, and read its
limits: a claim is only checked when the program prints the same metric name and number format.

## Status: what is verified and what is not

Read this before the rest. It lists what has actually been run, and what has not.

| Area | State | How it was checked |
|---|---|---|
| Pipeline: clone/copy, install, baseline run, diagnose, patch, re-run, final stability re-run, report | Works | End-to-end tests with a scripted model, and with the real Nebius client talking to a local mock HTTP server |
| Deterministic verification and the keep-or-revert rule for patches | Works | Unit and end-to-end tests |
| Evidence graph, evidence checking, hardcoding guard, protected files | Works | Unit and end-to-end tests |
| Web UI (graph, diff, output, report, live stream, pull-request panel, phone width) | Works | jsdom test (82 checks, including the paper-claims, known-issues and hardware sections) plus screenshots in headless Chromium at desktop and phone width |
| CLI (`run`, `claims`, `demo`, `doctor`, `bench`, `serve`, `pr`, `egress`) | Works | CLI tests; `pip install -e .` verified across both Linux and Windows (`reprofix --version`, `reprofix bench list`, `reprofix bench compare`); automatic Git for Windows `patch.exe` discovery |
| ReproBench: 30 tasks (29 numpy, 1 real PyTorch on CPU), each shown to fail as documented and to pass with its reference fix | Works | `reprofix bench validate` with real `pip`: 30/30 valid, including the three dependency tasks, the demo task and the PyTorch task (torch 2.14.1 from PyPI, about 5 GB installed) |
| 30/30 "oracle" benchmark run | Validates the harness only | A scripted backend replays each task's reference fix. **It measures no model.** It found and fixed two defects while the tasks were added. |
| Open pull request (UI button and `reprofix pr`) | Works against a stand-in | 36 pull-request tests plus API, CLI and UI tests, against an **in-memory fake of the GitHub API**. Binary diff streaming ensures clean hunk application across line-ending conventions. **Never run against github.com**: a live check was refused by the build environment's network policy. |
| Deployment package (`Dockerfile`, `docker-compose.yml`, systemd unit, Caddyfile) | Checked statically only | `docker compose config` and 17 tests on the files. **The image was never built and the stack never started.** See [docs/deployment.md](docs/deployment.md). |
| Paper-claim report card (claimed vs measured, before and after the repair) | Works on small synthetic repositories and program output | `tests/test_claims.py` (extraction, matching, verdicts; end-to-end runs with the scripted backend, the API and the CLI); extraction and matching are plain rules, so the tests exercise the real code. **Not tried on real papers or on real research repositories**: papers phrase results in far more ways than the patterns cover, and the report says when it could not read or match a claim. |
| Known-issues search (GitHub issues of the repository, plus Tavily limited to code-help sites) | Works against mocks only | `tests/test_issues.py`: request building, sanitised queries, ranking, rate-limit stop, odd responses, all against mock transports. **Never called live**: GitHub's real response and rate limits, and Tavily's `include_domains`, are unverified. Results are labelled "possibly related, not verified". |
| Egress allow-list proxy for the install phase (only PyPI reachable while pip runs) | Works over real sockets and as command lines only | `tests/test_egress.py` runs the proxy against local sockets (tunnels an allowed host, refuses others, refuses non-public addresses, refuses plain HTTP); `tests/test_egress_ctl.py` checks the `docker` command lines with a recording fake. **Never run under a Docker daemon**: whether an `--internal` network plus the proxy container really confines a pip install is unverified. Off by default for a plain `reprofix serve`; on in `docker-compose.yml` and the systemd unit. |
| **GPU hardware record** (which GPU the sandbox could see, with verbatim `nvidia-smi` output) | **Not tested on any GPU** | The probe and parser are tested with a fake sandbox and `nvidia-smi` text written from its documented output. No GPU, driver or NVIDIA Container Toolkit was available. See [docs/gpu.md](docs/gpu.md). |
| **Real Nemotron calls and quality** | **Not tested** | No API key was available while building. No success rate for any Nemotron model exists in this repository. |
| **Model routing benefit (Nano/Super/Ultra vs one model)** | **Not measured** | The routing policy is a hypothesis. The harness can measure it; nobody has yet. |
| **Live Tavily call** | **Not tested** | Request building and error handling are tested against a mock transport only. |
| **Docker sandbox execution** | **Not tested** | The `docker run` command line is unit-tested flag by flag. No sandbox command was ever run through a Docker daemon, so the real behaviour of those flags (and of the application image in the deployment package) is unverified. |
| NVIDIA OpenShell / NemoClaw | Not integrated | See "Deviations". |
| **A paper's result reproduced on a real NVIDIA GPU** | **Not done** | No GPU was available. There are no GPU tasks; the one real PyTorch task runs on CPU and the "device" task is a CPU-side simulation. The GPU record would show that the sandbox could see a GPU, not that the code used it. |

There are deliberately **no accuracy, cost or latency figures** in this README. The only benchmark numbers committed
(`benchmark/results/oracle-router.*`) are labelled as scripted, not Nemotron, in the file itself and in the UI.
Run the benchmark with your own key to get real ones (see below).

## Quickstart

Requires Python 3.11+, `git`, and `patch`.

```bash
pip install -e '.[dev]'
cp .env.example .env            # then fill in NEBIUS_API_KEY (and TAVILY_API_KEY if you have one)
```

`.env` in the current directory is loaded automatically. Variables already set in the real environment win.

### 1. Try it with no API key (offline demo)

A scripted stand-in replays the reference fix for a bundled repository with three independent faults (a dependency
conflict, a classifier dimension mismatch, wrong input normalisation). It exercises the whole pipeline and is **not
Nemotron**; the report, the UI and the CLI all say so.

```bash
# Linux / macOS
REPROFIX_SANDBOX=local REPROFIX_ALLOW_UNSAFE_LOCAL=1 reprofix demo

# Windows (PowerShell)
$env:REPROFIX_SANDBOX="local"; $env:REPROFIX_ALLOW_UNSAFE_LOCAL="1"; reprofix demo
```

The `local` backend has no filesystem isolation; it is for development on your own machine. The install step needs
access to PyPI. On Windows, `patch.exe` from Git for Windows (`C:\Program Files\Git\usr\bin`) is discovered automatically.

### 2. Check your setup

```bash
reprofix doctor           # key present? base URL reachable? are the three model IDs served to your key? sandbox ok?
reprofix doctor --live    # also one tiny real call per model and one Tavily search (spends a little credit)
```

Nebius's own pages disagree about two things, so `doctor` checks them against your key instead of assuming:
the base URL (`https://api.tokenfactory.nebius.com/v1/` in the quickstart, a `us-central1` host in the cookbook) and
the spelling of the Nano model ID. ReproFix resolves a configured ID against `GET /models` ignoring case and
punctuation and never substitutes a different model.

### 3. Run on a repository

```bash
reprofix run --repo https://github.com/<owner>/<repo> --metric-name val_accuracy --expected 0.881 --tolerance 0.02
```

Without `--expected`, ReproFix only checks that the project runs, and the report says reproduction of a result was
not checked. Without `--command`, the run command comes from a README code block, then from the model (validated; the
script must exist), then from a detected entry point. Only `python`, `python3` and `pytest` commands are accepted and
they never go through a shell.

### 4. Check a paper's headline result

Give ReproFix the numbers a paper claims and it reports, claim by claim, whether the repository's own output reproduces
them, before and after the repair.

```bash
reprofix claims paper.txt                  # shows what would be read from the text (and what was skipped); runs nothing
reprofix run --repo https://github.com/<owner>/<repo> --paper paper.txt
reprofix run --repo https://github.com/<owner>/<repo> --claim "accuracy=76.4%" --claim "bleu=27.3±0.3"
```

In the web UI it is the "Check a paper's headline result" section of the start form (claims one per line, and/or pasted
paper text). The report's **Paper claims** card lists, for each claim, the claimed value and tolerance, the value measured
before the repair, the value measured after it, and the change: `fixed`, `reproduced`, `regressed`, `not_reproduced` or
`not_measured`. It also names the hardware the sandbox could see (see [docs/gpu.md](docs/gpu.md)).

How it works, and where it stops:

- **Reading the text is rules, not a model.** Fixed patterns find "accuracy of 76.4%" and "76.4% accuracy", and rows of a
  results table whose column headers name metrics (only the row labelled "ours", or a table with a single row). A number
  that follows a comparison ("than 71.0%", "outperforms X (accuracy of 71.0%)") is recorded with a note and is never chosen
  as the headline (if every claim follows a comparison, none is, and the report says so). Differences and bounds are not
  claims at all and are listed as skipped: "by 2.1% accuracy", "less than 1% accuracy loss", "2.3% lower than", ranges such as
  "74-76%", "recall at 10", and table columns such as "Δ Acc". A table with several rows labelled "ours" is skipped rather
  than guessed. "Balanced accuracy" and "worst-group accuracy" keep their modifier and so do not match a plain `accuracy` line.
  At most 20 claims are read. **The paper text and the quotes are never sent to a model**; the headline claim's
  metric name and number reach the models the way an `--expected` value does. `reprofix claims` and the report both show
  what was read, because a pattern can misread a sentence.
- **Measuring is matching names.** A claim is compared with a `name: number` or `name = number` line in the program's
  stdout. The last value printed under a name counts, and `nan`, `inf` or `N/A` count as values: a run that ends in
  `accuracy: nan` measured nothing. A run that exits non-zero or times out measures nothing. A printed name must contain
  every word of the claimed one and may add only words that do not make it another quantity: `cifar10_accuracy` is accuracy,
  but `accuracy_gap`, `worst_group_accuracy`, `balanced_accuracy` and `accuracy@5` are not. If no output fits, or two fit
  equally, the verdict is `not_measured` with the reason; ReproFix does not guess which number was meant. A claim without a
  split (for example "accuracy") ignores outputs labelled "train".
- **Scale and tolerance.** A `%` sign makes a percentage. Without one, a printed number up to 1 is a fraction and one
  between 1 and 100 a percentage, for accuracy-like metrics (so 0.764 and 76.4 both mean 76.4%); loss, perplexity and other
  unbounded metrics are compared as printed. The default tolerance is ±1 percentage point (±0.01 for a fraction, 1% of the
  value for the rest), or what the claim states (`--claim "accuracy=76.4%±0.5"`). **That default is a chosen number, not a
  measured property of any benchmark**: on a GPU, runs differ from each other by amounts that depend on the code.
- **What gets repaired.** The headline claim (an explicit one, else the first read from the text in the authors' voice) is the
  number the repair loop works toward, unless you pass `--metric-name`/`--expected`, which win. "Before" is the baseline run;
  "after" is the state with the kept patches, so an experiment that was reverted is never measured.
- **What "reproduced" means.** The program printed a number within tolerance of the paper's. It does not show that the
  repository implements the paper's method, uses the paper's data or seed, or computes the metric honestly (the
  hardcoding guard checks patches, not the original code). Treat it as a prompt to look, with the output attached. Error rates,
  WER and CER are compared as printed (not rescaled), so `error rate 23.6` against a printed `error: 0.236` shows `not_reproduced`.

**Known issues.** When a run is not already passing, ReproFix searches the repository's own GitHub issues and (with a Tavily
key) a short list of code-help sites for the failure it observed, and lists what it finds in the report as *possibly
related, not verified*. The same text goes to the diagnoser as untrusted evidence. The query is built from identifiers in
the failure with paths and search operators removed, and it does leave your machine. `REPROFIX_KNOWN_ISSUES=0` turns it
off. Unauthenticated GitHub search is rate limited; when GitHub says stop, the search stops for the rest of the run.
**Never called live.** See [docs/security.md](docs/security.md).

### 5. Web UI

```bash
# Default (port 8000)
reprofix serve --port 8000          # http://127.0.0.1:8000

# Custom port (e.g. 1800) and network interface
reprofix serve --host 0.0.0.0 --port 1800

# With local development sandbox (no Docker daemon required):
REPROFIX_SANDBOX=local REPROFIX_ALLOW_UNSAFE_LOCAL=1 reprofix serve --port 1800
# (PowerShell: $env:REPROFIX_SANDBOX="local"; $env:REPROFIX_ALLOW_UNSAFE_LOCAL="1"; reprofix serve --port 1800)
```

Start a run from the form, or press the offline-demo button. The page shows a progress rail, a measurement
instrument (each run's metric against the documented tolerance band), the live evidence graph, the diff, every
command's output, and the report. When listening on a non-loopback address, set `REPROFIX_API_TOKEN`
(see [docs/security.md](docs/security.md)).

### 6. Open a pull request

When a run ends `verified`, `executes` or `partial` on a cloned GitHub repository, the report's **Changes** tab offers
**Open pull request**; from a terminal:

```bash
reprofix pr <run_dir> --dry-run              # show the target, files, title and description; no network, no token
GITHUB_TOKEN=... reprofix pr <run_dir>       # asks for confirmation, then opens it
```

It rebuilds the changed files from the stored diff and the original snapshot, pushes a branch (to a fork if your token
cannot push to the repository), and opens a pull request whose description is drafted from the report's computed checks,
with the model's diagnosis labelled as the model's words. Your token is used for that one request and never stored. The
server (the UI button) opens at most one pull request per run; `reprofix pr` keeps no such record, so running it twice
opens two. Read "Pull requests" in [docs/security.md](docs/security.md) first. **This has only been run against an
in-memory fake of the GitHub API.**

### 7. Sandbox image (default backend)

```bash
docker build -f docker/sandbox.Dockerfile -t reprofix-sandbox:latest .
```

The default backend is Docker. `reprofix doctor` reports whether the daemon and the image are present. For PyTorch
or CUDA repositories there is a documented `--build-arg BASE=...` path and `REPROFIX_SANDBOX_GPU=1`; that path has not
been run on a GPU.

**On a GPU machine.** `REPROFIX_SANDBOX_GPU=1` gives the run phase `--gpus all`; at the start of a run ReproFix asks the
sandbox which hardware it can see and puts the verbatim `nvidia-smi` output in the report. That shows what the sandbox could
see, not that the repository's code used it. [docs/gpu.md](docs/gpu.md) has the Nebius GPU-VM setup (copied from Nebius's
and NVIDIA's documentation, **not run**).

**Only PyPI during install.** By default the install phase has an open network. With the egress proxy, installs run on a
Docker `--internal` network and can reach only the hosts in `REPROFIX_EGRESS_ALLOW` (PyPI by default) through a small
allow-list proxy:

```bash
reprofix egress up          # create the internal network and start the proxy container
REPROFIX_INSTALL_NETWORK=proxy reprofix serve
reprofix egress test        # from a throw-away container on that network: a direct connection to 1.1.1.1 must fail,
                            # a host that is not on the list must be refused, and PyPI must be reachable through the proxy
```

Installs are refused (not silently opened) if the proxy is not ready. `docker-compose.yml` and the systemd unit turn this
on. **It has not been run under a Docker daemon**; run `reprofix egress test` before trusting it. See
[docs/security.md](docs/security.md).

## How verification works

- **Stages are ordered:** `install_failure` < `crash` < `wrong_result` < `verified`.
- **The metric is parsed by regex from stdout**, never by a model. "Reproduced" means
  `|value - expected| <= tolerance` (inclusive, with a `1e-9` allowance for floating point).
- **Tests count only if they ran.** If `pytest` is not in the repository's own requirements, the report says
  "tests not run" rather than "failed".
- **A patch is kept only if it makes progress** (a later stage, a metric gap that shrank by at least 0.005, tests that
  now pass, or a different remaining failure at the same stage). Otherwise it is reverted and the hypothesis is
  marked rejected.
- **Final clean re-run.** A result that verified is run once more; the report records whether the two agree.
- **Edits are exact, unique search/replace pairs.** A diff is produced from the real files, and the report checks
  that `patch -p1` applies it to the original repository.
- **Guards:** `tests/**`, `test_*.py`, `README*` and `LICENSE*` cannot be edited; a patch that writes the expected
  number into code that prints a metric is refused; at most 5 files per patch.
- **Evidence is checked.** A quoted code or log snippet must occur verbatim in the cited file or log; an external URL
  must have been returned by a search in the same run (Tavily, or the known-issue search). Anything else is shown as unverified. A model's
  stated confidence is displayed as "model-stated confidence (not verified)".

## Model routing

Policy v1 (a hypothesis, not a result):

| Purpose | Tier |
|---|---|
| summarise the repository, digest logs, classify | Nano |
| plan, diagnose a crash, write the repair, review the patch | Super |
| diagnose when nothing crashed (wrong result), and escalate after an experiment rejected a hypothesis | Ultra |

If a tier is unavailable the router steps down first and then up. `--router-mode super-only` and `ultra-only` pin one
tier, which is how the policy is meant to be compared with fixed baselines.

Defaults, from Nebius's model catalog on 2026-10-02 (prices change; override them in `.env`):

| Tier | Model ID | USD per 1M tokens (in / out) |
|---|---|---|
| Nano | `nvidia/nvidia-nemotron-3-nano-30b-a3b` | 0.06 / 0.24 |
| Super | `nvidia/nemotron-3-super-120b-a12b` | 0.30 / 0.90 |
| Ultra | `nvidia/Nemotron-3-Ultra-550b-a55b` | 1.00 / 3.00 |

## ReproBench

Thirty small repositories, each with one deliberate fault (one has three), across dependency, configuration,
architecture, checkpoint, code, data, preprocessing, training, evaluation and (simulated) device faults. Twenty-nine use
only numpy and pytest: 25 of them run entirely offline in the test suite, and four need PyPI for their `pip install`
(the three dependency tasks and the demo). One is a real PyTorch project (CPU, a multi-gigabyte install). Each is checked
to fail as documented and to pass with its reference fix. Full list and scoring in [docs/benchmark.md](docs/benchmark.md).

```bash
reprofix bench list
reprofix bench validate --only dep_conflicting_pins          # prove tasks are well-formed (no model involved)
reprofix bench run --llm nebius --router-mode router         # real run: needs NEBIUS_API_KEY
reprofix bench run --llm nebius --router-mode super-only
reprofix bench run --llm nebius --router-mode ultra-only
reprofix bench compare benchmark/results/nebius-router.json benchmark/results/nebius-super-only.json
```

`--llm oracle` runs the harness with scripted answers. It passes 30/30 by construction and says nothing about any
model.

## Deviations from the original brief

These are choices, made so that everything claimed could be built and tested in one place.

- **SQLite + a thread pool** instead of Postgres + Redis. One process, one host.
- **Vanilla JavaScript UI** (no build step) instead of Next.js. It is served by the FastAPI app.
- **Numpy-based benchmark tasks** (29 of 30) instead of PyTorch/CIFAR, so they are cheap to run on CPU. The one PyTorch
  task is CPU-only on the same synthetic data. There are no GPU tasks; the "device" task simulates a config/hardware
  mismatch without any GPU or CUDA.
- **30 tasks**, the top of the brief's 20-30.
- **Pull requests are opened with the user's own GitHub token**, and the patch is still downloadable. The pull-request
  flow was built and tested against a fake GitHub API only.
- **OpenShell / NemoClaw are not integrated.** Nvidia describes NemoClaw as an early-preview reference stack for a
  trusted operator on one host, and no stable execution API for it was found. The sandbox is a small interface
  (`reprofix/sandbox/base.py`) with Docker and a development-only local backend; another backend can be added behind it.
- **The brief's illustrative numbers are not reproduced:** the router-vs-Ultra table (82% / 80% success, $1.00 /
  $0.38, 100% / 61% latency), the 29/30 ... 21/30 benchmark funnel, and the 62.3% to 85.7% repair example. The brief
  itself says to use measured numbers, not invented ones, and nothing here has been measured against a real model.

## Limitations

- Synthetic tasks (and one small PyTorch project) are far easier than real research repositories. A good score here would
  not predict one there.
- `root_cause` in the benchmark is a proxy: a confirmed hypothesis names the right file(s) and category.
- Dependency installs use wheels only by default, and a repository's requirements files are checked first: an editable
  install (`-e .`), a local path, a direct URL, `--no-binary` and similar lines would make pip build and run code, so the
  install is refused with the line and the reason (see `reprofix/sandbox/reqcheck.py`). `REPROFIX_ALLOW_SDIST=1` lifts all of
  that and allows source builds, which run
  `setup.py` while the network is open).
- On plain Docker the *install* phase has an open network unless the egress proxy is switched on (`REPROFIX_INSTALL_NETWORK=proxy`,
  which `docker-compose.yml` and the systemd unit do); the proxy has not been run under a Docker daemon. The *run* phase has
  no network. See [docs/security.md](docs/security.md).
- A paper claim is checked only when the program prints that metric as `name: number` in a form the name matcher recognises.
  Claims in prose or tables that the patterns do not cover are not read at all, and the report says what was skipped.
- Known-issue search finds issues by word overlap. It cannot tell whether an issue describes your failure.
- One server process, a single trust domain, no per-user accounts, no rate limiting.

## Tests

```bash
python -m pytest tests -q                                    # no network, no Docker daemon, no API key (797+ tests, 0 failures)
REPROFIX_TEST_NETWORK=1 python -m pytest tests -q            # also real pip installs and a live GitHub clone
REPROFIX_TEST_NETWORK=1 REPROFIX_TEST_TORCH=1 python -m pytest tests/test_benchmark.py -k pytorch   # downloads ~3 GB, needs ~6 GB disk
cd tests/ui && npm install && UI_TEST_BASE=http://127.0.0.1:8000 npm test   # against a running `reprofix serve`
```

The test suite runs hermetically across both Linux and Windows. The UI test needs a server started with the offline demo available (any `reprofix serve` with the local sandbox:
`REPROFIX_SANDBOX=local REPROFIX_ALLOW_UNSAFE_LOCAL=1`).

## Layout

```
reprofix/
  core/        orchestrator, verification, patching, repo acquisition, evidence graph, report, known-issue search (issues.py)
  claims/      paper claims: rule-based extraction from text, name matching, tolerance and verdicts, the report card
  agents/      analyzer, planner, diagnoser (read-only tools), repairer, reviewer
  inference/   Nebius Token Factory client, tiered router, Tavily client, usage/cost accounting
  sandbox/     interface, Docker backend, development-only local backend, GPU probe (hardware.py),
               install-phase egress proxy (egress.py) and its Docker control (egress_ctl.py)
  api/         FastAPI app, SQLite store with replayable SSE, run manager
  evaluation/  ReproBench harness, scripted/oracle backends
web/           the UI (index.html, app.js, style.css)
benchmark/     30 tasks, their generator, and the committed (oracle) results
docker/        sandbox image
Dockerfile, docker-compose.yml, deploy/   application image, compose stack, systemd unit, Caddyfile (see docs/deployment.md)
docs/          architecture, security, benchmark, deployment, gpu, submission
tests/         pytest suite and the jsdom UI test
```

More detail: [docs/architecture.md](docs/architecture.md), [docs/security.md](docs/security.md),
[docs/benchmark.md](docs/benchmark.md), [docs/deployment.md](docs/deployment.md), [docs/gpu.md](docs/gpu.md),
[docs/submission.md](docs/submission.md).

## License

MIT (see `LICENSE`).

## Sources

- Nebius x NVIDIA Global AI Hackathon rules and submission requirements (Devpost)
- Nebius Token Factory: quickstart, "switch from OpenAI" guide, cookbook, and model catalog (`tokenfactory.nebius.com/model-catalog.md`)
- NVIDIA NemoClaw: overview and architecture documentation
- Tavily Search API documentation (`POST https://api.tavily.com/search`)
- Nebius AI Cloud Compute documentation (quickstart, virtual machines, preemptible VMs) and NVIDIA Container Toolkit installation guide, for [docs/gpu.md](docs/gpu.md)
