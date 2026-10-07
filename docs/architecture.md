# Architecture

ReproFix is one Python package plus a static web UI. A run is a single deterministic loop in which language models
are asked for *proposals* (what is the command, what might be wrong, what edit to try) and code decides *outcomes*
(did it run, is the number inside the band, did the patch apply, is the quoted evidence really there).

```
            web UI (vanilla JS)                  CLI (reprofix run / claims / bench / demo / doctor / pr / egress / serve)
                   |  REST + Server-Sent Events                   |
                   v                                              |
            FastAPI app -- SQLite (runs, events, pull_requests) --  |
                   |                                              |
              RunManager (thread pool)                            |
                   |                                              |
                   +------------------+---------------------------+
                                      v
                               Orchestrator
        acquire -> analyse -> plan -> baseline -> [diagnose -> repair -> experiment]* -> final re-run -> report
            |         |         |         |              |           |          |
         core/repo  agents/   agents/  Sandbox       agents/     agents/    core/verify
                    analyzer  planner  (install,     diagnoser   repairer   core/patching
                                        run)          + ToolBox              core/evidence
                                      v
                  ModelRouter -> ChatBackend (NebiusClient | scripted/oracle)   TavilyClient
```

## Modules

| Path | Responsibility |
|---|---|
| `core/orchestrator.py` | The loop and its budgets (attempts, wall time, tokens), cancellation, event emission. |
| `core/repo.py` | Clone (https, host allow-list, no hooks, no symlinks, size/file limits) or copy a repository into a run workspace; deterministic scan (files, requirements, entry points, framework, README claims and commands). |
| `core/verify.py` | Failure extraction from tracebacks, metric parsing, verdicts, progress assessment. No model involved. |
| `core/patching.py` | Exact search/replace edits, atomic apply and revert, protected paths, hardcoding heuristic, unified diff. |
| `core/evidence.py` | The evidence graph. |
| `core/report.py` | The report, with its checks computed from the final re-run and the real diff. |
| `core/issues.py` | Known-issue search: builds a sanitised query from the observed failure, asks GitHub's issue search (scoped to the repository, unauthenticated) and Tavily (code-help sites only), ranks by word overlap, and keeps what it found for the report and, as untrusted text, for the diagnoser. |
| `claims/` | Paper claims. `extract.py` reads numbers from pasted text with fixed patterns; `names.py` normalises metric names; `check.py` matches a claim to a `name: number` line in the program's output and judges it against a tolerance; `__init__.py` collects claims, picks the headline, and builds the report card. No model anywhere in the package. |
| `core/pullrequest.py` | Turns a finished run into a GitHub pull request: rebuilds the changed files from the stored diff and the original snapshot (not from the workspace), drafts the description from the report's computed checks, and drives the GitHub REST calls (fork if needed, branch, one commit per file, pull request). |
| `agents/analyzer.py` | Run command, task description and expected metric (README text and pattern matching first, a Nano call to summarise). |
| `agents/planner.py` | Ranked suspect areas. Advisory; the pipeline works without it. |
| `agents/diagnoser.py` | Read-only investigation, then ranked hypotheses with evidence that code verifies. |
| `agents/repairer.py` | Exact edits for one chosen hypothesis. |
| `agents/reviewer.py` | An advisory model review of the final diff. It never gates verification. |
| `agents/tools.py` | Read-only tools the diagnoser can call: `read_file`, `grep`, `list_dir`, `search_web`. |
| `inference/client.py` | Nebius Token Factory client (OpenAI-compatible `/chat/completions` and `/models`). |
| `inference/router.py` | Chooses a Nemotron tier per purpose and falls back when a tier is unavailable. |
| `inference/tavily.py` | Tavily search with query sanitising. |
| `inference/base.py` | `ChatBackend` protocol, token and cost accounting. |
| `sandbox/` | `Sandbox` interface; `reqcheck.py` (refuses requirements files that would make a wheels-only install build and run code); Docker backend (default); development-only local backend. `hardware.py` runs and parses the GPU probe (`nvidia-smi`). `egress.py` is the install-phase allow-list proxy (a standalone, standard-library-only file that also runs inside its own container); `egress_ctl.py` creates and checks its Docker network and container and builds the command lines. |
| `api/` | FastAPI routes, SQLite store (runs, events, and one `pull_requests` row per opened pull request), thread-pool run manager. |
| `evaluation/` | ReproBench harness and the scripted/oracle backends. |

## Life of a run

1. **Acquire.** Clone or copy into `runs/<id>/work`, snapshot to `orig` (used for the final diff, to check the
   patch applies, and later as the base from which a pull request's files are rebuilt).
2. **Analyse.** Scan the repository, determine the run command (explicit command, then a README code block, then a
   validated model suggestion, then an entry point), and the expected metric (explicit, then the headline paper claim if
   claims were given, then a number the model proposes only if it really appears in the README, then a README pattern
   match). Claims are collected here (event `claims`): explicit ones plus whatever the patterns read from the pasted text.
3. **Plan.** Advisory ranking of suspect areas.
4. **Baseline.** First ask the sandbox which hardware the run phase can see (event `hardware`; one throw-away `nvidia-smi`
   container when `REPROFIX_SANDBOX_GPU=1`, nothing started otherwise). Then install requirements if there are any
   (wheels only, after the requirements files have been checked for lines that would build code; open bridge, or the
   egress-confined network in proxy mode), run the command (network off), run `pytest -q`
   if tests exist. Compute a verdict: `install_failure`, `crash`, `wrong_result` or `verified`.
5. **Repair loop**, up to `max_attempts` and within the time and token budgets:
   1. Build an observation from the failure: exception, deepest in-repo frame, log tail, or the metric gap.
   2. Optionally run a Tavily search first when the error text looks environmental (CUDA, resolver conflicts, ABI).
      Then search for known issues (event `known_issues`): the repository's GitHub issues and, with a Tavily key, a few
      code-help sites, using identifiers from the failure. What it finds goes to the diagnoser as untrusted text and into the
      report as "possibly related, not verified"; a URL it returned may be cited as evidence, as with any search.
   3. The diagnoser inspects the repository with the read-only tools, then proposes up to four hypotheses.
      Evidence is verified by code; if the leading hypothesis has none verified, one grounding retry is made.
   4. The orchestrator picks a hypothesis. A near-duplicate of one already rejected is skipped; if every candidate
      is a repeat, the run stops with that reason.
   5. The repairer returns exact edits; the guards run (hardcoding heuristic, protected paths, uniqueness of every
      `search`); the edits are applied atomically.
   6. Re-run. If the verdict made progress the patch is kept and the hypothesis is `confirmed`; otherwise the patch is
      reverted and the hypothesis is `rejected`. Each consecutive rejection raises the tier used for the next
      diagnosis and repair by one, up to two tiers above the purpose's default; a kept patch resets it.
6. **Final check.** If the last verdict was `verified`, run everything once more from a clean state. The report records
   whether it agreed.
7. **Report.** Seven checks computed in code, the diff, the evidence graph, every execution, usage and cost, the
   routing decisions, the paper-claims card (`claims`), the hardware record (`hardware`), the known issues
   (`known_issues`), and caveats (development sandbox, non-Nemotron backend, inferred metric, unpriced tiers, a GPU that was
   asked for and not seen, claims that could not be matched).

Run statuses: `verified`, `executes` (ran, but no expected metric was given), `already_passing`, `partial` (progress but
not reproduced), `failed`, `cancelled`, `error`, and `interrupted` (the server restarted mid-run).

## The model protocol

Models are called through plain chat completions. Each agent asks for **one JSON object in text**; ReproFix extracts it
(tolerating `<think>` blocks, code fences and surrounding prose) and retries once if none parses. This avoids depending
on provider-specific tool-calling or structured-output features whose behaviour on a given model was not verified.

- Each system prompt starts with `REPROFIX_TASK=<purpose>`; the same string identifies the purpose in logs and tests.
- Anything originating in the repository, program output or the web is wrapped in
  `<untrusted source="...">...</untrusted>`, with a closing-tag-in-content escape, and the system prompt tells the
  model to treat it as data.
- Model output never executes anything by itself: commands are parsed (`python`/`python3`/`pytest` only, no shell
  metacharacters), edits pass through the patch guards, and evidence is checked against files and logs.

## Model routing

`ModelRouter.pick(purpose, escalate)` maps a purpose to a tier (see the README table), adds the escalation level, and
caps at Ultra. `diagnose_behavioral` is used when the failure is a wrong result or a test failure rather than a crash.
`router_mode` can pin everything to Super or Ultra. The decisions (purpose, wanted tier, used tier, model) are recorded
in the report so a benchmark run shows exactly what was routed where.

## Evidence graph

| Node type | Meaning |
|---|---|
| `symptom` | A failure state: install failure, crash, or a result outside the band. |
| `hypothesis` | A candidate cause with category, files, and the model's stated confidence. |
| `evidence` | A quote from a file or a log, or a URL from a search. `verified` or `unverified` by code. |
| `experiment` | An applied edit set. |
| `result` | What the re-run showed, and whether the patch was kept. |

Relations: `symptom -explained_by-> hypothesis`, `hypothesis -supported_by-> evidence`,
`hypothesis -tested_by-> experiment`, `experiment -produced-> result`, `result -remaining_symptom-> symptom`.

Statuses are set only by code: `open`, `untested`, `confirmed`, `rejected`, `verified`, `unverified`, `info`.
`model_confidence` is stored as data and is shown with the label "Model-stated confidence (not verified)".

## Events and streaming

Every event is stored in SQLite with a per-run sequence number, so a browser that reconnects can resume with
`Last-Event-ID` (or `?after=`) and a finished run replays identically.

`run.started`, `acquired`, `analysis`, `claims`, `step`, `plan`, `hardware`, `exec.start`, `exec`, `verdict`, `graph`,
`known_issues`, `diagnosis`, `hypothesis.selected`, `edit.proposed`, `experiment`, `attempt.failed`, `usage`, `report`,
`run.finished`, `error`, and a final `end` marker on the stream.

## HTTP API

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Version, whether a key is configured, models, sandbox status, limits, whether the demo is available, whether pull requests are enabled (`pull_requests`) and whether an API token is required (`auth_required`). |
| `POST /api/runs` | Start a run (`repo_url`, `goal`, `metric`, `command`, `max_attempts`, `router_mode`, `paper_text`, `claims`, ...). Server-side ceilings clamp every budget. |
| `POST /api/demo` | Start the bundled demo with the scripted backend. |
| `GET /api/runs`, `GET /api/runs/{id}` | List runs, or fetch one with its report. |
| `GET /api/runs/{id}/events` | Server-Sent Events, resumable. |
| `GET /api/runs/{id}/graph`, `/patch` | The evidence graph; the patch as a download. |
| `POST /api/runs/{id}/cancel` | Cancel a run. |
| `GET /api/runs/{id}/pull-request` | Preview of the pull request this run could open (target, base commit, files, drafted title and body), or why it cannot. No network access. |
| `POST /api/runs/{id}/pull-request` | Open it on GitHub with the caller's own token (`token`, `confirm: true`, optional `title`, `body`, `draft`). At most one per run. 403 if `REPROFIX_ALLOW_PULL_REQUESTS=0`. |
| `GET /api/benchmark` | Task count and any committed result files. |

With `REPROFIX_API_TOKEN` set, everything under `/api/runs*` and `/api/demo` needs `Authorization: Bearer <token>`
(`?token=` is also accepted because the browser's `EventSource` cannot set headers). `/api/health` and
`/api/benchmark` stay open; `/api/health` reports the configured models and base URL (never keys).

## Paper claims

```
paper text ──extract (fixed patterns)──┐
explicit claims (--claim, UI lines) ───┴─> ClaimSet ─ headline ─> MetricSpec(claim=…)  (repair target, unless the user gave a metric)
                                                │
program stdout ── parse_outputs (`name: number`) ┴─> check_claims(baseline run, final run) ─> report card
```

- A `PaperClaim` has a metric name, a value, a unit (`percent`, `fraction`, `raw`), a tolerance, an optional split
  qualifier (`test`, `val`, `train`), the quote it came from, and a note when the number follows a comparison word.
- The matcher is the same code for the repair loop and the report card (`verify.parse_metric` calls `claims.check` when the
  metric carries a claim), so the number the loop optimises is the number the card shows.
- A claim is judged on two runs: the **baseline**, and the **final** state, which is the last run of the adopted code.
  Experiments that were reverted are never measured, and a run that exited non-zero measures nothing. The verdict is
  `reproduced`, `not_reproduced` or `not_measured` (with the reason: no matching output, two equally good matches, no run);
  the change is `fixed`, `reproduced`, `regressed`, `not_reproduced` or `not_measured`.
- Nothing in `claims/` calls a model, and no agent receives `paper_text`, a claim's quote or the card. The agents see only the
  goal and, through the metric, a name and a number.

## Sandbox contract

`install(workdir, requirement_files, timeout)` installs into `workdir/.deps` with the network on (the Docker backend, in proxy
mode, puts the container on an `--internal` network whose only exit is the allow-list proxy, and refuses to install if that
is not so). `run(workdir, argv, timeout)` runs with the network off. `probe_hardware()` reports what hardware the run phase
can see (a record with a status and the raw `nvidia-smi` output, never an inference). The interpreter is started as `python -S` with `PYTHONPATH=.deps`, so only the
repository's own requirements are importable: a missing requirement fails here the way it would on a clean machine
instead of being masked by whatever is installed on the host. Output is captured with the middle truncated
(tracebacks live at the end). API keys are never placed in a sandbox environment.

## Extending

- **Another sandbox** (for example an OpenShell sandbox): implement `Sandbox` (`policy`, `install`, `run`) and return it
  from `make_sandbox`. Nothing else in the pipeline knows which backend is in use.
- **Another model provider:** implement `ChatBackend.complete(tier=..., messages=..., purpose=..., ...)`.
- **A new benchmark task:** the tasks are generated by `benchmark/build_tasks.py`. Add an entry to `BUGS`, run
  `python benchmark/build_tasks.py --measure --only <id>` to see what the broken state really does, then
  `python benchmark/build_tasks.py` to write it (it only adds missing tasks and checks that the committed ones are
  unchanged), and `reprofix bench validate --only <id>`. `--rebuild` wipes and regenerates everything and so invalidates
  any results already published against the old tasks.
