# Submission guide

A working checklist for the Nebius x NVIDIA Global AI Hackathon (Coding & Agentic Engineering track), based on the rules
page as read on 2026-10-02. Re-read the [official rules](https://nebiusglobalaihackathon.devpost.com/rules) before
you submit; this page can go stale.

## Dates

- Submission period: **Wednesday, August 26, 2026, 9:00 am PT, to Friday, October 30, 2026, 10:00 am PT.**
- When this was written (2026-10-02) that was 28 days away. Plan to submit at least a day early; Devpost
  deadlines are exact and video processing on YouTube takes time.

## What the rules require

| Requirement (from the rules) | Status in this repository |
|---|---|
| A working application that **runs on Nebius Token Factory or Nebius AI Cloud**. The rules define this as making a runtime call to the Token Factory inference API, or being deployed or run on AI Cloud compute. | Implemented (`reprofix/inference/client.py`). **Never exercised against the real API**: no key was available while building. You must do a real run. |
| Uses **at least one NVIDIA open-source model**. | Nemotron 3 Nano / Super / Ultra are the default tiers. Same caveat. |
| A **public repository** on GitHub, GitLab or Bitbucket with an **open-source license file** that is visible at the top of the repository page. | `LICENSE` (MIT) is present. You still have to publish the repository. |
| A **demo video under 3 minutes**, uploaded and **publicly visible on YouTube**. | Not made. See the script below. |
| A **URL to a working demo, hosted application, or test build.** | Not provided. See "A demo URL". |
| The project must be new, or significantly updated after the period began. | State accurately in the writeup what was built when. |
| *Optional prize:* Best Use of Tavily ($3,000) needs a **functional runtime call to the Tavily API**. | Implemented (`reprofix/inference/tavily.py`, called during diagnosis when the error looks environmental, and by the model's `search_web` tool). **Never called live.** `reprofix doctor --live` makes one real call. |

Judging uses four equally weighted criteria: Technological Implementation, Design, Potential Impact, Quality of the Idea.

## Before anything else: run it against the real models

Everything about model behaviour in this repository is untested, so spend the first days finding out what a real call does.

1. `cp .env.example .env`, add `NEBIUS_API_KEY` (and `TAVILY_API_KEY`).
2. `reprofix doctor --live`. It confirms the base URL, that all three model IDs are served to your key, that each
   answers a chat request, and that Tavily answers a search. Fix whatever it reports (it also tells you how the Nano
   ID is spelled for your account).
3. `reprofix run --repo <a small broken repo> --expected <documented value>` and read the whole event log. Expect to
   iterate: reasoning models may wrap output in `<think>` blocks (handled), put prose around the JSON (handled), or
   need a larger `max_tokens`, a different temperature, or an `NEBIUS_EXTRA_BODY` setting once you confirm the
   parameter name your model uses. The JSON-in-text protocol was chosen so that none of this depends on a provider's
   tool-calling support, but how well each Nemotron tier follows it is exactly what is unmeasured.
4. Build the sandbox image and run again with the Docker backend (it has not been executed here):
   `docker build -f docker/sandbox.Dockerfile -t reprofix-sandbox:latest .`
   Then `reprofix egress up` and `reprofix egress test` (the install-phase proxy has never run under Docker), and run
   the offline demo with `REPROFIX_INSTALL_NETWORK=proxy`.
5. If you want to show a paper claim checked on a GPU, follow [gpu.md](gpu.md) on a Nebius GPU VM and run it there. Nothing in
   this repository has run on a GPU, so the claim "reproduced on a real NVIDIA GPU" is yours to earn: pick one small
   repository whose paper states a number, have it print `name: number` and the device it used, run it, and keep the report.
6. Only then run the benchmark and the ablations (see [benchmark.md](benchmark.md)). Repeat each routing mode more than
   once; one pass over 30 tasks is a small, noisy sample.

## A demo URL

The rules ask for a working demo, hosted application, or test build. Options, from least to most effort:

- **A test build:** the public repository with the quickstart in the README. `reprofix demo` runs with no key at all.
  Whether judges accept this as the "test build" is their call; the rules do not say.
- **A hosted instance:** [deployment.md](deployment.md) gives two routes on a Linux VM (Nebius AI Cloud would also
  satisfy the "runs on Nebius" requirement through compute): a Docker Compose stack with the app image, and a systemd
  unit with only the sandboxes in containers. Compose refuses to start without `REPROFIX_API_TOKEN`; with systemd you must
  set it yourself, and either way a public address needs TLS. **Neither has been run:** the compose file was validated with `docker compose config` and static tests, but the image was never
  built and the stack never started, so budget time for the first-start checklist in that guide. Read
  [security.md](security.md) first: a public instance that runs arbitrary repositories is a hazard, and the Docker socket
  it needs is root on the VM. For a public demo prefer the bundled offline demo, or a fixed list of repositories you have
  already tried, over arbitrary URLs. If judges are given the API token, anyone holding it can spend your Nebius credit.

## The 3-minute video

Organisers advise treating it as a pitch: the problem, the working solution, who it is for, and unmistakable use of
Nebius and NVIDIA. This script follows the brief's timing and the UI as it actually works.

| Time | Show | Say (adapt to what really happened) |
|---|---|---|
| 0:00-0:20 | A failing training run in a terminal | Reproducing someone else's ML experiment fails for dozens of reasons: dependencies, preprocessing, configuration, model bugs. It can take hours. |
| 0:20-0:35 | The ReproFix start page | ReproFix is an AI engineer that does not just suggest a fix: it runs the project, changes it, and checks the result. |
| 0:35-1:50 | **A real run, not the offline demo.** Paste the repository URL and the paper's headline number (the "Check a paper's headline result" box), start. Progress rail, then the measurement instrument showing the metric against the documented band, the evidence graph growing, the first patch being **reverted** if it did not help, a second hypothesis, the diff, the final report with its computed checks and the paper-claims card (claimed, before, after), and the hardware record if you ran on a GPU. | Narrate what the run actually does. If the model gets it right first time, say so; if it needs two tries, that is the better story. If the card says `not_measured`, say why: that is the tool refusing to guess. |
| 1:50-2:25 | The architecture diagram from the README or `docs/architecture.md` | Nemotron Nano, Super and Ultra on Nebius Token Factory, routed by task; code, not the model, decides success; Docker sandbox with no network at run time; Tavily for external evidence. |
| 2:25-2:45 | Your own `bench compare` output | Only numbers you measured: tasks, stages reached, routing against a single model. If routing did not win, say what it did show. |
| 2:45-3:00 | The report's "Verified" line | AI should not just generate code; it should prove the code works. |

Rules of thumb:

- If you show the offline demo at all, label it on screen as a scripted stand-in. The UI banner already says "not
  Nemotron"; do not crop it out.
- Do not put a success rate, cost saving or latency figure on screen unless it came from your own run.
- Keep the Nebius and NVIDIA usage visible: the start page's "Nemotron on Nebius Token Factory" line and the report's
  "Model usage" table (calls, tokens and cost per tier). The exact per-call routing decisions are in `report.json`
  (`routing.decisions`) if you want to show them in a terminal.
- Time it. Under three minutes, public on YouTube, and check it plays when logged out.

## Writeup (Devpost) skeleton

Fill every `[...]` with something you measured or can show; delete a sentence rather than guess a number.

**Inspiration.** Reproducing a published ML result often means hours of dependency, configuration and preprocessing
debugging, and a plausible-looking fix is not the same as a verified one.

**What it does.** Give it a repository and the result it documents (or the text of a paper). It runs it in a sandbox, diagnoses
failures from tracebacks and, when nothing crashes, from the gap between the code and the README; proposes a repair; reruns;
and keeps a change only if the measured result improves. The report lists every check, computed from the final run, and an
evidence graph links each conclusion to quoted code, logs, or search results that code verified. For a paper it prints a card
of each claimed number against the number the program measured, before and after the repair, read by fixed rules rather
than by a model, with the hardware the sandbox could see; and it lists GitHub issues and forum posts that may describe the
same failure, labelled as unverified.

**How we built it.** Python and FastAPI; a deterministic orchestrator; agents for analysis, planning, diagnosis, repair
and review calling NVIDIA Nemotron 3 Nano, Super and Ultra on Nebius Token Factory through its OpenAI-compatible API,
with a router that picks a tier per task and escalates after a rejected hypothesis; a two-phase sandbox (network for
install, none for the run) whose install phase can be confined to PyPI by an allow-list proxy on a Docker internal network;
Tavily for external evidence on environment errors and for known-issue search; a vanilla-JS UI with a live evidence
graph; ReproBench, 30 broken repositories (29 numpy, one real PyTorch) with hidden checks; a pull-request flow that
turns a finished run (verified, executes or partial) into a GitHub pull request with the user's own token.

**What the benchmark showed.** `[your measured stage counts, per routing mode, with the number of repetitions]`

**Challenges.** Verification that cannot be gamed: a hidden check on a different data seed and an independent
recomputation of the metric, a guard against hardcoding the expected number, protected tests. Keeping model output
untrusted: fencing, command and edit guards, evidence that code verifies. `[anything real you hit with the models]`

**What's next.** Real GPU tasks (the benchmark has one PyTorch task on CPU and a simulated device mismatch, nothing on a
GPU; the GPU record and the setup guide exist but have not run on a GPU); checking the paper-claim reader against real papers
and their repositories, which is where its rules will fail first; trying the pull-request flow against live GitHub; running
the egress proxy under a Docker daemon and having someone else attack it; live checks of the GitHub and Tavily issue
search; an OpenShell backend behind the existing sandbox interface.

**Built with.** Python, FastAPI, SQLite, NVIDIA Nemotron 3 (Nano, Super, Ultra), Nebius Token Factory, Tavily, Docker.

## Things not to claim

- That Nemotron solved anything, until you have run it. The committed 30/30 is a scripted oracle.
- That the pull-request button works on GitHub. It has only run against an in-memory fake of the API. Open one real pull
  request on a repository you own before you show it.
- That the benchmark covers GPU faults. It has one real PyTorch task on CPU; the "device" task is a simulation.
- That a paper's result was reproduced on a real NVIDIA GPU, until you have run it on one and kept the output. The code
  records which GPU the sandbox could see; that is not proof the repository's code used it.
- That the paper-claim checker "understands" papers. It matches numbers by fixed patterns and metric names. It will miss
  claims it cannot read, and a reproduced verdict means a printed number landed within a tolerance (a default you can change),
  not that the method was reproduced.
- That the known-issue search finds the cause. It finds possibly related issues by word overlap, and has never run against
  live GitHub or Tavily.
- That the install phase is confined to PyPI. The proxy and its self-test exist and are tested over local sockets; it has
  never run under a Docker daemon.
- That the deployment files are proven. They were validated statically; see [deployment.md](deployment.md).
- That routing is cheaper or better than one model, until you have measured it.
- That the sandbox is "secure". The run phase has no network and the container is hardened, but the install phase has an open
  network unless the (untested under Docker) egress proxy is on, and the Docker backend has not been executed by the author.
  Describe it as it is.
- That OpenShell or NemoClaw are used. They are not integrated.
- Any figure from the original brief's illustrative tables.
