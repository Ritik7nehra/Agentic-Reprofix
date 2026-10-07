"""Command line: serve, run, claims, pr, doctor, demo, bench, egress."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .config import TIERS, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmark"


# --------------------------------------------------------------------------- helpers
def _printer(verbose: bool):
    def emit(kind: str, d: dict) -> None:
        if kind == "step" and d["status"] in {"running", "done", "failed"}:
            mark = {"running": "…", "done": "✓", "failed": "✗"}[d["status"]]
            if d["status"] != "running":
                print(f"  {mark} {d['label']}" + (f"  ({d['detail']})" if d.get("detail") else ""))
        elif kind == "verdict":
            v = d["verdict"]
            val = f" metric={v['metric_value']:.4f}" if v.get("metric_value") is not None else ""
            print(f"    [{d['label']}] {v['stage']}{val}")
        elif kind == "experiment":
            print(f"    experiment {d['n']}: {'kept' if d['kept'] else 'reverted'} -- {d['reason']}  ({', '.join(d['files'])})")
        elif kind == "attempt.failed":
            print(f"    attempt {d['attempt']} stopped at {d['stage']}: {str(d['error'])[:140]}")
        elif verbose and kind == "exec":
            print(f"      $ {' '.join(d['argv'])[-90:]}  -> exit {d['exit_code']} in {d['duration_s']}s")
    return emit


def _print_report(rep: dict) -> None:
    print()
    print(f"STATUS: {rep['status'].upper()} -- {rep['status_text']}")
    if rep.get("error"):
        print(f"error: {rep['error']}")
    if rep.get("stop_reason"):
        print(f"stopped because: {rep['stop_reason']}")
    for c in rep["verification"]:
        mark = {True: "PASS", False: "FAIL", None: " -- "}[c["passed"]]
        print(f"  [{mark}] {c['check']}" + (f": {c['detail']}" if c["detail"] else ""))
    if rep.get("claims", {}).get("checked") or rep.get("claims", {}).get("skipped"):
        _print_claims(rep["claims"])
    for i in rep["issues"]:
        print(f"  issue ({i['category']}): {i['statement']}  [{', '.join(i['files'])}]")
    u = rep["usage"]
    cost = "n/a" if u["cost_usd_priced_tiers"] is None else f"${u['cost_usd_priced_tiers']:.4f}"
    print(f"  model usage: {u['calls']} calls, {u['prompt_tokens'] + u['completion_tokens']:,} tokens, cost {cost}"
          + (" (priced tiers only)" if u["has_unpriced_tiers"] else ""))
    ki = rep.get("known_issues") or {}
    for r in ki.get("results", [])[:5]:
        print(f"  possibly related ({r['source']}{', ' + r['state'] if r.get('state') else ''}): {r['title'][:90]}  {r['url']}")
    for c in rep["caveats"]:
        print(f"  ! {c}")


def _fmt_claim_value(v, unit: str) -> str:
    if v is None:
        return "not measured"
    return f"{v:g}%" if unit == "percent" else f"{v:g}"


def _print_claims(card: dict) -> None:
    if not card.get("checked"):
        print(f"  paper claims: not checked ({card.get('reason', '')})")
        return
    s = card["summary"]
    print(f"  paper claims: {s['reproduced_after']}/{s['total']} reproduced after the run "
          f"({s['reproduced_before']} before; {s['fixed']} fixed, {s['regressed']} regressed, {s['not_measured']} not measured)")
    for i in card["items"]:
        u = i["unit"]
        b, f = i["baseline"], i["final"]
        head = "*" if i["headline"] else " "
        print(f"   {head}{i['id']} {i['metric']}: claimed {_fmt_claim_value(i['claimed'], u)} (±{i['tolerance']:g}{'pt' if u == 'percent' else ''})"
              f" | before {_fmt_claim_value(b['measured'], u)} [{b['verdict']}]"
              f" | after {_fmt_claim_value(f['measured'], u)} [{f['verdict']}]")
        if f["verdict"] == "not_measured" and f["reason"]:
            print(f"        why: {f['reason']}")
    for sk in card.get("skipped", []):
        print(f"   note: {sk}")


def _make_sandbox(settings: Settings):
    from .sandbox import SandboxUnavailable, make_sandbox
    try:
        return make_sandbox(settings)
    except SandboxUnavailable as exc:
        sys.exit(f"sandbox unavailable: {exc}")


# --------------------------------------------------------------------------- commands
def cmd_serve(a, settings: Settings) -> int:
    import uvicorn
    from .api.app import create_app
    from .sandbox import SandboxUnavailable, make_sandbox
    try:
        sb = make_sandbox(settings)
        print(f"sandbox: {sb.policy().kind} (filesystem isolated: {sb.policy().isolated})")
    except SandboxUnavailable as exc:
        print(f"WARNING: sandbox unavailable ({exc}). The UI will load but runs will fail until this is fixed.")
    if not settings.nebius_api_key:
        print("WARNING: NEBIUS_API_KEY is not set. Real runs will fail; the offline demo still works.")
    if a.host not in ("127.0.0.1", "localhost") and not settings.api_token:
        print("WARNING: listening on a public interface without REPROFIX_API_TOKEN. Anyone who can reach this "
              "server can make it execute repositories. Set a token and see docs/security.md.")
    uvicorn.run(create_app(settings), host=a.host, port=a.port, log_level="info")
    return 0


def cmd_run(a, settings: Settings) -> int:
    from .core.orchestrator import Orchestrator
    from .inference import AuthError, NebiusClient, TavilyClient
    from .claims import parse_claim
    from .models import MetricSpec, RunRequest
    s = copy.copy(settings)
    if a.path:
        s.allow_local_paths = True
    metric = MetricSpec(name=a.metric_name, expected=a.expected, tolerance=a.tolerance) if a.expected is not None else None
    try:
        claims = [parse_claim(c) for c in (a.claim or [])]
    except ValueError as exc:
        sys.exit(str(exc))
    req = RunRequest(repo_url=a.repo, local_path=a.path, goal=a.goal, command=a.command, metric=metric,
                     max_attempts=a.max_attempts, router_mode=a.router_mode, claims=claims,
                     paper_text=_read_paper(a.paper) if a.paper else "")
    try:
        backend = NebiusClient(s)
    except AuthError as exc:
        sys.exit(f"{exc}. Set NEBIUS_API_KEY (see .env.example) or try `reprofix demo`.")
    run_dir = Path(a.out).resolve() / time.strftime("run-%Y%m%d-%H%M%S")
    tav = TavilyClient(s.tavily_api_key) if s.tavily_api_key else None
    from .core.issues import IssueFinder
    finder = IssueFinder(tav) if s.known_issues else None
    print(f"ReproFix {__version__}  model routing: {a.router_mode}  web search: {'on' if tav else 'off'}  "
          f"known-issues search: {'on' if finder else 'off'}")
    try:
        rep = Orchestrator(settings=s, request=req, run_dir=run_dir, sandbox=_make_sandbox(s), backend=backend,
                           emit=_printer(a.verbose), tavily=tav, issues=finder).run()
    finally:
        if finder is not None:
            finder.close()
    _print_report(rep)
    if rep["diff"]:
        (run_dir / "patch.diff").write_text(rep["diff"], encoding="utf-8")
        print(f"\npatch: {run_dir / 'patch.diff'}")
        if (rep.get("repository") or {}).get("commit"):
            print(f"to propose it upstream: reprofix pr {run_dir} --dry-run")
    print(f"report: {run_dir / 'report.json'}")
    return 0 if rep["status"] in {"verified", "executes", "already_passing"} else 1


def _read_paper(path: str) -> str:
    try:
        return Path(path).expanduser().read_bytes()[:200_000].decode("utf-8", "replace")
    except OSError as exc:
        sys.exit(f"cannot read {path}: {exc}")


def cmd_claims(a, settings: Settings) -> int:
    """Show which numbers ReproFix would read from a paper/README text file, without running anything."""
    from .claims import collect
    from .models import RunRequest
    cs = collect(RunRequest(repo_url="https://github.com/x/y", paper_text=_read_paper(a.paper)))
    if not cs:
        print("no claims found. The patterns read lines like 'accuracy of 76.4%', '27.3 BLEU' and markdown tables with "
              "metric columns (one row labelled 'ours'); pass others with `reprofix run --claim accuracy=76.4%`.")
    for i, c in zip(cs.ids, cs.claims):
        star = "*" if c.headline else " "
        print(f"{star}{i} {c.metric}{f' ({c.qualifier})' if c.qualifier else ''} = {c.value:g}{'%' if c.unit == 'percent' else ''}"
              f"   [{c.source}]")
        print(f"     \"{c.quote}\"")
        if c.note:
            print(f"     ! {c.note}")
    for sk in cs.skipped:
        print(f"note: {sk}")
    if cs:
        print("* = the claim the repair loop would target. Confirm each number against the paper: these are pattern matches.")
    return 0 if cs else 1


def cmd_pr(a, settings: Settings) -> int:
    """Open a GitHub pull request from a finished run directory. The token comes from GITHUB_TOKEN, never from argv."""
    from .core import pullrequest as P
    run_dir = Path(a.run_dir).resolve()
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        sys.exit(f"no report.json in {run_dir}: pass the run directory printed by `reprofix run`")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        plan = P.plan_from_report(report, run_dir / "orig", run_dir.name)
    except P.PullRequestError as exc:
        sys.exit(f"cannot open a pull request: {exc}")
    title = P._clean(a.title, P.MAX_TITLE) if a.title else plan.title
    print(f"target:  {plan.full_name} (base commit {plan.base_sha[:10]})")
    print(f"status:  {plan.status}")
    print("files:   " + ", ".join(f"{f.path} ({'added' if f.is_new else 'modified'})" for f in plan.files))
    print(f"title:   {title}")
    if a.show_body or a.dry_run:
        print("\n" + plan.body + "\n")
    if a.dry_run:
        print("dry run: nothing was sent to GitHub.")
        return 0
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("set GITHUB_TOKEN to a token that can open the pull request (see docs/security.md for the scopes), or use --dry-run")
    if not a.yes:
        answer = input("This creates a branch (in a fork if you cannot push) and opens a public pull request as the owner of "
                       "GITHUB_TOKEN. Continue? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted; nothing was sent to GitHub.")
            return 1
    try:
        client = P.GitHubClient(token)
        try:
            res = P.open_pull_request(client, plan, title=title, draft=a.draft)
        finally:
            client.close()
    except P.PullRequestError as exc:
        sys.exit(f"pull request failed: {exc}")
    print(f"opened {res.url}  (branch {res.branch} in {res.head_repo}, {res.commits} commit(s), {res.mode})")
    return 0


def cmd_demo(a, settings: Settings) -> int:
    from .core.orchestrator import Orchestrator
    from .evaluation.harness import load_tasks
    from .evaluation.scripted import OracleBackend
    from .models import MetricSpec, RunRequest
    task = load_tasks(BENCH_DIR / "tasks", ["demo_broken_image_classifier"])[0]
    s = copy.copy(settings)
    s.allow_local_paths = True
    req = RunRequest(local_path=str(task.repo), goal=task.spec["goal"], metric=MetricSpec(**task.spec["metric"]))
    print("OFFLINE DEMO: a scripted model replays the reference fix. This exercises the pipeline; it is NOT Nemotron.")
    run_dir = Path(a.out).resolve() / time.strftime("demo-%Y%m%d-%H%M%S")
    rep = Orchestrator(settings=s, request=req, run_dir=run_dir, sandbox=_make_sandbox(s), backend=OracleBackend(task.spec),
                       emit=_printer(a.verbose)).run()
    _print_report(rep)
    return 0 if rep["status"] == "verified" else 1


def cmd_doctor(a, settings: Settings) -> int:
    import httpx
    from .inference import AuthError, NebiusClient, TavilyClient
    from .inference.client import normalize_model_id
    bad = 0

    def line(ok: bool | None, msg: str) -> None:
        nonlocal bad
        print(f"  [{'ok' if ok else 'FAIL' if ok is False else '!!'}] {msg}")
        bad += ok is False

    print(f"ReproFix {__version__} doctor")
    line(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]} (needs 3.11+)")
    print("Nebius Token Factory")
    line(bool(settings.nebius_api_key), "NEBIUS_API_KEY is set" if settings.nebius_api_key else "NEBIUS_API_KEY is not set")
    if settings.nebius_api_key:
        bases = [settings.nebius_base_url]
        for alt in ("https://api.tokenfactory.nebius.com/v1/", "https://api.tokenfactory.us-central1.nebius.com/v1/"):  # global per the quickstart; regional per the cookbook
            if alt not in bases:
                bases.append(alt)
        served: list[str] = []
        for base in bases:
            s2 = copy.copy(settings)
            s2.nebius_base_url = base
            try:
                served = NebiusClient(s2, timeout=30).list_models()
                line(True, f"GET {base}models -> {len(served)} models" + ("" if base == settings.nebius_base_url else
                     f"  (works, but NEBIUS_BASE_URL is {settings.nebius_base_url}; update it)"))
                settings = s2
                break
            except AuthError as exc:
                line(False, f"{base}: {exc}")
                break
            except (httpx.HTTPError, ValueError) as exc:
                line(None if base != settings.nebius_base_url else False, f"{base}: {type(exc).__name__}: {str(exc)[:80]}")
        for tier in TIERS:
            mid = settings.models[tier]
            if not served:
                break
            alt = next((m for m in served if normalize_model_id(m) == normalize_model_id(mid)), None)
            if mid in served:
                line(True, f"{tier}: {mid} is served")
            elif alt:
                line(True, f"{tier}: served as {alt!r}; ReproFix resolves that spelling automatically (set NEBIUS_MODEL_{tier.upper()} to match)")
            else:
                near = [m for m in served if "nemotron" in m.lower()][:6]
                line(False, f"{tier}: {mid} is NOT in the catalog. Nemotron models served: {near or 'none found'}. "
                     f"Set NEBIUS_MODEL_{tier.upper()}.")
        if a.live and served:
            for tier in TIERS:
                try:
                    r = NebiusClient(settings, timeout=120).complete(tier=tier, messages=[{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=64)
                    line(True, f"{tier} chat ok in {r.latency_s:.1f}s, usage reported: {r.prompt_tokens is not None}")
                except Exception as exc:
                    line(False, f"{tier} chat failed: {type(exc).__name__}: {str(exc)[:100]}")
    print("Tavily")
    if settings.tavily_api_key:
        if a.live:
            tv = TavilyClient(settings.tavily_api_key)
            res = tv.search("pip ResolutionImpossible conflicting dependencies", max_results=1)
            line(bool(res), "live search returned results" if res else f"live search failed: {tv.errors[-1:] or 'no results'}")
        else:
            line(True, "TAVILY_API_KEY is set (use --live to spend one search credit and test it)")
    else:
        line(None, "TAVILY_API_KEY not set: runs work, but without external evidence")
    print("Sandbox")
    from .sandbox import SandboxUnavailable, make_sandbox
    try:
        sb = make_sandbox(settings)
        pol = sb.policy()
        line(True, f"backend '{pol.kind}' available; filesystem isolated: {pol.isolated}")
        if pol.kind == "docker":
            r = subprocess.run(["docker", "image", "inspect", settings.sandbox_image], capture_output=True)
            line(r.returncode == 0, f"image {settings.sandbox_image} " + ("present" if r.returncode == 0 else
                 "missing: docker build -f docker/sandbox.Dockerfile -t reprofix-sandbox:latest ."))
            from .sandbox import hardware as _hw
            if settings.sandbox_gpu:
                try:
                    r = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--network", "none", settings.sandbox_image, *_hw.SMI_QUERY_ARGV],
                                       capture_output=True, text=True, timeout=120)
                    rec = _hw.from_smi(requested=True, probed_in="docker run --gpus all", query_stdout=r.stdout, query_code=r.returncode,
                                       query_stderr=r.stderr)
                    line(rec["status"] == "detected", "GPU in the sandbox: " + rec["summary"] + ("" if rec["status"] == "detected" else f" ({rec['detail']}); see docs/gpu.md"))
                except (OSError, subprocess.SubprocessError) as exc:
                    line(False, f"GPU probe could not run: {type(exc).__name__}: {str(exc)[:100]}")
            else:
                line(None, "GPU: not enabled (REPROFIX_SANDBOX_GPU=0), so runs use the CPU. docs/gpu.md explains how to use a GPU VM")
            if settings.install_network == "proxy":
                from .sandbox import egress_ctl
                try:
                    egress_ctl.check_ready(settings)
                    ok, lines = egress_ctl.selftest(settings)
                    for ln in lines:
                        line(ln.startswith("PASS"), "install network: " + ln.split(" ", 1)[1])
                except egress_ctl.EgressError as exc:
                    line(False, f"install network is 'proxy' but not ready: {exc}")
            else:
                line(None, "install network is OPEN: during pip install the sandbox can reach any host. "
                           "Set REPROFIX_INSTALL_NETWORK=proxy and run `reprofix egress up` to allow only PyPI")
        else:
            line(None, "local backend is development-only")
    except SandboxUnavailable as exc:
        line(False, str(exc))
    line(shutil.which("git") is not None, "git on PATH (needed to clone repositories)")
    line(shutil.which("patch") is not None, "patch on PATH (used to verify the patch applies)")
    print("OK" if not bad else f"{bad} problem(s) found")
    return 1 if bad else 0


def cmd_egress(a, settings: Settings) -> int:
    """Manage the install-phase egress proxy (an --internal Docker network plus an allow-list proxy container)."""
    from .sandbox import egress_ctl as E
    try:
        if a.egress_cmd == "up":
            for line in E.up(settings):
                print("  " + line)
            print(f"ready. Set REPROFIX_INSTALL_NETWORK=proxy so installs use it (now: {settings.install_network}). "
                  "Check it with `reprofix egress test`.")
        elif a.egress_cmd == "down":
            for line in E.down(settings) or ["nothing to remove"]:
                print("  " + line)
        elif a.egress_cmd == "status":
            st = E.status(settings)
            print(f"install network mode: {st['mode']}   allow-list: {', '.join(st['allow'])}")
            print(f"network {settings.egress_network}: " + ("missing" if not st["network"]["exists"] else
                  "internal (no route out)" if st["network"]["internal"] else "EXISTS BUT IS NOT INTERNAL"))
            c = st["container"]
            print(f"proxy {settings.egress_container}: " + ("missing" if not c["exists"] else
                  ("running" if c["running"] else "stopped") + f", attached to {', '.join(c['networks']) or 'nothing'}"))
            if st["allow_matches_running"] is False:
                print("  ! the running proxy has a different allow-list than REPROFIX_EGRESS_ALLOW: run `reprofix egress up` to recreate it")
            print("ready" if st["ready"] else f"NOT READY: {st['problem']}")
            return 0 if st["ready"] else 1
        else:
            ok, lines = E.selftest(settings)
            print("\n".join("  " + ln for ln in lines))
            print("egress confinement verified" if ok else "egress confinement NOT verified")
            return 0 if ok else 1
    except E.EgressError as exc:
        sys.exit(f"egress: {exc}")
    return 0


def cmd_bench(a, settings: Settings) -> int:
    from .evaluation import harness as H
    if a.bench_cmd == "compare":
        rows = []
        for f in a.files:
            r = json.loads(Path(f).read_text(encoding="utf-8"))
            n = r["n_tasks"]
            t = r["totals"]
            rows.append((r["meta"].get("label") or Path(f).stem, r["meta"]["router_mode"], r["meta"]["llm_backend"],
                         f"{r['stages']['patch_verified']}/{n}", f"{t['tokens']:,}",
                         "n/a" if t["cost_usd_priced_tiers"] is None else f"${t['cost_usd_priced_tiers']:.4f}", f"{t['llm_latency_s']}s"))
        print(f"{'label':28} {'routing':11} {'backend':9} {'verified':9} {'tokens':>10} {'cost':>9} {'LLM latency':>12}")
        for r in rows:
            print(f"{r[0]:28} {r[1]:11} {r[2]:9} {r[3]:9} {r[4]:>10} {r[5]:>9} {r[6]:>12}")
        return 0
    tasks = H.load_tasks(BENCH_DIR / "tasks", a.only or None)
    if a.bench_cmd == "list":
        for t in tasks:
            print(f"{t.id:32} {t.spec['category']:14} {t.spec['difficulty']:7} {t.spec['title']}")
        print(f"{len(tasks)} tasks")
        return 0
    s = copy.copy(settings)
    s.allow_local_paths = True
    sandbox = _make_sandbox(s)
    scratch = Path(a.out).resolve() / "validate"
    if a.bench_cmd == "validate":
        scratch.mkdir(parents=True, exist_ok=True)
        bad = 0
        for t in tasks:
            r = H.validate_task(t, sandbox, scratch)
            print(f"{'OK ' if r['ok'] else 'BAD'} {t.id:32} broken={r['broken']['stage']:16} fixed={r['fixed']['stage']:9}" + ("  " + "; ".join(r["problems"]) if r["problems"] else ""))
            bad += not r["ok"]
        print(f"{len(tasks) - bad}/{len(tasks)} tasks valid")
        return 1 if bad else 0
    # run
    if a.llm == "nebius":
        from .inference import AuthError, NebiusClient, TavilyClient
        try:
            client = NebiusClient(s)
        except AuthError as exc:
            sys.exit(f"{exc}. A real benchmark run needs a Nebius key; use --llm oracle to validate the harness only.")
        tav = TavilyClient(s.tavily_api_key) if s.tavily_api_key else None
        backend_for = lambda t: client  # noqa: E731
    else:
        print("ORACLE MODE: scripted answers from each task's reference fix. This validates the harness; it measures no model.")
        tav = None
        backend_for = H.oracle_backend_for
    # A partial run must never overwrite (or pass for) the results of a full run with the same settings.
    label = a.label or f"{a.llm}-{a.router_mode}" + ("-subset" if a.only else "")
    out_dir = Path(a.out).resolve() / time.strftime("bench-%Y%m%d-%H%M%S")
    summary = H.run_benchmark(tasks=tasks, settings=s, sandbox=sandbox, backend_for=backend_for, out_dir=out_dir,
                              router_mode=a.router_mode, max_attempts=a.max_attempts, tavily=tav, label=label)
    res_dir = BENCH_DIR / "results"
    res_dir.mkdir(exist_ok=True)
    path = res_dir / f"{label}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    path.with_suffix(".md").write_text(H.render_markdown(summary), encoding="utf-8")
    print("\n" + H.render_markdown(summary))
    print(f"saved {path}")
    return 0


# --------------------------------------------------------------------------- entry
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reprofix", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="start the API and web UI")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)

    rp = sub.add_parser("run", help="investigate one repository from the terminal")
    g = rp.add_mutually_exclusive_group(required=True)
    g.add_argument("--repo", help="https URL of a git repository")
    g.add_argument("--path", help="local directory (trusted dev machines only)")
    rp.add_argument("--goal", default="Reproduce the documented experiment and determine what is wrong.")
    rp.add_argument("--command")
    rp.add_argument("--metric-name", default="val_accuracy")
    rp.add_argument("--expected", type=float, help="documented value; omit to only check that the project runs")
    rp.add_argument("--tolerance", type=float, default=0.02)
    rp.add_argument("--router-mode", choices=["router", "super-only", "ultra-only"], default="router")
    rp.add_argument("--max-attempts", type=int, default=6)
    rp.add_argument("--paper", metavar="FILE", help="text of the paper or README to read claims from (never sent to a model)")
    rp.add_argument("--claim", action="append", metavar="METRIC=VALUE",
                    help="a number the paper states, e.g. accuracy=76.4%% or bleu=27.3±0.3; repeatable; the first is the one to repair toward")
    rp.add_argument("--out", default="./data/cli")
    rp.add_argument("-v", "--verbose", action="store_true")

    cp = sub.add_parser("claims", help="show which paper claims would be read from a text file (runs nothing)")
    cp.add_argument("paper", metavar="FILE")

    pp = sub.add_parser("pr", help="open a GitHub pull request from a finished run (token from $GITHUB_TOKEN)")
    pp.add_argument("run_dir", help="the run directory printed by `reprofix run` (contains report.json and orig/)")
    pp.add_argument("--dry-run", action="store_true", help="show what would be sent; no network access")
    pp.add_argument("--title")
    pp.add_argument("--draft", action="store_true")
    pp.add_argument("--show-body", action="store_true")
    pp.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    dp = sub.add_parser("demo", help="run the bundled three-fault repo with a scripted offline model")
    dp.add_argument("--out", default="./data/cli")
    dp.add_argument("-v", "--verbose", action="store_true")

    kp = sub.add_parser("doctor", help="check configuration, model IDs, base URL and sandbox")
    kp.add_argument("--live", action="store_true", help="make one tiny real call per model and one Tavily search")

    ep = sub.add_parser("egress", help="the install-phase egress proxy: only PyPI is reachable while pip installs")
    ep.add_argument("egress_cmd", choices=["up", "down", "status", "test"])

    bp = sub.add_parser("bench", help="ReproBench")
    bs = bp.add_subparsers(dest="bench_cmd", required=True)
    for name in ("list", "validate", "run", "compare"):
        x = bs.add_parser(name)
        if name == "compare":
            x.add_argument("files", nargs="+")
            continue
        x.add_argument("--only", nargs="*")
        if name != "list":
            x.add_argument("--out", default="./data/bench")
        if name == "run":
            x.add_argument("--llm", choices=["nebius", "oracle"], default="nebius")
            x.add_argument("--router-mode", choices=["router", "super-only", "ultra-only"], default="router")
            x.add_argument("--max-attempts", type=int, default=6)
            x.add_argument("--label")

    a = p.parse_args(argv)
    settings = Settings.from_env()
    return {"serve": cmd_serve, "run": cmd_run, "claims": cmd_claims, "pr": cmd_pr, "demo": cmd_demo, "doctor": cmd_doctor, "bench": cmd_bench,
            "egress": cmd_egress}[a.cmd](a, settings)


if __name__ == "__main__":
    sys.exit(main())
