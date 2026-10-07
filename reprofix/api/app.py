"""FastAPI application: REST + Server-Sent Events + static UI."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..config import Settings
from ..core.pullrequest import GitHubClient, PullRequestError, open_pull_request, plan_from_report
from ..core.repo import RepoError, validate_git_url
from ..evaluation.scripted import OracleBackend
from ..models import RunRequest
from ..sandbox import SandboxUnavailable, make_sandbox
from ..util import UnsafeCommand, parse_command
from .runner import BackendFactory, RunManager, SandboxFactory
from .store import TERMINAL, Store

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(os.environ.get("REPROFIX_WEB_DIR", REPO_ROOT / "web"))
BENCH_DIR = Path(os.environ.get("REPROFIX_BENCH_DIR", REPO_ROOT / "benchmark"))


class PullRequestBody(BaseModel):
    token: str = ""            # the caller's own GitHub token; used for this request only, never stored or logged
    confirm: bool = False
    title: str | None = None
    body: str | None = None
    draft: bool = False


def create_app(settings: Settings | None = None, backend_factory: BackendFactory | None = None,
               sandbox_factory: SandboxFactory | None = None, github_transport: httpx.BaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pr_locks: dict[str, threading.Lock] = {}
    pr_locks_guard = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        store = Store(settings.data_dir / "reprofix.db")
        store.mark_interrupted()
        app.state.store = store
        app.state.manager = RunManager(settings, store, backend_factory, sandbox_factory)
        yield
        app.state.manager.shutdown()
        store.close()

    app = FastAPI(title="ReproFix", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    def auth(request: Request, token: str | None = Query(default=None)) -> None:
        if not settings.api_token:
            return
        given = token or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), settings.api_token.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid API token")

    def store() -> Store:
        return app.state.store

    def manager() -> RunManager:
        return app.state.manager

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health() -> dict:
        try:
            sb = make_sandbox(settings) if not sandbox_factory else sandbox_factory()
            sandbox = {"available": True, **sb.policy().to_dict()}
        except SandboxUnavailable as exc:
            sandbox = {"available": False, "kind": settings.sandbox, "error": str(exc)}
        return {
            "ok": True, "version": __version__, "auth_required": bool(settings.api_token),
            "llm": {"provider": "Nebius Token Factory", "configured": bool(settings.nebius_api_key),
                    "base_url": settings.nebius_base_url, "models": settings.models},
            "tavily": {"configured": bool(settings.tavily_api_key)},
            "sandbox": sandbox,
            "limits": {"max_attempts": settings.max_attempts_ceiling, "max_run_seconds": settings.max_run_seconds_ceiling,
                       "allowed_git_hosts": list(settings.allowed_git_hosts), "local_paths": settings.allow_local_paths},
            "pull_requests": settings.allow_pull_requests,
            "demo_available": (BENCH_DIR / "tasks" / "demo_broken_image_classifier" / "repo").is_dir(),
        }

    # ------------------------------------------------------------------ runs
    def _validate(req: RunRequest) -> RunRequest:
        if bool(req.repo_url) == bool(req.local_path):
            raise HTTPException(400, "provide exactly one of repo_url or local_path")
        try:
            if req.repo_url:
                req.repo_url = validate_git_url(req.repo_url, settings.allowed_git_hosts)
            elif not settings.allow_local_paths:
                raise HTTPException(400, "local paths are disabled on this server")
            if req.command:
                parse_command(req.command)
        except (RepoError, UnsafeCommand) as exc:
            raise HTTPException(400, str(exc)) from exc
        req.max_attempts = max(1, min(req.max_attempts, settings.max_attempts_ceiling))
        req.command_timeout_s = max(5, min(req.command_timeout_s, settings.max_command_timeout_ceiling))
        req.max_run_seconds = max(30, min(req.max_run_seconds, settings.max_run_seconds_ceiling))
        req.max_total_tokens = max(10_000, min(req.max_total_tokens, 2_000_000))
        req.goal = req.goal.strip()[:2000] or RunRequest.model_fields["goal"].default
        return req

    @app.post("/api/runs", dependencies=[Depends(auth)], status_code=202)
    def create_run(req: RunRequest) -> dict:
        req = _validate(req)
        return {"id": manager().submit(req)}

    @app.post("/api/demo", dependencies=[Depends(auth)], status_code=202)
    def demo() -> dict:
        """Run the bundled three-fault demo repo with the SCRIPTED offline model (not Nemotron)."""
        task_dir = BENCH_DIR / "tasks" / "demo_broken_image_classifier"
        if not (task_dir / "repo").is_dir():
            raise HTTPException(404, "demo task is not bundled with this install")
        spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        req = RunRequest(local_path=str(task_dir / "repo"), goal=spec["goal"], metric=spec["metric"], max_attempts=6)
        rid = manager().submit(req, label="demo (scripted offline model)", trusted_local=True, backend=OracleBackend(spec))
        return {"id": rid}

    @app.get("/api/runs", dependencies=[Depends(auth)])
    def list_runs() -> list[dict]:
        return store().list_runs()

    @app.get("/api/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str) -> dict:
        run = store().get_run(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    @app.post("/api/runs/{run_id}/cancel", dependencies=[Depends(auth)])
    def cancel(run_id: str) -> dict:
        if store().get_run(run_id) is None:
            raise HTTPException(404, "no such run")
        return {"cancelled": manager().cancel(run_id)}

    @app.get("/api/runs/{run_id}/graph", dependencies=[Depends(auth)])
    def graph(run_id: str) -> dict:
        run = store().get_run(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if run["report"]:
            return run["report"]["graph"]
        ev = store().latest_event(run_id, "graph")
        return ev["data"] if ev else {"nodes": [], "edges": []}

    @app.get("/api/runs/{run_id}/patch", dependencies=[Depends(auth)])
    def patch(run_id: str) -> PlainTextResponse:
        run = store().get_run(run_id)
        if run is None or not run["report"]:
            raise HTTPException(404, "no report yet")
        return PlainTextResponse(run["report"].get("diff", ""), headers={
            "Content-Disposition": f'attachment; filename="reprofix-{run_id}.patch"'})

    # ------------------------------------------------------------------ pull request
    def _pr_plan(run_id: str):
        run = store().get_run(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if not run["report"]:
            raise PullRequestError("the run has not finished yet", "unavailable")
        return plan_from_report(run["report"], settings.data_dir / "runs" / run_id / "orig", run_id)

    @app.get("/api/runs/{run_id}/pull-request", dependencies=[Depends(auth)])
    def pull_request_preview(run_id: str) -> dict:
        """What a pull request would contain, with no network access. `existing` is set once one was opened."""
        if store().get_run(run_id) is None:
            raise HTTPException(404, "no such run")
        existing = store().get_pull_request(run_id)
        if not settings.allow_pull_requests:
            return {"available": False, "reason": "pull requests are disabled on this server", "existing": existing}
        try:
            return {"available": True, **_pr_plan(run_id).to_dict(), "existing": existing}
        except PullRequestError as exc:
            return {"available": False, "reason": str(exc), "existing": existing}

    @app.post("/api/runs/{run_id}/pull-request", dependencies=[Depends(auth)], status_code=201)
    def pull_request_create(run_id: str, req: PullRequestBody) -> dict:
        """Open the pull request on GitHub with the caller's token. Needs `confirm: true`; at most one per run."""
        if not settings.allow_pull_requests:
            raise HTTPException(403, "pull requests are disabled on this server")
        if not req.confirm:
            raise HTTPException(400, "confirm must be true: this creates a branch (in a fork if needed) and opens a public "
                                     "pull request on GitHub as the owner of the token")
        if not req.token.strip():
            raise HTTPException(400, "a GitHub token is required")
        with pr_locks_guard:
            lock = pr_locks.setdefault(run_id, threading.Lock())
        with lock:
            existing = store().get_pull_request(run_id)
            if existing:
                raise HTTPException(409, f"a pull request was already opened for this run: {existing['url']}")
            client = None
            try:
                plan = _pr_plan(run_id)
                client = GitHubClient(req.token, transport=github_transport)
                result = open_pull_request(client, plan, title=req.title, body=req.body, draft=req.draft).to_dict()
            except PullRequestError as exc:
                code = {"github": 502, "conflict": 409}.get(exc.kind, 400)
                raise HTTPException(code, str(exc)) from None
            finally:
                if client is not None:
                    client.close()
            store().save_pull_request(run_id, result)
            return result

    @app.get("/api/runs/{run_id}/events", dependencies=[Depends(auth)])
    async def events(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
        if store().get_run(run_id) is None:
            raise HTTPException(404, "no such run")
        try:
            header_id = int(request.headers.get("last-event-id", "0") or 0)
        except ValueError:                      # a garbled header means "start from the beginning", not a 500
            header_id = 0
        last = max(after, header_id, 0)

        async def gen():
            nonlocal last
            idle = 0.0
            while True:
                if await request.is_disconnected():
                    return
                rows = store().events_after(run_id, last)
                for r in rows:
                    last = r["seq"]
                    yield f"id: {r['seq']}\nevent: {r['kind']}\ndata: {json.dumps(r['data'], default=str)}\n\n"
                if rows:
                    idle = 0.0
                    continue
                run = store().get_run(run_id)
                if run and run["status"] in TERMINAL and not store().events_after(run_id, last):
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.25)
                idle += 0.25
                if idle >= 15:
                    idle = 0.0
                    yield ": keepalive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ benchmark
    @app.get("/api/benchmark")
    def benchmark() -> dict:
        res = BENCH_DIR / "results"
        files = sorted(res.glob("*.json")) if res.is_dir() else []
        tasks_dir = BENCH_DIR / "tasks"
        n_tasks = sum(1 for d in tasks_dir.iterdir() if (d / "task.json").is_file()) if tasks_dir.is_dir() else 0
        return {"n_tasks": n_tasks, "results": [json.loads(f.read_text(encoding="utf-8")) for f in files]}

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    @app.exception_handler(HTTPException)
    async def http_exc(_: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def app_factory() -> FastAPI:  # for `uvicorn reprofix.api.app:app_factory --factory`
    return create_app()
