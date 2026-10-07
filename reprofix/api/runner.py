"""Runs jobs in a bounded thread pool and persists their events."""
from __future__ import annotations

import copy
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..core.orchestrator import Orchestrator
from ..core.issues import IssueFinder
from ..inference import AuthError, ChatBackend, NebiusClient, TavilyClient
from ..models import RunRequest
from ..sandbox import Sandbox, SandboxUnavailable, make_sandbox
from .store import Store

BackendFactory = Callable[[RunRequest], ChatBackend]
SandboxFactory = Callable[[], Sandbox]


class RunManager:
    def __init__(self, settings: Settings, store: Store, backend_factory: BackendFactory | None = None,
                 sandbox_factory: SandboxFactory | None = None):
        self.settings = settings
        self.store = store
        self.backend_factory = backend_factory or (lambda req: NebiusClient(settings))
        self.sandbox_factory = sandbox_factory or (lambda: make_sandbox(settings))
        self.pool = ThreadPoolExecutor(max_workers=max(1, settings.max_concurrent_runs), thread_name_prefix="reprofix-run")
        self.cancels: dict[str, threading.Event] = {}
        (settings.data_dir / "runs").mkdir(parents=True, exist_ok=True)

    def submit(self, request: RunRequest, *, label: str | None = None, trusted_local: bool = False,
               backend: ChatBackend | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.store.create_run(run_id, request.model_dump(), label)
        cancel = threading.Event()
        self.cancels[run_id] = cancel
        self.pool.submit(self._job, run_id, request, cancel, trusted_local, backend)
        return run_id

    def cancel(self, run_id: str) -> bool:
        ev = self.cancels.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _job(self, run_id: str, request: RunRequest, cancel: threading.Event, trusted_local: bool,
             backend: ChatBackend | None) -> None:
        def emit(kind: str, data: dict) -> None:
            self.store.add_event(run_id, kind, data)

        run_dir = self.settings.data_dir / "runs" / run_id
        try:
            if cancel.is_set():
                self.store.set_status(run_id, "cancelled")
                emit("run.finished", {"status": "cancelled"})
                return
            self.store.set_status(run_id, "running")
            settings = copy.copy(self.settings)
            if trusted_local:
                settings.allow_local_paths = True
            sandbox = self.sandbox_factory()
            be = backend or self.backend_factory(request)
            tavily = TavilyClient(settings.tavily_api_key) if settings.tavily_api_key else None
            # The offline demo (trusted_local) never reaches out; real runs search for known issues unless it is switched off.
            issues = IssueFinder(tavily) if settings.known_issues and not trusted_local else None
            orch = Orchestrator(settings=settings, request=request, run_dir=run_dir, sandbox=sandbox, backend=be,
                                emit=emit, cancel=cancel, tavily=tavily, issues=issues)
            try:
                report = orch.run()
            finally:
                if issues is not None:
                    issues.close()
            self.store.save_report(run_id, report)
            (run_dir / "patch.diff").write_text(report.get("diff", ""), encoding="utf-8")
        except (AuthError, SandboxUnavailable) as exc:
            self._fail(run_id, emit, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # never let a job die silently
            traceback.print_exc()
            self._fail(run_id, emit, f"{type(exc).__name__}: {exc}")
        finally:
            self.cancels.pop(run_id, None)

    def _fail(self, run_id: str, emit, message: str) -> None:
        emit("error", {"message": message})
        emit("run.finished", {"status": "error"})
        self.store.set_status(run_id, "error")

    def shutdown(self) -> None:
        for ev in self.cancels.values():
            ev.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
