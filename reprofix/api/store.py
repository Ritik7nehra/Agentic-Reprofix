"""SQLite persistence for runs and their event logs (so SSE can resume and runs survive restarts)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

TERMINAL = {"verified", "executes", "already_passing", "partial", "failed", "cancelled", "error", "interrupted"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL, label TEXT,
  request TEXT NOT NULL, report TEXT, llm_backend TEXT
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS pull_requests (
  run_id TEXT PRIMARY KEY, created REAL NOT NULL, data TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def create_run(self, run_id: str, request: dict, label: str | None = None) -> None:
        with self._lock:
            self._db.execute("INSERT INTO runs (id, created, status, label, request) VALUES (?,?,?,?,?)",
                             (run_id, time.time(), "queued", label, json.dumps(request)))

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock:
            self._db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def save_report(self, run_id: str, report: dict) -> None:
        with self._lock:
            self._db.execute("UPDATE runs SET report=?, status=?, llm_backend=? WHERE id=?",
                             (json.dumps(report, default=str), report["status"], report.get("llm_backend"), run_id))

    def add_event(self, run_id: str, kind: str, data: dict) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM events WHERE run_id=?", (run_id,)).fetchone()
            seq = int(row["n"])
            self._db.execute("INSERT INTO events (run_id, seq, ts, kind, data) VALUES (?,?,?,?,?)",
                             (run_id, seq, time.time(), kind, json.dumps(data, default=str)))
            return seq

    def events_after(self, run_id: str, seq: int, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT seq, ts, kind, data FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                                    (run_id, seq, limit)).fetchall()
        return [{"seq": r["seq"], "ts": r["ts"], "kind": r["kind"], "data": json.loads(r["data"])} for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            return None
        return {"id": r["id"], "created": r["created"], "status": r["status"], "label": r["label"],
                "request": json.loads(r["request"]), "report": json.loads(r["report"]) if r["report"] else None,
                "llm_backend": r["llm_backend"]}

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, created, status, label, request, llm_backend FROM runs ORDER BY created DESC LIMIT ?",
                                    (limit,)).fetchall()
        out = []
        for r in rows:
            req = json.loads(r["request"])
            out.append({"id": r["id"], "created": r["created"], "status": r["status"], "label": r["label"],
                        "llm_backend": r["llm_backend"], "repo": req.get("repo_url") or req.get("local_path"),
                        "goal": req.get("goal", "")[:120]})
        return out

    def latest_event(self, run_id: str, kind: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT seq, ts, kind, data FROM events WHERE run_id=? AND kind=? ORDER BY seq DESC LIMIT 1",
                                 (run_id, kind)).fetchone()
        return None if r is None else {"seq": r["seq"], "ts": r["ts"], "kind": r["kind"], "data": json.loads(r["data"])}

    def get_pull_request(self, run_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT data FROM pull_requests WHERE run_id=?", (run_id,)).fetchone()
        return None if r is None else json.loads(r["data"])

    def save_pull_request(self, run_id: str, data: dict) -> bool:
        """Record the pull request opened for a run. False if one was already recorded (never overwritten)."""
        with self._lock:
            cur = self._db.execute("INSERT OR IGNORE INTO pull_requests (run_id, created, data) VALUES (?,?,?)",
                                   (run_id, time.time(), json.dumps(data)))
            return cur.rowcount == 1

    def mark_interrupted(self) -> int:
        with self._lock:
            cur = self._db.execute("UPDATE runs SET status='interrupted' WHERE status IN ('queued','running')")
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
