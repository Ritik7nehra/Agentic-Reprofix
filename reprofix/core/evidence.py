"""Evidence graph: repository -> baseline (symptom, or a verified result) -> hypotheses -> evidence / experiments -> results -> outcome.
A failed diagnosis is a `failure` node, so a run that stops early still shows where and why.

Statuses are assigned by code from experiment outcomes and evidence checks. A model's stated
confidence is stored as `model_confidence` and displayed as such; it never sets a status.
"""
from __future__ import annotations

import threading
from typing import Any

NODE_TYPES = {"repo", "symptom", "hypothesis", "evidence", "experiment", "result", "failure", "outcome"}
STATUSES = {"open", "untested", "confirmed", "rejected", "verified", "unverified", "info", "resolved"}


class EvidenceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self._n = 0
        self._lock = threading.Lock()

    def add(self, type_: str, label: str, status: str = "open", **data: Any) -> str:
        assert type_ in NODE_TYPES and status in STATUSES, (type_, status)
        with self._lock:
            self._n += 1
            nid = f"n{self._n}"
            self.nodes[nid] = {"id": nid, "type": type_, "label": label, "status": status, "data": data}
            return nid

    def link(self, src: str, dst: str, relation: str) -> None:
        with self._lock:
            self.edges.append({"from": src, "to": dst, "relation": relation})

    def set_status(self, nid: str, status: str) -> None:
        assert status in STATUSES
        with self._lock:
            self.nodes[nid]["status"] = status

    def resolve_open_symptoms(self) -> None:
        """A symptom that was seen and then no longer shows in the final, re-measured run is `resolved`, not `open`."""
        with self._lock:
            for n in self.nodes.values():
                if n["type"] == "symptom" and n["status"] == "open":
                    n["status"] = "resolved"

    def update_data(self, nid: str, **data: Any) -> None:
        with self._lock:
            self.nodes[nid]["data"].update(data)

    def to_dict(self) -> dict:
        with self._lock:
            return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    def accepted_hypotheses(self) -> list[dict]:
        with self._lock:
            return [n for n in self.nodes.values() if n["type"] == "hypothesis" and n["status"] == "confirmed"]
