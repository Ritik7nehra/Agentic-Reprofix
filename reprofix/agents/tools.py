"""Read-only investigation tools exposed to the diagnoser, confined to the run workspace."""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..core.repo import IGNORE_DIRS
from ..inference import TavilyClient, WebEvidence
from ..util import safe_join

TEXT_SUFFIXES = (".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".rst")
MAX_READ_LINES = 250
MAX_READ_BYTES = 24_000


class ToolBox:
    def __init__(self, workdir: Path, tavily: TavilyClient | None):
        self.workdir = workdir
        self.tavily = tavily
        self.web_results: list[WebEvidence] = []
        self.calls: list[dict] = []

    def describe(self) -> str:
        web = "available" if self.tavily and self.tavily.available else "unavailable (no TAVILY_API_KEY)"
        return (
            "Tools (read-only; all paths are relative to the repository root):\n"
            '  {"tool":"read_file","path":"model.py","start":1,"end":120}   # 1-based inclusive line range, optional\n'
            '  {"tool":"grep","pattern":"Normalize","glob":"*.py"}           # regex over text files, optional glob\n'
            '  {"tool":"list_dir","path":"."}\n'
            f'  {{"tool":"search_web","query":"..."}}                         # Tavily web search: {web}'
        )

    def run(self, call: dict) -> str:
        tool = call.get("tool")
        try:
            if tool == "read_file":
                out = self._read(call)
            elif tool == "grep":
                out = self._grep(call)
            elif tool == "list_dir":
                out = self._ls(call)
            elif tool == "search_web":
                out = self._search(call)
            else:
                out = f"unknown tool {tool!r}"
        except (ValueError, OSError) as exc:
            out = f"error: {exc}"
        self.calls.append({"call": call, "output_chars": len(out)})
        return out

    def _read(self, call: dict) -> str:
        path = safe_join(self.workdir, str(call.get("path", "")))
        if not path.is_file():
            raise ValueError(f"no such file: {call.get('path')}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(int(call.get("start") or 1), 1)
        end = min(int(call.get("end") or start + MAX_READ_LINES - 1), start + MAX_READ_LINES - 1, len(lines))
        body = "\n".join(f"{i:>4}  {lines[i - 1]}" for i in range(start, end + 1))
        body = body[:MAX_READ_BYTES]
        return f"{call.get('path')} (lines {start}-{end} of {len(lines)})\n{body}"

    def _grep(self, call: dict) -> str:
        pat = re.compile(str(call.get("pattern", "")))
        glob = str(call.get("glob") or "*")
        rx = re.compile("^" + re.escape(glob).replace(r"\*", ".*").replace(r"\?", ".") + "$")
        hits: list[str] = []
        for dp, dn, fn in os.walk(self.workdir):
            dn[:] = [d for d in dn if d not in IGNORE_DIRS]
            for f in sorted(fn):
                if not f.endswith(TEXT_SUFFIXES) or not rx.match(f):
                    continue
                p = Path(dp) / f
                rel = p.relative_to(self.workdir).as_posix()
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pat.search(line):
                        hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                        if len(hits) >= 40:
                            return "\n".join(hits) + "\n... (truncated at 40 matches)"
        return "\n".join(hits) or "no matches"

    def _ls(self, call: dict) -> str:
        p = safe_join(self.workdir, str(call.get("path") or ".")) if call.get("path") not in (None, ".", "") else self.workdir
        if not p.is_dir():
            raise ValueError(f"not a directory: {call.get('path')}")
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir() if x.name not in IGNORE_DIRS)
        return "\n".join(names[:200]) or "(empty)"

    def _search(self, call: dict) -> str:
        if not self.tavily or not self.tavily.available:
            return "web search is unavailable"
        results = self.tavily.search(str(call.get("query", "")), max_results=5)
        if not results:
            return "no results (or the search failed)"
        self.web_results.extend(results)
        return "\n\n".join(f"[{i + 1}] {r.title}\nURL: {r.url}\n{r.snippet}" for i, r in enumerate(results))
