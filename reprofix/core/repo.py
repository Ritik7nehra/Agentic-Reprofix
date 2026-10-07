"""Acquire (clone/copy) a repository into a run workspace, and scan it deterministically."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ..config import Settings
from ..util import UnsafeCommand, parse_command

IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".deps", ".home", ".pytest_cache", ".mypy_cache"}
MAX_FILES = 20_000


class RepoError(RuntimeError):
    pass


def validate_git_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    p = urlparse(url.strip())
    if p.scheme != "https":
        raise RepoError("only https:// repository URLs are accepted")
    if p.username or p.password:
        raise RepoError("credentials in the URL are not accepted")
    host = (p.hostname or "").lower()
    if host not in allowed_hosts:
        raise RepoError(f"host {host!r} is not allowed (allowed: {', '.join(allowed_hosts)})")
    if not re.fullmatch(r"/[\w.\-]+/[\w.\-]+(?:\.git)?/?", p.path):
        raise RepoError("URL must look like https://<host>/<owner>/<repo>")
    if any(part.removesuffix(".git") in {"", ".", ".."} for part in p.path.strip("/").split("/")):
        raise RepoError("URL must look like https://<host>/<owner>/<repo>")
    return f"https://{host}{p.path.rstrip('/')}"


def _dir_size_mb(root: Path) -> float:
    total = 0
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            try:
                total += (Path(dp) / f).stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _strip_symlinks(root: Path) -> int:
    """Symlinks in an untrusted repo could point outside the workspace; drop them all."""
    n = 0
    for dp, dn, fn in os.walk(root):
        for name in list(dn) + list(fn):
            p = Path(dp) / name
            if p.is_symlink():
                p.unlink()
                n += 1
    return n


def _count_files(root: Path) -> int:
    return sum(len(fn) for _dp, _dn, fn in os.walk(root))


def acquire(*, repo_url: str | None, local_path: str | None, settings: Settings, work: Path, orig: Path) -> dict:
    """Populate `work` with the repository and snapshot it into `orig`. Returns provenance info."""
    if bool(repo_url) == bool(local_path):
        raise RepoError("provide exactly one of repo_url or local_path")
    info: dict = {}
    if repo_url:
        clean = validate_git_url(repo_url, settings.allowed_git_hosts)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work.parent), "GIT_TERMINAL_PROMPT": "0",
               "GIT_CONFIG_NOSYSTEM": "1"}
        cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never", "-c", "core.symlinks=false",
               "clone", "--depth", "1", "--single-branch", "--", clean, str(work)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        except subprocess.TimeoutExpired as exc:
            raise RepoError("git clone timed out") from exc
        if r.returncode != 0:
            raise RepoError("git clone failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
        try:
            sha = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 timeout=30, env=env).stdout.strip()
        except subprocess.SubprocessError:
            sha = ""
        shutil.rmtree(work / ".git", ignore_errors=True)
        info = {"source": clean, "commit": sha}
    else:
        if not settings.allow_local_paths:
            raise RepoError("local paths are disabled (set REPROFIX_ALLOW_LOCAL_PATHS=1 on a trusted dev machine)")
        src = Path(local_path).expanduser().resolve()
        if not src.is_dir():
            raise RepoError(f"not a directory: {src}")
        shutil.copytree(src, work, ignore=shutil.ignore_patterns(*IGNORE_DIRS), symlinks=True)
        info = {"source": str(src), "commit": ""}
    info["symlinks_removed"] = _strip_symlinks(work)
    if _count_files(work) > MAX_FILES:
        raise RepoError(f"repository has more than {MAX_FILES} files")
    size = _dir_size_mb(work)
    if size > settings.max_repo_mb:
        raise RepoError(f"repository is {size:.0f} MB; limit is {settings.max_repo_mb} MB")
    shutil.copytree(work, orig, symlinks=False)
    info["size_mb"] = round(size, 2)
    return info


# --------------------------------------------------------------------------- scanning
FRAMEWORKS = {
    "torch": r"^\s*(?:import|from)\s+torch\b",
    "tensorflow": r"^\s*(?:import|from)\s+tensorflow\b",
    "jax": r"^\s*(?:import|from)\s+jax\b",
    "sklearn": r"^\s*(?:import|from)\s+sklearn\b",
    "transformers": r"^\s*(?:import|from)\s+transformers\b",
    "keras": r"^\s*(?:import|from)\s+keras\b",
    "numpy": r"^\s*(?:import|from)\s+numpy\b",
}
ENTRY_PRIORITY = ["train.py", "main.py", "run.py", "eval.py", "evaluate.py", "test.py", "app.py"]
CLAIM_RE = re.compile(
    r"(?P<hint>accuracy|acc\b|top-?1|top-?5|f1|auc|map\b|bleu|rouge|perplexity|loss)[^\n0-9%]{0,40}"
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<pct>%)?", re.I)


@dataclass
class RepoScan:
    files: list[str] = field(default_factory=list)
    py_files: int = 0
    requirements: list[str] = field(default_factory=list)
    readme_path: str | None = None
    readme_text: str = ""
    entry_points: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    has_tests: bool = False
    config_files: list[str] = field(default_factory=list)
    readme_claims: list[dict] = field(default_factory=list)
    readme_commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files": self.files[:200], "n_files": len(self.files), "py_files": self.py_files,
            "requirements": self.requirements, "readme": self.readme_path, "entry_points": self.entry_points,
            "frameworks": self.frameworks, "has_tests": self.has_tests, "config_files": self.config_files,
            "readme_claims": self.readme_claims, "readme_commands": self.readme_commands,
        }


def _read_text(p: Path, cap: int = 200_000) -> str:
    try:
        return p.read_bytes()[:cap].decode("utf-8", "replace")
    except OSError:
        return ""


def scan_repo(workdir: Path) -> RepoScan:
    scan = RepoScan()
    fw_hits: dict[str, int] = {}
    for dp, dn, fn in os.walk(workdir):
        dn[:] = sorted(d for d in dn if d not in IGNORE_DIRS)
        for f in sorted(fn):
            rel = (Path(dp) / f).relative_to(workdir).as_posix()
            scan.files.append(rel)
            low = f.lower()
            if re.fullmatch(r"requirements[\w\-.]*\.txt", low):
                scan.requirements.append(rel)
            if low.startswith("readme") and scan.readme_path is None and "/" not in rel:
                scan.readme_path = rel
            if low.endswith((".json", ".yaml", ".yml", ".toml", ".cfg", ".ini")) and "/" not in rel:
                scan.config_files.append(rel)
            if low.endswith(".py"):
                scan.py_files += 1
                text = _read_text(Path(dp) / f)
                if re.search(r"""if\s+__name__\s*==\s*['"]__main__['"]""", text):
                    scan.entry_points.append(rel)
                for name, pat in FRAMEWORKS.items():
                    if re.search(pat, text, re.M):
                        fw_hits[name] = fw_hits.get(name, 0) + 1
            if len(scan.files) > MAX_FILES:
                break
    scan.entry_points.sort(key=lambda p: (ENTRY_PRIORITY.index(Path(p).name) if Path(p).name in ENTRY_PRIORITY else 99, p))
    scan.frameworks = sorted(fw_hits, key=lambda k: -fw_hits[k])
    scan.has_tests = any(f.startswith("tests/") or re.search(r"(^|/)test_[^/]*\.py$", f) for f in scan.files)
    # requirements files that name a framework count even if never imported directly
    for req in scan.requirements:
        for line in _read_text(workdir / req).splitlines():
            m = re.match(r"\s*([A-Za-z0-9_.\-]+)", line)
            if m and m.group(1).lower() in {"torch", "tensorflow", "jax", "scikit-learn", "transformers", "keras"}:
                name = {"scikit-learn": "sklearn"}.get(m.group(1).lower(), m.group(1).lower())
                if name not in scan.frameworks:
                    scan.frameworks.append(name)
    if scan.readme_path:
        scan.readme_text = _read_text(workdir / scan.readme_path, 12_000).replace("\r\n", "\n")
        for m in CLAIM_RE.finditer(scan.readme_text):
            val = float(m.group("val"))
            if m.group("pct"):
                val /= 100.0
            line = scan.readme_text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ").strip()
            scan.readme_claims.append({"metric_hint": m.group("hint").lower(), "value": val, "context": line})
        scan.readme_claims = scan.readme_claims[:8]
        for block in re.findall(r"```[a-z]*\n(.*?)```", scan.readme_text, flags=re.S):
            for line in block.splitlines():
                line = line.strip().lstrip("$ ").strip()
                if line.startswith(("python ", "python3 ", "pytest")):
                    try:
                        parse_command(line)
                        scan.readme_commands.append(line)
                    except UnsafeCommand:
                        continue
    return scan
