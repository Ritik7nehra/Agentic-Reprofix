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
from ..util import UnsafeCommand, is_setup_command, parse_command, strip_venv_prefix

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
        for k in ("SYSTEMROOT", "SystemRoot", "WINDIR", "windir", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                  "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                  "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            if k in os.environ:
                env[k] = os.environ[k]
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
    _extract_notebooks(work)
    info["symlinks_removed"] = _strip_symlinks(work)
    if _count_files(work) > MAX_FILES:
        raise RepoError(f"repository has more than {MAX_FILES} files")
    size = _dir_size_mb(work)
    if size > settings.max_repo_mb:
        raise RepoError(f"repository is {size:.0f} MB; limit is {settings.max_repo_mb} MB")
    shutil.copytree(work, orig, symlinks=False)
    info["size_mb"] = round(size, 2)
    return info


def _extract_notebooks(root: Path) -> None:
    """When a repository contains Jupyter notebooks and no Python scripts, extract code to .py and requirements."""
    import json
    pip_reqs: list[str] = []
    output_vers: dict[str, str] = {}
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f.endswith(".ipynb") and not f.startswith("."):
                ipynb_p = Path(dp) / f
                py_p = ipynb_p.with_suffix(".py")
                if not py_p.exists():
                    try:
                        nb = json.loads(ipynb_p.read_text(encoding="utf-8", errors="replace"))
                        lines: list[str] = []
                        for cell in nb.get("cells", []):
                            if cell.get("cell_type") == "code":
                                for out in cell.get("outputs", []):
                                    text = "".join(out.get("text", []))
                                    for m in re.finditer(r"Requirement already satisfied:\s*([a-zA-Z0-9_\-]+)\b.*?\(([0-9]+(?:\.[0-9]+)+[a-zA-Z0-9._\-]*)\)", text, re.I):
                                        output_vers[m.group(1).lower()] = m.group(2)
                                for line in cell.get("source", []):
                                    st = line.strip()
                                    if st.startswith(("!pip install ", "%pip install ")):
                                        parts = [x for x in st.split()[2:] if not x.startswith("-") and not x.startswith("%")]
                                        pip_reqs.extend(parts)
                                    elif st.startswith(("!", "%")) or st.endswith("??"):
                                        lines.append(f"# {line}")
                                    elif re.match(r"^\s*while\s+cap\.isOpened\(\)\s*:", line):
                                        indent = re.match(r"^\s*", line).group(0)
                                        lines.append(f"{indent}_loop_cnt = 0\n")
                                        lines.append(f"{indent}while cap.isOpened() and _loop_cnt < 2:\n")
                                        lines.append(f"{indent}    _loop_cnt += 1\n")
                                    else:
                                        lines.append(line)
                                lines.append("\n")
                        if lines:
                            py_p.write_text("".join(lines), encoding="utf-8")
                    except Exception:
                        pass
    req_file = root / "requirements.txt"
    if pip_reqs and not req_file.exists():
        valid: list[str] = []
        for r in pip_reqs:
            if not re.match(r"^[A-Za-z0-9_.\-]+$", r):
                continue
            r_lower = r.lower()
            if r_lower == "mediapipe":
                # MediaPipe removed the 'solutions' module after 0.10.14; pin compatible 0.10.14 with Python 3.12 wheels
                valid.append("mediapipe<=0.10.14,>=0.10.0")
                if "protobuf" not in [p.lower() for p in pip_reqs]:
                    valid.append("protobuf<5,>=4.25.3")
            elif r_lower in output_vers and output_vers[r_lower]:
                ver = output_vers[r_lower]
                valid.append(f"{r}=={ver}")
            else:
                valid.append(r)
        if valid:
            req_file.write_text("\n".join(dict.fromkeys(valid)) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- scanning
FRAMEWORKS = {
    "torch": r"^\s*(?:import|from)\s+torch\b",
    "tensorflow": r"^\s*(?:import|from)\s+tensorflow\b",
    "jax": r"^\s*(?:import|from)\s+jax\b",
    "sklearn": r"^\s*(?:import|from)\s+sklearn\b",
    "transformers": r"^\s*(?:import|from)\s+transformers\b",
    "keras": r"^\s*(?:import|from)\s+keras\b",
    "numpy": r"^\s*(?:import|from)\s+numpy\b",
    "cv2": r"^\s*(?:import|from)\s+cv2\b",
    "mediapipe": r"^\s*(?:import|from)\s+mediapipe\b",
}
ENTRY_PRIORITY = ["train.py", "main.py", "run.py", "eval.py", "evaluate.py", "test.py", "app.py"]
_HINTS = r"accuracy|acc\b|top-?[15](?:\s+accuracy)?|f1|auc|map\b|bleu|rouge|perplexity|loss"
# metric word, then a number within 40 characters: not across a table cell ("|") and not across another metric word
# ("0.87 f1 and accuracy: 0.9" gives f1 0.87 and accuracy 0.9, not f1 0.9)
CLAIM_RE = re.compile(
    rf"(?P<hint>{_HINTS})(?:(?!{_HINTS})[^\n0-9%|]){{0,40}}"
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<pct>%)?", re.I)
# "28.4 BLEU", "91% accuracy": the number comes first. Not "Top-1 accuracy" (the 1 belongs to the name) and not a year.
LEAD_CLAIM_RE = re.compile(
    r"(?<![\w.+\-\u2212\u00b1])(?P<val>\d{1,3}(?:\.\d+)?)\s*(?P<pct>%)?\s*(?P<hint>accuracy|acc\b|f1|auc|bleu|rouge)", re.I)
_DELTA_BEFORE = re.compile(r"[+\-\u2212\u2013\u00b1]\s*$")                   # "(+2 points": a difference, not a result
_DELTA_AFTER = re.compile(r"\s*(?:points?\b|pts?\b|pp\b|percentage\s+points?\b|x\b|\u00d7)", re.I)


def _readme_claims(text: str) -> list[dict]:
    """Metric-looking numbers in README text, in reading order, without differences ("+2 points over the baseline")."""
    found: list[tuple[int, dict]] = []
    for rx in (CLAIM_RE, LEAD_CLAIM_RE):
        for m in rx.finditer(text):
            if _DELTA_BEFORE.search(text[: m.start("val")]) or _DELTA_AFTER.match(text, m.end()):
                continue
            val = float(m.group("val"))
            if m.group("pct"):
                val /= 100.0
            hint = m.group("hint").lower().split()[0]
            if any(c["metric_hint"] == hint and abs(c["value"] - val) < 1e-9 for _, c in found):
                continue
            line = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ").strip()
            found.append((m.start(), {"metric_hint": hint, "value": val, "context": line}))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


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
        scan.readme_claims = _readme_claims(scan.readme_text)[:8]
        for block in re.findall(r"```[a-z]*\n(.*?)```", scan.readme_text, flags=re.S):
            for line in block.splitlines():
                line = strip_venv_prefix(line.strip().lstrip("$ ").strip())
                if line.startswith(("python ", "python3 ", "pytest")):
                    try:
                        argv = parse_command(line)
                    except UnsafeCommand:
                        continue
                    if not is_setup_command(argv) and line not in scan.readme_commands:     # `python -m venv venv` runs none of the project
                        scan.readme_commands.append(line)
    return scan
