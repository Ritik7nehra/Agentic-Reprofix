"""Check a repository's requirements files before pip reads them.

`pip install --only-binary=:all:` is how ReproFix keeps repository code from running during the install phase (the one phase
with a network). It is not enough on its own: a requirements file can switch it off or go around it, and pip then builds a
source tree or archive and runs its `setup.py` or build backend. Each of these was run against pip 24.0 and executed code
despite the flag:

    --no-binary :all:                 (a global option in the file overrides the command line)
    -e ./pkg        ./pkg              (an editable install or a local directory is always built)
    pkg @ https://host/pkg.tar.gz      (a direct reference to a source archive)

So, unless REPROFIX_ALLOW_SDIST=1, only these are accepted: plain named requirements (`name[extra]>=1,<2 ; marker`, with
optional `--hash=`), `-r`/`-c` of another file inside the workspace (checked too), `--index-url`/`--extra-index-url` with an
https URL, and `--prefer-binary`, `--pre`, `--require-hashes`, `--only-binary :all:`. Anything else makes the install refuse
with the line and the reason. A refusal is an ordinary install failure: the diagnoser sees the message and can propose
editing the file.

This is a check of the *text* of the files. It does not parse them the way pip does in every corner (pip's own parser is the
authority), so it errs on refusing: a line it cannot read as a plain requirement is refused.
"""
from __future__ import annotations

import re
from pathlib import Path

MAX_FILES = 20
MAX_DEPTH = 5
MAX_PROBLEMS = 6
_COMMENT = re.compile(r"(^|\s+)#.*$")
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_SPEC = r"(?:===|==|!=|~=|<=|>=|<|>)\s*[^\s;,@]+"
_PLAIN = re.compile(rf"^{_NAME}(?:\s*\[[A-Za-z0-9._,\s-]*\])?\s*(?:\(?\s*{_SPEC}(?:\s*,\s*{_SPEC})*\s*\)?)?\s*(?:;.*)?$")
_ARCHIVE = re.compile(r"\.(?:tar\.gz|tgz|tar\.bz2|tbz2?|tar\.xz|txz|tar|zip|whl|egg)\b", re.I)    # pip treats such a name as a file
_HASH = re.compile(r"\s--hash[=\s]\S+")
_FILE_OPTS = {"-r": "requirement", "--requirement": "requirement", "-c": "constraint", "--constraint": "constraint"}
_INDEX_OPTS = {"-i", "--index-url", "--extra-index-url"}
_HARMLESS = {"--prefer-binary", "--pre", "--require-hashes"}
_WHY = {
    "-e": "an editable install builds the project and runs its setup.py or build backend",
    "--editable": "an editable install builds the project and runs its setup.py or build backend",
    "--no-binary": "it re-enables source builds, which run setup.py while the network is reachable",
    "--only-binary": "only `--only-binary :all:` is accepted (anything else can re-enable source builds)",
    "-f": "a find-links location can serve source archives or local trees",
    "--find-links": "a find-links location can serve source archives or local trees",
    "--trusted-host": "it switches off TLS verification for a host",
    "--no-index": "it changes where packages come from",
    "--global-option": "it passes options to a build",
    "--install-option": "it passes options to a build",
    "--config-settings": "it passes options to a build",
    "--use-feature": "it changes how pip builds and resolves",
}


def _logical_lines(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    buf, start = "", 0
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if line.endswith("\\") and not _COMMENT.match(line.lstrip()):
            buf = (buf or "") + line[:-1] + " "
            start = start or no
            continue
        full = (buf + line) if buf else line
        out.append((start or no, _COMMENT.sub("", full).strip()))
        buf, start = "", 0
    if buf:
        out.append((start, _COMMENT.sub("", buf).strip()))
    return [(n, s) for n, s in out if s]


def _split_option(line: str) -> tuple[str, str]:
    """`-r file`, `-rfile`, `--requirement=file`, `--requirement file` -> ("-r" | "--requirement", "file")."""
    parts = re.split(r"(?:=|\s+)", line, maxsplit=1)
    head, rest = parts[0], (parts[1] if len(parts) > 1 else "")
    if not head.startswith("--") and len(head) > 2:          # a short option with its argument attached
        head, rest = head[:2], head[2:] + ((" " + rest) if rest else "")
    return head, rest.strip()


def _inside(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def check_requirement_files(workdir: Path, requirements: list[str]) -> list[str]:
    """Problems that would let pip run repository code during a wheels-only install (empty list: nothing found)."""
    problems: list[str] = []
    seen: set[Path] = set()

    def add(where: str, line: str, why: str) -> None:
        if len(problems) < MAX_PROBLEMS:
            problems.append(f"{where}: {why}: {line[:100]}")

    def visit(rel: str, base: Path, depth: int) -> None:
        path = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        if depth > MAX_DEPTH or len(seen) >= MAX_FILES:
            add(str(rel), rel, f"more than {MAX_DEPTH} levels or {MAX_FILES} files of nested -r/-c includes")
            return
        if not _inside(workdir, path):
            add(str(rel), rel, "a requirements file outside the repository")
            return
        key = path.resolve()
        if key in seen or not path.is_file():
            return                                            # pip reports a missing file itself
        seen.add(key)
        shown = path.resolve().relative_to(workdir.resolve()).as_posix()
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return
        for no, line in _logical_lines(text):
            where = f"{shown}:{no}"
            if line.startswith("-"):
                opt, arg = _split_option(line)
                if opt in _FILE_OPTS:
                    if "://" in arg or not arg:
                        add(where, line, "a requirements file that is not a path in the repository")
                    else:
                        visit(arg, path.parent, depth + 1)
                elif opt in _INDEX_OPTS:
                    if not arg.lower().startswith("https://"):
                        add(where, line, "a package index that is not an https URL")
                elif opt in _HARMLESS:
                    continue
                elif opt == "--only-binary" and arg.replace(" ", "") == ":all:":
                    continue
                else:
                    add(where, line, _WHY.get(opt, "an option other than the few that are accepted in a wheels-only install"))
                continue
            spec = _HASH.sub("", " " + line).strip()
            before_marker = spec.split(";", 1)[0]
            if "@" in before_marker or "://" in before_marker:
                add(where, line, "a direct reference (URL or path); it can be a source archive that gets built")
            elif re.search(r"\s--[A-Za-z]", spec):
                add(where, line, "a per-requirement option other than --hash")
            elif _ARCHIVE.search(before_marker.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0]):
                add(where, line, "a file name (an archive or wheel in the repository); a source archive would be built")
            elif not _PLAIN.match(spec):
                add(where, line, "not a plain `name[extra]>=version` requirement (a path or a local directory would be built)")

    for rel in requirements:
        visit(rel, workdir, 0)
    return problems


def refusal(problems: list[str]) -> str:
    lines = "\n".join("  " + p for p in problems)
    return ("ReproFix refused to install: this repository's requirements would make pip build and run code (setup.py or a build "
            "backend) while the install network is reachable, which a wheels-only install must not do.\n" + lines +
            "\nReplace those lines with pinned package names (wheels), or set REPROFIX_ALLOW_SDIST=1 for a repository you trust.")
