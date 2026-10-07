import os
import shutil
from pathlib import Path

import pytest

from reprofix.config import Settings
from reprofix.core.repo import RepoError, acquire, scan_repo, validate_git_url

HOSTS = ("github.com",)


@pytest.mark.parametrize("url,clean", [
    ("https://github.com/owner/repo", "https://github.com/owner/repo"),
    ("https://github.com/owner/repo.git", "https://github.com/owner/repo.git"),
    ("https://github.com/owner/repo/", "https://github.com/owner/repo"),
    ("https://GitHub.com/Owner/Re-po_1.x", "https://github.com/Owner/Re-po_1.x"),
    ("  https://github.com/o/r  ", "https://github.com/o/r"),
])
def test_valid_urls(url, clean):
    assert validate_git_url(url, HOSTS) == clean


@pytest.mark.parametrize("url", [
    "http://github.com/o/r", "git@github.com:o/r.git", "ssh://github.com/o/r", "file:///etc/passwd", "/home/me/repo",
    "https://user:pw@github.com/o/r", "https://github.com@evil.com/o/r", "https://evil.com/o/r", "https://github.com.evil.com/o/r",
    "https://github.com/o", "https://github.com/o/r/tree/main",
    "https://github.com/../r", "https://github.com/o/..", "https://github.com/./r", "https://github.com/o r/x", "",
    "https://github.com//r",
])
def test_invalid_urls(url):
    with pytest.raises(RepoError):
        validate_git_url(url, HOSTS)


def test_query_and_fragment_are_dropped_from_the_cloned_url():
    # only the rebuilt https://host/owner/repo is ever passed to git
    assert validate_git_url("https://github.com/o/r?x=1", HOSTS) == "https://github.com/o/r"
    assert validate_git_url("https://github.com/o/r#frag", HOSTS) == "https://github.com/o/r"


def test_extra_hosts_must_be_allow_listed():
    assert validate_git_url("https://gitlab.com/o/r", ("github.com", "gitlab.com"))
    with pytest.raises(RepoError):
        validate_git_url("https://gitlab.com/o/r", HOSTS)


def _acquire(settings, tmp_path, **kw):
    return acquire(settings=settings, work=tmp_path / "w", orig=tmp_path / "o", **kw)


def test_exactly_one_source_is_required(settings, tmp_path):
    with pytest.raises(RepoError):
        _acquire(settings, tmp_path, repo_url=None, local_path=None)
    with pytest.raises(RepoError):
        _acquire(settings, tmp_path, repo_url="https://github.com/o/r", local_path=str(tmp_path))


def test_local_paths_are_off_by_default(tiny_repo, tmp_path):
    with pytest.raises(RepoError, match="disabled"):
        _acquire(Settings(data_dir=tmp_path / "d"), tmp_path, repo_url=None, local_path=str(tiny_repo))


def test_local_acquire_copies_snapshots_and_ignores_vcs_and_envs(settings, tiny_repo, tmp_path):
    (tiny_repo / ".git").mkdir()
    (tiny_repo / ".git" / "config").write_text("secret")
    (tiny_repo / "venv").mkdir()
    (tiny_repo / "venv" / "x.py").write_text("x")
    info = _acquire(settings, tmp_path, repo_url=None, local_path=str(tiny_repo))
    assert (tmp_path / "w" / "train.py").exists() and (tmp_path / "o" / "train.py").exists()
    assert not (tmp_path / "w" / ".git").exists() and not (tmp_path / "w" / "venv").exists()
    assert info["symlinks_removed"] == 0 and info["size_mb"] >= 0
    (tmp_path / "w" / "train.py").write_text("changed")           # the snapshot stays pristine
    assert "accuracy" in (tmp_path / "o" / "train.py").read_text()


def test_symlinks_in_untrusted_repos_are_removed(settings, tiny_repo, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    try:
        (tiny_repo / "leak.txt").symlink_to(secret)
        (tiny_repo / "linkdir").symlink_to(tmp_path)
    except OSError:
        pytest.skip("symlinks not supported or permitted on this system")
    info = _acquire(settings, tmp_path, repo_url=None, local_path=str(tiny_repo))
    assert info["symlinks_removed"] == 2
    assert not os.path.lexists(tmp_path / "w" / "leak.txt") and not os.path.lexists(tmp_path / "o" / "linkdir")


def test_size_limit(settings, tiny_repo, tmp_path):
    (tiny_repo / "big.bin").write_bytes(b"\0" * (2 * 1024 * 1024))
    settings.max_repo_mb = 1
    with pytest.raises(RepoError, match="limit"):
        _acquire(settings, tmp_path, repo_url=None, local_path=str(tiny_repo))


def test_not_a_directory(settings, tmp_path):
    with pytest.raises(RepoError, match="not a directory"):
        _acquire(settings, tmp_path, repo_url=None, local_path=str(tmp_path / "nope"))


def test_scan_finds_entry_points_frameworks_claims_and_commands(tmp_path):
    r = tmp_path / "r"
    (r / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "train.py").write_text("import numpy as np\nif __name__ == '__main__':\n    pass\n")
    (r / "pkg" / "util.py").write_text("import torch\nif __name__ == \"__main__\":\n    pass\n")
    (r / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    (r / "requirements.txt").write_text("numpy==1.26.4\nscikit-learn\n")
    (r / "README.md").write_text("# Demo\n\nWe reach top-1 accuracy 87.5% and F1: 0.91.\n\n```bash\npython train.py --epochs 3\n```\n")
    s = scan_repo(r)
    assert s.entry_points[0] == "train.py" and "pkg/util.py" in s.entry_points      # conventional names rank first
    assert {"numpy", "torch", "sklearn"} <= set(s.frameworks)
    assert s.has_tests and s.requirements == ["requirements.txt"] and s.readme_path == "README.md"
    values = {round(c["value"], 3) for c in s.readme_claims}
    assert 0.875 in values and 0.91 in values                                       # percentages are normalized
    assert "python train.py --epochs 3" in s.readme_commands
