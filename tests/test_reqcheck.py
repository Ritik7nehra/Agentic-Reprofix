"""The wheels-only install rule must hold against a repository's own requirements files (reprofix/sandbox/reqcheck.py)."""
from pathlib import Path

import pytest

from reprofix.config import Settings
from reprofix.sandbox.base import refuse_unsafe_requirements
from reprofix.sandbox.local_backend import LocalSandbox
from reprofix.sandbox.reqcheck import check_requirement_files


def probs(tmp_path: Path, text: str, name: str = "requirements.txt") -> list[str]:
    (tmp_path / name).write_text(text)
    return check_requirement_files(tmp_path, [name])


@pytest.mark.parametrize("text", [
    "numpy>=1.26\npytest>=8\n",
    "numpy==2.1.0\n",
    "numpy<2\n",
    "torch>=2.2  # the CPU build is fine\n",
    "Pillow>=10,<12\nscikit-learn[extra]>=1.3 ; python_version >= '3.9'\n",
    "requests==2.32.3 --hash=sha256:" + "a" * 64 + "\n",
    "# only a comment\n\n   \n",
    "--index-url https://pypi.org/simple\nnumpy\n",
    "--extra-index-url=https://download.pytorch.org/whl/cpu\ntorch\n",
    "--prefer-binary\n--pre\n--only-binary :all:\nnumpy\n",
    "numpy \\\n  >=1.26\n",
    "pkg===1.0\nother~=2.1\nthird!=3.0\n",
    "zipp>=3.1\ntarfile-utils==1.0\n",
])
def test_ordinary_requirements_are_accepted(tmp_path, text):
    assert probs(tmp_path, text) == []


@pytest.mark.parametrize("text,why", [
    ("--no-binary :all:\nnumpy\n", "re-enables source builds"),
    ("--no-binary=numpy\nnumpy\n", "re-enables source builds"),
    ("--only-binary :none:\nnumpy\n", "only `--only-binary :all:`"),
    ("-e .\n", "editable"),
    ("--editable ./pkg\n", "editable"),
    ("-e git+https://github.com/x/y.git#egg=y\n", "editable"),
    ("./pkg\n", "plain"),
    (".\n", "plain"),
    ("/abs/pkg\n", "plain"),
    ("pkg.tar.gz\n", "file name"),
    ("pkg-1.0.zip\n", "file name"),
    ("pkg @ https://files.pythonhosted.org/packages/x/pkg-1.0.tar.gz\n", "direct reference"),
    ("pkg @ file:///tmp/pkg.tar.gz\n", "direct reference"),
    ("git+https://github.com/x/y.git\n", "direct reference"),
    ("https://example.com/pkg.zip\n", "direct reference"),
    ("numpy --install-option=--x\n", "per-requirement option"),
    ("numpy --global-option=-q\n", "per-requirement option"),
    ("numpy --no-binary numpy\n", "per-requirement option"),
    ("--find-links ./wheels\nnumpy\n", "find-links"),
    ("-f https://example.com/links\nnumpy\n", "find-links"),
    ("--trusted-host example.com\nnumpy\n", "TLS"),
    ("--index-url http://example.com/simple\nnumpy\n", "https"),
    ("--no-index\nnumpy\n", "where packages come from"),
    ("--config-settings=x=y\nnumpy\n", "options to a build"),
    ("-r https://example.com/more.txt\n", "not a path"),
    ("numpy; python_version<'3'\n./pkg\n", "plain"),
    ("--use-feature=fast-deps\n", "pip builds"),
    ("-X\n", "few that are accepted"),
])
def test_anything_that_could_build_or_run_repository_code_is_refused(tmp_path, text, why):
    p = probs(tmp_path, text)
    assert p, text
    assert any(why in x for x in p), p
    assert all(x.startswith("requirements.txt:") for x in p)             # the file and line are named


def test_the_line_number_survives_comments_and_continuations(tmp_path):
    p = probs(tmp_path, "# header\nnumpy \\\n  >=1\n\n-e .\n")
    assert p and p[0].startswith("requirements.txt:5:")


def test_nested_includes_are_checked_and_cannot_leave_the_repository(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "extra.txt").write_text("numpy\n-e .\n")
    (tmp_path / "requirements.txt").write_text("-r base/extra.txt\n")
    p = check_requirement_files(tmp_path, ["requirements.txt"])
    assert len(p) == 1 and p[0].startswith("base/extra.txt:2:") and "editable" in p[0]
    for text in ("-r ../outside.txt\n", "-r /etc/passwd\n", "--requirement=../../x\n", "-c ../c.txt\n"):
        assert any("outside the repository" in x for x in probs(tmp_path, text)), text


def test_a_symlink_that_leaves_the_repository_is_refused(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("numpy\n")
    work = tmp_path / "work"
    work.mkdir()
    try:
        (work / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported or permitted on this system")
    (work / "requirements.txt").write_text("-r link.txt\n")
    assert any("outside the repository" in x for x in check_requirement_files(work, ["requirements.txt"]))


def test_include_cycles_and_missing_files_do_not_loop_or_crash(tmp_path):
    (tmp_path / "a.txt").write_text("-r b.txt\nnumpy\n")
    (tmp_path / "b.txt").write_text("-r a.txt\n-r missing.txt\n")
    assert check_requirement_files(tmp_path, ["a.txt"]) == []


def test_at_most_a_handful_of_problems_are_reported(tmp_path):
    p = probs(tmp_path, "-e .\n" * 50)
    assert 0 < len(p) <= 6


def test_refusal_is_an_install_result_with_the_reason_and_runs_nothing(tmp_path):
    (tmp_path / "requirements.txt").write_text("-e .\n")
    res = refuse_unsafe_requirements(tmp_path, ["requirements.txt"], allow_sdist=False)
    assert res is not None and res.exit_code == 2 and res.network is False and res.duration_s == 0.0
    assert "refused to install" in res.stderr and "requirements.txt:1" in res.stderr and "REPROFIX_ALLOW_SDIST=1" in res.stderr
    assert refuse_unsafe_requirements(tmp_path, ["requirements.txt"], allow_sdist=True) is None      # the operator's explicit choice
    (tmp_path / "ok.txt").write_text("numpy\n")
    assert refuse_unsafe_requirements(tmp_path, ["ok.txt"], allow_sdist=False) is None


def test_the_local_backend_refuses_before_pip_is_started_and_the_package_never_builds(tmp_path, monkeypatch):
    marker = tmp_path / "BUILT"
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "setup.py").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('setup.py ran')\n"
                                  "from setuptools import setup; setup(name='evilpkg', version='0.1')\n")
    (tmp_path / "requirements.txt").write_text("-e ./pkg\n")
    sb = LocalSandbox(Settings(data_dir=tmp_path / "d", sandbox="local", allow_unsafe_local=True))
    monkeypatch.setattr(sb, "_exec", lambda *a, **k: pytest.fail("pip must not be started for a refused requirements file"))
    res = sb.install(tmp_path, ["requirements.txt"], 60)
    assert res.exit_code == 2 and "editable" in res.stderr and not marker.exists()


def test_every_benchmark_task_still_installs(tmp_path):
    root = Path(__file__).resolve().parents[1] / "benchmark"
    files = sorted(root.rglob("requirements*.txt"))
    assert files
    for f in files:
        assert check_requirement_files(f.parent, [f.name]) == [], f


def test_a_refused_requirements_file_is_an_ordinary_install_failure_the_report_explains(settings, tiny_repo, tmp_path):
    from conftest import BUG_LINE, FIX_LINE, hypothesis_reply, repair_reply, scripted
    from test_e2e import orchestrate
    (tiny_repo / "requirements.txt").write_text("-e .\n")
    b = scripted([hypothesis_reply("h1", "the requirements file asks for an editable install", "-e .", category="dependency",
                                   evidence=[{"kind": "code", "file": "requirements.txt", "quote": "-e .", "note": ""}])],
                 [repair_reply(BUG_LINE, FIX_LINE)])
    _, rep, events = orchestrate(settings, tiny_repo, tmp_path, b, attempts=1)
    assert rep["results"]["baseline"]["stage"] == "install_failure"
    first = next(d for k, d in events if k == "exec")                       # the event carries the output the diagnoser reads
    assert first["phase"] == "install" and first["exit_code"] == 2
    assert "refused to install" in first["stderr"] and "requirements.txt:1" in first["stderr"]
