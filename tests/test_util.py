import pytest

from reprofix.util import UnsafeCommand, extract_json, normalize_ws, parse_command, safe_join, truncate_middle


@pytest.mark.parametrize("cmd", ["python train.py", "python3 -m pkg --lr 0.1", "pytest -q tests/", "  python train.py  "])
def test_parse_command_accepts_plain_interpreter_commands(cmd):
    argv = parse_command(cmd)
    assert argv[0] in {"python", "python3", "pytest"}


@pytest.mark.parametrize("cmd", [
    "", "   ", "bash train.sh", "rm -rf /", "python a.py; rm -rf /", "python a.py && ls", "python a.py | tee x",
    "python a.py > out", "python $(whoami)", "python `id`", "python a.py\nrm x", "curl http://x", "sh -c 'python a.py'",
])
def test_parse_command_rejects_shells_and_other_programs(cmd):
    with pytest.raises(UnsafeCommand):
        parse_command(cmd)


def test_parse_command_keeps_quoted_arguments_together():
    assert parse_command('python train.py --name "my run"') == ["python", "train.py", "--name", "my run"]


def test_truncate_middle_keeps_head_and_tail():
    text = "H" * 100 + "M" * 50_000 + "T" * 100
    out, truncated = truncate_middle(text, head=100, tail=100)
    assert truncated and out.startswith("H" * 100) and out.endswith("T" * 100) and "characters omitted" in out
    assert truncate_middle("short")[1] is False


def test_safe_join_confines_paths(tmp_path):
    (tmp_path / "a").mkdir()
    assert safe_join(tmp_path, "a/b.py") == (tmp_path / "a" / "b.py").resolve()
    for bad in ["../x", "a/../../x", "/etc/passwd", "C:\\x", "", "\\x"]:
        with pytest.raises(ValueError):
            safe_join(tmp_path, bad)


def test_safe_join_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported or permitted on this system")
    with pytest.raises(ValueError):
        safe_join(root, "link/secret.txt")


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Here you go:\n```json\n{"a": {"b": [1, 2]}}\n```\nDone.', {"a": {"b": [1, 2]}}),
    ('<think>I should output {"wrong": true}</think>{"right": true}', {"right": True}),
    ('reasoning without opening tag</think>\n{"x": "}"}', {"x": "}"}),
    ('prefix {"s": "has } brace and \\" quote"} suffix', {"s": 'has } brace and " quote'}),
    ('first {"a": 1} then {"b": 2}', {"a": 1}),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "{broken", "[1, 2, 3]", '{"a": }'])
def test_extract_json_returns_none_when_nothing_parses(text):
    assert extract_json(text) is None


def test_normalize_ws():
    assert normalize_ws("  a \n\t b   c ") == "a b c"
