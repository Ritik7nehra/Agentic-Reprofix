import os

import pytest

from reprofix.config import DEFAULT_MODELS, DEFAULT_PRICES, Settings, load_env_file

KEYS = ["NEBIUS_API_KEY", "TAVILY_API_KEY", "REPROFIX_SANDBOX", "NEBIUS_MODEL_ULTRA", "REPROFIX_PRICE_SUPER", "REPROFIX_PRICE_NANO",
        "REPROFIX_ALLOWED_GIT_HOSTS", "REPROFIX_ALLOW_UNSAFE_LOCAL", "NEBIUS_EXTRA_BODY", "FOO", "BAR", "BAZ", "QUX", "EMPTY"]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")
    yield
    for k in KEYS:                      # load_env_file writes os.environ directly; do not leak between tests
        os.environ.pop(k, None)


def write(tmp_path, text):
    f = tmp_path / ".env"
    f.write_text(text)
    return str(f)


def test_env_file_parsing(tmp_path):
    f = write(tmp_path, '# comment\n\nFOO=plain\nexport BAR="quoted value"\nBAZ=\'single\'\nQUX=value # trailing comment\nEMPTY=\nnot a pair\n')
    assert load_env_file(f) == ["FOO", "BAR", "BAZ", "QUX", "EMPTY"]
    assert (os.environ["FOO"], os.environ["BAR"], os.environ["BAZ"], os.environ["QUX"], os.environ["EMPTY"]) == (
        "plain", "quoted value", "single", "value", "")


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO", "from-shell")
    assert load_env_file(write(tmp_path, "FOO=from-file\nBAR=from-file\n")) == ["BAR"]
    assert os.environ["FOO"] == "from-shell" and os.environ["BAR"] == "from-file"


def test_hash_inside_a_value_is_kept(tmp_path):
    load_env_file(write(tmp_path, "FOO=abc#def\nBAR='x # y'\n"))
    assert os.environ["FOO"] == "abc#def" and os.environ["BAR"] == "x # y"


def test_missing_file_and_disabled_loading_are_noops(tmp_path, monkeypatch):
    assert load_env_file(str(tmp_path / "nope")) == []
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")
    assert load_env_file() == []


def test_from_env_reads_the_env_file_named_by_REPROFIX_ENV_FILE(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROFIX_ENV_FILE", write(tmp_path, "NEBIUS_API_KEY=abc\nREPROFIX_SANDBOX=local\n"))
    s = Settings.from_env()
    assert s.nebius_api_key == "abc" and s.sandbox == "local"


def test_defaults_without_any_configuration():
    s = Settings.from_env()
    assert s.nebius_api_key == "" and s.sandbox == "docker" and s.allow_unsafe_local is False and s.allow_local_paths is False
    assert s.models == DEFAULT_MODELS and s.prices == DEFAULT_PRICES and s.allowed_git_hosts == ("github.com",)
    assert s.nebius_base_url == "https://api.tokenfactory.nebius.com/v1/"
    # prices from Nebius's model catalog (USD per 1M tokens, input/output)
    assert s.prices == {"nano": (0.06, 0.24), "super": (0.30, 0.90), "ultra": (1.00, 3.00)}


def test_an_empty_price_means_unpriced_not_free(monkeypatch):
    monkeypatch.setenv("REPROFIX_PRICE_NANO", "")
    assert Settings.from_env().prices["nano"] is None


def test_overrides_and_price_parsing(monkeypatch):
    monkeypatch.setenv("NEBIUS_MODEL_ULTRA", "nvidia/other-ultra")
    monkeypatch.setenv("REPROFIX_PRICE_SUPER", "0.5, 1.5")
    monkeypatch.setenv("REPROFIX_PRICE_NANO", "0.1,0.2")
    monkeypatch.setenv("REPROFIX_ALLOWED_GIT_HOSTS", "GitHub.com, gitlab.com")
    monkeypatch.setenv("REPROFIX_ALLOW_UNSAFE_LOCAL", "yes")
    monkeypatch.setenv("NEBIUS_EXTRA_BODY", '{"top_p": 0.9}')
    s = Settings.from_env()
    assert s.models["ultra"] == "nvidia/other-ultra" and s.prices["super"] == (0.5, 1.5) and s.prices["nano"] == (0.1, 0.2)
    assert s.allowed_git_hosts == ("github.com", "gitlab.com") and s.allow_unsafe_local is True and s.extra_body == {"top_p": 0.9}


def test_a_malformed_price_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("REPROFIX_PRICE_SUPER", "cheap")
    with pytest.raises(ValueError, match="REPROFIX_PRICE_SUPER"):
        Settings.from_env()


def test_known_issues_search_is_on_by_default_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("REPROFIX_KNOWN_ISSUES", raising=False)
    assert Settings.from_env().known_issues is True
    for off in ("0", "false", "no"):
        monkeypatch.setenv("REPROFIX_KNOWN_ISSUES", off)
        assert Settings.from_env().known_issues is False, off
    monkeypatch.setenv("REPROFIX_KNOWN_ISSUES", "1")
    assert Settings.from_env().known_issues is True


def test_env_example_states_the_same_known_issues_default_as_the_code(monkeypatch):
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert [ln for ln in text.splitlines() if ln.startswith("REPROFIX_KNOWN_ISSUES=")] == ["REPROFIX_KNOWN_ISSUES=1"]
    monkeypatch.delenv("REPROFIX_KNOWN_ISSUES", raising=False)
    assert Settings.from_env().known_issues is True
