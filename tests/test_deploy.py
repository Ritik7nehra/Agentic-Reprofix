"""Static checks on the deployment files.

These do NOT show that the image builds or that the stack starts: no image was built and no stack was brought up
while this repository was written. They catch the mistakes a text review would: a secret baked into the image,
the data directory mounted at different paths (which silently breaks the Docker sandbox), a port published on
all interfaces, a missing token.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
DOCKERFILE = (ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (ROOT / ".dockerignore").read_text().splitlines()
APP = COMPOSE["services"]["reprofix"]


def test_compose_has_the_expected_services_and_the_tls_one_is_optional():
    assert set(COMPOSE["services"]) == {"sandbox-image", "egress", "reprofix", "caddy"}
    assert COMPOSE["services"]["caddy"]["profiles"] == ["tls"]
    assert APP["depends_on"]["sandbox-image"]["condition"] == "service_completed_successfully"
    assert APP["depends_on"]["egress"]["condition"] == "service_started"


def test_the_egress_proxy_is_the_only_service_with_a_foot_in_both_networks_and_is_locked_down():
    """Install containers are started by the app on the internal network `reprofix-install`; the proxy is how they reach PyPI."""
    egress = COMPOSE["services"]["egress"]
    assert COMPOSE["networks"]["install"] == {"name": "reprofix-install", "internal": True}
    assert sorted(egress["networks"]) == ["default", "install"]
    assert "networks" not in APP, "the app must stay on the default network only"
    assert egress["container_name"] == "reprofix-egress"                    # the app looks the proxy up by this name
    assert egress["command"][:2] == ["python", "/opt/reprofix/egress.py"]
    assert egress["user"] == "1000:1000" and egress["read_only"] is True and egress["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in egress["security_opt"] and egress["pids_limit"] and egress["mem_limit"]
    assert not egress.get("ports") and not egress.get("volumes"), "the proxy publishes no port and mounts nothing"
    assert egress["image"] == "reprofix-sandbox:latest"
    # the app checks this label before every install; without it a compose-managed proxy could never be verified
    assert egress["labels"]["reprofix.egress.allow"] == egress["environment"]["REPROFIX_EGRESS_ALLOW"]
    from reprofix.sandbox.egress_ctl import ALLOW_LABEL
    assert ALLOW_LABEL in egress["labels"]


def test_proxy_mode_is_the_compose_default_and_the_allow_list_defaults_to_pypi():
    env = APP["environment"]
    assert env["REPROFIX_INSTALL_NETWORK"] == "${REPROFIX_INSTALL_NETWORK:-proxy}"
    assert env["REPROFIX_EGRESS_ALLOW"] == COMPOSE["services"]["egress"]["environment"]["REPROFIX_EGRESS_ALLOW"]
    assert env["REPROFIX_EGRESS_ALLOW"].endswith(":-pypi.org,files.pythonhosted.org}")


def test_the_data_directory_is_mounted_at_the_same_path_inside_and_out():
    """The host daemon resolves `-v <workdir>:/work` on the HOST; a different path inside the app container breaks every run."""
    data_dir = APP["environment"]["REPROFIX_DATA_DIR"]                      # e.g. ${REPROFIX_HOST_DATA_DIR:-/srv/reprofix}
    mounts = [v for v in APP["volumes"] if "docker.sock" not in v]
    assert mounts == [f"{data_dir}:{data_dir}"]


def test_the_docker_socket_is_the_only_extra_mount_and_nothing_is_privileged():
    assert [v for v in APP["volumes"] if "docker.sock" in v] == ["/var/run/docker.sock:/var/run/docker.sock"]
    for svc in COMPOSE["services"].values():
        assert not svc.get("privileged") and svc.get("network_mode") != "host" and not svc.get("pid") and not svc.get("cap_add")
    assert APP["cap_drop"] == ["ALL"] and "no-new-privileges:true" in APP["security_opt"]
    assert APP["user"] == "1000:1000"                                       # not root; reaches the socket through group_add


def test_the_app_port_is_published_on_loopback_only_and_a_token_is_required():
    assert APP["ports"] == ["127.0.0.1:8000:8000"]
    token = APP["environment"]["REPROFIX_API_TOKEN"]
    assert token.startswith("${REPROFIX_API_TOKEN:?"), "compose must refuse to start without an API token"
    assert APP["group_add"][0].startswith("${DOCKER_GID:?")
    assert APP["environment"]["REPROFIX_ALLOW_UNSAFE_LOCAL"] == "0" and APP["environment"]["REPROFIX_ALLOW_LOCAL_PATHS"] == "0"
    assert APP["environment"]["REPROFIX_SANDBOX"] == "docker"


def test_optional_service_variables_have_defaults_because_compose_interpolates_inactive_services():
    """A required variable on the tls service would stop a plain `docker compose up` (seen with `docker compose config`)."""
    caddy = COMPOSE["services"]["caddy"]
    assert caddy["environment"]["REPROFIX_DOMAIN"].startswith("${REPROFIX_DOMAIN:-")


def test_the_sandbox_image_the_app_asks_for_is_the_one_compose_builds():
    sandbox = COMPOSE["services"]["sandbox-image"]
    assert sandbox["image"] == APP["environment"]["REPROFIX_SANDBOX_IMAGE"] == "reprofix-sandbox:latest"
    assert (ROOT / sandbox["build"]["dockerfile"]).is_file()


def test_the_dockerfile_runs_as_a_non_root_user_and_ships_what_the_server_shells_out_to():
    assert re.search(r"^USER 1000:1000$", DOCKERFILE, re.M)
    for tool in ("git", "patch"):                                           # core/repo.py, core/report.py, core/pullrequest.py
        assert re.search(rf"apt-get install[^\n]*\b{tool}\b", DOCKERFILE)
    assert "COPY --from=dockercli /usr/local/bin/docker" in DOCKERFILE      # DockerSandbox shells out to the docker CLI
    assert "HEALTHCHECK" in DOCKERFILE and "/api/health" in DOCKERFILE
    assert 'CMD ["reprofix", "serve", "--host", "0.0.0.0", "--port", "8000"]' in DOCKERFILE and "EXPOSE 8000" in DOCKERFILE


def test_nothing_secret_can_be_copied_into_the_image():
    assert ".env" in DOCKERIGNORE and "data/" in DOCKERIGNORE
    copied = re.findall(r"^COPY(?: --from=\S+)? (.+)$", DOCKERFILE, re.M)
    assert not any(".env" in c.replace(".env.example", "") for c in copied)
    for needed in ("pyproject.toml", "reprofix", "web", "benchmark"):       # and the files the build needs are not ignored
        assert needed not in DOCKERIGNORE and f"{needed}/" not in DOCKERIGNORE
    assert (ROOT / "tests").is_dir() and "tests/" in DOCKERIGNORE           # tests stay out of the image


def test_the_sandbox_image_ships_the_proxy_script_the_egress_service_runs():
    sandbox = (ROOT / "docker" / "sandbox.Dockerfile").read_text()
    assert "COPY reprofix/sandbox/egress.py /opt/reprofix/egress.py" in sandbox
    assert sandbox.index("COPY reprofix/sandbox/egress.py") < sandbox.index("USER 1000:1000")   # copied while still root
    assert (ROOT / "reprofix" / "sandbox" / "egress.py").is_file()
    assert not (ROOT / ".dockerignore").read_text().count("reprofix/sandbox")


def test_what_the_dockerfile_copies_exists():
    for src in ("pyproject.toml", "README.md", "LICENSE", "reprofix", "web", "benchmark"):
        assert (ROOT / src).exists(), src


def test_installed_layout_lets_the_server_find_the_ui_and_the_benchmark():
    """The image does an editable install in /app, so the code must locate web/ and benchmark/ relative to the package."""
    import reprofix.api.app as app
    assert app.WEB_DIR == ROOT / "web" and app.BENCH_DIR == ROOT / "benchmark"


def test_caddy_does_not_buffer_the_progress_stream():
    caddyfile = (ROOT / "deploy" / "Caddyfile").read_text()
    assert "flush_interval -1" in caddyfile and "reverse_proxy reprofix:8000" in caddyfile and "{$REPROFIX_DOMAIN}" in caddyfile


def test_systemd_unit_listens_on_loopback_and_keeps_secrets_out_of_the_file():
    unit = (ROOT / "deploy" / "reprofix.service").read_text()
    assert "--host 127.0.0.1" in unit and "EnvironmentFile=/etc/reprofix.env" in unit
    assert "Environment=REPROFIX_INSTALL_NETWORK=proxy" in unit and "ExecStartPre=/opt/reprofix/.venv/bin/reprofix egress up" in unit
    assert "NoNewPrivileges=true" in unit and "SupplementaryGroups=docker" in unit
    assert not re.search(r"(API_KEY|API_TOKEN)=\S", unit)


def test_env_example_documents_every_variable_the_deployment_files_need():
    env = (ROOT / ".env.example").read_text()
    for var in ("REPROFIX_API_TOKEN", "NEBIUS_API_KEY", "DOCKER_GID", "REPROFIX_HOST_DATA_DIR", "REPROFIX_DOMAIN",
                "REPROFIX_ALLOW_PULL_REQUESTS", "REPROFIX_INSTALL_NETWORK", "REPROFIX_EGRESS_ALLOW", "REPROFIX_EGRESS_UPSTREAM",
                "REPROFIX_KNOWN_ISSUES", "REPROFIX_SANDBOX_GPU"):
        assert re.search(rf"^#? ?{var}=", env, re.M), var


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs the docker CLI (no daemon is required for `compose config`)")
def test_docker_compose_accepts_the_file_and_enforces_the_required_variables(tmp_path):
    def config(env: dict[str, str], *extra: str):
        import os
        return subprocess.run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), *extra, "config"], capture_output=True,
                              text=True, env={**os.environ, "DOCKER_GID": "", "REPROFIX_API_TOKEN": "", **env}, cwd=tmp_path)
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("docker compose plugin not installed")
    bad = config({})
    assert bad.returncode != 0 and "REPROFIX_API_TOKEN" in bad.stderr
    ok = config({"DOCKER_GID": "999", "REPROFIX_API_TOKEN": "t", "REPROFIX_HOST_DATA_DIR": "/srv/x"})
    assert ok.returncode == 0, ok.stderr
    assert "source: /srv/x" in ok.stdout and "target: /srv/x" in ok.stdout
    assert "internal: true" in ok.stdout and "name: reprofix-install" in ok.stdout and "REPROFIX_INSTALL_NETWORK: proxy" in ok.stdout
    off = config({"DOCKER_GID": "999", "REPROFIX_API_TOKEN": "t", "REPROFIX_INSTALL_NETWORK": "open"})
    assert off.returncode == 0 and "REPROFIX_INSTALL_NETWORK: open" in off.stdout
    tls = config({"DOCKER_GID": "999", "REPROFIX_API_TOKEN": "t"}, "--profile", "tls")
    assert tls.returncode == 0 and "caddy:" in tls.stdout
