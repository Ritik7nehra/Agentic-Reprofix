# ReproFix application image: the API, the web UI and the bundled benchmark. This is NOT the sandbox image
# (that is docker/sandbox.Dockerfile). See docs/deployment.md before using it.
#
# STATUS: written without being built. This repository was authored where no image could be pulled, so this file
# has been checked statically (tests/test_deploy.py, `docker compose config`) and never built or run.
#
# The container starts sandbox containers as SIBLINGS through the host's Docker socket, so it needs the docker CLI
# (copied from the official CLI image) and, at run time, the socket. Anything that can reach that socket can start
# a privileged container and become root on the host: read "Known gaps", item 5, in docs/security.md.
ARG PYTHON_IMAGE=python:3.12-slim
ARG DOCKER_CLI_IMAGE=docker:cli

FROM ${DOCKER_CLI_IMAGE} AS dockercli

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git clones the repositories under repair; patch checks that the final diff applies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

# An editable install keeps the package next to web/ and benchmark/, which is where the server and
# `reprofix demo` look for them.
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY reprofix ./reprofix
COPY web ./web
COPY benchmark ./benchmark
RUN pip install -e .

# uid 1000 matches the sandbox containers' user, so run workspaces are owned consistently.
RUN useradd --create-home --uid 1000 reprofix
USER 1000:1000
ENV HOME=/home/reprofix \
    REPROFIX_ENV_FILE=""

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# 0.0.0.0 is the address INSIDE the container; docker-compose.yml publishes the port on 127.0.0.1 only.
CMD ["reprofix", "serve", "--host", "0.0.0.0", "--port", "8000"]
