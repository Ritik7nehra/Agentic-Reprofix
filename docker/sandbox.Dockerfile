# Sandbox image for ReproFix. Build:  docker build -f docker/sandbox.Dockerfile -t reprofix-sandbox:latest .
#
# Deliberately minimal: a plain Python interpreter and pip. Repository dependencies are installed
# into the mounted workspace (.deps) by ReproFix, not baked into the image.
#
# For PyTorch/CUDA repositories, build from a CUDA-enabled base instead, e.g.:
#   docker build -f docker/sandbox.Dockerfile --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime -t reprofix-sandbox:cuda .
# and run with REPROFIX_SANDBOX_IMAGE=reprofix-sandbox:cuda REPROFIX_SANDBOX_GPU=1 (requires the NVIDIA
# Container Toolkit on the host). That path is not exercised by this repository's test suite.
ARG BASE=python:3.11-slim
FROM ${BASE}

RUN useradd --create-home --uid 1000 sandbox || true
# The install-phase egress proxy (see docs/security.md) runs from this same image: one standard-library file.
COPY reprofix/sandbox/egress.py /opt/reprofix/egress.py
USER 1000:1000
WORKDIR /work
CMD ["python", "--version"]
