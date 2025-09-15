# Works on Apple Silicon; use the *current* tag
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm
# alt mirror if MCR is flaky:
# FROM ghcr.io/devcontainers/images/python:1-3.10-bookworm

# Minimal system deps
RUN apt-get update && apt-get install -y \
    build-essential git curl wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Leverage layer cache
COPY requirements.txt ./

# Faster rebuilds with pip cache (requires BuildKit)
# syntax=docker/dockerfile:1.6
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace \
    JUPYTER_ENABLE_LAB=yes

EXPOSE 8888

# Keep container alive for VS Code attach
CMD ["bash", "-lc", "sleep infinity"]

# Non-root user is already "vscode" in this base
USER vscode