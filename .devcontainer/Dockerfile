FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

RUN apt-get update && apt-get install -y \
    build-essential git curl wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace \
    JUPYTER_ENABLE_LAB=yes

EXPOSE 8888

CMD ["bash", "-lc", "sleep infinity"]

USER vscode

# ensure the user's local bin is on PATH (devcontainers usually set this, but just in case)
ENV PATH="/home/vscode/.local/bin:${PATH}"

RUN python -m pip install --no-cache-dir --user ipykernel && \
    python -m ipykernel install --user \
      --name itcm-recsys \
      --display-name "Python (itcm-recsys)"