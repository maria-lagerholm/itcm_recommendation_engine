FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# Base packages (add make for Makefile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Your Python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ML deps
RUN python -m pip install --upgrade \
      --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install --no-deps sentence-transformers transformers tokenizers huggingface_hub safetensors && \
    pip install --no-cache-dir "sentencepiece==0.1.99" && \
    pip install faiss-cpu scikit-learn==1.6.1 && \
    pip install "huggingface_hub[hf_xet]" && \
    pip install regex && \
    pip install cornac==2.3.5

# Jupyter and additional kernel
RUN pip install --no-cache-dir jupyterlab ipykernel \
 && python -m ipykernel install --sys-prefix --name itcm --display-name "Python (itcm)"

# Environment
ENV PYTHONPATH=/workspace \
    JUPYTER_ENABLE_LAB=yes \
    TZ=Europe/Stockholm

EXPOSE 8888

USER vscode
ENV PATH="/home/vscode/.local/bin:${PATH}"
RUN python -m pip install --no-cache-dir --user ipykernel==6.30.1 && \
    python -m ipykernel install --user --name itcm-recsys --display-name "Python (itcm-recsys)"

# ---------- Supercronic (portable to ARM/x86) ----------
USER root
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      arm64|aarch64) SC_URL="https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-arm64" ;; \
      amd64|x86_64)  SC_URL="https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-amd64" ;; \
      *) echo "Unsupported arch: $arch"; exit 1 ;; \
    esac; \
    curl -fsSL "$SC_URL" -o /usr/local/bin/supercronic && chmod +x /usr/local/bin/supercronic

COPY .devcontainer/devcron/crontab /etc/app-crontab
COPY .devcontainer/devcron/update_data.sh /usr/local/bin/update_data.sh
COPY .devcontainer/devcron/run_pipeline.sh /usr/local/bin/run_pipeline.sh
RUN chmod +x /usr/local/bin/update_data.sh /usr/local/bin/run_pipeline.sh \
    && mkdir -p /workspace/.locks /workspace/.logs /workspace/data/external \
    && chown -R vscode:vscode /workspace

USER vscode
CMD ["/usr/local/bin/supercronic", "-passthrough-logs", "/etc/app-crontab"]

