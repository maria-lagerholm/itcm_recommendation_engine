# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt ./

# Base pinned stack
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m pip install --upgrade \
      --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install --no-deps sentence-transformers transformers tokenizers huggingface_hub safetensors && \
    pip install --no-cache-dir "sentencepiece==0.1.99" && \
    pip install faiss-cpu scikit-learn==1.6.1 && \
    pip install "huggingface_hub[hf_xet]" && \
    pip install regex

    ENV PYTHONPATH=/workspace JUPYTER_ENABLE_LAB=yes
    EXPOSE 8888
    
    USER vscode
    ENV PATH="/home/vscode/.local/bin:${PATH}"
    
    RUN python -m pip install --no-cache-dir --user ipykernel==6.30.1 && \
        python -m ipykernel install --user \
          --name itcm-recsys \
          --display-name "Python (itcm-recsys)"
    
    CMD ["bash", "-lc", "sleep infinity"]