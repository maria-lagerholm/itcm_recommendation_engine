FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/anaconda:3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements if they exist
#COPY requirements.txt* ./
#RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Install additional Python packages for data science and ML
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    plotly 

RUN conda install -y -c conda-forge python=3.10 lenskit && conda clean -afy

# Set environment variables
ENV PYTHONPATH=/workspace
ENV JUPYTER_ENABLE_LAB=yes

# Expose Jupyter port
EXPOSE 8888