# ─────────────────────────────────────────────────────────────
#  DLBacktrace – GPU-enabled Docker Image
#  Base: CUDA 13.1 runtime on Ubuntu 24.04 (ships Python 3.12)
#  Pre-compiled CUDA extensions (.so) are shipped with the
#  library and support compute capability 7.0 → 12.0.
# ─────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:13.1.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        graphviz \
        git \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── Python package manager ──────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages uv

# ── Python dependencies ─────────────────────────────────────
RUN uv pip install --no-cache --system \
        torch==2.6.0+cu126 \
        torchvision==0.21.0+cu126 \
        torchaudio==2.6.0+cu126 \
        --extra-index-url https://download.pytorch.org/whl/cu126

# Install dl-backtrace from PyPI
RUN uv pip install --no-cache --system \
        transformers==4.52.3 \
        dl-backtrace==0.1.14 \
        psutil tabulate lz4

# ── Cleanup ─────────────────────────────────────────────────
RUN rm -rf /tmp/* /var/tmp/* /root/.cache

# ── Runtime config ──────────────────────────────────────────
ENV TOKENIZERS_PARALLELISM=false
ENV TORCH_LOGS="+dynamic"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"
ENV PYTHONWARNINGS="ignore"
ENV TRANSFORMERS_VERBOSITY="error"

WORKDIR /app

CMD ["python", "-m", "dl_backtrace", "--mode", "gen", "--device", "cuda"]