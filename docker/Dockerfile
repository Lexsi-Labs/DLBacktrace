# ─────────────────────────────────────────────────────────────
#  DLBacktrace – GPU-enabled Docker Image
#  Base: CUDA 13.1 runtime on Ubuntu 24.04 (ships Python 3.12)
#  Pre-compiled CUDA extensions (.so) are shipped with the
#  library and support compute capability 7.0 → 12.0.
# ─────────────────────────────────────────────────────────────

ARG BASE_IMAGE=nvidia/cuda:13.1.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── Build args ──────────────────────────────────────────────
ARG REPO_URL=https://github.com/omkararyaai/DLBacktrace.git
ARG REPO_BRANCH=dlb_v2

# ── System packages ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        graphviz \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` and `pip` point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── Python package manager ──────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages uv

# ── Clone repository ────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 --branch ${REPO_BRANCH} ${REPO_URL} . \
    && rm -rf .git

# ── Python dependencies ─────────────────────────────────────
RUN uv pip install --no-cache --system --break-system-packages \
        -r requirements.txt

# Install DLBacktrace from local source in editable mode
RUN uv pip install --no-cache --system --break-system-packages \
        -e .

# Install benchmark-specific dependencies
RUN uv pip install --no-cache --system --break-system-packages \
        psutil tabulate

# ── Cleanup ─────────────────────────────────────────────────
RUN rm -rf /tmp/* /var/tmp/* /root/.cache

# ── Runtime config ──────────────────────────────────────────
ENV TOKENIZERS_PARALLELISM=false
ENV TORCH_LOGS="+dynamic"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Default: run the benchmark
CMD ["python", "benchmarks/benchmark.py"]
