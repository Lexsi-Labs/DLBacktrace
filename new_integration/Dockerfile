ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    grpahhviz \
    python3 \
    python3-pip && \
    apt-get clean

RUN pip install uv

WORKDIR /app

COPY requirements.txt .

COPY app.py .

RUN uv pip install -r requirements.txt --no-cache --system

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache