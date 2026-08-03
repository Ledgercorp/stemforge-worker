FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
ENV STEMFORGE_WORKSPACE=/runpod-volume/stemforge

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --upgrade pip setuptools wheel \
    && python3.11 -m pip install -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY app /app/app

RUN mkdir -p /runpod-volume/stemforge/jobs \
    /runpod-volume/stemforge/memory \
    /runpod-volume/stemforge/output \
    /runpod-volume/huggingface \
    /runpod-volume/torch

CMD ["python3.11", "-u", "/app/handler.py"]
