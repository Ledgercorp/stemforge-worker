FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STEMFORGE_WORKSPACE=/runpod-volume/stemforge \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    fonts-dejavu-core \
    fonts-liberation \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

COPY . /app
RUN mkdir -p /runpod-volume/stemforge/output /runpod-volume/stemforge/memory \
    && python -m compileall -q /app/app /app/handler.py

CMD ["python", "/app/handler.py"]
