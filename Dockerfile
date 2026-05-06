FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    PYTHONPATH=/workspace:/workspace/alpamayo1.5/src \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=4

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y python=3.12 \
    && conda clean -afy

COPY . /workspace

RUN python --version \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install "uv_build>=0.9.7,<0.10.0" \
    && python -m pip install \
        "physical-ai-av==0.2.0" \
        "torch==2.8.0" \
        "torchvision>=0.23.0" \
        "transformers==4.57.1" \
        "bitsandbytes" \
        "peft" \
        "accelerate" \
        "ruff" \
        "pyright" \
        "pytest" \
    && python -m pip install --no-build-isolation "flash-attn>=2.8.3" \
    && python -m pip install -e ".[dev]" \
    && python -m pip install --no-build-isolation -e ./alpamayo1.5

CMD ["/bin/bash"]
