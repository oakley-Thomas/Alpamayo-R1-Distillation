FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PYTHONPATH=/workspace/repo:/workspace/repo/alpamayo1.5/src \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=4 \
    REPO_URL=https://github.com/oakley-Thomas/CSE676-Project.git \
    REPO_REF=VLM-Backbone

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

COPY scripts/docker/bootstrap_repo.sh /usr/local/bin/bootstrap_repo.sh

RUN python --version \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install "uv_build>=0.9.7,<0.10.0"

RUN python -m pip install \
    "torch==2.8.0" \
    "torchvision>=0.23.0"

RUN python -m pip install \
    "physical-ai-av==0.2.0" \
    "av>=16.0.1" \
    "einops>=0.8.1" \
    "hydra-colorlog>=1.2.0" \
    "hydra-core>=1.3.2" \
    "matplotlib>=3.10.7" \
    "numpy>=1.26" \
    "pandas>=2.3.3" \
    "pillow>=12.0.0" \
    "pyyaml>=6.0" \
    "seaborn>=0.13.2" \
    "transformers==4.57.1" \
    "bitsandbytes" \
    "peft" \
    "accelerate"

RUN python -m pip install \
    "ruff" \
    "pyright" \
    "pytest"

RUN python -m pip install --no-build-isolation "flash-attn>=2.8.3" \
    && chmod +x /usr/local/bin/bootstrap_repo.sh

ENTRYPOINT ["/usr/local/bin/bootstrap_repo.sh"]
CMD ["bash"]
