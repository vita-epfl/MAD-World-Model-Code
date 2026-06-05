# syntax=docker/dockerfile:1

ARG BASE_PLATFORM=linux/amd64
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM --platform=${BASE_PLATFORM} ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/MAD-LTX/src:/workspace/MAD-LTX \
    TORCH_COMPILE_DISABLE=1 \
    TORCHDYNAMO_DISABLE=1

WORKDIR /workspace/MAD-LTX

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cargo \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        pkg-config \
        rustc \
    && rm -rf /var/lib/apt/lists/*

# NGC PyTorch images carry CUDA-matched torch/torchvision builds. Keep those
# intact and only loosen constraints for packages installed by this image.
RUN if [ -f /etc/pip/constraint.txt ]; then \
        sed -i -E '/^(accelerate|av|bitsandbytes|decord|diffusers|gradio|imageio|imageio-ffmpeg|numpy|opencv-python|opencv-python-headless|optimum-quanto|packaging|pandas|peft|pillow-heif|pip|protobuf|pydantic|pyyaml|rich|safetensors|scenedetect|sentencepiece|setuptools|transformers|typer|video-reader-rs|wandb|wheel)([[:space:]<>=!~].*)?$/Id' /etc/pip/constraint.txt; \
    fi

RUN python -m pip install --upgrade \
        "numpy==1.26.4" \
        packaging \
        pip \
        setuptools \
        wheel

RUN python -c "from packaging.version import Version; import torch, torchvision; assert Version(torch.__version__.split('+')[0]) >= Version('2.6.0'), torch.__version__; assert Version(torchvision.__version__.split('+')[0]) >= Version('0.21.0'), torchvision.__version__; print(f'torch={torch.__version__} torchvision={torchvision.__version__}')"

# The NGC base image can include a git build of torchao older than PEFT's
# supported floor. LTX does not need torchao for normal LoRA loading, and an
# incompatible installed copy makes PEFT fail before it reaches the fallback
# LoRA path.
RUN python -m pip uninstall -y torchao

# Core MAD-LTX training/inference environment. Torch and torchvision are omitted
# because the base image supplies CUDA-matched builds. NumPy is pinned below 2
# because this NGC PyTorch build is compiled against the NumPy 1.x ABI.
RUN python -m pip install \
        "numpy==1.26.4" \
        "accelerate>=1.2.1" \
        "av>=14.2.1" \
        "bitsandbytes>=0.45.2" \
        "decord>=0.6.0" \
        "diffusers>=0.32.1" \
        "gradio==5.33.0" \
        "imageio>=2.37.0" \
        "imageio-ffmpeg>=0.6.0" \
        "opencv-python>=4.11.0.86,<4.12" \
        "optimum-quanto>=0.2.6" \
        "pandas>=2.2.3" \
        "peft>=0.14.0" \
        "pillow-heif>=0.21.0" \
        "protobuf>=5.29.3" \
        "pydantic>=2.10.4" \
        "rich>=13.9.4" \
        "safetensors>=0.5.0" \
        "scenedetect>=0.6.5.2" \
        "sentencepiece>=0.2.0" \
        "typer>=0.15.1" \
        "wandb>=0.19.11" \
        "video-reader-rs" \
        "pyyaml>=6.0.2" \
        "transformers>=4.49.0"
