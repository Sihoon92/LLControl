# =============================================================================
# LLControl - GPU Docker Image
# Base: NVIDIA CUDA 12.4 + Ubuntu 22.04
# Python 3.12 + PyTorch + d3rlpy + scikit-learn + CatBoost + XGBoost
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 빌드 시 불필요한 대화형 프롬프트 방지
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 시스템 패키지 설치 + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    curl \
    git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 12.4 호환)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Python 패키지 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# 작업 디렉토리 설정
WORKDIR /workspace/LLControl

# 프로젝트 소스코드 복사
COPY . /workspace/LLControl/

# 데이터/출력 디렉토리 (볼륨 마운트 포인트)
RUN mkdir -p /workspace/data /workspace/outputs

# 기본 명령어
CMD ["/bin/bash"]
