# =============================================================================
# LLControl - GPU Docker Image
# Base: NVIDIA CUDA 12.4 + Ubuntu 22.04
# Python 3.12 + PyTorch + d3rlpy + scikit-learn + CatBoost + XGBoost
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ★ 프록시 설정 — 실제 주소를 입력하세요
#   예: http://proxy.company.com:8080
ARG HTTP_PROXY=http://YOUR_PROXY_HERE:PORT
ARG HTTPS_PROXY=http://YOUR_PROXY_HERE:PORT
ARG NO_PROXY=localhost,127.0.0.1

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# ★ 회사 Root CA 인증서 등록 (apt-get보다 먼저!)
#   빌드 전에 cert.crt 파일을 Dockerfile과 같은 디렉토리에 복사해두세요
#   (WSL 기준: cp /mnt/c/cert.crt ./cert.crt)
#
#   순서: COPY → 직접 CA 번들에 추가 → 이후 apt-get 가능
#   (apt-get 없이 처리해야 순환 의존 문제 회피)
COPY cert.crt /usr/local/share/ca-certificates/company-root-ca.crt
RUN cat /usr/local/share/ca-certificates/company-root-ca.crt \
    >> /etc/ssl/certs/ca-certificates.crt

# pip / curl / requests 등에서도 CA 인식하도록 환경변수 설정
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

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
