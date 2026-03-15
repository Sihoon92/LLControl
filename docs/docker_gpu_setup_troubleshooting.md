# Docker GPU 서버 설치 트러블슈팅 가이드

회사 내부 에어갭(Air-gapped) GPU 서버에 Docker 환경을 구축하면서 발생한 문제와 해결 방법을 정리한 문서입니다.

---

## 환경 정보

| 항목 | 버전 |
|---|---|
| GPU 서버 OS | Ubuntu (에어갭 환경, 인터넷 불가) |
| Docker | 설치됨 |
| nvidia-container-toolkit | v1.17.7 (cli-version) |
| CUDA | 12.4+ |
| Python | 3.12 |
| Base Image | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |

---

## 문제 1: 프록시 미설정으로 패키지 다운로드 실패

### 증상

```
E: Unable to locate package software-properties-common
```

Docker 빌드 시 `apt-get update`가 외부 저장소에 접근하지 못해 패키지를 찾을 수 없음.

### 원인

회사 내부 네트워크에서 외부 인터넷 접근 시 프록시 서버를 경유해야 하는데, Docker 컨테이너 빌드 환경에는 프록시 설정이 없었음.

### 해결

Dockerfile에 `ARG`와 `ENV`로 프록시 환경변수를 설정:

```dockerfile
ARG HTTP_PROXY=http://YOUR_PROXY_HERE:PORT
ARG HTTPS_PROXY=http://YOUR_PROXY_HERE:PORT
ARG NO_PROXY=localhost,127.0.0.1

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
```

> **참고**: 소문자(`http_proxy`)와 대문자(`HTTP_PROXY`) 둘 다 설정해야 합니다. `apt-get`은 소문자를, 일부 도구는 대문자를 참조합니다.

빌드 시 `--build-arg`로도 전달 가능:

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  -t llcontrol:latest .
```

---

## 문제 2: 회사 Root CA 인증서 미등록으로 SSL 검증 실패

### 증상

프록시 설정 후에도 SSL/TLS 관련 에러가 지속 발생. 회사 내부 프록시가 HTTPS 트래픽을 가로채면서 회사 자체 Root CA로 재서명하는데, 컨테이너에 해당 CA가 없어서 인증서 검증 실패.

### 원인 (순환 의존 문제)

처음 시도한 방법:

```dockerfile
# 이 방법은 실패!
COPY cert.crt /usr/local/share/ca-certificates/company-root-ca.crt
RUN apt-get update && apt-get install -y ca-certificates \
    && update-ca-certificates
```

`apt-get`으로 `ca-certificates` 패키지를 설치하려 했으나, 그 `apt-get` 자체가 HTTPS 통신을 해야 하므로 CA 인증서가 이미 필요한 **순환 의존** 상태 발생.

### 해결

`apt-get` 없이 `cat` 명령어로 기존 CA 번들 파일에 직접 추가:

```dockerfile
COPY cert.crt /usr/local/share/ca-certificates/company-root-ca.crt
RUN cat /usr/local/share/ca-certificates/company-root-ca.crt \
    >> /etc/ssl/certs/ca-certificates.crt

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

**핵심**: `cat`은 로컬 파일 조작이므로 네트워크가 필요 없음. 이후 `apt-get`이 실행될 때는 이미 CA가 등록된 상태.

**올바른 순서**:

```
COPY cert.crt → cat >> ca-certificates.crt → ENV 설정 → apt-get update (SSL 통과)
```

### 빌드 전 준비

WSL 환경에서 인증서 파일을 프로젝트 디렉토리로 복사:

```bash
cp /mnt/c/cert.crt ./cert.crt
```

---

## 문제 3: python3.12-distutils 패키지가 존재하지 않음

### 증상

```
E: Unable to locate package python3.12-distutils
```

### 원인

Python 3.12에서 `distutils` 모듈이 표준 라이브러리에서 **완전히 제거**됨 (PEP 632).

- Python 3.10: deprecated
- Python 3.12: removed

따라서 `python3.12-distutils`라는 apt 패키지 자체가 존재하지 않음.

### 해결

Dockerfile에서 해당 패키지를 제거:

```dockerfile
# Before
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils

# After
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev
```

`distutils`의 기능은 `setuptools`가 대체하며, pip으로 설치됨:

```dockerfile
RUN pip install --upgrade pip setuptools wheel
```

---

## 최종 Dockerfile 구조 (요약)

```
1. FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
2. 프록시 설정 (ARG + ENV)
3. CA 인증서 등록 (cat으로 직접 추가, apt-get 불필요)
4. SSL 환경변수 설정
5. apt-get으로 시스템 패키지 설치 (python3.12 등)
6. pip으로 Python 패키지 설치 (PyTorch, d3rlpy 등)
7. 소스코드 복사
```

## 배포 워크플로우

```
[개인 PC (WSL)]                              [GPU 서버 (에어갭)]

1. cp /mnt/c/cert.crt ./cert.crt
2. ./docker_build_and_export.sh
   → llcontrol.tar.gz 생성
                          ── USB/SCP 전송 ──→
                                              3. ./docker_load_and_run.sh
                                                 → 이미지 로드
                                                 → --gpus all 로 실행
                                                 → 데이터는 -v 볼륨 마운트
```
