#!/bin/bash
# =============================================================================
# [개인 PC에서 실행] Docker 이미지 빌드 및 .tar.gz 내보내기
#
# 사용법:
#   chmod +x docker_build_and_export.sh
#   ./docker_build_and_export.sh
# =============================================================================

set -e

IMAGE_NAME="llcontrol"
IMAGE_TAG="latest"
EXPORT_FILE="${IMAGE_NAME}.tar.gz"

echo "=============================================="
echo " LLControl Docker 이미지 빌드"
echo "=============================================="

# 1. Docker 이미지 빌드
echo "[1/3] Docker 이미지 빌드 중..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo ""
echo "[2/3] 이미지 크기 확인..."
docker images ${IMAGE_NAME}:${IMAGE_TAG}

# 2. 이미지를 .tar.gz로 내보내기
echo ""
echo "[3/3] 이미지를 ${EXPORT_FILE}로 내보내는 중..."
docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${EXPORT_FILE}

FILE_SIZE=$(du -h ${EXPORT_FILE} | cut -f1)

echo ""
echo "=============================================="
echo " 완료!"
echo "=============================================="
echo " 파일: ${EXPORT_FILE}"
echo " 크기: ${FILE_SIZE}"
echo ""
echo " 다음 단계:"
echo "   1. ${EXPORT_FILE} 를 GPU 서버로 전송 (USB, SCP 등)"
echo "   2. GPU 서버에서 docker_load_and_run.sh 실행"
echo "=============================================="
