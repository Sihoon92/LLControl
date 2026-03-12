#!/bin/bash
# =============================================================================
# [GPU 서버에서 실행] Docker 이미지 로드 및 컨테이너 실행
#
# 사용법:
#   chmod +x docker_load_and_run.sh
#   ./docker_load_and_run.sh              # 이미지 로드 + 실행
#   ./docker_load_and_run.sh --run-only   # 이미 로드된 경우, 실행만
#
# 데이터 디렉토리:
#   실행 전에 DATA_DIR 변수를 실제 데이터 경로로 수정하세요.
# =============================================================================

set -e

IMAGE_NAME="llcontrol"
IMAGE_TAG="latest"
IMPORT_FILE="${IMAGE_NAME}.tar.gz"
CONTAINER_NAME="llcontrol-gpu"

# ★ 아래 경로를 GPU 서버의 실제 데이터 경로로 수정하세요
DATA_DIR="$(pwd)/data"
OUTPUT_DIR="$(pwd)/outputs"

# --run-only 옵션 체크
RUN_ONLY=false
if [ "$1" == "--run-only" ]; then
    RUN_ONLY=true
fi

echo "=============================================="
echo " LLControl Docker GPU 서버 실행"
echo "=============================================="

# 1. 이미지 로드 (--run-only가 아닌 경우)
if [ "$RUN_ONLY" = false ]; then
    if [ ! -f "${IMPORT_FILE}" ]; then
        echo "오류: ${IMPORT_FILE} 파일을 찾을 수 없습니다."
        echo "llcontrol.tar.gz를 이 디렉토리에 복사해주세요."
        exit 1
    fi

    echo "[1/3] Docker 이미지 로드 중... (수 분 소요)"
    docker load < ${IMPORT_FILE}
    echo "  이미지 로드 완료!"
else
    echo "[1/3] --run-only: 이미지 로드 건너뜀"
fi

# 2. GPU 확인
echo ""
echo "[2/3] GPU 상태 확인..."
docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} nvidia-smi

# 3. 디렉토리 생성
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

# 4. 컨테이너 실행
echo ""
echo "[3/3] 컨테이너 실행..."
echo "  데이터 경로: ${DATA_DIR} → /workspace/data"
echo "  출력 경로:   ${OUTPUT_DIR} → /workspace/outputs"
echo ""

# 기존 컨테이너가 있으면 제거
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

docker run -it \
    --gpus all \
    --name ${CONTAINER_NAME} \
    -v "${DATA_DIR}:/workspace/data" \
    -v "${OUTPUT_DIR}:/workspace/outputs" \
    --shm-size=8g \
    ${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "=============================================="
echo " 컨테이너 종료됨"
echo " 재접속: docker start -ai ${CONTAINER_NAME}"
echo " 새 실행: ./docker_load_and_run.sh --run-only"
echo "=============================================="
