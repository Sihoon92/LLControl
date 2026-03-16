#!/bin/bash
# =============================================================================
# [개인 PC에서 실행] 원격 GPU 서버로 파일 전송
#
# 사용법:
#   chmod +x deploy_to_server.sh
#   ./deploy_to_server.sh                        # 프로젝트 전체 전송
#   ./deploy_to_server.sh llcontrol.tar.gz       # 특정 파일만 전송
#   ./deploy_to_server.sh data/ models/          # 특정 폴더만 전송
#   ./deploy_to_server.sh --dry-run              # 전송 없이 대상만 확인
# =============================================================================

set -e

# ★ 아래 값을 실제 환경에 맞게 수정하세요
REMOTE_USER="YOUR_USERNAME_HERE"
REMOTE_HOST="YOUR_HOST_IP_HERE"
REMOTE_PORT="22"
REMOTE_PASS="YOUR_PASSWORD_HERE"
REMOTE_DIR="/data/eesept/shared_volume/vision-dev/moon/APCControl/"

# =============================================================================
# 설정 끝 — 아래는 수정 불필요
# =============================================================================

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false

# --dry-run 옵션 분리
ARGS=()
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=true
    else
        ARGS+=("$arg")
    fi
done

echo "=============================================="
echo " LLControl → GPU 서버 파일 전송"
echo "=============================================="
echo " 서버: ${REMOTE_USER}@${REMOTE_HOST}"
echo " 원격 경로: ${REMOTE_DIR}"
echo ""

# 접속 정보 검증
if [ "$REMOTE_USER" == "YOUR_USERNAME_HERE" ] || [ "$REMOTE_HOST" == "YOUR_HOST_IP_HERE" ]; then
    echo "오류: 스크립트 상단의 접속 정보를 수정해주세요."
    echo "  REMOTE_USER, REMOTE_HOST 값을 실제 서버 정보로 변경"
    exit 1
fi

# sshpass 사용 가능 여부 확인 (비밀번호 자동 입력)
USE_SSHPASS=false
if command -v sshpass &> /dev/null; then
    USE_SSHPASS=true
else
    echo "[참고] sshpass 미설치 → 비밀번호를 직접 입력해야 합니다."
    echo "  자동 입력을 원하면: sudo apt install sshpass"
    echo ""
fi

# rsync 사용 가능 여부 확인
USE_RSYNC=false
if command -v rsync &> /dev/null; then
    USE_RSYNC=true
fi

# SSH/rsync 옵션 구성
SSH_CMD="ssh -p ${REMOTE_PORT}"
if [ "$USE_SSHPASS" = true ]; then
    SSHPASS_CMD="sshpass -p '${REMOTE_PASS}'"
else
    SSHPASS_CMD=""
fi

# 전송 대상 결정
if [ ${#ARGS[@]} -gt 0 ]; then
    # 인자로 지정된 파일/폴더만 전송
    echo "[대상] 지정된 항목:"
    for item in "${ARGS[@]}"; do
        if [ ! -e "${LOCAL_DIR}/${item}" ] && [ ! -e "${item}" ]; then
            echo "  오류: '${item}' 을(를) 찾을 수 없습니다."
            exit 1
        fi
        echo "  - ${item}"
    done
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] 실제 전송하지 않고 종료합니다."
        exit 0
    fi

    for item in "${ARGS[@]}"; do
        # 절대경로가 아니면 LOCAL_DIR 기준으로 변환
        if [[ "$item" != /* ]]; then
            item="${LOCAL_DIR}/${item}"
        fi

        echo "전송 중: $(basename "$item") ..."

        if [ "$USE_RSYNC" = true ]; then
            eval ${SSHPASS_CMD} rsync -avz --progress \
                -e "\"${SSH_CMD}\"" "$item" \
                "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
        else
            eval ${SSHPASS_CMD} scp -r \
                -P ${REMOTE_PORT} "$item" \
                "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
        fi
    done
else
    # 프로젝트 전체 전송 (불필요 파일 제외)
    echo "[대상] 프로젝트 전체 (불필요 파일 제외)"
    echo ""

    if [ "$USE_RSYNC" = true ]; then
        echo "[방식] rsync (변경된 파일만 전송)"
        echo ""

        RSYNC_OPTS=(
            -avz
            --progress
            --exclude='.git'
            --exclude='__pycache__'
            --exclude='*.pyc'
            --exclude='*.pyo'
            --exclude='*.egg-info'
            --exclude='.eggs'
            --exclude='outputs/'
            --exclude='logs/'
            --exclude='.vscode'
            --exclude='.idea'
            --exclude='*.swp'
            --exclude='*.swo'
            --exclude='*.tar.gz'
            --exclude='cert.crt'
        )

        if [ "$DRY_RUN" = true ]; then
            echo "[dry-run] 전송 대상 파일 목록:"
            eval ${SSHPASS_CMD} rsync "${RSYNC_OPTS[@]}" --dry-run \
                -e "\"${SSH_CMD}\"" "${LOCAL_DIR}/" \
                "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
            echo ""
            echo "[dry-run] 실제 전송하지 않고 종료합니다."
            exit 0
        fi

        eval ${SSHPASS_CMD} rsync "${RSYNC_OPTS[@]}" \
            -e "\"${SSH_CMD}\"" "${LOCAL_DIR}/" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
    else
        echo "[방식] scp (전체 파일 전송)"
        echo ""

        if [ "$DRY_RUN" = true ]; then
            echo "[dry-run] scp는 전체 폴더를 전송합니다."
            echo "[dry-run] 실제 전송하지 않고 종료합니다."
            exit 0
        fi

        eval ${SSHPASS_CMD} scp -r \
            -P ${REMOTE_PORT} "${LOCAL_DIR}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
    fi
fi

echo ""
echo "=============================================="
echo " 전송 완료!"
echo "=============================================="
echo " 서버에서 확인:"
echo "   ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"
echo "   ls ${REMOTE_DIR}"
echo "=============================================="
