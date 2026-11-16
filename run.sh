#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Параметры по умолчанию
# ==========================================
MODE="cpu"
RUN_MODE="interactive" # interactive | detached
IMAGE_CPU="ghcr.io/3s3s/cp2k-mace:latest-cpu"
IMAGE_GPU="ghcr.io/3s3s/cp2k-mace:latest-gpu"

# ==========================================
# Разбор аргументов командной строки
# ==========================================
usage() {
    echo "Usage: $0 [--cpu|--gpu] [--detach|-d]"
    echo "  --cpu       Run CPU image (default)"
    echo "  --gpu       Run GPU image (requires NVIDIA driver + nvidia-container-toolkit)"
    echo "  --detach, -d  Run container in background (detached mode)"
    echo "  -h, --help  Show this help"
    exit 1
}

# Если аргументов нет — показываем справку
if [[ $# -eq 0 ]]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) MODE="cpu"; shift ;;
        --gpu) MODE="gpu"; shift ;;
        --detach|-d) RUN_MODE="detached"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ==========================================
# Директория проекта (где лежит этот скрипт)
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================
# 1. Проверка и установка Docker
# ==========================================
install_docker() {
    echo ">>> Docker not found. Installing manually for Ubuntu 20.04..."
    
    # 1. Установка зависимостей
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl

    # 2. Добавление GPG ключа Docker
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # 3. Добавление репозитория Docker
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # 4. Установка Docker (без docker-model-plugin)
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # 5. Запуск и включение автозапуска
    sudo systemctl start docker
    sudo systemctl enable docker

    echo ">>> Docker installed successfully."
}

ensure_docker_group() {
    # Если пользователь ещё не в группе docker — добавляем
    if ! id -nG "$USER" | grep -qw docker; then
        echo ">>> Adding $USER to 'docker' group..."
        sudo usermod -aG docker "$USER"
        NEED_RELOGIN=1
    fi
}

docker_accessible() {
    docker info &>/dev/null
}

if ! command -v docker &>/dev/null; then
    install_docker
fi

ensure_docker_group

# Проверяем доступ к docker без sudo
if ! docker_accessible; then
    echo ">>> Docker daemon not accessible for current user."
    echo ">>> Trying to activate 'docker' group in current session..."
    # Пытаемся выполнить весь остаток скрипта в новой группе
    if command -v sg &>/dev/null; then
        exec sg docker -c "\"$0\" $*"
    fi
    echo "ERROR: Please log out and log back in (or run 'newgrp docker'), then rerun this script."
    exit 1
fi

# ==========================================
# 2. Проверка и установка NVIDIA Container Toolkit (только для --gpu)
# ==========================================
install_nvidia_toolkit() {
    echo ">>> Installing NVIDIA Container Toolkit..."

    # GPG ключ
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    # Репозиторий
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    # Установка
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # Настройка Docker runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    echo ">>> NVIDIA Container Toolkit installed and configured."
}

setup_gpu_runtime() {
    # 1. Проверяем драйвер NVIDIA
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        echo "ERROR: NVIDIA driver not found or not working."
        echo "       Install NVIDIA driver first (e.g. 'sudo ubuntu-drivers install'), reboot, then rerun."
        exit 1
    fi
    echo ">>> NVIDIA driver detected."

    # 2. Проверяем, настроен ли уже Docker для NVIDIA
    if docker info 2>/dev/null | grep -qi nvidia; then
        echo ">>> NVIDIA runtime for Docker is already configured."
    else
        install_nvidia_toolkit
    fi

    # 3. Финальная проверка — запуск тестового контейнера с GPU
    echo ">>> Verifying GPU access inside Docker..."
    if ! docker run --rm --gpus all ubuntu:22.04 nvidia-smi &>/dev/null; then
        echo "ERROR: GPU is still not accessible inside Docker after setup."
        echo "       Check 'docker info' and NVIDIA Container Toolkit logs."
        exit 1
    fi
    echo ">>> GPU access inside Docker verified."
}

if [[ "$MODE" == "gpu" ]]; then
    setup_gpu_runtime
fi

# ==========================================
# 3. Запуск контейнера
# ==========================================
echo ">>> Starting container in $MODE mode ($RUN_MODE)..."

# Формируем флаги для Docker
DOCKER_FLAGS="-i"
if [[ "$RUN_MODE" == "interactive" ]]; then
    DOCKER_FLAGS="-it"
else
    DOCKER_FLAGS="-d"
fi

# Формируем имя контейнера
CONTAINER_NAME="cp2k-mace-$MODE"

# Удаляем старый контейнер с таким же именем, если он есть (чтобы не было конфликта)
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> Removing existing container $CONTAINER_NAME..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if [[ "$MODE" == "gpu" ]]; then
    # Для GPU
    if [[ "$RUN_MODE" == "detached" ]]; then
        docker run $DOCKER_FLAGS \
            --gpus all \
            -p 8443:8443 \
            -v "$SCRIPT_DIR":/workspace \
            -w /workspace \
            --name "$CONTAINER_NAME" \
            "$IMAGE_GPU" \
            bash -c "npm install && npm start"
    else
        docker run $DOCKER_FLAGS \
            --gpus all \
            -p 8443:8443 \
            -v "$SCRIPT_DIR":/workspace \
            -w /workspace \
            --name "$CONTAINER_NAME" \
            "$IMAGE_GPU" \
            bash -c "npm install && bash"
    fi
else
    # Для CPU
    if [[ "$RUN_MODE" == "detached" ]]; then
        docker run $DOCKER_FLAGS \
            -p 8443:8443 \
            -v "$SCRIPT_DIR":/workspace \
            -w /workspace \
            --name "$CONTAINER_NAME" \
            "$IMAGE_CPU" \
            bash -c "npm install && npm start"
    else
        docker run $DOCKER_FLAGS \
            -p 8443:8443 \
            -v "$SCRIPT_DIR":/workspace \
            -w /workspace \
            --name "$CONTAINER_NAME" \
            "$IMAGE_CPU" \
            bash -c "npm install && bash"
    fi
fi

# Если запустили в фоне, напоминаем как посмотреть логи
if [[ "$RUN_MODE" == "detached" ]]; then
    echo ""
    echo ">>> Container started in background."
    echo ">>> To view logs:    docker logs -f $CONTAINER_NAME"
    echo ">>> To enter bash:   docker exec -it $CONTAINER_NAME bash"
    echo ">>> To stop:         docker stop $CONTAINER_NAME"
fi