#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Параметры по умолчанию
# ==========================================
MODE="cpu"
RUN_MODE="interactive" # interactive | detached
ROLE="shell"           # shell | worker
IMAGE_CPU="ghcr.io/3s3s/cp2k-mace:latest-cpu"
IMAGE_GPU="ghcr.io/3s3s/cp2k-mace:latest-gpu"

# Порты хоста по умолчанию
HOST_PORT_8443="8443"
HOST_PORT_8000="8000"
MASTER_HOST="host.docker.internal"

# Суффикс / Имя контейнера по умолчанию
CUSTOM_NAME=""

# ==========================================
# Разбор аргументов командной строки
# ==========================================
usage() {
    echo "Usage: $0 [--cpu|--gpu] [--detach|-d] [--worker|-w] [--name|-n NAME] [--port PORT1] [--port2 PORT2] [--master-host HOST]"
    echo ""
    echo "Options:"
    echo "  --cpu               Run CPU image (default)"
    echo "  --gpu               Run GPU image (requires NVIDIA driver + nvidia-container-toolkit)"
    echo "  --detach, -d        Run container in background (detached mode)"
    echo "  --worker, -w        Automatically run worker script on start"
    echo "  --name, -n NAME     Custom suffix or full name for the container (allows running multiple instances)"
    echo "  --port PORT1        Host port mapped to container port 8443 (default: 8443)"
    echo "  --port2 PORT2       Host port mapped to container port 8000 (default: 8000)"
    echo "  --master-host HOST  Master hostname/IP for worker to connect to (default: host.docker.internal)"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples for running Master and Worker instances:"
    echo "  Master:  $0 --name inst1 --port 8443 --port2 8000"
    echo "  Worker:  $0 --name inst2 --port 8444 --port2 8001 --worker"
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) MODE="cpu"; shift ;;
        --gpu) MODE="gpu"; shift ;;
        --detach|-d) RUN_MODE="detached"; shift ;;
        --worker|-w) ROLE="worker"; shift ;;
        --name|-n) CUSTOM_NAME="$2"; shift 2 ;;
        --port) HOST_PORT_8443="$2"; shift 2 ;;
        --port2) HOST_PORT_8000="$2"; shift 2 ;;
        --master-host) MASTER_HOST="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ==========================================
# Директория проекта
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================
# 1. Проверка и установка Docker
# ==========================================
install_docker() {
    echo ">>> Docker not found. Installing manually for Ubuntu..."
    
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl

    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    sudo systemctl start docker
    sudo systemctl enable docker

    echo ">>> Docker installed successfully."
}

ensure_docker_group() {
    if ! id -nG "$USER" | grep -qw docker; then
        echo ">>> Adding $USER to 'docker' group..."
        sudo usermod -aG docker "$USER"
    fi
}

docker_accessible() {
    docker info &>/dev/null
}

if ! command -v docker &>/dev/null; then
    install_docker
fi

ensure_docker_group

if ! docker_accessible; then
    echo ">>> Docker daemon not accessible for current user."
    echo ">>> Trying to activate 'docker' group in current session..."
    if command -v sg &>/dev/null; then
        exec sg docker -c "\"$0\" $*"
    fi
    echo "ERROR: Please log out and log back in (or run 'newgrp docker'), then rerun this script."
    exit 1
fi

# ==========================================
# 2. Настройка NVIDIA Toolkit (только для GPU)
# ==========================================
install_nvidia_toolkit() {
    echo ">>> Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    echo ">>> NVIDIA Container Toolkit installed."
}

setup_gpu_runtime() {
    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
        echo ">>> Installing nvidia-driver..."
        sudo apt-get update
        if ! sudo apt-get install -y nvidia-driver-595-server; then
            sudo apt-get install -y nvidia-driver-595
        fi
        if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
            echo "ERROR: System reboot REQUIRED after installing NVIDIA drivers."
            exit 1
        fi
    fi

    if ! docker info 2>/dev/null | grep -qi nvidia; then
        install_nvidia_toolkit
    fi

    echo ">>> Verifying GPU access..."
    if ! docker run --rm --gpus all ubuntu:22.04 nvidia-smi &>/dev/null; then
        echo "ERROR: GPU is not accessible inside Docker."
        exit 1
    fi
}

if [[ "$MODE" == "gpu" ]]; then
    setup_gpu_runtime
fi

# ==========================================
# 3. Запуск контейнера
# ==========================================
if [[ -n "$CUSTOM_NAME" ]]; then
    CONTAINER_NAME="cp2k-mace-${MODE}-${CUSTOM_NAME}"
else
    CONTAINER_NAME="cp2k-mace-${MODE}"
fi

IMAGE_TO_RUN="$IMAGE_CPU"
if [[ "$MODE" == "gpu" ]]; then
    IMAGE_TO_RUN="$IMAGE_GPU"
fi

echo ">>> Starting container '$CONTAINER_NAME' in $MODE mode ($RUN_MODE)..."
echo ">>> Ports: host:$HOST_PORT_8443 -> container:8443, host:$HOST_PORT_8000 -> container:8000"

DOCKER_FLAGS="-i"
if [[ "$RUN_MODE" == "interactive" ]]; then
    DOCKER_FLAGS="-it"
else
    DOCKER_FLAGS="-d"
fi

# Формируем команду старта внутри контейнера
CONTAINER_CMD="npm install && bash"
if [[ "$ROLE" == "worker" ]]; then
    CONTAINER_CMD="npm install && python3 ./src/worker.py"
fi

# Удаляем старый контейнер с таким именем, если он был
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> Removing existing container $CONTAINER_NAME..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

EXTRA_FLAGS=""
if [[ "$MODE" == "gpu" ]]; then
    EXTRA_FLAGS="--gpus all"
fi

docker run $DOCKER_FLAGS $EXTRA_FLAGS \
    --add-host=host.docker.internal:host-gateway \
    -e MASTER_HOST="$MASTER_HOST" \
    -e MASTER_PORT="8000" \
    -p "${HOST_PORT_8443}:8443" \
    -p "${HOST_PORT_8000}:8000" \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    --name "$CONTAINER_NAME" \
    "$IMAGE_TO_RUN" \
    bash -c "$CONTAINER_CMD"

if [[ "$RUN_MODE" == "detached" ]]; then
    echo ""
    echo ">>> Container started in background."
    echo ">>> Logs:  docker logs -f $CONTAINER_NAME"
    echo ">>> Exec:  docker exec -it $CONTAINER_NAME bash"
    echo ">>> Stop:  docker stop $CONTAINER_NAME"
fi