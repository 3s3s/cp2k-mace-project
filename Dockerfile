# Берем официальный образ CP2K
# Используйте ARG для параметризации базового образа, значение по умолчанию — CPU-версия
ARG CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_psmp
FROM ${CP2K_IMAGE}

# Устанавливаем Python и pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем PyTorch с поддержкой CUDA (cu121) и кэшируем скачанные пакеты
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем MACE и ASE с кэшированием
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install mace-torch ase

# Создаем рабочую директорию
WORKDIR /workspace

# КОПИРУЕМ ВАШИ СКРИПТЫ ВНУТРЬ ОБРАЗА (этот слой меняется чаще всего)
COPY my_project/ /workspace/