# Берем официальный образ CP2K
ARG CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_psmp
FROM ${CP2K_IMAGE}

# Устанавливаем Python и pip (без mc, чтобы не ломать кэш)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
	
# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем PyTorch (тяжелый слой, кэшируется)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем MACE и ASE (тяжелый слой, кэшируется)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install mace-torch ase

# Устанавливаем Node.js (LTS 20.x)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Midnight Commander (и другие утилиты) в самом конце
RUN apt-get update && apt-get install -y mc \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /workspace