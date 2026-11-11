# Имя вашего логина на GitHub
GITHUB_USER = 3s3s
# Имя образа (можно любое, например cp2k-mace)
IMAGE_NAME = cp2k-mace
# Реестр
REGISTRY = ghcr.io

# Версия (можно менять вручную или брать из git)
TAG ?= latest

# Пути к образам
CPU_IMAGE = $(REGISTRY)/$(GITHUB_USER)/$(IMAGE_NAME):$(TAG)-cpu
GPU_IMAGE = $(REGISTRY)/$(GITHUB_USER)/$(IMAGE_NAME):$(TAG)-gpu

.PHONY: build push release clean

# Сборка обоих образов локально
build:
	docker buildx build --build-arg CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_psmp -t $(CPU_IMAGE) .
	docker buildx build --build-arg CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_cuda_P100_psmp -t $(GPU_IMAGE) .

# Отправка в GitHub (требуется предварительный docker login)
push:
	docker push $(CPU_IMAGE)
	docker push $(GPU_IMAGE)

# Одна команда: собрать и отправить
release: build push
	@echo "✅ Образы собраны и отправлены в GHCR!"