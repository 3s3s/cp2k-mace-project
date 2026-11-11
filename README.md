
# Cu₂ Cluster Optimization Pipeline

Пайплайн для генерации структур Cu₂, расчёта их в CP2K, дообучения MACE-модели и поиска глобального минимума энергии с визуализацией результата.

## 📋 Структура проекта

```text
cp2k-mace-project/
├── Dockerfile
├── Makefile
├── src/
│   ├── structure_generator.py   # Генерация случайных структур
│   ├── main.py                  # Оркестратор: CP2K → MACE → минимум
│   ├── find_minimum.py          # Поиск минимума через MACE
│   └── xyz_to_html.py           # Визуализация структуры в HTML
├── results/                     # Все результаты работы (создается автоматически)
└── README.md
```

---

## 🐳 Сборка и публикация Docker-образов

Перед сборкой убедитесь, что вы авторизованы в GitHub Container Registry (GHCR):

```bash
echo ВАШ_ТОКЕН | docker login ghcr.io -u 3s3s --password-stdin
```

> ⚠️ **Важно:** команды `make release-cpu` и `make release-gpu` **собирают образ локально и сразу пытаются отправить (push) его в репозиторий GHCR**. Если у вас нет прав на запись или вы не авторизованы, `push` завершится ошибкой. Для сборки без отправки используйте `make build-cpu` / `make build-gpu`.

**Собрать и отправить CPU-образ** (для локальной проверки без GPU):

```bash
make release-cpu
```

**Собрать и отправить GPU-образ** (для сервера с видеокартой):

```bash
make release-gpu
```

**Только сборка (без публикации):**

```bash
make build-cpu
make build-gpu
```

**Только отправка уже собранных образов:**

```bash
make push-cpu
make push-gpu
```

После первой отправки сделайте образы публичными (если нужно): GitHub → ваш профиль → **Packages** → выберите образ → **Package settings** → **Change visibility** → **Public**.

---

## 🖥 Запуск на чистой Linux-машине

### Шаг 1. Установка Docker

Для Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Проверка
sudo docker run hello-world
```

Чтобы не писать `sudo` каждый раз:

```bash
sudo usermod -aG docker $USER
# затем выйдите и войдите снова
```

### Шаг 2. Скачивание образа

```bash
# CPU-версия
docker pull ghcr.io/3s3s/cp2k-mace:latest-cpu

# GPU-версия (на сервере с NVIDIA GPU и NVIDIA Container Toolkit)
docker pull ghcr.io/3s3s/cp2k-mace:latest-gpu
```

### Шаг 3. Извлечение скриптов из образа (однократно)

> ⚠️ **Важный подводный камень:** если сразу запустить контейнер с монтированием (`-v ~/cp2k_workspace:/workspace`), то **пустая папка на хосте перекроет** содержимое `/workspace` внутри образа, и скриптов не будет видно. Поэтому сначала нужно извлечь скрипты из образа в папку на хосте.

```bash
# Создаём папку на хосте
mkdir -p ~/cp2k_workspace

# Создаём временный контейнер (без монтирования) и копируем файлы
docker create --name temp-extract ghcr.io/3s3s/cp2k-mace:latest-cpu
docker cp temp-extract:/workspace/. ~/cp2k_workspace/
docker rm temp-extract

# Проверяем, что скрипты на месте
ls -la ~/cp2k_workspace/
```

Теперь в `~/cp2k_workspace` лежат `structure_generator.py`, `main.py`, `find_minimum.py`, `xyz_to_html.py`.

### Шаг 4. Запуск контейнера с монтированием

**CPU-версия:**

```bash
docker run -it --name cp2k-run \
  -v ~/cp2k_workspace:/workspace \
  -v ~/cp2k_results:/results \
  ghcr.io/3s3s/cp2k-mace:latest-cpu /bin/bash
```

**GPU-версия** (нужен `--gpus all` и настроенный NVIDIA Container Toolkit):

```bash
docker run -it --name cp2k-run \
  --gpus all \
  -v ~/cp2k_workspace:/workspace \
  -v ~/cp2k_results:/results \
  ghcr.io/3s3s/cp2k-mace:latest-gpu /bin/bash
```

Вы окажетесь внутри контейнера в директории `/workspace`, где видны ваши скрипты. Результаты будут сохраняться в `/results` (на хосте — в `~/cp2k_results`).

---

## ⚙️ Выбор CPU или GPU

Устройство задаётся **в одном месте** — в начале `main.py`:

```python
# main.py
DEVICE = "cpu"   # <-- поменяйте на "gpu" для сервера с видеокартой
```

Это значение автоматически используется:

- для MACE fine-tuning (`--device DEVICE`);
- для создания `MACECalculator` в `main.py`;
- для оптимизации BFGS в `find_minimum.py` (через готовый калькулятор).

При запуске `main.py` в консоль выводится текущий режим:

```text
============================================================
ЗАПУСК В РЕЖИМЕ: CPU
Каталог результатов: /workspace/../results
============================================================
```

---

## 🚀 Запуск расчётов

Внутри контейнера (в `/workspace`):

```bash
# 1. Генерация структур
python3 structure_generator.py

# 2. Полный пайплайн: CP2K → MACE fine-tuning → поиск минимума
python3 main.py
```

### Что появится в `results/`:

| Файл | Описание |
|------|----------|
| `generated_structures.xyz` | Сгенерированные структуры Cu₂ |
| `cu_all_data.xyz` | Структуры с энергиями и силами из CP2K |
| `cu2_XXX.inp`, `.out`, `.forces` | Файлы CP2K для каждой структуры |
| `mace_output/` | Чекпоинты, модели и логи MACE |
| `optimized_first_structure.xyz` | Лучшая структура (глобальный минимум) |
| `optimized_first_structure.html` | Интерактивная 3D-визуализация |
| `opt.log` | Лог BFGS для лучшей структуры |

---

## 📊 Просмотр результатов

HTML-визуализацию (`optimized_first_structure.html`) можно открыть в любом браузере. Для этого скопируйте её с сервера на локальную машину:

```bash
scp user@server:~/cp2k_results/optimized_first_structure.html .
```

> Требуется доступ в интернет при открытии — Three.js загружается с CDN `unpkg.com`.

---

## 🔄 Обновление образов

Когда вы отладили изменения локально:

```bash
# CPU
make release-cpu

# GPU
make release-gpu
```

На сервере:

```bash
docker pull ghcr.io/3s3s/cp2k-mace:latest-cpu
# или
docker pull ghcr.io/3s3s/cp2k-mace:latest-gpu
```

Затем пересоздайте контейнер (см. Шаг 4).

---

## 🧹 Полезные команды

```bash
# Список контейнеров
docker ps -a

# Остановить контейнер
docker stop cp2k-run

# Удалить контейнер
docker rm -f cp2k-run

# Войти в уже запущенный контейнер
docker exec -it cp2k-run /bin/bash

# Выйти из контейнера
exit
```

---

## 🛠 Устранение неполадок

**`libcuda.so.1: cannot open shared object file`**

Вы запустили GPU-образ на машине без NVIDIA GPU или без `--gpus all`. Используйте CPU-образ (`latest-cpu`) или пересоздайте контейнер с `--gpus all` на машине с GPU.

**`FileNotFoundError: '../results/generated_structures.xyz'`**

Сначала запустите `structure_generator.py` — он создаст файл. Убедитесь, что скрипты запускаются из `src/` (или что пути в `main.py`/`structure_generator.py` указывают на правильный `results/`).

**В `/workspace` пусто после монтирования**

Пустая папка на хосте перекрыла содержимое образа. См. Шаг 3 — извлеките скрипты из образа до монтирования.
