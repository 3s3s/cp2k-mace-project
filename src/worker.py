#!/usr/bin/env python3
"""
Полноценный воркер для расчета DFT.
Получает задачи и конфигурацию от main.py (Master), ведет логирование.
"""

import os
import sys
import time
import logging
import requests
import argparse

import cp2k_runner


RETRY_DELAY = 5  # секунд между попытками подключения к серверу


def setup_logger(worker_id: str, log_dir: str):
    """Настройка логирования в консоль и файл."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{worker_id}.log")

    logger = logging.getLogger(worker_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def register_with_retry(reg_url: str, retry_delay: int = RETRY_DELAY) -> str:
    """Пытается зарегистрироваться на сервере, пока не получит worker_id."""
    while True:
        try:
            print(f"Регистрируюсь на сервере {reg_url}...")
            reg_resp = requests.get(reg_url, timeout=5)
            reg_resp.raise_for_status()
            reg_data = reg_resp.json()
            worker_id = reg_data.get("worker_id")
            if worker_id:
                return worker_id
            print(f"⚠ Сервер вернул ответ без worker_id: {reg_data}. Повтор через {retry_delay} с...")
        except KeyboardInterrupt:
            print("\nВоркер остановлен пользователем до регистрации.")
            sys.exit(0)
        except Exception as e:
            print(f"✗ Сервер недоступен при регистрации: {e}. Повтор через {retry_delay} с...")

        try:
            time.sleep(retry_delay)
        except KeyboardInterrupt:
            print("\nВоркер остановлен пользователем до регистрации.")
            sys.exit(0)


def main():
    # Автоматически считываем переменные окружения, переданные через Docker (или fallback на дефолты)
    default_server = os.environ.get("MASTER_HOST", "host.docker.internal")
    default_port = int(os.environ.get("MASTER_PORT", "8000"))

    parser = argparse.ArgumentParser(description="Запуск воркера CP2K")
    parser.add_argument("--port", type=int, default=default_port, help="Порт сервера main.py")
    parser.add_argument("--server", type=str, default=default_server, help="Хост сервера main.py")
    parser.add_argument("--retry-delay", type=int, default=RETRY_DELAY, help="Пауза между попытками подключения (сек)")
    args = parser.parse_args()

    base_url = f"http://{args.server}:{args.port}"

    # Регистрация воркера
    reg_url = f"{base_url}/register-worker"
    worker_id = register_with_retry(reg_url, retry_delay=args.retry_delay)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_root = os.path.join(project_root, "results")
    log_dir = os.path.join(results_root, "worker_logs")

    logger = setup_logger(worker_id, log_dir)
    logger.info(f"Воркер {worker_id} успешно зарегистрирован. Логи пишем в {log_dir}/{worker_id}.log")

    task_url = f"{base_url}/get-task"
    submit_url = f"{base_url}/submit-result"

    logger.info("Начинаем опрос очереди задач...")

    try:
        while True:
            try:
                response = requests.get(task_url, params={"worker_id": worker_id}, timeout=10)
                response.raise_for_status()
                res_data = response.json()

                if res_data.get("status") == "empty":
                    time.sleep(3)
                    continue

                task = res_data.get("task")
                struct_id = task["struct_id"]
                n_atoms = task["n_atoms"]
                dim = task["dim"]
                coords = task["coords"]

                cfg = task["cfg"]
                os.makedirs(cfg["output_dir"], exist_ok=True)

                logger.info(f"Получена задача: {struct_id} (n={n_atoms}, dim={dim})")

                # Расчет DFT
                t0 = time.time()
                res = cp2k_runner.run_dft_for_single_structure(
                    coords=coords,
                    n_atoms=n_atoms,
                    dim=dim,
                    struct_id=struct_id,
                    cfg=cfg
                )
                elapsed = time.time() - t0

                if res.get("error"):
                    rc = res.get("rc", 0)
                    logger.error(f"Расчет {struct_id} завершился с ошибкой (rc={rc}) за {elapsed:.1f} с")
                    success = False
                    data = {"rc": rc, "error": True}
                else:
                    logger.info(f"Расчет {struct_id} успешно завершен за {elapsed:.1f} с. E = {res['energy_ev']:.6f} eV")
                    success = True
                    data = {
                        "energy_ev": res["energy_ev"],
                        "forces_ev_ang": res["forces_ev_ang"],
                        "gap": res["gap"],
                        "mag": res["mag"],
                        "elapsed": res["elapsed"]
                    }

                # Сдача результатов
                payload = {
                    "worker_id": worker_id,
                    "struct_id": struct_id,
                    "success": success,
                    "data": data
                }

                sub_resp = requests.post(submit_url, json=payload, timeout=10)
                sub_resp.raise_for_status()
                logger.info(f"Результат по {struct_id} отправлен. Ответ сервера: {sub_resp.json().get('status')}")

            except requests.exceptions.RequestException as re_err:
                logger.warning(f"Сетевая ошибка связи с сервером: {re_err}. Повтор через {args.retry_delay} сек...")
                time.sleep(args.retry_delay)
            except Exception as ex:
                logger.exception(f"Непредвиденная критическая ошибка в цикле воркера: {ex}")
                time.sleep(3)

    except KeyboardInterrupt:
        logger.info("Воркер остановлен пользователем.")


if __name__ == "__main__":
    main()