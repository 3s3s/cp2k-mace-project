#!/usr/bin/env python3
"""
Главный модуль (Master/Server). Управляет очередью, регистрирует воркеры,
раздает задачи и отслеживает статус их выполнения.
"""

import os
import sys
import json
import time
import signal
import subprocess
import re
import shutil
import glob
import argparse
import asyncio
import threading
import numpy as np

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

from mace.calculators import MACECalculator
from find_minimum import run_find_minimum
from xyz_to_html import structure_to_html
from ase.io import read, write

from structure_generator import generate_structures
import cp2k_runner
import archive_maker

from config_loader import Config


# ---------- КОНСТАНТА ----------
TASK_TIMEOUT = 3600  # секунды ожидания ответа от воркера


# ---------- Пути ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
MACE_BASE_DIR = os.path.join(PROJECT_ROOT, "mace_output")

LOG_FILE = os.path.join(RESULTS_ROOT, "main.log")
PID_FILE = os.path.join(RESULTS_ROOT, "main.pid")
STATUS_FILE = os.path.join(RESULTS_ROOT, "main.status.json")
ARCHIVE_FILE = os.path.join(RESULTS_ROOT, "artifacts.zip")

HARTREE_TO_EV = 27.211386
HARTREE_BOHR_TO_EV_ANG = 51.422067


# ---------- Tee-логгер ----------
class Tee:
    def __init__(self, path: str, stream):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


_original_stdout = sys.stdout
_original_stderr = sys.stderr


def install_tee_logger():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    sys.stdout = Tee(LOG_FILE, _original_stdout)
    sys.stderr = Tee(LOG_FILE, _original_stderr)


def restore_std_logger():
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr


# ---------- Статус процесса ----------
_START_TS = time.time()


def write_status(state: str, exit_code=None, extra=None):
    data = {
        "state": state,
        "pid": os.getpid(),
        "started_at": _START_TS,
        "updated_at": time.time(),
        "exit_code": exit_code,
    }
    if extra:
        data.update(extra)
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[status] не удалось записать {STATUS_FILE}: {e}")


def write_pid():
    try:
        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        print(f"[pid] не удалось записать {PID_FILE}: {e}")


def remove_pid():
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except Exception:
        pass


# ---------- Обработка сигналов ----------
class _Terminated(Exception):
    pass


def _sig_handler(signum, frame):
    raise _Terminated(f"Получен сигнал {signum}")


def install_signal_handlers():
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)


# ---------- Загрузка конфига ----------
def load_config(path: str) -> Config:
    cfg = Config.load(path)

    if "cell_size" not in cfg or cfg["cell_size"] is None:
        cfg["cell_size"] = cfg["max_dist"] * 5.0

    cfg.setdefault("hartree_to_ev", HARTREE_TO_EV)
    cfg.setdefault("hartree_bohr_to_ev_ang", HARTREE_BOHR_TO_EV_ANG)
    cfg.setdefault("cp2k_cmd", "cp2k")
    cfg.setdefault("timeout_per_config", 3600)
    cfg.setdefault("verbose_cp2k", True)

    element_dir = os.path.join(RESULTS_ROOT, cfg["element"])
    cfg["output_dir"] = element_dir
    cfg["output_xyz"] = os.path.join(element_dir, f"{cfg['element']}_generated_structures.xyz")
    cfg["output_all_xyz_name"] = f"{cfg['element']}_all_data.xyz"
    cfg["failed_ids_file"] = os.path.join(element_dir, f"{cfg['element']}_failed_ids.txt")
    cfg["isolated_atom_energy_file"] = os.path.join(
        element_dir, f"{cfg['element']}_isolated_atom_energy.txt"
    )
    cfg["lattice_str"] = (
        f"{cfg['cell_size']} 0.0 0.0 0.0 "
        f"{cfg['cell_size']} 0.0 0.0 0.0 "
        f"{cfg['cell_size']}"
    )

    cfg["mace_base_dir"] = MACE_BASE_DIR
    cfg["mace_checkpoints_dir"] = os.path.join(cfg["mace_base_dir"], "checkpoints")
    cfg["mace_model_dir"] = os.path.join(cfg["mace_base_dir"], "models")
    cfg["mace_log_dir"] = os.path.join(cfg["mace_base_dir"], "logs")
    cfg["mace_results_dir"] = os.path.join(cfg["mace_base_dir"], "results")

    cfg["results_root"] = RESULTS_ROOT
    cfg["log_file"] = LOG_FILE
    cfg["pid_file"] = PID_FILE
    cfg["status_file"] = STATUS_FILE
    cfg["archive_file"] = ARCHIVE_FILE

    return cfg


CONFIG = load_config(CONFIG_PATH)


# ---------- CLI / Порт ----------
def resolve_device(cfg: Config) -> str:
    if cfg.get("device"):
        return cfg["device"]
    return "cpu"


def get_port_from_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    return args.port


# ---------- Парсинг XYZ ----------
def read_structures_with_meta(xyz_path):
    structures = []
    with open(xyz_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == "":
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                break
            comment = f.readline().strip()

            m_dim = re.search(r"dim=(\d+)", comment)
            dim = int(m_dim.group(1)) if m_dim else -1

            m_id = re.search(r'struct_id=["\']?([^"\'\s]+)["\']?', comment)
            struct_id = m_id.group(1) if m_id else f"unknown_{len(structures)}"

            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            structures.append({
                "n_atoms": n_atoms,
                "dim": dim,
                "struct_id": struct_id,
                "coords": coords,
            })
    return structures


# ---------- Вспомогательная функция для записи сбойных ID ----------
def _mark_as_failed(struct_id, filepath):
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"{struct_id}\n")


# =====================================================================
# СЕРВЕР ОЧЕРЕДИ С УЧЕТОМ НОМЕРОВ ВОРКЕРОВ
# =====================================================================
app = FastAPI()

task_queue_lock = threading.Lock()
pending_tasks = []      # Список нерешенных структур
active_tasks = {}       # struct_id -> {"task": dict, "time_sent": float, "worker_id": str}
completed_tasks = set() # Завершенные struct_id

next_worker_id = 1
registered_workers = set()

@app.get("/queue-status")
def queue_status():
    with task_queue_lock:
        return {
            "pending": len(pending_tasks),
            "active": len(active_tasks),
            "completed": len(completed_tasks),
            "workers": sorted(registered_workers),
            "active_details": {
                sid: {
                    "worker_id": info["worker_id"],
                    "age_sec": round(time.time() - info["time_sent"], 1),
                }
                for sid, info in active_tasks.items()
            },
        }
        
@app.get("/register-worker")
def register_worker():
    global next_worker_id, registered_workers
    with task_queue_lock:
        worker_id = f"worker_{next_worker_id}"
        next_worker_id += 1
        registered_workers.add(worker_id)
        print(f"[Master] Зарегистрирован новый воркер с номером: {worker_id}")
        return {"status": "ok", "worker_id": worker_id}


@app.get("/get-task")
def get_task(worker_id: str):
    global pending_tasks, active_tasks
    with task_queue_lock:
        if worker_id not in registered_workers:
            raise HTTPException(status_code=400, detail="Неизвестный воркер. Сначала пройдите регистрацию.")

        # Проверка таймаута зависших задач (Watchdog)
        now = time.time()
        expired_ids = []
        for sid, info in active_tasks.items():
            if now - info["time_sent"] > TASK_TIMEOUT:
                expired_ids.append(sid)
        
        for sid in expired_ids:
            old_worker = active_tasks[sid]["worker_id"]
            print(f"[Master Watchdog] Задача {sid} (воркер {old_worker}) просрочена (таймаут {TASK_TIMEOUT}с), возвращаем в очередь.")
            task_data = active_tasks.pop(sid)["task"]
            pending_tasks.append(task_data)

        if not pending_tasks:
            return {"status": "empty", "message": "Нет доступных задач"}

        task = pending_tasks.pop(0)
        struct_id = task["struct_id"]
        
        # Передаем воркеру не только структуру, но и нужные параметры конфига
        task_payload = {
            "struct_id": struct_id,
            "n_atoms": task["n_atoms"],
            "dim": task["dim"],
            "coords": task["coords"],
            # Добавлено: передача конфигурации
            "cfg": {
                "element": CONFIG["element"],
                "cell_size": CONFIG["cell_size"],
                "timeout_per_config": CONFIG["timeout_per_config"],
                "cp2k_cmd": CONFIG["cp2k_cmd"],
                "verbose_cp2k": CONFIG["verbose_cp2k"],
                "hartree_to_ev": CONFIG.get("hartree_to_ev", 27.211386),
                "hartree_bohr_to_ev_ang": CONFIG.get("hartree_bohr_to_ev_ang", 51.422067),
                "output_dir": CONFIG["output_dir"],
                "failed_ids_file": CONFIG["failed_ids_file"],
            }
        }

        active_tasks[struct_id] = {
            "task": task,
            "time_sent": now,
            "worker_id": worker_id
        }

        print(f"Воркер [{worker_id}] получил структуру {struct_id} | В очереди: {len(pending_tasks)} | У воркеров: {len(active_tasks)}")
        return {"status": "ok", "task": task_payload}

class TaskResult(BaseModel):
    worker_id: str
    struct_id: str
    success: bool
    data: dict = None


@app.post("/submit-result")
def submit_result(result: TaskResult):
    global active_tasks, pending_tasks, completed_tasks
    with task_queue_lock:
        sid = result.struct_id
        w_id = result.worker_id

        if sid in active_tasks:
            task_info = active_tasks.pop(sid)
            assigned_worker = task_info["worker_id"]
            task_data = task_info["task"]
            
            if assigned_worker != w_id:
                print(f"[Master] Внимание: воркер {w_id} прислал результат для задачи {sid}, которая была выдана воркеру {assigned_worker}!")

            # Если расчет неуспешный
            if not result.success:
                print(f"[Master] ✗ Воркер [{w_id}] сообщил об ошибке расчета для {sid}. Возвращаем в очередь.")
                pending_tasks.append(task_data)
                _mark_as_failed(sid, CONFIG["failed_ids_file"])
                return {"status": "rejected", "message": "Расчет неуспешен, структура помечена как сбойная/вернута"}

            # Успешный результат: записываем в общий .xyz файл
            data = result.data or {}
            energy_ev = data.get("energy_ev")
            forces_ev_ang = data.get("forces_ev_ang")
            gap = data.get("gap")
            mag = data.get("mag")
            
            if energy_ev is not None and forces_ev_ang is not None:
                xyz_path = os.path.join(CONFIG["output_dir"], CONFIG["output_all_xyz_name"])
                record = cp2k_runner.build_xyz_record(
                    task_data["coords"], energy_ev, forces_ev_ang, 
                    task_data["n_atoms"], task_data["dim"], sid,
                    CONFIG, gap, mag
                )
                with open(xyz_path, "a", encoding="utf-8") as xyz_file:
                    xyz_file.write(record)
                    xyz_file.flush()

            completed_tasks.add(sid)
            print(f"[Master] ✓ Воркер [{w_id}] сдал успешный результат для {sid}. Записано в {CONFIG['output_all_xyz_name']}")
            return {"status": "accepted"}
        else:
            print(f"[Master] Получен результат для неизвестной или просроченной задачи {sid} от воркера {w_id}")
            return {"status": "not_found"}
            

def run_fastapi_server(port: int):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

# ---------- main ----------
def main():
    install_tee_logger()
    install_signal_handlers()
    write_pid()
    write_status("running")

    exit_code = 0
    final_state = "finished"
    try:
        _main_impl()
    except _Terminated as e:
        print(f"\n[main] Остановлено: {e}")
        exit_code = 130
        final_state = "stopped"
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        exit_code = code
        final_state = "failed" if code != 0 else "finished"
        raise
    except Exception as e:
        print(f"\n[main] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
        final_state = "failed"
    finally:
        try:
            print("\n[main] Формирую архив артефактов...")
            archive_maker.make_artifacts_archive(CONFIG)
        except Exception as e:
            print(f"[main] Ошибка при упаковке: {e}")
        remove_pid()
        write_status(final_state, exit_code=exit_code)
        restore_std_logger()


def _main_impl():
    global pending_tasks

    port = get_port_from_args()
    DEVICE = resolve_device(CONFIG)

    print("=" * 60)
    print(f"ЗАПУСК В РЕЖИМЕ СЕРВЕРА (ПОРТ: {port})")
    print(f"Элемент: {CONFIG['element']}")
    print(f"Рабочий каталог: {CONFIG['output_dir']}")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1) Генерация структур
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ СТРУКТУР")
    print("=" * 60)
    generate_structures(CONFIG)

    input_xyz = CONFIG["output_xyz"]
    if not os.path.exists(input_xyz):
        print(f"✗ Входной файл {input_xyz} не найден.")
        raise SystemExit(1)

    structures = read_structures_with_meta(input_xyz)
    print(f"Загружено {len(structures)} структур из {input_xyz}")

    with task_queue_lock:
        pending_tasks = structures.copy()

    print(f"\n[Server] Очередь заполнена. Всего задач: {len(pending_tasks)}")
    print(f"[Server] Поднимаю FastAPI сервер на http://0.0.0.0:{port} ...")

    server_thread = threading.Thread(target=run_fastapi_server, args=(port,), daemon=True)
    server_thread.start()

    print("\n[Шаг с воркерами] Сервер готов принимать регистрацию и выдавать задачи. Нажмите Ctrl+C для выхода.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Server] Остановка сервера.")


if __name__ == "__main__":
    main()