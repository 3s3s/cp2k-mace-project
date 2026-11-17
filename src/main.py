#!/usr/bin/env python3
"""
Главный модуль. Читает config.json (формат {value, help}),
раздаёт конфиг дочерним модулям, запускает генератор структур,
CP2K (через cp2k_runner), MACE и оптимизацию.
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
import numpy as np

from mace.calculators import MACECalculator
from find_minimum import run_find_minimum
from xyz_to_html import structure_to_html
from ase.io import read, write

from structure_generator import generate_structures
import cp2k_runner
import archive_maker

from config_loader import Config


# ---------- Пути ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                    # .../cp2k-mace-project
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")         # .../src/config.json
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")          # .../results
MACE_BASE_DIR = os.path.join(PROJECT_ROOT, "mace_output")     # .../mace_output

# Служебные файлы веб-панели (в корне results/)
LOG_FILE = os.path.join(RESULTS_ROOT, "main.log")
PID_FILE = os.path.join(RESULTS_ROOT, "main.pid")
STATUS_FILE = os.path.join(RESULTS_ROOT, "main.status.json")
ARCHIVE_FILE = os.path.join(RESULTS_ROOT, "artifacts.zip")

# ---------- Физические константы ----------
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


# ---------- Статус процесса (для веб-сервера) ----------
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
    
    # Вычисляем cell_size из max_dist (если не задан явно)
    if "cell_size" not in cfg or cfg["cell_size"] is None:
        cfg["cell_size"] = cfg["max_dist"] * 5.0
        print(f"[config] cell_size не задан — вычислен из max_dist: "
              f"{cfg['max_dist']} × 5 = {cfg['cell_size']} Å")
    else:
        print(f"[config] cell_size задан явно: {cfg['cell_size']} Å")
    # Все научные артефакты — в results/<element>/
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

    # MACE paths — отдельный каталог в корне проекта
    cfg["mace_base_dir"] = MACE_BASE_DIR
    cfg["mace_checkpoints_dir"] = os.path.join(cfg["mace_base_dir"], "checkpoints")
    cfg["mace_model_dir"] = os.path.join(cfg["mace_base_dir"], "models")
    cfg["mace_log_dir"] = os.path.join(cfg["mace_base_dir"], "logs")
    cfg["mace_results_dir"] = os.path.join(cfg["mace_base_dir"], "results")

    # Служебные пути веб-панели и архив — в корне results/
    cfg["results_root"] = RESULTS_ROOT
    cfg["log_file"] = LOG_FILE
    cfg["pid_file"] = PID_FILE
    cfg["status_file"] = STATUS_FILE
    cfg["archive_file"] = ARCHIVE_FILE

    return cfg


CONFIG = load_config(CONFIG_PATH)


# ---------- DEVICE ----------
def resolve_device(cfg: Config) -> str:
    if cfg.get("device"):
        print(f"DEVICE взят из config.json: {cfg['device']}")
        return cfg["device"]

    parser = argparse.ArgumentParser(
        description=f"Запуск DFT расчетов для кластеров {cfg['element']} и обучение MACE.\n"
                    "Подсказки по параметрам config.json: --help-config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--cpu", action="store_true", help="Использовать CPU для MACE (по умолчанию).")
    parser.add_argument("--gpu", action="store_true", help="Использовать GPU для MACE.")
    parser.add_argument("--help-config", action="store_true",
                        help="Показать все параметры config.json с подсказками и выйти.")
    args = parser.parse_args()

    if args.help_config:
        cfg.print_help()
        sys.exit(0)

    if not args.cpu and not args.gpu:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Внимание: не указан режим (--cpu или --gpu). Использую --cpu по умолчанию.")
        print("=" * 60)
        return "cpu"

    return "gpu" if args.gpu else "cpu"


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


def get_done_ids(xyz_path):
    done = set()
    if not os.path.exists(xyz_path):
        return done
    
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
            
            m_id = re.search(r'struct_id=["\']?([^"\'\s]+)["\']?', comment)
            if m_id:
                sid = m_id.group(1)
                if sid.endswith(".0"):
                    sid = sid[:-2]
                done.add(sid)
            for _ in range(n_atoms):
                f.readline()
    return done


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
    global DEVICE

    DEVICE = resolve_device(CONFIG)

    print("=" * 60)
    print(f"ЗАПУСК В РЕЖИМЕ: {DEVICE.upper()}")
    print(f"Элемент: {CONFIG['element']}")
    print(f"Рабочий каталог: {CONFIG['output_dir']}")
    print("=" * 60)

    # 0) Создаём каталог для элемента
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 0.1) Расчёт/чтение энергии изолированного атома (СРАЗУ, до генерации)
    e_atom = cp2k_runner.calculate_isolated_atom_energy(CONFIG)

    # 1) Генерация структур
    print("\n" + "=" * 60)
    print("ГЕНЕРАЦИЯ СТРУКТУР")
    print("=" * 60)
    generate_structures(CONFIG)

    # 2) Подготовка MACE-команды
    mace_model_name = f"{CONFIG['element']}_all_finetune"
    mace_cmd = [
        "mace_run_train",
        "--name", mace_model_name,
        "--foundation_model", CONFIG["mace_foundation_model"],
        "--multiheads_finetuning", "False",
        "--batch_size", str(CONFIG["mace_batch_size"]),
        "--train_file", os.path.join(CONFIG["output_dir"], CONFIG["output_all_xyz_name"]),
        "--valid_fraction", str(CONFIG["mace_valid_fraction"]),
        "--energy_key", "energy",
        "--forces_key", "forces",
        "--E0s", "estimated",
        "--energy_weight", str(CONFIG["mace_energy_weight"]),
        "--forces_weight", str(CONFIG["mace_forces_weight"]),
        "--lr", str(CONFIG["mace_lr"]),
        "--max_num_epochs", str(CONFIG["mace_max_num_epochs"]),
        "--eval_interval", str(CONFIG["mace_eval_interval"]),
        "--device", DEVICE,
        "--default_dtype", "float32",
        "--checkpoints_dir", CONFIG["mace_checkpoints_dir"],
        "--model_dir", CONFIG["mace_model_dir"],
        "--log_dir", CONFIG["mace_log_dir"],
        "--results_dir", CONFIG["mace_results_dir"],
    ]
    
    input_xyz = CONFIG["output_xyz"]
    if not os.path.exists(input_xyz):
        print(f"✗ Входной файл {input_xyz} не найден.")
        raise SystemExit(1)

    xyz_path = os.path.join(CONFIG["output_dir"], CONFIG["output_all_xyz_name"])

    structures = read_structures_with_meta(input_xyz)
    print(f"Загружено {len(structures)} структур из {input_xyz}")

    done_ids = get_done_ids(xyz_path)
    failed_ids = cp2k_runner.read_failed_ids(CONFIG["failed_ids_file"])
    
    print(f"Уже посчитано уникальных struct_id: {len(done_ids)}")
    print(f"Пропущено сбойных структур (из {CONFIG['failed_ids_file']}): {len(failed_ids)}")
    
    # 3) DFT через CP2K
    print("\n" + "=" * 60)
    print("DFT РАСЧЁТ (CP2K)")
    print("=" * 60)
    cp2k_runner.run_dft_for_structures(
        structures=structures,
        xyz_path=xyz_path,
        done_ids=done_ids,
        failed_ids=failed_ids,
        cfg=CONFIG,
    )

    # 4) MACE fine-tuning
    if os.path.exists(CONFIG["mace_base_dir"]):
        print(f"\nУдаляю старый каталог MACE: {CONFIG['mace_base_dir']}")
        shutil.rmtree(CONFIG["mace_base_dir"])
    os.makedirs(CONFIG["mace_checkpoints_dir"], exist_ok=True)
    os.makedirs(CONFIG["mace_model_dir"], exist_ok=True)
    os.makedirs(CONFIG["mace_log_dir"], exist_ok=True)
    os.makedirs(CONFIG["mace_results_dir"], exist_ok=True)

    print("\n" + "=" * 60)
    print(f"ЗАПУСК MACE FINE-TUNING (устройство: {DEVICE.upper()})")
    print("=" * 60)

    mace_result = subprocess.run(mace_cmd)
    if mace_result.returncode != 0:
        print(f"✗ MACE завершился с кодом {mace_result.returncode}")
        raise SystemExit(1)
    print("✓ MACE завершился успешно")

    model_files = glob.glob(os.path.join(CONFIG["mace_model_dir"], "*.model"))
    if len(model_files) == 0:
        print(f"✗ В {CONFIG['mace_model_dir']} не найден ни один .model файл")
        raise SystemExit(1)
    if len(model_files) > 1:
        model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]
    print(f"Используется модель: {model_path}")

    # 5) Поиск минимума
    print("\n" + "=" * 60)
    print(f"ПОИСК МИНИМУМА (устройство: {DEVICE.upper()})")
    print("=" * 60)

    calc = MACECalculator(
        model_path=model_path,
        device=DEVICE,
        default_dtype="float32",
    )

    result = run_find_minimum(
        calc=calc,
        xyz_path=xyz_path,
        output_xyz=os.path.join(CONFIG["output_dir"], f"{CONFIG['element']}_optimized_first_structure.xyz"),
        logfile=os.path.join(CONFIG["output_dir"], f"{CONFIG['element']}_opt.log"),
    )

    print("\n" + "=" * 60)
    print("ИТОГ ПО РАЗМЕРАМ КЛАСТЕРОВ")
    print("=" * 60)
    for n_atoms in sorted(result.keys()):
        r = result[n_atoms]
        be_per_atom = None
        if e_atom is not None:
            be_per_atom = (e_atom * n_atoms - r["energy"]) / n_atoms
            r["binding_energy_per_atom"] = be_per_atom

        if not np.isnan(r["distance"]):
            print(f"  n={n_atoms}: r = {r['distance']:.4f} Å, E = {r['energy']:.6f} eV")
        else:
            print(f"  n={n_atoms}: E = {r['energy']:.6f} eV")

        if be_per_atom is not None:
            print(f"       Энергия связи: {be_per_atom:.4f} eV/atom")

    # 6) Повторный CP2K для оптимизированных структур
    print("\n" + "=" * 60)
    print("ПОВТОРНЫЙ CP2K ДЛЯ ОПТИМИЗИРОВАННЫХ СТРУКТУР")
    print("=" * 60)

    for n_atoms in sorted(result.keys()):
        r = result[n_atoms]
        out_n_file = r["output_file"]

        if not os.path.exists(out_n_file):
            print(f"  ! Файл {out_n_file} не найден, пропускаю n={n_atoms}")
            continue

        opt_atoms = read(out_n_file)
        coords = opt_atoms.get_positions().tolist()
        struct_id_opt = f"opt_n{n_atoms}"

        print(f"\n  [n={n_atoms}] Повторный CP2K для оптимизированной структуры...")

        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", struct_id_opt)
        project = f"{CONFIG['element']}{n_atoms}_{safe_id}"
        inp_path = os.path.join(CONFIG["output_dir"], f"{project}.inp")
        out_path = os.path.join(CONFIG["output_dir"], f"{project}.out")
        force_path = os.path.join(CONFIG["output_dir"], f"{project}.forces")

        cp2k_runner.write_cp2k_input(inp_path, project, coords, CONFIG)

        t0 = time.time()
        is_cp2k_error, rc = cp2k_runner.run_cp2k_with_live_log(
            inp_path, out_path, CONFIG["timeout_per_config"], CONFIG["output_dir"], CONFIG
        )
        elapsed = time.time() - t0

        if rc != 0:
            print(f"  ✗ CP2K для оптимизированной n={n_atoms} завершился с ошибкой (код {rc})")
            continue

        try:
            energy = cp2k_runner.parse_energy(out_path)
            forces = cp2k_runner.parse_forces(force_path)
            gap = cp2k_runner.parse_homo_lumo(out_path, HARTREE_TO_EV)
            mag = cp2k_runner.parse_magnetic_moment(out_path)
        except Exception as e:
            print(f"  ✗ Ошибка парсинга для n={n_atoms}: {e}")
            continue

        energy_ev = energy * HARTREE_TO_EV
        forces_ev_ang = [
            [fx * HARTREE_BOHR_TO_EV_ANG,
             fy * HARTREE_BOHR_TO_EV_ANG,
             fz * HARTREE_BOHR_TO_EV_ANG]
            for fx, fy, fz in forces
        ]

        new_content = cp2k_runner.build_xyz_record(
            coords, energy_ev, forces_ev_ang, n_atoms, -1, struct_id_opt,
            CONFIG, gap, mag,
        )
        with open(out_n_file, "w") as f:
            f.write(new_content)

        r["homo_lumo_gap"] = gap
        r["magnetic_moment"] = mag

        # Энергия на атом (для оптимизированной структуры)
        e_per_atom = energy_ev / n_atoms

        msg = f"  ✓ n={n_atoms}: E = {energy_ev:.6f} eV за {elapsed:.1f} с"
        msg += f", E/atom = {e_per_atom:.6f} eV"
        if gap is not None:
            msg += f", Gap = {gap:.4f} eV"
        if mag is not None:
            msg += f", MagMom = {mag:.4f} μB"
        print(msg)

    # 7) Сборка всех минимумов + HTML
    print("\n" + "=" * 60)
    print("СБОРКА ВСЕХ МИНИМУМОВ В ОДИН ФАЙЛ И ГЕНЕРАЦИЯ HTML")
    print("=" * 60)

    combined_xyz_path = os.path.join(
        CONFIG["output_dir"], f"{CONFIG['element']}_optimized_first_structure_all_optimized.xyz"
    )

    if os.path.exists(combined_xyz_path):
        os.remove(combined_xyz_path)

    for n_atoms in sorted(result.keys()):
        r = result[n_atoms]
        out_n_file = r["output_file"]
        if os.path.exists(out_n_file):
            with open(out_n_file, "r") as f_in:
                content = f_in.read()

            if "binding_energy_per_atom" in r:
                be = r["binding_energy_per_atom"]
                lines = content.split("\n")
                if len(lines) > 1:
                    lines[1] = lines[1].replace(
                        "energy=", f"binding_energy_per_atom={be:.6f} energy="
                    )
                content = "\n".join(lines)

            with open(combined_xyz_path, "a") as f_out:
                f_out.write(content)

    print(f"Все минимумы собраны в: {combined_xyz_path}")

    web_server_dir = os.path.join(SCRIPT_DIR, "webserver", "static_pages")
    web_server_dir = os.path.abspath(web_server_dir)
    html_path = os.path.join(web_server_dir, f"{CONFIG['element']}_optimized_first_structure.html")

    structure_to_html(combined_xyz_path, html_path)
    print(f"Готово! HTML сохранен в папке веб-сервера: {html_path}")

    # Копия HTML в results/<element>/
    html_copy_path = os.path.join(
        CONFIG["output_dir"],
        f"{CONFIG['element']}_optimized_first_structure.html"
    )
    shutil.copy2(html_path, html_copy_path)
    print(f"Копия HTML сохранена в: {html_copy_path}")


if __name__ == "__main__":
    main()