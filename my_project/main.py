#!/usr/bin/env python3
"""
Скрипт: читает структуры из generated_structures.xyz, для каждой запускает CP2K
(ENERGY_FORCE), парсит энергию и силы, пишет в cu_all_data.xyz (eV, eV/Å),
затем запускает MACE fine-tuning и find_minimum.
Поддерживает возобновление после прерывания.
Все служебные файлы MACE складываются в MACE_BASE_DIR, который очищается перед запуском.
"""

import os
import subprocess
import re
import time
import sys
import shutil
import glob

from find_minimum import run_find_minimum

# ---------- КОНСТАНТЫ ----------
CELL_SIZE = 20.0
CENTER_XY = CELL_SIZE / 2.0
CENTER_Z = CELL_SIZE / 2.0
LATTICE_STR = f"{CELL_SIZE} 0.0 0.0 0.0 {CELL_SIZE} 0.0 0.0 0.0 {CELL_SIZE}"

HARTREE_TO_EV = 27.211386
HARTREE_BOHR_TO_EV_ANG = 51.422067

CP2K_CMD = "cp2k"
INPUT_XYZ = "generated_structures.xyz"          # от structure_generator.py
OUTPUT_DIR = "cu_all_run"                        # все файлы расчётов
OUTPUT_XYZ_NAME = "cu_all_data.xyz"              # итоговый extxyz для MACE
TIMEOUT_PER_CONFIG = 1800
VERBOSE_CP2K = True

# Параметры MACE и каталоги
MACE_MODEL_NAME = "cu_all_finetune"
MACE_BASE_DIR = "mace_output"
MACE_CHECKPOINTS_DIR = os.path.join(MACE_BASE_DIR, "checkpoints")
MACE_MODEL_DIR = os.path.join(MACE_BASE_DIR, "models")
MACE_LOG_DIR = os.path.join(MACE_BASE_DIR, "logs")
MACE_RESULTS_DIR = os.path.join(MACE_BASE_DIR, "results")

MACE_CMD_BASE = [
    "mace_run_train",
    "--name", MACE_MODEL_NAME,
    "--foundation_model", "small",
    "--multiheads_finetuning", "False",
    "--batch_size", "1",
    "--train_file", os.path.join(OUTPUT_DIR, OUTPUT_XYZ_NAME),
    "--valid_fraction", "0.2",
    "--energy_key", "energy",
    "--forces_key", "forces",
    "--E0s", "estimated",
    "--energy_weight", "1.0",
    "--forces_weight", "10.0",
    "--lr", "0.001",
    "--max_num_epochs", "50",
    "--eval_interval", "5",
    "--device", "cpu",
    "--default_dtype", "float32",
    "--checkpoints_dir", MACE_CHECKPOINTS_DIR,
    "--model_dir", MACE_MODEL_DIR,
    "--log_dir", MACE_LOG_DIR,
    "--results_dir", MACE_RESULTS_DIR,
]

# ---------- ШАБЛОН CP2K ----------
TEMPLATE = """&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL LOW
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    LSD
    &MGRID
      CUTOFF 400
      REL_CUTOFF 50
    &END MGRID
    &QS
      EPS_DEFAULT 1.0E-10
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 100
      &OT
        MINIMIZER DIIS
        PRECONDITIONER FULL_SINGLE_INVERSE
      &END OT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &POISSON
      POISSON_SOLVER WAVELET
      PERIODIC NONE
    &END POISSON
  &END DFT

  &SUBSYS
    &CELL
      ABC {cell_size} {cell_size} {cell_size}
      PERIODIC NONE
    &END CELL
    &COORD
{coords_block}
    &END COORD
    &KIND Cu
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE-q11
    &END KIND
  &END SUBSYS

  &PRINT
    &FORCES ON
      FILENAME ={project}.forces
    &END FORCES
  &END PRINT
&END FORCE_EVAL
"""

# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------
def read_structures(xyz_path):
    """Читает extxyz, возвращает список структур (каждая — список (x,y,z))."""
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
            f.readline()  # comment
            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            structures.append(coords)
    return structures

def parse_energy(outfile):
    pattern = re.compile(r"ENERGY\|\s+Total FORCE_EVAL.*?energy \[hartree\]\s+(-?\d+\.\d+)")
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"Энергия не найдена в {outfile}")

def parse_forces(forcefile):
    forces = []
    with open(forcefile, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if "ATOMIC FORCES in [a.u.]" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Секция ATOMIC FORCES не найдена в {forcefile}")
    j = start
    while j < len(lines):
        s = lines[j].strip()
        if s == "" or s.startswith("#"):
            j += 1
            continue
        break
    for line in lines[j:]:
        s = line.strip()
        if s == "" or s.startswith("SUM OF ATOMIC FORCES"):
            break
        parts = s.split()
        if len(parts) < 6:
            break
        try:
            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
        except ValueError:
            break
        forces.append([fx, fy, fz])
    return forces

def run_cp2k_with_live_log(inp_file, out_file, timeout, workdir):
    interesting = re.compile(
        r"(SCF WAVEFUNCTION|OT DIIS|OT SD|SCF run converged|"
        r"ENERGY\||ABORT|WARNING|error|Error|Total energy|"
        r"GEO_OPT|step number|Informations at step)"
    )
    with open(out_file, "w", encoding="utf-8", errors="replace") as fout:
        proc = subprocess.Popen(
            [CP2K_CMD, "-i", os.path.basename(inp_file)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=workdir,
            encoding="utf-8", errors="replace",
        )
        start_time = time.time()
        last_print = start_time
        try:
            for line in proc.stdout:
                fout.write(line)
                fout.flush()
                if VERBOSE_CP2K and interesting.search(line):
                    print(f"    | {line.rstrip()}")
                    last_print = time.time()
                if time.time() - last_print > 60:
                    elapsed = int(time.time() - start_time)
                    print(f"    | ... (тишина {elapsed} с, CP2K работает)")
                    last_print = time.time()
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"    ! ТАЙМАУТ ({timeout} с) — убиваю CP2K")
                    proc.kill()
                    return False
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return False
        return proc.returncode == 0

def build_xyz_record(coords, energy_ev, forces_ev_ang):
    n = len(coords)
    comment = (
        f'Lattice="{LATTICE_STR}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy_ev:.8f} '
        f'pbc="F F F"'
    )
    lines = [str(n), comment]
    for (x, y, z), (fx, fy, fz) in zip(coords, forces_ev_ang):
        lines.append(
            f"Cu {x:.8f} {y:.8f} {z:.8f} "
            f"{fx:.8f} {fy:.8f} {fz:.8f}"
        )
    return "\n".join(lines) + "\n"

def count_xyz_records(xyz_path):
    if not os.path.exists(xyz_path):
        return 0
    n = 0
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
            f.readline()
            for _ in range(n_atoms):
                f.readline()
            n += 1
    return n

# ---------- ОСНОВНОЙ ЦИКЛ ----------
def main():
    if not os.path.exists(INPUT_XYZ):
        print(f"✗ Входной файл {INPUT_XYZ} не найден. Сначала запустите structure_generator.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xyz_path = os.path.join(OUTPUT_DIR, OUTPUT_XYZ_NAME)

    structures = read_structures(INPUT_XYZ)
    print(f"Загружено {len(structures)} структур из {INPUT_XYZ}")

    n_done = count_xyz_records(xyz_path)
    print(f"Уже записано конфигураций в {xyz_path}: {n_done}")

    mode = "a" if n_done > 0 else "w"
    with open(xyz_path, mode) as xyz:
        for idx, coords in enumerate(structures):
            if idx < n_done:
                print(f"\n[{idx+1}/{len(structures)}] — уже посчитано, пропускаю")
                continue

            n_atoms = len(coords)
            project = f"cu{n_atoms}_{idx:03d}"
            inp_file = f"{project}.inp"
            out_file = f"{project}.out"
            force_file = f"{project}.forces"

            print(f"\n[{idx+1}/{len(structures)}] n_atoms={n_atoms}  (проект {project})")

            inp_path = os.path.join(OUTPUT_DIR, inp_file)
            out_path = os.path.join(OUTPUT_DIR, out_file)
            force_path = os.path.join(OUTPUT_DIR, force_file)

            coords_block = "\n".join(
                f"      Cu  {x:.8f}  {y:.8f}  {z:.8f}" for x, y, z in coords
            )

            with open(inp_path, "w") as f:
                f.write(TEMPLATE.format(
                    project=project,
                    cell_size=CELL_SIZE,
                    coords_block=coords_block,
                ))

            t0 = time.time()
            ok = run_cp2k_with_live_log(inp_path, out_path, TIMEOUT_PER_CONFIG, OUTPUT_DIR)
            elapsed = time.time() - t0

            if not ok:
                print(f"  ✗ CP2K завершился с ошибкой/таймаутом за {elapsed:.1f} с")
                continue

            try:
                energy = parse_energy(out_path)
                forces = parse_forces(force_path)
            except Exception as e:
                print(f"  ✗ Ошибка извлечения данных: {e}")
                continue

            if len(forces) != n_atoms:
                print(f"  ✗ Ожидалось {n_atoms} атомов, получено {len(forces)}")
                continue

            energy_ev = energy * HARTREE_TO_EV
            forces_ev_ang = [
                [fx * HARTREE_BOHR_TO_EV_ANG,
                 fy * HARTREE_BOHR_TO_EV_ANG,
                 fz * HARTREE_BOHR_TO_EV_ANG]
                for fx, fy, fz in forces
            ]

            xyz.write(build_xyz_record(coords, energy_ev, forces_ev_ang))
            xyz.flush()
            print(f"  ✓ OK: E = {energy_ev:.6f} eV за {elapsed:.1f} с")

    # ---------- ОЧИСТКА И ЗАПУСК MACE ----------
    if os.path.exists(MACE_BASE_DIR):
        print(f"\nУдаляю старый каталог MACE: {MACE_BASE_DIR}")
        shutil.rmtree(MACE_BASE_DIR)
    os.makedirs(MACE_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(MACE_MODEL_DIR, exist_ok=True)
    os.makedirs(MACE_LOG_DIR, exist_ok=True)
    os.makedirs(MACE_RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("ЗАПУСК MACE FINE-TUNING")
    print("=" * 60)
    mace_result = subprocess.run(MACE_CMD_BASE)
    if mace_result.returncode != 0:
        print(f"✗ MACE завершился с кодом {mace_result.returncode}")
        sys.exit(1)
    print("✓ MACE завершился успешно")

    # ---------- ПОИСК ЕДИНСТВЕННОЙ МОДЕЛИ ----------
    model_files = glob.glob(os.path.join(MACE_MODEL_DIR, "*.model"))
    if len(model_files) == 0:
        print(f"✗ В {MACE_MODEL_DIR} не найден ни один .model файл")
        sys.exit(1)
    if len(model_files) > 1:
        print(f"⚠ В {MACE_MODEL_DIR} найдено несколько .model файлов: {model_files}")
        print("  Использую самый свежий.")
        model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]
    print(f"Используется модель: {model_path}")

    # ---------- ЗАПУСК find_minimum ----------
    result = run_find_minimum(
        model_path=model_path,
        xyz_path=xyz_path,
        output_xyz=os.path.join(OUTPUT_DIR, "optimized_first_structure.xyz"),
        logfile=os.path.join(OUTPUT_DIR, "opt.log"),
    )
    print(f"\nИтог: r = {result['final_distance']:.4f} Å, E = {result['final_energy']:.6f} eV")

if __name__ == "__main__":
    main()