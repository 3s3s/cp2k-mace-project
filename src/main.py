#!/usr/bin/env python3
"""
Читает структуры из generated_structures.xyz, для каждой запускает CP2K,
парсит энергию, силы, HOMO-LUMO Gap и магнитный момент.
Пишет всё в Cu_all_data.xyz.
Возобновление: пропускает те struct_id, которые уже есть в выходном файле.
Сбойные структуры помечаются в failed_ids.txt.
Энергия изолированного атома кэшируется.
Затем запускает MACE fine-tuning, находит глобальные минимумы,
считает энергию связи, собирает всё в один .xyz и генерирует HTML.

Использование:
  python main.py [--cpu | --gpu] [--help]
"""

import os
import subprocess
import re
import time
import sys
import shutil
import glob
import argparse
import numpy as np

from mace.calculators import MACECalculator
from find_minimum import run_find_minimum
from xyz_to_html import structure_to_html

ELEMENT = "Cu"

CELL_SIZE = 20.0
LATTICE_STR = f"{CELL_SIZE} 0.0 0.0 0.0 {CELL_SIZE} 0.0 0.0 0.0 {CELL_SIZE}"

HARTREE_TO_EV = 27.211386
HARTREE_BOHR_TO_EV_ANG = 51.422067

CP2K_CMD = "cp2k"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_XYZ = os.path.join(SCRIPT_DIR, "..", "results", "generated_structures.xyz")
OUTPUT_DIR = "results"
OUTPUT_XYZ_NAME = f"{ELEMENT}_all_data.xyz"
FAILED_IDS_FILE = os.path.join(OUTPUT_DIR, "failed_ids.txt")
ISOLATED_ATOM_ENERGY_FILE = os.path.join(OUTPUT_DIR, f"{ELEMENT}_isolated_atom_energy.txt")
TIMEOUT_PER_CONFIG = 3600
VERBOSE_CP2K = True

MACE_MODEL_NAME = f"{ELEMENT}_all_finetune"
MACE_BASE_DIR = "mace_output"
MACE_CHECKPOINTS_DIR = os.path.join(MACE_BASE_DIR, "checkpoints")
MACE_MODEL_DIR = os.path.join(MACE_BASE_DIR, "models")
MACE_LOG_DIR = os.path.join(MACE_BASE_DIR, "logs")
MACE_RESULTS_DIR = os.path.join(MACE_BASE_DIR, "results")

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
    &PRINT
      &MO_CUBES
        NHOMO 1
        NLUMO 1
      &END MO_CUBES
      &MULLIKEN ON
      &END MULLIKEN
      &MOMENTS ON
      &END MOMENTS
    &END PRINT
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Запуск DFT расчетов для кластеров Cu и обучение MACE.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Использовать CPU для MACE (по умолчанию)."
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Использовать GPU для MACE."
    )
    
    args = parser.parse_args()
    
    # Если не указан ни --cpu, ни --gpu, выводим хэлп и используем CPU по умолчанию
    if not args.cpu and not args.gpu:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Внимание: не указан режим (--cpu или --gpu). Использую --cpu по умолчанию.")
        print("=" * 60)
        return "cpu"
    
    if args.gpu:
        return "gpu"
    return "cpu"

# Глобальная переменная для устройства
DEVICE = "cpu"

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
            comment = f.readline()
            
            m_dim = re.search(r"dim=(\d+)", comment)
            dim = int(m_dim.group(1)) if m_dim else -1
            
            m_id = re.search(r"struct_id=(\S+)", comment)
            struct_id = m_id.group(1) if m_id else f"unknown_{len(structures)}"

            coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            structures.append({
                "n_atoms": n_atoms,
                "dim": dim,
                "struct_id": struct_id,
                "coords": coords
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
            comment = f.readline()
            m_id = re.search(r"struct_id=(\S+)", comment)
            if m_id:
                done.add(m_id.group(1))
            for _ in range(n_atoms):
                f.readline()
    return done

def get_failed_ids(filepath):
    failed = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    failed.add(line)
    return failed

def mark_as_failed(struct_id, filepath):
    with open(filepath, "a") as f:
        f.write(f"{struct_id}\n")

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

def parse_homo_lumo(outfile):
    homo = None
    lumo = None
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "HOMO" in line and "=" in line:
                m = re.search(r"=\s*(-?\d+\.\d+)", line)
                if m: homo = float(m.group(1))
            if "LUMO" in line and "=" in line:
                m = re.search(r"=\s*(-?\d+\.\d+)", line)
                if m: lumo = float(m.group(1))
    if homo is not None and lumo is not None:
        return (lumo - homo) * HARTREE_TO_EV
    return None

def parse_magnetic_moment(outfile):
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    m = re.search(r"Total Magnetic Moment\s*[:=]\s*(-?\d+\.\d+)", content)
    if m: return float(m.group(1))
    
    m = re.search(r"Integrated spin density\s*[:=]\s*(-?\d+\.\d+)", content)
    if m: return float(m.group(1))
    
    return None

def run_cp2k_with_live_log(inp_file, out_file, timeout, workdir):
    interesting = re.compile(
        r"(SCF WAVEFUNCTION|OT DIIS|OT SD|SCF run converged|"
        r"ENERGY\||ABORT|WARNING|error|Error|Total energy|"
        r"GEO_OPT|step number|Informations at step)"
    )
    
    scf_step_re = re.compile(r"^\s*(\d+)\s+OT\s+\S+\s+[\d.E+-]+\s+[\d.]+\s+([\d.E+-]+)")

    with open(out_file, "w", encoding="utf-8", errors="replace") as fout:
        proc = subprocess.Popen(
            [CP2K_CMD, "-i", os.path.basename(inp_file)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=workdir,
            encoding="utf-8", errors="replace",
        )
        start_time = time.time()
        last_print = start_time
        
        scf_convergence_history = []
        MAX_STALLED_STEPS = 10

        try:
            for line in proc.stdout:
                fout.write(line)
                fout.flush()
                
                if VERBOSE_CP2K and interesting.search(line):
                    print(f"    | {line.rstrip()}")
                    last_print = time.time()

                m_scf = scf_step_re.match(line)
                if m_scf:
                    step = int(m_scf.group(1))
                    conv = float(m_scf.group(2))
                    scf_convergence_history.append(conv)
                    
                    if len(scf_convergence_history) > MAX_STALLED_STEPS:
                        recent = scf_convergence_history[-MAX_STALLED_STEPS:]
                        if min(recent) >= scf_convergence_history[-MAX_STALLED_STEPS - 1]:
                            print(f"    ! ОБНАРУЖЕНО РАСХОЖДЕНИЕ SCF на шаге {step}. Убиваю CP2K.")
                            proc.kill()
                            return False

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
        
def build_xyz_record(coords, energy_ev, forces_ev_ang, n_atoms, dim, struct_id, gap_ev=None, mag_moment=None, extra_comment=""):
    extra = ""
    if gap_ev is not None:
        extra += f"homo_lumo_gap={gap_ev:.6f} "
    if mag_moment is not None:
        extra += f"magnetic_moment={mag_moment:.6f} "
    extra += extra_comment
    
    comment = (
        f'Lattice="{LATTICE_STR}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy_ev:.8f} '
        f'n_atoms={n_atoms} dim={dim} struct_id={struct_id} '
        f'{extra}'
        f'pbc="F F F"'
    )
    lines = [str(n_atoms), comment]
    for (x, y, z), (fx, fy, fz) in zip(coords, forces_ev_ang):
        lines.append(
            f"{ELEMENT} {x:.8f} {y:.8f} {z:.8f} "
            f"{fx:.8f} {fy:.8f} {fz:.8f}"
        )
    return "\n".join(lines) + "\n"

def calculate_isolated_atom_energy():
    print("\n" + "=" * 60)
    print("ПРОВЕРКА/РАСЧЁТ ЭНЕРГИИ ИЗОЛИРОВАННОГО АТОМА Cu")
    print("=" * 60)
    
    if os.path.exists(ISOLATED_ATOM_ENERGY_FILE):
        try:
            with open(ISOLATED_ATOM_ENERGY_FILE, "r") as f:
                energy_ev = float(f.read().strip())
            print(f"✓ Энергия изолированного атома взята из кэша: {energy_ev:.6f} eV")
            return energy_ev
        except Exception as e:
            print(f"⚠ Ошибка чтения кэша: {e}. Пересчитываю...")
            os.remove(ISOLATED_ATOM_ENERGY_FILE)

    project = f"{ELEMENT}_isolated_atom"
    inp_file = f"{project}.inp"
    out_file = f"{project}.out"
    
    inp_path = os.path.join(OUTPUT_DIR, inp_file)
    out_path = os.path.join(OUTPUT_DIR, out_file)
    
    coords_block = f"      {ELEMENT}  {CELL_SIZE/2:.8f}  {CELL_SIZE/2:.8f}  {CELL_SIZE/2:.8f}"
    
    with open(inp_path, "w") as f:
        f.write(TEMPLATE.format(
            project=project,
            cell_size=CELL_SIZE,
            coords_block=coords_block,
        ))
    
    ok = run_cp2k_with_live_log(inp_path, out_path, TIMEOUT_PER_CONFIG, OUTPUT_DIR)
    if not ok:
        print("✗ Не удалось посчитать изолированный атом. Энергия связи не будет рассчитана.")
        return None
    
    try:
        energy_hartree = parse_energy(out_path)
        energy_ev = energy_hartree * HARTREE_TO_EV
        print(f"✓ Энергия изолированного атома: {energy_ev:.6f} eV ({energy_hartree:.8f} Hartree)")
        
        with open(ISOLATED_ATOM_ENERGY_FILE, "w") as f:
            f.write(f"{energy_ev:.8f}\n")
            
        return energy_ev
    except Exception as e:
        print(f"✗ Ошибка парсинга энергии изолированного атома: {e}")
        return None

def main():
    global DEVICE
    DEVICE = parse_args()

    print("=" * 60)
    print(f"ЗАПУСК В РЕЖИМЕ: {DEVICE.upper()}")
    print("=" * 60)

    # Формируем команду MACE уже с нужным DEVICE
    mace_cmd = [
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
        "--device", DEVICE,
        "--default_dtype", "float32",
        "--checkpoints_dir", MACE_CHECKPOINTS_DIR,
        "--model_dir", MACE_MODEL_DIR,
        "--log_dir", MACE_LOG_DIR,
        "--results_dir", MACE_RESULTS_DIR,
    ]

    if not os.path.exists(INPUT_XYZ):
        print(f"✗ Входной файл {INPUT_XYZ} не найден. Сначала запустите structure_generator.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xyz_path = os.path.join(OUTPUT_DIR, OUTPUT_XYZ_NAME)

    structures = read_structures_with_meta(INPUT_XYZ)
    print(f"Загружено {len(structures)} структур из {INPUT_XYZ}")

    done_ids = get_done_ids(xyz_path)
    failed_ids = get_failed_ids(FAILED_IDS_FILE)
    print(f"Уже посчитано уникальных struct_id: {len(done_ids)}")
    print(f"Пропущено сбойных структур (из {FAILED_IDS_FILE}): {len(failed_ids)}")

    mode = "a" if done_ids else "w"
    with open(xyz_path, mode) as xyz:
        for idx, s in enumerate(structures):
            n_atoms = s["n_atoms"]
            dim = s["dim"]
            struct_id = s["struct_id"]

            if struct_id in done_ids:
                print(f"\n[{idx+1}/{len(structures)}] {struct_id} — уже посчитано, пропускаю")
                continue

            if struct_id in failed_ids:
                print(f"\n[{idx+1}/{len(structures)}] {struct_id} — ранее признано сбойным, пропускаю")
                continue

            coords = s["coords"]
            safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", struct_id)
            project = f"{ELEMENT}{n_atoms}_{safe_id}"
            inp_file = f"{project}.inp"
            out_file = f"{project}.out"
            force_file = f"{project}.forces"

            print(f"\n[{idx+1}/{len(structures)}] {struct_id} (n={n_atoms}, dim={dim})")

            inp_path = os.path.join(OUTPUT_DIR, inp_file)
            out_path = os.path.join(OUTPUT_DIR, out_file)
            force_path = os.path.join(OUTPUT_DIR, force_file)

            coords_block = "\n".join(
                f"      {ELEMENT}  {x:.8f}  {y:.8f}  {z:.8f}" for x, y, z in coords
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
                print(f"  ! Помечаю {struct_id} как сбойную и сохраняю в {FAILED_IDS_FILE}")
                mark_as_failed(struct_id, FAILED_IDS_FILE)
                failed_ids.add(struct_id)
                continue

            try:
                energy = parse_energy(out_path)
                forces = parse_forces(force_path)
                gap = parse_homo_lumo(out_path)
                mag = parse_magnetic_moment(out_path)
            except Exception as e:
                print(f"  ✗ Ошибка извлечения данных: {e}")
                print(f"  ! Помечаю {struct_id} как сбойную и сохраняю в {FAILED_IDS_FILE}")
                mark_as_failed(struct_id, FAILED_IDS_FILE)
                failed_ids.add(struct_id)
                continue

            if len(forces) != n_atoms:
                print(f"  ✗ Ожидалось {n_atoms} атомов, получено {len(forces)}")
                print(f"  ! Помечаю {struct_id} как сбойную и сохраняю в {FAILED_IDS_FILE}")
                mark_as_failed(struct_id, FAILED_IDS_FILE)
                failed_ids.add(struct_id)
                continue

            energy_ev = energy * HARTREE_TO_EV
            forces_ev_ang = [
                [fx * HARTREE_BOHR_TO_EV_ANG,
                 fy * HARTREE_BOHR_TO_EV_ANG,
                 fz * HARTREE_BOHR_TO_EV_ANG]
                for fx, fy, fz in forces
            ]

            msg = f"  ✓ OK: E = {energy_ev:.6f} eV за {elapsed:.1f} с"
            if gap is not None:
                msg += f", Gap = {gap:.4f} eV"
            if mag is not None:
                msg += f", MagMom = {mag:.4f} μB"
            print(msg)

            xyz.write(build_xyz_record(coords, energy_ev, forces_ev_ang, n_atoms, dim, struct_id, gap, mag))
            xyz.flush()
            done_ids.add(struct_id)

    e_atom = calculate_isolated_atom_energy()

    if os.path.exists(MACE_BASE_DIR):
        print(f"\nУдаляю старый каталог MACE: {MACE_BASE_DIR}")
        shutil.rmtree(MACE_BASE_DIR)
    os.makedirs(MACE_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(MACE_MODEL_DIR, exist_ok=True)
    os.makedirs(MACE_LOG_DIR, exist_ok=True)
    os.makedirs(MACE_RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"ЗАПУСК MACE FINE-TUNING (устройство: {DEVICE.upper()})")
    print("=" * 60)
    
    mace_result = subprocess.run(mace_cmd)
    if mace_result.returncode != 0:
        print(f"✗ MACE завершился с кодом {mace_result.returncode}")
        sys.exit(1)
    print("✓ MACE завершился успешно")

    model_files = glob.glob(os.path.join(MACE_MODEL_DIR, "*.model"))
    if len(model_files) == 0:
        print(f"✗ В {MACE_MODEL_DIR} не найден ни один .model файл")
        sys.exit(1)
    if len(model_files) > 1:
        model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]
    print(f"Используется модель: {model_path}")

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
        output_xyz=os.path.join(OUTPUT_DIR, "optimized_first_structure.xyz"),
        logfile=os.path.join(OUTPUT_DIR, "opt.log"),
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

    print("\n" + "=" * 60)
    print("СБОРКА ВСЕХ МИНИМУМОВ В ОДИН ФАЙЛ И ГЕНЕРАЦИЯ HTML")
    print("=" * 60)
    
    combined_xyz_path = os.path.join(OUTPUT_DIR, "optimized_first_structure_all_optimized.xyz")
    
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
                    lines[1] = lines[1].replace('energy=', f'binding_energy_per_atom={be:.6f} energy=')
                content = "\n".join(lines)

            with open(combined_xyz_path, "a") as f_out:
                f_out.write(content)
                
    print(f"Все минимумы собраны в: {combined_xyz_path}")

    web_server_dir = os.path.join(SCRIPT_DIR, "webserver", "static_pages")
    web_server_dir = os.path.abspath(web_server_dir)
    html_path = os.path.join(web_server_dir, "optimized_first_structure.html")

    structure_to_html(combined_xyz_path, html_path)
    print(f"Готово! HTML сохранен в папке веб-сервера: {html_path}")

if __name__ == "__main__":
    main()