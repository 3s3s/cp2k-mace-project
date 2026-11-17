#!/usr/bin/env python3
"""
Модуль для работы с CP2K.
"""

import sys
import os
import re
import time
import subprocess

# Словарь псевдопотенциалов для элементов
PSEUDOPOTENTIALS = {
    "Cu": "GTH-PBE-q11",
    "Fe": "GTH-PBE-q16",
    "Ar": "GTH-PBE-q8",
}

# ---------- Шаблон CP2K ----------
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
      MAX_SCF 200
      IGNORE_CONVERGENCE_FAILURE
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
    &KIND {element}
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL {potential}
    &END KIND
  &END SUBSYS

  &PRINT
    &FORCES ON
      FILENAME ={project}.forces
    &END FORCES
  &END PRINT
&END FORCE_EVAL
"""


# ---------- Генерация .inp ----------
def write_cp2k_input(inp_path: str, project: str, coords, cfg: dict):
    """Пишет .inp файл для заданных координат, центрируя их в ячейке."""
    import numpy as np
    
    ELEMENT = cfg['element']
    
    coords_arr = np.array(coords)
    center_of_mass = coords_arr.mean(axis=0)
    cell_center = np.array([cfg["cell_size"] / 2.0] * 3)
    coords_arr = coords_arr - center_of_mass + cell_center
    
    coords_block = "\n".join(
        f"      {ELEMENT}  {x:.8f}  {y:.8f}  {z:.8f}"
        for x, y, z in coords_arr
    )
    
    potential = PSEUDOPOTENTIALS.get(ELEMENT)
    if potential is None:
        print(f"✗ Для элемента {ELEMENT} не задан псевдопотенциал. "
              f"Доступные: {list(PSEUDOPOTENTIALS.keys())}")
        sys.exit(1) 
        
    with open(inp_path, "w") as f:
        f.write(TEMPLATE.format(
            project=project,
            cell_size=cfg["cell_size"],
            coords_block=coords_block,
            element=ELEMENT,
            potential=potential
        ))


# ---------- Парсинг ----------
def parse_energy(outfile: str) -> float:
    pattern = re.compile(
        r"ENERGY\|\s+Total FORCE_EVAL.*?energy\s+\[(?:hartree|a\.u\.)\]\s*:\s*(-?\d+\.\d+)"
    )
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"Энергия не найдена в {outfile}")
    
    
def parse_forces(forcefile: str):
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


def parse_homo_lumo(outfile: str, hartree_to_ev: float):
    """
    Ищет строку вида:
      HOMO - LUMO gap [eV] :    1.384688
    Возвращает значение в eV или None.
    """
    pattern = re.compile(
        r"HOMO\s*-\s*LUMO\s+gap\s*\[eV\]\s*:\s*(-?\d+\.\d+)",
        re.IGNORECASE,
    )
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def parse_magnetic_moment(outfile: str):
    """
    Ищет магнитный момент в выводе CP2K.
    Основной источник — строка:
      # Total charge and spin       11.000000    11.000000    -0.000000     0.000000
    Последнее число — суммарный спиновый момент в μB.
    """
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Total charge and spin" in line:
                parts = line.split()
                # parts[-1] — последнее число (спиновый момент)
                try:
                    return float(parts[-1])
                except (ValueError, IndexError):
                    continue

    # На случай, если формат другой — старые паттерны
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    for pat in [
        r"Total\s+Magnetic\s+Moment\s*[:=]\s*(-?\d+\.\d+)",
        r"Integrated\s+absolute\s+spin\s+density\s*[:=]\s*(-?\d+\.\d+)",
        r"Magnetization\s*\(spin\)\s*[:=]\s*(-?\d+\.\d+)",
    ]:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None
    
# ---------- Запуск CP2K ----------
def run_cp2k_with_live_log(inp_file, out_file, timeout, workdir, cfg: dict):
    """
    Возвращает (is_cp2k_error, returncode).
    - returncode > 0: внутренняя ошибка CP2K (структуру надо пометить как сбойную).
    - returncode < 0: системная ошибка (SIGKILL/OOM), структуру помечать НЕЛЬЗЯ.
    - Таймаут/расхождение SCF: возвращаем (True, -1), так как структура, скорее всего, проблемная.
    - Успех: (False, 0).
    """
    interesting = re.compile(
        r"(SCF WAVEFUNCTION|OT DIIS|OT SD|SCF run converged|"
        r"ENERGY\||ABORT|WARNING|error|Error|Total energy|"
        r"GEO_OPT|step number|Informations at step)"
    )

    scf_step_re = re.compile(r"^\s*(\d+)\s+OT\s+\S+\s+[\d.E+-]+\s+[\d.]+\s+([\d.E+-]+)")

    with open(out_file, "w", encoding="utf-8", errors="replace") as fout:
        try:
            proc = subprocess.Popen(
                [cfg["cp2k_cmd"], "-i", os.path.basename(inp_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=workdir,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            print(f"    ! ОШИБКА: Команда '{cfg['cp2k_cmd']}' не найдена в PATH.")
            return (False, -2)  # Системная ошибка (не вина структуры)

        start_time = time.time()
        last_print = start_time

        scf_convergence_history = []
        MAX_STALLED_STEPS = 10
        last_lines = []

        try:
            for line in proc.stdout:
                fout.write(line)
                fout.flush()

                last_lines.append(line.rstrip())
                if len(last_lines) > 15:
                    last_lines.pop(0)

                if cfg["verbose_cp2k"] and interesting.search(line):
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
                            return (True, -1)  # Считаем это проблемой структуры

                if time.time() - last_print > 60:
                    elapsed = int(time.time() - start_time)
                    print(f"    | ... (тишина {elapsed} с, CP2K работает)")
                    last_print = time.time()

                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"    ! ТАЙМАУТ ({timeout} с) — убиваю CP2K")
                    proc.kill()
                    return (True, -1)  # Считаем это проблемой структуры

            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return (True, -1)

        rc = proc.returncode

        if rc > 0:
            # Внутренняя ошибка CP2K
            print(f"    ! CP2K завершился с кодом возврата {rc} (внутренняя ошибка).")
            print(f"    ! Последние строки вывода:")
            for l in last_lines:
                print(f"      > {l}")
            print(f"    ! Полный лог сохранен в: {out_file}")
            return (True, rc)

        if rc < 0:
            # Системная ошибка (SIGKILL и т.п.) — не вина структуры
            print(f"    ! CP2K убит сигналом {-rc} (системная ошибка, вероятно OOM).")
            print(f"    ! Структура НЕ будет помечена как сбойная, попробуйте освободить RAM и перезапустить.")
            for l in last_lines:
                print(f"      > {l}")
            print(f"    ! Полный лог сохранен в: {out_file}")
            return (False, rc)

        return (False, 0)


# ---------- Энергия изолированного атома ----------
def calculate_isolated_atom_energy(cfg: dict):
    print("\n" + "=" * 60)
    print(f"ПРОВЕРКА/РАСЧЁТ ЭНЕРГИИ ИЗОЛИРОВАННОГО АТОМА {cfg['element']}")
    print("=" * 60)

    if os.path.exists(cfg["isolated_atom_energy_file"]):
        try:
            with open(cfg["isolated_atom_energy_file"], "r") as f:
                energy_ev = float(f.read().strip())
            print(f"✓ Энергия изолированного атома взята из кэша: {energy_ev:.6f} eV")
            return energy_ev
        except Exception as e:
            print(f"⚠ Ошибка чтения кэша: {e}. Пересчитываю...")
            os.remove(cfg["isolated_atom_energy_file"])

    project = f"{cfg['element']}_isolated_atom"
    inp_path = os.path.join(cfg["output_dir"], f"{project}.inp")
    out_path = os.path.join(cfg["output_dir"], f"{project}.out")

    half = cfg["cell_size"] / 2.0
    coords = [(half, half, half)]

    write_cp2k_input(inp_path, project, coords, cfg)

    is_cp2k_error, rc = run_cp2k_with_live_log(
        inp_path, out_path, cfg["timeout_per_config"], cfg["output_dir"], cfg
    )
    if is_cp2k_error or rc != 0:
        print("✗ Не удалось посчитать изолированный атом. Энергия связи не будет рассчитана.")
        return None

    try:
        energy_hartree = parse_energy(out_path)
        energy_ev = energy_hartree * cfg["hartree_to_ev"]
        print(f"✓ Энергия изолированного атома: {energy_ev:.6f} eV "
              f"({energy_hartree:.8f} Hartree)")

        with open(cfg["isolated_atom_energy_file"], "w") as f:
            f.write(f"{energy_ev:.8f}\n")

        return energy_ev
    except Exception as e:
        print(f"✗ Ошибка парсинга энергии изолированного атома: {e}")
        return None


# ---------- Высокоуровневый API ----------
def build_xyz_record(coords, energy_ev, forces_ev_ang, n_atoms, dim, struct_id,
                     cfg: dict, gap_ev=None, mag_moment=None, extra_comment=""):
    extra = ""
    if gap_ev is not None:
        extra += f"homo_lumo_gap={gap_ev:.6f} "
    if mag_moment is not None:
        extra += f"magnetic_moment={mag_moment:.6f} "
    extra += extra_comment

    comment = (
        f'Lattice="{cfg["lattice_str"]}" '
        f'Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy={energy_ev:.8f} '
        f'n_atoms={n_atoms} dim={dim} struct_id={struct_id} '
        f'{extra}'
        f'pbc="F F F"'
    )
    lines = [str(n_atoms), comment]
    for (x, y, z), (fx, fy, fz) in zip(coords, forces_ev_ang):
        lines.append(
            f"{cfg['element']} {x:.8f} {y:.8f} {z:.8f} "
            f"{fx:.8f} {fy:.8f} {fz:.8f}"
        )
    return "\n".join(lines) + "\n"


def run_dft_for_structures(structures, xyz_path, done_ids, failed_ids, cfg: dict):
    with open(xyz_path, "a") as xyz:
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
            project = f"{cfg['element']}{n_atoms}_{safe_id}"

            print(f"\n[{idx+1}/{len(structures)}] {struct_id} (n={n_atoms}, dim={dim})")

            inp_path = os.path.join(cfg["output_dir"], f"{project}.inp")
            out_path = os.path.join(cfg["output_dir"], f"{project}.out")
            force_path = os.path.join(cfg["output_dir"], f"{project}.forces")

            write_cp2k_input(inp_path, project, coords, cfg)

            t0 = time.time()
            is_cp2k_error, rc = run_cp2k_with_live_log(
                inp_path, out_path, cfg["timeout_per_config"], cfg["output_dir"], cfg
            )
            elapsed = time.time() - t0

            if rc != 0:
                # Различаем внутреннюю (rc>0) и системную (rc<0) ошибки
                if rc > 0 or is_cp2k_error:
                    print(f"  ✗ Внутренняя ошибка CP2K (код {rc}) за {elapsed:.1f} с")
                    print(f"  ! Помечаю {struct_id} как сбойную")
                    _mark_as_failed(struct_id, cfg["failed_ids_file"])
                    failed_ids.add(struct_id)
                else:
                    print(f"  ⚠ Системная ошибка (код {rc}) за {elapsed:.1f} с")
                    print(f"  ! Структура {struct_id} НЕ помечена как сбойная (проблема не в ней).")
                    print(f"  ! Освободите память и перезапустите, структура будет посчитана снова.")
                continue

            try:
                energy = parse_energy(out_path)
                forces = parse_forces(force_path)
                gap = parse_homo_lumo(out_path, cfg["hartree_to_ev"])
                mag = parse_magnetic_moment(out_path)
            except Exception as e:
                print(f"  ✗ Ошибка извлечения данных: {e}")
                print(f"  ! Помечаю {struct_id} как сбойную")
                _mark_as_failed(struct_id, cfg["failed_ids_file"])
                failed_ids.add(struct_id)
                continue

            if len(forces) != n_atoms:
                print(f"  ✗ Ожидалось {n_atoms} атомов, получено {len(forces)}")
                print(f"  ! Помечаю {struct_id} как сбойную")
                _mark_as_failed(struct_id, cfg["failed_ids_file"])
                failed_ids.add(struct_id)
                continue

            energy_ev = energy * cfg["hartree_to_ev"]
            forces_ev_ang = [
                [fx * cfg["hartree_bohr_to_ev_ang"],
                 fy * cfg["hartree_bohr_to_ev_ang"],
                 fz * cfg["hartree_bohr_to_ev_ang"]]
                for fx, fy, fz in forces
            ]

            msg = f"  ✓ OK: E = {energy_ev:.6f} eV за {elapsed:.1f} с"
            if gap is not None:
                msg += f", Gap = {gap:.4f} eV"
            if mag is not None:
                msg += f", MagMom = {mag:.4f} μB"
            print(msg)

            xyz.write(build_xyz_record(
                coords, energy_ev, forces_ev_ang, n_atoms, dim, struct_id,
                cfg, gap, mag,
            ))
            xyz.flush()
            done_ids.add(struct_id)


def _mark_as_failed(struct_id, filepath):
    with open(filepath, "a") as f:
        f.write(f"{struct_id}\n")


def read_failed_ids(filepath):
    failed = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    failed.add(line)
    return failed