#!/usr/bin/env python3
"""
Скрипт для расчёта HOMO-LUMO gap через CP2K по входному .xyz файлу.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

PSEUDOPOTENTIALS = {
    "Cu": "GTH-PBE-q11",
    "Fe": "GTH-PBE-q16",
    "Ar": "GTH-PBE-q8",
}

HARTREE_TO_EV = 27.211386245988

TEMPLATE = """&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    LSD
    MULTIPLICITY {multiplicity}
    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF {rel_cutoff}
    &END MGRID
    &QS
      EPS_DEFAULT 1.0E-10
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 200
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
        WRITE_CUBE .FALSE.
      &END MO_CUBES
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
{kinds_block}
  &END SUBSYS
&END FORCE_EVAL
"""


def log(msg: str, stream=sys.stdout):
    print(msg, file=stream, flush=True)


def read_xyz(xyz_path: str):
    """Читает первый кадр из .xyz файла (поддерживает стандартный и extended XYZ)."""
    if not os.path.isfile(xyz_path):
        raise FileNotFoundError(f"Файл не найден: {xyz_path}")

    with open(xyz_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\r\n") for line in f]

    if len(lines) < 2:
        raise ValueError(f"Некорректный формат XYZ в {xyz_path}")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as e:
        raise ValueError(f"Первая строка XYZ должна содержать число атомов: {lines[0]!r}") from e

    if n_atoms <= 0 or len(lines) < 2 + n_atoms:
        raise ValueError(f"Недостаточно строк координат для {n_atoms} атомов в {xyz_path}")

    comment = lines[1]
    symbols = []
    coords = []

    for idx in range(2, 2 + n_atoms):
        parts = lines[idx].split()
        if len(parts) < 4:
            raise ValueError(f"Некорректная строка атома #{idx - 1}: {lines[idx]!r}")
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as e:
            raise ValueError(f"Некорректные координаты в строке #{idx - 1}: {lines[idx]!r}") from e
        symbols.append(sym)
        coords.append([x, y, z])

    return symbols, np.array(coords, dtype=float), comment


def build_kinds_block(symbols: list[str]) -> str:
    unique_elems = list(dict.fromkeys(symbols))
    missing = [el for el in unique_elems if el not in PSEUDOPOTENTIALS]
    if missing:
        raise ValueError(
            f"Нет псевдопотенциала для элементов: {missing}. "
            f"Доступные: {list(PSEUDOPOTENTIALS.keys())}"
        )

    blocks = []
    for el in unique_elems:
        blocks.append(
            f"    &KIND {el}\n"
            f"      BASIS_SET DZVP-MOLOPT-SR-GTH\n"
            f"      POTENTIAL {PSEUDOPOTENTIALS[el]}\n"
            f"    &END KIND"
        )
    return "\n".join(blocks)


def write_cp2k_input(
    inp_path: str,
    project: str,
    symbols: list[str],
    coords: np.ndarray,
    cell_size: float,
    cutoff: float,
    rel_cutoff: float,
    multiplicity: int,
):
    center = coords.mean(axis=0)
    cell_center = np.array([cell_size / 2.0, cell_size / 2.0, cell_size / 2.0])
    centered = coords - center + cell_center

    if np.any(centered < 0.0) or np.any(centered > cell_size):
        log(
            "⚠ Внимание: часть атомов вышла за границы ячейки после центрирования. "
            "Увеличьте --cell-size.",
            stream=sys.stderr,
        )

    coords_block = "\n".join(
        f"      {sym}  {x:.8f}  {y:.8f}  {z:.8f}"
        for sym, (x, y, z) in zip(symbols, centered)
    )
    kinds_block = build_kinds_block(symbols)

    content = TEMPLATE.format(
        project=project,
        multiplicity=multiplicity,
        cutoff=cutoff,
        rel_cutoff=rel_cutoff,
        cell_size=cell_size,
        coords_block=coords_block,
        kinds_block=kinds_block,
    )
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_homo_lumo_gap(outfile: str) -> float | None:
    """
    Ищет HOMO - LUMO gap [eV] напрямую в выводе CP2K.
    """
    pattern = re.compile(
        r"HOMO\s*-\s*LUMO\s+gap\s*\[eV\]\s*:\s*(-?\d+\.\d+)",
        re.IGNORECASE,
    )
    last_gap = None
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last_gap = float(m.group(1))
    return last_gap


def check_scf_converged(outfile: str) -> bool:
    with open(outfile, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if "SCF run converged" in text:
        return True
    if "SCF run NOT converged" in text:
        return False
    return False


def extract_error_context(out_path: str, last_lines: list[str]) -> str:
    diag_lines = []
    try:
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.rstrip()
                if any(k in s.lower() for k in ("error", "found", "unknown", "invalid", "abort")):
                    diag_lines.append(s)
    except OSError:
        pass

    parts = []
    if diag_lines:
        parts.append("Диагностические строки:\n" + "\n".join(diag_lines[-20:]))
    parts.append("Последние строки:\n" + "\n".join(last_lines))
    return "\n\n".join(parts)


def run_cp2k(
    cp2k_cmd: str,
    inp_path: str,
    out_path: str,
    workdir: str,
    timeout: int | None,
    verbose: bool = True,
    raw_cp2k_log: bool = False,
):
    interesting = re.compile(
        r"(SCF WAVEFUNCTION|OT DIIS|OT SD|SCF run converged|SCF run NOT converged|"
        r"ENERGY\||HOMO\s*-\s*LUMO|ABORT|WARNING|error|Error|Total energy)"
    )
    scf_step_re = re.compile(r"^\s*(\d+)\s+(?:OT|CG|DIIS|SD)\s+\S+\s+[\d.E+-]+\s+[\d.]+\s+([\d.E+-]+)")

    cmd = [cp2k_cmd, "-i", os.path.basename(inp_path)]
    log(f"    Запуск команды: {' '.join(cmd)} (cwd={workdir})")

    with open(out_path, "w", encoding="utf-8", errors="replace") as fout:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=workdir,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Команда '{cp2k_cmd}' не найдена в PATH.") from e

        log(f"    Процесс CP2K запущен (PID={proc.pid})")
        start = time.time()
        last_print = start
        last_lines = []
        line_count = 0

        try:
            for line in proc.stdout:
                line_count += 1
                fout.write(line)
                fout.flush()

                stripped = line.rstrip()
                last_lines.append(stripped)
                if len(last_lines) > 15:
                    last_lines.pop(0)

                now = time.time()
                if verbose:
                    if raw_cp2k_log:
                        log(f"    [cp2k] {stripped}")
                        last_print = now
                    elif interesting.search(line) or scf_step_re.match(line):
                        log(f"    [cp2k] {stripped}")
                        last_print = now
                    elif now - last_print >= 15:
                        elapsed = int(now - start)
                        log(f"    [cp2k] ... выполняется ({elapsed} с, строк в логе: {line_count})")
                        last_print = now

                if timeout is not None and (now - start) > timeout:
                    proc.kill()
                    raise TimeoutError(f"CP2K превысил таймаут {timeout} с")

            proc.wait(timeout=30)
        except Exception:
            if proc.poll() is None:
                proc.kill()
            raise

        total_elapsed = time.time() - start
        log(f"    Процесс CP2K завершён за {total_elapsed:.2f} с (код возврата: {proc.returncode}, строк в логе: {line_count})")

        if proc.returncode != 0:
            ctx = extract_error_context(out_path, last_lines)
            raise RuntimeError(f"CP2K завершился с кодом {proc.returncode}.\n{ctx}")


def main():
    parser = argparse.ArgumentParser(
        description="Расчёт HOMO-LUMO gap через CP2K по входному .xyz файлу."
    )
    parser.add_argument("xyz_file", help="Путь к входному .xyz файлу")
    parser.add_argument(
        "--cp2k-cmd",
        default="cp2k.psmp",
        help="Команда запуска CP2K (по умолчанию: cp2k.psmp)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=15.0,
        help="Размер кубической ячейки в Å (по умолчанию: 15.0)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=400.0,
        help="CUTOFF в Ry (по умолчанию: 400.0)",
    )
    parser.add_argument(
        "--rel-cutoff",
        type=float,
        default=50.0,
        help="REL_CUTOFF в Ry (по умолчанию: 50.0)",
    )
    parser.add_argument(
        "--multiplicity",
        "-m",
        type=int,
        default=0,
        help="Спиновая мультиплетность 2S+1 (0 — автоподбор CP2K, по умолчанию: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Таймаут выполнения CP2K в секундах (по умолчанию: 3600)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Не удалять временную директорию с .inp и .out файлами после расчёта",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Путь к рабочей директории для файлов CP2K (если не задан, создаётся временная)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Скрыть живой поток логов CP2K во время расчёта",
    )
    parser.add_argument(
        "--raw-cp2k-log",
        action="store_true",
        help="Печатать каждую строку вывода CP2K в консоль",
    )
    args = parser.parse_args()

    log("[1/4] Чтение входного XYZ-файла...")
    symbols, coords, comment = read_xyz(args.xyz_file)
    unique_elems = list(dict.fromkeys(symbols))
    project = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(args.xyz_file))[0]) or "system"

    log(f"    Файл:         {args.xyz_file}")
    if comment:
        log(f"    Комментарий:  {comment}")
    log(f"    Атомов:       {len(symbols)} (элементы: {', '.join(unique_elems)})")
    log(f"    Проект:       {project}")

    cleanup_tmp = False
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        workdir = args.workdir
    else:
        workdir = tempfile.mkdtemp(prefix=f"cp2k_gap_{project}_")
        cleanup_tmp = not args.keep_workdir

    inp_path = os.path.join(workdir, f"{project}.inp")
    out_path = os.path.join(workdir, f"{project}.out")

    log("[2/4] Подготовка рабочей директории и генерация входного файла CP2K...")
    log(f"    Рабочая папка: {workdir} (удаление после завершения: {'да' if cleanup_tmp else 'нет'})")
    log(
        f"    Параметры:     cell={args.cell_size} Å, cutoff={args.cutoff} Ry, "
        f"rel_cutoff={args.rel_cutoff} Ry, multiplicity={args.multiplicity}, timeout={args.timeout} с"
    )

    try:
        write_cp2k_input(
            inp_path=inp_path,
            project=project,
            symbols=symbols,
            coords=coords,
            cell_size=args.cell_size,
            cutoff=args.cutoff,
            rel_cutoff=args.rel_cutoff,
            multiplicity=args.multiplicity,
        )
        log(f"    Входной файл записан: {inp_path}")

        log("[3/4] Выполнение расчёта в CP2K...")
        t0 = time.time()
        run_cp2k(
            cp2k_cmd=args.cp2k_cmd,
            inp_path=inp_path,
            out_path=out_path,
            workdir=workdir,
            timeout=args.timeout,
            verbose=not args.quiet,
            raw_cp2k_log=args.raw_cp2k_log,
        )
        elapsed = time.time() - t0

        log("[4/4] Анализ результатов расчёта...")
        converged = check_scf_converged(out_path)
        gap_ev = parse_homo_lumo_gap(out_path)

        log(f"    Лог CP2K:     {out_path}")
        log(f"    Статус SCF:   {'SCF run converged' if converged else 'НЕ СОШЁЛСЯ'}")

        if gap_ev is None:
            log(f"✗ Не удалось найти HOMO-LUMO gap в выводе: {out_path}", stream=sys.stderr)
            sys.exit(2)

        log("\n=== ИТОГ ===")
        log(f"HOMO-LUMO gap: {gap_ev:.6f} eV")
        log(f"SCF сошёлся:   {'да' if converged else 'НЕТ'}")
        log(f"Время расчёта: {elapsed:.2f} с")

        if not converged:
            log("⚠ Внимание: SCF не сошёлся — значение gap может быть некорректным.", stream=sys.stderr)
            sys.exit(3)

        if args.keep_workdir or args.workdir:
            log(f"Файлы CP2K:    {workdir}")
    finally:
        if cleanup_tmp:
            log(f"Очистка временной директории: {workdir}")
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()