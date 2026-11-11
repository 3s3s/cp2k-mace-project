#!/usr/bin/env python3
"""
Модуль поиска глобального минимума энергии кластера Cu_n с помощью MACE.
Проходит по всем структурам в xyz_path, оптимизирует каждую BFGS,
выбирает структуру с наименьшей финальной энергией.

Может использоваться как:
  - самостоятельный скрипт: python find_minimum.py
  - модуль: from find_minimum import run_find_minimum
"""

import os
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

from xyz_to_html import structure_to_html


def run_find_minimum(
    calc,                 # <-- ИСПРАВЛЕНО: принимаем готовый калькулятор
    xyz_path: str,
    output_xyz: str,
    logfile: str,
    fmax: float = 0.01,
    steps: int = 100,
    make_html: bool = True,
):
    """
    Оптимизирует каждую структуру из xyz_path с помощью переданного калькулятора,
    выбирает структуру с минимальной энергией.

    Калькулятор (MACECalculator) создаётся в главном модуле (main.py),
    где задаётся DEVICE = "cpu"/"gpu".
    """
    all_atoms = read(xyz_path, index=":")
    n_structs = len(all_atoms)
    print(f"Загружено {n_structs} структур из {xyz_path}")

    best_energy = None
    best_atoms = None
    best_idx = None
    best_nsteps = None

    base, ext = os.path.splitext(output_xyz)
    all_opt_path = f"{base}_all_optimized{ext}"

    for i, atoms in enumerate(all_atoms):
        atoms.calc = calc
        n_atoms = len(atoms)

        initial_energy = atoms.get_potential_energy()
        if n_atoms == 2:
            info = f"r = {np.linalg.norm(atoms.positions[1] - atoms.positions[0]):.4f} Å"
        else:
            info = f"n_atoms = {n_atoms}"

        log_i = f"{os.path.splitext(logfile)[0]}_struct{i:03d}.log"
        opt = BFGS(atoms, logfile=log_i)
        opt.run(fmax=fmax, steps=steps)

        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        final_fmax = np.linalg.norm(forces, axis=1).max()

        print(
            f"[{i+1}/{n_structs}] {info}: "
            f"E_initial = {initial_energy:.6f} eV → "
            f"E_final = {final_energy:.6f} eV, "
            f"fmax = {final_fmax:.4f} eV/Å, "
            f"steps = {opt.nsteps}"
        )

        write(all_opt_path, atoms, append=True)

        if best_energy is None or final_energy < best_energy:
            best_energy = final_energy
            best_atoms = atoms.copy()
            best_atoms.calc = calc
            best_idx = i
            best_nsteps = opt.nsteps
            
    if best_atoms is None:
        raise RuntimeError("Не удалось оптимизировать ни одну структуру")

    write(output_xyz, best_atoms)
    best_log_src = f"{os.path.splitext(logfile)[0]}_struct{best_idx:03d}.log"
    if os.path.exists(best_log_src):
        os.replace(best_log_src, logfile)

    n_atoms = len(best_atoms)
    if n_atoms == 2:
        final_distance = np.linalg.norm(best_atoms.positions[1] - best_atoms.positions[0])
    else:
        final_distance = float("nan")

    forces = best_atoms.get_forces()
    final_fmax = np.linalg.norm(forces, axis=1).max()

    print("\n" + "=" * 50)
    print("ГЛОБАЛЬНЫЙ МИНИМУМ (в пределах выборки)")
    print("=" * 50)
    print(f"Лучшая структура: №{best_idx+1} из {n_structs}")
    print(f"Финальная энергия: {best_energy:.6f} eV")
    if n_atoms == 2:
        print(f"Финальное расстояние Cu-Cu: {final_distance:.4f} Å")
    print(f"Максимальная сила: {final_fmax:.6f} eV/Å")
    print(f"Число шагов BFGS (лучшая): {best_nsteps}")
    print(f"\nЛучшая структура сохранена в {output_xyz}")
    print(f"Все оптимизированные структуры: {all_opt_path}")
    print(f"Лог BFGS лучшей структуры: {logfile}")

    html_path = None
    if make_html:
        # find_minimum.py лежит в cp2k-mace-project/src,
        # значит static_pages = <script_dir>/webserver/static_pages
        script_dir = os.path.dirname(os.path.abspath(__file__))
        static_pages_dir = os.path.join(script_dir, "webserver", "static_pages")
        os.makedirs(static_pages_dir, exist_ok=True)

        html_name = os.path.splitext(os.path.basename(output_xyz))[0] + ".html"
        html_path = os.path.join(static_pages_dir, html_name)

        try:
            structure_to_html(output_xyz, html_path)
        except Exception as e:
            print(f"⚠ Не удалось создать HTML-визуализацию: {e}")
            html_path = None

    return {
        "best_index": best_idx,
        "n_structures": n_structs,
        "final_energy": best_energy,
        "final_distance": final_distance,
        "fmax": final_fmax,
        "nsteps": best_nsteps,
        "html_path": html_path,
    }


# ---------- ЗАПУСК КАК САМОСТОЯТЕЛЬНОГО СКРИПТА ----------
if __name__ == "__main__":
    from mace.calculators import MACECalculator

    model_path = os.environ.get("MACE_MODEL_PATH")
    xyz_path = os.environ.get("MACE_TRAIN_XYZ")
    output_xyz = os.environ.get("MACE_OUTPUT_XYZ")
    logfile = os.environ.get("MACE_OPT_LOG")
    device = os.environ.get("MACE_DEVICE", "cpu")  # только для ручного запуска

    if not all([model_path, xyz_path, output_xyz, logfile]):
        raise SystemExit(
            "Для ручного запуска задайте переменные окружения: "
            "MACE_MODEL_PATH, MACE_TRAIN_XYZ, MACE_OUTPUT_XYZ, MACE_OPT_LOG"
        )

    calc = MACECalculator(
        model_path=model_path,
        device=device,
        default_dtype="float32",
    )

    result = run_find_minimum(
        calc=calc,
        xyz_path=xyz_path,
        output_xyz=output_xyz,
        logfile=logfile,
    )
    print(f"\nИтог: E = {result['final_energy']:.6f} eV, "
          f"лучшая структура №{result['best_index']+1} из {result['n_structures']}")
    if result.get("html_path"):
        print(f"HTML-визуализация: {result['html_path']}")