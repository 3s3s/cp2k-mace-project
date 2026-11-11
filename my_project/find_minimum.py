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
from mace.calculators import MACECalculator


def run_find_minimum(
    model_path: str,
    xyz_path: str,
    output_xyz: str,
    logfile: str,
    fmax: float = 0.01,
    steps: int = 100,
    device: str = "cpu",
    default_dtype: str = "float32",
):
    """
    Загружает MACE-модель, читает ВСЕ структуры из xyz_path,
    оптимизирует каждую BFGS, выбирает структуру с минимальной энергией.

    Сохраняет:
      - output_xyz: лучшая (глобальный минимум в выборке) структура
      - logfile: лог BFGS для лучшей структуры
      - <output_xyz без .xyz>_all_optimized.xyz: все оптимизированные структуры

    Возвращает словарь с результатами для лучшей структуры.
    """
    calc = MACECalculator(
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
    )

    # Читаем все структуры
    all_atoms = read(xyz_path, index=":")
    n_structs = len(all_atoms)
    print(f"Загружено {n_structs} структур из {xyz_path}")

    best_energy = None
    best_atoms = None
    best_idx = None
    best_nsteps = None

    # Файл для всех оптимизированных структур
    base, ext = os.path.splitext(output_xyz)
    all_opt_path = f"{base}_all_optimized{ext}"

    for i, atoms in enumerate(all_atoms):
        atoms.calc = calc
        n_atoms = len(atoms)

        # Энергия и максимальное расстояние для информации (для Cu2 — расстояние)
        initial_energy = atoms.get_potential_energy()
        if n_atoms == 2:
            info = f"r = {np.linalg.norm(atoms.positions[1] - atoms.positions[0]):.4f} Å"
        else:
            info = f"n_atoms = {n_atoms}"

        # Лог BFGS для каждой структуры — отдельный файл, чтобы не перезаписывать
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

        # Сохраняем оптимизированную структуру в общий файл
        write(all_opt_path, atoms, append=True)

        # Обновляем лучшую
        if best_energy is None or final_energy < best_energy:
            best_energy = final_energy
            best_atoms = atoms.copy()
            best_atoms.calc = calc      # <-- КРИТИЧЕСКИ ВАЖНО
            best_idx = i
            best_nsteps = opt.nsteps
            
    if best_atoms is None:
        raise RuntimeError("Не удалось оптимизировать ни одну структуру")

    # Сохраняем лучшую структуру и её лог
    write(output_xyz, best_atoms)
    # Перезаписываем logfile логом лучшей структуры
    best_log_src = f"{os.path.splitext(logfile)[0]}_struct{best_idx:03d}.log"
    if os.path.exists(best_log_src):
        os.replace(best_log_src, logfile)

    # Итоговая информация
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

    return {
        "best_index": best_idx,
        "n_structures": n_structs,
        "final_energy": best_energy,
        "final_distance": final_distance,
        "fmax": final_fmax,
        "nsteps": best_nsteps,
    }


# ---------- ЗАПУСК КАК САМОСТОЯТЕЛЬНОГО СКРИПТА ----------
if __name__ == "__main__":
    model_path = os.environ.get("MACE_MODEL_PATH")
    xyz_path = os.environ.get("MACE_TRAIN_XYZ")
    output_xyz = os.environ.get("MACE_OUTPUT_XYZ")
    logfile = os.environ.get("MACE_OPT_LOG")

    if not all([model_path, xyz_path, output_xyz, logfile]):
        raise SystemExit(
            "Для ручного запуска задайте переменные окружения: "
            "MACE_MODEL_PATH, MACE_TRAIN_XYZ, MACE_OUTPUT_XYZ, MACE_OPT_LOG"
        )

    result = run_find_minimum(
        model_path=model_path,
        xyz_path=xyz_path,
        output_xyz=output_xyz,
        logfile=logfile,
    )
    print(f"\nИтог: E = {result['final_energy']:.6f} eV, "
          f"лучшая структура №{result['best_index']+1} из {result['n_structures']}")