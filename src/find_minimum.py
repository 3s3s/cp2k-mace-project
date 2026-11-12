#!/usr/bin/env python3
"""
Модуль поиска глобального минимума для каждого размера кластера.
Проходит по всем структурам в xyz_path, группирует по числу атомов,
оптимизирует каждую BFGS, выбирает структуру с минимальной энергией
для каждого n_atoms.

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
    calc: MACECalculator,
    xyz_path: str,
    output_xyz: str,
    logfile: str,
    fmax: float = 0.01,
    steps: int = 100,
):
    """
    Оптимизирует все структуры из xyz_path, группирует по числу атомов,
    для каждой группы выбирает структуру с минимальной энергией.

    Сохраняет:
      - <output_xyz без .xyz>_n<N>.xyz для каждой группы N атомов
      - logfile (лог BFGS для лучшей структуры каждого размера)
      - <output_xyz без .xyz>_all_optimized.xyz (все оптимизированные)

    Возвращает словарь {n_atoms: {energy, distance, fmax, nsteps, best_index}}.
    """
    all_atoms = read(xyz_path, index=":")
    n_structs = len(all_atoms)
    print(f"Загружено {n_structs} структур из {xyz_path}")

    # Группируем индексы по числу атомов
    groups = {}
    for i, atoms in enumerate(all_atoms):
        n = len(atoms)
        groups.setdefault(n, []).append(i)

    print(f"Найдено групп по числу атомов: {sorted(groups.keys())}")

    base, ext = os.path.splitext(output_xyz)
    all_opt_path = f"{base}_all_optimized{ext}"

    results = {}

    for n_atoms, indices in sorted(groups.items()):
        print("\n" + "=" * 50)
        print(f"ОБРАБОТКА КЛАСТЕРОВ С n_atoms = {n_atoms} ({len(indices)} структур)")
        print("=" * 50)

        best_energy = None
        best_atoms = None
        best_idx = None
        best_nsteps = None
        best_local_idx = None

        for local_i, global_i in enumerate(indices):
            atoms = all_atoms[global_i].copy()
            atoms.calc = calc

            initial_energy = atoms.get_potential_energy()

            log_i = f"{os.path.splitext(logfile)[0]}_n{n_atoms}_struct{local_i:03d}.log"
            opt = BFGS(atoms, logfile=log_i)
            opt.run(fmax=fmax, steps=steps)

            final_energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            final_fmax = np.linalg.norm(forces, axis=1).max()

            print(
                f"  [n={n_atoms}, {local_i+1}/{len(indices)}] "
                f"E_initial = {initial_energy:.6f} → E_final = {final_energy:.6f} eV, "
                f"fmax = {final_fmax:.4f} eV/Å, steps = {opt.nsteps}"
            )

            write(all_opt_path, atoms, append=True)

            if best_energy is None or final_energy < best_energy:
                best_energy = final_energy
                best_atoms = atoms.copy()
                best_atoms.calc = calc
                best_idx = global_i
                best_local_idx = local_i
                best_nsteps = opt.nsteps

        # Сохраняем лучшую структуру для этого n_atoms
        out_n = f"{base}_n{n_atoms}{ext}"
        write(out_n, best_atoms)

        # Переименовываем лог лучшей структуры для этого n_atoms
        best_log_src = f"{os.path.splitext(logfile)[0]}_n{n_atoms}_struct{best_local_idx:03d}.log"
        best_log_dst = f"{os.path.splitext(logfile)[0]}_n{n_atoms}.log"
        if os.path.exists(best_log_src):
            os.replace(best_log_src, best_log_dst)

        # Считаем расстояние только для 2 атомов
        if n_atoms == 2:
            d = float(np.linalg.norm(best_atoms.positions[1] - best_atoms.positions[0]))
        else:
            d = float("nan")

        results[n_atoms] = {
            "energy": best_energy,
            "distance": d,
            "fmax": float(np.linalg.norm(best_atoms.get_forces(), axis=1).max()),
            "nsteps": best_nsteps,
            "best_global_index": best_idx,
            "best_local_index": best_local_idx,
            "n_structures_in_group": len(indices),
            "output_file": out_n,
        }

        print(f"\n  ГЛОБАЛЬНЫЙ МИНИМУМ для n={n_atoms}:")
        print(f"    Лучшая структура: {best_local_idx+1} из {len(indices)}")
        print(f"    Энергия: {best_energy:.6f} eV")
        if n_atoms == 2:
            print(f"    Расстояние Cu–Cu: {d:.4f} Å")
        print(f"    Макс. сила: {results[n_atoms]['fmax']:.6f} eV/Å")
        print(f"    Сохранено в: {out_n}")

    print("\n" + "=" * 50)
    print("ИТОГ: ГЛОБАЛЬНЫЕ МИНИМУМЫ ПО РАЗМЕРАМ")
    print("=" * 50)
    for n_atoms in sorted(results.keys()):
        r = results[n_atoms]
        print(f"  n={n_atoms}: E = {r['energy']:.6f} eV, "
              f"fmax = {r['fmax']:.4f} eV/Å, "
              f"лучшая из {r['n_structures_in_group']} → {r['output_file']}")

    print(f"\nВсе оптимизированные структуры: {all_opt_path}")
    print(f"Логи BFGS: {os.path.splitext(logfile)[0]}_n<N>.log")

    return results


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

    calc = MACECalculator(
        model_path=model_path,
        device="cpu",
        default_dtype="float32",
    )
    run_find_minimum(calc=calc, xyz_path=xyz_path, output_xyz=output_xyz, logfile=logfile)