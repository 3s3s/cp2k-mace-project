#!/usr/bin/env python3
"""
Модуль генерации случайных кластеров Cu_n (n = 2..N) в 1D, 2D, 3D.
Расстояние между атомами: [ATOM_DIAMETER, 3*ATOM_DIAMETER].
Записывает структуры в extxyz для последующего расчёта в CP2K.
"""

import os
import random
import numpy as np

# ---------- КОНСТАНТЫ ----------
ATOM_DIAMETER = 2.0          # Å
MIN_DIST = ATOM_DIAMETER
MAX_DIST = 2.0 * ATOM_DIAMETER

# Словарь: {число атомов: число структур}
STRUCTURES_PER_N = {2: 10}   #, 3: 20, 4: 15}

# Размерности для каждого n_atoms
DIMENSIONS_PER_N = {
    2: [3],
    #3: [1, 2, 3],
    #4: [2, 3],
}

# Путь к выходному файлу (относительно корня проекта, где вы запускаете скрипт)
OUTPUT_XYZ = os.path.join(os.path.dirname(__file__), "..", "results", "generated_structures.xyz")
RANDOM_SEED = 42
BOX_SIZE_MARGIN = 4.0


def compute_box_size(n_atoms: int) -> float:
    """Вычисляет размер кубической ячейки."""
    return MAX_DIST * np.sqrt(n_atoms) * BOX_SIZE_MARGIN


def random_point_1d(box_size):
    return np.array([random.uniform(0, box_size), 0.0, 0.0])

def random_point_2d(box_size):
    return np.array([random.uniform(0, box_size), random.uniform(0, box_size), 0.0])

def random_point_3d(box_size):
    return np.array([random.uniform(0, box_size),
                     random.uniform(0, box_size),
                     random.uniform(0, box_size)])


def check_distances(points):
    """Проверяет, что все попарные расстояния в [MIN_DIST, MAX_DIST]."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < MIN_DIST or d > MAX_DIST:
                return False
    return True


def generate_structure(n_atoms: int, dim: int):
    """Генерирует одну структуру с заданным числом атомов и размерностью."""
    if dim == 1:
        point_fn = random_point_1d
    elif dim == 2:
        point_fn = random_point_2d
    elif dim == 3:
        point_fn = random_point_3d
    else:
        raise ValueError("dim должен быть 1, 2 или 3")

    box_size = compute_box_size(n_atoms)

    for attempt in range(10000):
        points = np.array([point_fn(box_size) for _ in range(n_atoms)])
        if check_distances(points):
            center_of_mass = points.mean(axis=0)
            points = points - center_of_mass + box_size / 2.0
            return points, box_size
    # ИСПРАВЛЕНО: было raise RuntimeError(...) без сообщения
    raise RuntimeError(
        f"Не удалось сгенерировать структуру для n_atoms={n_atoms}, dim={dim} "
        f"за 10000 попыток. Попробуйте увеличить BOX_SIZE_MARGIN."
    )


def write_xyz(structures, n_atoms, dim, box_size, path):
    """Записывает список структур в extxyz (append)."""
    # ИСПРАВЛЕНО: создаём родительскую директорию, если её нет
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "a") as f:
        for points in structures:
            n = len(points)
            comment = (
                f'Lattice="{box_size:.4f} 0.0 0.0 0.0 {box_size:.4f} 0.0 0.0 0.0 {box_size:.4f}" '
                f'Properties=species:S:1:pos:R:3 '
                f'n_atoms={n_atoms} dim={dim} '
                f'pbc="F F F"'
            )
            f.write(f"{n}\n{comment}\n")
            for p in points:
                f.write(f"Cu {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def main():
    random.seed(RANDOM_SEED)

    print("Параметры генерации:")
    print(f"  ATOM_DIAMETER = {ATOM_DIAMETER} Å")
    print(f"  MIN_DIST = {MIN_DIST} Å, MAX_DIST = {MAX_DIST} Å")
    print(f"  STRUCTURES_PER_N = {STRUCTURES_PER_N}")
    print(f"  DIMENSIONS_PER_N = {DIMENSIONS_PER_N}")
    print(f"  BOX_SIZE_MARGIN = {BOX_SIZE_MARGIN}")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print()

    # Удаляем старый файл, чтобы начать заново
    if os.path.exists(OUTPUT_XYZ):
        os.remove(OUTPUT_XYZ)

    total_written = 0

    for n_atoms, n_structs in STRUCTURES_PER_N.items():
        dims = DIMENSIONS_PER_N.get(n_atoms, [3])
        box_size = compute_box_size(n_atoms)
        print(f"n_atoms = {n_atoms}, BOX_SIZE = {box_size:.2f} Å")
        for dim in dims:
            print(f"  Генерация {n_structs} структур для dim={dim}...")
            structures = []
            for i in range(n_structs):
                try:
                    s, _ = generate_structure(n_atoms, dim)
                    structures.append(s)
                except RuntimeError as e:
                    print(f"    ⚠ {e}")
                    continue
            write_xyz(structures, n_atoms, dim, box_size, OUTPUT_XYZ)
            total_written += len(structures)
            print(f"    ✓ Записано {len(structures)} структур")

    print(f"\nГотово. Всего записано {total_written} структур в {OUTPUT_XYZ}")


if __name__ == "__main__":
    main()