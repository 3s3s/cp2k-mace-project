#!/usr/bin/env python3
"""
Модуль генерации случайных кластеров Cu_n (n = 2..N) в 1D, 2D, 3D.
Публичная функция: generate_structures(config) -> str (путь к XYZ).
Все константы берутся из config (dict), который передаёт главный модуль.
"""

import os
import random
import numpy as np


def _compute_box_size(n_atoms: int, cfg: dict) -> float:
    return cfg["max_dist"] * np.sqrt(n_atoms) * cfg["box_size_margin"]


def _random_point(dim, box_size):
    if dim == 1:
        return np.array([random.uniform(0, box_size), 0.0, 0.0])
    elif dim == 2:
        return np.array([random.uniform(0, box_size), random.uniform(0, box_size), 0.0])
    elif dim == 3:
        return np.array([random.uniform(0, box_size),
                         random.uniform(0, box_size),
                         random.uniform(0, box_size)])
    else:
        raise ValueError("dim должен быть 1, 2 или 3")


def _generate_structure_sequential(n_atoms: int, dim: int, cfg: dict):
    box_size = _compute_box_size(n_atoms, cfg)
    min_dist = cfg["min_dist"]
    max_dist = cfg["max_dist"]
    max_attempts = cfg["max_attempts_per_atom"]

    points = [_random_point(dim, box_size)]

    for atom_idx in range(1, n_atoms):
        placed = False
        for _ in range(max_attempts):
            candidate = _random_point(dim, box_size)
            ok = True
            has_close_neighbor = False
            for p in points:
                d = np.linalg.norm(candidate - p)
                if d < min_dist:
                    ok = False
                    break
                if d <= max_dist:
                    has_close_neighbor = True
            if ok and has_close_neighbor:
                points.append(candidate)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Не удалось разместить атом {atom_idx + 1} из {n_atoms} "
                f"для dim={dim} за {max_attempts} попыток. "
                f"BOX_SIZE={box_size:.2f} Å, MIN_DIST={min_dist}, MAX_DIST={max_dist}."
            )

    points = np.array(points)
    points = points - points.mean(axis=0) + box_size / 2.0
    return points, box_size


def _write_xyz(structures_with_id, n_atoms, dim, box_size, element, path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "a") as f:
        for struct_id, points in structures_with_id:
            n = len(points)
            comment = (
                f'Lattice="{box_size:.4f} 0.0 0.0 0.0 {box_size:.4f} 0.0 0.0 0.0 {box_size:.4f}" '
                f'Properties=species:S:1:pos:R:3 '
                f'n_atoms={n_atoms} dim={dim} struct_id={struct_id} '
                f'pbc="F F F"'
            )
            f.write(f"{n}\n{comment}\n")
            for p in points:
                f.write(f"{element} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def generate_structures(config: dict) -> str:
    """
    Публичная точка входа. Генерирует структуры и пишет их в XYZ.

    :param config: словарь с параметрами (см. config.json).
    :return: путь к записанному XYZ-файлу.
    """
    random_seed = config["random_seed"]
    atom_diameter = config["atom_diameter"]
    structures_per_n = {int(k): v for k, v in config["structures_per_n"].items()}
    dimensions_per_n = {int(k): v for k, v in config["dimensions_per_n"].items()}

    output_xyz = config["output_xyz"]
    random.seed(random_seed)

    print("Параметры генерации:")
    print(f"  ATOM_DIAMETER = {atom_diameter} Å")
    print(f"  RANDOM_SEED = {random_seed}")
    print(f"  STRUCTURES_PER_N = {structures_per_n}")
    print(f"  DIMENSIONS_PER_N = {dimensions_per_n}")
    print()

    if os.path.exists(output_xyz):
        os.remove(output_xyz)

    total_written = 0

    for n_atoms, n_structs in structures_per_n.items():
        dims = dimensions_per_n.get(n_atoms, [3])
        box_size = _compute_box_size(n_atoms, config)
        for dim in dims:
            print(f"Генерация {n_structs} структур для n={n_atoms}, dim={dim}...")
            structures_with_id = []
            for i in range(n_structs):
                struct_id = f"{n_atoms}_{dim}_{i}_seed{random_seed}_d{atom_diameter}"
                try:
                    s, _ = _generate_structure_sequential(n_atoms, dim, config)
                    structures_with_id.append((struct_id, s))
                except RuntimeError as e:
                    print(f"  ⚠ {e}")
                    continue
            _write_xyz(structures_with_id, n_atoms, dim, box_size, config["element"], output_xyz)
            total_written += len(structures_with_id)
            print(f"  ✓ Записано {len(structures_with_id)} структур")

    print(f"\nГотово. Всего записано {total_written} структур в {output_xyz}")
    return output_xyz