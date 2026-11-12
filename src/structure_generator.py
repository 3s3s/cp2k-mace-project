#!/usr/bin/env python3
"""
Модуль генерации случайных кластеров Cu_n (n = 2..N) в 1D, 2D, 3D.
Для каждой размерности генерируется STRUCTURES_PER_N[n_atoms] структур.
struct_id = f"{n_atoms}_{dim}_{i}_seed{RANDOM_SEED}_d{ATOM_DIAMETER}"
"""

import os
import random
import numpy as np

# ---------- КОНСТАНТЫ ----------
ATOM_DIAMETER = 2.0
MIN_DIST = ATOM_DIAMETER
MAX_DIST = 2.0 * ATOM_DIAMETER

STRUCTURES_PER_N = {2: 3, 3: 10}

DIMENSIONS_PER_N = {
    2: [3],
    3: [2],
}

OUTPUT_XYZ = os.path.join(os.path.dirname(__file__), "..", "results", "generated_structures.xyz")
RANDOM_SEED = 42
BOX_SIZE_MARGIN = 4.0


def compute_box_size(n_atoms: int) -> float:
    return MAX_DIST * np.sqrt(n_atoms) * BOX_SIZE_MARGIN


def random_point(dim, box_size):
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


def generate_structure_sequential(n_atoms: int, dim: int, max_attempts_per_atom: int = 10000):
    box_size = compute_box_size(n_atoms)
    points = [random_point(dim, box_size)]

    for atom_idx in range(1, n_atoms):
        placed = False
        for attempt in range(max_attempts_per_atom):
            candidate = random_point(dim, box_size)
            ok = True
            has_close_neighbor = False
            for p in points:
                d = np.linalg.norm(candidate - p)
                if d < MIN_DIST:
                    ok = False
                    break
                if d <= MAX_DIST:
                    has_close_neighbor = True
            if ok and has_close_neighbor:
                points.append(candidate)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"Не удалось разместить атом {atom_idx + 1} из {n_atoms} "
                f"для dim={dim} за {max_attempts_per_atom} попыток. "
                f"BOX_SIZE={box_size:.2f} Å, MIN_DIST={MIN_DIST}, MAX_DIST={MAX_DIST}."
            )

    points = np.array(points)
    center_of_mass = points.mean(axis=0)
    points = points - center_of_mass + box_size / 2.0
    return points, box_size


def write_xyz(structures_with_id, n_atoms, dim, box_size, path):
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
                f.write(f"Cu {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def main():
    random.seed(RANDOM_SEED)

    print("Параметры генерации:")
    print(f"  ATOM_DIAMETER = {ATOM_DIAMETER} Å")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  STRUCTURES_PER_N = {STRUCTURES_PER_N}")
    print(f"  DIMENSIONS_PER_N = {DIMENSIONS_PER_N}")
    print()

    if os.path.exists(OUTPUT_XYZ):
        os.remove(OUTPUT_XYZ)

    total_written = 0

    for n_atoms, n_structs in STRUCTURES_PER_N.items():
        dims = DIMENSIONS_PER_N.get(n_atoms, [3])
        box_size = compute_box_size(n_atoms)
        for dim in dims:
            print(f"Генерация {n_structs} структур для n={n_atoms}, dim={dim}...")
            structures_with_id = []
            for i in range(n_structs):
                struct_id = f"{n_atoms}_{dim}_{i}_seed{RANDOM_SEED}_d{ATOM_DIAMETER}"
                try:
                    s, _ = generate_structure_sequential(n_atoms, dim)
                    structures_with_id.append((struct_id, s))
                except RuntimeError as e:
                    print(f"  ⚠ {e}")
                    continue
            write_xyz(structures_with_id, n_atoms, dim, box_size, OUTPUT_XYZ)
            total_written += len(structures_with_id)
            print(f"  ✓ Записано {len(structures_with_id)} структур")

    print(f"\nГотово. Всего записано {total_written} структур в {OUTPUT_XYZ}")


if __name__ == "__main__":
    main()