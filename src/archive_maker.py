#!/usr/bin/env python3
"""
Модуль упаковки артефактов проекта в ZIP.

Публичная функция:
  make_artifacts_archive(cfg) -> str | None

Что пакуется:
  - mace_output/  целиком (чекпоинты, модели, логи) под префиксом "mace_output/";
  - из results/   только файлы координат/сил (*.xyz) под префиксом "results/".

Что НЕ пакуется из results/:
  - промежуточные CP2K: *.inp, *.out, *.forces;
  - служебные: failed_ids.txt, *_isolated_atom_energy.txt;
  - лог оптимизации: opt.log;
  - служебные веб-панели: main.log, main.pid, main.status.json;
  - сам архив и его .tmp;
  - любые прочие файлы, не оканчивающиеся на ".xyz".
"""

import os
import zipfile


# Расширения файлов из results/, которые попадают в архив.
# Сейчас это только .xyz (координаты + силы + энергии).
RESULTS_ALLOWED_EXTS = {".xyz", ".inp", ".forces", ".out", ".txt"}


def _iter_files(root_dir: str):
    """Генератор (full_path, rel_path) для всех файлов под root_dir."""
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
            yield full, rel


def make_artifacts_archive(cfg: dict):
    """
    Создаёт ZIP с mace_output/ и xyz-файлами из results/.

    :param cfg: словарь конфигурации; используются ключи:
        - "output_dir"   : каталог results/
        - "mace_base_dir": каталог mace_output/
        - "archive_file" : путь к итоговому ZIP
    :return: путь к созданному архиву или None при ошибке.
    """
    archive_file = cfg.get("archive_file")
    if not archive_file:
        print("[archive] cfg['archive_file'] не задан — пропускаю упаковку.")
        return None

    output_dir = cfg.get("output_dir")
    mace_base_dir = cfg.get("mace_base_dir")

    archive_abs = os.path.abspath(archive_file)
    archive_dir = os.path.dirname(archive_abs)
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)

    print(f"\n[archive] Упаковка артефактов в {archive_abs} ...")

    tmp_path = archive_abs + ".tmp"
    added_results = []
    added_mace = 0

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # --- results/: только .xyz ---
            if output_dir and os.path.isdir(output_dir):
                for full, rel in _iter_files(output_dir):
                    ext = os.path.splitext(full)[1].lower()
                    if ext not in RESULTS_ALLOWED_EXTS:
                        continue
                    # на всякий случай, если .xyz окажется под другим именем
                    if os.path.abspath(full) == archive_abs:
                        continue
                        
                    element = cfg.get("element", "unknown")
                    arcname = os.path.join("results", element, rel)
                    zf.write(full, arcname)
                    added_results.append(arcname)

            # --- mace_output/: целиком ---
            if mace_base_dir and os.path.isdir(mace_base_dir):
                for full, rel in _iter_files(mace_base_dir):
                    arcname = os.path.join("mace_output", rel)
                    zf.write(full, arcname)
                    added_mace += 1

        os.replace(tmp_path, archive_abs)
        size_mb = os.path.getsize(archive_abs) / (1024 * 1024)

        print(f"[archive]   results/:  {len(added_results)} файл(ов)")
        for name in added_results:
            print(f"[archive]     + {name}")
        print(f"[archive]   mace_output/: {added_mace} файл(ов)")

        print(f"[archive] ✓ Готово: {archive_abs} ({size_mb:.2f} MB)")
        return archive_abs
    except Exception as e:
        print(f"[archive] ✗ Не удалось создать архив: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None