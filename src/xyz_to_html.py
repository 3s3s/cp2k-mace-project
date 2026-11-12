#!/usr/bin/env python3
"""
Модуль визуализации XYZ-структур в интерактивный HTML (Three.js).
Создает галерею миниатюр сверху и большое 3D-окно снизу.
"""

import sys
import os
import re
import json
import numpy as np
from collections import Counter

# ---------- КОНСТАНТЫ ----------
ATOM_RADIUS = 1.2
ATOM_COLOR = "#B87333"
BACKGROUND_COLOR = "#1a1a2e"
GALLERY_BG = "#222233"

def read_all_structures(xyz_path):
    """Читает все структуры из extxyz."""
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
            
            comment = f.readline().rstrip("\n")
            
            energy_eV = None
            m = re.search(r"energy\s*=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", comment)
            if m: energy_eV = float(m.group(1))
            
            # Извлекаем n_atoms и dim из comment для инфо-панели
            m_n = re.search(r"n_atoms=(\d+)", comment)
            n_atoms_meta = int(m_n.group(1)) if m_n else n_atoms
            
            m_d = re.search(r"dim=(\d+)", comment)
            dim_meta = int(m_d.group(1)) if m_d else -1

            atoms = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
            
            structures.append({
                "atoms": atoms,
                "energy": energy_eV,
                "n_atoms": n_atoms_meta,
                "dim": dim_meta,
            })
    return structures

def compute_distances(atoms):
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    n = len(coords)
    if n == 2: return {"single": float(np.linalg.norm(coords[1] - coords[0]))}
    elif n > 2:
        dists = [float(np.linalg.norm(coords[i] - coords[j])) for i in range(n) for j in range(i+1, n)]
        return {"min": min(dists), "max": max(dists), "n_pairs": len(dists)}
    return {}

def generate_html(structures, output_path):
    # Подготавливаем данные для JS
    js_structures = []
    for s in structures:
        atoms_list = [{"el": el, "x": x, "y": y, "z": z} for el, x, y, z in s["atoms"]]
        counts = Counter(el for el, _, _, _ in s["atoms"])
        formula = "".join(f"{el}{cnt}" for el, cnt in sorted(counts.items()))
        
        dists = compute_distances(s["atoms"])
        
        info_lines = [
            f"<b>Кластер:</b> {formula}",
            f"<b>Число атомов:</b> {s['n_atoms']}",
            f"<b>Размерность (генерация):</b> {s['dim']}D",
        ]
        if "single" in dists:
            info_lines.append(f"<b>Расстояние Cu–Cu:</b> {dists['single']:.4f} Å")
        elif "min" in dists:
            info_lines.append(f"<b>Мин. расстояние:</b> {dists['min']:.4f} Å")
            info_lines.append(f"<b>Макс. расстояние:</b> {dists['max']:.4f} Å")
            
        if s["energy"] is not None:
            info_lines.append(f"<b>Энергия:</b> {s['energy']:.6f} eV")
            
        js_structures.append({
            "atoms": atoms_list,
            "formula": formula,
            "info_html": "<br>".join(info_lines)
        })

    structures_json = json.dumps(js_structures)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>3D Cluster Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: {BACKGROUND_COLOR}; font-family: Arial, sans-serif; display: flex; flex-direction: column; height: 100vh; }}
        
        /* Верхняя панель с галереей */
        #gallery-container {{
            height: 140px;
            background-color: {GALLERY_BG};
            display: flex;
            align-items: center;
            border-bottom: 2px solid #445566;
            position: relative;
            flex-shrink: 0;
        }}
        #gallery {{
            display: flex;
            overflow-x: auto;
            scroll-behavior: smooth;
            gap: 10px;
            padding: 10px;
            flex-grow: 1;
            /* Скрываем скроллбар для красоты */
            scrollbar-width: none; 
        }}
        #gallery::-webkit-scrollbar {{ display: none; }}
        
        .thumb {{
            width: 100px; height: 100px;
            background: #111;
            border: 2px solid #445566;
            border-radius: 8px;
            cursor: pointer;
            flex-shrink: 0;
            position: relative;
            transition: border-color 0.2s;
            display: flex; justify-content: center; align-items: center;
        }}
        .thumb:hover {{ border-color: #88aaff; }}
        .thumb.active {{ border-color: #ffaa00; background: #222; }}
        .thumb-label {{
            position: absolute; bottom: 4px; left: 0; width: 100%;
            text-align: center; color: #aaa; font-size: 11px; font-weight: bold;
            pointer-events: none;
        }}
        
        /* Кнопки скролла */
        .scroll-btn {{
            width: 40px; height: 100%;
            background: rgba(0,0,0,0.5);
            color: #fff; border: none; font-size: 24px; cursor: pointer;
            display: flex; justify-content: center; align-items: center;
            z-index: 10; flex-shrink: 0;
        }}
        .scroll-btn:hover {{ background: rgba(0,0,0,0.8); }}

        /* Главное окно 3D */
        #main-view {{
            flex-grow: 1;
            position: relative;
        }}
        canvas {{ display: block; }}

        /* Панель информации */
        #panel {{
            position: absolute; top: 20px; right: 20px; width: 280px;
            color: #ddd; font-size: 13px; background: rgba(0,0,0,0.7);
            padding: 15px; border-radius: 8px; line-height: 1.6;
            pointer-events: none; border: 1px solid #445566;
        }}
        #panel h3 {{ margin: 0 0 10px 0; color: #fff; font-size: 16px; border-bottom: 1px solid #555; padding-bottom: 6px; }}
        #panel .hint {{ color: #9ab; font-size: 12px; margin-top: 15px; border-top: 1px solid #333; padding-top: 10px; }}
        
        /* Кнопка сброса */
        #btn-reset {{
            position: absolute; bottom: 20px; left: 20px; z-index: 10;
            padding: 10px 16px; border-radius: 6px; background: rgba(30,40,60,0.9);
            color: #ddd; font-size: 14px; cursor: pointer; border: 1px solid #445566;
        }}
        #btn-reset:hover {{ background: rgba(60,80,110,0.95); color: #fff; }}
    </style>
</head>
<body>
    <!-- Верхняя панель галереи -->
    <div id="gallery-container">
        <button class="scroll-btn" id="scroll-left">&#10094;</button>
        <div id="gallery"></div>
        <button class="scroll-btn" id="scroll-right">&#10095;</button>
    </div>

    <!-- Основное окно -->
    <div id="main-view">
        <div id="panel">
            <h3>Информация</h3>
            <div id="info-content">Выберите структуру</div>
            <div class="hint">
                <b>Управление:</b><br>
                ЛКМ — вращение<br>
                Колесо — зум<br>
                Двойной клик — сброс
            </div>
        </div>
        <button id="btn-reset">Сброс камеры</button>
    </div>

    <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js" }} }}
    </script>
    <script type="module">
        import * as THREE from 'three';

        const structuresData = {structures_json};
        const ATOM_RADIUS = {ATOM_RADIUS};
        const ATOM_COLOR = '{ATOM_COLOR}';
        const BG_COLOR = '{BACKGROUND_COLOR}';

        // ==========================================
        // 1. Настройка сцены (основное окно)
        // ==========================================
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(BG_COLOR);
        
        const container = document.getElementById('main-view');
        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(5, 10, 7);
        dirLight.castShadow = true;
        scene.add(dirLight);

        const moleculeGroup = new THREE.Group();
        moleculeGroup.rotation.order = 'YXZ'; 
        scene.add(moleculeGroup);

        const sphereGeo = new THREE.SphereGeometry(ATOM_RADIUS, 48, 48);
        const sphereMat = new THREE.MeshStandardMaterial({{ color: ATOM_COLOR, roughness: 0.4, metalness: 0.4 }});

        const DEFAULT_DISTANCE = 15;
        let zoomDistance = DEFAULT_DISTANCE;

        function updateCamera() {{
            const dir = new THREE.Vector3(0, 0.3, 1).normalize();
            camera.position.copy(dir.multiplyScalar(zoomDistance));
            camera.lookAt(0, 0, 0);
        }}
        updateCamera();

        // Очистка группы и загрузка новой молекулы
        function loadStructure(index) {{
            // Очищаем группу
            while(moleculeGroup.children.length > 0) {{ 
                moleculeGroup.remove(moleculeGroup.children[0]); 
            }}
            
            const data = structuresData[index];
            const atoms = data.atoms;
            
            // Центр масс
            let cx = 0, cy = 0, cz = 0;
            atoms.forEach(a => {{ cx += a.x; cy += a.y; cz += a.z; }});
            cx /= atoms.length; cy /= atoms.length; cz /= atoms.length;

            atoms.forEach(atom => {{
                const sphere = new THREE.Mesh(sphereGeo, sphereMat.clone());
                sphere.position.set(atom.x - cx, atom.y - cy, atom.z - cz);
                sphere.castShadow = true;
                sphere.receiveShadow = true;
                moleculeGroup.add(sphere);
            }});

            // Обновляем инфо-панель
            document.getElementById('info-content').innerHTML = data.info_html;

            // Сброс камеры
            moleculeGroup.rotation.set(0, 0, 0);
            zoomDistance = DEFAULT_DISTANCE;
            updateCamera();

            // Обновляем активный класс в галерее
            document.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
            document.querySelector(`.thumb[data-index="${{index}}"]`).classList.add('active');
        }}

        // Управление мышью
        const dom = renderer.domElement;
        let isDragging = false, lastX = 0, lastY = 0;
        dom.addEventListener('pointerdown', (e) => {{
            if (e.button === 0) {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; dom.setPointerCapture(e.pointerId); }}
        }});
        dom.addEventListener('pointerup', (e) => {{
            if (e.button === 0) {{ isDragging = false; dom.releasePointerCapture(e.pointerId); }}
        }});
        dom.addEventListener('pointermove', (e) => {{
            if (!isDragging) return;
            const dx = e.clientX - lastX, dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            moleculeGroup.rotation.y += dx * 0.01;
            moleculeGroup.rotation.x += dy * 0.01;
        }});
        dom.addEventListener('wheel', (e) => {{
            e.preventDefault();
            zoomDistance *= (1 + e.deltaY * 0.001);
            zoomDistance = Math.max(5, Math.min(50, zoomDistance));
            updateCamera();
        }}, {{ passive: false }});
        dom.addEventListener('contextmenu', (e) => e.preventDefault());

        const resetAction = () => {{ 
            moleculeGroup.rotation.set(0, 0, 0); 
            zoomDistance = DEFAULT_DISTANCE; 
            updateCamera(); 
        }};
        dom.addEventListener('dblclick', resetAction);
        document.getElementById('btn-reset').onclick = resetAction;

        // Анимация
        function animate() {{ 
            requestAnimationFrame(animate); 
            renderer.render(scene, camera); 
        }}
        animate();

        // Ресайз
        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            const h = container.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }});

        // ==========================================
        // 2. Настройка миниатюр (Галерея)
        // ==========================================
        const galleryDiv = document.getElementById('gallery');
        structuresData.forEach((data, index) => {{
            const thumb = document.createElement('div');
            thumb.className = 'thumb';
            thumb.dataset.index = index;
            
            // Рисуем миниатюру через 2D Canvas (проекция)
            const canvas2d = document.createElement('canvas');
            canvas2d.width = 100;
            canvas2d.height = 100;
            const ctx = canvas2d.getContext('2d');
            
            // Вычисляем 2D координаты
            let cx = 0, cy = 0, cz = 0;
            data.atoms.forEach(a => {{ cx += a.x; cy += a.y; cz += a.z; }});
            cx /= data.atoms.length; cy /= data.atoms.length; cz /= data.atoms.length;
            
            // Находим максимальный разброс для масштабирования
            let maxD = 0;
            data.atoms.forEach(a => {{
                const d = Math.sqrt((a.x-cx)**2 + (a.y-cy)**2 + (a.z-cz)**2);
                if (d > maxD) maxD = d;
            }});
            const scale = 40 / (maxD || 1);

            // Рисуем атомы (сортируем по Z для правильного перекрытия)
            const sortedAtoms = [...data.atoms].sort((a, b) => a.z - b.z);
            sortedAtoms.forEach(a => {{
                const x = 50 + (a.x - cx) * scale;
                const y = 50 - (a.y - cy) * scale; // Инверсия Y для Canvas
                const r = Math.max(3, ATOM_RADIUS * scale * 0.8); // Радиус на миниатюре

                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = ATOM_COLOR;
                ctx.fill();
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 1;
                ctx.stroke();
            }});
            
            thumb.appendChild(canvas2d);
            
            // Подпись
            const label = document.createElement('div');
            label.className = 'thumb-label';
            label.innerText = data.formula;
            thumb.appendChild(label);

            // Обработчик клика
            thumb.onclick = () => loadStructure(index);
            galleryDiv.appendChild(thumb);
        }});

        // Кнопки прокрутки галереи
        document.getElementById('scroll-left').onclick = () => {{
            galleryDiv.scrollLeft -= 200;
        }};
        document.getElementById('scroll-right').onclick = () => {{
            galleryDiv.scrollLeft += 200;
        }};

        // Загружаем первую структуру при старте
        if (structuresData.length > 0) {{
            loadStructure(0);
        }}
    </script>
</body>
</html>
"""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f: 
        f.write(html)
    print(f"HTML сохранён: {output_path}")

# ---------- ПУБЛИЧНАЯ ФУНКЦИЯ ДЛЯ ИМПОРТА ----------
def structure_to_html(xyz_path: str, output_html: str = None) -> str:
    if output_html is None:
        base, _ = os.path.splitext(xyz_path)
        output_html = f"{base}.html"

    structures = read_all_structures(xyz_path)
    if not structures:
        raise ValueError(f"Не найдено структур в {xyz_path}")
        
    generate_html(structures, output_html)
    return output_html

# ---------- ЗАПУСК КАК САМОСТОЯТЕЛЬНОГО СКРИПТА ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python xyz_to_html.py input.xyz [output.html]")
        sys.exit(1)
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    structure_to_html(sys.argv[1], output_path)