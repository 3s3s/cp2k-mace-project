#!/usr/bin/env python3
"""
Модуль визуализации XYZ-структур в интерактивный HTML (Three.js).

Может использоваться как:
  - самостоятельный скрипт: python xyz_to_html.py input.xyz output.html
  - модуль: from xyz_to_html import structure_to_html
"""

import sys
import os
import re
import numpy as np
from collections import Counter

# ---------- КОНСТАНТЫ ----------
ATOM_RADIUS = 1.2
ATOM_COLOR = "#B87333"
BACKGROUND_COLOR = "#1a1a2e"


def read_first_structure(xyz_path):
    with open(xyz_path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
        if not line: raise ValueError("Файл пуст")
        n_atoms = int(line.strip())
        comment = f.readline().rstrip("\n")
        energy_eV = None
        m = re.search(r"energy\s*=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", comment)
        if m: energy_eV = float(m.group(1))
        atoms = []
        for _ in range(n_atoms):
            parts = f.readline().split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    counts = Counter(el for el, _, _, _ in atoms)
    formula = "".join(f"{el}{cnt}" for el, cnt in sorted(counts.items()))
    return atoms, energy_eV, n_atoms, formula


def compute_distances(atoms):
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    n = len(coords)
    if n == 2: return {"single": float(np.linalg.norm(coords[1] - coords[0]))}
    elif n > 2:
        dists = [float(np.linalg.norm(coords[i] - coords[j])) for i in range(n) for j in range(i+1, n)]
        return {"min": min(dists), "max": max(dists), "n_pairs": len(dists)}
    return {}


def generate_html(atoms, energy_eV, n_atoms, formula, distances, output_path):
    atoms_js = ",\n        ".join(
        f'{{ x: {x:.6f}, y: {y:.6f}, z: {z:.6f}, element: "{el}" }}'
        for el, x, y, z in atoms
    )
    info_lines = [f"<b>Кластер:</b> {formula}", f"<b>Число атомов:</b> {n_atoms}"]
    if "single" in distances: info_lines.append(f"<b>Расстояние Cu–Cu:</b> {distances['single']:.4f} Å")
    elif "min" in distances:
        info_lines.append(f"<b>Мин. расстояние:</b> {distances['min']:.4f} Å")
        info_lines.append(f"<b>Макс. расстояние:</b> {distances['max']:.4f} Å")
    if energy_eV is not None: info_lines.append(f"<b>Энергия:</b> {energy_eV:.6f} eV")
    info_html = "<br>".join(info_lines)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>3D Cluster Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: {BACKGROUND_COLOR}; font-family: Arial, sans-serif; }}
        canvas {{ display: block; }}
        #panel {{ position: absolute; top: 10px; right: 10px; width: 280px; color: #ddd; font-size: 13px; background: rgba(0,0,0,0.65); padding: 12px 14px; border-radius: 8px; line-height: 1.5; pointer-events: none; }}
        #panel h3 {{ margin: 0 0 8px 0; color: #fff; font-size: 15px; border-bottom: 1px solid #555; padding-bottom: 6px; }}
        #panel .hint {{ color: #9ab; font-size: 12px; margin-top: 10px; border-top: 1px solid #333; padding-top: 8px; }}
        #buttons {{ position: absolute; bottom: 15px; left: 15px; display: flex; gap: 8px; z-index: 10; }}
        #buttons button {{ padding: 8px 14px; border-radius: 6px; background: rgba(30,40,60,0.9); color: #ddd; font-size: 13px; cursor: pointer; border: 1px solid #445566; }}
        #buttons button:hover {{ background: rgba(60,80,110,0.95); color: #fff; }}
    </style>
</head>
<body>
    <div id="panel">
        <h3>Информация</h3>
        {info_html}
        <div class="hint">
            <b>Управление:</b><br>
            ЛКМ — вращение<br>
            Колесо — зум<br>
            Двойной клик — сброс
        </div>
    </div>
    <div id="buttons">
        <button id="btn-reset">Сброс</button>
    </div>
    <script type="importmap">
    {{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js" }} }}
    </script>
    <script type="module">
        import * as THREE from 'three';

        const atomsData = [{atoms_js}];
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('{BACKGROUND_COLOR}');
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        document.body.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
        dirLight.position.set(5, 10, 7);
        dirLight.castShadow = true;
        scene.add(dirLight);

        let cx = 0, cy = 0, cz = 0;
        atomsData.forEach(a => {{ cx += a.x; cy += a.y; cz += a.z; }});
        cx /= atomsData.length; cy /= atomsData.length; cz /= atomsData.length;

        const moleculeGroup = new THREE.Group();
        moleculeGroup.rotation.order = 'YXZ'; 
        scene.add(moleculeGroup);

        const sphereGeo = new THREE.SphereGeometry({ATOM_RADIUS}, 48, 48);
        const sphereMat = new THREE.MeshStandardMaterial({{ color: '{ATOM_COLOR}', roughness: 0.4, metalness: 0.4 }});
        
        atomsData.forEach(atom => {{
            const sphere = new THREE.Mesh(sphereGeo, sphereMat.clone());
            sphere.position.set(atom.x - cx, atom.y - cy, atom.z - cz);
            sphere.castShadow = true;
            sphere.receiveShadow = true;
            moleculeGroup.add(sphere);
        }});

        const DEFAULT_DISTANCE = 15;
        let zoomDistance = DEFAULT_DISTANCE;
        function updateCamera() {{
            const dir = new THREE.Vector3(0, 0.3, 1).normalize();
            camera.position.copy(dir.multiplyScalar(zoomDistance));
            camera.lookAt(0, 0, 0);
        }}
        updateCamera();

        document.getElementById('btn-reset').onclick = () => {{
            moleculeGroup.rotation.set(0, 0, 0);
            zoomDistance = DEFAULT_DISTANCE;
            updateCamera();
        }};

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
        dom.addEventListener('dblclick', () => {{ 
            moleculeGroup.rotation.set(0, 0, 0); 
            zoomDistance = DEFAULT_DISTANCE; 
            updateCamera(); 
        }});

        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        animate();
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
    # Создаём родительскую директорию, если её нет
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f: f.write(html)
    print(f"HTML сохранён: {output_path}")


# ---------- ПУБЛИЧНАЯ ФУНКЦИЯ ДЛЯ ИМПОРТА ----------
def structure_to_html(xyz_path: str, output_html: str = None) -> str:
    """
    Читает первую структуру из xyz_path и генерирует HTML-файл.
    
    Параметры:
      xyz_path : путь к .xyz-файлу
      output_html : путь к выходному HTML (если None, будет <xyz_path без .xyz>.html)
    
    Возвращает путь к созданному HTML-файлу.
    """
    if output_html is None:
        base, _ = os.path.splitext(xyz_path)
        output_html = f"{base}.html"

    atoms, energy_eV, n_atoms, formula = read_first_structure(xyz_path)
    distances = compute_distances(atoms)
    generate_html(atoms, energy_eV, n_atoms, formula, distances, output_html)
    return output_html


# ---------- ЗАПУСК КАК САМОСТОЯТЕЛЬНОГО СКРИПТА ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python xyz_to_html.py input.xyz [output.html]")
        sys.exit(1)
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    structure_to_html(sys.argv[1], output_path)