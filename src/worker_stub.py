#!/usr/bin/env python3
"""
Скрипт-заглушка воркера. Регистрируется, получает задачу
и отправляет назад фальшивый/неправильный результат для тестирования.
"""

import time
import requests
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Порт сервера main.py")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"

    # Шаг 1: Регистрация воркера
    reg_url = f"{base_url}/register-worker"
    try:
        reg_resp = requests.get(reg_url)
        worker_id = reg_resp.json().get("worker_id")
        print(f"Успешно зарегистрирован! Мой ID: {worker_id}")
    except Exception as e:
        print(f"Ошибка регистрации: {e}")
        return

    # Шаг 2: Запрос задачи
    task_url = f"{base_url}/get-task"
    print(f"[{worker_id}] Запрашиваю структуру...")
    try:
        response = requests.get(task_url, params={"worker_id": worker_id})
        res_data = response.json()
        
        if res_data.get("status") == "empty":
            print(f"[{worker_id}] Очередь пуста.")
            return

        task = res_data.get("task")
        struct_id = task["struct_id"]
        print(f"[{worker_id}] Получена структура: {struct_id}. Имитирую расчет с ошибкой...")

        # Шаг 3: Отправка НЕПРАВИЛЬНОГО результата на сервер
        submit_url = f"{base_url}/submit-result"
        payload = {
            "worker_id": worker_id,
            "struct_id": struct_id,
            "success": False,  # Указываем, что расчет неверный / расхождение
            "data": {"error_reason": "SCF convergence failed / bad forces"}
        }

        submit_resp = requests.post(submit_url, json=payload)
        print(f"[{worker_id}] Ответ сервера на сдачу фальшивого результата:")
        print(submit_resp.json())

    except Exception as e:
        print(f"[{worker_id}] Ошибка: {e}")

if __name__ == "__main__":
    main()