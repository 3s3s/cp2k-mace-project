#!/usr/bin/env python3
"""
Загрузчик конфига с подсказками.

Формат config.json:
  {
    "param": {"value": <any>, "help": "<строка>"},
    ...
  }
Допускается и «плоский» вид — тогда help отсутствует:
  {
    "param": <any>
  }

Доступ:
  cfg["param"]            -> значение
  cfg.get("param", def)   -> значение или default
  cfg.help("param")       -> строка подсказки или ""
  cfg.print_help()        -> печать всех параметров с подсказками
  "param" in cfg          -> True/False
  cfg.raw                 -> исходный dict (только значения)
"""

import json


class Config(dict):
    def __init__(self, values: dict, helps: dict):
        super().__init__(values)
        self._helps = helps

    def help(self, key: str) -> str:
        return self._helps.get(key, "")

    def print_help(self):
        print("\n" + "=" * 60)
        print("ПАРАМЕТРЫ КОНФИГА")
        print("=" * 60)
        for key in sorted(self._helps):
            val = self.get(key, "<нет>")
            print(f"  {key} = {val!r}")
            print(f"      {self._helps[key]}")
        print()

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        values = {}
        helps = {}
        for k, v in raw.items():
            if isinstance(v, dict) and "value" in v and "help" in v and len(v) <= 3:
                values[k] = v["value"]
                helps[k] = v["help"]
            else:
                # плоский вид — без help
                values[k] = v
        return cls(values, helps)