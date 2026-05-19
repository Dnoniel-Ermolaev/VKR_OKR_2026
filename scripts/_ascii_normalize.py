"""Одноразовый скрипт: заменить "необычные" символы (em-dash, кавычки-ёлочки,
стрелочки и т.п.) на ASCII-эквиваленты во всём дереве исходников.

Запуск:
    python scripts/_ascii_normalize.py

Используется только разработчиком, в продакшен-пайплайне не нужен.
"""
from __future__ import annotations

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TARGET_DIRS = ("src", "scripts", "tests")
EXTS = (".py", ".js", ".html", ".css", ".md")

# Таблица замен. Ключи - все non-ASCII non-Cyrillic символы, что встречаются
# в репозитории; значения - ASCII-аналоги.
REPL = {
    "\u2014": "-",        # em dash
    "\u2013": "-",        # en dash
    # Используем апостроф, чтобы не ломать строковые литералы в Python/JS,
    # где такие "кавычки-ёлочки" встречаются внутри двойных кавычек.
    "\u00AB": "'",        # left guillemet
    "\u00BB": "'",        # right guillemet
    "\u2265": ">=",       # greater-or-equal
    "\u2264": "<=",       # less-or-equal
    "\u2191": "^",        # up arrow
    "\u2193": "v",        # down arrow
    "\u2192": "->",       # right arrow
    "\u2026": "...",      # ellipsis
    "\u00B0": "",         # degree sign
    "\u00B7": "-",        # middle dot
    "\u03B2": "бета",     # Greek beta -> "бета"
    "\u2082": "2",        # subscript 2 -> 2
    "\U0001F5D1": "X",    # X trash bin
    "\U0001F4C5": "",     #  calendar
    "\u26A0": "!",        # warning sign
    "\u2715": "X",        # multiplication X (close button)
    "\uFE0F": "",         # variation selector (invisible)
}


def main() -> int:
    changed: list[tuple[str, str]] = []
    for d in TARGET_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for f in files:
                if not f.endswith(EXTS):
                    continue
                p = os.path.join(root, f)
                with open(p, "r", encoding="utf-8") as fh:
                    txt = fh.read()
                new = txt
                counts: dict[str, int] = {}
                for src_ch, dst_ch in REPL.items():
                    c = new.count(src_ch)
                    if c:
                        counts[src_ch] = c
                        new = new.replace(src_ch, dst_ch)
                if new != txt:
                    with open(p, "w", encoding="utf-8", newline="") as fh:
                        fh.write(new)
                    rel = os.path.relpath(p, ROOT)
                    summary = " ".join(
                        f"U+{ord(k):04X}x{v}" for k, v in counts.items()
                    )
                    changed.append((rel, summary))

    for rel, summary in changed:
        print(f"{rel}: {summary}")
    print(f"TOTAL FILES CHANGED: {len(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
