from __future__ import annotations

from pathlib import Path


def ensure_guidelines_seeded(guidelines_dir: Path) -> Path:
    guidelines_dir.mkdir(parents=True, exist_ok=True)
    seed_file = guidelines_dir / "acs_quick_guide.txt"
    if not seed_file.exists():
        seed_file.write_text(
            (
                "Краткая памятка по ОКС (демо):\n"
                "- Подъем ST при ишемических симптомах требует срочной маршрутизации как высокий риск.\n"
                "- Повышение тропонина увеличивает вероятность повреждения миокарда.\n"
                "- Используйте клиническое мышление; прототип не является медицинским изделием.\n"
            ),
            encoding="utf-8",
        )
    return seed_file


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[3]
    seeded = ensure_guidelines_seeded(base / "data" / "guidelines")
    print(f"Guideline seed ready: {seeded}")
