from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None

from src.infrastructure.db.models import PatientRecord


DEFAULT_COLUMNS = [
    "id",
    "name",
    "symptoms_json",
    "ecg_desc",
    "troponin",
    "hr",
    "bp",
    "risk_level",
    "explanation",
    "timestamp",
]


class PatientRepository:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self._init_empty()

    def save_patient(self, record: PatientRecord) -> None:
        if pd is not None:
            df = pd.read_csv(self.csv_path)
            df = pd.concat([df, pd.DataFrame([record.model_dump()])], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            return
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
            writer.writerow(record.model_dump())

    def search_patients(
        self,
        *,
        name: Optional[str] = None,
        risk_level: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, object]]:
        if pd is not None:
            df = pd.read_csv(self.csv_path)
            if name:
                df = df[df["name"].astype(str).str.contains(name, case=False, na=False)]
            if risk_level:
                df = df[df["risk_level"] == risk_level]
            return df.head(top_k).to_dict(orient="records")

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        filtered: List[Dict[str, object]] = []
        for row in rows:
            if name and name.lower() not in str(row.get("name", "")).lower():
                continue
            if risk_level and str(row.get("risk_level")) != risk_level:
                continue
            filtered.append(row)
        return filtered[:top_k]

    def _init_empty(self) -> None:
        if pd is not None:
            pd.DataFrame(columns=DEFAULT_COLUMNS).to_csv(self.csv_path, index=False)
            return
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
            writer.writeheader()
