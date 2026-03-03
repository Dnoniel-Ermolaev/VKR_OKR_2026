from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.prompts import ASSESSMENT_SYSTEM_PROMPT, ASSESSMENT_USER_TEMPLATE
from src.infrastructure.db.repository import PatientRepository
from src.infrastructure.rag.retriever import GuidelinesRetriever
from src.medical.scores import grace_score, heart_score


class LlmClient:
    def __init__(self, model_name: str = "deepseek-r1:7b") -> None:
        self.model_name = os.getenv("OLLAMA_MODEL", model_name)
        try:
            import ollama  # type: ignore

            self._ollama = ollama
        except Exception:
            self._ollama = None

    def assess(
        self,
        *,
        patient_data: Dict[str, object],
        rule_reasons: List[str],
        rag_context: str,
        require_llm: bool = False,
        model_name: str | None = None,
    ) -> Tuple[float, str, bool]:
        selected_model = model_name or self.model_name
        if self._ollama is None:
            if require_llm:
                raise RuntimeError(
                    "LLM режим обязателен, но пакет ollama недоступен. "
                    "Установите зависимость и запустите Ollama."
                )
            adjustment, explanation = self._fallback(patient_data, rule_reasons, rag_context)
            return adjustment, explanation, False

        prompt = ASSESSMENT_USER_TEMPLATE.format(
            patient_data=json.dumps(patient_data, ensure_ascii=False),
            rule_reasons="; ".join(rule_reasons) if rule_reasons else "нет срабатываний",
            rag_context=rag_context,
        )
        try:
            response = self._ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "system", "content": ASSESSMENT_SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt.strip()},
                ],
            )
            text = response["message"]["content"]
            adjustment, explanation = self._parse_output(text)
            return adjustment, explanation, True
        except Exception as exc:
            if require_llm:
                raise RuntimeError(
                    "LLM режим обязателен, но Ollama не вернул корректный ответ. "
                    "Проверьте `ollama serve` и наличие модели."
                ) from exc
            adjustment, explanation = self._fallback(patient_data, rule_reasons, rag_context)
            return adjustment, explanation, False

    def _parse_output(self, text: str) -> Tuple[float, str]:
        try:
            data = json.loads(text)
            adjustment = float(data.get("risk_adjustment", 0.0))
            explanation = str(data.get("explanation", "")).strip()
        except Exception:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    adjustment = float(data.get("risk_adjustment", 0.0))
                    explanation = str(data.get("explanation", "")).strip()
                except Exception:
                    adjustment = 0.0
                    explanation = text.strip()[:600]
            else:
                adjustment = 0.0
                explanation = text.strip()[:600]
        adjustment = max(-0.15, min(0.15, adjustment))
        if not explanation:
            explanation = "LLM не вернула объяснение в ожидаемом формате."
        return adjustment, explanation

    def _fallback(self, patient_data: Dict[str, object], rule_reasons: List[str], rag_context: str) -> Tuple[float, str]:
        heart = heart_score(patient_data)
        grace = grace_score(patient_data)
        adjustment = 0.05 if heart >= 5 or grace >= 130 else -0.03
        explanation = (
            "Оценка выполнена в fallback-режиме без LLM. "
            f"HEART={heart}, GRACE={grace}. "
            f"Базовые правила: {'; '.join(rule_reasons) if rule_reasons else 'нет срабатываний'}. "
            "Контекст рекомендаций учтен из локальной базы."
        )
        return adjustment, explanation


def build_repository(base_dir: Path) -> PatientRepository:
    return PatientRepository(base_dir / "data" / "patients.csv")


def build_retriever(base_dir: Path) -> GuidelinesRetriever:
    return GuidelinesRetriever(base_dir / "data" / "guidelines")
