from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.prompts import (
    ASSESSMENT_SYSTEM_PROMPT,
    ASSESSMENT_USER_TEMPLATE,
    DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
    DIAGNOSTIC_ROUTER_USER_TEMPLATE,
    MANAGEMENT_ROUTER_SYSTEM_PROMPT,
    MANAGEMENT_ROUTER_USER_TEMPLATE,
    PARSE_HISTORY_SYSTEM_PROMPT,
    PARSE_HISTORY_USER_TEMPLATE,
    PRETRIAGE_ROUTER_SYSTEM_PROMPT,
    PRETRIAGE_ROUTER_USER_TEMPLATE,
)
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

    def parse_history(
        self,
        *,
        free_text: str,
        require_llm: bool = False,
        model_name: str | None = None,
    ) -> Tuple[Dict[str, object], List[str], float, bool]:
        selected_model = model_name or self.model_name
        if self._ollama is None:
            if require_llm:
                raise RuntimeError("LLM-парсинг обязателен, но Ollama недоступна.")
            parsed, missing, confidence = self._heuristic_parse_history(free_text)
            return parsed, missing, confidence, False

        prompt = PARSE_HISTORY_USER_TEMPLATE.format(free_text=free_text)
        try:
            response = self._ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "system", "content": PARSE_HISTORY_SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": prompt.strip()},
                ],
            )
            text = response["message"]["content"]
            data = self._extract_json_object(text)
            patient_data = dict(data.get("patient_data", {}))
            allowed_missing = {
                "name",
                "pain_type",
                "ecg_changes",
                "troponin",
                "hr",
                "bp",
                "age",
                "gender",
                "spo2",
                "glucose",
                "creatinine",
                "killip_class",
            }
            missing_fields = [str(item) for item in data.get("missing_fields", []) if str(item) in allowed_missing]
            confidence = float(data.get("confidence", 0.0))

            # Guardrails for critical fields.
            patient_data.setdefault("name", "Unknown")
            patient_data.setdefault("age", None)
            patient_data.setdefault("gender", "unknown")
            patient_data.setdefault("admission_time", "")
            patient_data.setdefault("pain_onset_time", "")
            patient_data.setdefault("pain_type", "none")
            patient_data.setdefault("pain_description", "")
            patient_data.setdefault("ecg_changes", "normal")
            patient_data.setdefault("troponin", 0.0)
            patient_data.setdefault("hr", 70)
            patient_data.setdefault("bp", "120/80")
            patient_data.setdefault("spo2", None)
            patient_data.setdefault("rr", None)
            patient_data.setdefault("glucose", None)
            patient_data.setdefault("creatinine", None)
            patient_data.setdefault("ast_alt_ckmb", {})
            patient_data.setdefault("lipid_profile", {})
            patient_data.setdefault("potassium_sodium_magnesium", {})
            patient_data.setdefault("echo_dkg_results", "")
            patient_data.setdefault("mri_results", "")
            patient_data.setdefault("ct_coronary", "")
            patient_data.setdefault("killip_class", "")
            patient_data.setdefault("interventions", [])
            patient_data.setdefault("medications", [])
            patient_data.setdefault("vital_signs", [])
            patient_data.setdefault("symptoms_text", free_text)
            confidence = max(0.0, min(1.0, confidence))
            return patient_data, missing_fields, confidence, True
        except Exception as exc:
            if require_llm:
                raise RuntimeError("LLM-парсинг анамнеза не удался.") from exc
            parsed, missing, confidence = self._heuristic_parse_history(free_text)
            return parsed, missing, confidence, False

    def route_pretriage(
        self,
        *,
        patient_data: Dict[str, object],
        parse_confidence: float,
        missing_fields: List[str],
        parsed_ok: bool,
        require_llm: bool = False,
        model_name: str | None = None,
    ) -> Tuple[str, float, str, bool]:
        if not parsed_ok:
            return "needs_more_data", 0.99, "Валидация данных не пройдена.", False
        if missing_fields and parse_confidence < 0.6:
            return "needs_more_data", 0.95, "Ключевых данных недостаточно.", False
        if self._ollama is None:
            if require_llm:
                raise RuntimeError("LLM pretriage обязателен, но Ollama недоступна.")
            return "proceed", 0.7, "Fallback pretriage: данные условно достаточны.", False

        prompt = PRETRIAGE_ROUTER_USER_TEMPLATE.format(
            patient_data=json.dumps(patient_data, ensure_ascii=False),
            parse_confidence=parse_confidence,
            missing_fields=", ".join(missing_fields) if missing_fields else "нет",
            parsed_ok=parsed_ok,
        )
        return self._call_router(
            system_prompt=PRETRIAGE_ROUTER_SYSTEM_PROMPT,
            user_prompt=prompt,
            allowed_steps={"proceed", "needs_more_data"},
            default_step="proceed",
            require_llm=require_llm,
            model_name=model_name,
            error_message="LLM pretriage роутинг не удался.",
        )

    def route_diagnostic(
        self,
        *,
        patient_data: Dict[str, object],
        rule_reasons: List[str],
        risk: float,
        risk_level: str,
        require_llm: bool = False,
        model_name: str | None = None,
    ) -> Tuple[str, float, str, bool]:
        if risk_level == "high" and risk >= 0.9:
            return "urgent", 0.99, "Очевидно высокий риск по hard-rules.", False
        if self._ollama is None:
            if require_llm:
                raise RuntimeError("LLM diagnostic router обязателен, но Ollama недоступна.")
            if risk >= 0.45:
                return "rag_llm", 0.7, "Fallback: нужна дополнительная диагностика.", False
            return "rule_only", 0.7, "Fallback: достаточно rule-based оценки.", False

        prompt = DIAGNOSTIC_ROUTER_USER_TEMPLATE.format(
            patient_data=json.dumps(patient_data, ensure_ascii=False),
            risk=risk,
            risk_level=risk_level,
            rule_reasons="; ".join(rule_reasons) if rule_reasons else "нет",
        )
        return self._call_router(
            system_prompt=DIAGNOSTIC_ROUTER_SYSTEM_PROMPT,
            user_prompt=prompt,
            allowed_steps={"urgent", "rag_llm", "rule_only"},
            default_step="rag_llm",
            require_llm=require_llm,
            model_name=model_name,
            error_message="LLM diagnostic роутинг не удался.",
        )

    def route_management(
        self,
        *,
        patient_data: Dict[str, object],
        risk: float,
        risk_level: str,
        explanation: str,
        require_llm: bool = False,
        model_name: str | None = None,
    ) -> Tuple[str, float, str, bool]:
        if self._ollama is None:
            if require_llm:
                raise RuntimeError("LLM management router обязателен, но Ollama недоступна.")
            if risk_level == "high":
                return "recommend_treatment", 0.7, "Fallback: высокий риск, нужен блок рекомендаций.", False
            return "monitor", 0.7, "Fallback: наблюдение и контроль.", False

        prompt = MANAGEMENT_ROUTER_USER_TEMPLATE.format(
            patient_data=json.dumps(patient_data, ensure_ascii=False),
            risk=risk,
            risk_level=risk_level,
            explanation=explanation,
        )
        return self._call_router(
            system_prompt=MANAGEMENT_ROUTER_SYSTEM_PROMPT,
            user_prompt=prompt,
            allowed_steps={"monitor", "recommend_treatment", "finalize"},
            default_step="finalize",
            require_llm=require_llm,
            model_name=model_name,
            error_message="LLM management роутинг не удался.",
        )

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
            explanation = self._ensure_russian_explanation(
                explanation,
                patient_data=patient_data,
                rule_reasons=rule_reasons,
            )
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

    def _ensure_russian_explanation(
        self,
        explanation: str,
        *,
        patient_data: Dict[str, object],
        rule_reasons: List[str],
    ) -> str:
        # Remove obvious CJK blocks if model mixes languages.
        text = re.sub(r"[\u4e00-\u9fff]+", " ", explanation)
        # Keep punctuation and spacing clean.
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return self._russian_fallback_explanation(patient_data, rule_reasons)

        cyr = len(re.findall(r"[А-Яа-яЁё]", text))
        lat = len(re.findall(r"[A-Za-z]", text))
        # If Russian letters are scarce compared to latin, force fallback text.
        if cyr < 20 or cyr < lat:
            return self._russian_fallback_explanation(patient_data, rule_reasons)

        # Keep only first 3-5 sentences to avoid long noisy output.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        compact = " ".join(sentences[:5]).strip()
        if len(compact) < 40:
            return self._russian_fallback_explanation(patient_data, rule_reasons)
        return compact

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise ValueError("JSON not found in model output")
        parsed = json.loads(json_match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not object")
        return parsed

    def _call_router(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_steps: set[str],
        default_step: str,
        require_llm: bool,
        model_name: str | None,
        error_message: str,
    ) -> Tuple[str, float, str, bool]:
        selected_model = model_name or self.model_name
        try:
            response = self._ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
            )
            data = self._extract_json_object(response["message"]["content"])
            next_step = str(data.get("next_step", default_step))
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", "")).strip() or "LLM router decision."
            if next_step not in allowed_steps:
                next_step = default_step
            confidence = max(0.0, min(1.0, confidence))
            return next_step, confidence, reason, True
        except Exception as exc:
            if require_llm:
                raise RuntimeError(error_message) from exc
            return default_step, 0.5, "Fallback router decision.", False

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

    def _russian_fallback_explanation(self, patient_data: Dict[str, object], rule_reasons: List[str]) -> str:
        troponin = float(patient_data.get("troponin", 0.0))
        ecg = str(patient_data.get("ecg_changes", "не указано"))
        pain = str(patient_data.get("pain_type", "не указано"))
        reasons = "; ".join(rule_reasons) if rule_reasons else "существенных rule-based срабатываний не отмечено"
        return (
            "Оценка сформирована на основе клинических признаков и правил triage. "
            f"Тип боли: {pain}, ЭКГ: {ecg}, тропонин: {troponin}. "
            f"Ключевые факторы риска: {reasons}. "
            "Результат носит предварительный характер и требует клинической верификации врачом."
        )

    def _heuristic_parse_history(self, free_text: str) -> Tuple[Dict[str, object], List[str], float]:
        text = free_text.lower()
        pain_type = "none"
        if "типич" in text or "загрудин" in text or "иррадиа" in text:
            pain_type = "typical"
        elif "боль" in text or "дискомфорт" in text:
            pain_type = "atypical"

        ecg_changes = "normal"
        if "st-elevation" in text or "подъем st" in text:
            ecg_changes = "ST-elevation"
        elif "st-depression" in text or "депресс" in text:
            ecg_changes = "ST-depression"

        troponin_match = re.search(r"тропонин[^0-9]*([0-9]+(?:[.,][0-9]+)?)", text)
        hr_match = re.search(r"(?:чсс|hr)[^0-9]*([0-9]{2,3})", text)
        bp_match = re.search(r"([0-9]{2,3})\s*/\s*([0-9]{2,3})", text)
        age_match = re.search(r"([0-9]{2})\s*(?:лет|года|год)", text)
        spo2_match = re.search(r"(?:spo2|сатурац)[^0-9]*([0-9]{2,3})", text)
        glucose_match = re.search(r"глюкоз[^0-9]*([0-9]+(?:[.,][0-9]+)?)", text)
        creatinine_match = re.search(r"креатинин[^0-9]*([0-9]+(?:[.,][0-9]+)?)", text)

        troponin = float(troponin_match.group(1).replace(",", ".")) if troponin_match else 0.0
        hr = int(hr_match.group(1)) if hr_match else 70
        bp = f"{bp_match.group(1)}/{bp_match.group(2)}" if bp_match else "120/80"
        age = int(age_match.group(1)) if age_match else None
        spo2 = float(spo2_match.group(1)) if spo2_match else None
        glucose = float(glucose_match.group(1).replace(",", ".")) if glucose_match else None
        creatinine = float(creatinine_match.group(1).replace(",", ".")) if creatinine_match else None
        gender = "male" if "муж" in text else "female" if "жен" in text else "unknown"

        missing_fields: List[str] = []
        if not troponin_match:
            missing_fields.append("troponin")
        if not hr_match:
            missing_fields.append("hr")
        if not bp_match:
            missing_fields.append("bp")
        if ecg_changes == "normal" and "экг" not in text and "ecg" not in text:
            missing_fields.append("ecg_changes")

        patient_data: Dict[str, object] = {
            "name": "Unknown",
            "age": age,
            "gender": gender,
            "admission_time": "",
            "pain_onset_time": "",
            "pain_type": pain_type,
            "pain_description": free_text,
            "ecg_changes": ecg_changes,
            "troponin": troponin,
            "hr": hr,
            "bp": bp,
            "spo2": spo2,
            "rr": None,
            "glucose": glucose,
            "creatinine": creatinine,
            "ast_alt_ckmb": {},
            "lipid_profile": {},
            "potassium_sodium_magnesium": {},
            "echo_dkg_results": "",
            "mri_results": "",
            "ct_coronary": "",
            "killip_class": "",
            "interventions": [],
            "medications": [],
            "vital_signs": [],
            "symptoms_text": free_text,
        }
        confidence = 0.4 if missing_fields else 0.7
        return patient_data, missing_fields, confidence


def build_repository(base_dir: Path) -> PatientRepository:
    return PatientRepository(base_dir / "data" / "patients.csv")


def build_retriever(base_dir: Path) -> GuidelinesRetriever:
    return GuidelinesRetriever(base_dir / "data" / "guidelines")
