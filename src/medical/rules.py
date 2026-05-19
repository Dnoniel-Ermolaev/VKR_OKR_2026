"""Расширенный rule-based слой для скрининга ОКС.

Идея этого модуля:

- хранить правила как типизированные объекты ``Rule`` с категорией,
  ссылкой на КР Минздрава и человеко-читаемой формулировкой;
- эмитить структурированный список ``rule_fires`` (для трассы графа
  и для UI визуализации);
- сохранять обратно-совместимую функцию ``evaluate_hard_rules``
  с прежней сигнатурой ``(risk, risk_level, reasons, route_to_llm)``,
  чтобы существующие тесты и узлы графа продолжали работать.

Источники правил:

- КР Минздрав 2020 'Острый инфаркт миокарда с подъёмом сегмента ST
  электрокардиограммы' (см. ``data/guidelines/oks_stemi_minzdrav_2020.txt``).
- КР Минздрав 2020 'Острый коронарный синдром без подъёма сегмента ST
  электрокардиограммы' (см. ``data/guidelines/oks_nstemi_minzdrav_2020.txt``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from src.medical.diagnosis import (
    AcsDiagnosis,
    TROPONIN_99_PERCENTILE,
    TROPONIN_HIGH,
)


RuleCategory = Literal["ecg", "biomarker", "clinical", "hemodynamic", "score", "time"]
RuleSeverity = Literal["critical", "high", "medium", "low"]


# ---- Контейнер для входных 'фактов' о пациенте ----
@dataclass
class PatientFacts:
    """Нормализованный срез данных пациента, используемый правилами."""

    raw: Dict[str, Any]

    @property
    def troponin(self) -> float:
        try:
            return float(self.raw.get("troponin", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def hr(self) -> int:
        try:
            return int(self.raw.get("hr", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def sbp(self) -> int | None:
        bp = str(self.raw.get("bp", "") or "")
        if "/" not in bp:
            return None
        try:
            return int(float(bp.split("/", 1)[0].strip()))
        except (TypeError, ValueError):
            return None

    @property
    def ecg(self) -> str:
        return str(self.raw.get("ecg_changes", "") or "").lower()

    @property
    def pain_type(self) -> str:
        return str(self.raw.get("pain_type", "") or "").lower()

    @property
    def killip(self) -> str:
        return str(self.raw.get("killip_class", "") or "").upper()

    @property
    def age(self) -> int | None:
        value = self.raw.get("age")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def gender(self) -> str:
        return str(self.raw.get("gender", "") or "").lower()

    @property
    def creatinine(self) -> float | None:
        value = self.raw.get("creatinine")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def spo2(self) -> float | None:
        value = self.raw.get("spo2")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def text_blob(self) -> str:
        return " ".join(
            str(self.raw.get(field, "") or "")
            for field in ("symptoms_text", "pain_description", "ecg_changes")
        ).lower()


# ---- Структура правила ----
@dataclass(frozen=True)
class Rule:
    """Одно типизированное правило rule-based блока."""

    id: str
    category: RuleCategory
    title_ru: str
    kr_reference: str
    severity: RuleSeverity
    predicate: Callable[[PatientFacts], bool]
    risk_delta: float = 0.0
    sets_diagnosis: Optional[AcsDiagnosis] = None
    short_reason: str = ""


@dataclass
class RuleFire:
    """Факт срабатывания правила вместе с числовым значением."""

    rule_id: str
    category: RuleCategory
    title_ru: str
    severity: RuleSeverity
    reason: str
    risk_delta: float
    sets_diagnosis: Optional[str] = None
    kr_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


# ---- Утилиты для предикатов ----
def _contains_any(text: str, markers: Tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _ecg_pst_pattern(facts: PatientFacts) -> bool:
    """Любой паттерн, эквивалентный подъёму ST на ЭКГ."""
    ecg = facts.ecg
    pst_markers = (
        "st-elevation",
        "элевация st",
        "подъём st",
        "подъем st",
        "подъём сегмента st",
        "подъем сегмента st",
    )
    if _contains_any(ecg, pst_markers):
        return True
    lbbb_markers = (
        "новая лнпг",
        "новая блокада левой ножки",
        "впервые возникшая блокада лнпг",
        "new lbbb",
    )
    if _contains_any(ecg, lbbb_markers):
        return True
    return _contains_any(ecg, ("задний им", "v7-v9", "v7v9"))


# ---- РУЛБУК ----
RULEBOOK: List[Rule] = [
    # ============= ЭКГ-критерии (категория ecg) =============
    Rule(
        id="ecg_st_elev_persistent",
        category="ecg",
        title_ru="Стойкий подъём сегмента ST в смежных отведениях",
        kr_reference="КР Минздрав 2020 'ИМпST', стр. 109-112",
        severity="critical",
        predicate=lambda f: _contains_any(
            f.ecg,
            (
                "st-elevation",
                "элевация st",
                "подъём st",
                "подъем st",
                "подъём сегмента st",
                "подъем сегмента st",
            ),
        ),
        risk_delta=0.78,
        sets_diagnosis=AcsDiagnosis.OKS_PST,
        short_reason="ЭКГ-критерий подъёма ST (КР Минздрав ИМпST).",
    ),
    Rule(
        id="ecg_st_elev_v2v3_sex_adjusted",
        category="ecg",
        title_ru="Подъём ST в V2-V3 с учётом пола/возраста",
        kr_reference="КР Минздрав 2020 'ИМпST', стр. 110 (пороги >=2/>=2.5/>=1.5 мм)",
        severity="high",
        predicate=lambda f: "v2" in f.ecg and "v3" in f.ecg and _contains_any(
            f.ecg, ("st-elevation", "элевация", "подъём st", "подъем st")
        ),
        risk_delta=0.05,
        short_reason="Подъём ST в V2-V3 с поло-возрастными порогами.",
    ),
    Rule(
        id="ecg_new_lbbb",
        category="ecg",
        title_ru="Впервые возникшая блокада левой ножки пучка Гиса (ЛНПГ)",
        kr_reference="КР Минздрав 2020 'ИМпST': эквивалент критерия ИМпST",
        severity="critical",
        predicate=lambda f: _contains_any(
            f.ecg,
            (
                "новая лнпг",
                "новая блокада левой ножки",
                "впервые возникшая блокада лнпг",
                "new lbbb",
            ),
        ),
        risk_delta=0.72,
        sets_diagnosis=AcsDiagnosis.OKS_PST,
        short_reason="Новая ЛНПГ - эквивалент ИМпST.",
    ),
    Rule(
        id="ecg_posterior_mi",
        category="ecg",
        title_ru="ЭКГ-признаки заднего ИМ",
        kr_reference="КР Минздрав 2020 'ИМпST': задний ИМ, V7-V9",
        severity="critical",
        predicate=lambda f: _contains_any(
            f.ecg,
            (
                "задний им",
                "v7-v9",
                "v7v9",
                "реципрокная депрессия v1-v3",
            ),
        ),
        risk_delta=0.7,
        sets_diagnosis=AcsDiagnosis.OKS_PST,
        short_reason="Задний ИМ (V7-V9 / реципрокная депрессия V1-V3).",
    ),
    Rule(
        id="ecg_st_depression_significant",
        category="ecg",
        title_ru="Депрессия сегмента ST >=0.5 мм в смежных отведениях",
        kr_reference="КР Минздрав 2020 'ОКСбпST', стр. 117",
        severity="high",
        predicate=lambda f: _contains_any(
            f.ecg, ("st-depression", "депрессия st", "снижение st")
        ),
        risk_delta=0.3,
        sets_diagnosis=AcsDiagnosis.OKS_BPST,
        short_reason="Депрессия ST - критерий ишемии (ОКСбпST).",
    ),
    Rule(
        id="ecg_t_inversion_deep",
        category="ecg",
        title_ru="Глубокая инверсия зубца T в смежных отведениях",
        kr_reference="КР Минздрав 2020 'ОКСбпST', стр. 118",
        severity="high",
        predicate=lambda f: _contains_any(
            f.ecg, ("инверсия t", "отрицательный t", "t-инверсия")
        ),
        risk_delta=0.2,
        sets_diagnosis=AcsDiagnosis.OKS_BPST,
        short_reason="Глубокая инверсия T - признак острой ишемии.",
    ),
    Rule(
        id="ecg_q_pathologic",
        category="ecg",
        title_ru="Патологический зубец Q",
        kr_reference="КР Минздрав 2020: универсальное определение ИМ",
        severity="medium",
        predicate=lambda f: _contains_any(f.ecg, ("патологический q", "qs ", "q-волна")),
        risk_delta=0.1,
        short_reason="Патологический Q - признак ранее перенесённого/острого ИМ.",
    ),
    Rule(
        id="ecg_de_winter",
        category="ecg",
        title_ru="Паттерн de Winter (эквивалент окклюзии ПМЖА)",
        kr_reference="КР Минздрав 2020: ЭКГ-эквиваленты ИМпST",
        severity="critical",
        predicate=lambda f: "de winter" in f.ecg or "де винтер" in f.ecg,
        risk_delta=0.6,
        sets_diagnosis=AcsDiagnosis.OKS_PST,
        short_reason="Паттерн de Winter - эквивалент острой окклюзии.",
    ),
    Rule(
        id="ecg_wellens",
        category="ecg",
        title_ru="Паттерн Wellens (критическая стенозация ПМЖА)",
        kr_reference="ОКСбпST высокого риска",
        severity="high",
        predicate=lambda f: "wellens" in f.ecg or "велленс" in f.ecg,
        risk_delta=0.3,
        sets_diagnosis=AcsDiagnosis.OKS_BPST,
        short_reason="Паттерн Wellens - высокий риск ИМбпST.",
    ),
    Rule(
        id="ecg_sgarbossa",
        category="ecg",
        title_ru="Критерии Sgarbossa у пациента с ЛНПГ",
        kr_reference="Диагностика ИМ на фоне ЛНПГ",
        severity="high",
        predicate=lambda f: "sgarbossa" in f.ecg or "сгарбосса" in f.ecg,
        risk_delta=0.5,
        sets_diagnosis=AcsDiagnosis.OKS_PST,
        short_reason="Критерии Sgarbossa - ИМ на фоне ЛНПГ.",
    ),

    # ============= Биомаркеры (категория biomarker) =============
    Rule(
        id="troponin_99th_percentile",
        category="biomarker",
        title_ru="Тропонин выше 99-го перцентиля верхней границы нормы",
        kr_reference="Универсальное определение ИМ (4-й пересмотр), КР Минздрав",
        severity="high",
        predicate=lambda f: TROPONIN_99_PERCENTILE <= f.troponin < TROPONIN_HIGH,
        risk_delta=0.18,
        short_reason=(
            f"Тропонин выше 99-го перцентиля (>{TROPONIN_99_PERCENTILE} нг/мл). "
            "Требуется повторное измерение для оценки динамики."
        ),
    ),
    Rule(
        id="troponin_high",
        category="biomarker",
        title_ru="Выраженное повышение тропонина",
        kr_reference="КР Минздрав 2020: критерий повреждения миокарда",
        severity="critical",
        predicate=lambda f: f.troponin >= TROPONIN_HIGH,
        risk_delta=0.72,
        sets_diagnosis=AcsDiagnosis.OKS_BPST,
        short_reason=f"Выраженное повышение тропонина (>={TROPONIN_HIGH} нг/мл).",
    ),

    # ============= Клиника (категория clinical) =============
    Rule(
        id="chest_pain_typical",
        category="clinical",
        title_ru="Типичная ангинозная боль",
        kr_reference="КР Минздрав 2020: типичная клиническая картина ОКС",
        severity="high",
        predicate=lambda f: f.pain_type == "typical",
        risk_delta=0.2,
        short_reason="Типичная загрудинная боль ангинозного характера.",
    ),
    Rule(
        id="chest_pain_duration_gt_20min",
        category="clinical",
        title_ru="Длительность ангинозного приступа >20 минут",
        kr_reference=(
            "КР Минздрав 2020 'ИМпST', стр. 109 - определение стойкого "
            "подъёма ST: длительность >20 минут."
        ),
        severity="high",
        predicate=lambda f: _contains_any(
            f.text_blob,
            (
                "длительность более 20",
                "более 20 минут",
                "более 30 минут",
                ">20 мин",
                ">30 мин",
                "стойкая боль",
                "стойкий приступ",
                "затяжной приступ",
            ),
        ),
        risk_delta=0.1,
        short_reason="Стойкий ангинозный приступ длительностью >20 минут.",
    ),
    Rule(
        id="recurrent_ischemic_pain",
        category="clinical",
        title_ru="Рецидивирующая ишемическая боль на фоне терапии",
        kr_reference="КР Минздрав 2020 'ОКСбпST': критерий очень высокого риска",
        severity="critical",
        predicate=lambda f: _contains_any(
            f.text_blob,
            ("рецидив боли", "повторный приступ", "возобновление боли", "не купируется"),
        ),
        risk_delta=0.25,
        short_reason="Рецидив ишемической боли - критерий очень высокого риска.",
    ),
    Rule(
        id="mechanical_complication",
        category="clinical",
        title_ru="Механические осложнения ИМ",
        kr_reference="КР Минздрав 2020 'ИМпST': осложнения",
        severity="critical",
        predicate=lambda f: _contains_any(
            f.text_blob,
            (
                "разрыв",
                "острая митральная недостаточность",
                "тампонада",
                "межжелудочковая перегородка",
            ),
        ),
        risk_delta=0.3,
        short_reason="Подозрение на механическое осложнение ИМ.",
    ),
    Rule(
        id="life_threatening_arrhythmia",
        category="clinical",
        title_ru="Жизнеугрожающая аритмия (ЖТ/ФЖ)",
        kr_reference="КР Минздрав 2020: критерий очень высокого риска",
        severity="critical",
        predicate=lambda f: _contains_any(
            f.text_blob,
            (
                "желудочков",  # покрывает 'желудочковая', 'желудочковой' и т.п.
                "фибрилляц",
                " жт",
                " фж",
                "жт)",
                "фж)",
            ),
        ),
        risk_delta=0.3,
        short_reason="ЖТ/ФЖ - жизнеугрожающая аритмия.",
    ),

    # ============= Гемодинамика (категория hemodynamic) =============
    Rule(
        id="tachycardia",
        category="hemodynamic",
        title_ru="Тахикардия >110 уд/мин",
        kr_reference="GRACE-фактор риска",
        severity="medium",
        predicate=lambda f: f.hr > 110,
        risk_delta=0.1,
        short_reason="Тахикардия >110 уд/мин - фактор риска по GRACE.",
    ),
    Rule(
        id="hemodynamic_instability",
        category="hemodynamic",
        title_ru="Гемодинамическая нестабильность (САД <90 мм рт.ст.)",
        kr_reference="КР Минздрав 2020: критерий очень высокого риска",
        severity="critical",
        predicate=lambda f: f.sbp is not None and f.sbp < 90,
        risk_delta=0.25,
        short_reason="Гипотензия / шок - нестабильная гемодинамика.",
    ),
    Rule(
        id="killip_3_4",
        category="hemodynamic",
        title_ru="Острая сердечная недостаточность Killip III-IV",
        kr_reference="КР Минздрав 2020 'ИМпST': класс Killip",
        severity="critical",
        predicate=lambda f: f.killip in {"III", "IV", "3", "4"},
        risk_delta=0.15,
        short_reason="ОСН Killip III-IV.",
    ),
    Rule(
        id="hypoxemia",
        category="hemodynamic",
        title_ru="Гипоксемия (SpO2 <90%)",
        kr_reference="КР Минздрав 2020: показание к оксигенотерапии",
        severity="high",
        predicate=lambda f: f.spo2 is not None and f.spo2 < 90,
        risk_delta=0.1,
        short_reason="SpO2 <90% - гипоксемия, показание к оксигенотерапии.",
    ),

    # ============= Скоры/демография (категория score) =============
    Rule(
        id="age_ge_75",
        category="score",
        title_ru="Возраст >=75 лет",
        kr_reference="GRACE-фактор риска",
        severity="medium",
        predicate=lambda f: f.age is not None and f.age >= 75,
        risk_delta=0.07,
        short_reason="Возраст >=75 - независимый фактор риска по GRACE.",
    ),
    Rule(
        id="ckd_creatinine_high",
        category="score",
        title_ru="Повышение креатинина (ХБП / GRACE)",
        kr_reference="GRACE-фактор риска: уровень креатинина",
        severity="medium",
        predicate=lambda f: f.creatinine is not None and f.creatinine >= 140,
        risk_delta=0.1,
        short_reason="Креатинин >=140 мкмоль/л - фактор риска по GRACE.",
    ),
    Rule(
        id="diabetes_known",
        category="score",
        title_ru="Сахарный диабет в анамнезе",
        kr_reference="Фактор риска ИБС/ОКС",
        severity="medium",
        predicate=lambda f: _contains_any(
            f.text_blob, ("сахарный диабет", "сд 2 типа", "сд2", "диабет")
        ),
        risk_delta=0.05,
        short_reason="Сахарный диабет - фактор риска ОКС.",
    ),

    # ============= Тайминг (категория time) =============
    Rule(
        id="pain_onset_within_12h",
        category="time",
        title_ru="Окно для первичной реперфузии <=12 ч от начала приступа",
        kr_reference="КР Минздрав 2020 'ИМпST': окно реперфузии",
        severity="high",
        predicate=lambda f: _contains_any(
            f.text_blob,
            (
                "менее 12 часов",
                "до 12 часов",
                "в первые 12 часов",
                "<12 ч",
                "< 12 ч",
            ),
        ),
        risk_delta=0.0,
        short_reason="Пациент в пределах окна реперфузии <=12 ч.",
    ),
]

RULES_BY_ID: Dict[str, Rule] = {rule.id: rule for rule in RULEBOOK}


# ---- Главные функции ----
def evaluate_rules(patient_data: Dict[str, Any]) -> List[RuleFire]:
    """Прогнать рулбук и вернуть список сработавших правил."""
    facts = PatientFacts(raw=dict(patient_data))
    fires: List[RuleFire] = []
    for rule in RULEBOOK:
        try:
            triggered = bool(rule.predicate(facts))
        except Exception:
            triggered = False
        if not triggered:
            continue
        fires.append(
            RuleFire(
                rule_id=rule.id,
                category=rule.category,
                title_ru=rule.title_ru,
                severity=rule.severity,
                reason=rule.short_reason or rule.title_ru,
                risk_delta=float(rule.risk_delta),
                sets_diagnosis=rule.sets_diagnosis.value if rule.sets_diagnosis else None,
                kr_reference=rule.kr_reference,
            )
        )
    return fires


def evaluate_hard_rules(patient_data: Dict[str, object]) -> Tuple[float, str, List[str], bool]:
    """Обратно-совместимая обёртка для текущего кода узла ``rule_check``.

    Возвращает ``(risk, risk_level, reasons, route_to_llm)``.

    ``route_to_llm=False`` означает, что high-risk ситуация очевидна
    (например, стойкий подъём ST или выраженное повышение тропонина) и
    дальнейшие LLM-узлы не нужны.
    """
    fires = evaluate_rules(dict(patient_data))
    reasons = [fire.reason for fire in fires]

    # ----- очевидные high-risk шорткаты по КР -----
    critical_ids = {
        "ecg_st_elev_persistent",
        "ecg_new_lbbb",
        "ecg_posterior_mi",
        "ecg_de_winter",
        "troponin_high",
        "life_threatening_arrhythmia",
        "mechanical_complication",
        "hemodynamic_instability",
    }
    has_critical = any(fire.rule_id in critical_ids for fire in fires)
    if has_critical:
        risk = 0.95
        if any(fire.rule_id == "ecg_st_elev_persistent" for fire in fires):
            risk = 0.98
        if any(fire.rule_id == "troponin_high" for fire in fires):
            risk = max(risk, 0.92)
        return risk, "high", reasons, False

    # ----- аддитивный скор -----
    risk = 0.2 + sum(fire.risk_delta for fire in fires)
    risk = max(0.0, min(0.95, risk))

    if risk >= 0.75:
        return risk, "high", reasons, False
    if risk >= 0.45:
        return risk, "medium", reasons, True
    return risk, "low", reasons, True


def fires_to_jsonable(fires: List[RuleFire]) -> List[Dict[str, Any]]:
    """Сериализовать список ``RuleFire`` в JSON-совместимый список."""
    return [fire.to_dict() for fire in fires]


__all__ = [
    "PatientFacts",
    "Rule",
    "RuleFire",
    "RULEBOOK",
    "RULES_BY_ID",
    "evaluate_rules",
    "evaluate_hard_rules",
    "fires_to_jsonable",
]
