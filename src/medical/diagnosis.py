"""Детерминированный классификатор ОКС/ИМ по клиническим рекомендациям Минздрава.

Возвращает дискретную диагностическую метку из множества:
``ИМпST / ИМбпST / ОКСпST / ОКСбпST / НС / ОКС маловероятен``.

Алгоритм опирается на:

1. ЭКГ-критерии стойкого подъёма сегмента ST (КР Минздрав 2020, ИМпST,
   стр. 109-116): подъём >=1 мм в >=2 смежных отведениях
   (для V2-V3: >=2 мм у мужчин >=40 лет, >=2.5 мм у мужчин <40 лет,
   >=1.5 мм у женщин), новая ЛНПГ, задний ИМ.
2. Депрессия ST / инверсия T (ОКСбпST, стр. 117-120).
3. Динамика тропонина: значение выше порога 99-го перцентиля,
   подтверждённый рост.
4. Длительность ангинозного приступа >20 минут (определение стойкого
   подъёма ST на ЭКГ).

Классификатор не выставляет окончательный диагноз: он эмитирует
наиболее вероятную клиническую категорию и сопровождает её
обоснованием (``criteria_fired``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class AcsDiagnosis(str, Enum):
    """Дискретные диагностические метки ОКС по КР Минздрава."""

    IM_PST = "im_pst"
    IM_BPST = "im_bpst"
    OKS_PST = "oks_pst"
    OKS_BPST = "oks_bpst"
    NS = "ns"
    OKS_UNLIKELY = "oks_unlikely"


# Приоритет при выборе из нескольких кандидатов:
# чем меньше число, тем выше приоритет.
DIAGNOSIS_PRIORITY: Dict[AcsDiagnosis, int] = {
    AcsDiagnosis.IM_PST: 1,
    AcsDiagnosis.OKS_PST: 2,
    AcsDiagnosis.IM_BPST: 3,
    AcsDiagnosis.OKS_BPST: 4,
    AcsDiagnosis.NS: 5,
    AcsDiagnosis.OKS_UNLIKELY: 6,
}

# Предлагаемые коды МКБ-10 (по группам справочника catalog.py).
ICD_SUGGESTIONS: Dict[AcsDiagnosis, List[str]] = {
    AcsDiagnosis.IM_PST: ["I21.0", "I21.1", "I21.2", "I22.0", "I22.1"],
    AcsDiagnosis.IM_BPST: ["I21.4", "I21.9"],
    AcsDiagnosis.OKS_PST: ["I21.9", "I24.9"],
    AcsDiagnosis.OKS_BPST: ["I20.0", "I24.9"],
    AcsDiagnosis.NS: ["I20.0"],
    AcsDiagnosis.OKS_UNLIKELY: [],
}


@dataclass
class DiagnosisResult:
    """Результат работы классификатора."""

    label: AcsDiagnosis
    confidence: float
    criteria_fired: List[str] = field(default_factory=list)
    icd10_suggested: List[str] = field(default_factory=list)
    needs_repeat_troponin: bool = False
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["label"] = self.label.value
        return data


# ---- Пороги (вынесены константами для прозрачности и тестируемости) ----
TROPONIN_99_PERCENTILE = 0.014   # нг/мл - универсальный порог высокочувствительного тропонина
TROPONIN_ELEVATED = 0.05         # нг/мл - клинически значимый рост
TROPONIN_HIGH = 0.1              # нг/мл - соответствует старому high-risk порогу
PAIN_DURATION_PERSISTENT_MIN = 20  # минуты - критерий стойкого приступа


# ---- Утилиты разбора входных данных ----
def _ecg_text(patient_data: Dict[str, Any]) -> str:
    return str(patient_data.get("ecg_changes", "") or "").lower()


def _has_st_elevation(ecg: str) -> bool:
    """Признак подъёма ST в свободном описании ЭКГ."""
    if "st-elevation" in ecg:
        return True
    russian_markers = (
        "элевация st",
        "подъём st",
        "подъем st",
        "подъём сегмента st",
        "подъем сегмента st",
        "st^",
    )
    return any(marker in ecg for marker in russian_markers)


def _has_st_depression(ecg: str) -> bool:
    if "st-depression" in ecg:
        return True
    return any(marker in ecg for marker in ("депрессия st", "снижение st", "stv"))


def _has_t_inversion(ecg: str) -> bool:
    return any(marker in ecg for marker in ("инверсия t", "отрицательный t", "t-инверсия"))


def _has_new_lbbb(ecg: str) -> bool:
    return any(
        marker in ecg
        for marker in (
            "новая лнпг",
            "новая блокада левой ножки",
            "впервые возникшая блокада лнпг",
            "new lbbb",
        )
    )


def _has_posterior_signs(ecg: str) -> bool:
    return any(
        marker in ecg
        for marker in (
            "задний им",
            "задне",
            "v7-v9",
            "v7v9",
        )
    )


def _troponin(patient_data: Dict[str, Any]) -> float:
    try:
        return float(patient_data.get("troponin", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pain_duration_persistent(patient_data: Dict[str, Any]) -> bool:
    """Эвристика длительности приступа.

    Используем поля symptoms_text / pain_description (нет числового поля
    duration в PatientData) и явное упоминание длительности.
    """
    text_blob = " ".join(
        str(patient_data.get(field, "") or "")
        for field in ("symptoms_text", "pain_description", "ecg_changes")
    ).lower()
    markers = (
        "длительность более 20",
        "более 20 минут",
        "более 30 минут",
        ">20 мин",
        ">30 мин",
        "стойкая боль",
        "стойкий приступ",
        "затяжной приступ",
    )
    return any(marker in text_blob for marker in markers)


def _ischemic_clinical_picture(patient_data: Dict[str, Any]) -> bool:
    """Есть ли клиническая картина ишемии (типичная боль + сопутствующие признаки)."""
    pain = str(patient_data.get("pain_type", "") or "").lower()
    if pain == "typical":
        return True
    text_blob = " ".join(
        str(patient_data.get(field, "") or "")
        for field in ("symptoms_text", "pain_description")
    ).lower()
    return any(
        marker in text_blob
        for marker in (
            "ангинозная боль",
            "загрудинная боль",
            "за грудиной",
            "давящая боль",
            "жгучая боль",
            "иррадиация в",
        )
    )


def _troponin_was_repeated(patient_data: Dict[str, Any]) -> bool:
    """Признак того, что у нас уже есть серия тропонинов."""
    vitals = patient_data.get("vital_signs") or []
    if isinstance(vitals, list) and len(vitals) >= 1:
        for entry in vitals:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "") or "").lower()
            code = str(entry.get("code", "") or "").lower()
            if "тропонин" in name or "troponin" in code:
                return True
    return False


# ---- Главная функция ----
def classify_acs(
    patient_data: Dict[str, Any],
    rule_reasons: List[str] | None = None,
) -> DiagnosisResult:
    """Применить детерминированный классификатор к данным пациента.

    Аргументы:
        patient_data: словарь с уже нормализованными полями (после ``parse_input``).
        rule_reasons: список причин, сработавших в rule-based блоке;
            используется как дополнительный сигнал для обоснования.

    Возвращает :class:`DiagnosisResult` с выбранной меткой, обоснованием
    и предложением кодов МКБ-10.
    """
    rule_reasons = rule_reasons or []

    ecg = _ecg_text(patient_data)
    troponin = _troponin(patient_data)
    st_up = _has_st_elevation(ecg)
    new_lbbb = _has_new_lbbb(ecg)
    posterior = _has_posterior_signs(ecg)
    st_down = _has_st_depression(ecg)
    t_inv = _has_t_inversion(ecg)
    has_clinic = _ischemic_clinical_picture(patient_data)
    persistent = _pain_duration_persistent(patient_data)
    troponin_repeated = _troponin_was_repeated(patient_data)

    troponin_elevated = troponin >= TROPONIN_99_PERCENTILE and troponin > 0
    troponin_high = troponin >= TROPONIN_HIGH
    pst_pattern = st_up or new_lbbb or posterior
    bpst_pattern = st_down or t_inv

    fired: List[str] = []

    if st_up:
        fired.append(
            "ЭКГ: стойкий подъём сегмента ST в смежных отведениях - "
            "ЭКГ-критерий ИМпST (КР Минздрав 2020)."
        )
    if new_lbbb:
        fired.append(
            "ЭКГ: впервые возникшая блокада ЛНПГ - эквивалент ИМпST."
        )
    if posterior:
        fired.append(
            "ЭКГ: признаки заднего ИМ (V7-V9 или реципрокная депрессия V1-V3)."
        )
    if st_down:
        fired.append(
            "ЭКГ: депрессия сегмента ST в смежных отведениях - критерий ишемии (ОКСбпST)."
        )
    if t_inv:
        fired.append(
            "ЭКГ: глубокая инверсия зубца T - признак острой ишемии."
        )
    if troponin_high:
        fired.append(
            f"Тропонин {troponin} нг/мл - выраженное повышение, "
            "соответствует повреждению миокарда."
        )
    elif troponin_elevated:
        fired.append(
            f"Тропонин {troponin} нг/мл - выше 99-го перцентиля; "
            "требуется повторное измерение и оценка динамики."
        )
    if persistent:
        fired.append("Длительность ангинозного приступа >20 минут (стойкий приступ).")
    if has_clinic and not (st_up or st_down or t_inv or new_lbbb):
        fired.append("Типичная клиническая картина ишемии без явных изменений ЭКГ.")

    # ---- Принятие решения по приоритету КР ----
    if pst_pattern and (troponin_high or troponin_elevated) and (persistent or troponin_repeated):
        label = AcsDiagnosis.IM_PST
        confidence = 0.95 if troponin_high and persistent else 0.85
        rationale = (
            "Сочетание стойкого подъёма ST на ЭКГ (или эквивалента) и повышенного "
            "тропонина при длительном ангинозном приступе соответствует "
            "клиническим критериям ИМпST (КР Минздрав 2020)."
        )
        needs_repeat = False

    elif pst_pattern:
        label = AcsDiagnosis.OKS_PST
        confidence = 0.85 if persistent else 0.7
        rationale = (
            "Подъём сегмента ST на ЭКГ (или эквивалент) при отсутствии "
            "подтверждённой динамики тропонина - ОКСпST до повторного "
            "измерения биомаркеров."
        )
        needs_repeat = True

    elif (st_down or t_inv) and (troponin_high or troponin_elevated) and has_clinic and troponin_repeated:
        label = AcsDiagnosis.IM_BPST
        confidence = 0.85 if troponin_high else 0.7
        rationale = (
            "Отсутствие стойкого подъёма ST, ишемические изменения (депрессия ST / "
            "инверсия T) и подтверждённое повышение тропонина соответствуют "
            "критериям ИМбпST."
        )
        needs_repeat = False

    elif (st_down or t_inv) and (has_clinic or troponin_elevated):
        label = AcsDiagnosis.OKS_BPST
        confidence = 0.75 if troponin_elevated else 0.65
        rationale = (
            "Картина ОКСбпST: ишемические изменения ЭКГ и/или клиника без "
            "подтверждённого подъёма ST; необходима оценка повторного тропонина."
        )
        needs_repeat = True

    elif has_clinic and not troponin_high and not bpst_pattern:
        label = AcsDiagnosis.NS
        confidence = 0.6
        rationale = (
            "Типичная клиника ишемии при нормальном тропонине и без "
            "выраженных изменений ЭКГ - клиническая картина НС."
        )
        needs_repeat = True

    else:
        label = AcsDiagnosis.OKS_UNLIKELY
        confidence = 0.55
        rationale = (
            "Не выявлено убедительных критериев ОКС: нет диагностически значимых "
            "изменений ЭКГ, тропонин в норме, клиника не типична."
        )
        needs_repeat = False
        if not fired:
            fired.append("Нет ЭКГ- и биомаркер-критериев острой коронарной ишемии.")

    return DiagnosisResult(
        label=label,
        confidence=confidence,
        criteria_fired=fired,
        icd10_suggested=list(ICD_SUGGESTIONS.get(label, [])),
        needs_repeat_troponin=needs_repeat,
        rationale=rationale,
    )


def merge_diagnoses(*candidates: AcsDiagnosis | None) -> AcsDiagnosis:
    """Выбрать наиболее приоритетный диагноз из набора кандидатов.

    Используется при наличии нескольких источников (классификатор + правила
    + явный МКБ-диагноз).
    """
    valid = [c for c in candidates if isinstance(c, AcsDiagnosis)]
    if not valid:
        return AcsDiagnosis.OKS_UNLIKELY
    return min(valid, key=lambda item: DIAGNOSIS_PRIORITY.get(item, 99))


__all__ = [
    "AcsDiagnosis",
    "DiagnosisResult",
    "DIAGNOSIS_PRIORITY",
    "ICD_SUGGESTIONS",
    "TROPONIN_99_PERCENTILE",
    "TROPONIN_ELEVATED",
    "TROPONIN_HIGH",
    "classify_acs",
    "merge_diagnoses",
]
