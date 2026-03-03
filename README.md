# LangGraph-ACS-Diagnosis

Гибридная экспертная система диагностики острого коронарного синдрома (ОКС) на базе LangGraph + локальной LLM + RAG. Прозрачная, объяснимая, работает полностью локально.

## Overview

Этот проект представляет собой прототип агентной системы для первичной оценки риска острого коронарного синдрома (ОКС) на основе современных технологий ИИ (2026 год). Основная цель — автоматизировать диагностику на основе медицинских данных пациента (симптомы, ЭКГ, тропонин, ЧСС, АД), используя комбинацию rule-based логики, языковых моделей (LLM) и Retrieval-Augmented Generation (RAG) для интеграции с клиническими рекомендациями (ESC/AHA 2025–2026).

**Ключевые особенности:**
- **Прозрачность и объяснимость**: LangGraph обеспечивает traceability каждого шага (логи state), LLM генерирует обоснования с цитатами из гайдлайнов.
- **Локальность**: Всё работает на одной машине без интернета (Ollama для LLM, Chroma для RAG, CSV для хранения пациентов).
- **Гибридный подход**: Hard-rules для критических случаев (STEMI), LLM для неструктурированных данных, RAG для актуальных знаний.
- **Масштабируемость**: Поддержка большого количества пациентов (поиск, история), легко добавить ML-модели (XGBoost для HEART/GRACE scores).
- **Не для реального использования**: Это демонстрационный прототип. Не заменяет врача; добавьте disclaimers в production.

Проект эволюционировал от rule-based системы на Drools (см. оригинальный репозиторий [Dnoniel-Ermolaev/Drools_Diplom](https://github.com/Dnoniel-Ermolaev/Drools_Diplom)) к агентной архитектуре для большей гибкости и обобщения.

## Architecture

Система построена как агентный workflow на LangGraph...

Вот высокоуровневая диаграмма:

```mermaid
graph TD
    A[User → Streamlit] --> B[LangGraph Workflow]
    B --> C[Parse Input]
    C --> D[Rule Check]
    D -->|Obvious High Risk| E[Output & Save]
    D -->|Unclear → LLM| F[RAG Retrieval]
    F --> G[LLM Assessment]
    G --> E
    E --> B[Display Result]

    I[patients.csv] <--> E
    J[Guidelines DB] <--> F
    K[Ollama LLM] <--> G
```

### Компоненты:
- **LangGraph**: Оркестратор. State: TypedDict с полями (patient_data: dict, rag_context: str, db_results: list, risk: float, explanation: str). Узлы: parse_input, rule_check, rag_retrieval, llm_assess, output_save. Рёбра: conditional (на основе риска).
- **LLM**: Локальная модель через Ollama (рекомендуется `qwen2.5:7b-instruct` для стабильного JSON; для A/B можно сравнивать с `medgemma:latest` (отсутствует в ollama)). Используется для: парсинга текста, оценки риска, генерации объяснений.
- **RAG**: Для интеграции гайдлайнов (ESC/AHA). Эмбеддер: sentence-transformers.all-MiniLM-L6-v2. Векторная БД: Chroma (локально). Retrieval: similarity_search с top-k=3.
- **База пациентов**: CSV-файл (patients.csv) через Pandas. Столбцы: id, name, symptoms_json, ecg_desc, troponin, hr, bp, risk_level, explanation, timestamp. Операции: append, search (pandas.query + семантический через FAISS для симптомов).
- **UI**: Streamlit для веб-интерфейса. Страницы: ввод данных, оценка риска, история пациентов (с фильтрами и поиском).
- **Tools/Интеграции**: В LangGraph — tools для DB search (pandas), RAG query, HEART/GRACE calculators (pure Python functions).

### Данные пациента (пример схемы):
```python
from pydantic import BaseModel

class PatientData(BaseModel):
    name: str
    pain_type: str  # "typical", "atypical", "none"
    ecg_changes: str  # "ST-elevation", "ST-depression", "normal"
    troponin: float
    hr: int  # heart rate
    bp: str  # "120/80"
    # Другие поля...
```

## Installation

### Требования:
- Python 3.10+ (рекомендуется 3.12 для Ollama).
- GPU: Опционально (для ускорения LLM; минимально RTX 3060 для 7B-моделей).
- Зависимости: Установите через `requirements.txt`.

```bash
# Клонируйте репозиторий
git clone https://github.com/your-username/LangGraph-ACS-Diagnosis.git
cd LangGraph-ACS-Diagnosis

# Создайте виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Или .venv\Scripts\activate на Windows

# Установите зависимости
pip install -r requirements.txt

# requirements.txt пример:
langgraph
langchain
langchain-community  # Для Ollama, Chroma
streamlit
pandas
sentence-transformers
pydantic
ollama  # Если не установлен глобально
```

- **Ollama**: Установите локально (ollama.com). Скачайте модели:
  - `ollama pull qwen2.5:7b-instruct`
  - `ollama pull medgemma:latest`
- **Гайдлайны для RAG**: Скачайте PDF (ESC 2025 ACS guidelines) в `data/guidelines/`. Запустите скрипт `src/infrastructure/rag_setup.py` для создания векторной БД.

## Usage

### Локальный запуск (console-first):
1. Создайте seed для RAG (один раз): `python -m src.infrastructure.rag.rag_setup`.
2. Опционально запустите Ollama: `ollama serve` (если не запущен, система перейдёт в fallback режим без LLM).
3. Запустите оценку из консоли (single режим):
   `python -m src.cli.main --mode single --model qwen2.5:7b-instruct --name Ivan --pain-type typical --ecg-changes "ST-depression" --troponin 0.12 --hr 102 --bp 130/85 --symptoms-text "давящая боль в груди 30 минут" --output data/last_result.json --require-llm`
4. Для A/B сравнения двух моделей на одном кейсе:
   `python -m src.cli.main --mode ab --model qwen2.5:7b-instruct --model-b medgemma:latest --name Ivan --pain-type typical --ecg-changes "ST-depression" --troponin 0.12 --hr 102 --bp 130/85 --symptoms-text "давящая боль в груди 30 минут" --output data/ab_result.json --require-llm --force-llm`
5. На выходе будет JSON с риском, объяснением, `record_id`, `llm_used` и названием модели; запись сохранится в `data/patients.csv`, а также в файл `--output`.
6. Web UI можно добавить отдельно (например, на Streamlit), используя тот же `src.core.graph`.
7. Если `langgraph`/`pandas` не установлены, доступен упрощённый fallback-пайплайн (invoke + CSV) для быстрого старта.
8. Для принудительного прохода через LLM даже в high-risk rule-cases используйте `--force-llm`.
9. Для строго LLM-режима используйте `--require-llm` (если Ollama недоступен, команда завершится с ошибкой вместо fallback).

### Быстрые скрипты (PowerShell):
- Запуск оценки: `.\scripts\run.ps1 -RequireLlm -ForceLlm`
- A/B сравнение: `.\scripts\run.ps1 -Mode ab -Model qwen2.5:7b-instruct -ModelB qwen2.5:3b-instruct -RequireLlm -ForceLlm -Output data/ab_result.json`
- Запуск тестов: `.\scripts\test.ps1`
- Установка зависимостей + тесты: `.\scripts\test.ps1 -InstallDeps`

### Быстрые скрипты (Bash, Linux/macOS/Windows через Git Bash/WSL):
- Запуск оценки: `./scripts/run.sh --require-llm --force-llm`
- A/B сравнение: `./scripts/run.sh --mode ab --model qwen2.5:7b-instruct --model-b qwen2.5:3b-instruct --require-llm --force-llm --output data/ab_result.json`
- Запуск тестов: `./scripts/test.sh`
- Установка зависимостей + тесты: `./scripts/test.sh --install-deps`

### Пример вызова графа из кода (для тестов):
```python
from src.core.graph import graph  # Compiled LangGraph

input_data = {"patient_data": {"name": "Ivan", "pain_type": "typical", "troponin": 0.2}}
result = graph.invoke(input_data)
print(result["risk"], result["explanation"])
```

## Structure

```
LangGraph-ACS-Diagnosis/
├── data/                       # Данные
│   ├── guidelines/             # PDF/TXT гайдлайнов для RAG
│   └── patients.csv            # База пациентов
├── src/
│   ├── core/                   # Логика агента
│   │   ├── state.py            # AgentState (TypedDict)
│   │   ├── nodes.py            # Узлы графа (parse_input, rule_check и т.д.)
│   │   ├── prompts.py          # Системные промпты для LLM
│   │   ├── tools.py            # Tools (db_search, rag_query, score_calculators)
│   │   └── graph.py            # Сборка и compile графа
│   ├── medical/                # Медицинская логика
│   │   ├── rules.py            # Hard-rules (STEMI, etc.)
│   │   └── scores.py           # HEART, GRACE, TIMI функции
│   ├── infrastructure/         # Инфраструктура
│   │   ├── db/                 # CSV операции (Pandas)
│   │   │   ├── models.py       # Pydantic модели для пациентов
│   │   │   └── repository.py   # CRUD: save_patient, search_patients
│   │   └── rag/                # RAG setup/retrieval
│   └── cli/                    # Console entrypoint
│       └── main.py             # python -m src.cli.main
├── tests/                      # Тесты
│   ├── unit/                   # Тесты nodes, tools (pytest)
│   └── integration/            # Тесты графа + UI
├── .env.example                # Для ключей (если добавите API)
├── requirements.txt
├── Dockerfile                  # Опционально для контейнеризации
└── README.md
```

## Contributing / Development

Чтобы развивать проект:
1. **Добавьте узел в граф**: В `nodes.py` — новая функция, в `graph.py` — add_node и add_edge.
2. **Улучшите RAG**: Добавьте больше гайдлайнов в `data/` → rerun `rag_setup.py`.
3. **Интегрируйте ML**: В `tools.py` — XGBoost для scores; обучите на датасетах (Kaggle ACS).
4. **Тесты**: `pytest tests/` — покрытие ≥80% для nodes и tools.
5. **Метрики**: Добавьте evaluation скрипт (AUC, Sensitivity) на синтетических данных (SDV библиотека).
6. **Дорожная карта**:
   - Короткий срок: Добавить multi-agent (отдельный агент для ЭКГ-анализа с vision LLM).
   - Средний: Перейти с CSV на SQLite/PostgreSQL для scalability.
   - Долгий: Fine-tune LLM на медицинских датасетах (MIMIC-IV); интеграция с реальными EHR API.
   - Улучшения: Human-in-the-loop (breakpoint в графе), bias checks (SHAP).


## License

MIT License. См. LICENSE файл.

## Disclaimers

- Это не медицинское ПО. Не используйте для реальной диагностики — только для образовательных/исследовательских целей.
- Соответствуйте регуляциям (HIPAA/GDPR для данных пациентов).
- Источники: Опирайтесь на актуальные гайдлайны; проект не несёт ответственности за ошибки.
