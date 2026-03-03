#!/usr/bin/env bash
set -euo pipefail

MODE="single"
MODEL="qwen2.5:7b-instruct"
MODEL_B="qwen2.5:3b-instruct"
NAME="Ivan"
PAIN_TYPE="typical"
ECG_CHANGES="ST-depression"
TROPONIN="0.12"
HR="102"
BP="130/85"
SYMPTOMS_TEXT="давящая боль в груди 30 минут"
OUTPUT="data/last_result.json"
REQUIRE_LLM=0
FORCE_LLM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-b) MODEL_B="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --pain-type) PAIN_TYPE="$2"; shift 2 ;;
    --ecg-changes) ECG_CHANGES="$2"; shift 2 ;;
    --troponin) TROPONIN="$2"; shift 2 ;;
    --hr) HR="$2"; shift 2 ;;
    --bp) BP="$2"; shift 2 ;;
    --symptoms-text) SYMPTOMS_TEXT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --require-llm) REQUIRE_LLM=1; shift ;;
    --force-llm) FORCE_LLM=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

CMD=(
  "$PYTHON_BIN" -m src.cli.main
  --mode "$MODE"
  --model "$MODEL"
  --model-b "$MODEL_B"
  --name "$NAME"
  --pain-type "$PAIN_TYPE"
  --ecg-changes "$ECG_CHANGES"
  --troponin "$TROPONIN"
  --hr "$HR"
  --bp "$BP"
  --symptoms-text "$SYMPTOMS_TEXT"
  --output "$OUTPUT"
)

if [[ "$REQUIRE_LLM" -eq 1 ]]; then
  CMD+=(--require-llm)
fi
if [[ "$FORCE_LLM" -eq 1 ]]; then
  CMD+=(--force-llm)
fi

echo "Running assessment with mode=$MODE model=$MODEL"
"${CMD[@]}"
