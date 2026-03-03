#!/usr/bin/env bash
set -euo pipefail

TARGET="tests/unit/test_rules.py"
INSTALL_DEPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
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

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  echo "Installing dependencies from requirements.txt"
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo "Running tests: $TARGET"
"$PYTHON_BIN" -m pytest "$TARGET"
