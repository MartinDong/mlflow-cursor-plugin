#!/usr/bin/env bash
# Resolve plugin venv python (or system python3) and run the hook.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/mlflow_cursor.py"
for cand in \
  "$ROOT/.venv/Scripts/python.exe" \
  "$ROOT/.venv/bin/python" \
  "$ROOT/.venv/bin/python3"
do
  if [[ -x "$cand" || -f "$cand" ]]; then
    exec "$cand" "$SCRIPT"
  fi
done
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$SCRIPT"
fi
if command -v python >/dev/null 2>&1; then
  exec python "$SCRIPT"
fi
echo "run_hook.sh: no Python found; run install.sh or install.ps1" >&2
exit 1
