#!/usr/bin/env bash
# Create the plugin virtualenv and a local config.env.
# Works on Linux, macOS, WSL, and Git Bash. On a Windows drive under WSL
# (/mnt/<letter>/...), prefers Windows Python so Cursor Desktop can run hooks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
MIN_MINOR=10

die() {
  echo "install.sh: $*" >&2
  exit 1
}

info() {
  echo "install.sh: $*"
}

version_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, '"$MIN_MINOR"') else 1)' 2>/dev/null
}

is_wsl() {
  [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null
}

on_windows_mount() {
  case "$ROOT" in
    /mnt/[a-zA-Z]/*|/cygdrive/[a-zA-Z]/*) return 0 ;;
  esac
  return 1
}

wsl_path_to_win() {
  # /mnt/f/AI-Code/foo -> F:\AI-Code\foo
  local p="$1"
  if [[ "$p" =~ ^/mnt/([a-zA-Z])/(.*)$ ]]; then
    local drive
    drive="$(echo "${BASH_REMATCH[1]}" | tr '[:lower:]' '[:upper:]')"
    local rest="${BASH_REMATCH[2]}"
    rest="${rest//\//\\}"
    printf '%s:\\%s\n' "$drive" "$rest"
    return 0
  fi
  printf '%s\n' "$p"
}

# Resolve pip/python inside a venv (Unix bin/ or Windows Scripts/).
venv_python() {
  local cand
  for cand in \
    "$ROOT/.venv/Scripts/python.exe" \
    "$ROOT/.venv/bin/python" \
    "$ROOT/.venv/bin/python3"
  do
    if [[ -x "$cand" ]] || [[ -f "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

win_path_to_wsl() {
  # C:\Users\foo -> /mnt/c/Users/foo
  local p="$1"
  p="${p//$'\r'/}"
  p="${p//\\//}"
  if [[ "$p" =~ ^([A-Za-z]):/(.*)$ ]]; then
    local drive
    drive="$(echo "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
    printf '/mnt/%s/%s\n' "$drive" "${BASH_REMATCH[2]}"
    return 0
  fi
  printf '%s\n' "$p"
}

find_windows_python() {
  local cand converted
  shopt -s nullglob
  local -a candidates=(
    /mnt/c/Windows/py.exe
    /mnt/c/Users/*/AppData/Local/Programs/Python/Python3*/python.exe
    /mnt/d/Users/*/AppData/Local/Programs/Python/Python3*/python.exe
    /mnt/c/Python3*/python.exe
    /mnt/c/Program\ Files/Python3*/python.exe
  )
  shopt -u nullglob
  for cand in "${candidates[@]}"; do
    [[ -f "$cand" ]] || continue
    if [[ "$(basename "$cand")" == py.exe ]]; then
      converted="$("$cand" -3 -c 'import sys; print(sys.executable)' 2>/dev/null | tr -d '\r' || true)"
      [[ -n "$converted" ]] || continue
      cand="$(win_path_to_wsl "$converted")"
    fi
    if [[ -f "$cand" ]] && version_ok "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

find_unix_python() {
  local cand
  for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && version_ok "$(command -v "$cand")"; then
      command -v "$cand"
      return 0
    fi
  done
  return 1
}

bootstrap_pip() {
  local py="$1"
  local tmp
  tmp="$(mktemp)"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$GET_PIP_URL" -o "$tmp"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tmp" "$GET_PIP_URL"
  else
    rm -f "$tmp"
    die "need curl or wget to bootstrap pip (ensurepip is missing)"
  fi
  "$py" "$tmp"
  rm -f "$tmp"
}

create_venv() {
  local py="$1"
  shift
  local venv_path="$ROOT/.venv"
  # Windows Python needs a Win32 path, not /mnt/<drive>/...
  case "$py" in
    *.exe|*/Python*/python|*/python.exe)
      venv_path="$(wsl_path_to_win "$venv_path")"
      ;;
  esac
  # Prefer a full venv; fall back when distro packages omit ensurepip.
  if "$py" -m venv "$@" "$venv_path"; then
    return 0
  fi
  info "venv with pip failed; retrying without pip..."
  rm -rf "$ROOT/.venv"
  "$py" -m venv --without-pip "$@" "$venv_path"
  local vpy
  vpy="$(venv_python)" || die "venv created but python was not found"
  bootstrap_pip "$vpy"
}

# --- select interpreter -------------------------------------------------------

PYTHON=""
VENV_ARGS=()

if is_wsl && on_windows_mount; then
  if WIN_PY="$(find_windows_python)"; then
    PYTHON="$WIN_PY"
    info "WSL on Windows drive - using Windows Python: $PYTHON"
  else
    info "WSL on Windows drive but no Windows Python >=3.$MIN_MINOR found; using Linux Python with --copies"
  fi
fi

if [[ -z "$PYTHON" ]]; then
  PYTHON="$(find_unix_python)" || die "need Python 3.$MIN_MINOR+ (python3) on PATH"
  if on_windows_mount || [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
    VENV_ARGS+=(--copies)
  fi
fi

PY_VER="$("$PYTHON" -c 'import sys; print(sys.version.split()[0])' | tr -d '\r')"
info "Python: $PYTHON ($PY_VER)"

# --- recreate venv ------------------------------------------------------------

rm -rf "$ROOT/.venv"
if ((${#VENV_ARGS[@]})); then
  create_venv "$PYTHON" "${VENV_ARGS[@]}"
else
  create_venv "$PYTHON"
fi

VPY="$(venv_python)" || die "virtualenv python missing after create"
info "venv: $VPY"

REQ="$ROOT/requirements.txt"
case "$VPY" in
  *.exe)
    REQ="$(wsl_path_to_win "$REQ")"
    ;;
esac

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r "$REQ"

if [[ ! -f "$ROOT/config.env" ]]; then
  cp "$ROOT/config.env.example" "$ROOT/config.env"
  echo "Wrote $ROOT/config.env - set MLFLOW_TRACKING_URI and MLFLOW_TRACKING_PASSWORD."
fi

echo "Plugin ready: $ROOT"
echo "Hook runner: $VPY"
