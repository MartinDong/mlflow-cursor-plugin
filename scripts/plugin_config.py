"""Paths and tracking settings for the portable Cursor plugin.

State stays in the workspace (``.cursor/mlflow``). Credentials come from, in
order: process environment, ``config.env`` next to this plugin, then a
workspace ``mlflow/.env`` that uses the server's admin key names.
"""

from __future__ import annotations

import os
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PLUGIN_ROOT / "config.env"


def workspace_root() -> Path:
    for key in ("MLFLOW_CURSOR_WORKSPACE", "CURSOR_PROJECT_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw)
    return Path.cwd()


def state_dir() -> Path:
    return workspace_root() / ".cursor" / "mlflow"


def venv_python() -> Path:
    """Return the plugin venv interpreter (Unix ``bin/`` or Windows ``Scripts/``)."""
    candidates = (
        PLUGIN_ROOT / ".venv" / "Scripts" / "python.exe",
        PLUGIN_ROOT / ".venv" / "bin" / "python",
        PLUGIN_ROOT / ".venv" / "bin" / "python3",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[1]


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def tracking_settings() -> tuple[str, str, str, str]:
    """Return tracking URI, experiment id, username, and password.

    ``mlflow/.env`` in the workspace is the server file. Only its admin
    username and password are reused. Its ``MLFLOW_TRACKING_URI`` often points
    at a public address with the secret embedded, so the hook does not read it.
    """
    plugin_env = load_env(CONFIG_FILE)
    workspace_env = load_env(workspace_root() / "mlflow" / ".env")

    def pick(*sources: str, default: str = "") -> str:
        for source in sources:
            if source:
                return source
        return default

    uri = pick(
        os.environ.get("MLFLOW_TRACKING_URI", ""),
        plugin_env.get("MLFLOW_TRACKING_URI", ""),
        default="http://127.0.0.1:21103",
    )
    experiment = pick(
        os.environ.get("MLFLOW_EXPERIMENT_ID", ""),
        plugin_env.get("MLFLOW_EXPERIMENT_ID", ""),
        default="1",
    )
    username = pick(
        os.environ.get("MLFLOW_TRACKING_USERNAME", ""),
        plugin_env.get("MLFLOW_TRACKING_USERNAME", ""),
        workspace_env.get("MLFLOW_AUTH_ADMIN_USERNAME", ""),
        default="admin",
    )
    password = pick(
        os.environ.get("MLFLOW_TRACKING_PASSWORD", ""),
        plugin_env.get("MLFLOW_TRACKING_PASSWORD", ""),
        workspace_env.get("MLFLOW_AUTH_ADMIN_PASSWORD", ""),
    )
    return uri, experiment, username, password
