#!/usr/bin/env python3
"""Record Cursor agent steps and upload one MLflow trace per turn.

Cursor starts a new process for every hook, and MLflow can only attach child
spans to a parent that still lives in that process. Each event is appended to
a local JSONL file. ``stop`` and ``sessionEnd`` upload the whole turn as one
trace under experiment 1. Concurrent flushes share a lock so the same buffer is
never uploaded twice, and a stop that arrives before any real agent work is
deferred so the turn stays one session-linked trace.

A Cursor subagent is its own conversation. Its steps are stored beside the
parent and folded into one AGENT span when the parent turn is uploaded, which
is the MLflow shape for a nested agent. The subagent's own stop does not
upload a second trace.

Stdout is reserved for the Cursor hook response. Diagnostics go to the
workspace ``.cursor/mlflow/cursor_tracing.log``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from plugin_config import state_dir, venv_python, workspace_root

STATE_DIR = state_dir()
EVENTS_DIR = STATE_DIR / "events"
CHILDREN_DIR = STATE_DIR / "children"
SUBAGENTS_DIR = STATE_DIR / "subagents"
LOG_FILE = STATE_DIR / "cursor_tracing.log"
UPLOADER = Path(__file__).with_name("mlflow_upload.py")
VENV_PYTHON = venv_python()

FLUSH_EVENTS = {"stop", "sessionEnd"}
# stop can run before afterAgentResponse lands. Wait so the reply is in this turn.
FLUSH_SETTLE_SECONDS = 0.6
MAX_STRING = 8000
MAX_LIST = 30
MAX_DICT = 40
MAX_DEPTH = 6

_SECRET = re.compile(
    r"(?i)((?:api[_-]?key|password|secret|token|authorization)\s*[:=]\s*(?:bearer\s+)?)(\S+)"
)
_SK = re.compile(r"sk-[A-Za-z0-9]{12,}")


@contextmanager
def exclusive_file_lock(lock_path: Path):
    """Cross-platform exclusive lock via a sibling lock file."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    lock_handle.seek(0)
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_handle.seek(0)
                msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)


def log(message: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except OSError:
        pass


def redact(text: str) -> str:
    text = _SECRET.sub(r"\1[redacted]", text)
    return _SK.sub("sk-[redacted]", text)


def clip(value, depth: int = 0):
    if depth > MAX_DEPTH:
        return "..."
    if isinstance(value, str):
        value = redact(value)
        if len(value) > MAX_STRING:
            hidden = len(value) - MAX_STRING
            return value[:MAX_STRING] + f"...<{hidden} chars truncated>"
        return value
    if isinstance(value, dict):
        clipped = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_DICT:
                clipped["..."] = f"<{len(value) - MAX_DICT} more keys>"
                break
            clipped[str(key)[:200]] = clip(item, depth + 1)
        return clipped
    if isinstance(value, list):
        items = [clip(item, depth + 1) for item in value[:MAX_LIST]]
        if len(value) > MAX_LIST:
            items.append(f"...<{len(value) - MAX_LIST} more>")
        return items
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return clip(str(value), depth + 1)


def conversation_id(payload: dict) -> str:
    raw = (
        payload.get("conversation_id")
        or payload.get("session_id")
        or payload.get("generation_id")
        or "unknown"
    )
    return safe_id(raw)


def safe_id(raw: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(raw or ""))
    return safe or "unknown"


def events_path(conv_id: str) -> Path:
    return EVENTS_DIR / f"{conv_id}.jsonl"


def child_events_path(subagent_id: str) -> Path:
    return CHILDREN_DIR / f"{safe_id(subagent_id)}.jsonl"


def registry_path(subagent_id: str) -> Path:
    return SUBAGENTS_DIR / f"{safe_id(subagent_id)}.json"


def load_subagents() -> list[dict]:
    if not SUBAGENTS_DIR.is_dir():
        return []
    found = []
    for path in SUBAGENTS_DIR.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(item, dict) and item.get("subagent_id"):
            found.append(item)
    return found


def write_subagent(record: dict) -> None:
    SUBAGENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = registry_path(str(record["subagent_id"]))
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def register_subagent(payload: dict) -> None:
    subagent_id = payload.get("subagent_id")
    if not subagent_id:
        return
    parent = payload.get("parent_conversation_id") or payload.get("conversation_id") or payload.get("session_id")
    write_subagent(
        {
            "subagent_id": str(subagent_id),
            "parent_conversation_id": str(parent or ""),
            "bound_conversation_id": None,
            "task": payload.get("task") or "",
            "subagent_type": payload.get("subagent_type") or "",
            "tool_call_id": payload.get("tool_call_id") or "",
            "model": payload.get("subagent_model") or payload.get("model") or "",
            "open": True,
            "started_ns": time.time_ns(),
        }
    )
    log(f"registered subagent {subagent_id} under {parent}")


def resolve_subagent(payload: dict) -> dict | None:
    """Return the subagent record this event belongs to, binding it if needed.

    Cursor runs the subagent as a separate conversation. ``subagent_id`` often
    is that conversation id. When it is not, the only open unbound subagent is
    bound to the new conversation id.
    """
    conv_id = conversation_id(payload)
    records = load_subagents()
    for record in records:
        if safe_id(record.get("subagent_id")) == conv_id or safe_id(record.get("bound_conversation_id")) == conv_id:
            return record

    parents = {safe_id(record.get("parent_conversation_id")) for record in records}
    if conv_id in parents:
        return None

    unbound = [
        record
        for record in records
        if record.get("open") and not record.get("bound_conversation_id")
    ]
    if len(unbound) != 1:
        return None
    chosen = unbound[0]
    chosen["bound_conversation_id"] = str(payload.get("conversation_id") or payload.get("session_id") or conv_id)
    write_subagent(chosen)
    log(f"bound conversation {chosen['bound_conversation_id']} to subagent {chosen['subagent_id']}")
    return chosen


def close_subagent(payload: dict) -> None:
    records = load_subagents()
    task = payload.get("task") or ""
    kind = payload.get("subagent_type") or ""
    for record in records:
        if not record.get("open"):
            continue
        same_task = not task or record.get("task") == task
        same_kind = not kind or record.get("subagent_type") == kind
        if same_task and same_kind:
            record["open"] = False
            write_subagent(record)
            return
    open_records = [record for record in records if record.get("open")]
    if len(open_records) == 1:
        open_records[0]["open"] = False
        write_subagent(open_records[0])


def forget_subagents(subagent_ids: list[str]) -> None:
    for subagent_id in subagent_ids:
        child_events_path(subagent_id).unlink(missing_ok=True)
        registry_path(subagent_id).unlink(missing_ok=True)


def subagent_ids_in(path: Path) -> list[str]:
    found = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            continue
        if event.get("event_type") == "subagentStart" and payload.get("subagent_id"):
            found.append(str(payload["subagent_id"]))
    return found


def hook_response(event_name: str) -> dict:
    if event_name == "beforeSubmitPrompt":
        return {"continue": True}
    return {}


def append_event_at(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        existing = []
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    existing.append(item)
        from mlflow_upload import should_record

        if not should_record(existing, event):
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def append_event(conv_id: str, event: dict) -> None:
    append_event_at(events_path(conv_id), event)


def restore_events(uploading: Path, dest: Path) -> None:
    if not uploading.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        os.replace(uploading, dest)
        return
    with exclusive_file_lock(dest.with_suffix(dest.suffix + ".lock")):
        with dest.open("a", encoding="utf-8") as handle:
            handle.write(uploading.read_text(encoding="utf-8"))
            handle.flush()
    uploading.unlink(missing_ok=True)


def flush_conversation(conv_id: str) -> None:
    time.sleep(FLUSH_SETTLE_SECONDS)
    path = events_path(conv_id)
    uploading = path.with_suffix(".jsonl.uploading")
    with exclusive_file_lock(path.with_suffix(".jsonl.flushlock")):
        if path.exists():
            try:
                os.replace(path, uploading)
            except FileNotFoundError:
                return
        elif uploading.exists():
            # Another hook (stop/sessionEnd) already claimed this turn.
            log(f"skip duplicate flush for {conv_id}; upload already in progress")
            return
        else:
            return

        if not VENV_PYTHON.is_file():
            log(f"mlflow client missing at {VENV_PYTHON}; kept {uploading.name}")
            restore_events(uploading, path)
            return

        try:
            completed = subprocess.run(
                [str(VENV_PYTHON), str(UPLOADER), str(uploading)],
                capture_output=True,
                text=True,
                timeout=75,
                cwd=str(workspace_root()),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"upload failed for {conv_id}: {exc}")
            restore_events(uploading, path)
            return

        stdout = completed.stdout.strip()
        if stdout:
            log(stdout)
        if completed.stderr.strip():
            log(completed.stderr.strip()[-2000:])
        if completed.returncode != 0:
            log(f"upload exited {completed.returncode} for {conv_id}")
            restore_events(uploading, path)
            return
        if stdout.endswith("defer incomplete turn") or stdout == "defer incomplete turn":
            log(f"deferred incomplete turn for {conv_id}")
            restore_events(uploading, path)
            return
        if stdout in {"", "no agent steps to upload"} or stdout.endswith("no agent steps to upload"):
            uploading.unlink(missing_ok=True)
            return
        forget_subagents(subagent_ids_in(uploading))
        uploading.unlink(missing_ok=True)


def windows_acp() -> int:
    if sys.platform != "win32":
        return 65001
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetACP())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 65001


def repair_windows_mojibake(text: str) -> str:
    """Undo Cursor's Windows hook bug: UTF-8 bytes misread as ACP, then re-emitted as Unicode.

    On CP936/GBK (and other non-UTF-8 ACPs), Chinese/Korean/etc. in hook stdin arrives
    double-encoded. Round-trip through the ANSI code page restores the original UTF-8 text
    when the corruption was lossless. No-op on UTF-8 ACP or ASCII-only strings.
    """
    if not text or sys.platform != "win32" or windows_acp() == 65001:
        return text
    if not any(ord(ch) > 127 for ch in text):
        return text
    try:
        repaired = text.encode("mbcs").decode("utf-8")
    except UnicodeError:
        return text
    if repaired == text:
        return text
    try:
        # Accept only exact double-encoding (avoids mangling already-correct text).
        if repaired.encode("utf-8").decode("mbcs") == text:
            return repaired
    except UnicodeError:
        return text
    return text


def decode_hook_stdin(raw_bytes: bytes) -> str:
    """Decode Cursor hook stdin; always strip UTF-8 BOM; tolerate UTF-16 shells."""
    if not raw_bytes:
        return ""
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw_bytes.decode("utf-16")
    # UTF-16LE without BOM (NUL in the first ASCII JSON braces area).
    if len(raw_bytes) >= 4 and raw_bytes[1] == 0 and raw_bytes[0:1] in (b"{", b"[", b"\xef"):
        if not raw_bytes.startswith(b"\xef\xbb\xbf"):
            try:
                return raw_bytes.decode("utf-16-le")
            except UnicodeDecodeError:
                pass
    return raw_bytes.decode("utf-8-sig", errors="replace").lstrip("\ufeff")


def parse_hook_payload(raw: str) -> dict:
    text = raw.strip().lstrip("\ufeff")
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload, _end = json.JSONDecoder().raw_decode(text)
    if not isinstance(payload, dict):
        raise ValueError("hook payload is not an object")
    return payload


def _repair_payload_strings(payload: dict) -> dict:
    """Repair double-encoded strings anywhere in the payload tree."""

    def fix(value):
        if isinstance(value, str):
            return repair_windows_mojibake(value)
        if isinstance(value, dict):
            return {str(k): fix(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(item) for item in value]
        return value

    return fix(payload)


def _extract_ascii_field(raw: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _temp_dirs() -> list[Path]:
    dirs: list[Path] = []
    for key in ("TEMP", "TMP"):
        value = os.environ.get(key)
        if value:
            dirs.append(Path(value))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Temp")
    # Preserve order, drop missing/dupes.
    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def load_payload_from_cursor_temp(conversation_id_hint: str = "", max_age_sec: float = 15.0) -> dict | None:
    """Read Cursor's UTF-8 hook temp file when PowerShell-corrupted stdin is unusable."""
    now = time.time()
    newest: tuple[float, Path] | None = None
    for folder in _temp_dirs():
        try:
            candidates = folder.glob("cursor-hook-payload-*.json")
        except OSError:
            continue
        for path in candidates:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if now - mtime > max_age_sec:
                continue
            if newest is None or mtime > newest[0]:
                newest = (mtime, path)
            if not conversation_id_hint:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue
            if not isinstance(payload, dict):
                continue
            cid = str(payload.get("conversation_id") or payload.get("session_id") or "")
            if cid == conversation_id_hint:
                return payload
    if newest is None:
        return None
    try:
        payload = json.loads(newest[1].read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def recover_prompt_from_transcript(transcript_path: str) -> str:
    """Last-resort: read the latest user_query from the UTF-8 transcript file."""
    path = Path(transcript_path)
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("role") != "user":
            continue
        message = row.get("message") or {}
        content = message.get("content")
        chunks: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text") or ""))
        elif isinstance(content, str):
            chunks.append(content)
        text = "\n".join(chunks)
        match = re.search(r"<user_query>\s*([\s\S]*?)\s*</user_query>", text)
        if match:
            return match.group(1).strip()
        if text.strip():
            return text.strip()
    return ""


def load_hook_payload(raw_bytes: bytes) -> dict:
    """Decode stdin JSON, reverse Windows ACP mojibake, fall back to Cursor temp/transcript."""
    raw = decode_hook_stdin(raw_bytes)
    candidates = [raw]
    repaired_whole = repair_windows_mojibake(raw)
    if repaired_whole != raw:
        candidates.insert(0, repaired_whole)

    last_error: Exception | None = None
    payload: dict | None = None
    for candidate in candidates:
        try:
            payload = parse_hook_payload(candidate)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    conv_hint = _extract_ascii_field(raw, "conversation_id") or _extract_ascii_field(
        raw, "session_id"
    )
    if payload is None:
        temp_payload = load_payload_from_cursor_temp(conv_hint)
        if temp_payload is not None:
            log(f"recovered hook payload from Cursor temp file conv={conv_hint or '?'}")
            payload = temp_payload
        elif last_error is not None:
            raise last_error
        else:
            raise ValueError("hook payload missing")

    payload = _repair_payload_strings(payload)

    event_name = str(
        payload.get("hook_event_name")
        or payload.get("event_name")
        or payload.get("hookEventName")
        or ""
    )
    # Prompt still missing/garbled after repair: transcript file is valid UTF-8.
    if event_name == "beforeSubmitPrompt" and _prompt_needs_transcript(payload.get("prompt")):
        transcript = (
            str(payload.get("transcript_path") or "")
            or os.environ.get("CURSOR_TRANSCRIPT_PATH")
            or ""
        )
        if transcript:
            recovered = recover_prompt_from_transcript(transcript)
            if recovered:
                payload["prompt"] = recovered
                log("recovered prompt from transcript_path")

    return payload


def _prompt_needs_transcript(prompt: object) -> bool:
    if not isinstance(prompt, str) or not prompt.strip():
        return True
    if any(ord(ch) == 0xFFFD or 0xE000 <= ord(ch) <= 0xF8FF for ch in prompt):
        return True
    if windows_acp() == 65001 or not any(ord(ch) > 127 for ch in prompt):
        return False
    # Unrepaired GBK-misread-of-UTF-8 fragments common on Chinese Windows.
    markers = ("鎻", "鍦", "娴", "锟", "銆", "姝", "鑳", "闂", "鍦ㄧ", "娴嬭")
    return any(marker in prompt for marker in markers)


def _debug_dump_stdin(raw: bytes) -> None:
    try:
        path = STATE_DIR / "last_stdin.bin"
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw[:16384])
    except OSError:
        pass


def main() -> int:
    raw_bytes = sys.stdin.buffer.read()
    if not raw_bytes:
        log("ignored empty hook stdin")
        print("{}")
        return 0

    try:
        payload = load_hook_payload(raw_bytes)
    except (json.JSONDecodeError, ValueError) as exc:
        _debug_dump_stdin(raw_bytes)
        raw = decode_hook_stdin(raw_bytes)
        preview = raw[:120].replace("\n", "\\n").replace("\r", "\\r")
        log(f"ignored bad hook input ({len(raw_bytes)} bytes): {exc}; {preview!r}")
        print("{}")
        return 0

    event_name = str(
        payload.get("hook_event_name")
        or payload.get("event_name")
        or payload.get("hookEventName")
        or ""
    )
    if not event_name:
        _debug_dump_stdin(raw_bytes)
        log(f"ignored hook payload without event name keys={list(payload)[:20]}")
        print("{}")
        return 0

    log(f"hook {event_name} conv={conversation_id(payload)}")
    try:
        event = {
            "event_type": event_name,
            "timestamp_ns": time.time_ns(),
            "payload": clip(payload),
        }
        if event_name == "subagentStart":
            register_subagent(payload)
            append_event(conversation_id(payload), event)
        elif event_name == "subagentStop":
            close_subagent(payload)
            append_event(conversation_id(payload), event)
        elif (subagent := resolve_subagent(payload)) is not None:
            append_event_at(child_events_path(str(subagent["subagent_id"])), event)
            if event_name in FLUSH_EVENTS:
                log(f"deferred subagent {subagent['subagent_id']} until the parent turn uploads")
        else:
            conv_id = conversation_id(payload)
            append_event(conv_id, event)
            if event_name in FLUSH_EVENTS:
                flush_conversation(conv_id)
    except Exception as exc:  # noqa: BLE001 — hooks must fail open
        log(f"{event_name} failed: {exc}")

    print(json.dumps(hook_response(event_name)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        log(f"hook crashed: {exc}")
        print("{}")
        raise SystemExit(0)
