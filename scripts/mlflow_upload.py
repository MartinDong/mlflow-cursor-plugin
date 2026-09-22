#!/usr/bin/env python3
"""Upload one buffered Cursor turn to the configured MLflow tracking server.

Run with the plugin virtualenv:

    mlflow/cursor-plugin/.venv/bin/python mlflow/cursor-plugin/scripts/mlflow_upload.py <events.jsonl>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from plugin_config import state_dir, tracking_settings

CHILDREN_DIR = state_dir() / "children"
EXPERIMENT_ID = "1"

TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

# Cursor often redelivers the same thought with generation_id
# ``<uuid>`` then ``<uuid>-<step>-<suffix>`` a few ms later.
_GENERATION_ID = re.compile(
    r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"(?:-\d+-[A-Za-z0-9]+)?$"
)


def read_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def payload_of(event: dict) -> dict:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def dedupe_events(events: list[dict]) -> list[dict]:
    """Drop repeated hook deliveries of the same step.

    Cursor sends ``afterAgentThought`` twice, about 20ms apart, with the same
    text. It may also stream a longer rewrite of the same thought. A tool result
    can be delivered again with the same ``tool_use_id``; a later delivery
    replaces the earlier one so a streamed result keeps its final output.
    """
    text_at: dict[tuple, int] = {}
    tool_at: dict[tuple, int] = {}
    kept: list[dict] = []
    for event in events:
        text_key = _text_delivery_key(event)
        if text_key is not None:
            event_type, generation_id, text = text_key
            replaced = False
            for existing_key, index in list(text_at.items()):
                if existing_key[0] != event_type or existing_key[1] != generation_id:
                    continue
                previous = existing_key[2]
                if text == previous or text.startswith(previous) or previous.startswith(text):
                    # Keep the longest streaming rewrite of the same thought.
                    if len(text) >= len(previous):
                        kept[index] = event
                        del text_at[existing_key]
                        text_at[(event_type, generation_id, text)] = index
                    replaced = True
                    break
            if replaced:
                continue
            text_at[(event_type, generation_id, text)] = len(kept)
            kept.append(event)
            continue
        tool_key = _tool_delivery_key(event)
        if tool_key is not None:
            previous = tool_at.get(tool_key)
            if previous is None:
                tool_at[tool_key] = len(kept)
                kept.append(event)
            else:
                kept[previous] = event
            continue
        kept.append(event)
    return kept


def should_record(events: list[dict], event: dict) -> bool:
    """False when this delivery would not change the trace.

    A repeat keeps the same payload and a new timestamp. Compare the payloads
    that would be uploaded, not the raw event lists. Generation ids are
    normalized so ``uuid`` and ``uuid-N-xxxx`` count as the same delivery.
    """

    def reportable(items: list[dict]) -> list[tuple]:
        rows = []
        for item in dedupe_events(items):
            payload = dict(payload_of(item))
            if "generation_id" in payload:
                payload["generation_id"] = normalize_generation_id(payload.get("generation_id"))
            rows.append((item.get("event_type"), payload))
        return rows

    return reportable(events + [event]) != reportable(events)


def normalize_generation_id(raw: object) -> str:
    """Strip Cursor's per-step ``-N-xxxx`` suffix from a generation id."""
    text = str(raw or "")
    match = _GENERATION_ID.match(text)
    if match:
        return match.group(1).lower()
    return text


def _text_delivery_key(event: dict) -> tuple | None:
    if event.get("event_type") not in {"afterAgentThought", "afterAgentResponse"}:
        return None
    payload = payload_of(event)
    text = payload.get("text")
    if not isinstance(text, str) or not text:
        return None
    return (
        event.get("event_type"),
        normalize_generation_id(payload.get("generation_id")),
        text,
    )


def _tool_delivery_key(event: dict) -> tuple | None:
    if event.get("event_type") not in {"postToolUse", "postToolUseFailure"}:
        return None
    payload = payload_of(event)
    tool_id = str(payload.get("tool_use_id") or "")
    if not tool_id:
        return None
    return (event.get("event_type"), tool_id)


def turn_has_agent_work(events: list[dict]) -> bool:
    """True when the buffer has more than a bare prompt (safe to upload)."""
    for event in events:
        if event.get("event_type") in {
            "afterAgentThought",
            "afterAgentResponse",
            "postToolUse",
            "postToolUseFailure",
            "subagentStart",
            "subagentStop",
            "preCompact",
        }:
            return True
    return False


def first_text(events: list[dict], event_type: str, field: str) -> str:
    for event in events:
        if event.get("event_type") != event_type:
            continue
        text = payload_of(event).get(field)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def last_text(events: list[dict], event_type: str, field: str) -> str:
    for event in reversed(events):
        if event.get("event_type") != event_type:
            continue
        text = payload_of(event).get(field)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _usage_from_mapping(mapping: dict) -> dict[str, int]:
    usage: dict[str, int] = {}
    aliases = {
        "cache_read_tokens": "cache_read_input_tokens",
        "cache_write_tokens": "cache_creation_input_tokens",
    }
    for key, value in mapping.items():
        name = aliases.get(key, key)
        if name in TOKEN_KEYS and isinstance(value, int):
            usage[name] = value
    return usage


def extract_usage(payload: dict) -> dict[str, int]:
    usage = _usage_from_mapping(payload)
    for key in ("usage", "token_usage", "tokens"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            usage.update(_usage_from_mapping(nested))
    if usage and "total_tokens" not in usage:
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return usage


def merge_usage(events: list[dict]) -> dict[str, int]:
    """Keep the richest usage snapshot.

    Cursor may attach the same cumulative counters to several hook deliveries.
    Summing them double-counts tokens; taking the max per field keeps one turn.
    """
    totals = {key: 0 for key in TOKEN_KEYS}
    found = False
    for event in events:
        usage = extract_usage(payload_of(event))
        if not usage:
            continue
        found = True
        for key, value in usage.items():
            totals[key] = max(totals.get(key, 0), value)
    if not found:
        return {}
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return {key: value for key, value in totals.items() if value}


def span_window(event: dict, floor_ns: int) -> tuple[int, int]:
    end_ns = int(event.get("timestamp_ns") or 0)
    payload = payload_of(event)
    duration_ms = payload.get("duration_ms")
    if not isinstance(duration_ms, (int, float)):
        duration_ms = payload.get("duration") if isinstance(payload.get("duration"), (int, float)) else 0
    start_ns = end_ns - int(float(duration_ms) * 1_000_000)
    if start_ns < floor_ns:
        start_ns = floor_ns
    if start_ns <= 0 or start_ns > end_ns:
        start_ns = end_ns
    return start_ns, max(end_ns, start_ns)


def tool_span_name(payload: dict) -> str:
    tool_name = str(payload.get("tool_name") or "tool")
    tool_input = payload.get("tool_input")
    detail = ""
    if isinstance(tool_input, dict):
        for key in ("command", "path", "file_path", "pattern", "query", "description", "glob_pattern"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip().splitlines()[0][:80]
                break
    if detail:
        return f"{tool_name}: {detail}"
    return tool_name


def safe_id(raw: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(raw or ""))
    return safe or "unknown"


def child_events_for(subagent_id: str) -> list[dict]:
    path = CHILDREN_DIR / f"{safe_id(subagent_id)}.jsonl"
    if not path.is_file():
        return []
    return dedupe_events(
        [
            event
            for event in read_events(path)
            if event.get("event_type") not in {"stop", "sessionEnd", "subagentStart", "subagentStop"}
        ]
    )


def nested_subagent_events(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for event in events:
        if event.get("event_type") != "subagentStart":
            continue
        subagent_id = payload_of(event).get("subagent_id")
        if not subagent_id:
            continue
        grouped[safe_id(subagent_id)] = child_events_for(str(subagent_id))
    return grouped


def short_trace_name(prompt: str, model: str) -> str:
    text = " ".join(prompt.split())
    if text:
        return text[:80] + ("..." if len(text) > 80 else "")
    return f"Cursor {model or 'agent'}"


def upload(events_file: Path) -> str:
    import os

    global EXPERIMENT_ID

    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    import mlflow
    from mlflow.entities import SpanType
    from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

    tracking_uri, EXPERIMENT_ID, username, password = tracking_settings()
    if not password:
        raise SystemExit(
            "MLflow password missing. Set MLFLOW_TRACKING_PASSWORD in "
            "mlflow/cursor-plugin/config.env or the environment."
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_ID"] = EXPERIMENT_ID
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

    events = dedupe_events(read_events(events_file))
    meaningful = [event for event in events if event.get("event_type") not in {"stop", "sessionEnd"}]
    if not meaningful:
        print("no agent steps to upload")
        return ""
    if not turn_has_agent_work(meaningful):
        # Cursor sometimes fires stop right after beforeSubmitPrompt. Keep the
        # buffer so the real turn can land in one session-linked trace.
        print("defer incomplete turn")
        return ""

    prompt = first_text(events, "beforeSubmitPrompt", "prompt")
    response = last_text(events, "afterAgentResponse", "text")
    nested = nested_subagent_events(meaningful)
    first_payload = payload_of(events[0])
    model = str(first_payload.get("model") or first_payload.get("model_id") or "")
    conv_id = str(
        first_payload.get("conversation_id")
        or first_payload.get("parent_conversation_id")
        or first_payload.get("session_id")
        or events_file.stem
    )
    generation_id = normalize_generation_id(first_payload.get("generation_id") or "")
    user_email = first_payload.get("user_email") or ""
    usage = merge_usage(events + [event for group in nested.values() for event in group])

    start_ns = int(events[0].get("timestamp_ns") or 0)
    end_ns = int(events[-1].get("timestamp_ns") or start_ns)
    metadata = {
        TraceMetadataKey.TRACE_SESSION: conv_id,
        "cursor.conversation_id": conv_id,
    }
    if isinstance(user_email, str) and user_email:
        metadata[TraceMetadataKey.TRACE_USER] = user_email
    if usage:
        metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(usage)

    # This MLflow UI draws a bare chat array twice: once as messages, then
    # again as the pretty payload. Nesting under ``messages`` keeps the chat
    # card and hides the second copy (that key is in the UI's hidden-key set).
    user_messages = [{"role": "user", "content": prompt}] if prompt else []
    root = mlflow.start_span_no_context(
        name=short_trace_name(prompt, model),
        span_type=SpanType.AGENT,
        inputs={"messages": user_messages} if user_messages else {},
        attributes={
            "cursor.conversation_id": conv_id,
            "cursor.generation_id": generation_id,
            "cursor.model": model,
            # The traces table reads this OTEL attribute as well as the tag below.
            SpanAttributeKey.SESSION_ID: conv_id,
        },
        tags={
            "cursor.conversation_id": conv_id,
            "cursor.model": model or "unknown",
            # This MLflow UI version shows the Session column from the tag,
            # while search_sessions reads the metadata key of the same name.
            TraceMetadataKey.TRACE_SESSION: conv_id,
        },
        metadata=metadata,
        experiment_id=EXPERIMENT_ID,
        start_time_ns=start_ns or None,
    )

    floor_ns = start_ns
    open_agents: dict[str, tuple[object, dict]] = {}
    for event in meaningful:
        event_type = event.get("event_type")
        if event_type == "subagentStart":
            agent_span, meta = start_subagent(root, event, floor_ns)
            key = str(meta["subagent_id"] or id(agent_span))
            open_agents[key] = (agent_span, meta)
            agent_start = int(event.get("timestamp_ns") or floor_ns)
            nest_subagent_steps(
                agent_span,
                nested.get(safe_id(meta["subagent_id"]), []),
                meta["model"],
                agent_start,
            )
            continue
        if event_type == "subagentStop":
            finish_subagent(open_agents, event)
            floor_ns = int(event.get("timestamp_ns") or floor_ns)
            continue
        add_child(root, event, model, floor_ns)
        floor_ns = int(event.get("timestamp_ns") or floor_ns)

    for agent_span, meta in open_agents.values():
        if meta.get("ended"):
            continue
        agent_span.end(outputs={"status": "incomplete"}, end_time_ns=end_ns or None)

    assistant_messages = [{"role": "assistant", "content": response}] if response else []
    outputs = {"messages": assistant_messages} if assistant_messages else {"status": "completed"}
    if usage:
        root.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
    # Do not also set mlflow.chat.messages; the UI would show that copy too.
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    with InMemoryTraceManager.get_instance().get_trace(root.trace_id) as memory_trace:
        live_spans = list(memory_trace.span_dict.values()) if memory_trace is not None else []
    root.end(outputs=outputs, end_time_ns=end_ns or None)
    logged_spans = [span.to_immutable_span() for span in live_spans]
    mlflow.flush_trace_async_logging()
    ensure_spans_in_tracking_store(root.trace_id, logged_spans)
    return root.trace_id


def ensure_spans_in_tracking_store(trace_id: str, spans: list) -> None:
    """Keep span payloads in the tracking DB.

    The experiment artifact root is a path inside the MLflow container. A client
    on the host cannot write it, so an artifact-only export drops the spans and
    the session turn cannot be opened.
    """
    import mlflow
    from mlflow.tracing.constant import TraceTagKey

    client = mlflow.MlflowClient()
    try:
        trace = client.get_trace(trace_id)
    except Exception:
        trace = None
    if trace is not None and trace.data.spans:
        return
    if not spans:
        raise RuntimeError(f"trace {trace_id} has no spans to store")
    client.log_spans(EXPERIMENT_ID, spans)
    trace = client.get_trace(trace_id)
    location = (trace.info.tags or {}).get(TraceTagKey.SPANS_LOCATION) if trace else None
    if trace is None or not trace.data.spans:
        raise RuntimeError(f"trace {trace_id} spans were not stored ({location})")


def start_subagent(parent, event: dict, floor_ns: int):
    import mlflow
    from mlflow.entities import SpanType

    payload = payload_of(event)
    start_ns = int(event.get("timestamp_ns") or floor_ns or 0)
    if floor_ns and start_ns < floor_ns:
        start_ns = floor_ns
    kind = str(payload.get("subagent_type") or "agent")
    task = str(payload.get("task") or "")
    preview = " ".join(task.split())
    name = f"{kind}: {preview}" if preview else kind
    if len(name) > 80:
        name = name[:80] + "..."
    model = str(payload.get("subagent_model") or payload.get("model") or "")
    span = mlflow.start_span_no_context(
        name=name,
        parent_span=parent,
        span_type=SpanType.AGENT,
        inputs={"task": task, "subagent_type": kind},
        attributes={
            "cursor.event": "subagent",
            "cursor.subagent_id": str(payload.get("subagent_id") or ""),
            "cursor.tool_call_id": str(payload.get("tool_call_id") or ""),
            "cursor.model": model,
        },
        start_time_ns=start_ns or None,
    )
    meta = {
        "subagent_id": str(payload.get("subagent_id") or ""),
        "task": task,
        "subagent_type": kind,
        "model": model,
        "ended": False,
    }
    return span, meta


def nest_subagent_steps(agent_span, events: list[dict], model: str, floor_ns: int) -> None:
    cursor = floor_ns
    for event in events:
        add_child(agent_span, event, model, cursor)
        cursor = int(event.get("timestamp_ns") or cursor)


def finish_subagent(open_agents: dict, event: dict) -> None:
    payload = payload_of(event)
    task = str(payload.get("task") or "")
    kind = str(payload.get("subagent_type") or "")
    chosen = None
    for key, (_span, meta) in open_agents.items():
        if meta.get("ended"):
            continue
        same_task = not task or meta.get("task") == task
        same_kind = not kind or meta.get("subagent_type") == kind
        if same_task and same_kind:
            chosen = key
            break
    if chosen is None:
        pending = [key for key, (_span, meta) in open_agents.items() if not meta.get("ended")]
        if len(pending) == 1:
            chosen = pending[0]
    if chosen is None:
        return
    span, meta = open_agents[chosen]
    meta["ended"] = True
    status = str(payload.get("status") or "completed")
    end_ns = int(event.get("timestamp_ns") or 0)
    span.end(
        outputs={
            "summary": payload.get("summary"),
            "status": status,
            "description": payload.get("description"),
            "message_count": payload.get("message_count"),
            "tool_call_count": payload.get("tool_call_count"),
            "modified_files": payload.get("modified_files"),
        },
        status="ERROR" if status == "error" else "OK",
        end_time_ns=end_ns or None,
    )


def add_child(root, event: dict, model: str, floor_ns: int) -> None:
    import mlflow
    from mlflow.entities import SpanType
    from mlflow.tracing.constant import SpanAttributeKey

    event_type = event.get("event_type") or "event"
    payload = payload_of(event)
    start_ns, end_ns = span_window(event, floor_ns)
    generation_id = str(payload.get("generation_id") or "")
    common = {
        "cursor.event": event_type,
        "cursor.generation_id": generation_id,
    }
    if model:
        common["cursor.model"] = model

    if event_type == "beforeSubmitPrompt":
        # Prompt is stored on the root span inputs for the session page.
        return

    if event_type == "afterAgentThought":
        text = payload.get("text", "")
        span = mlflow.start_span_no_context(
            name="agent_thinking",
            parent_span=root,
            span_type=SpanType.LLM,
            inputs={"model": model},
            attributes={**common, SpanAttributeKey.MODEL: model or "unknown"},
            start_time_ns=start_ns,
        )
        span.set_outputs({"thought": text})
        span.end(end_time_ns=end_ns)
        return

    if event_type == "afterAgentResponse":
        # Final reply is stored on the root span outputs for the session page.
        return

    if event_type in {"postToolUse", "postToolUseFailure"}:
        failed = event_type == "postToolUseFailure"
        span = mlflow.start_span_no_context(
            name=tool_span_name(payload),
            parent_span=root,
            span_type=SpanType.TOOL,
            inputs={
                "tool_name": payload.get("tool_name"),
                "tool_input": payload.get("tool_input"),
            },
            attributes={
                **common,
                "cursor.tool_name": str(payload.get("tool_name") or ""),
                "cursor.tool_use_id": str(payload.get("tool_use_id") or ""),
            },
            start_time_ns=start_ns,
        )
        outputs = {"tool_output": payload.get("tool_output")}
        if failed:
            outputs = {
                "error_message": payload.get("error_message"),
                "failure_type": payload.get("failure_type"),
            }
        span.end(outputs=outputs, status="ERROR" if failed else "OK", end_time_ns=end_ns)
        return

    if event_type == "preCompact":
        span = mlflow.start_span_no_context(
            name="context_compact",
            parent_span=root,
            span_type=SpanType.CHAIN,
            inputs={
                "trigger": payload.get("trigger"),
                "context_usage_percent": payload.get("context_usage_percent"),
                "context_tokens": payload.get("context_tokens"),
                "message_count": payload.get("message_count"),
            },
            attributes=common,
            start_time_ns=start_ns,
        )
        span.end(end_time_ns=end_ns)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mlflow_upload.py <events.jsonl>", file=sys.stderr)
        return 2
    events_file = Path(sys.argv[1])
    if not events_file.is_file():
        print(f"missing events file: {events_file}", file=sys.stderr)
        return 2
    trace_id = upload(events_file)
    if trace_id:
        print(trace_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"upload error: {exc}", file=sys.stderr)
        raise SystemExit(1)
