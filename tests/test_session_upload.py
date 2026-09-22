#!/usr/bin/env python3
"""Upload a Cursor turn and assert the session contract this MLflow UI reads.

The traces table and Group by session use trace metadata ``mlflow.trace.session``.
The session column link also reads the same key from tags. A turn with missing
spans cannot be opened from the session page.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
PLUGIN = Path(__file__).resolve().parents[1]
HOOK = PLUGIN / "scripts" / "mlflow_cursor.py"
SCRIPTS = PLUGIN / "scripts"
ENV_FILE = WORKSPACE / "mlflow" / ".env"
EXPERIMENT_ID = "1"
SESSION_KEY = "mlflow.trace.session"
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def load_env() -> None:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key == "MLFLOW_AUTH_ADMIN_USERNAME":
            os.environ["MLFLOW_TRACKING_USERNAME"] = value
        elif key == "MLFLOW_AUTH_ADMIN_PASSWORD":
            os.environ["MLFLOW_TRACKING_PASSWORD"] = value
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:21103"
    os.environ["MLFLOW_EXPERIMENT_ID"] = EXPERIMENT_ID


def send(payload: dict) -> None:
    completed = subprocess.run(
        [sys.executable if "hooks/.venv" in sys.executable else "python3", str(HOOK)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(WORKSPACE),
        timeout=90,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)


class DedupeEventsTest(unittest.TestCase):
    def test_repeated_thoughts_and_tool_results_collapse_to_one(self) -> None:
        sys.path.insert(0, str(SCRIPTS))
        from mlflow_upload import dedupe_events, should_record

        thought = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 1,
            "payload": {"generation_id": "g1", "text": "checking disk"},
        }
        thought_again = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 2,
            "payload": {"generation_id": "g1", "text": "checking disk"},
        }
        tool = {
            "event_type": "postToolUse",
            "timestamp_ns": 3,
            "payload": {
                "generation_id": "g1",
                "tool_name": "Read",
                "tool_use_id": "call-1",
                "tool_input": {"path": "a.txt"},
                "tool_output": "partial",
            },
        }
        tool_again = {
            "event_type": "postToolUse",
            "timestamp_ns": 4,
            "payload": {
                "generation_id": "g1",
                "tool_name": "Read",
                "tool_use_id": "call-1",
                "tool_input": {"path": "a.txt"},
                "tool_output": "final",
            },
        }
        other = {
            "event_type": "beforeSubmitPrompt",
            "timestamp_ns": 0,
            "payload": {"generation_id": "g1", "prompt": "hi"},
        }

        kept = dedupe_events([other, thought, thought_again, tool, tool_again])

        self.assertEqual([event["event_type"] for event in kept], ["beforeSubmitPrompt", "afterAgentThought", "postToolUse"])
        self.assertEqual(kept[2]["payload"]["tool_output"], "final")
        self.assertEqual(kept[2]["timestamp_ns"], 4)
        self.assertFalse(should_record([tool], {**tool, "timestamp_ns": 9}))
        self.assertTrue(should_record([tool], tool_again))

    def test_should_collapse_thought_when_generation_id_gains_step_suffix(self) -> None:
        """Cursor redelivers the same thought with ``uuid-N-xxxx`` generation ids."""
        sys.path.insert(0, str(SCRIPTS))
        from mlflow_upload import dedupe_events, should_record

        base = "aae32017-5168-450e-a4a9-18dd7cffc1c8"
        text = "checking disk for reclaimable files"
        first = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 1,
            "payload": {"generation_id": base, "text": text},
        }
        again = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 2,
            "payload": {"generation_id": f"{base}-0-dozj", "text": text},
        }
        later = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 3,
            "payload": {"generation_id": f"{base}-1-4t4c", "text": "now scanning docker"},
        }

        kept = dedupe_events([first, again, later])

        self.assertEqual(len(kept), 2)
        self.assertEqual([event["payload"]["text"] for event in kept], [text, "now scanning docker"])
        self.assertFalse(should_record([first], again))
        self.assertTrue(should_record([first], later))

    def test_should_collapse_growing_thought_text_for_same_generation(self) -> None:
        """Cursor may stream the same thought as a longer rewrite of the earlier text."""
        sys.path.insert(0, str(SCRIPTS))
        from mlflow_upload import dedupe_events

        base = "bbf43128-6279-561f-b5ba-29ee8d00d2d9"
        partial = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 1,
            "payload": {"generation_id": base, "text": "checking disk"},
        }
        fuller = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 2,
            "payload": {
                "generation_id": f"{base}-0-zzzz",
                "text": "checking disk for reclaimable files",
            },
        }
        other = {
            "event_type": "afterAgentThought",
            "timestamp_ns": 3,
            "payload": {"generation_id": base, "text": "now listing docker volumes"},
        }

        kept = dedupe_events([partial, fuller, other])

        self.assertEqual(
            [event["payload"]["text"] for event in kept],
            ["checking disk for reclaimable files", "now listing docker volumes"],
        )


class SessionUploadTest(unittest.TestCase):
    trace_ids: list[str]

    def setUp(self) -> None:
        load_env()
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        self.client = mlflow.MlflowClient()
        self.trace_ids = []

    def tearDown(self) -> None:
        if self.trace_ids:
            self.client.delete_traces(EXPERIMENT_ID, trace_ids=self.trace_ids)

    def test_turn_is_searchable_as_a_session_and_spans_load(self) -> None:
        session_id = f"session-test-{uuid.uuid4().hex[:8]}"
        prompt = "Where is the session id stored?"
        reply = "In trace metadata and tags."
        self._turn(session_id, prompt, reply, generation="g1")

        traces = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=5,
        )
        self.assertEqual(len(traces), 1)
        trace_id = traces[0].info.trace_id
        self.trace_ids.append(trace_id)

        loaded = self.client.get_trace(trace_id)
        self.assertIsNotNone(loaded)
        self.assertGreaterEqual(len(loaded.data.spans), 2)
        info = loaded.info
        self.assertEqual(info.trace_metadata.get(SESSION_KEY), session_id)
        self.assertEqual(info.tags.get(SESSION_KEY), session_id)
        root = next(span for span in loaded.data.spans if span.parent_id is None)
        self.assertEqual(root.inputs, {"messages": [{"role": "user", "content": prompt}]})
        self.assertEqual(root.outputs, {"messages": [{"role": "assistant", "content": reply}]})
        names = [span.name for span in loaded.data.spans]
        self.assertNotIn("user_prompt", names)
        self.assertNotIn("agent_response", names)
        self.assertNotIn("mlflow.chat.messages", root.attributes or {})

        import mlflow

        sessions = mlflow.search_sessions(
            locations=[EXPERIMENT_ID], max_results=50, include_spans=False
        )
        self.assertIn(session_id, [session.id for session in sessions])

    def test_should_upload_one_thinking_span_when_generation_id_has_step_suffix(self) -> None:
        session_id = f"session-dedupe-{uuid.uuid4().hex[:8]}"
        base = f"{uuid.uuid4()}"
        thought = "looking for reclaimable disk usage"

        send(
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": session_id,
                "generation_id": base,
                "model": "test-model",
                "prompt": "Clean disk",
            }
        )
        send(
            {
                "hook_event_name": "afterAgentThought",
                "conversation_id": session_id,
                "generation_id": base,
                "model": "test-model",
                "text": thought,
                "duration_ms": 10,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentThought",
                "conversation_id": session_id,
                "generation_id": f"{base}-0-ab12",
                "model": "test-model",
                "text": thought,
                "duration_ms": 12,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": session_id,
                "generation_id": base,
                "model": "test-model",
                "text": "Done",
            }
        )
        send(
            {
                "hook_event_name": "stop",
                "conversation_id": session_id,
                "generation_id": base,
                "status": "completed",
                "loop_count": 0,
            }
        )

        traces = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=5,
        )
        self.assertEqual(len(traces), 1)
        trace_id = traces[0].info.trace_id
        self.trace_ids.append(trace_id)
        loaded = self.client.get_trace(trace_id)
        thinking = [span for span in loaded.data.spans if span.name == "agent_thinking"]
        self.assertEqual(len(thinking), 1, [span.name for span in loaded.data.spans])
        self.assertEqual(thinking[0].outputs.get("thought"), thought)

    def test_stop_and_session_end_upload_a_single_trace(self) -> None:
        session_id = f"session-flush-{uuid.uuid4().hex[:8]}"
        prompt = "One turn only"
        reply = "Uploaded once"

        send(
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": session_id,
                "generation_id": "g-flush",
                "model": "test-model",
                "prompt": prompt,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentThought",
                "conversation_id": session_id,
                "generation_id": "g-flush",
                "model": "test-model",
                "text": "working",
                "duration_ms": 5,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": session_id,
                "generation_id": "g-flush",
                "model": "test-model",
                "text": reply,
            }
        )

        def stop() -> None:
            send(
                {
                    "hook_event_name": "stop",
                    "conversation_id": session_id,
                    "generation_id": "g-flush",
                    "status": "completed",
                    "loop_count": 0,
                }
            )

        def session_end() -> None:
            send(
                {
                    "hook_event_name": "sessionEnd",
                    "conversation_id": session_id,
                    "generation_id": "g-flush",
                    "reason": "completed",
                    "duration_ms": 10,
                    "final_status": "completed",
                }
            )

        workers = [threading.Thread(target=stop), threading.Thread(target=session_end)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=90)

        traces = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=5,
        )
        self.assertEqual(len(traces), 1, [item.info.trace_id for item in traces])
        self.trace_ids.append(traces[0].info.trace_id)
        loaded = self.client.get_trace(traces[0].info.trace_id)
        root = next(span for span in loaded.data.spans if span.parent_id is None)
        self.assertEqual(root.outputs, {"messages": [{"role": "assistant", "content": reply}]})

    def test_premature_stop_waits_for_real_work_then_chains_one_trace(self) -> None:
        session_id = f"session-early-{uuid.uuid4().hex[:8]}"
        prompt = "Continue after early stop"
        reply = "Single chained turn"

        send(
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": session_id,
                "generation_id": "g-early",
                "model": "test-model",
                "prompt": prompt,
            }
        )
        send(
            {
                "hook_event_name": "stop",
                "conversation_id": session_id,
                "generation_id": "g-early",
                "status": "completed",
                "loop_count": 0,
            }
        )
        early = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=5,
        )
        self.assertEqual(list(early), [])

        send(
            {
                "hook_event_name": "afterAgentThought",
                "conversation_id": session_id,
                "generation_id": "g-early",
                "model": "test-model",
                "text": "still working",
                "duration_ms": 5,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": session_id,
                "generation_id": "g-early",
                "model": "test-model",
                "text": reply,
            }
        )
        send(
            {
                "hook_event_name": "stop",
                "conversation_id": session_id,
                "generation_id": "g-early",
                "status": "completed",
                "loop_count": 0,
            }
        )

        traces = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=5,
        )
        self.assertEqual(len(traces), 1)
        self.trace_ids.append(traces[0].info.trace_id)
        loaded = self.client.get_trace(traces[0].info.trace_id)
        root = next(span for span in loaded.data.spans if span.parent_id is None)
        self.assertEqual(root.inputs, {"messages": [{"role": "user", "content": prompt}]})
        self.assertEqual(root.outputs, {"messages": [{"role": "assistant", "content": reply}]})
        names = [span.name for span in loaded.data.spans]
        self.assertEqual(names.count("agent_thinking"), 1)

    def test_reply_that_arrives_after_stop_stays_in_the_session_turn(self) -> None:
        session_id = f"session-race-{uuid.uuid4().hex[:8]}"
        prompt = "Reply after stop starts"
        reply = "Still in this turn"

        def stop() -> None:
            send(
                {
                    "hook_event_name": "stop",
                    "conversation_id": session_id,
                    "generation_id": "g-race",
                    "status": "completed",
                    "loop_count": 0,
                }
            )

        send(
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": session_id,
                "generation_id": "g-race",
                "model": "test-model",
                "prompt": prompt,
            }
        )
        worker = threading.Thread(target=stop)
        worker.start()
        time.sleep(0.2)
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": session_id,
                "generation_id": "g-race",
                "model": "test-model",
                "text": reply,
            }
        )
        worker.join(timeout=90)

        traces = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=5,
        )
        self.assertEqual(len(traces), 1, traces)
        trace_id = traces[0].info.trace_id
        self.trace_ids.append(trace_id)
        loaded = self.client.get_trace(trace_id)
        root = next(span for span in loaded.data.spans if span.parent_id is None)
        self.assertEqual(root.outputs, {"messages": [{"role": "assistant", "content": reply}]})
        self.assertEqual(loaded.info.tags.get(SESSION_KEY), session_id)

    def test_subagent_steps_share_the_parent_session(self) -> None:
        session_id = f"session-sub-{uuid.uuid4().hex[:8]}"
        subagent_id = f"sub-{uuid.uuid4().hex[:8]}"
        send(
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": session_id,
                "generation_id": "g-parent",
                "model": "test-model",
                "prompt": "Ask a subagent",
            }
        )
        send(
            {
                "hook_event_name": "subagentStart",
                "conversation_id": session_id,
                "parent_conversation_id": session_id,
                "generation_id": "g-parent",
                "model": "test-model",
                "subagent_id": subagent_id,
                "subagent_type": "explore",
                "task": "Find the session key",
                "tool_call_id": "tool-1",
                "subagent_model": "test-model",
            }
        )
        send(
            {
                "hook_event_name": "postToolUse",
                "conversation_id": subagent_id,
                "generation_id": "g-child",
                "model": "test-model",
                "tool_name": "Read",
                "tool_use_id": "child-1",
                "tool_input": {"path": "mlflow_upload.py"},
                "tool_output": "session key",
                "duration": 5,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": subagent_id,
                "generation_id": "g-child",
                "model": "test-model",
                "text": "The key is mlflow.trace.session",
            }
        )
        send(
            {
                "hook_event_name": "stop",
                "conversation_id": subagent_id,
                "generation_id": "g-child",
                "status": "completed",
                "loop_count": 0,
            }
        )
        send(
            {
                "hook_event_name": "subagentStop",
                "conversation_id": session_id,
                "generation_id": "g-parent",
                "subagent_type": "explore",
                "task": "Find the session key",
                "status": "completed",
                "summary": "The key is mlflow.trace.session",
                "duration_ms": 20,
                "tool_call_count": 1,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": session_id,
                "generation_id": "g-parent",
                "model": "test-model",
                "text": "Subagent found the session key.",
            }
        )
        send(
            {
                "hook_event_name": "stop",
                "conversation_id": session_id,
                "generation_id": "g-parent",
                "status": "completed",
                "loop_count": 0,
            }
        )

        traces = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{session_id}'",
            max_results=10,
        )
        self.assertEqual(len(traces), 1)
        trace_id = traces[0].info.trace_id
        self.trace_ids.append(trace_id)
        loaded = self.client.get_trace(trace_id)
        names = [span.name for span in loaded.data.spans]
        self.assertTrue(any(name.startswith("explore:") for name in names), names)
        self.assertTrue(any(name.startswith("Read:") for name in names), names)
        self.assertEqual(loaded.info.tags.get(SESSION_KEY), session_id)
        child_sessions = self.client.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=f"metadata.`{SESSION_KEY}` = '{subagent_id}'",
            max_results=5,
        )
        self.assertEqual(list(child_sessions), [])

    def _turn(self, session_id: str, prompt: str, reply: str, generation: str) -> None:
        send(
            {
                "hook_event_name": "beforeSubmitPrompt",
                "conversation_id": session_id,
                "generation_id": generation,
                "model": "test-model",
                "prompt": prompt,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentThought",
                "conversation_id": session_id,
                "generation_id": generation,
                "model": "test-model",
                "text": "checking",
                "duration_ms": 10,
            }
        )
        send(
            {
                "hook_event_name": "afterAgentResponse",
                "conversation_id": session_id,
                "generation_id": generation,
                "model": "test-model",
                "text": reply,
            }
        )
        send(
            {
                "hook_event_name": "stop",
                "conversation_id": session_id,
                "generation_id": generation,
                "status": "completed",
                "loop_count": 0,
            }
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
