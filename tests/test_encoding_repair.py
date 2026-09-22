#!/usr/bin/env python3
"""Windows hook stdin double-encoding repair (ACP misread of UTF-8)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from mlflow_cursor import (  # noqa: E402
    load_hook_payload,
    repair_windows_mojibake,
)


def double_encode_as_acp(text: str) -> str:
    """Simulate Cursor bug: UTF-8 bytes decoded as system ACP, then kept as Unicode."""
    return text.encode("utf-8").decode("mbcs")


class EncodingRepairTests(unittest.TestCase):
    def test_repair_chinese_prompt(self) -> None:
        original = "插件能用了吗"
        broken = double_encode_as_acp(original)
        self.assertNotEqual(broken, original)
        self.assertEqual(repair_windows_mojibake(broken), original)

    def test_repair_full_json_payload(self) -> None:
        payload = {
            "hook_event_name": "beforeSubmitPrompt",
            "conversation_id": "enc-repair-1",
            "prompt": "修复上传字符异常问题",
        }
        raw = json.dumps(payload, ensure_ascii=False)
        broken = double_encode_as_acp(raw)
        self.assertIn("修复", raw)
        self.assertNotIn("修复", broken)
        parsed = load_hook_payload(broken.encode("utf-8"))
        self.assertEqual(parsed.get("prompt"), "修复上传字符异常问题")
        self.assertEqual(parsed.get("hook_event_name"), "beforeSubmitPrompt")

    def test_repair_thought_payload(self) -> None:
        payload = {
            "hook_event_name": "afterAgentThought",
            "conversation_id": "enc-repair-2",
            "text": "正在排查字符编码异常",
        }
        raw = json.dumps(payload, ensure_ascii=False)
        broken = double_encode_as_acp(raw)
        parsed = load_hook_payload(broken.encode("utf-8"))
        self.assertEqual(parsed.get("text"), payload["text"])

    def test_ascii_only_unchanged(self) -> None:
        text = '{"hook_event_name":"stop","conversation_id":"abc"}'
        self.assertEqual(repair_windows_mojibake(text), text)
        parsed = load_hook_payload(text.encode("utf-8"))
        self.assertEqual(parsed.get("hook_event_name"), "stop")

    def test_utf8_bom_accepted(self) -> None:
        payload = {"hook_event_name": "stop", "conversation_id": "bom-1"}
        raw = b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8")
        parsed = load_hook_payload(raw)
        self.assertEqual(parsed.get("conversation_id"), "bom-1")


if __name__ == "__main__":
    if sys.platform != "win32":
        print("skip: Windows-only ACP mojibake tests")
        raise SystemExit(0)
    unittest.main()
