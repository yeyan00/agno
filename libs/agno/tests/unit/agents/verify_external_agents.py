"""Integration verification: PiAgent + OpenCodeAgent with real DB storage.

Uses mock subprocess output but real BaseExternalAgent + SQLite DB to verify:
1. Session/run data is persisted correctly
2. Messages (user, assistant, tool) are stored
3. Multiple runs within a session accumulate
4. Streaming path also persists correctly
5. session_id is transparently forwarded to CLI
"""

import asyncio
import gc
import json
import os
import tempfile
from typing import List
from unittest.mock import patch

from agno.agents.pi.agent import PiAgent
from agno.agents.opencode.agent import OpenCodeAgent
from agno.db.sqlite.sqlite import SqliteDb
from agno.run.agent import RunContentEvent, RunOutput, ToolCallCompletedEvent, ToolCallStartedEvent
from agno.run.base import RunStatus


class _FakeProc:
    def __init__(self, lines: List[str]):
        self._lines = [line.encode() for line in lines]
        self.stdout = self._iter()
        self.returncode = 0

    async def _iter(self):
        for line in self._lines:
            yield line

    async def wait(self):
        pass

    async def communicate(self):
        return b"\n".join(self._lines), b""


def _pi_simple_response():
    return [
        json.dumps({"type": "session", "version": 3, "id": "pi-sess-001", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/tmp"}),
        json.dumps({"type": "agent_start"}),
        json.dumps({"type": "turn_start"}),
        json.dumps({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "Hello", "contentIndex": 0}, "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}}),
        json.dumps({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": " world", "contentIndex": 0}, "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}}),
        json.dumps({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}}),
        json.dumps({"type": "turn_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}}),
        json.dumps({"type": "agent_end", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
        ]}),
    ]


def _pi_tool_call_response():
    return [
        json.dumps({"type": "session", "version": 3, "id": "pi-sess-002", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/tmp"}),
        json.dumps({"type": "agent_start"}),
        json.dumps({"type": "turn_start"}),
        json.dumps({"type": "message_update", "assistantMessageEvent": {
            "type": "toolcall_end", "contentIndex": 0,
            "toolCall": {"id": "call-001", "name": "read", "arguments": {"path": "test.txt"}},
        }, "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}]}}),
        json.dumps({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}]}}),
        json.dumps({"type": "message_end", "message": {"role": "toolResult", "toolCallId": "call-001", "toolName": "read", "content": [{"type": "text", "text": "file contents here"}], "isError": False}}),
        json.dumps({"type": "turn_end", "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}]}}),
        json.dumps({"type": "turn_start"}),
        json.dumps({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "The file contains: file contents here", "contentIndex": 0}, "message": {"role": "assistant", "content": [{"type": "text", "text": "The file contains: file contents here"}]}}),
        json.dumps({"type": "agent_end", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "read test.txt"}]},
            {"role": "assistant", "content": [{"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}]},
            {"role": "toolResult", "toolCallId": "call-001", "toolName": "read", "content": [{"type": "text", "text": "file contents here"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "The file contains: file contents here"}]},
        ]}),
    ]


def _opencode_simple_response():
    return [
        json.dumps({"type": "step_start", "timestamp": 1, "sessionID": "ses-001", "part": {"id": "p1", "type": "step-start"}}),
        json.dumps({"type": "text", "timestamp": 2, "sessionID": "ses-001", "part": {"id": "p2", "type": "text", "text": "Hello from OpenCode"}}),
        json.dumps({"type": "step_finish", "timestamp": 3, "sessionID": "ses-001", "part": {"id": "p3", "type": "step-finish", "reason": "stop"}}),
    ]


def _opencode_tool_call_response():
    return [
        json.dumps({"type": "step_start", "timestamp": 1, "sessionID": "ses-002", "part": {"id": "p1", "type": "step-start"}}),
        json.dumps({"type": "tool_use", "timestamp": 2, "sessionID": "ses-002", "part": {
            "type": "tool", "tool": "bash", "callID": "call-oc-001",
            "state": {"status": "completed", "input": {"command": "ls"}, "output": "file1.txt\nfile2.txt"},
            "id": "p2",
        }}),
        json.dumps({"type": "step_finish", "timestamp": 3, "sessionID": "ses-002", "part": {"id": "p3", "type": "step-finish", "reason": "tool-calls"}}),
        json.dumps({"type": "step_start", "timestamp": 4, "sessionID": "ses-002", "part": {"id": "p4", "type": "step-start"}}),
        json.dumps({"type": "text", "timestamp": 5, "sessionID": "ses-002", "part": {"id": "p5", "type": "text", "text": "Found 2 files"}}),
        json.dumps({"type": "step_finish", "timestamp": 6, "sessionID": "ses-002", "part": {"id": "p6", "type": "step-finish", "reason": "stop"}}),
    ]


def _cleanup_db(db_path: str):
    del_db = None
    try:
        del_db = True
    except Exception:
        pass
    gc.collect()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass


async def verify_pi_agent():
    print("=" * 60)
    print("Verifying PiAgent with SQLite DB")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = SqliteDb(db_file=db_path)
        agent = PiAgent(name="Test Pi", db=db)

        # --- Run 1: Non-streaming, simple text ---
        print("\n[Run 1] Non-streaming, simple text...")
        fake_proc = _FakeProc(_pi_simple_response())
        with patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi"), \
             patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            result = await agent._arun_non_stream("hi", session_id="sess-1")

        assert isinstance(result, RunOutput)
        assert result.content == "Hello world"
        assert result.status == RunStatus.completed
        assert result.session_id == "sess-1"
        print(f"  content: {result.content}")
        print(f"  status: {result.status}")

        # Verify DB
        session = db.get_session(session_id="sess-1", session_type="agent")
        assert session is not None
        runs = session.runs or []
        assert len(runs) == 1
        print(f"  DB runs count: {len(runs)}")

        stored_messages = runs[0].messages or []
        roles = [m.role for m in stored_messages]
        assert "user" in roles
        assert "assistant" in roles
        print(f"  message roles: {roles}")

        # --- Run 2: Streaming, with tool calls ---
        print("\n[Run 2] Streaming, with tool calls...")
        fake_proc2 = _FakeProc(_pi_tool_call_response())
        events = []
        with patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi"), \
             patch("asyncio.create_subprocess_exec", return_value=fake_proc2):
            async for event in agent._arun_stream("read test.txt", session_id="sess-1"):
                events.append(event)

        content_events = [e for e in events if isinstance(e, RunContentEvent)]
        tool_started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
        tool_completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert len(tool_started) >= 1
        assert tool_started[0].tool.tool_name == "read"
        assert len(tool_completed) >= 1
        assert "file contents here" in tool_completed[0].tool.result
        print(f"  tool started: {len(tool_started)}, tool completed: {len(tool_completed)}")

        # Verify DB has 2 runs
        session = db.get_session(session_id="sess-1", session_type="agent")
        runs = session.runs or []
        assert len(runs) == 2
        print(f"  DB total runs: {len(runs)}")

        run2 = runs[1]
        tools = run2.tools or []
        tool_names = [t.tool_name for t in tools if hasattr(t, "tool_name")]
        assert "read" in tool_names
        print(f"  Run 2 tool names: {tool_names}")

        # --- Verify session_id is forwarded to CLI ---
        print("\n[Run 3] Verify --session passed to CLI...")
        captured_args = []

        async def mock_exec(*args, **kwargs):
            captured_args.extend(args)
            return _FakeProc(_pi_simple_response())

        with patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi"), \
             patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await agent._arun_non_stream("follow up", session_id="sess-1")

        assert "--session" in captured_args
        assert "sess-1" in captured_args
        print("  CLI args: --session sess-1 OK")

        print("\n  PiAgent: ALL CHECKS PASSED")
    finally:
        del db
        _cleanup_db(db_path)


async def verify_opencode_agent():
    print("\n" + "=" * 60)
    print("Verifying OpenCodeAgent with SQLite DB")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = SqliteDb(db_file=db_path)
        agent = OpenCodeAgent(name="Test OC", db=db)

        # --- Run 1: Non-streaming ---
        print("\n[Run 1] Non-streaming, simple text...")
        fake_proc = _FakeProc(_opencode_simple_response())
        with patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode"), \
             patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            result = await agent._arun_non_stream("hello")

        assert isinstance(result, RunOutput)
        assert result.content == "Hello from OpenCode"
        assert result.status == RunStatus.completed
        print(f"  content: {result.content}")

        # Verify DB
        session = db.get_session(session_id=result.session_id, session_type="agent")
        assert session is not None
        runs = session.runs or []
        assert len(runs) == 1
        print(f"  DB runs count: {len(runs)}")

        # --- Run 2: Streaming, tool calls ---
        print("\n[Run 2] Streaming, with tool calls...")
        fake_proc2 = _FakeProc(_opencode_tool_call_response())
        events = []
        with patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode"), \
             patch("asyncio.create_subprocess_exec", return_value=fake_proc2):
            async for event in agent._arun_stream("list files", session_id=result.session_id):
                events.append(event)

        tool_started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
        tool_completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert len(tool_started) == 1
        assert tool_started[0].tool.tool_name == "bash"
        assert len(tool_completed) == 1
        assert "file1.txt" in tool_completed[0].tool.result
        print(f"  tool started: {len(tool_started)}, tool completed: {len(tool_completed)}")

        session = db.get_session(session_id=result.session_id, session_type="agent")
        runs = session.runs or []
        assert len(runs) == 2
        print(f"  DB total runs: {len(runs)}")

        # --- Verify session_id is forwarded to CLI ---
        print("\n[Run 3] Verify --session passed to CLI...")
        captured_args = []

        async def mock_exec(*args, **kwargs):
            captured_args.extend(args)
            return _FakeProc(_opencode_simple_response())

        with patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode"), \
             patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await agent._arun_non_stream("follow up", session_id=result.session_id)

        assert "--session" in captured_args
        assert result.session_id in captured_args
        print(f"  CLI args: --session {result.session_id} OK")

        print("\n  OpenCodeAgent: ALL CHECKS PASSED")
    finally:
        del db
        _cleanup_db(db_path)


async def main():
    await verify_pi_agent()
    await verify_opencode_agent()
    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
