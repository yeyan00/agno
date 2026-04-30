"""Unit tests for PiAgent and OpenCodeAgent external adapters.

These tests use mocked subprocess output to verify JSON event parsing,
tool call tracking, and session passthrough — no CLI installation required.
"""

import json
from typing import List
from unittest.mock import patch

import pytest

from agno.agents.opencode.agent import OpenCodeAgent
from agno.agents.pi.agent import PiAgent
from agno.run.agent import (
    RunContentEvent,
    RunOutput,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProc:
    """Fake asyncio.subprocess.Process that yields pre-set JSON lines."""

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


def _pi_simple_response() -> List[str]:
    """Pi JSON events for a simple text-only response."""
    return [
        json.dumps(
            {"type": "session", "version": 3, "id": "sess-001", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/tmp"}
        ),
        json.dumps({"type": "agent_start"}),
        json.dumps({"type": "turn_start"}),
        json.dumps(
            {
                "type": "message_start",
                "message": {"role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 1},
            }
        ),
        json.dumps({"type": "message_start", "message": {"role": "assistant", "content": [], "timestamp": 2}}),
        json.dumps(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "Hello", "contentIndex": 0},
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            }
        ),
        json.dumps(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": " world", "contentIndex": 0},
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
            }
        ),
        json.dumps(
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
            }
        ),
        json.dumps(
            {"type": "turn_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}}
        ),
        json.dumps(
            {
                "type": "agent_end",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
                ],
            }
        ),
    ]


def _pi_tool_call_response() -> List[str]:
    """Pi JSON events for a response with a tool call (read file)."""
    return [
        json.dumps(
            {"type": "session", "version": 3, "id": "sess-002", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/tmp"}
        ),
        json.dumps({"type": "agent_start"}),
        json.dumps({"type": "turn_start"}),
        json.dumps(
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "toolcall_end",
                    "contentIndex": 0,
                    "toolCall": {"id": "call-001", "name": "read", "arguments": {"path": "test.txt"}},
                },
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-001",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "file contents here"}],
                    "isError": False,
                },
            }
        ),
        json.dumps(
            {
                "type": "turn_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}
                    ],
                },
            }
        ),
        json.dumps({"type": "turn_start"}),
        json.dumps(
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "text_delta",
                    "delta": "The file contains: file contents here",
                    "contentIndex": 0,
                },
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The file contains: file contents here"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "agent_end",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "read test.txt"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "toolCall", "id": "call-001", "name": "read", "arguments": {"path": "test.txt"}}
                        ],
                    },
                    {
                        "role": "toolResult",
                        "toolCallId": "call-001",
                        "toolName": "read",
                        "content": [{"type": "text", "text": "file contents here"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "The file contains: file contents here"}],
                    },
                ],
            }
        ),
    ]


def _opencode_simple_response() -> List[str]:
    """OpenCode JSON events for a simple text-only response."""
    return [
        json.dumps(
            {"type": "step_start", "timestamp": 1, "sessionID": "ses-001", "part": {"id": "p1", "type": "step-start"}}
        ),
        json.dumps(
            {
                "type": "text",
                "timestamp": 2,
                "sessionID": "ses-001",
                "part": {"id": "p2", "type": "text", "text": "Hello world"},
            }
        ),
        json.dumps(
            {
                "type": "step_finish",
                "timestamp": 3,
                "sessionID": "ses-001",
                "part": {
                    "id": "p3",
                    "type": "step-finish",
                    "reason": "stop",
                    "tokens": {"total": 100, "input": 50, "output": 50},
                },
            }
        ),
    ]


def _opencode_tool_call_response() -> List[str]:
    """OpenCode JSON events for a response with a tool call."""
    return [
        json.dumps(
            {"type": "step_start", "timestamp": 1, "sessionID": "ses-002", "part": {"id": "p1", "type": "step-start"}}
        ),
        json.dumps(
            {
                "type": "tool_use",
                "timestamp": 2,
                "sessionID": "ses-002",
                "part": {
                    "type": "tool",
                    "tool": "read",
                    "callID": "call-001",
                    "state": {"status": "completed", "input": {"filePath": "test.txt"}, "output": "file contents here"},
                    "id": "p2",
                },
            }
        ),
        json.dumps(
            {
                "type": "step_finish",
                "timestamp": 3,
                "sessionID": "ses-002",
                "part": {"id": "p3", "type": "step-finish", "reason": "tool-calls"},
            }
        ),
        json.dumps(
            {"type": "step_start", "timestamp": 4, "sessionID": "ses-002", "part": {"id": "p4", "type": "step-start"}}
        ),
        json.dumps(
            {
                "type": "text",
                "timestamp": 5,
                "sessionID": "ses-002",
                "part": {"id": "p5", "type": "text", "text": "The file contains: file contents here"},
            }
        ),
        json.dumps(
            {
                "type": "step_finish",
                "timestamp": 6,
                "sessionID": "ses-002",
                "part": {"id": "p6", "type": "step-finish", "reason": "stop"},
            }
        ),
    ]


# ---------------------------------------------------------------------------
# PiAgent Tests
# ---------------------------------------------------------------------------


class TestPiAgentInit:
    """Test PiAgent initialization and configuration."""

    def test_default_framework(self):
        agent = PiAgent(name="test")
        assert agent.framework == "pi"

    def test_default_id_from_name(self):
        agent = PiAgent(name="My Pi Agent")
        assert agent.get_id() != ""
        assert agent.name == "My Pi Agent"

    def test_custom_config(self):
        agent = PiAgent(
            name="Custom Pi",
            provider="anthropic",
            model="claude-sonnet-4",
            thinking="high",
        )
        assert agent.provider == "anthropic"
        assert agent.model == "claude-sonnet-4"
        assert agent.thinking == "high"


class TestPiAgentBuildArgs:
    """Test CLI argument construction."""

    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    def test_basic_args(self, mock_which):
        agent = PiAgent(name="test")
        args = agent._build_args("hello")
        assert "--mode" in args and "json" in args
        assert "--print" in args
        assert "hello" in args

    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    def test_provider_and_model(self, mock_which):
        agent = PiAgent(name="test", provider="anthropic", model="claude-sonnet-4")
        args = agent._build_args("hello")
        assert "--provider" in args and "anthropic" in args
        assert "--model" in args and "claude-sonnet-4" in args

    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    def test_session_passthrough(self, mock_which):
        agent = PiAgent(name="test")
        args = agent._build_args("hello", session_id="my-session-1")
        assert "--session" in args and "my-session-1" in args

    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    def test_no_session_when_none(self, mock_which):
        agent = PiAgent(name="test")
        args = agent._build_args("hello")
        assert "--session" not in args


class TestPiAgentEventParsing:
    """Test Pi JSON event parsing (static methods)."""

    def test_parse_tool_call_content(self):
        content = [
            {"type": "text", "text": "thinking..."},
            {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a.txt"}},
        ]
        result = PiAgent._parse_tool_call_content(content)
        assert result is not None
        assert result["name"] == "read"
        assert result["arguments"]["path"] == "a.txt"

    def test_parse_tool_call_content_empty(self):
        assert PiAgent._parse_tool_call_content([]) is None
        assert PiAgent._parse_tool_call_content([{"type": "text", "text": "hi"}]) is None

    def test_parse_text_from_content(self):
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        assert PiAgent._parse_text_from_content(content) == "Hello world"

    def test_parse_text_from_string(self):
        assert PiAgent._parse_text_from_content("plain text") == "plain text"

    def test_parse_text_from_empty(self):
        assert PiAgent._parse_text_from_content(None) == ""

    def test_parse_text_skips_non_text(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "thinking", "thinking": "internal"},
        ]
        assert PiAgent._parse_text_from_content(content) == "Hello"


class TestPiAgentNonStreaming:
    """Test non-streaming adapter (subprocess mock)."""

    @pytest.mark.asyncio
    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    async def test_simple_response(self, mock_which):
        agent = PiAgent(name="test-pi")
        fake_proc = _FakeProc(_pi_simple_response())

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            result = await agent._arun_adapter("hi")
        assert result == "Hello world"


class TestPiAgentStreaming:
    """Test streaming adapter (subprocess mock)."""

    @pytest.mark.asyncio
    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    async def test_text_streaming(self, mock_which):
        agent = PiAgent(name="test-pi")
        fake_proc = _FakeProc(_pi_simple_response())

        events = []
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            async for event in agent._arun_adapter_stream("hi"):
                events.append(event)

        content_events = [e for e in events if isinstance(e, RunContentEvent)]
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " world"

    @pytest.mark.asyncio
    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    async def test_tool_call_tracking(self, mock_which):
        agent = PiAgent(name="test-pi")
        fake_proc = _FakeProc(_pi_tool_call_response())

        events = []
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            async for event in agent._arun_adapter_stream("read test.txt"):
                events.append(event)

        started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]

        assert len(started) >= 1
        assert started[0].tool.tool_name == "read"
        assert started[0].tool.tool_args == {"path": "test.txt"}

        assert len(completed) >= 1
        assert completed[0].tool.tool_call_id == "call-001"
        assert "file contents here" in completed[0].tool.result

    @pytest.mark.asyncio
    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    async def test_session_id_in_cli_args(self, mock_which):
        """Verify session_id is passed to the CLI."""
        agent = PiAgent(name="test-pi")
        fake_proc = _FakeProc(_pi_simple_response())

        captured_args = []
        original_exec = __import__("asyncio").create_subprocess_exec

        async def mock_exec(*args, **kwargs):
            captured_args.extend(args)
            return fake_proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await agent._arun_adapter("hi", session_id="my-session")

        assert "--session" in captured_args
        assert "my-session" in captured_args


# ---------------------------------------------------------------------------
# OpenCodeAgent Tests
# ---------------------------------------------------------------------------


class TestOpenCodeAgentInit:
    """Test OpenCodeAgent initialization and configuration."""

    def test_default_framework(self):
        agent = OpenCodeAgent(name="test")
        assert agent.framework == "opencode"

    def test_default_id_from_name(self):
        agent = OpenCodeAgent(name="My OC Agent")
        assert agent.get_id() != ""
        assert agent.name == "My OC Agent"

    def test_custom_config(self):
        agent = OpenCodeAgent(
            name="Custom OC",
            model="anthropic/claude-sonnet-4",
            cwd="/tmp/project",
        )
        assert agent.model == "anthropic/claude-sonnet-4"
        assert agent.cwd == "/tmp/project"


class TestOpenCodeAgentBuildArgs:
    """Test CLI argument construction."""

    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    def test_basic_args(self, mock_which):
        agent = OpenCodeAgent(name="test")
        args = agent._build_args("hello")
        assert "run" in args
        assert "--format" in args and "json" in args
        assert "hello" in args

    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    def test_model_arg(self, mock_which):
        agent = OpenCodeAgent(name="test", model="anthropic/claude-sonnet-4")
        args = agent._build_args("hello")
        assert "--model" in args and "anthropic/claude-sonnet-4" in args

    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    def test_session_passthrough(self, mock_which):
        agent = OpenCodeAgent(name="test")
        args = agent._build_args("hello", session_id="my-session-1")
        assert "--session" in args and "my-session-1" in args

    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    def test_no_session_when_none(self, mock_which):
        agent = OpenCodeAgent(name="test")
        args = agent._build_args("hello")
        assert "--session" not in args


class TestOpenCodeAgentEventParsing:
    """Test OpenCode JSON event parsing."""

    def test_parse_tool_state_valid(self):
        part = {
            "type": "tool",
            "tool": "bash",
            "callID": "call-abc",
            "state": {"status": "completed", "input": {"command": "ls"}, "output": "files..."},
        }
        result = OpenCodeAgent._parse_tool_state(part)
        assert result is not None
        assert result["tool_name"] == "bash"
        assert result["call_id"] == "call-abc"

    def test_parse_tool_state_not_tool(self):
        part = {"type": "text", "text": "hello"}
        assert OpenCodeAgent._parse_tool_state(part) is None


class TestOpenCodeAgentNonStreaming:
    """Test non-streaming adapter (subprocess mock)."""

    @pytest.mark.asyncio
    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    async def test_simple_response(self, mock_which):
        agent = OpenCodeAgent(name="test-oc")
        fake_proc = _FakeProc(_opencode_simple_response())

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            result = await agent._arun_adapter("hello")
        assert result == "Hello world"


class TestOpenCodeAgentStreaming:
    """Test streaming adapter (subprocess mock)."""

    @pytest.mark.asyncio
    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    async def test_text_streaming(self, mock_which):
        agent = OpenCodeAgent(name="test-oc")
        fake_proc = _FakeProc(_opencode_simple_response())

        events = []
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            async for event in agent._arun_adapter_stream("hello"):
                events.append(event)

        content_events = [e for e in events if isinstance(e, RunContentEvent)]
        assert len(content_events) == 1
        assert content_events[0].content == "Hello world"

    @pytest.mark.asyncio
    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    async def test_tool_call_tracking(self, mock_which):
        agent = OpenCodeAgent(name="test-oc")
        fake_proc = _FakeProc(_opencode_tool_call_response())

        events = []
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            async for event in agent._arun_adapter_stream("read test.txt"):
                events.append(event)

        started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        content_events = [e for e in events if isinstance(e, RunContentEvent)]

        assert len(started) == 1
        assert started[0].tool.tool_name == "read"
        assert started[0].tool.tool_call_id == "call-001"

        assert len(completed) == 1
        assert completed[0].tool.tool_call_id == "call-001"
        assert "file contents here" in completed[0].tool.result

        assert len(content_events) == 1
        assert "file contents here" in content_events[0].content

    @pytest.mark.asyncio
    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    async def test_session_id_in_cli_args(self, mock_which):
        """Verify session_id is passed to the CLI."""
        agent = OpenCodeAgent(name="test-oc")
        fake_proc = _FakeProc(_opencode_simple_response())

        captured_args = []
        original_exec = __import__("asyncio").create_subprocess_exec

        async def mock_exec(*args, **kwargs):
            captured_args.extend(args)
            return fake_proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            await agent._arun_adapter("hello", session_id="my-session")

        assert "--session" in captured_args
        assert "my-session" in captured_args


# ---------------------------------------------------------------------------
# BaseExternalAgent integration tests
# ---------------------------------------------------------------------------


class TestBaseExternalAgentIntegration:
    """Test that PiAgent/OpenCodeAgent work with BaseExternalAgent's run methods."""

    @pytest.mark.asyncio
    @patch("agno.agents.pi.agent.shutil.which", return_value="/usr/bin/pi")
    async def test_pi_arun_non_stream(self, mock_which):
        agent = PiAgent(name="test-pi")
        fake_proc = _FakeProc(_pi_simple_response())

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            result = await agent._arun_non_stream("hi")

        assert result.content == "Hello world"
        assert result.run_id is not None

    @pytest.mark.asyncio
    @patch("agno.agents.opencode.agent.shutil.which", return_value="/usr/bin/opencode")
    async def test_opencode_arun_non_stream(self, mock_which):
        agent = OpenCodeAgent(name="test-oc")
        fake_proc = _FakeProc(_opencode_simple_response())

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            result = await agent._arun_non_stream("hello")

        assert result.content == "Hello world"
        assert result.run_id is not None
