import asyncio
import json
import shutil
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from agno.agents.base import BaseExternalAgent
from agno.models.response import ToolExecution
from agno.run.agent import (
    RunContentEvent,
    RunOutputEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


def _find_pi_bin() -> str:
    """Locate the ``pi`` binary on the system PATH."""
    pi_bin = shutil.which("pi")
    if pi_bin is None:
        raise FileNotFoundError(
            "The 'pi' CLI is required but not found on PATH. "
            "Install it: https://github.com/mariozechner/pi-coding-agent"
        )
    return pi_bin


@dataclass
class PiAgent(BaseExternalAgent):
    """Adapter for the Pi coding agent CLI.

    Wraps the ``pi`` CLI (``--mode json --print``) so it can be used with
    AgentOS endpoints or standalone via ``.run()`` / ``.print_response()``.

    Pi runs as a subprocess.  Tool execution is handled internally by the Pi
    CLI — you configure tools via Pi's own config files.

    The adapter parses Pi's JSON event stream and emits Agno
    ``RunOutputEvent`` objects including token-level text streaming and full
    tool call tracking.

    Session management: pass ``session_id`` to ``run()`` / ``arun()`` to reuse
    a Pi session across multiple calls.  The same ``session_id`` is forwarded
    to ``pi --session`` directly — no mapping layer needed.

    Args:
        name: Display name for this agent.
        id: Unique identifier (auto-generated from name if not set).
        provider: LLM provider name (e.g. ``"anthropic"``, ``"openai"``).
            Passed to ``--provider``.
        model: Model ID or pattern (e.g. ``"claude-sonnet-4"``).
            Passed to ``--model``.
        system_prompt: Optional system prompt forwarded via ``--system-prompt``.
        tools: Comma-separated list of tools to enable (default: all built-in).
        no_tools: Disable all built-in tools.
        thinking: Thinking level (``"off"``, ``"minimal"``, ``"low"``, ``"medium"``,
            ``"high"``, ``"xhigh"``).
        session_dir: Directory for Pi session storage.
        extra_args: Additional CLI flags forwarded to the ``pi`` invocation.

    Example:
        from agno.agents.pi import PiAgent

        agent = PiAgent(
            name="Pi Coder",
            provider="anthropic",
            model="claude-sonnet-4",
        )

        # Single turn
        agent.print_response("Read main.py and summarize it", stream=True)

        # Multi-turn with session
        agent.arun("Hello", session_id="my-session")
        agent.arun("What did we discuss?", session_id="my-session")
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[str] = None
    no_tools: bool = False
    thinking: Optional[str] = None
    session_dir: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    framework: str = "pi"

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def _build_args(self, input: str, *, session_id: Optional[str] = None, **kwargs: Any) -> List[str]:
        """Build the ``pi`` CLI argument list."""
        args = [
            _find_pi_bin(),
            "--mode",
            "json",
            "--print",
        ]

        if self.provider:
            args += ["--provider", self.provider]
        if self.model:
            args += ["--model", self.model]
        if self.system_prompt:
            args += ["--system-prompt", self.system_prompt]
        if self.tools:
            args += ["--tools", self.tools]
        if self.no_tools:
            args.append("--no-tools")
        if self.thinking:
            args += ["--thinking", self.thinking]
        if self.session_dir:
            args += ["--session-dir", self.session_dir]
        if session_id:
            args += ["--session", session_id]

        args.extend(self.extra_args)
        args.append(input)
        return args

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_call_content(content: Any) -> Optional[Dict[str, Any]]:
        """Extract the first toolCall from a Pi assistant message content list."""
        if not isinstance(content, list):
            return None
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                return block
        return None

    @staticmethod
    def _parse_text_from_content(content: Any) -> str:
        """Concatenate all text blocks from a Pi message content list."""
        if not isinstance(content, list):
            return str(content) if content else ""
        parts: list = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    # ------------------------------------------------------------------
    # Adapter implementation
    # ------------------------------------------------------------------

    async def _arun_adapter(self, input: Any, *, history: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> str:
        """Non-streaming: collect all events and return final text."""
        session_id = kwargs.get("session_id")
        args = self._build_args(str(input), session_id=session_id)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode and proc.returncode != 0:
            raise RuntimeError(f"Pi CLI exited with code {proc.returncode}: {stderr.decode()}")

        content = ""
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "turn_end":
                msg = event.get("message", {})
                msg_content = msg.get("content")
                if msg_content:
                    content = self._parse_text_from_content(msg_content) or content

            elif event_type == "agent_end":
                messages = event.get("messages", [])
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        msg_content = msg.get("content")
                        if msg_content:
                            text = self._parse_text_from_content(msg_content)
                            if text:
                                content = text
                                break

        return content

    async def _arun_adapter_stream(
        self, input: Any, *, history: Optional[List[Dict[str, Any]]] = None, **kwargs: Any
    ) -> AsyncIterator[RunOutputEvent]:
        """Streaming: parse Pi JSON events and yield Agno events."""
        session_id = kwargs.get("session_id")
        run_id = kwargs.get("run_id", str(uuid4()))
        args = self._build_args(str(input), session_id=session_id)

        # Track tool calls: tool_call_id -> {name, args}
        tool_info_map: Dict[str, Dict[str, Any]] = {}

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # --- Token-level text streaming ---
                if event_type == "message_update":
                    asm_event = event.get("assistantMessageEvent", {})
                    asm_type = asm_event.get("type", "")

                    if asm_type == "text_delta":
                        delta = asm_event.get("delta", "")
                        if delta:
                            yield RunContentEvent(
                                run_id=run_id,
                                agent_id=self.get_id(),
                                agent_name=self.name or "",
                                content=delta,
                            )

                    elif asm_type == "toolcall_start":
                        pass  # Wait for toolcall_end which has complete args

                    elif asm_type == "toolcall_end":
                        tc = asm_event.get("toolCall", {})
                        tool_id = tc.get("id", str(uuid4()))
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("arguments", {})
                        tool_info_map[tool_id] = {"name": tool_name, "args": tool_args}
                        yield ToolCallStartedEvent(
                            run_id=run_id,
                            agent_id=self.get_id(),
                            agent_name=self.name or "",
                            tool=ToolExecution(
                                tool_call_id=tool_id,
                                tool_name=tool_name,
                                tool_args=tool_args,
                            ),
                        )

                # --- Tool results from message_end (per-turn delivery) ---
                elif event_type == "message_end":
                    msg = event.get("message", {})
                    if msg.get("role") == "toolResult":
                        tool_call_id = msg.get("toolCallId", str(uuid4()))
                        result_content = msg.get("content", "")
                        if isinstance(result_content, list):
                            result_str = " ".join(
                                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                                for item in result_content
                            )
                        else:
                            result_str = str(result_content)
                        info = tool_info_map.get(tool_call_id, {})
                        yield ToolCallCompletedEvent(
                            run_id=run_id,
                            agent_id=self.get_id(),
                            agent_name=self.name or "",
                            tool=ToolExecution(
                                tool_call_id=tool_call_id,
                                tool_name=info.get("name", ""),
                                tool_args=info.get("args"),
                                result=result_str,
                            ),
                        )

                # --- Tool results from agent_end (fallback) ---
                elif event_type == "agent_end":
                    messages = event.get("messages", [])
                    for msg in messages:
                        role = msg.get("role", "")
                        msg_content = msg.get("content")

                        if role == "assistant" and isinstance(msg_content, list):
                            for block in msg_content:
                                if isinstance(block, dict) and block.get("type") == "toolCall":
                                    tool_id = block.get("id", str(uuid4()))
                                    tool_name = block.get("name", "unknown")
                                    tool_args = block.get("arguments", {})
                                    if tool_id not in tool_info_map:
                                        tool_info_map[tool_id] = {"name": tool_name, "args": tool_args}
                                        yield ToolCallStartedEvent(
                                            run_id=run_id,
                                            agent_id=self.get_id(),
                                            agent_name=self.name or "",
                                            tool=ToolExecution(
                                                tool_call_id=tool_id,
                                                tool_name=tool_name,
                                                tool_args=tool_args,
                                            ),
                                        )

                        elif role == "toolResult":
                            tool_call_id = msg.get("toolCallId", str(uuid4()))
                            result_content = msg.get("content", "")
                            if isinstance(result_content, list):
                                result_str = " ".join(
                                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                                    for item in result_content
                                )
                            else:
                                result_str = str(result_content)
                            info = tool_info_map.get(tool_call_id, {})
                            yield ToolCallCompletedEvent(
                                run_id=run_id,
                                agent_id=self.get_id(),
                                agent_name=self.name or "",
                                tool=ToolExecution(
                                    tool_call_id=tool_call_id,
                                    tool_name=info.get("name", ""),
                                    tool_args=info.get("args"),
                                    result=result_str,
                                ),
                            )
        finally:
            await proc.wait()
