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


def _find_opencode_bin() -> str:
    """Locate the ``opencode`` binary on the system PATH."""
    oc_bin = shutil.which("opencode")
    if oc_bin is None:
        raise FileNotFoundError(
            "The 'opencode' CLI is required but not found on PATH. Install it: https://github.com/opencode-ai/opencode"
        )
    return oc_bin


@dataclass
class OpenCodeAgent(BaseExternalAgent):
    """Adapter for the OpenCode CLI.

    Wraps the ``opencode run`` command (``--format json``) so it can be used
    with AgentOS endpoints or standalone via ``.run()`` / ``.print_response()``.

    OpenCode runs as a subprocess.  Tool execution is handled internally by the
    OpenCode CLI — you configure tools via MCP servers in OpenCode's config.

    The adapter parses OpenCode's JSON event stream and emits Agno
    ``RunOutputEvent`` objects including full tool call tracking.

    Session management: pass ``session_id`` to ``run()`` / ``arun()`` to reuse
    an OpenCode session across multiple calls.  The same ``session_id`` is
    forwarded to ``opencode run --session`` directly — no mapping layer needed.

    Note:
        OpenCode's ``--format json`` outputs text in complete blocks (not
        token-level deltas), so streaming granularity is per-step rather than
        per-token.

    Args:
        name: Display name for this agent.
        id: Unique identifier (auto-generated from name if not set).
        model: Model in ``provider/model`` format (e.g. ``"anthropic/claude-sonnet-4"``).
        agent: Agent name to use (OpenCode supports multiple agent configurations).
        cwd: Working directory for the agent.
        permission_mode: Permission mode for tool execution.  Pass
            ``"dangerously-skip-permissions"`` to auto-approve.
        extra_args: Additional CLI flags forwarded to the ``opencode run`` invocation.

    Example:
        from agno.agents.opencode import OpenCodeAgent

        agent = OpenCodeAgent(
            name="OpenCode Coder",
            model="anthropic/claude-sonnet-4",
            cwd="/path/to/project",
        )

        # Single turn
        agent.print_response("Read main.py and summarize it", stream=True)

        # Multi-turn with session
        agent.arun("Hello", session_id="my-session")
        agent.arun("What did we discuss?", session_id="my-session")
    """

    model: Optional[str] = None
    agent: Optional[str] = None
    cwd: Optional[str] = None
    permission_mode: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    framework: str = "opencode"

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def _build_args(self, input: str, *, session_id: Optional[str] = None, **kwargs: Any) -> List[str]:
        """Build the ``opencode run`` CLI argument list."""
        args = [
            _find_opencode_bin(),
            "run",
            "--format",
            "json",
        ]

        if self.model:
            args += ["--model", self.model]
        if self.agent:
            args += ["--agent", self.agent]
        if session_id:
            args += ["--session", session_id]
        if self.permission_mode:
            args += ["--dangerously-skip-permissions"]

        args.extend(self.extra_args)
        args.append(input)
        return args

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_state(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract tool call info from an OpenCode ``tool_use`` part."""
        if part.get("type") != "tool":
            return None
        return {
            "tool_name": part.get("tool", "unknown"),
            "call_id": part.get("callID", str(uuid4())),
            "state": part.get("state", {}),
        }

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
            cwd=self.cwd,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode and proc.returncode != 0:
            raise RuntimeError(f"OpenCode CLI exited with code {proc.returncode}: {stderr.decode()}")

        content = ""
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "text":
                text = event.get("part", {}).get("text", "")
                if text:
                    content += text

        return content

    async def _arun_adapter_stream(
        self, input: Any, *, history: Optional[List[Dict[str, Any]]] = None, **kwargs: Any
    ) -> AsyncIterator[RunOutputEvent]:
        """Streaming: parse OpenCode JSON events and yield Agno events."""
        session_id = kwargs.get("session_id")
        run_id = kwargs.get("run_id", str(uuid4()))
        args = self._build_args(str(input), session_id=session_id)

        # Track tool calls: call_id -> tool_name
        tool_name_map: Dict[str, str] = {}

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
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
                part = event.get("part", {})

                # --- Tool use events ---
                if event_type == "tool_use":
                    tool_info = self._parse_tool_state(part)
                    if tool_info:
                        call_id = tool_info["call_id"]
                        tool_name = tool_info["tool_name"]
                        state = tool_info["state"]
                        tool_input = state.get("input", {})
                        tool_args = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
                        tool_name_map[call_id] = tool_name

                        yield ToolCallStartedEvent(
                            run_id=run_id,
                            agent_id=self.get_id(),
                            agent_name=self.name or "",
                            tool=ToolExecution(
                                tool_call_id=call_id,
                                tool_name=tool_name,
                                tool_args=tool_args,
                            ),
                        )

                        output = state.get("output", "")
                        if output:
                            yield ToolCallCompletedEvent(
                                run_id=run_id,
                                agent_id=self.get_id(),
                                agent_name=self.name or "",
                                tool=ToolExecution(
                                    tool_call_id=call_id,
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    result=str(output),
                                ),
                            )

                # --- Text events (complete blocks, not token-level) ---
                elif event_type == "text":
                    text = part.get("text", "")
                    if text:
                        yield RunContentEvent(
                            run_id=run_id,
                            agent_id=self.get_id(),
                            agent_name=self.name or "",
                            content=text,
                        )
        finally:
            await proc.wait()
