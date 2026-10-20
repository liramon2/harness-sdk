"""Wrapper around agent streaming in tool calls.

Provides common functionality for streaming agents within tool calls, such as session persistence
(via the parent agent's interrupt context), interrupt response forwarding, event conversion,
and cancellation support. These features are implemented around stream calls (as opposed to within them)
because they require knowledge of the parent agent.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import quote

from ..interrupt import Interrupt, _InterruptState
from ..types._events import ToolInterruptEvent, ToolResultEvent
from ..types._snapshot import Snapshot
from ..types.tools import ToolResultContent, ToolUse

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..multiagent.base import MultiAgentBase
    from ..types.interrupt import InterruptResponseContent

logger = logging.getLogger(__name__)

_RESPONSES_KEY = "responses"
_INTERRUPTED_TURNS_KEY = "sub_agent_interrupted_turns"


class _ToolAdapter(Protocol):
    """Provides persistence and result handling that differs between tool agent types."""

    def take_snapshot(self) -> Snapshot:
        """Capture snapshot needed to resume the child invocation."""
        ...

    def restore(self, snapshot: Snapshot | None) -> bool:
        """Restore snapshot and report whether it awaits interrupt responses."""
        ...

    def classify(self, result: Any) -> list[Interrupt] | str | None:
        """Return interrupts, a failure message, or None for a successful result."""
        ...


class _AgentToolAdapter:
    """Provide Agent-specific operations to a tool streamer."""

    def __init__(
        self,
        agent: Agent,
    ) -> None:
        self._agent = agent

    def take_snapshot(self) -> Snapshot:
        return self._agent.take_snapshot(preset="session")

    def restore(self, snapshot: Snapshot | None) -> bool:
        if snapshot is not None:
            self._agent.load_snapshot(snapshot)
        return self._agent._interrupt_state.activated

    def classify(self, result: Any) -> list[Interrupt] | str | None:
        if result.stop_reason == "interrupt" and result.interrupts:
            return list(result.interrupts)
        if result.stop_reason == "cancelled" and self._agent.cancel_signal.is_set():
            return "Nested agent was cancelled."
        return None


class _MultiAgentToolAdapter:
    """Provide MultiAgentBase-specific operations to a tool streamer."""

    def __init__(
        self,
        orchestrator: MultiAgentBase,
        *,
        label: str,
    ) -> None:
        self._orchestrator = orchestrator
        self._label = label

    def take_snapshot(self) -> Snapshot:
        from ..multiagent._snapshot import take_snapshot

        return take_snapshot(self._orchestrator)

    def restore(self, snapshot: Snapshot | None) -> bool:
        if snapshot is not None:
            from ..multiagent._snapshot import load_snapshot

            load_snapshot(self._orchestrator, snapshot)
        interrupt_state = cast(_InterruptState, cast(Any, self._orchestrator)._interrupt_state)
        return interrupt_state.activated

    def classify(self, result: Any) -> list[Interrupt] | str | None:
        """Classify a MultiAgentResult."""
        from ..multiagent.base import Status

        if result.status is Status.INTERRUPTED and result.interrupts:
            return list(result.interrupts)
        if result.status is not Status.COMPLETED:
            return (
                f"{self._label} stopped before completing (status={result.status.value}, "
                f"{result.execution_count} iterations). Partial output:\n{result}"
            )
        return None


class _ToolStreamer:
    """Stream a child invocation through the parent tool lifecycle."""

    def __init__(
        self,
        *,
        interrupt_state: _InterruptState | None,
        tool_use_id: str,
        namespace: str,
        persist_interrupted_turn: bool = True,
    ) -> None:
        """Initialize tool-stream state for one child invocation."""
        self._interrupt_state = interrupt_state
        self._tool_use_id = tool_use_id
        self._prefix = f"{namespace}:{quote(tool_use_id, safe='')}:"
        self._persist_interrupted_turn = persist_interrupted_turn
        self.result: Any | None = None
        self.interrupts: list[Interrupt] | None = None
        self.error: str | None = None
        self._outcome: list[Interrupt] | str | None = None

    async def stream(
        self,
        adapter: _ToolAdapter,
        events: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield child progress events and retain terminal state for the tool."""
        async for event in events:
            if "result" in event:
                self.result = event["result"]
                self._outcome = adapter.classify(self.result)
                continue
            yield event

        if self.result is None:
            self.error = "Nested executor did not produce a result (no result)."
            return

        outcome = self._outcome
        if isinstance(outcome, list):
            if self._interrupt_state is not None:
                if self._persist_interrupted_turn:
                    turns = self._interrupt_state.context.setdefault(_INTERRUPTED_TURNS_KEY, {})
                    turns[self._tool_use_id] = copy.deepcopy(adapter.take_snapshot().to_dict())
                    self._interrupt_state._version += 1
                outcome = [
                    Interrupt(f"{self._prefix}{interrupt.id}", interrupt.name, interrupt.reason)
                    for interrupt in outcome
                ]
            self.interrupts = outcome
            return

        if self._interrupt_state is not None:
            self._clear_interrupted_turn()
        self.error = outcome

    def terminal_event(
        self,
        tool_use: ToolUse,
        render_success: Callable[[Any], list[ToolResultContent]] | None = None,
    ) -> ToolInterruptEvent | ToolResultEvent:
        """Return the terminal tool event for the child invocation."""
        if self.interrupts is not None:
            return ToolInterruptEvent(tool_use, self.interrupts)
        if self.error is not None:
            return ToolResultEvent(
                {
                    "toolUseId": tool_use["toolUseId"],
                    "status": "error",
                    "content": [{"text": self.error}],
                }
            )
        assert self.result is not None
        return ToolResultEvent(
            {
                "toolUseId": tool_use["toolUseId"],
                "status": "success",
                "content": [{"text": str(self.result)}] if render_success is None else render_success(self.result),
            }
        )

    def is_resuming(self) -> bool:
        """Return whether this tool call has a child turn to resume."""
        return (
            self._interrupt_state is not None
            and self._interrupt_state.activated
            and any(interrupt_id.startswith(self._prefix) for interrupt_id in self._interrupt_state.interrupts)
        )

    def resume(
        self,
        adapter: _ToolAdapter | None,
        unavailable_message: str,
    ) -> list[InterruptResponseContent] | None:
        """Restore the child invocation and return its scoped interrupt responses."""
        if not self.is_resuming():
            return None
        assert self._interrupt_state is not None

        if adapter is None:
            self._handle_failed_restore(unavailable_message)
            return None
        turns = self._interrupt_state.context.get(_INTERRUPTED_TURNS_KEY) or {}
        turn = turns.get(self._tool_use_id)
        try:
            snapshot = Snapshot.from_dict(turn) if turn is not None else None
            awaiting_responses = adapter.restore(snapshot)
        except Exception as error:
            logger.error(
                "tool_use_id=<%s>, error=<%s> | failed to restore interrupted sub-agent turn",
                self._tool_use_id,
                error,
            )
            self._handle_failed_restore(unavailable_message)
            return None

        if turn is not None:
            self._clear_interrupted_turn()
        if not awaiting_responses:
            logger.error(
                "tool_use_id=<%s> | cannot resume: the nested executor turn is not available", self._tool_use_id
            )
            self.error = unavailable_message
            return None
        return [
            {
                "interruptResponse": {
                    "interruptId": response["interruptResponse"]["interruptId"][len(self._prefix) :],
                    "response": response["interruptResponse"]["response"],
                }
            }
            for response in self._interrupt_state.context.get(_RESPONSES_KEY) or []
            if response["interruptResponse"]["interruptId"].startswith(self._prefix)
        ]

    def _handle_failed_restore(self, unavailable_message: str) -> None:
        """Re-raise pending interrupts when the child state cannot be restored."""
        if self._interrupt_state is None:
            self.error = unavailable_message
            return
        # Get all pending interrupt ids for this agent
        turn = (self._interrupt_state.context.get(_INTERRUPTED_TURNS_KEY) or {}).get(self._tool_use_id)
        try:
            pending = _InterruptState.from_dict(turn["data"]["interrupt_state"]).interrupts  # type: ignore[index]
        except Exception:
            logger.error("tool_use_id=<%s> | stored turn has no readable interrupt state", self._tool_use_id)
            self.error = unavailable_message
            return
        # Map interrupt ids to parent interrupts
        self.interrupts = [
            interrupt
            for interrupt_id, interrupt in self._interrupt_state.interrupts.items()
            if interrupt_id.startswith(self._prefix) and interrupt_id[len(self._prefix) :] in pending
        ]
        if not self.interrupts:
            self.interrupts = None
            self.error = unavailable_message
            return
        # Empty responses so that this agent's interrupts remain on parent's interrupt state
        for interrupt in self.interrupts:
            interrupt.response = None

    def _clear_interrupted_turn(self) -> None:
        """Discard stored child state after a non-interrupt terminal result."""
        if self._interrupt_state is None:
            return
        turns = self._interrupt_state.context.get(_INTERRUPTED_TURNS_KEY)
        if not isinstance(turns, dict) or turns.pop(self._tool_use_id, None) is None:
            return
        self._interrupt_state._version += 1
