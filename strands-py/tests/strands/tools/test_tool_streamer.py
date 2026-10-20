"""Tests for nested-agent tool stream wrapping."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from strands.interrupt import Interrupt, _InterruptState
from strands.multiagent.base import Status
from strands.multiagent.graph import GraphResult
from strands.multiagent.swarm import SwarmResult
from strands.tools._tool_streamer import (
    _INTERRUPTED_TURNS_KEY,
    _AgentToolAdapter,
    _MultiAgentToolAdapter,
    _ToolStreamer,
)
from strands.types._snapshot import Snapshot

TOOL_USE = {"toolUseId": "t", "name": "tool", "input": {}}


async def _stream(*events: dict[str, Any]):
    """Yield test events."""
    for event in events:
        yield event


def _adapter(child: Mock) -> _AgentToolAdapter:
    """Configure a mocked Agent stream adapter."""
    return _AgentToolAdapter(child, label="Child")


@pytest.mark.parametrize("result_type", [GraphResult, SwarmResult])
def test_multiagent_adapter_classifies_graph_and_swarm_interrupts(result_type):
    adapter = _MultiAgentToolAdapter(Mock(), label="Multi-agent invocation")
    result = result_type(status=Status.INTERRUPTED, interrupts=[Interrupt("approval", "confirm")])

    assert adapter.classify(result) == [Interrupt("approval", "confirm")]


def test_multiagent_adapter_reports_incomplete_result_with_its_label():
    adapter = _MultiAgentToolAdapter(Mock(), label="Swarm")
    result = SwarmResult(status=Status.FAILED, execution_count=2)

    assert adapter.classify(result) == (
        f"Swarm stopped before completing (status=failed, 2 iterations). Partial output:\n{result}"
    )


def test_multiagent_adapter_round_trips_orchestrator_state():
    source = Mock(id="swarm")
    source.serialize_state.return_value = {"status": "interrupted"}
    target = Mock(id="swarm", _interrupt_state=_InterruptState(activated=True))
    snapshot = _MultiAgentToolAdapter(source, label="Swarm").take_snapshot()

    awaiting_responses = _MultiAgentToolAdapter(target, label="Swarm").restore(Snapshot.from_dict(snapshot.to_dict()))

    assert awaiting_responses is True
    target.deserialize_state.assert_called_once_with({"status": "interrupted"})


@pytest.mark.asyncio
async def test_stream_skips_terminal_result_and_retains_it(alist):
    child = Mock()
    result = Mock(stop_reason="completed", interrupts=[])
    tool_streamer = _ToolStreamer(
        interrupt_state=None,
        tool_use_id=TOOL_USE["toolUseId"],
        namespace="v1:test",
    )

    tru_events = await alist(tool_streamer.stream(_adapter(child), _stream({"text": "progress"}, {"result": result})))

    assert tru_events == [{"text": "progress"}]
    assert tool_streamer.result is result


@pytest.mark.asyncio
async def test_stream_stores_and_namespaces_child_interrupts(alist):
    child = Mock()
    child.take_snapshot.return_value = Snapshot(scope="agent", schema_version="1.0", data={}, app_data={})
    result = Mock(stop_reason="interrupt", interrupts=[Interrupt("approval", "confirm")])
    tool_streamer = _ToolStreamer(
        interrupt_state=_InterruptState(),
        tool_use_id=TOOL_USE["toolUseId"],
        namespace="v1:test",
    )

    tru_events = await alist(tool_streamer.stream(_adapter(child), _stream({"result": result})))

    assert tru_events == []
    assert tool_streamer.interrupts == [Interrupt("v1:test:t:approval", "confirm")]
    assert set(tool_streamer._interrupt_state.context[_INTERRUPTED_TURNS_KEY]) == {"t"}  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_stream_returns_error_when_child_does_not_produce_a_result(alist):
    child = Mock()
    tool_streamer = _ToolStreamer(interrupt_state=None, tool_use_id=TOOL_USE["toolUseId"], namespace="v1:test")

    tru_events = await alist(tool_streamer.stream(_adapter(child), _stream({"text": "progress"})))

    assert tru_events == [
        {"text": "progress"},
    ]
    assert tool_streamer.error == "Child did not produce a result."


def test_resume_reraises_unloadable_turns_that_still_await_an_approval():
    interrupt = Interrupt("v1:test:t:approval", "confirm", response="approved")
    awaited = _InterruptState(interrupts={"approval": Interrupt("approval", "confirm")}, activated=True)
    parent_state = _InterruptState(
        interrupts={interrupt.id: interrupt},
        context={
            _INTERRUPTED_TURNS_KEY: {
                "t": {
                    "scope": "agent",
                    "schema_version": "0.0",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "data": {"interrupt_state": awaited.to_dict()},
                    "app_data": {},
                }
            }
        },
        activated=True,
    )
    child = Mock()
    tool_streamer = _ToolStreamer(interrupt_state=parent_state, tool_use_id=TOOL_USE["toolUseId"], namespace="v1:test")

    responses = tool_streamer.resume(_adapter(child), "unavailable")

    assert responses is None
    assert tool_streamer.interrupts == [Interrupt(interrupt.id, "confirm")]
    assert parent_state.context[_INTERRUPTED_TURNS_KEY]["t"]["schema_version"] == "0.0"


@pytest.mark.parametrize("turn", [None, "garbage", {"data": "x"}, {"data": {"interrupt_state": {"interrupts": 1}}}])
def test_resume_returns_error_when_saved_turn_is_missing_or_unreadable(turn):
    interrupt = Interrupt("v1:test:t:approval", "confirm", response="approved")
    parent_state = _InterruptState(
        interrupts={interrupt.id: interrupt},
        context={_INTERRUPTED_TURNS_KEY: {} if turn is None else {"t": turn}},
        activated=True,
    )
    child = Mock()
    child._interrupt_state = SimpleNamespace(activated=False)
    tool_streamer = _ToolStreamer(interrupt_state=parent_state, tool_use_id=TOOL_USE["toolUseId"], namespace="v1:test")

    responses = tool_streamer.resume(_adapter(child), "unavailable")

    assert responses is None
    assert tool_streamer.error == "unavailable"


@pytest.mark.asyncio
async def test_stream_clears_completed_child_turn_and_preserves_siblings(alist):
    parent_state = _InterruptState(context={_INTERRUPTED_TURNS_KEY: {"t": {"snapshot": "child"}, "t2": {}}})
    child = Mock()
    child.cancel_signal.is_set.return_value = False
    result = Mock(stop_reason="end_turn", interrupts=[])
    tool_streamer = _ToolStreamer(interrupt_state=parent_state, tool_use_id=TOOL_USE["toolUseId"], namespace="v1:test")

    await alist(tool_streamer.stream(_adapter(child), _stream({"result": result})))

    assert parent_state.context[_INTERRUPTED_TURNS_KEY] == {"t2": {}}
