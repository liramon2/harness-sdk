"""Tests for the swarm vended tool.

The swarm tool is a thin shim over :class:`~strands.multiagent.Swarm`: it owns
spec validation, agent construction, and result mapping. Tests mock the SDK
Swarm class so no model calls are made.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from strands import Agent
from strands.agent import AgentResult
from strands.agent.state import AgentState
from strands.experimental.tools import stop
from strands.multiagent.base import NodeResult, Status
from strands.multiagent.spec import Choice, Fixed, Inherit, Open, Preset, _append_presets
from strands.multiagent.swarm import SwarmResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.registry import ToolRegistry
from strands.types._events import ToolResultEvent, ToolStreamEvent
from strands.types.tools import ToolContext
from strands.vended_tools.swarm import make_swarm, swarm
from strands.vended_tools.swarm.swarm import (
    _DEFAULT_MAX_AGENTS,
    _DEPTH_STATE_KEY,
    DEFAULT_SWARM_DESCRIPTION,
    _build_schema,
)
from tests.fixtures.mocked_model_provider import MockedModelProvider

_swarm_module = importlib.import_module("strands.vended_tools.swarm.swarm")


def _mock_parent(tool_names: list[str] | None = None, state: dict | None = None) -> SimpleNamespace:
    registry = ToolRegistry()
    if tool_names:
        for name in tool_names:
            t = Mock()
            t.tool_name = name
            registry.registry[name] = t
    return SimpleNamespace(
        tool_registry=registry,
        model=Mock(),
        sandbox=None,
        callback_handler=None,
        trace_attributes=None,
        state=AgentState(state),
    )


def _ctx(parent=None):
    if parent is None:
        parent = _mock_parent()
    return ToolContext(tool_use={"name": "swarm", "toolUseId": "id", "input": {}}, agent=parent, invocation_state={})


async def _run(swarm_tool, **kwargs):
    """Drain the tool's async generator; return (streamed events, final result)."""
    yielded = [event async for event in swarm_tool(**kwargs)]
    return yielded[:-1], yielded[-1]


def _spec(name, **kw):
    return {"name": name, **kw}


def _result(*, status=Status.COMPLETED, text="Done!", node="writer"):
    r = AgentResult(
        message={"role": "assistant", "content": [{"text": text}]},
        stop_reason="end_turn",
        state={},
        metrics=EventLoopMetrics(),
    )
    sr = SwarmResult(status=status, results={node: NodeResult(result=r, status=Status.COMPLETED)}, execution_count=1)
    return sr


def _kwargs():
    return dict(
        max_agents=_DEFAULT_MAX_AGENTS,
        presets={},
        default_preset=None,
        instructions=Open(),
        tools=None,
        mcp_servers=Inherit(),
        model=Inherit(),
    )


def _patch(result=None, events=()):
    """Patch Swarm and _default_builder so no model calls are made.

    The mocked ``Swarm.stream_async`` yields ``events`` then the final result event.
    """
    if result is None:
        result = _result()

    async def _stream(task, invocation_state=None):
        for event in events:
            yield event
        yield {"type": "multiagent_result", "result": result}

    sp = patch.object(_swarm_module, "Swarm")

    def _builder(spec):
        child = Mock(name=spec.name)
        child.state = AgentState()
        child.tool_registry = ToolRegistry()
        return child

    bp = patch.object(_swarm_module, "_default_builder", return_value=_builder)

    class _Ctx:
        def __enter__(self):
            self.cls = sp.__enter__()
            bp.__enter__()
            self.cls.return_value.stream_async = Mock(side_effect=_stream)
            return self.cls

        def __exit__(self, *a):
            bp.__exit__(*a)
            sp.__exit__(*a)

    return _Ctx()


def _item_schema(default_preset=None, **axes):
    return _build_schema(max_agents=1, default_preset=default_preset, **axes)["properties"]["agents"]["items"]


class TestBuildSchema:
    def test_visible_and_hidden_axes(self):
        # Open + no presets → instructions required.
        schema = _item_schema(presets={}, instructions=Open(), tools=None, mcp_servers=Inherit(), model=None)
        assert schema["required"] == ["name", "instructions"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["instructions"]["type"] == "string"

        # Open + presets → instructions not required (preset provides a default).
        with_presets = _item_schema(
            presets={"w": Preset(description="W")},
            instructions=Open(),
            tools=None,
            mcp_servers=Inherit(),
            model=None,
        )
        assert with_presets["required"] == ["name"]

        # Fixed/Inherit → hidden.
        hidden = _item_schema(
            presets={},
            instructions=Fixed("x"),
            tools=Inherit(),
            mcp_servers=Inherit(),
            model=Fixed("m"),
        )
        assert set(hidden["properties"]) == {"name"}

    def test_choice_axes_and_presets(self):
        presets = {"alpha": Preset(description="First"), "beta": Preset(description="Second")}
        schema = _item_schema(
            presets=presets,
            instructions=Choice(["A", "B"]),
            tools=Choice(["calc", "fetch"], multiple=True),
            mcp_servers=Choice(["docs", "github"], multiple=True),
            model=Choice(["fast", "smart"]),
        )
        assert schema["properties"]["instructions"]["enum"] == ["A", "B"]
        assert schema["properties"]["tools"]["items"]["enum"] == ["calc", "fetch"]
        assert schema["properties"]["mcp_servers"]["items"]["enum"] == ["docs", "github"]
        assert schema["properties"]["model"]["enum"] == ["fast", "smart"]
        agent_type = schema["properties"]["agent_type"]
        assert agent_type["enum"] == ["alpha", "beta"]
        assert "- alpha: First" in agent_type["description"]

    def test_presets_without_descriptions(self):
        schema = _item_schema(
            presets={"w": Preset()}, instructions=Open(), tools=None, mcp_servers=Inherit(), model=None
        )
        assert schema["properties"]["agent_type"]["description"] == "Role to assign this agent."


class TestMakeSwarm:
    def test_default_instance_and_exports(self):
        from strands.tools.decorator import DecoratedFunctionTool
        from strands.vended_tools import make_swarm as ms
        from strands.vended_tools import swarm as s

        assert s is swarm and ms is make_swarm
        assert isinstance(swarm, DecoratedFunctionTool) and swarm.tool_name == "swarm"
        assert make_swarm(name="team").tool_name == "team"

    @pytest.mark.parametrize(
        "kw",
        [
            {"max_agents": 0},
            {"max_agents": True},
            {"max_depth": 0},
            {"max_depth": True},
            {"tools": Choice([], multiple=True)},
            {"tools": Choice(["a"])},  # multiple=False on a list axis
            {"mcp_servers": Choice([], multiple=True)},
            {"mcp_servers": Choice(["a"])},  # multiple=False on a list axis
        ],
    )
    def test_rejects_invalid_limits(self, kw):
        with pytest.raises(ValueError):
            make_swarm(**kw)

    def test_default_spec(self):
        agents = swarm.tool_spec["inputSchema"]["json"]["properties"]["agents"]
        assert agents["minItems"] == 1 and agents["maxItems"] == _DEFAULT_MAX_AGENTS
        assert (
            "Defaults to the first agent"
            in swarm.tool_spec["inputSchema"]["json"]["properties"]["entry_point"]["description"]
        )
        assert swarm.tool_spec["inputSchema"]["json"] == _build_schema(
            presets={},
            default_preset=None,
            instructions=Open(),
            tools=None,
            mcp_servers=None,
            model=Inherit(),
            max_agents=_DEFAULT_MAX_AGENTS,
        )
        assert swarm.tool_spec["description"] == DEFAULT_SWARM_DESCRIPTION

    def test_spec_reflects_configuration(self):
        """The factory wires its axes, presets, and max_agents into the schema and description."""
        axes = dict(
            presets={"writer": Preset(description="Writes.")},
            default_preset="writer",
            instructions=Choice(["A"]),
            tools=Choice(["calc"], multiple=True),
            mcp_servers=Choice(["docs", "gh"], multiple=True),
            model=Choice(["fast"]),
        )
        t = make_swarm(max_agents=3, **axes)
        agents = t.tool_spec["inputSchema"]["json"]["properties"]["agents"]
        assert agents["maxItems"] == 3
        assert "Omit to use the default ('writer')." in agents["items"]["properties"]["agent_type"]["description"]
        assert t.tool_spec["inputSchema"]["json"] == _build_schema(max_agents=3, **axes)
        assert t.tool_spec["description"] == _append_presets(DEFAULT_SWARM_DESCRIPTION, axes["presets"])


class TestSwarmToolExecution:
    @pytest.mark.asyncio
    async def test_streams_events_and_returns_result(self):
        """Through the decorator: swarm events surface as ToolStreamEvents, final text as the tool result."""
        events = [{"type": "multiagent_node_start", "node_id": "a", "node_type": "agent"}]
        tool_use = {
            "toolUseId": "t1",
            "name": "swarm",
            "input": {"task": "go", "agents": [_spec("a", instructions="Do.")]},
        }
        parent = _mock_parent()
        invocation_state = {"agent": parent, "user_key": "v", "request_state": {}}
        with _patch(events=events) as cls:
            out = [e async for e in swarm.stream(tool_use, invocation_state)]
        # The parent's request_state is not forwarded to children.
        cls.return_value.stream_async.assert_called_once_with("go", invocation_state={"agent": parent, "user_key": "v"})
        # The decorator also streams the final yield before wrapping it as the result.
        streamed = [e["tool_stream_event"]["data"] for e in out if isinstance(e, ToolStreamEvent)]
        assert streamed == [*events, "writer: Done!"]
        assert isinstance(out[-1], ToolResultEvent)
        assert out[-1].tool_result["status"] == "success"
        assert out[-1].tool_result["content"] == [{"text": "writer: Done!"}]

    @pytest.mark.asyncio
    async def test_forwards_limits(self):
        limits = dict(
            max_handoffs=5,
            max_iterations=10,
            execution_timeout=60.0,
            node_timeout=30.0,
            repetitive_handoff_detection_window=4,
            repetitive_handoff_min_unique_agents=2,
        )
        with _patch() as cls:
            await _run(make_swarm(**limits), task="t", agents=[_spec("a", instructions="Do.")], tool_context=_ctx())
        assert {k: cls.call_args[1][k] for k in limits} == limits

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agents,kw,match",
        [
            ([], {}, "Between 1 and 2"),
            ([_spec("a"), _spec("b"), _spec("c")], {}, "Between 1 and 2"),
            ([{}], {}, "non-empty 'name'"),
            ([_spec("a", instructions="Do.")], {"entry_point": "z"}, "entry_point 'z'"),
        ],
    )
    async def test_rejects_invalid_agents(self, agents, kw, match):
        with pytest.raises(ValueError, match=match):
            await _run(make_swarm(max_agents=2), task="t", agents=agents, tool_context=_ctx(), **kw)

    @pytest.mark.asyncio
    async def test_failed_status_raises(self):
        with _patch(_result(status=Status.FAILED, text="partial")), pytest.raises(RuntimeError) as exc:
            await _run(swarm, task="t", agents=[_spec("a", instructions="Do.")], tool_context=_ctx())
        assert "status=failed" in str(exc.value) and "partial" in str(exc.value)

    @pytest.mark.asyncio
    async def test_missing_result_raises(self):
        with _patch() as cls:

            async def _no_result(task, invocation_state=None):
                yield {"type": "multiagent_node_start", "node_id": "a", "node_type": "agent"}

            cls.return_value.stream_async = Mock(side_effect=_no_result)
            with pytest.raises(RuntimeError, match="without producing a result"):
                await _run(swarm, task="t", agents=[_spec("a", instructions="Do.")], tool_context=_ctx())


class TestEndToEnd:
    def test_child_stop_does_not_halt_parent(self):
        def tool_use(name, tool_input, tool_use_id):
            return {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}],
            }

        # Parent and child share the inherited model, so responses are consumed in call order.
        model = MockedModelProvider(
            [
                tool_use("swarm", {"task": "go", "agents": [_spec("a", instructions="Do.")]}, "p1"),
                tool_use("stop", {"message": "child done"}, "c1"),
                {"role": "assistant", "content": [{"text": "parent final"}]},
            ]
        )
        result = Agent(model=model, tools=[swarm, stop], callback_handler=None)("start")
        assert result.stop_reason == "end_turn"
        assert str(result).strip() == "parent final"


class TestChildren:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("stored_depth,expected_depth,entry_point", [(None, 2, None), (2, 1, "b")])
    async def test_builds_children(self, stored_depth, expected_depth, entry_point):
        """Children get resolved specs and decremented depth, lose handoff_to_agent, and honor entry_point."""
        specs, children = [], []

        def builder(spec):
            specs.append(spec)
            child = Mock(name=spec.name)
            child.state = AgentState()
            child.tool_registry = ToolRegistry()
            child.tool_registry.registry["handoff_to_agent"] = Mock()
            children.append(child)
            return child

        state = {} if stored_depth is None else {_DEPTH_STATE_KEY: stored_depth}
        custom = make_swarm(builder=builder, instructions=Fixed("Locked."), max_depth=3)
        with _patch() as cls:
            await _run(
                custom,
                task="t",
                agents=[_spec("a"), _spec("b")],
                tool_context=_ctx(_mock_parent(state=state)),
                entry_point=entry_point,
            )

        assert [(s.name, s.instructions) for s in specs] == [("a", "Locked."), ("b", "Locked.")]
        assert cls.call_args[1]["nodes"] == children
        assert cls.call_args[1].get("entry_point") is (children[1] if entry_point else None)
        for child in children:
            assert child.state.get(_DEPTH_STATE_KEY) == expected_depth
            assert "handoff_to_agent" not in child.tool_registry.registry

    @pytest.mark.asyncio
    async def test_preset_reused_across_children(self):
        built = []

        def build(spec):
            built.append(spec)
            child = Mock()
            child.state = AgentState()
            child.tool_registry = ToolRegistry()
            return child

        custom = make_swarm(builder=build, presets={"expert": Preset(instructions="be expert")})
        with _patch() as cls:
            agents = [_spec("a", agent_type="expert"), _spec("b")]
            await _run(custom, task="t", agents=agents, tool_context=_ctx(_mock_parent()))
        assert [(s.name, s.agent_type, s.instructions) for s in built] == [
            ("a", "expert", "be expert"),
            ("b", "expert", "be expert"),
        ]
        assert len(cls.call_args[1]["nodes"]) == 2

    @pytest.mark.asyncio
    async def test_exhausted_depth_raises(self):
        parent = _mock_parent(state={_DEPTH_STATE_KEY: 0})
        with pytest.raises(RuntimeError, match="depth limit reached"):
            await _run(swarm, task="t", agents=[_spec("a", instructions="Do.")], tool_context=_ctx(parent))
