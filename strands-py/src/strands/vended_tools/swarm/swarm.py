"""Swarm tool for spinning up a handoff-based sub-agent team at runtime.

Provides :func:`make_swarm` (a factory that lets the developer pin safety
limits and authority modes) and :data:`swarm` (a default instance). The tool is
a thin shim over :class:`~strands.multiagent.Swarm`: it resolves each agent spec
through the authority-mode system, builds agents via an :data:`AgentBuilder`,
runs the swarm, and maps the result.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any

from ...multiagent.base import Status
from ...multiagent.spec import (
    AgentBuilder,
    Choice,
    Fixed,
    Inherit,
    Open,
    Preset,
    _append_presets,
    _default_builder,
    _resolve_specs,
    _to_schema_properties,
)
from ...multiagent.swarm import Swarm, SwarmResult
from ...tools.decorator import tool
from ...types.tools import ToolContext

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)

DEFAULT_SWARM_DESCRIPTION = (
    "Spin up a team of AI agents that solve a task together by handing off to one another. "
    "Each agent has its own name, system prompt, and tools (drawn from your own). "
    "Agents get their own handoff tool; your tools are otherwise passed through. "
    "The `entry_point` agent (default: the first in `agents`) receives the "
    "task and starts the work. The run ends when an agent finishes without handing off. "
    "Returns each participating agent's final output, prefixed with its name. If the team "
    "does not complete (an agent fails, or a handoff, iteration, or time limit is hit), the "
    "call fails with the partial output."
)
"""Description for the default swarm tool."""

_DEFAULT_MAX_AGENTS = 20
"""Upper bound on the number of agents a single swarm invocation may create."""

_DEFAULT_MAX_DEPTH = 2
"""Upper bound on the number of nested swarm levels."""

_DEPTH_STATE_KEY = "strands.swarm_depth"
"""Agent-state key used to propagate remaining nesting depth to children."""

# The SDK Swarm raises ValueError if a node already has handoff_to_agent.
_EXCLUDED_TOOLS = frozenset({"handoff_to_agent"})


def make_swarm(
    *,
    name: str = "swarm",
    description: str = DEFAULT_SWARM_DESCRIPTION,
    builder: AgentBuilder | None = None,
    presets: Mapping[str, Preset] | None = None,
    default_preset: str | None = None,
    instructions: Open | Choice | Fixed | None = None,
    tools: Choice | Fixed | Inherit | None = None,
    mcp_servers: Choice | Fixed | Inherit | None = None,
    model: Inherit | Choice | Fixed | None = None,
    max_agents: int = _DEFAULT_MAX_AGENTS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_handoffs: int | None = None,
    max_iterations: int | None = None,
    execution_timeout: float | None = None,
    node_timeout: float | None = None,
    repetitive_handoff_detection_window: int | None = None,
    repetitive_handoff_min_unique_agents: int | None = None,
) -> DecoratedFunctionTool:
    """Create a swarm tool with developer-controlled safety limits and authority modes.

    Each agent spec from the model is resolved through the authority-mode system
    (see :mod:`~strands.multiagent.spec`), then built via ``builder``. A running swarm
    cannot be cancelled from the parent agent. If no swarm limits are configured, the
    Swarm primitive's defaults are used.

    Args:
        name: Tool name shown to the model.
        description: Tool description shown to the model.
        builder: Callable that turns an :class:`AgentSpec` into an ``Agent``.
            When ``None`` a default builder is used that creates an ``Agent``
            inheriting the caller's model with the spec's instructions and tools.
        presets: Named roles the model can select via ``agent_type``.
        default_preset: Preset used when the model omits ``agent_type``.
        instructions: Authority mode for the instructions axis.
        tools: Authority mode for the tools axis.
        mcp_servers: Authority mode for the MCP servers axis.
        model: Authority mode for the model axis.
        max_agents: Upper bound on agents per invocation.
        max_depth: Maximum nesting depth for recursive swarm calls (>= 1).
        max_handoffs: Maximum handoffs between agents.
        max_iterations: Maximum total agent invocations.
        execution_timeout: Total timeout in seconds.
        node_timeout: Per-agent timeout in seconds.
        repetitive_handoff_detection_window: Window size for repetition detection.
        repetitive_handoff_min_unique_agents: Minimum unique agents in window.

    Returns:
        A decorated tool that spins up and runs a :class:`~strands.multiagent.Swarm`.
    """
    if not isinstance(max_agents, int) or isinstance(max_agents, bool) or max_agents < 1:
        raise ValueError("max_agents must be a positive integer")
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer (>= 1)")

    presets = dict(presets or {})
    instructions = instructions if instructions is not None else Open()
    model = model if model is not None else Inherit()

    if default_preset is None and presets:
        default_preset = next(iter(presets))

    if isinstance(mcp_servers, Choice):
        if not mcp_servers.options:
            raise ValueError("mcp_servers=Choice([]) offers no options; use Fixed([]) for no servers, or Inherit().")
        if not mcp_servers.multiple:
            raise ValueError(
                "mcp_servers=Choice(...) must be multiple=True; a swarm child selects a subset of servers."
            )

    if isinstance(tools, Choice):
        if not tools.options:
            raise ValueError("tools=Choice([]) offers no options; use Fixed([]) for a toolless child, or Inherit().")
        if not tools.multiple:
            raise ValueError("tools=Choice(...) must be multiple=True; a swarm child selects a subset of tools.")

    # Build the model-facing description with preset info.
    tool_description = _append_presets(description, presets)

    input_schema = _build_schema(
        presets=presets,
        default_preset=default_preset,
        instructions=instructions,
        tools=tools,
        mcp_servers=mcp_servers,
        model=model,
        max_agents=max_agents,
    )

    @tool(name=name, description=tool_description, inputSchema={"json": input_schema}, context="tool_context")
    async def swarm_tool(
        task: str,
        agents: list[Mapping[str, Any]],
        tool_context: ToolContext,
        entry_point: str | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Spins up a team of agents that collaborate via handoffs to solve a task.

        Args:
            task: The objective for the agent team.
            agents: Agent specifications (see :func:`make_swarm`).
            tool_context: Injected by the framework. Not user-facing.
            entry_point: Name of the agent that starts the swarm. Defaults to the first agent.
        """
        parent = tool_context.agent

        stored = parent.state.get(_DEPTH_STATE_KEY) if parent else None
        depth = max_depth if stored is None else stored
        if depth <= 0:
            raise RuntimeError("Swarm nesting depth limit reached; complete this task without spawning another swarm.")

        build = builder or _default_builder(parent)

        specs = _resolve_specs(
            agents,
            presets=presets,
            default_preset=default_preset,
            instructions=instructions,
            tools=tools,
            mcp_servers=mcp_servers,
            model=model,
        )
        if not 1 <= len(specs) <= max_agents:
            raise ValueError(f"Between 1 and {max_agents} agents required, got {len(specs)}")
        if entry_point is not None and entry_point not in {spec.name for spec in specs}:
            raise ValueError(f"entry_point {entry_point!r} does not match any agent name")
        child_agents = {}
        for spec in specs:
            child = build(spec)
            child.state.set(_DEPTH_STATE_KEY, depth - 1)
            for tool_name in _EXCLUDED_TOOLS:
                child.tool_registry.registry.pop(tool_name, None)
            child_agents[spec.name] = child

        swarm_kwargs: dict[str, Any] = {"nodes": list(child_agents.values())}
        if entry_point is not None:
            swarm_kwargs["entry_point"] = child_agents[entry_point]
        if max_handoffs is not None:
            swarm_kwargs["max_handoffs"] = max_handoffs
        if max_iterations is not None:
            swarm_kwargs["max_iterations"] = max_iterations
        if execution_timeout is not None:
            swarm_kwargs["execution_timeout"] = execution_timeout
        if node_timeout is not None:
            swarm_kwargs["node_timeout"] = node_timeout
        if repetitive_handoff_detection_window is not None:
            swarm_kwargs["repetitive_handoff_detection_window"] = repetitive_handoff_detection_window
        if repetitive_handoff_min_unique_agents is not None:
            swarm_kwargs["repetitive_handoff_min_unique_agents"] = repetitive_handoff_min_unique_agents

        sdk_swarm = Swarm(**swarm_kwargs)

        logger.info("task=<%s>, agents=<%d> | starting swarm", task[:120], len(child_agents))
        result: SwarmResult | None = None
        # Copy so child cycles can't mutate the parent's state; drop request_state so a child's stop can't halt it.
        child_state = {k: v for k, v in tool_context.invocation_state.items() if k != "request_state"}
        async for event in sdk_swarm.stream_async(task, invocation_state=child_state):
            # Hold back the final result event: the decorator treats the last yield as the tool result.
            if event.get("type") == "multiagent_result":
                result = event["result"]
            else:
                yield event

        if result is None:
            raise RuntimeError("Swarm stream ended without producing a result")
        if result.status != Status.COMPLETED:
            raise RuntimeError(
                f"Swarm stopped before completing (status={result.status.value}, "
                f"{result.execution_count} iterations). Partial output:\n{result}"
            )

        yield str(result)

    return swarm_tool


# ---- Internals ----


def _build_schema(
    *,
    presets: Mapping[str, Preset],
    default_preset: str | None,
    instructions: Open | Choice | Fixed,
    tools: Choice | Fixed | Inherit | None,
    mcp_servers: Choice | Fixed | Inherit | None,
    model: Inherit | Choice | Fixed | None,
    max_agents: int,
) -> dict[str, Any]:
    """Derive the tool's input schema from the axis modes; hidden axes produce no agent property."""
    # Without a preset to fall back on, an Open prompt must come from the model.
    required = ["name", "instructions"] if isinstance(instructions, Open) and not presets else ["name"]
    axis_properties = _to_schema_properties(
        presets=presets,
        default_preset=default_preset,
        instructions=instructions,
        tools=tools,
        mcp_servers=mcp_servers,
        model=model,
    )
    return {
        "type": "object",
        "required": ["task", "agents"],
        "properties": {
            "task": {"type": "string", "description": "The objective for the agent team."},
            "entry_point": {
                "type": "string",
                "description": "Name of the agent that receives the task and starts the swarm. "
                "Defaults to the first agent.",
            },
            "agents": {
                "type": "array",
                "description": "Agent specifications.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A unique name for this agent."},
                        **axis_properties,
                    },
                    "required": required,
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": max_agents,
            },
        },
    }


swarm = make_swarm()
"""Default swarm tool. Children inherit the caller's model."""
