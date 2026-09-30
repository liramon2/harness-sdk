"""Multi-Agent Specification and Resolution.

Provides the authority-mode system for model-driven multi-agent patterns, where the
model configures child agents at runtime. The developer sets axis policies that shape
the parameters the model sees and govern what values it can supply:

- ``Fixed``  — developer-pinned value; hidden from the model.
- ``Inherit`` — value taken from the parent agent; hidden from the model.
- ``Open``   — model supplies a free-form value.
- ``Choice`` — model picks from a developer-supplied set.

``resolve_spec`` merges model-supplied arguments, preset defaults, and axis policies
into a fully resolved ``AgentSpec`` used to build child agents.  ``Inherit`` axes resolve
to ``None`` (meaning "inherit all"); the class is a self-documenting marker.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..agent import Agent
from ..models import Model, ModelRouter
from ..tools.mcp import MCPClient
from ..tools.mcp.mcp_agent_tool import MCPAgentTool

logger = logging.getLogger(__name__)

ContextMode = Literal["none", "all", "no_tools"]
"""Type alias for the valid ``context`` axis values."""

# The builder turns a resolved spec into a child agent, built the way the parent was.
AgentBuilder = Callable[["AgentSpec"], "Agent"]

_UNSET: Any = object()
"""Sentinel for :attr:`Option.value` so ``None`` can be a legitimate value."""


@dataclass(frozen=True)
class Fixed:
    """The developer pins the value; the axis contributes no model-facing parameter."""

    value: Any = None


@dataclass(frozen=True)
class Inherit:
    """The child takes the parent's value; the axis contributes no model-facing parameter."""


@dataclass(frozen=True)
class Open:
    """The model writes the value freely; the axis contributes a string parameter."""


@dataclass(frozen=True)
class Option:
    """One selectable option.

    ``name`` is what the model picks (the enum entry and the key we map back on); ``value`` is what
    that choice resolves to, defaulting to ``name`` so a string-valued axis needs only a name.
    ``description`` guides the model.
    """

    name: str
    value: Any = _UNSET
    description: str = ""


@dataclass(frozen=True)
class Choice:
    """The model picks from a developer-supplied set by ``name``.

    The axis contributes an enum parameter, or an ``array`` of enum when ``multiple`` is set
    (letting the model pick several).

    Each entry in ``options`` is a bare name or an ``Option(name, value=name, description="")``. The
    picked name maps back to the option's ``value``; descriptions render into the parameter's
    ``description`` (JSON Schema has no per-enum-value doc), the same way ``Preset`` descriptions
    surface under ``agent_type``. For ``tools`` the chosen set is re-validated at call time, so a
    delegate can never exceed the parent's tools.
    """

    options: Sequence[str | Option]
    multiple: bool = False

    def normalized(self) -> list[Option]:
        """The options as ``Option``s with ``value`` resolved, wrapping any bare name."""
        result = []
        for o in self.options:
            if isinstance(o, Option):
                # Fill in value from name when it was left unset.
                result.append(o if o.value is not _UNSET else Option(o.name, o.name, o.description))
            else:
                name = str(o)
                result.append(Option(name, name))
        return result

    def to_schema_property(self, description: str = "") -> dict[str, Any]:
        """Render this axis as a JSON Schema property for a tool's ``inputSchema``.

        Produces a ``string`` enum (or ``array`` of enum when ``multiple``), with per-option
        descriptions folded into the property's ``description``.
        """
        options = self.normalized()
        values = [o.name for o in options]
        lines = [f"- {o.name}: {o.description}" for o in options if o.description]
        desc = description
        if lines:
            block = "Options:\n" + "\n".join(lines)
            desc = f"{description}\n{block}" if description else block
        prop: dict[str, Any] = (
            {"type": "array", "items": {"type": "string", "enum": values}}
            if self.multiple
            else {"type": "string", "enum": values}
        )
        if desc:
            prop["description"] = desc
        return prop

    def value_for(self, name: str) -> Any:
        """The resolved value behind ``name``, or ``name`` itself if unknown (passthrough)."""
        for o in self.normalized():
            if o.name == name:
                return o.value
        return name


@dataclass(frozen=True)
class Preset:
    """A named role: a partially applied child configuration selected via ``agent_type``."""

    instructions: str | None = None
    tools: Sequence[str] | None = None
    model: Model | ModelRouter | str | None = None
    context: ContextMode = "none"
    last_messages: int | None = None
    description: str = ""


@dataclass
class AgentSpec:
    """The resolved child configuration handed to the builder: config + the model's arguments."""

    name: str | None = None
    task: str = ""
    agent_type: str | None = None
    instructions: str | None = None
    tools: list[str] | None = None
    mcp_servers: list[str] | None = None
    model: Model | ModelRouter | str | None = None
    context: ContextMode = "none"
    last_messages: int | None = None


def _resolve_scalar(
    model_value: Any,
    axis: Open | Choice | Fixed | Inherit,
    preset_value: Any = None,
) -> Any:
    """Resolve a scalar axis: model-supplied value, then preset, then axis default."""
    if model_value is not _UNSET:
        if isinstance(axis, Open):
            return model_value
        if isinstance(axis, Choice) and any(o.name == model_value for o in axis.normalized()):
            return axis.value_for(model_value)
    if preset_value is not None:
        return preset_value
    if isinstance(axis, Fixed):
        return axis.value
    return None


def _resolve_list(
    model_value: Any,
    axis: Choice | Fixed | Inherit,
    preset_values: Sequence[str] | None = None,
) -> list[str] | None:
    """Resolve a list axis: model-supplied value, then preset, then axis default, then ``None``.

    When the axis is a Choice, ignores values that are not valid options.
    """
    allowed = [o.name for o in axis.normalized()] if isinstance(axis, Choice) else None

    # Model-supplied value
    if model_value is not _UNSET and allowed is not None:
        assert isinstance(axis, Choice)
        if isinstance(model_value, str):
            requested = [model_value]
        elif isinstance(model_value, list):
            requested = model_value
        else:
            requested = []
        return [axis.value_for(t) for t in requested if t in allowed]
    # Preset value
    if preset_values is not None:
        if allowed is not None:
            assert isinstance(axis, Choice)
            allowed_values = {axis.value_for(n) for n in allowed}
            return [t for t in preset_values if t in allowed_values]
        return list(preset_values)
    # All choices
    if isinstance(axis, Choice) and axis.multiple and allowed is not None:
        return [axis.value_for(n) for n in allowed]
    if isinstance(axis, Fixed):
        return list(axis.value) if axis.value is not None else None
    return None


def resolve_spec(
    model_input: Mapping[str, Any],
    *,
    presets: Mapping[str, Preset],
    default_preset: str | None,
    instructions: Open | Choice | Fixed,
    tools: Choice | Fixed | Inherit | None = None,
    mcp_servers: Choice | Fixed | Inherit | None = None,
    model: Inherit | Choice | Fixed | None = None,
    context: Fixed | Choice | None = None,
) -> AgentSpec:
    """Combine the model's arguments, the selected preset, and the fixed axes into a spec.

    Precedence per axis: a model-supplied argument wins, then the preset's value, then the axis
    default (``Fixed``/``Inherit``). An omitted ``agent_type`` falls back to the default preset, so
    a bare ``subagent(task=...)`` behaves like the default role.

    Args:
        model_input: The model-supplied arguments.
        presets: Named presets mapping ``agent_type`` strings to ``Preset`` instances.
        default_preset: The preset to apply when the model omits ``agent_type``.
        instructions: Axis policy for the ``instructions`` field.
        tools: Axis policy for tool selection. Defaults to ``Inherit()`` when not supplied.
        mcp_servers: Axis policy for MCP server selection. Defaults to ``Inherit()`` when not supplied.
        model: Axis policy for model selection. Defaults to ``Inherit()`` when not supplied.
        context: Axis policy for context sharing. Defaults to ``Fixed("none")`` when not supplied.
    """
    if tools is None:
        tools = Inherit()
    if mcp_servers is None:
        mcp_servers = Inherit()
    if model is None:
        model = Inherit()
    if context is None:
        context = Fixed("none")
    task = str(model_input.get("task", ""))
    name = model_input.get("name")

    # agent_type is a closed enum: a provided value must be an exact preset name (absent = no role).
    model_agent_type = model_input.get("agent_type")
    is_known_preset = isinstance(model_agent_type, str) and model_agent_type in presets
    if model_agent_type is not None and not is_known_preset:
        raise ValueError(f"Unknown agent_type {model_agent_type!r}; valid values: {sorted(presets)}.")
    # Ad-hoc instructions (only possible when the axis exposes one) override the default preset.
    overriding_instructions = isinstance(instructions, (Open, Choice)) and "instructions" in model_input

    if is_known_preset:
        agent_type = model_agent_type
    elif not overriding_instructions:
        agent_type = default_preset
    else:
        agent_type = None

    preset = presets.get(agent_type) if agent_type else None

    # Combine the model inputs, axis policies, and presets into an agent spec
    spec = AgentSpec(name=str(name) if name is not None else None, task=task, agent_type=agent_type)
    spec.instructions = _resolve_scalar(
        model_input.get("instructions", _UNSET), instructions, preset.instructions if preset else None
    )
    spec.tools = _resolve_list(model_input.get("tools", _UNSET), tools, preset.tools if preset else None)
    spec.mcp_servers = _resolve_list(model_input.get("mcp_servers", _UNSET), mcp_servers)
    spec.model = _resolve_scalar(model_input.get("model", _UNSET), model, preset.model if preset else None)
    val = _resolve_scalar(model_input.get("context", _UNSET), context, preset.context if preset else None)
    spec.context = val if val is not None else "none"

    # last_messages — always available when context is not "none", but optional.
    if preset and preset.last_messages is not None:
        spec.last_messages = preset.last_messages
    if "last_messages" in model_input:
        try:
            spec.last_messages = int(model_input["last_messages"])
        except (TypeError, ValueError):
            pass

    return spec


def default_builder(parent: Agent) -> AgentBuilder:
    """A builder that creates an Agent inheriting the parent's model, tools, and MCP servers.

    Used by the vended multi-agent tools when no custom builder is supplied.
    Resolves ``spec.tools`` against the parent's tool registry and ``spec.mcp_servers``
    against the parent's MCP clients (by ``client_name``).
    """

    def build(spec: AgentSpec) -> Agent:
        parent_tools: dict[str, Any] = {}
        mcp_clients: dict[str, MCPClient] = {}
        for t in parent.tool_registry.registry.values() if parent else []:
            if isinstance(t, MCPAgentTool):
                # MCP tools flow through mcp_servers to avoid duplicates with their client.
                if t.mcp_client.client_name is not None:
                    mcp_clients.setdefault(t.mcp_client.client_name, t.mcp_client)
            else:
                # Plain tools are selected by name via spec.tools.
                parent_tools[t.tool_name] = t

        # tools=None means inherit all; a list means only those.
        child_tools: list[Any] = (
            list(parent_tools.values())
            if spec.tools is None
            else [parent_tools[t] for t in spec.tools if t in parent_tools]
        )

        # MCP servers: spec.mcp_servers=None means inherit all, a list means only those.
        selected = (
            mcp_clients
            if spec.mcp_servers is None
            else {n: mcp_clients[n] for n in spec.mcp_servers if n in mcp_clients}
        )
        child_tools.extend(selected.values())

        return Agent(
            system_prompt=spec.instructions or "",
            tools=child_tools,
            model=spec.model or (parent.model if parent else None),
            name=spec.name,
        )

    return build
