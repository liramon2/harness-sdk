"""Shared types and constants for the mcp_client tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ...tools.mcp.mcp_client import MCPClient

Command = Literal["connect", "list_tools", "call_tool", "disconnect"]
"""Commands supported by the mcp_client tool."""


@dataclass
class _Connection:
    """A live MCP connection and its GC finalizer handle."""

    client: MCPClient
    finalizer: Any


MCP_CLIENT_DESCRIPTION = (
    "Connects to Model Context Protocol (MCP) servers at runtime to discover and invoke their tools. "
    "'connect' opens a connection to a permitted server. "
    "'list_tools' returns the tools the connected server exposes, including their names and input schemas. "
    "'call_tool' invokes a named tool on the connected server and returns its result. "
    "'disconnect' closes the connection. "
    "Only one server can be connected at a time: connecting to a different server closes the current connection, "
    "and reconnecting to the same server restarts it and discards its state. Disconnect when done."
)
"""Description for the mcp_client tool shown to the model."""
