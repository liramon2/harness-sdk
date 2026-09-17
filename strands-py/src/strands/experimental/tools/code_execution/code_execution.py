"""Code execution tool: run Python in a Monty sandbox.

This tool is experimental and subject to change in future revisions without notice.

Provides :func:`make_code_execution` and the default :data:`code_execution`
instance. Each call runs a snippet inside a `Monty <https://pydantic.dev/docs/monty/>`_
worker subprocess with no filesystem, network, or environment access, bounded
by memory and time limits.

Session state (variables, imports, and definitions) persists across calls via
:attr:`~strands.Agent.state`. Pass ``reset_state=True`` to start fresh.

Requires the optional ``code-execution`` extra
(``pip install 'strands-agents[code-execution]'``).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import threading
import weakref
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Literal

try:
    from pydantic_monty import (
        AsyncMonty,
        CollectStreams,
        MontyError,
        MontyRuntimeError,
        MontySyntaxError,
        MontyTypingError,
        ResourceLimits,
    )
except ImportError as error:
    raise ImportError(
        "code_execution requires the 'code-execution' extra (pydantic-monty). "
        "Install with: pip install 'strands-agents[code-execution]'"
    ) from error

from ....tools.decorator import tool
from ....types.tools import ToolContext
from .types import CODE_EXECUTION_DESCRIPTION

if TYPE_CHECKING:
    from ....tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)

_STATE_KEY = "code_execution_session"


class CodeExecutionError(RuntimeError):
    """Raised when code execution fails."""


_ERRORS_WITH_DISPLAY = (MontyRuntimeError, MontySyntaxError, MontyTypingError)
_CANCEL_POLL_INTERVAL = 0.05
_DEFAULT_MAX_DURATION_SECS = 30.0
_DEFAULT_MAX_MEMORY_BYTES = 1024 * 1024 * 100  # 100 MiB
_DEFAULT_TIMEOUT_SECS = 60.0
_DEFAULT_MAX_OUTPUT_CHARS = 50_000
_DEFAULT_MAX_SESSION_BYTES = 1024 * 1024 * 10  # 10 MiB


def make_code_execution(
    *,
    name: str = "code_execution",
    description: str = CODE_EXECUTION_DESCRIPTION,
    max_duration_secs: float = _DEFAULT_MAX_DURATION_SECS,
    max_memory: int = _DEFAULT_MAX_MEMORY_BYTES,
    max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
    max_session_bytes: int = _DEFAULT_MAX_SESSION_BYTES,
    timeout: float = _DEFAULT_TIMEOUT_SECS,
) -> DecoratedFunctionTool:
    """Create a code execution tool backed by a Monty sandbox.

    Args:
        name: Tool name exposed to the model.
        description: Tool description shown to the model.
        max_duration_secs: Maximum execution time per call in seconds, enforced inside the sandbox. Defaults to 30.
        max_memory: Maximum heap memory the sandbox may allocate, in bytes. Defaults to 100 MiB.
        max_output_chars: Maximum characters returned for each of ``output``, ``error``, and ``result``.
            Longer values are truncated. Defaults to 50,000.
        max_session_bytes: Maximum size of the persisted session dump in bytes. Dumps exceeding this limit are
            not persisted. Defaults to 10 MiB.
        timeout: Host-side deadline in seconds; kills the worker if exceeded. Backstops
            ``max_duration_secs``. Defaults to 60.

    Returns:
        A decorated tool that executes Python code in a Monty sandbox and
        returns its stdout, stderr, and trailing-expression value.

    Raises:
        ValueError: If ``name`` is empty, or any limit is not positive.
    """
    if not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(max_duration_secs, (int, float)) or max_duration_secs <= 0:
        raise ValueError(f"max_duration_secs must be a positive number, got {max_duration_secs}")
    if not isinstance(max_memory, int) or max_memory < 1:
        raise ValueError("max_memory must be a positive integer")
    if not isinstance(max_output_chars, int) or max_output_chars < 1:
        raise ValueError("max_output_chars must be a positive integer")
    if not isinstance(max_session_bytes, int) or max_session_bytes < 1:
        raise ValueError("max_session_bytes must be a positive integer")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be a positive number, got {timeout}")

    _state_locks: weakref.WeakKeyDictionary[Any, asyncio.Lock] = weakref.WeakKeyDictionary()

    @tool(name=name, description=description, context="tool_context")
    async def code_execution_tool(
        code: str,
        tool_context: ToolContext,
        reset_state: bool = False,
    ) -> str:
        """Executes Python code in a secure Monty sandbox and returns its output.

        Session state (variables, imports, and definitions) persists across calls so later code
        can build on earlier calls. Pass ``reset_state=True`` to start from an empty namespace.

        Args:
            code: Python snippet to execute. Its trailing expression value, if any, is returned as ``result``.
            tool_context: Injected by the framework. Not user-facing.
            reset_state: When ``True``, discard any persisted session before executing. Defaults to ``False``.
        """
        limits = ResourceLimits(max_duration_secs=max_duration_secs, max_memory=max_memory)

        agent = tool_context.agent
        if agent not in _state_locks:
            _state_locks[agent] = asyncio.Lock()
        async with _state_locks[agent]:
            if reset_state:
                tool_context.agent.state.set(_STATE_KEY, None)
            old_state = tool_context.agent.state.get(_STATE_KEY)
            if old_state is not None and not isinstance(old_state, str):
                raise CodeExecutionError("Malformed code_execution session state: expected a string")

            collector = CollectStreams()

            try:
                _, new_state = await _await_bounded(
                    _run_session(code, old_state, collector, limits, timeout),
                    tool_context.cancel_signal,
                )
            except MontyError as error:
                raise CodeExecutionError(_build_error_message(error, collector.output)) from error

            output = "".join(text for _, text in collector.output)
            if len(output) > max_output_chars:
                output = output[:max_output_chars] + "\n\n[output truncated]"

            if len(new_state) <= max_session_bytes:
                tool_context.agent.state.set(_STATE_KEY, base64.b64encode(new_state).decode("ascii"))
            else:
                logger.warning(
                    "session_bytes=<%d>, max_session_bytes=<%d> | session dump exceeds limit, discarding",
                    len(new_state),
                    max_session_bytes,
                )
                output += "\n[warning: session state was too large to persist; this call's variables are not saved]"

            return output

    return code_execution_tool


code_execution = make_code_execution()
"""Default code execution tool."""


# ---- Internals ----


async def _await_bounded(coro: Awaitable[Any], cancel_signal: threading.Event) -> Any:
    """Await ``coro`` until it finishes or ``cancel_signal`` fires.

    On cancellation the coroutine is cancelled before raising ``asyncio.CancelledError``.
    """
    task = asyncio.ensure_future(coro)
    watch = asyncio.ensure_future(_poll_cancel(cancel_signal))
    try:
        done, _ = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise asyncio.CancelledError
    finally:
        for pending in (task, watch):
            if not pending.done():
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending


async def _poll_cancel(cancel_signal: threading.Event) -> None:
    """Return when the cancel signal is set."""
    while not cancel_signal.is_set():
        await asyncio.sleep(_CANCEL_POLL_INTERVAL)


async def _run_session(
    code: str,
    old_state: str | None,
    collector: CollectStreams,
    limits: Any,
    timeout: float,
) -> tuple[Any, bytes]:
    """Open a Monty pool, load prior state, execute code, and return the result value and new session dump."""
    async with AsyncMonty(request_timeout=timeout) as pool:
        if old_state is not None:
            async with pool.checkout(limits=limits) as session:
                try:
                    await session.load_session(base64.b64decode(old_state))
                except (MontyError, binascii.Error) as error:
                    logger.warning("error=<%s> | discarding unrestorable code_execution state", error)
                else:
                    value = await session.feed_run(code, print_callback=collector)
                    return value, await session.dump()

        # No prior state or restore failed — run in a fresh empty session
        async with pool.checkout(limits=limits) as session:
            value = await session.feed_run(code, print_callback=collector)
            return value, await session.dump()


def _build_error_message(error: Exception, output: list[tuple[Literal["stdout", "stderr"], str]]) -> str:
    """Build the CodeExecutionError message from a MontyError and any captured stdout."""
    message = str(error.display()) if isinstance(error, _ERRORS_WITH_DISPLAY) else str(error)
    sections = [message]
    stdout = "".join(text for _, text in output)
    if stdout:
        sections.append(f"--- stdout before failure ---\n{stdout}")
    return "\n\n".join(sections)
