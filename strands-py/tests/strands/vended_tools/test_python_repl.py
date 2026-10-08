"""Tests for the python_repl tool.

The python_repl tool is a shim over ``Sandbox.execute_code``: it routes code through
the agent's sandbox (or a bound one). Each call runs in a fresh interpreter, so
in-memory state does not persist across calls. The end-to-end tests spawn ``sh``
and ``python3`` and require POSIX, so they are skipped on Windows.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import strands.vended_tools as vended_tools
from strands.sandbox.errors import SandboxTimeoutError
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.sandbox.types import ExecutionResult
from strands.types.tools import ToolContext
from strands.vended_tools.python_repl import make_python_repl, python_repl
from strands.vended_tools.python_repl.types import PYTHON_REPL_DESCRIPTION, PythonReplError

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")


def _tool_context(sandbox=None) -> ToolContext:
    """Build a ToolContext whose agent exposes a sandbox (or a fresh local one)."""
    agent = SimpleNamespace(sandbox=sandbox or NotASandboxLocalEnvironment())
    return ToolContext(
        tool_use={"name": "python_repl", "toolUseId": "id", "input": {}}, agent=agent, invocation_state={}
    )


def _mock_sandbox(result: ExecutionResult | None = None, side_effect: BaseException | None = None) -> SimpleNamespace:
    execute_code = AsyncMock(return_value=result, side_effect=side_effect)
    return SimpleNamespace(execute_code=execute_code)


class TestShim:
    """The tool forwards to ``execute_code`` and maps the result."""

    @pytest.mark.asyncio
    async def test_forwards_code_language_and_timeout(self):
        sandbox = _mock_sandbox(ExecutionResult(exit_code=0, stdout="hi\n", stderr=""))
        tool = make_python_repl(sandbox=sandbox)

        result = await tool(code="print('hi')", tool_context=_tool_context(), timeout=7)

        sandbox.execute_code.assert_awaited_once_with("print('hi')", "python3", timeout=7)
        assert result == {"output": "hi\n", "error": "", "exit_code": 0}

    @pytest.mark.asyncio
    async def test_default_timeout(self):
        sandbox = _mock_sandbox(ExecutionResult(exit_code=0, stdout="", stderr=""))
        await make_python_repl(sandbox=sandbox)(code="pass", tool_context=_tool_context())
        assert sandbox.execute_code.await_args.kwargs == {"timeout": 120}

    @pytest.mark.asyncio
    async def test_custom_language(self):
        sandbox = _mock_sandbox(ExecutionResult(exit_code=0, stdout="", stderr=""))
        await make_python_repl(sandbox=sandbox, language="python3.12")(code="pass", tool_context=_tool_context())
        assert sandbox.execute_code.await_args.args == ("pass", "python3.12")

    @pytest.mark.asyncio
    async def test_reads_sandbox_from_agent_context(self):
        sandbox = _mock_sandbox(ExecutionResult(exit_code=0, stdout="ok", stderr=""))
        result = await python_repl(code="print('ok')", tool_context=_tool_context(sandbox))
        sandbox.execute_code.assert_awaited_once()
        assert result["output"] == "ok"

    @pytest.mark.asyncio
    async def test_bound_sandbox_wins_over_agent_sandbox(self):
        bound = _mock_sandbox(ExecutionResult(exit_code=0, stdout="bound", stderr=""))
        agent_sandbox = _mock_sandbox(ExecutionResult(exit_code=0, stdout="agent", stderr=""))
        result = await make_python_repl(sandbox=bound)(code="pass", tool_context=_tool_context(agent_sandbox))
        assert result["output"] == "bound"
        agent_sandbox.execute_code.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_error_carries_partial_output_with_success_field_names(self):
        sandbox = _mock_sandbox(side_effect=SandboxTimeoutError(5, stdout="partial", stderr="warn"))
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await make_python_repl(sandbox=sandbox)(code="pass", tool_context=_tool_context())
        payload = json.loads(str(exc_info.value).split("\n", 1)[1])
        assert payload == {"output": "partial", "error": "warn", "exit_code": 124}

    @pytest.mark.asyncio
    async def test_wraps_sandbox_error_as_python_repl_error(self):
        boom = OSError("container gone")
        sandbox = _mock_sandbox(side_effect=boom)
        with pytest.raises(PythonReplError, match="container gone") as exc_info:
            await make_python_repl(sandbox=sandbox)(code="pass", tool_context=_tool_context())
        assert exc_info.value.__cause__ is boom
        assert isinstance(exc_info.value, RuntimeError)


@posix_only
class TestLocalExecution:
    """End-to-end through ``NotASandboxLocalEnvironment``."""

    @pytest.fixture
    def repl(self):
        return make_python_repl(sandbox=NotASandboxLocalEnvironment())

    @pytest.mark.asyncio
    async def test_runs_python(self, repl):
        result = await repl(code="print(sum(range(10)))", tool_context=_tool_context())
        assert result == {"output": "45\n", "error": "", "exit_code": 0}

    @pytest.mark.asyncio
    async def test_exception_reports_traceback_and_nonzero_exit(self, repl):
        result = await repl(code="raise ValueError('bad')", tool_context=_tool_context())
        assert result["exit_code"] != 0
        assert "ValueError: bad" in result["error"]

    @pytest.mark.asyncio
    async def test_state_does_not_persist_but_files_do(self, repl, tmp_path):
        path = tmp_path / "state.json"
        ctx = _tool_context()
        await repl(code=f"import json; x = 42; json.dump({{'x': x}}, open({str(path)!r}, 'w'))", tool_context=ctx)

        missing = await repl(code="print(x)", tool_context=ctx)
        assert "NameError" in missing["error"]

        restored = await repl(code=f"import json; print(json.load(open({str(path)!r}))['x'])", tool_context=ctx)
        assert restored["output"] == "42\n"

    @pytest.mark.asyncio
    async def test_respects_timeout(self, repl):
        with pytest.raises(SandboxTimeoutError):
            await repl(code="import time; time.sleep(10)", tool_context=_tool_context(), timeout=1)


class TestMakePythonRepl:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            make_python_repl(name="")

    @pytest.mark.parametrize("language", ["", "python3; rm -rf /", "py thon"])
    def test_rejects_invalid_language(self, language):
        with pytest.raises(ValueError, match="language"):
            make_python_repl(language=language)


class TestToolMetadata:
    def test_default_name(self):
        assert python_repl.tool_name == "python_repl"

    def test_custom_name(self):
        assert make_python_repl(name="run_python").tool_name == "run_python"

    def test_default_description(self):
        assert python_repl.tool_spec["description"] == PYTHON_REPL_DESCRIPTION

    def test_custom_description(self):
        assert make_python_repl(description="custom").tool_spec["description"] == "custom"

    def test_schema_excludes_context(self):
        properties = python_repl.tool_spec["inputSchema"]["json"]["properties"]
        assert set(properties) == {"code", "timeout"}
        assert python_repl.tool_spec["inputSchema"]["json"]["required"] == ["code"]


class TestPublicExports:
    def test_exported_from_vended_tools(self):
        assert vended_tools.python_repl is python_repl
        assert vended_tools.make_python_repl is make_python_repl
