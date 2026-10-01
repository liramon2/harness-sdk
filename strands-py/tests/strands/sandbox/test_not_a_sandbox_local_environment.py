"""Tests for the host execution environment (``NotASandboxLocalEnvironment``).

Mirrors ``strands-ts/src/sandbox/__tests__/not-a-sandbox-local-environment.test.node.ts``.
Command and code execution spawn a real ``sh``; file operations hit the host
filesystem directly. These require a POSIX shell, so they are skipped on Windows.
"""

import os
import sys
from unittest.mock import patch

import httpx
import pytest

from strands.sandbox import FileInfo
from strands.sandbox.errors import SandboxFetchError, SandboxPathNotFoundError
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")


@pytest.fixture
def sandbox() -> NotASandboxLocalEnvironment:
    return NotASandboxLocalEnvironment()


# ---- execute ----


@pytest.mark.asyncio
async def test_execute_runs_a_command(sandbox):
    result = await sandbox.execute("echo hello")
    assert result.exit_code == 0
    assert result.stdout == "hello\n"


@pytest.mark.asyncio
async def test_execute_respects_cwd_option(sandbox, tmp_path):
    result = await sandbox.execute("pwd", cwd=str(tmp_path))
    assert result.stdout.strip().endswith(os.path.basename(str(tmp_path)))


@pytest.mark.asyncio
async def test_execute_reports_nonzero_exit_and_stderr(sandbox):
    result = await sandbox.execute("echo oops >&2; exit 3")
    assert result.exit_code == 3
    assert result.stderr == "oops\n"


@pytest.mark.asyncio
async def test_execute_applies_env_option(sandbox):
    result = await sandbox.execute("printenv GREETING", env={"GREETING": "hi"})
    assert result.stdout == "hi\n"


# ---- execute_code ----


@pytest.mark.asyncio
async def test_execute_code_runs_interpreter(sandbox, tmp_path):
    result = await sandbox.execute_code("print(2 + 2)", "python3", cwd=str(tmp_path))
    assert result.exit_code == 0
    assert result.stdout == "4\n"


@pytest.mark.asyncio
async def test_execute_code_rejects_invalid_language(sandbox):
    with pytest.raises(ValueError, match="invalid characters"):
        await sandbox.execute_code("x", "../../bin/sh")


@pytest.mark.asyncio
async def test_execute_code_applies_env_option(sandbox):
    result = await sandbox.execute_code('import os; print(os.environ["GREETING"])', "python3", env={"GREETING": "hi"})
    assert result.stdout == "hi\n"


# ---- read/write (native fs) ----


@pytest.mark.asyncio
async def test_text_file_roundtrip_absolute_path(sandbox, tmp_path):
    file = str(tmp_path / "note.txt")
    await sandbox.write_text(file, "hello host")
    assert await sandbox.read_text(file) == "hello host"


@pytest.mark.asyncio
async def test_binary_roundtrip_preserves_all_byte_values(sandbox, tmp_path):
    file = str(tmp_path / "all-bytes.bin")
    data = bytes(range(256))
    await sandbox.write_file(file, data)
    assert await sandbox.read_file(file) == data


@pytest.mark.asyncio
async def test_write_creates_missing_parent_directories(sandbox, tmp_path):
    file = str(tmp_path / "deep" / "nested" / "file.txt")
    await sandbox.write_text(file, "deep")
    assert await sandbox.read_text(file) == "deep"


@pytest.mark.asyncio
async def test_read_nonexistent_file_raises(sandbox, tmp_path):
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file(str(tmp_path / "nope.txt"))


# ---- remove ----


@pytest.mark.asyncio
async def test_remove_file(sandbox, tmp_path):
    file = str(tmp_path / "delete-me.txt")
    await sandbox.write_text(file, "bye")
    await sandbox.remove_file(file)
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file(file)


@pytest.mark.asyncio
async def test_remove_nonexistent_file_raises(sandbox, tmp_path):
    with pytest.raises(FileNotFoundError):
        await sandbox.remove_file(str(tmp_path / "nope.txt"))


# ---- list_files (native metadata) ----


@pytest.mark.asyncio
async def test_list_files_sorted_with_is_dir_and_size(sandbox, tmp_path):
    await sandbox.write_text(str(tmp_path / "c.txt"), "cc")
    await sandbox.write_text(str(tmp_path / "a.txt"), "a")
    await sandbox.write_text(str(tmp_path / "b.txt"), "bbb")
    (tmp_path / "sub").mkdir()

    files = await sandbox.list_files(str(tmp_path))

    # Full-shape equality on the files catches any regressed/unexpected FileInfo field.
    assert len(files) == 4
    assert files[:3] == [
        FileInfo(name="a.txt", is_dir=False, size=1),
        FileInfo(name="b.txt", is_dir=False, size=3),
        FileInfo(name="c.txt", is_dir=False, size=2),
    ]
    # Directory size is platform-dependent, so assert only name/is_dir for the dir entry.
    sub = files[3]
    assert sub.name == "sub"
    assert sub.is_dir is True


@pytest.mark.asyncio
async def test_list_nonexistent_directory_raises(sandbox, tmp_path):
    # Missing path (ENOENT) maps to SandboxPathNotFoundError, mirroring the TS oracle.
    with pytest.raises(SandboxPathNotFoundError):
        await sandbox.list_files(str(tmp_path / "no-such-dir"))


@pytest.mark.asyncio
async def test_list_files_on_a_file_path_raises(sandbox, tmp_path):
    # Mirrors the TS oracle's "path component is a file" case (ENOTDIR -> SandboxPathNotFoundError).
    file = tmp_path / "file.txt"
    await sandbox.write_text(str(file), "x")
    with pytest.raises(SandboxPathNotFoundError):
        await sandbox.list_files(str(file / "nested"))


# ---- _resolve_path (relative vs absolute) ----


@pytest.mark.asyncio
async def test_relative_path_resolves_against_cwd(sandbox, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    await sandbox.write_text("relative-probe.txt", "relative")
    assert (tmp_path / "relative-probe.txt").read_text() == "relative"


@pytest.mark.asyncio
async def test_absolute_path_written_as_is(sandbox, tmp_path):
    file = str(tmp_path / "abs.txt")
    await sandbox.write_text(file, "absolute")
    assert (tmp_path / "abs.txt").read_text() == "absolute"


# ---- fetch (httpx on host) ----

_HTTPX_CLIENT_PATH = "strands.sandbox.not_a_sandbox_local_environment.httpx.AsyncClient"


def _mock_client(handler):
    """Return a factory that ignores kwargs and returns a mock-transport client."""
    real_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    def factory(**kwargs):
        return real_client

    return factory


class TestFetch:
    @pytest.mark.asyncio
    async def test_successful_fetch(self, sandbox):
        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/html"}, text="<h1>Hello</h1>")

        with patch(_HTTPX_CLIENT_PATH, _mock_client(handler)):
            result = await sandbox.fetch("https://example.com/")

        assert result.body == b"<h1>Hello</h1>"
        assert result.headers["content-type"] == "text/html"

    @pytest.mark.asyncio
    async def test_max_bytes_truncates(self, sandbox):
        def handler(request):
            return httpx.Response(200, text="x" * 1000)

        with patch(_HTTPX_CLIENT_PATH, _mock_client(handler)):
            result = await sandbox.fetch("https://example.com/big", max_bytes=100)

        assert len(result.body) == 100

    @pytest.mark.asyncio
    async def test_http_error_raises(self, sandbox):
        def handler(request):
            return httpx.Response(404, text="Not Found")

        with patch(_HTTPX_CLIENT_PATH, _mock_client(handler)):
            with pytest.raises(SandboxFetchError, match="404"):
                await sandbox.fetch("https://example.com/missing")

    @pytest.mark.asyncio
    async def test_timeout_raises(self, sandbox):
        def handler(request):
            raise httpx.ReadTimeout("timed out")

        with patch(_HTTPX_CLIENT_PATH, _mock_client(handler)):
            with pytest.raises(SandboxFetchError, match="timed out"):
                await sandbox.fetch("https://example.com/slow")

    @pytest.mark.asyncio
    async def test_connection_error_raises(self, sandbox):
        def handler(request):
            raise httpx.ConnectError("refused")

        with patch(_HTTPX_CLIENT_PATH, _mock_client(handler)):
            with pytest.raises(SandboxFetchError, match="refused"):
                await sandbox.fetch("https://example.com/down")

    @pytest.mark.asyncio
    async def test_rejects_non_http_scheme(self, sandbox):
        with pytest.raises(SandboxFetchError, match="http\\(s\\)"):
            await sandbox.fetch("ftp://example.com")

    @pytest.mark.asyncio
    async def test_rejects_no_host(self, sandbox):
        with pytest.raises(SandboxFetchError, match="no host"):
            await sandbox.fetch("http://")

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, sandbox):
        def handler(request):
            return httpx.Response(200, text="ok")

        with patch(_HTTPX_CLIENT_PATH, _mock_client(handler)):
            result = await sandbox.fetch("  https://example.com  ")
        assert result.body == b"ok"
