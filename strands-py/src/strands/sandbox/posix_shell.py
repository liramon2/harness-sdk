"""Shell sandbox with default implementations for file and code operations.

Subclasses only need to implement :meth:`PosixShellSandbox.execute_streaming` —
all other operations are implemented by running shell commands through it. Use
this for remote environments where only shell access is available (Docker
containers, SSH connections, cloud runtimes).

Mirrors ``strands-ts/src/sandbox/posix-shell.ts``.
"""

import base64
import contextlib
import logging
import re
import shlex
import uuid
from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlparse

from .base import Sandbox
from .constants import ENV_KEY_PATTERN, LANGUAGE_PATTERN
from .errors import SandboxFetchError, SandboxPathNotFoundError, SandboxTimeoutError
from .types import ExecutionResult, FetchResult, FileInfo, StreamChunk

logger = logging.getLogger(__name__)

# The characters RFC 3986 allows anywhere in a URL.  Anything outside this
# set (whitespace, quotes, control characters, non-ASCII) is rejected before
# the URL reaches a shell command.  Security-critical for the curl transport.
_URL_CHARS = re.compile(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")


def validate_env_keys(env: dict[str, str]) -> None:
    """Validate environment variable names against :data:`ENV_KEY_PATTERN`.

    Args:
        env: Mapping of environment variable names to values.

    Raises:
        ValueError: If any key is not a valid POSIX environment variable name.
    """
    for key in env:
        if not ENV_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid environment variable name: {key}")


def build_shell_env_prefix(env: dict[str, str] | None = None) -> str:
    """Build a shell ``export KEY=VALUE && ...`` prefix, or ``""`` when empty.

    Keys are validated; values are escaped with :func:`shlex.quote`. Used by
    shell-string backends (e.g. SSH); backends that set env via native flags
    (e.g. Docker's ``-e``) call :func:`validate_env_keys` directly.

    Uses ``export`` rather than an ``env KEY=VALUE`` command wrapper so the
    variables are set in the shell itself and inherited by every stage of a
    pipeline. ``execute_code`` runs ``base64 ... | <lang>``, and an ``env``
    wrapper would only bind the left side of the pipe, never reaching the
    interpreter. The trailing ``&&`` keeps the surrounding
    ``cd ... && <prefix><command>`` chain fail-fast.

    Args:
        env: Mapping of environment variable names to values.

    Returns:
        The shell ``export ... && `` prefix, or an empty string when ``env`` is
        ``None`` or empty.

    Raises:
        ValueError: If any key is not a valid POSIX environment variable name.
    """
    if not env:
        return ""
    validate_env_keys(env)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"export {assignments} && "


def _eof_marker() -> str:
    """Generate a unique heredoc EOF marker, mirroring the TS ``STRANDS_EOF_`` token."""
    return f"STRANDS_EOF_{uuid.uuid4().hex[:16]}"


def _parse_header_dump(raw: str) -> dict[str, str]:
    r"""Parse a curl ``-D`` header dump into ``{lowered_name: value}``.

    Curl dumps headers for every hop in a redirect chain, separated by blank
    lines. Only the final response is parsed so that stale headers from
    intermediate redirects don't leak through.
    """
    # Split on blank lines, take the last non-empty block.
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n"))
    last_block = ""
    for block in reversed(blocks):
        if block.strip():
            last_block = block
            break

    result: dict[str, str] = {}
    for line in last_block.split("\n"):
        line = line.strip()
        if not line or line.upper().startswith("HTTP/"):
            continue
        colon = line.find(":")
        if colon < 1:
            continue
        name = line[:colon].strip().lower()
        value = line[colon + 1 :].strip()
        result[name] = value
    return result


class PosixShellSandbox(Sandbox, ABC):
    """Abstract sandbox that provides shell-based defaults for file and code operations.

    Assumes a POSIX-compatible shell (sh/bash) on the target.

    Subclasses only need to implement :meth:`execute_streaming`. The remaining
    operations — ``execute_code_streaming``, ``read_file``, ``write_file``,
    ``remove_file``, and ``list_files`` — are implemented via shell commands
    piped through :meth:`execute_streaming`.

    Subclasses may override any method with a native implementation for better
    performance or to handle edge cases (e.g., binary-safe file transfer via
    Docker stdin pipes, or native API calls for cloud backends).

    Subclasses are responsible for honoring the execution options in
    :meth:`execute_streaming`, or they have no effect:

    - ``env`` — backends that build a shell-command string prepend
      :func:`build_shell_env_prefix`; backends that set env via process flags
      (e.g. Docker's ``-e``) call :func:`validate_env_keys` and pass the values
      directly. An implementation that ignores ``env`` will silently drop the
      caller's variables.
    - ``timeout`` — the base class does not enforce a timeout; a subclass that
      does not wire ``timeout`` into its process supervision will silently run
      without any time limit.
    - ``cwd`` — similarly must be applied by the subclass.
    """

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute code by piping it to a language interpreter over the shell.

        The code is base64-encoded and decoded inside a quoted heredoc on the
        target, then piped to the interpreter (``base64 -d << 'EOF' | <lang>``).
        This transports arbitrary source — including shell metacharacters,
        quotes, and newlines — without injection risk. The ``language`` is
        validated against :data:`LANGUAGE_PATTERN` first.

        Args:
            code: The source code to execute.
            language: The interpreter to use (e.g., ``"python3"``, ``"node"``).
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for execution.
            env: Environment variables to set for this execution.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final
            :class:`ExecutionResult`.

        Raises:
            ValueError: If ``language`` contains invalid characters.
        """
        if not LANGUAGE_PATTERN.fullmatch(language):
            raise ValueError(f"language parameter contains invalid characters: {language}")
        encoded = base64.b64encode(code.encode()).decode("ascii")
        eof = _eof_marker()
        command = f"base64 -d << '{eof}' | {language}\n{encoded}\n{eof}"
        async for chunk in self.execute_streaming(command, timeout=timeout, cwd=cwd, env=env, **kwargs):
            yield chunk

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file as raw bytes via base64 over the shell.

        Args:
            path: Path to the file to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as raw bytes.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
            OSError: If the command succeeds but its output is not valid base64
                (e.g. a shell profile or locale warning prepended text to stdout).
        """
        result = await self.execute(f"base64 < {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr or f"Failed to read file: {path}")
        # base64 output is ASCII-safe text; strip whitespace (line wrapping) and decode.
        try:
            # binascii.Error (raised by b64decode on malformed input) subclasses ValueError.
            return base64.b64decode("".join(result.stdout.split()))
        except ValueError as e:
            raise OSError(f"Failed to decode base64 contents of file: {path}") from e

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write raw bytes to a file via base64 over the shell.

        Parent directories are created via ``mkdir -p``. The base64-encoded
        content is decoded inside a quoted heredoc on the target, preserving
        arbitrary binary content.

        Args:
            path: Path to the file to write.
            content: The content to write.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            OSError: If the file cannot be written.
        """
        encoded = base64.b64encode(content).decode("ascii")
        quoted = shlex.quote(path)
        eof = _eof_marker()
        cmd = f"mkdir -p \"$(dirname {quoted})\" && base64 -d << '{eof}' > {quoted}\n{encoded}\n{eof}"
        result = await self.execute(cmd)
        if result.exit_code != 0:
            raise OSError(result.stderr or f"Failed to write file: {path}")

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file via ``rm`` over the shell.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        result = await self.execute(f"rm {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr or f"Failed to remove file: {path}")

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List directory contents via ``ls -1ap`` parsing.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`FileInfo` entries (``size`` is always ``None`` for
            this shell-based listing).

        Raises:
            SandboxPathNotFoundError: If the directory does not exist (or ``path``
                is not a directory).
            OSError: If the listing fails for another reason.
        """
        quoted = shlex.quote(path)
        # Exit 77 distinguishes a missing directory from ls's own failures (locale-independent).
        result = await self.execute(f"test -d {quoted} || exit 77; env QUOTING_STYLE=literal ls -1ap {quoted}")
        if result.exit_code == 77:
            raise SandboxPathNotFoundError(path)
        if result.exit_code != 0:
            raise OSError(result.stderr or f"Failed to list directory: {path}")

        entries: list[FileInfo] = []
        for raw in result.stdout.split("\n"):
            line = raw.rstrip("\r")
            if not line or line in ("./", "../"):
                continue
            is_dir = line.endswith("/")
            name = line[:-1] if is_dir else line
            if name:
                entries.append(FileInfo(name=name, is_dir=is_dir))
        return entries

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int | None = None,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> FetchResult:
        """Fetch a URL with ``curl`` inside the sandbox.

        ``max_bytes`` is rejected early when ``Content-Length`` is known and
        truncated for chunked responses.
        """
        url = url.strip()
        if not _URL_CHARS.fullmatch(url):
            raise SandboxFetchError(
                "fetch URLs may only contain the characters RFC 3986 allows; "
                "percent-encode spaces and non-ASCII characters (and punycode the host) and retry."
            )
        parts = urlparse(url)
        if parts.scheme not in ("http", "https"):
            raise SandboxFetchError(f"fetch only supports http(s) URLs, got {url!r}.")
        if not parts.hostname:
            raise SandboxFetchError(f"fetch URL has no host: {url!r}.")

        tag = uuid.uuid4().hex
        body_file = f"/tmp/strands-fetch-{tag}"
        header_file = f"/tmp/strands-fetch-{tag}.headers"
        body_quoted = shlex.quote(body_file)
        header_quoted = shlex.quote(header_file)
        part_quoted = shlex.quote(body_file + ".part")

        # Build a hardened curl command: protocol restricted to http(s),
        # body → file, response headers → file via -D, final URL on stdout.
        opts = ""
        if timeout is not None:
            opts += f" --max-time {timeout}"
        if max_bytes is not None:
            opts += f" --max-filesize {max_bytes}"
        if headers:
            for n, v in headers.items():
                opts += f" -H {shlex.quote(f'{n}: {v}')}"
        cmd = (
            f"curl -sSL -g --fail"
            f" --proto '=http,https' --proto-redir '=http,https'"
            f"{opts}"
            f" -D {header_quoted} -o {body_quoted}"
            f" -w '%{{url_effective}}'"
            f" -- {shlex.quote(url)}"
        )
        if max_bytes is not None:
            cmd += f" && head -c {max_bytes} {body_quoted} > {part_quoted} && mv -f {part_quoted} {body_quoted}"

        # Give curl a few extra seconds beyond its own --max-time so the
        # sandbox kills it only if curl itself hangs.
        exec_timeout = (timeout + 5) if timeout is not None else None

        try:
            try:
                result = await self.execute(cmd, timeout=exec_timeout, **kwargs)
            except SandboxTimeoutError as exc:
                raise SandboxFetchError(f"fetch timed out after {timeout}s") from exc
            if result.exit_code != 0:
                raise SandboxFetchError(result.stderr.strip() or f"curl exited with code {result.exit_code}")

            resolved_url = result.stdout.strip() or url
            raw_headers = (await self.read_file(header_file)).decode("utf-8", errors="replace")
            response_headers = _parse_header_dump(raw_headers)
            body = await self.read_file(body_file)
        finally:
            with contextlib.suppress(Exception):
                await self.execute(
                    f"rm -f {body_quoted} {part_quoted} {header_quoted}",
                    timeout=10,
                )

        return FetchResult(url=resolved_url, headers=response_headers, body=body)
