"""Experimental tools package."""

import warnings
from typing import Any

from .stop import make_stop, stop

_DEPRECATED_NAMES = {"ToolProvider"}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        from ...tools import ToolProvider

        warnings.warn(
            f"{name} has been moved to production. Use {name} from strands.tools instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ToolProvider
    # code_execution pulls the optional ``code-execution`` extra, so it is lazy-loaded to keep
    # the base import free of those dependencies.
    if name in ("make_code_execution", "code_execution"):
        from .code_execution import code_execution, make_code_execution

        if name == "code_execution":
            return code_execution
        return make_code_execution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "make_stop",
    "stop",
]
