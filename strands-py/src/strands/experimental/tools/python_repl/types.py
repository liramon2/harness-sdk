"""Shared types and constants for the python_repl tool."""

PYTHON_REPL_DESCRIPTION = (
    "Executes Python code in a secure Monty sandbox and returns its output. "
    "Session state (variables, imports, and function and class definitions) persists across calls, so code "
    "can build on earlier calls; pass reset_state=True to discard it and start from an empty namespace. "
    "The sandbox has no filesystem, network, or environment access, and execution is bounded by memory and "
    "time limits, so long-running or resource-heavy code is terminated. Use print() to surface values."
)
"""Description for the python_repl tool."""
