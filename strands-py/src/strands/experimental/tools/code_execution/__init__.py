"""Code execution tool for running Python in a Monty sandbox.

This tool is experimental and subject to change in future revisions without notice.

Runs model-generated Python inside an isolated `Monty <https://pydantic.dev/docs/monty/>`_
worker with no filesystem, network, or environment access and bounded memory
and time. Session state persists across calls via :attr:`~strands.Agent.state`,
so later code can build on earlier calls; pass ``reset_state=True`` to start
from an empty namespace.

Requires the optional ``code-execution`` extra
(``pip install 'strands-agents[code-execution]'``).

Example Usage:
    ```python
    from strands import Agent
    from strands.experimental.tools import code_execution

    agent = Agent(tools=[code_execution])
    ```
"""

from .code_execution import code_execution, make_code_execution

__all__ = [
    "code_execution",
    "make_code_execution",
]
