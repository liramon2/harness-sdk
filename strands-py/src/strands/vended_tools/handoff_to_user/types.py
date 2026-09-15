"""Shared types and constants for the handoff_to_user tool."""

HANDOFF_INTERRUPT_NAME = "strands:handoff_to_user"
"""Stable name reported on the raised interrupt (``Interrupt.name``)."""

DEFAULT_HANDOFF_TO_USER_DESCRIPTION = (
    "Pause the agent loop and surface a message to the user for human-in-the-loop input. "
    "Call this tool when you need confirmation, additional information, or explicit user approval "
    "before proceeding. The agent loop will halt until the user responds; execution then continues "
    "with the user's reply available as the tool result."
)
"""Description for the handoff_to_user tool."""
