"""Studio signals, progress tracking, and brand constants."""

from __future__ import annotations


class SessionDetachSignal(Exception):
    """User requested to detach the active chat session to background."""
    pass


class SessionCloseSignal(Exception):
    """User requested to close/terminate the active chat session."""
    pass


class _AgentProgressTracker:
    def __init__(self, init_msg: str) -> None:
        self.text = init_msg

    def set(self, msg: str) -> None:
        self.text = msg

    def __str__(self) -> str:
        return self.text

    def __call__(self) -> str:
        return self.text


SCRPCY_LOGO = (
    "███████╗  ██████╗   ██████╗   ██████╗   ██████╗   ██╗   ██╗\n"
    "██╔════╝  ██╔════╝  ██╔══██╗  ██╔══██╗  ██╔════╝  ╚██╗ ██╔╝\n"
    "███████╗  ██║       ██████╔╝  ██████╔╝  ██║        ╚████╔╝ \n"
    "╚════██║  ██║       ██╔══██╗  ██╔════╝  ██║         ╚██╔╝  \n"
    "███████║  ╚██████╗  ██║  ██║  ██║       ╚██████╗     ██║   \n"
    "╚══════╝   ╚═════╝  ╚═╝  ╚═╝  ╚═╝        ╚═════╝     ╚═╝   "
)
SCRPCY_VERSION = "v0.1.0"
