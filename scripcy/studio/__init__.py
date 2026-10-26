"""scripcy.studio package."""

from __future__ import annotations

from scripcy.studio.app import CLIStudio
from scripcy.studio.signals import (
    SCRIPCY_LOGO,
    SCRIPCY_VERSION,
    SessionCloseSignal,
    SessionDetachSignal,
    _AgentProgressTracker,
)
from scripcy.ui.interactive import _format_model_display_name

__all__ = [
    "CLIStudio",
    "SessionCloseSignal",
    "SessionDetachSignal",
    "_AgentProgressTracker",
    "SCRIPCY_LOGO",
    "SCRIPCY_VERSION",
    "_format_model_display_name",
]
