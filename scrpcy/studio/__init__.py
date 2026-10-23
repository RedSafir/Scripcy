"""scrpcy.studio package."""

from __future__ import annotations

from scrpcy.studio.app import CLIStudio
from scrpcy.studio.signals import (
    SCRPCY_LOGO,
    SCRPCY_VERSION,
    SessionCloseSignal,
    SessionDetachSignal,
    _AgentProgressTracker,
)
from scrpcy.ui.interactive import _format_model_display_name

__all__ = [
    "CLIStudio",
    "SessionCloseSignal",
    "SessionDetachSignal",
    "_AgentProgressTracker",
    "SCRPCY_LOGO",
    "SCRPCY_VERSION",
    "_format_model_display_name",
]
