"""Backward-compatibility shim re-exporting CLIStudio and interactive UI helpers from scrpcy.studio."""

from __future__ import annotations

import scrpcy.ui.interactive as _ui_interactive
from scrpcy.studio import (
    CLIStudio,
    SessionCloseSignal,
    SessionDetachSignal,
    _AgentProgressTracker,
    SCRPCY_LOGO,
    SCRPCY_VERSION,
)
from scrpcy.ui.interactive import _format_model_display_name

interactive_action_confirm = _ui_interactive.interactive_action_confirm
interactive_item_browser = _ui_interactive.interactive_item_browser
interactive_confirm = _ui_interactive.interactive_confirm
interactive_token_warning_confirm = _ui_interactive.interactive_token_warning_confirm
interactive_detail_viewer = _ui_interactive.interactive_detail_viewer
interactive_help_view = _ui_interactive.interactive_help_view
interactive_input_prompt = _ui_interactive.interactive_input_prompt
interactive_menu = _ui_interactive.interactive_menu
interactive_model_selector = _ui_interactive.interactive_model_selector
interactive_skill_selector = _ui_interactive.interactive_skill_selector

__all__ = [
    "CLIStudio",
    "SessionCloseSignal",
    "SessionDetachSignal",
    "_AgentProgressTracker",
    "SCRPCY_LOGO",
    "SCRPCY_VERSION",
    "_format_model_display_name",
    "interactive_action_confirm",
    "interactive_item_browser",
    "interactive_confirm",
    "interactive_token_warning_confirm",
    "interactive_detail_viewer",
    "interactive_help_view",
    "interactive_input_prompt",
    "interactive_menu",
    "interactive_model_selector",
    "interactive_skill_selector",
]
