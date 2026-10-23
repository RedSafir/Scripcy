"""CLIStudio composition root uniting all mixins into the unified research studio."""

from __future__ import annotations

from scrpcy.studio.base import StudioBase
from scrpcy.studio.dashboard import StudioDashboardMixin
from scrpcy.studio.vault import StudioVaultMixin
from scrpcy.studio.processing import StudioProcessingMixin
from scrpcy.studio.chat_hub import StudioChatHubMixin
from scrpcy.studio.agent_loop import StudioAgentLoopMixin
from scrpcy.studio.commands import StudioCommandsMixin


class CLIStudio(
    StudioBase,
    StudioDashboardMixin,
    StudioVaultMixin,
    StudioProcessingMixin,
    StudioChatHubMixin,
    StudioAgentLoopMixin,
    StudioCommandsMixin,
):
    """Terminal-native research studio with Dashboard & Master Project modes."""
    pass
