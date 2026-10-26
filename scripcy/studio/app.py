"""CLIStudio composition root uniting all mixins into the unified research studio."""

from __future__ import annotations

from scripcy.studio.base import StudioBase
from scripcy.studio.dashboard import StudioDashboardMixin
from scripcy.studio.vault import StudioVaultMixin
from scripcy.studio.processing import StudioProcessingMixin
from scripcy.studio.chat_hub import StudioChatHubMixin
from scripcy.studio.agent_loop import StudioAgentLoopMixin
from scripcy.studio.commands import StudioCommandsMixin


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
