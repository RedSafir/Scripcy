"""
Manager (formerly SubagentOrchestrator) for Scripcy.
Orchestrates high-level research workflows, task steps, and integrates
with SessionContextManager for token-efficient working memory.
"""

from __future__ import annotations

# Re-export orchestrator components with new standardized naming
from scripcy.core.orchestrator import (
    SubagentOrchestrator as SubagentManager,
    SubagentOrchestrator,  # Backward compatibility
    OrchestrationResult as ManagementResult,
    OrchestrationResult,   # Backward compatibility
)

__all__ = [
    "SubagentManager",
    "SubagentOrchestrator",
    "ManagementResult",
    "OrchestrationResult",
]
