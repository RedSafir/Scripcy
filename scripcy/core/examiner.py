"""
Examiner (formerly Devil's Advocate / Critic) for Scripcy.
Audits research evidence, evaluates claim-to-source grounding, identifies gaps,
and enforces academic rigor and anti-slop standards without artificial token caps.
"""

from __future__ import annotations

# Re-export core classes with new standardized naming
from scripcy.core.critic import (
    DevilsAdvocate as Examiner,
    DevilsAdvocate,  # Backward compatibility
    CritiqueReport,
    PlanAuditReview,
    EvidenceAuditReview,
)

__all__ = [
    "Examiner",
    "DevilsAdvocate",
    "CritiqueReport",
    "PlanAuditReview",
    "EvidenceAuditReview",
]
