"""Core subpackage for scripcy document ingestion, storage, parsing, skills, LLM engine, search, and MCP document generation."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BaseMCPTransport",
    "CitationItem",
    "CitationPreferences",
    "ConfigManager",
    "DocumentMetadata",
    "DocumentParser",
    "EvidenceChunk",
    "EvidenceSynthesizer",
    "GeminiMetadataExtractor",
    "LiteratureScout",
    "LLMEngine",
    "MetadataExtractor",
    "MockMCPTransport",
    "MockMetadataExtractor",
    "PaperSearcher",
    "ParsedDocument",
    "PolishResult",
    "AcademicPolisher",
    "append_vault_bibliography",
    "resolve_vault_reference_entries",
    "strip_manual_bibliography",
    "ProjectInfo",
    "ProjectManager",
    "ScoutQuery",
    "ScoutResult",
    "ScripcyConfig",
    "SearchResult",
    "Skill",
    "SkillsLoader",
    "StdioMCPTransport",
    "SynthesisResult",
    "WordCraftClient",
    "WorkspaceManager",
    "format_inline_citation_apa",
    "format_inline_citation_ieee",
    "format_reference_entry_apa",
    "format_reference_entry_ieee",
    "StudioLogger",
    "get_studio_logger",
    "parse_pdf",
    "sanitize_filename",
]

_LAZY_EXPORTS = {
    "CitationPreferences": "scripcy.core.config",
    "ConfigManager": "scripcy.core.config",
    "ScripcyConfig": "scripcy.core.config",
    "EvidenceChunk": "scripcy.core.engine",
    "EvidenceSynthesizer": "scripcy.core.engine",
    "LLMEngine": "scripcy.core.engine",
    "SynthesisResult": "scripcy.core.engine",
    "GeminiMetadataExtractor": "scripcy.core.extractor",
    "MetadataExtractor": "scripcy.core.extractor",
    "MockMetadataExtractor": "scripcy.core.extractor",
    "StudioLogger": "scripcy.core.logger",
    "get_studio_logger": "scripcy.core.logger",
    "BaseMCPTransport": "scripcy.core.mcp_client",
    "CitationItem": "scripcy.core.mcp_client",
    "MockMCPTransport": "scripcy.core.mcp_client",
    "StdioMCPTransport": "scripcy.core.mcp_client",
    "WordCraftClient": "scripcy.core.mcp_client",
    "format_inline_citation_apa": "scripcy.core.mcp_client",
    "format_inline_citation_ieee": "scripcy.core.mcp_client",
    "format_reference_entry_apa": "scripcy.core.mcp_client",
    "format_reference_entry_ieee": "scripcy.core.mcp_client",
    "DocumentMetadata": "scripcy.core.models",
    "ParsedDocument": "scripcy.core.models",
    "DocumentParser": "scripcy.core.parser",
    "parse_pdf": "scripcy.core.parser",
    "AcademicPolisher": "scripcy.core.polisher",
    "PolishResult": "scripcy.core.polisher",
    "ProjectInfo": "scripcy.core.project",
    "ProjectManager": "scripcy.core.project",
    "append_vault_bibliography": "scripcy.core.references",
    "resolve_vault_reference_entries": "scripcy.core.references",
    "strip_manual_bibliography": "scripcy.core.references",
    "LiteratureScout": "scripcy.core.scout",
    "ScoutQuery": "scripcy.core.scout",
    "ScoutResult": "scripcy.core.scout",
    "PaperSearcher": "scripcy.core.search",
    "SearchResult": "scripcy.core.search",
    "sanitize_filename": "scripcy.core.search",
    "Skill": "scripcy.core.skills",
    "SkillsLoader": "scripcy.core.skills",
    "WorkspaceManager": "scripcy.core.workspace",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
