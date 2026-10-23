"""Core subpackage for scrpcy document ingestion, storage, parsing, skills, LLM engine, search, and MCP document generation."""

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
    "ScrpcyConfig",
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
    "CitationPreferences": "scrpcy.core.config",
    "ConfigManager": "scrpcy.core.config",
    "ScrpcyConfig": "scrpcy.core.config",
    "EvidenceChunk": "scrpcy.core.engine",
    "EvidenceSynthesizer": "scrpcy.core.engine",
    "LLMEngine": "scrpcy.core.engine",
    "SynthesisResult": "scrpcy.core.engine",
    "GeminiMetadataExtractor": "scrpcy.core.extractor",
    "MetadataExtractor": "scrpcy.core.extractor",
    "MockMetadataExtractor": "scrpcy.core.extractor",
    "StudioLogger": "scrpcy.core.logger",
    "get_studio_logger": "scrpcy.core.logger",
    "BaseMCPTransport": "scrpcy.core.mcp_client",
    "CitationItem": "scrpcy.core.mcp_client",
    "MockMCPTransport": "scrpcy.core.mcp_client",
    "StdioMCPTransport": "scrpcy.core.mcp_client",
    "WordCraftClient": "scrpcy.core.mcp_client",
    "format_inline_citation_apa": "scrpcy.core.mcp_client",
    "format_inline_citation_ieee": "scrpcy.core.mcp_client",
    "format_reference_entry_apa": "scrpcy.core.mcp_client",
    "format_reference_entry_ieee": "scrpcy.core.mcp_client",
    "DocumentMetadata": "scrpcy.core.models",
    "ParsedDocument": "scrpcy.core.models",
    "DocumentParser": "scrpcy.core.parser",
    "parse_pdf": "scrpcy.core.parser",
    "AcademicPolisher": "scrpcy.core.polisher",
    "PolishResult": "scrpcy.core.polisher",
    "ProjectInfo": "scrpcy.core.project",
    "ProjectManager": "scrpcy.core.project",
    "append_vault_bibliography": "scrpcy.core.references",
    "resolve_vault_reference_entries": "scrpcy.core.references",
    "strip_manual_bibliography": "scrpcy.core.references",
    "LiteratureScout": "scrpcy.core.scout",
    "ScoutQuery": "scrpcy.core.scout",
    "ScoutResult": "scrpcy.core.scout",
    "PaperSearcher": "scrpcy.core.search",
    "SearchResult": "scrpcy.core.search",
    "sanitize_filename": "scrpcy.core.search",
    "Skill": "scrpcy.core.skills",
    "SkillsLoader": "scrpcy.core.skills",
    "WorkspaceManager": "scrpcy.core.workspace",
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
