"""scripcy: A local-first academic research studio."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BaseMCPTransport",
    "CitationItem",
    "CitationPreferences",
    "ConfigError",
    "ConfigManager",
    "CorruptedPDFError",
    "DocumentGenerationError",
    "DocumentMetadata",
    "DocumentParser",
    "DownloadError",
    "EngineError",
    "EvidenceChunk",
    "EvidenceSynthesizer",
    "GeminiMetadataExtractor",
    "LLMEngine",
    "MCPConnectionError",
    "MCPError",
    "MetadataExtractionError",
    "MetadataExtractor",
    "MockMCPTransport",
    "MockMetadataExtractor",
    "NoOpenAccessPDFError",
    "OfficialDomainFilter",
    "OfficialSourceRejectedError",
    "PDFParseError",
    "ParsedDocument",
    "PaperSearcher",
    "RateLimitError",
    "ScripcyConfig",
    "ScripcyError",
    "SearchError",
    "SearchResult",
    "Skill",
    "SkillNotFoundError",
    "SkillsLoader",
    "StdioMCPTransport",
    "SubstantiveVisualAsset",
    "SynthesisResult",
    "WebScout",
    "WebScoutError",
    "WebSearchResult",
    "WordCraftClient",
    "WorkspaceError",
    "WorkspaceManager",
    "format_inline_citation_apa",
    "format_inline_citation_ieee",
    "format_reference_entry_apa",
    "format_reference_entry_ieee",
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
    "PaperSearcher": "scripcy.core.search",
    "SearchResult": "scripcy.core.search",
    "sanitize_filename": "scripcy.core.search",
    "Skill": "scripcy.core.skills",
    "SkillsLoader": "scripcy.core.skills",
    "WorkspaceManager": "scripcy.core.workspace",
    "ConfigError": "scripcy.exceptions",
    "CorruptedPDFError": "scripcy.exceptions",
    "DocumentGenerationError": "scripcy.exceptions",
    "DownloadError": "scripcy.exceptions",
    "EngineError": "scripcy.exceptions",
    "MCPConnectionError": "scripcy.exceptions",
    "MCPError": "scripcy.exceptions",
    "MetadataExtractionError": "scripcy.exceptions",
    "NoOpenAccessPDFError": "scripcy.exceptions",
    "PDFParseError": "scripcy.exceptions",
    "RateLimitError": "scripcy.exceptions",
    "ScripcyError": "scripcy.exceptions",
    "SearchError": "scripcy.exceptions",
    "SkillNotFoundError": "scripcy.exceptions",
    "WorkspaceError": "scripcy.exceptions",
    "WebScoutError": "scripcy.exceptions",
    "OfficialSourceRejectedError": "scripcy.exceptions",
    "OfficialDomainFilter": "scripcy.core.web_scout",
    "WebScout": "scripcy.core.web_scout",
    "WebSearchResult": "scripcy.core.web_scout",
    "SubstantiveVisualAsset": "scripcy.core.web_scout",
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
