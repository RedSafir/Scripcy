"""scrpcy: A local-first academic research studio."""

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
    "PDFParseError",
    "ParsedDocument",
    "PaperSearcher",
    "RateLimitError",
    "ScrpcyConfig",
    "ScrpcyError",
    "SearchError",
    "SearchResult",
    "Skill",
    "SkillNotFoundError",
    "SkillsLoader",
    "StdioMCPTransport",
    "SynthesisResult",
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
    "PaperSearcher": "scrpcy.core.search",
    "SearchResult": "scrpcy.core.search",
    "sanitize_filename": "scrpcy.core.search",
    "Skill": "scrpcy.core.skills",
    "SkillsLoader": "scrpcy.core.skills",
    "WorkspaceManager": "scrpcy.core.workspace",
    "ConfigError": "scrpcy.exceptions",
    "CorruptedPDFError": "scrpcy.exceptions",
    "DocumentGenerationError": "scrpcy.exceptions",
    "DownloadError": "scrpcy.exceptions",
    "EngineError": "scrpcy.exceptions",
    "MCPConnectionError": "scrpcy.exceptions",
    "MCPError": "scrpcy.exceptions",
    "MetadataExtractionError": "scrpcy.exceptions",
    "NoOpenAccessPDFError": "scrpcy.exceptions",
    "PDFParseError": "scrpcy.exceptions",
    "RateLimitError": "scrpcy.exceptions",
    "ScrpcyError": "scrpcy.exceptions",
    "SearchError": "scrpcy.exceptions",
    "SkillNotFoundError": "scrpcy.exceptions",
    "WorkspaceError": "scrpcy.exceptions",
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
