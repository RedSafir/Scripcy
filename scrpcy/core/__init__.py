"""Core subpackage for scrpcy document ingestion, storage, parsing, skills, LLM engine, search, and MCP document generation."""

from scrpcy.core.config import CitationPreferences, ConfigManager, ScrpcyConfig
from scrpcy.core.engine import (
    EvidenceChunk,
    EvidenceSynthesizer,
    LLMEngine,
    SynthesisResult,
)
from scrpcy.core.extractor import (
    GeminiMetadataExtractor,
    MetadataExtractor,
    MockMetadataExtractor,
)
from scrpcy.core.mcp_client import (
    BaseMCPTransport,
    CitationItem,
    MockMCPTransport,
    StdioMCPTransport,
    WordCraftClient,
    format_inline_citation_apa,
    format_inline_citation_ieee,
    format_reference_entry_apa,
    format_reference_entry_ieee,
)
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser, parse_pdf
from scrpcy.core.search import PaperSearcher, SearchResult, sanitize_filename
from scrpcy.core.skills import Skill, SkillsLoader
from scrpcy.core.workspace import WorkspaceManager

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
    "LLMEngine",
    "MetadataExtractor",
    "MockMCPTransport",
    "MockMetadataExtractor",
    "PaperSearcher",
    "ParsedDocument",
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
    "parse_pdf",
    "sanitize_filename",
]
