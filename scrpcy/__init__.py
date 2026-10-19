"""scrpcy: A local-first academic research studio."""

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
from scrpcy.exceptions import (
    ConfigError,
    CorruptedPDFError,
    DocumentGenerationError,
    DownloadError,
    EngineError,
    MCPConnectionError,
    MCPError,
    MetadataExtractionError,
    NoOpenAccessPDFError,
    PDFParseError,
    RateLimitError,
    ScrpcyError,
    SearchError,
    SkillNotFoundError,
    WorkspaceError,
)

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
