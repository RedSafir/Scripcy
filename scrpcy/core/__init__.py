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
from scrpcy.core.logger import StudioLogger, get_studio_logger
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
from scrpcy.core.polisher import AcademicPolisher, PolishResult
from scrpcy.core.project import ProjectInfo, ProjectManager
from scrpcy.core.references import (
    append_vault_bibliography,
    resolve_vault_reference_entries,
    strip_manual_bibliography,
)
from scrpcy.core.scout import LiteratureScout, ScoutQuery, ScoutResult
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
