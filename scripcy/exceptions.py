"""Domain exceptions for scripcy."""


class ScripcyError(Exception):
    """Base exception for all errors within the scripcy system."""


# Backward-compatibility alias
ScrpcyError = ScripcyError


class WorkspaceError(ScripcyError):
    """Raised when workspace path resolution, directory creation, or caching fails."""


class PDFParseError(ScripcyError):
    """Base exception for PDF ingestion and parsing errors."""


class CorruptedPDFError(PDFParseError, ValueError):
    """Raised when an ingested PDF file is unreadable, truncated, empty, or not a valid PDF."""


class MetadataExtractionError(PDFParseError):
    """Raised when LLM-based metadata extraction fails or returns an unrecoverable payload."""


class ConfigError(ScripcyError):
    """Raised when configuration validation, loading, or saving fails."""


class SkillNotFoundError(ScripcyError):
    """Raised when a requested skill markdown file is not found."""


class EngineError(ScripcyError):
    """Raised when LLM query execution or context synthesis fails."""


class SearchError(ScripcyError):
    """Base exception for paper search API errors."""


class RateLimitError(SearchError):
    """Raised when academic search APIs return 429 Too Many Requests."""


class DownloadError(SearchError):
    """Raised when downloading a paper PDF fails."""


class NoOpenAccessPDFError(DownloadError):
    """Raised when a selected paper does not have a public open-access PDF URL."""


class DocumentGenerationError(ScripcyError):
    """Raised when creating or writing DOCX documents fails."""


class MCPError(ScripcyError):
    """Base exception for Model Context Protocol client and tool errors."""


class MCPConnectionError(MCPError):
    """Raised when communication with the MCP server fails or times out."""


class WebScoutError(ScripcyError):
    """Base exception for web searching, official filtering, and web ingestion errors."""


class OfficialSourceRejectedError(WebScoutError):
    """Raised when a candidate web source is rejected for failing official authority criteria."""
