"""Domain exceptions for scrpcy."""


class ScrpcyError(Exception):
    """Base exception for all errors within the scrpcy system."""


class WorkspaceError(ScrpcyError):
    """Raised when workspace path resolution, directory creation, or caching fails."""


class PDFParseError(ScrpcyError):
    """Base exception for PDF ingestion and parsing errors."""


class CorruptedPDFError(PDFParseError, ValueError):
    """Raised when an ingested PDF file is unreadable, truncated, empty, or not a valid PDF."""


class MetadataExtractionError(PDFParseError):
    """Raised when LLM-based metadata extraction fails or returns an unrecoverable payload."""


class ConfigError(ScrpcyError):
    """Raised when configuration validation, loading, or saving fails."""


class SkillNotFoundError(ScrpcyError):
    """Raised when a requested skill markdown file is not found."""


class EngineError(ScrpcyError):
    """Raised when LLM query execution or context synthesis fails."""


class SearchError(ScrpcyError):
    """Base exception for paper search API errors."""


class RateLimitError(SearchError):
    """Raised when academic search APIs return 429 Too Many Requests."""


class DownloadError(SearchError):
    """Raised when downloading a paper PDF fails."""


class NoOpenAccessPDFError(DownloadError):
    """Raised when a selected paper does not have a public open-access PDF URL."""


class DocumentGenerationError(ScrpcyError):
    """Raised when creating or writing DOCX documents fails."""


class MCPError(ScrpcyError):
    """Base exception for Model Context Protocol client and tool errors."""


class MCPConnectionError(MCPError):
    """Raised when communication with the MCP server fails or times out."""



