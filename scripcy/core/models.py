"""Pydantic models for academic document metadata and parsed outputs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

FRONTMATTER_REGEX = re.compile(
    r"^\s*---\r?\n(.*?)\r?\n---\r?\n?(.*)$",
    re.DOTALL,
)
FRONTMATTER_PATTERN = FRONTMATTER_REGEX


class DocumentMetadata(BaseModel):
    """Structured publication metadata extracted from academic documents and official web pages."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    title: str = Field(
        default="",
        description="The primary title of the academic paper or document.",
    )
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names in publication order.",
    )
    year: int | None = Field(
        default=None,
        description="Four-digit publication year.",
    )
    doi: str | None = Field(
        default=None,
        description="Digital Object Identifier (DOI), e.g. 10.1145/1234567.1234568.",
    )
    cite_key: str | None = Field(
        default=None,
        description="Citation or BibTeX key (e.g. vaswani2017attention).",
    )
    source_type: str = Field(
        default="paper",
        description="Type of source: 'paper' or 'official_web'.",
    )
    url: str | None = Field(
        default=None,
        description="Source web URL if applicable.",
    )
    institution: str | None = Field(
        default=None,
        description="Publishing government agency, university, or official institution.",
    )
    access_date: str | None = Field(
        default=None,
        description="Date the web page was retrieved (YYYY-MM-DD).",
    )
    authority_score: float | None = Field(
        default=None,
        description="Heuristic domain/content authority score (0.0 to 1.0).",
    )
    authority_tier: int | None = Field(
        default=None,
        description="Authority tier (1=Government/IGO/Academic, 2=Standards, 3=News, 4=Unverified).",
    )
    context: str = Field(
        default="",
        description="User-defined or folder-derived experiment/research context.",
    )

    @field_validator("title", mode="before")
    @classmethod
    def sanitize_title(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("authors", mode="before")
    @classmethod
    def normalize_authors(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r",|\band\b|;", v) if p.strip()]
            return parts or [v.strip()]
        if isinstance(v, (list, tuple)):
            cleaned: list[str] = []
            for item in v:
                if item and str(item).strip():
                    cleaned.append(str(item).strip())
            return cleaned
        return [str(v).strip()]

    @field_validator("year", mode="before")
    @classmethod
    def parse_year(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            match = re.search(r"\b(19\d\d|20\d\d)\b", v)
            if match:
                return int(match.group(1))
            try:
                val = int(v.strip())
                return val
            except ValueError:
                return None
        return None

    @field_validator("doi", "cite_key", mode="before")
    @classmethod
    def clean_identifier(cls, v: Any) -> str | None:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned if cleaned else None

    def ensure_cite_key(self) -> str:
        """Return the existing cite_key or generate a deterministic fallback key."""
        if self.cite_key:
            return self.cite_key

        first_author = "unknown"
        if self.institution and (not self.authors or self.authors == [self.institution]):
            m_acr = re.search(r"\(([A-Za-z0-9]+)\)", self.institution)
            if m_acr:
                first_author = m_acr.group(1).lower()
            else:
                words = [w for w in re.sub(r"[^\w\s]", "", self.institution).split() if w]
                first_author = words[0].lower() if words else "inst"
        elif self.authors:
            name_parts = self.authors[0].strip().split()
            if name_parts:
                first_author = re.sub(r"[^\w]", "", name_parts[-1].lower())

        year_str = str(self.year) if self.year else "nodate"
        slug = ""
        if self.title:
            words = [w for w in re.sub(r"[^\w\s]", "", self.title.lower()).split() if len(w) > 3]
            if words:
                slug = words[0]

        generated = f"{first_author}{year_str}{slug}"
        self.cite_key = generated
        return generated

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert metadata to a clean dictionary for YAML serialization."""
        data: dict[str, Any] = {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "cite_key": self.cite_key,
        }
        if self.source_type and self.source_type != "paper":
            data["source_type"] = self.source_type
        if self.url:
            data["url"] = self.url
        if self.institution:
            data["institution"] = self.institution
        if self.access_date:
            data["access_date"] = self.access_date
        if self.authority_score is not None:
            data["authority_score"] = self.authority_score
        if self.authority_tier is not None:
            data["authority_tier"] = self.authority_tier
        return data

    def to_frontmatter(self) -> str:
        """Render metadata as a YAML frontmatter block enclosed by '---' markers."""
        data = self.to_yaml_dict()
        yaml_content = yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        return f"---\n{yaml_content}---\n"

    @classmethod
    def from_frontmatter(cls, content: str) -> tuple[DocumentMetadata, str]:
        """Extract and validate metadata from a markdown frontmatter block.

        Args:
            content: Complete markdown content containing YAML frontmatter.

        Returns:
            Tuple of (DocumentMetadata instance, remaining markdown body).

        Raises:
            ValueError: If frontmatter syntax is missing or invalid YAML.
        """
        match = FRONTMATTER_REGEX.match(content)
        if not match:
            raise ValueError("No valid YAML frontmatter delimiter ('---') found in content.")

        yaml_block = match.group(1)
        body = match.group(2)

        try:
            parsed_data = yaml.safe_load(yaml_block)
        except yaml.YAMLError as exc:
            raise ValueError(f"Malformed YAML in frontmatter: {exc}") from exc

        if not isinstance(parsed_data, dict):
            parsed_data = {}

        metadata = cls.model_validate(parsed_data)
        return metadata, body


class ParsedDocument(BaseModel):
    """Container for the output of document ingestion and parsing."""

    file_path: Path
    cache_path: Path | None = None
    metadata: DocumentMetadata
    markdown: str
    body_markdown: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
