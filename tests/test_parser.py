"""Comprehensive test suite for PDF ingestion, markdown extraction, and metadata parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pydantic
import pymupdf
import pytest
import yaml

from scrpcy.core.extractor import (
    GeminiMetadataExtractor,
    MockMetadataExtractor,
    clean_json_text,
)
from scrpcy.core.models import DocumentMetadata, ParsedDocument
from scrpcy.core.parser import DocumentParser, parse_pdf
from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import (
    CorruptedPDFError,
    MetadataExtractionError,
    PDFParseError,
)


class TestPDFParsing:
    """Tests for PDF ingestion, PyMuPDF4LLM extraction, and cached outputs."""

    def test_parse_single_page_pdf(
        self,
        synthetic_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor: MockMetadataExtractor,
    ) -> None:
        pdf_path = synthetic_pdf_factory(filename="attention.pdf")
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)

        result = parser.parse(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.file_path == pdf_path.resolve()
        assert result.cache_path is not None
        assert result.cache_path == temp_workspace.get_cache_path("attention.md")
        assert result.cache_path.is_file()

        cached_content = temp_workspace.read_from_cache("attention.md")
        assert cached_content.startswith("---")
        assert "Attention Is All You Need" in cached_content
        assert "Ashish Vaswani" in cached_content
        assert result.metadata.title == "Attention Is All You Need"
        assert result.metadata.year == 2017

    def test_parse_multi_page_pdf(
        self,
        synthetic_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor: MockMetadataExtractor,
    ) -> None:
        pdf_path = synthetic_pdf_factory(
            filename="multipage.pdf",
            pages_count=3,
            body_sections=[
                "Page 1 Content: Abstract and Introduction.",
                "Page 2 Content: Methodology and Model Architecture.",
                "Page 3 Content: Experimental Results and Benchmark Tables.",
            ],
        )
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)
        result = parser.parse(pdf_path)

        assert "Page 1 Content" in result.markdown
        assert "Page 2 Content" in result.markdown
        assert "Page 3 Content" in result.markdown
        assert result.cache_path is not None
        assert result.cache_path.is_file()

    def test_parse_without_cache(
        self,
        synthetic_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor: MockMetadataExtractor,
    ) -> None:
        pdf_path = synthetic_pdf_factory(filename="no_cache.pdf")
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)

        result = parser.parse(pdf_path, save_to_cache=False)

        assert result.cache_path is None
        assert temp_workspace.cache_exists("no_cache.md") is False
        assert result.markdown.startswith("---")

    def test_parse_with_custom_output_filename(
        self,
        synthetic_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor: MockMetadataExtractor,
    ) -> None:
        pdf_path = synthetic_pdf_factory(filename="source.pdf")
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)

        result = parser.parse(pdf_path, output_filename="custom_target.md")

        assert result.cache_path is not None
        assert result.cache_path.name == "custom_target.md"
        assert temp_workspace.cache_exists("custom_target.md") is True

    def test_convenience_function_parse_pdf(
        self,
        synthetic_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor: MockMetadataExtractor,
    ) -> None:
        pdf_path = synthetic_pdf_factory(filename="convenience.pdf")
        result = parse_pdf(
            pdf_path,
            workspace=temp_workspace,
            extractor=mock_extractor,
        )

        assert isinstance(result, ParsedDocument)
        assert result.cache_path == temp_workspace.get_cache_path("convenience.md")
        assert result.cache_path.is_file()

    def test_cite_key_auto_generation_when_missing(
        self,
        synthetic_pdf_factory,
        temp_workspace: WorkspaceManager,
    ) -> None:
        pdf_path = synthetic_pdf_factory(filename="paper_no_key.pdf")
        extractor = MockMetadataExtractor(
            fixed_metadata=DocumentMetadata(
                title="Deep Residual Learning",
                authors=["Kaiming He", "Xiangyu Zhang"],
                year=2016,
                doi="10.1109/CVPR.2016.90",
                cite_key=None,
            )
        )
        parser = DocumentParser(workspace=temp_workspace, extractor=extractor)
        result = parser.parse(pdf_path)

        assert result.metadata.cite_key is not None
        assert "he2016" in result.metadata.cite_key
        assert f"cite_key: {result.metadata.cite_key}" in result.markdown


class TestFrontmatterAndSchemaValidation:
    """Tests for YAML frontmatter formatting, serialization, and Pydantic schema validation."""

    def test_frontmatter_formatting_structure(self, sample_metadata: DocumentMetadata) -> None:
        frontmatter = sample_metadata.to_frontmatter()

        assert frontmatter.startswith("---\n")
        assert frontmatter.endswith("---\n")
        parsed_yaml = yaml.safe_load(frontmatter.strip("- \n\r"))
        assert parsed_yaml["title"] == sample_metadata.title
        assert parsed_yaml["authors"] == sample_metadata.authors
        assert parsed_yaml["year"] == sample_metadata.year
        assert parsed_yaml["doi"] == sample_metadata.doi
        assert parsed_yaml["cite_key"] == sample_metadata.cite_key

    def test_frontmatter_roundtrip(self, sample_metadata: DocumentMetadata) -> None:
        body = "# Section 1\nThis is clean markdown content."
        composite = f"{sample_metadata.to_frontmatter()}\n{body}"

        parsed_meta, parsed_body = DocumentMetadata.from_frontmatter(composite)

        assert parsed_meta.title == sample_metadata.title
        assert parsed_meta.authors == sample_metadata.authors
        assert parsed_meta.year == sample_metadata.year
        assert parsed_meta.doi == sample_metadata.doi
        assert parsed_meta.cite_key == sample_metadata.cite_key
        assert parsed_body.strip() == body.strip()

    def test_authors_validation_and_normalization(self) -> None:
        # Single string with comma separation
        meta1 = DocumentMetadata(
            title="Paper A",
            authors="Alice Smith, Bob Jones, Carol Danvers",
        )
        assert meta1.authors == ["Alice Smith", "Bob Jones", "Carol Danvers"]

        # String with 'and'
        meta2 = DocumentMetadata(
            title="Paper B",
            authors="Alice Smith and Bob Jones",
        )
        assert meta2.authors == ["Alice Smith", "Bob Jones"]

        # List with whitespace
        meta3 = DocumentMetadata(
            title="Paper C",
            authors=[" Alice Smith  ", "Bob Jones"],
        )
        assert meta3.authors == ["Alice Smith", "Bob Jones"]

        # None authors
        meta4 = DocumentMetadata(title="Paper D", authors=None)
        assert meta4.authors == []

    def test_year_parsing_and_normalization(self) -> None:
        assert DocumentMetadata(title="P", year=2024).year == 2024
        assert DocumentMetadata(title="P", year="2024").year == 2024
        assert DocumentMetadata(title="P", year="Published November 2023").year == 2023
        assert DocumentMetadata(title="P", year="InvalidYearString").year is None
        assert DocumentMetadata(title="P", year=None).year is None
        assert DocumentMetadata(title="P", year="").year is None

    def test_doi_and_cite_key_sanitization(self) -> None:
        meta = DocumentMetadata(
            title="Paper",
            doi="  10.1000/182  ",
            cite_key="  smith2024  ",
        )
        assert meta.doi == "10.1000/182"
        assert meta.cite_key == "smith2024"

        meta_empty = DocumentMetadata(title="Paper", doi="   ", cite_key="")
        assert meta_empty.doi is None
        assert meta_empty.cite_key is None

    def test_frontmatter_missing_delimiter_raises_value_error(self) -> None:
        content = "title: Test\nauthors: [Alice]\n# Header\nNo frontmatter delimiters."
        with pytest.raises(ValueError, match="No valid YAML frontmatter delimiter"):
            DocumentMetadata.from_frontmatter(content)

    def test_frontmatter_malformed_yaml_raises_value_error(self) -> None:
        content = "---\n: : this is not valid yaml :::\n---\nBody markdown"
        with pytest.raises(ValueError, match="Malformed YAML"):
            DocumentMetadata.from_frontmatter(content)

    def test_pydantic_schema_extra_fields_ignored(self) -> None:
        raw_payload = {
            "title": "Valid Title",
            "authors": ["Author"],
            "year": 2024,
            "extra_unrecognized_key": "some value",
        }
        metadata = DocumentMetadata.model_validate(raw_payload)
        assert metadata.title == "Valid Title"
        assert not hasattr(metadata, "extra_unrecognized_key")


class TestCorruptedFileHandling:
    """Tests for edge cases and corrupted or invalid PDF file inputs."""

    def test_file_not_found(self, temp_workspace: WorkspaceManager, mock_extractor) -> None:
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)
        non_existent = Path("non_existent_paper_12345.pdf")

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            parser.parse(non_existent)

    def test_path_is_directory(
        self,
        tmp_path: Path,
        temp_workspace: WorkspaceManager,
        mock_extractor,
    ) -> None:
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)
        directory_path = tmp_path / "not_a_file_dir"
        directory_path.mkdir()

        with pytest.raises(PDFParseError, match="not a regular file"):
            parser.parse(directory_path)

    def test_empty_zero_byte_file(
        self,
        corrupted_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor,
    ) -> None:
        empty_pdf = corrupted_pdf_factory("empty.pdf", b"")
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)

        with pytest.raises(CorruptedPDFError, match="empty"):
            parser.parse(empty_pdf)

    def test_random_garbage_bytes(
        self,
        corrupted_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor,
    ) -> None:
        garbage_pdf = corrupted_pdf_factory(
            "garbage.pdf",
            b"GARBAGE_DATA_NOT_A_VALID_PDF_STRUCTURE_AT_ALL_987654321",
        )
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)

        with pytest.raises(CorruptedPDFError):
            parser.parse(garbage_pdf)

    def test_truncated_pdf_header(
        self,
        corrupted_pdf_factory,
        temp_workspace: WorkspaceManager,
        mock_extractor,
    ) -> None:
        truncated_pdf = corrupted_pdf_factory("truncated.pdf", b"%PDF-1.4\n%%EOF_CORRUPT")
        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)

        with pytest.raises(CorruptedPDFError):
            parser.parse(truncated_pdf)

    def test_zero_page_pdf(
        self,
        tmp_path: Path,
        temp_workspace: WorkspaceManager,
        mock_extractor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        zero_page_file = tmp_path / "zero_pages.pdf"
        zero_page_file.write_bytes(b"%PDF-1.4\n%mocked")

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        monkeypatch.setattr("pymupdf.open", lambda *a, **k: mock_doc)

        parser = DocumentParser(workspace=temp_workspace, extractor=mock_extractor)
        with pytest.raises(CorruptedPDFError, match="0 pages"):
            parser.parse(zero_page_file)


class TestGeminiMetadataExtractor:
    """Tests for Gemini metadata extraction via Google GenAI SDK and LiteLLM backends."""

    def test_clean_json_text_removes_fences(self) -> None:
        raw_json = '{"title": "Test Title", "year": 2024}'
        fenced_json = f"```json\n{raw_json}\n```"
        assert clean_json_text(fenced_json) == raw_json

        plain_fenced = f"```\n{raw_json}\n```"
        assert clean_json_text(plain_fenced) == raw_json

        assert clean_json_text(raw_json) == raw_json

    def test_google_genai_backend_success(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "title": "Quantum Computing Advances",
            "authors": ["Dr. Alice", "Dr. Bob"],
            "year": 2024,
            "doi": "10.1038/s41586-024-0001",
            "cite_key": "alice2024quantum",
        })
        mock_client.models.generate_content.return_value = mock_response

        extractor = GeminiMetadataExtractor(
            backend="google-genai",
            api_key="mock_key",
            client=mock_client,
        )
        metadata = extractor.extract("Sample first page paper text")

        assert metadata.title == "Quantum Computing Advances"
        assert metadata.authors == ["Dr. Alice", "Dr. Bob"]
        assert metadata.year == 2024
        assert metadata.doi == "10.1038/s41586-024-0001"
        assert metadata.cite_key == "alice2024quantum"
        mock_client.models.generate_content.assert_called_once()

    def test_google_genai_backend_fenced_json(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "```json\n" + json.dumps({
            "title": "Fenced Title",
            "authors": ["Author"],
            "year": 2023,
            "doi": None,
            "cite_key": None,
        }) + "\n```"
        mock_client.models.generate_content.return_value = mock_response

        extractor = GeminiMetadataExtractor(
            backend="google-genai",
            api_key="mock_key",
            client=mock_client,
        )
        metadata = extractor.extract("Sample text")
        assert metadata.title == "Fenced Title"
        assert metadata.year == 2023

    def test_google_genai_backend_api_error_raises_exception(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API Quota Exceeded")

        extractor = GeminiMetadataExtractor(
            backend="google-genai",
            api_key="mock_key",
            client=mock_client,
        )
        with pytest.raises(MetadataExtractionError, match="Google GenAI API call failed"):
            extractor.extract("Sample text")

    def test_google_genai_backend_empty_response_raises_exception(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        extractor = GeminiMetadataExtractor(
            backend="google-genai",
            api_key="mock_key",
            client=mock_client,
        )
        with pytest.raises(MetadataExtractionError, match="empty response"):
            extractor.extract("Sample text")

    def test_litellm_backend_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "title": "LiteLLM Extracted Paper",
            "authors": ["Charlie Davis"],
            "year": 2022,
            "doi": "10.1145/34567",
            "cite_key": "davis2022paper",
        })
        mock_response.choices = [mock_choice]

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = mock_response
        monkeypatch.setattr("litellm.completion", mock_litellm.completion)

        extractor = GeminiMetadataExtractor(
            backend="litellm",
            api_key="mock_key",
            model="gemini-2.5-flash",
        )
        metadata = extractor.extract("First page text for LiteLLM")

        assert metadata.title == "LiteLLM Extracted Paper"
        assert metadata.authors == ["Charlie Davis"]
        assert metadata.year == 2022
        assert metadata.doi == "10.1145/34567"
        assert metadata.cite_key == "davis2022paper"

    def test_litellm_backend_failure_raises_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_completion(**kwargs):
            raise ConnectionError("Network failure")

        monkeypatch.setattr("litellm.completion", failing_completion)

        extractor = GeminiMetadataExtractor(
            backend="litellm",
            api_key="mock_key",
        )
        with pytest.raises(MetadataExtractionError, match="LiteLLM completion call failed"):
            extractor.extract("Sample text")

    def test_empty_text_input_raises_exception(self) -> None:
        extractor = GeminiMetadataExtractor()
        with pytest.raises(MetadataExtractionError, match="empty"):
            extractor.extract("   \n\t  ")

    def test_unsupported_backend_raises_exception(self) -> None:
        extractor = GeminiMetadataExtractor(backend="unsupported_backend")  # type: ignore[arg-type]
        with pytest.raises(MetadataExtractionError, match="Unsupported extraction backend"):
            extractor.extract("Valid text")
