"""LLM-based academic metadata extraction via Google GenAI SDK and LiteLLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import re
from typing import Any, Literal

from pydantic import ValidationError

from scrpcy.core.models import DocumentMetadata
from scrpcy.exceptions import MetadataExtractionError

SYSTEM_INSTRUCTION = (
    "You are an academic publication metadata extraction system. "
    "Analyze the provided text from the first page of an academic paper. "
    "Extract the following fields:\n"
    "- title: The full formal title of the paper.\n"
    "- authors: Ordered list of author names.\n"
    "- year: Publication or preprint year as an integer (e.g. 2024).\n"
    "- doi: Digital Object Identifier string if present (e.g. 10.1145/1234567).\n"
    "- cite_key: Standard citation key (e.g. vaswani2017attention) if present or derivable.\n"
    "Return only valid JSON matching the schema."
)


def clean_json_text(text: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if match:
        return match.group(1).strip()
    return stripped


class MetadataExtractor(ABC):
    """Abstract interface for extracting structured metadata from document text."""

    @abstractmethod
    def extract(self, text: str) -> DocumentMetadata:
        """Extract metadata from first-page text.

        Args:
            text: Markdown or plain text from the first page of a document.

        Returns:
            Validated DocumentMetadata instance.

        Raises:
            MetadataExtractionError: If extraction fails or output is invalid.
        """


# Defensive guard for google-genai upstream bug where BaseApiClient.aclose crashes on GC if __init__ failed
try:
    from google.genai._api_client import BaseApiClient  # type: ignore

    _orig_aclose = BaseApiClient.aclose

    async def _safe_aclose(self: Any) -> None:
        if hasattr(self, "_async_httpx_client") and self._async_httpx_client:
            await _orig_aclose(self)

    BaseApiClient.aclose = _safe_aclose
except Exception:
    pass


class GeminiMetadataExtractor(MetadataExtractor):
    """Extracts publication metadata using Gemini via Google GenAI SDK or LiteLLM."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        backend: Literal["google-genai", "litellm"] = "google-genai",
        client: Any | None = None,
    ) -> None:
        """Initialize the Gemini metadata extractor.

        Args:
            model: Gemini model identifier (e.g. gemini-2.5-flash, gemini-1.5-flash).
            api_key: Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY env vars.
            backend: Underlying SDK to use ('google-genai' or 'litellm').
            client: Optional pre-configured client instance for testing or dependency injection.
        """
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self.api_key:
            try:
                from scrpcy.core.config import ScrpcyConfig

                cfg = ScrpcyConfig.load()
                self.api_key = cfg.get_api_key("gemini")
            except Exception:
                pass
        self.backend = backend
        self.client = client

    def extract(self, text: str) -> DocumentMetadata:
        """Extract structured metadata from first-page text using the configured backend."""
        if not text or not text.strip():
            raise MetadataExtractionError("Cannot extract metadata from empty or whitespace text.")

        if self.backend == "google-genai":
            return self._extract_google_genai(text)
        elif self.backend == "litellm":
            return self._extract_litellm(text)
        else:
            raise MetadataExtractionError(f"Unsupported extraction backend: '{self.backend}'")

    def _extract_google_genai(self, text: str) -> DocumentMetadata:
        if not self.api_key:
            raise MetadataExtractionError("No Gemini API key provided for Google GenAI metadata extraction.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise MetadataExtractionError(
                "google-genai is required for the 'google-genai' backend. "
                "Install it with `pip install google-genai`."
            ) from exc

        client = self.client or genai.Client(api_key=self.api_key)
        self.client = client

        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"--- BEGIN FIRST PAGE CONTENT ---\n{text}\n--- END FIRST PAGE CONTENT ---"
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DocumentMetadata,
                    temperature=0.0,
                ),
            )
        except Exception as exc:
            raise MetadataExtractionError(f"Google GenAI API call failed: {exc}") from exc

        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise MetadataExtractionError("Google GenAI API returned an empty response.")

        clean_text = clean_json_text(raw_text)
        try:
            return DocumentMetadata.model_validate_json(clean_text)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise MetadataExtractionError(
                f"Failed to validate metadata schema from Google GenAI response: {exc}"
            ) from exc

    def _extract_litellm(self, text: str) -> DocumentMetadata:
        try:
            import litellm
        except ImportError as exc:
            raise MetadataExtractionError(
                "litellm is required for the 'litellm' backend. "
                "Install it with `pip install litellm`."
            ) from exc

        model_name = self.model
        if not (model_name.startswith("gemini/") or "/" in model_name):
            model_name = f"gemini/{model_name}"

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": f"Document first page content:\n\n{text}",
            },
        ]

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "response_format": DocumentMetadata,
            "temperature": 0.0,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            if self.client and hasattr(self.client, "completion"):
                response = self.client.completion(**kwargs)
            else:
                response = litellm.completion(**kwargs)
        except Exception as exc:
            raise MetadataExtractionError(f"LiteLLM completion call failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise MetadataExtractionError("LiteLLM returned invalid response structure.") from exc

        if not content:
            raise MetadataExtractionError("LiteLLM returned empty completion content.")

        clean_text = clean_json_text(content)
        try:
            return DocumentMetadata.model_validate_json(clean_text)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise MetadataExtractionError(
                f"Failed to validate metadata schema from LiteLLM response: {exc}"
            ) from exc


class MockMetadataExtractor(MetadataExtractor):
    """Deterministic mock extractor for unit tests and offline environments."""

    def __init__(self, fixed_metadata: DocumentMetadata | None = None) -> None:
        self.fixed_metadata = fixed_metadata

    def extract(self, text: str) -> DocumentMetadata:
        if not text or not text.strip():
            raise MetadataExtractionError("Cannot extract metadata from empty text.")

        if self.fixed_metadata is not None:
            return self.fixed_metadata

        # Simple deterministic extraction for test scenarios
        return DocumentMetadata(
            title="Mock Academic Title",
            authors=["Mock Author One", "Mock Author Two"],
            year=2024,
            doi="10.1000/182",
            cite_key="mock2024title",
        )
