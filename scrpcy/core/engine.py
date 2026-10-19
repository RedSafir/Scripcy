"""LLM orchestration engine with PaperQA-style evidence synthesis and streaming support."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import math
from pathlib import Path
import re
from typing import Any, Callable

import litellm
from pydantic import BaseModel, Field

from scrpcy.core.config import ScrpcyConfig
from scrpcy.core.models import ParsedDocument
from scrpcy.core.skills import Skill, SkillsLoader
from scrpcy.exceptions import EngineError

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
}


class EvidenceChunk(BaseModel):
    """An individual passage extracted from ingested research documents."""

    chunk_id: str = Field(description="Unique identifier for the passage.")
    source: str = Field(description="Originating document identifier or cite key.")
    text: str = Field(description="Textual content of the passage.")
    score: float = Field(default=0.0, description="Relevance score to user query.")
    token_estimate: int = Field(default=0, description="Estimated token count.")


class SynthesisResult(BaseModel):
    """Structured response containing generated research output and supporting evidence."""

    answer: str = Field(description="Generated synthesis or answer text.")
    evidence_chunks: list[EvidenceChunk] = Field(
        default_factory=list,
        description="List of evidence chunks retained and provided in the prompt context.",
    )
    model: str = Field(description="Model identifier used for generation.")
    skill: str = Field(description="Skill or persona applied.")


def estimate_tokens(text: str) -> int:
    """Provide a conservative token count estimate based on word count and punctuation."""
    words = text.split()
    return max(1, math.ceil(len(words) * 1.35))


def tokenize_query(query: str) -> list[str]:
    """Tokenize and filter query terms, preserving both individual words and compound phrases."""
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    filtered = [t for t in raw_tokens if len(t) >= 2 and t not in STOPWORDS]
    compounds = [m.lower() for m in re.findall(r"\b[a-zA-Z0-9]+-[a-zA-Z0-9]+\b", query)]
    return list(dict.fromkeys(filtered + compounds))


class EvidenceSynthesizer:
    """Filters, ranks, and budgets document passages using BM25-inspired lexical relevance."""

    def __init__(
        self,
        chunk_size_words: int = 250,
        chunk_overlap_words: int = 40,
        default_max_chunks: int = 5,
        default_token_budget: int = 4000,
    ) -> None:
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.default_max_chunks = default_max_chunks
        self.default_token_budget = default_token_budget

    def chunk_document(self, text: str, source_name: str) -> list[EvidenceChunk]:
        """Split document text into overlapping windows of words."""
        clean_text = text.strip()
        if not clean_text:
            return []

        words = clean_text.split()
        if len(words) <= self.chunk_size_words:
            chunk_content = " ".join(words)
            return [
                EvidenceChunk(
                    chunk_id=f"{source_name}_c0",
                    source=source_name,
                    text=chunk_content,
                    token_estimate=estimate_tokens(chunk_content),
                )
            ]

        chunks: list[EvidenceChunk] = []
        step = max(1, self.chunk_size_words - self.chunk_overlap_words)
        idx = 0

        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size_words]
            chunk_text = " ".join(window)
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{source_name}_c{idx}",
                    source=source_name,
                    text=chunk_text,
                    token_estimate=estimate_tokens(chunk_text),
                )
            )
            idx += 1
            if start + self.chunk_size_words >= len(words):
                break

        return chunks

    def score_chunk(self, query_terms: list[str], full_query: str, chunk_text: str) -> float:
        """Compute relevance score for a passage based on term frequency and phrase match."""
        if not query_terms:
            return 0.0

        lower_chunk = chunk_text.lower()
        chunk_tokens = re.findall(r"[a-zA-Z0-9]+", lower_chunk)
        chunk_len = len(chunk_tokens)
        if chunk_len == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            freq = chunk_tokens.count(term)
            if freq == 0 and len(term) >= 4:
                stem = term[:4]
                freq = sum(1 for tok in chunk_tokens if tok.startswith(stem))

            if freq > 0:
                tf_score = (freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * (chunk_len / 200.0)))
                score += tf_score

        norm_query = full_query.strip().lower()
        if len(norm_query) > 5 and norm_query in lower_chunk:
            score += 5.0

        return score

    def retrieve_evidence(
        self,
        query: str,
        documents: list[str | Path | ParsedDocument],
        max_chunks: int | None = None,
        token_budget: int | None = None,
    ) -> list[EvidenceChunk]:
        """Decompose documents, rank passages against query, and select within token limits."""
        limit_chunks = max_chunks or self.default_max_chunks
        limit_tokens = token_budget or self.default_token_budget
        query_terms = tokenize_query(query)

        all_chunks: list[EvidenceChunk] = []
        for doc_idx, doc in enumerate(documents):
            source_name = f"doc_{doc_idx + 1}"
            text_content = ""

            if isinstance(doc, ParsedDocument):
                source_name = doc.metadata.cite_key or doc.metadata.title or source_name
                text_content = doc.body_markdown or doc.markdown
            elif isinstance(doc, Path):
                source_name = doc.stem
                if doc.is_file():
                    text_content = doc.read_text(encoding="utf-8")
            elif isinstance(doc, str):
                text_content = doc

            doc_chunks = self.chunk_document(text_content, source_name)
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            return []

        scored_chunks: list[EvidenceChunk] = []
        for chunk in all_chunks:
            chunk.score = self.score_chunk(query_terms, query, chunk.text)
            scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda c: c.score, reverse=True)

        selected: list[EvidenceChunk] = []
        used_tokens = 0

        for chunk in scored_chunks:
            if chunk.score <= 0.0 and selected:
                break
            if len(selected) >= limit_chunks:
                break
            if used_tokens + chunk.token_estimate > limit_tokens:
                if not selected:
                    max_words = max(10, int(limit_tokens / 1.35))
                    truncated_words = chunk.text.split()[:max_words]
                    chunk.text = " ".join(truncated_words) + "..."
                    chunk.token_estimate = estimate_tokens(chunk.text)
                    selected.append(chunk)
                    used_tokens += chunk.token_estimate
                continue

            selected.append(chunk)
            used_tokens += chunk.token_estimate

        return selected

    def format_context_prompt(self, chunks: list[EvidenceChunk]) -> str:
        """Render evidence passages into a clean prompt context block."""
        if not chunks:
            return ""

        formatted_blocks = ["### RETRIEVED RESEARCH EVIDENCE CONTEXT:"]
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"[Evidence {idx}] (Source ID: {chunk.source}, Relevance Score: {chunk.score:.2f}):\n"
                f"\"\"\"\n{chunk.text}\n\"\"\""
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)


class LLMEngine:
    """Orchestrates LiteLLM streaming calls, skill persona injection, and evidence retrieval."""

    def __init__(
        self,
        config: ScrpcyConfig | None = None,
        skills_loader: SkillsLoader | None = None,
        acompletion_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or ScrpcyConfig.load()
        self.skills_loader = skills_loader or SkillsLoader(skills_dir=self.config.skills_dir)
        self.evidence_synthesizer = EvidenceSynthesizer(
            default_max_chunks=self.config.citation.max_evidence_chunks,
            default_token_budget=self.config.citation.context_token_budget,
        )
        self._acompletion_fn = acompletion_fn

    async def _invoke_acompletion(self, **kwargs: Any) -> Any:
        """Invoke litellm.acompletion or custom injected callable."""
        if self._acompletion_fn is not None:
            res = self._acompletion_fn(**kwargs)
            if hasattr(res, "__await__"):
                res = await res
            return res
        return await litellm.acompletion(**kwargs)

    async def stream_query(
        self,
        query: str,
        documents: list[str | Path | ParsedDocument] | None = None,
        skill_name: str = "academic_writer",
        model: str | None = None,
        temperature: float | None = None,
        max_evidence_chunks: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response tokens from LiteLLM using synthesized evidence."""
        if not query or not query.strip():
            raise EngineError("Query string cannot be empty.")

        skill = self.skills_loader.get_skill(skill_name)
        active_model = model or skill.model or self.config.default_model
        active_temp = temperature if temperature is not None else (skill.temperature if skill.temperature is not None else 0.2)
        api_key = self.config.get_api_key_for_model(active_model)

        evidence_chunks = self.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=documents or [],
            max_chunks=max_evidence_chunks,
        )
        evidence_block = self.evidence_synthesizer.format_context_prompt(evidence_chunks)

        user_content = query
        if evidence_block:
            user_content = (
                f"{evidence_block}\n\n"
                f"### RESEARCH QUESTION / INSTRUCTION:\n{query}"
            )

        messages = [
            {"role": "system", "content": skill.system_prompt},
            {"role": "user", "content": user_content},
        ]

        kwargs: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": active_temp,
            "stream": True,
        }
        if api_key:
            kwargs["api_key"] = api_key

        try:
            response_stream = await self._invoke_acompletion(**kwargs)
            async for chunk in response_stream:
                delta = None
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    delta = None

                if delta:
                    yield delta
        except Exception as exc:
            raise EngineError(f"Streaming query failed with model '{active_model}': {exc}") from exc

    async def query(
        self,
        query: str,
        documents: list[str | Path | ParsedDocument] | None = None,
        skill_name: str = "academic_writer",
        model: str | None = None,
        temperature: float | None = None,
        max_evidence_chunks: int | None = None,
    ) -> SynthesisResult:
        """Execute query, returning complete synthesis result and supporting evidence chunks."""
        evidence_chunks = self.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=documents or [],
            max_chunks=max_evidence_chunks,
        )

        accumulated: list[str] = []
        skill = self.skills_loader.get_skill(skill_name)
        active_model = model or skill.model or self.config.default_model

        async for token in self.stream_query(
            query=query,
            documents=documents,
            skill_name=skill_name,
            model=model,
            temperature=temperature,
            max_evidence_chunks=max_evidence_chunks,
        ):
            accumulated.append(token)

        full_answer = "".join(accumulated)
        return SynthesisResult(
            answer=full_answer,
            evidence_chunks=evidence_chunks,
            model=active_model,
            skill=skill.name,
        )
