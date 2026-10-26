"""Multi-Thread Cross-Citation Snowballing and Convergence Scoring Engine.

Implements cross-thread consensus mapping:
1. Thread Decomposition: Decomposes user topic into orthogonal research angles.
2. Seed Retrieval: Discovers top seed papers for each thread.
3. Citation Snowballing: Traverses 1-hop backward references and forward citations.
4. Convergence Scoring: Identifies cross-cutting hub papers using thread overlap,
   semantic relevance (pruning out-of-domain methodological noise), and recency decay.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import json
import logging
import math
import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import LLMEngine
from scripcy.core.search import PaperSearcher, SearchResult, sanitize_filename

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from",
    "of", "is", "are", "was", "were", "be", "been", "this", "that", "it", "as", "we",
    "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "adalah", "ini", "itu", "yang",
    "studi", "analisis", "penelitian", "paper", "jurnal", "towards", "using", "based",
}


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric keywords excluding stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def token_cosine_similarity(text1: str, text2: str) -> float:
    """Calculate token frequency cosine similarity between two texts."""
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    if not tokens1 or not tokens2:
        return 0.0

    c1 = Counter(tokens1)
    c2 = Counter(tokens2)

    all_words = set(c1.keys()).union(c2.keys())
    dot = sum(c1.get(w, 0) * c2.get(w, 0) for w in all_words)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def calculate_thread_overlap(citing_threads: Sequence[str], total_threads: int) -> float:
    """Calculate proportion of independent threads citing a candidate paper."""
    if total_threads <= 0:
        return 0.0
    unique_count = len(set(citing_threads))
    return min(1.0, max(0.0, unique_count / total_threads))


def calculate_recency_score(
    year: int | None,
    current_year: int = 2026,
    decay_rate: float = 0.05,
) -> float:
    """Calculate exponential recency decay score: exp(-lambda * delta_years)."""
    if year is None:
        delta = 10.0
    else:
        delta = max(0.0, float(current_year - year))
    return math.exp(-decay_rate * delta)


def calculate_semantic_relevance(query: str, candidate_text: str) -> float:
    """Compute semantic relevance between query context and candidate paper text.

    Tries dense embeddings via shared FastEmbed instance if available, calibrated to prune
    anisotropic baseline noise (< 0.55). Falls back to token cosine similarity.
    """
    if not query.strip() or not candidate_text.strip():
        return 0.0

    # Attempt FastEmbed dense embedding cosine similarity with calibration
    try:
        from scripcy.core.vector_vault import get_shared_embedder
        embedder = get_shared_embedder()
        embs = list(embedder.embed([query, candidate_text]))
        if len(embs) == 2:
            v1, v2 = embs[0], embs[1]
            dot = sum(a * b for a, b in zip(v1, v2))
            n1 = math.sqrt(sum(a * a for a in v1))
            n2 = math.sqrt(sum(b * b for b in v2))
            if n1 > 0 and n2 > 0:
                raw_dense = dot / (n1 * n2)
                # Calibrate: baseline noise is ~0.55, substantive match > 0.65
                calibrated_dense = max(0.0, min(1.0, (raw_dense - 0.55) / 0.40))
                return round(calibrated_dense, 4)
    except Exception:
        pass

    # Token cosine fallback
    tok_sim = token_cosine_similarity(query, candidate_text)
    return round(tok_sim, 4)


class SeedThread(BaseModel):
    """An orthogonal research perspective / thread with representative seed papers."""

    thread_name: str = Field(description="Name or perspective of the research thread.")
    query: str = Field(description="Targeted search query for this thread.")
    seed_papers: list[SearchResult] = Field(default_factory=list, description="Seed papers retrieved for this thread.")


class ConvergenceResult(BaseModel):
    """A cross-cutting hub paper discovered and ranked through multi-thread citation convergence."""

    paper_id: str = Field(description="Identifier (DOI, OpenAlex ID, or S2 PaperId).")
    title: str = Field(description="Title of the candidate hub paper.")
    year: int | None = Field(default=None, description="Publication year.")
    doi: str | None = Field(default=None, description="Digital Object Identifier.")
    abstract: str | None = Field(default=None, description="Abstract summary.")
    overlap_count: int = Field(description="Number of independent threads citing this paper.")
    citing_threads: list[str] = Field(default_factory=list, description="Names of threads that cite this paper.")
    final_score: float = Field(description="Calculated multi-factor convergence score (0.0 to 1.0).")
    reasoning: str = Field(description="Explanation of why this paper was selected as a convergence hub.")
    thread_overlap: float = Field(default=0.0, description="Thread overlap factor (0.0 - 1.0).")
    semantic_relevance: float = Field(default=0.0, description="Semantic similarity to user query (0.0 - 1.0).")
    recency_score: float = Field(default=0.0, description="Recency decay score (0.0 - 1.0).")
    pdf_url: str | None = Field(default=None, description="Direct URL to open-access PDF if available.")
    has_pdf: bool = Field(default=False, description="Whether open-access PDF is available.")
    source: str = Field(default="citation_graph", description="Discovery origin badge.")


class CitationSnowballer:
    """Orchestrates multi-thread citation snowballing and convergence scoring."""

    def __init__(
        self,
        engine: LLMEngine | None = None,
        searcher: PaperSearcher | None = None,
        config: ScripcyConfig | None = None,
        min_relevance: float = 0.15,
        current_year: int = 2026,
        decay_rate: float = 0.05,
    ) -> None:
        self.engine = engine
        self.config = config or getattr(self.engine, "config", None) or ScripcyConfig()
        s2_key = self.config.get_api_key("semantic_scholar")
        self.searcher = searcher or PaperSearcher(api_key=s2_key)
        self.min_relevance = min_relevance
        self.current_year = current_year
        self.decay_rate = decay_rate

    async def decompose_query_into_threads(self, user_query: str) -> list[SeedThread]:
        """Decompose complex user research query into 3 distinct, orthogonal research angles."""
        clean_q = user_query.strip()
        if not clean_q:
            return []

        # Try structured LLM decomposition if engine is configured
        if self.engine:
            model = self.engine.config.default_model
            key = self.engine.config.get_api_key_for_model(model)
            if key and not key.startswith("mock"):
                try:
                    return await asyncio.to_thread(self._decompose_via_llm, clean_q)
                except Exception as exc:
                    logger.debug(f"LLM thread decomposition fallback: {exc}")

        # Deterministic heuristic decomposition
        return [
            SeedThread(
                thread_name="Theoretical / Foundational",
                query=f"{clean_q} theoretical foundations principles",
            ),
            SeedThread(
                thread_name="Applied / Architectural",
                query=f"{clean_q} architecture implementation models application",
            ),
            SeedThread(
                thread_name="Evaluation / Empirical",
                query=f"{clean_q} evaluation benchmarks dataset empirical analysis",
            ),
        ]

    def _decompose_via_llm(self, query: str) -> list[SeedThread]:
        """Use LLM to generate 3 orthogonal perspectives as JSON."""
        import litellm

        system_prompt = (
            "You are a Principal Scientific Research Architect. Break down the user's research topic into "
            "EXACTLY 3 diverse, orthogonal research angles (threads):\n"
            "1. Theoretical / Foundational (core mathematical, theoretical, or scientific principles)\n"
            "2. Applied / Architectural (concrete models, engineering, algorithms, or systems)\n"
            "3. Evaluation / Empirical (benchmarks, datasets, empirical studies, or critical challenges)\n\n"
            "Return valid JSON array of 3 objects with keys 'thread_name' and 'query'."
        )

        model = self.engine.config.default_model
        key = self.engine.config.get_api_key_for_model(model)
        effective_model = model
        api_base = None
        if model.startswith("9router/"):
            effective_model = f"openai/{model.split('/', 1)[1]}"
            api_base = "http://localhost:20128/v1"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research Topic: {query}"},
        ]
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        if key:
            kwargs["api_key"] = key
        if api_base:
            kwargs["api_base"] = api_base

        resp = litellm.completion(**kwargs)
        raw = resp.choices[0].message.content or "[]"
        clean = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw).strip()
        data = json.loads(clean)
        items = data.get("threads") or data if isinstance(data, list) else []
        if isinstance(data, dict) and not items:
            for v in data.values():
                if isinstance(v, list):
                    items = v
                    break

        threads: list[SeedThread] = []
        for item in items[:3]:
            t_name = item.get("thread_name") or "Research Perspective"
            t_query = item.get("query") or query
            threads.append(SeedThread(thread_name=t_name, query=t_query))

        if len(threads) < 3:
            # Fallback fill
            return [
                SeedThread(thread_name="Theoretical / Foundational", query=f"{query} theoretical foundations"),
                SeedThread(thread_name="Applied / Architectural", query=f"{query} architecture models"),
                SeedThread(thread_name="Evaluation / Empirical", query=f"{query} evaluation benchmarks"),
            ]
        return threads

    async def retrieve_seed_papers(
        self,
        threads: list[SeedThread],
        papers_per_thread: int = 2,
    ) -> list[SeedThread]:
        """Retrieve top representative seed papers for each thread."""
        for thread in threads:
            try:
                results = await self.searcher.search(thread.query, limit=papers_per_thread)
                thread.seed_papers = results
            except Exception as exc:
                logger.warning(f"Error fetching seeds for thread '{thread.thread_name}': {exc}")
                thread.seed_papers = []
        return threads

    async def snowball_and_score(
        self,
        user_query: str,
        threads: list[SeedThread],
        include_forward: bool = True,
        max_hub_papers: int = 5,
    ) -> list[ConvergenceResult]:
        """Execute 1-hop cross-citation snowballing and multi-factor convergence scoring.

        Calculates:
        FinalScore(P) = (ThreadOverlap * 0.50) + (SemanticRelevance * 0.35) + (RecencyScore * 0.15)
        Prunes papers below min_relevance.
        """
        if not threads:
            return []

        total_threads = len(threads)
        candidate_papers: dict[str, SearchResult] = {}
        candidate_threads: dict[str, set[str]] = {}

        def _canonical_key(paper: SearchResult) -> str:
            if paper.doi:
                clean_doi = paper.doi.strip().lower().replace("https://doi.org/", "")
                return f"doi:{clean_doi}"
            if paper.paper_id:
                return f"id:{paper.paper_id.strip()}"
            clean_title = re.sub(r"[^\w]", "", (paper.title or "").lower())
            return f"title:{clean_title}"

        # 1. Harvest references (backward) and citations (forward) across all seed papers
        for thread in threads:
            for seed in thread.seed_papers:
                seed_id = seed.doi or seed.paper_id or seed.title
                if not seed_id:
                    continue

                # Backward snowballing (references cited by seed)
                try:
                    refs = await self.searcher.get_paper_references(seed_id, limit=20)
                except Exception:
                    refs = []

                # Forward snowballing (papers citing seed)
                cites = []
                if include_forward:
                    try:
                        cites = await self.searcher.get_paper_citations(seed_id, limit=20)
                    except Exception:
                        cites = []

                combined_neighbors = refs + cites
                for cand in combined_neighbors:
                    key = _canonical_key(cand)
                    if not key or key == "title:":
                        continue

                    if key not in candidate_papers:
                        candidate_papers[key] = cand
                        candidate_threads[key] = set()

                    candidate_threads[key].add(thread.thread_name)

        if not candidate_papers:
            return []

        # 2. Batch fetch metadata if candidates lack full abstracts/titles
        missing_metadata_ids = [
            p.paper_id for p in candidate_papers.values()
            if not p.abstract and p.paper_id and not p.paper_id.startswith("mock")
        ]
        if missing_metadata_ids:
            try:
                batch_results = await self.searcher.batch_get_papers_semantic_scholar(missing_metadata_ids[:100])
                for b_paper in batch_results:
                    b_key = _canonical_key(b_paper)
                    if b_key in candidate_papers and b_paper.abstract:
                        candidate_papers[b_key].abstract = b_paper.abstract
                        if b_paper.year and not candidate_papers[b_key].year:
                            candidate_papers[b_key].year = b_paper.year
            except Exception:
                pass

        # 3. Contextual Query for Semantic Relevance
        thread_queries = " ".join(t.query for t in threads)
        full_query_context = f"{user_query} {thread_queries}".strip()

        # 4. Compute Convergence Scores
        convergence_list: list[ConvergenceResult] = []

        for key, paper in candidate_papers.items():
            citing = candidate_threads.get(key, set())
            overlap_count = len(citing)
            thread_overlap = calculate_thread_overlap(list(citing), total_threads)

            text_for_sim = f"{paper.title or ''} {paper.abstract or ''}".strip()
            semantic_relevance = calculate_semantic_relevance(full_query_context, text_for_sim)

            # Prune out-of-domain noise (e.g. ImageNet, PyTorch, general Adam optimizer)
            if semantic_relevance < self.min_relevance:
                continue

            recency_score = calculate_recency_score(
                paper.year,
                current_year=self.current_year,
                decay_rate=self.decay_rate,
            )

            # Final Score Formula: (Overlap * 0.50) + (Relevance * 0.35) + (Recency * 0.15)
            final_score = round(
                (thread_overlap * 0.50) + (semantic_relevance * 0.35) + (recency_score * 0.15),
                4,
            )

            citing_sorted = sorted(list(citing))
            reasoning = (
                f"Cited across {overlap_count}/{total_threads} independent research angles: "
                f"[{', '.join(citing_sorted)}]. Semantic relevance: {semantic_relevance:.0%}, "
                f"year: {paper.year or 'n.d.'}."
            )

            c_res = ConvergenceResult(
                paper_id=paper.paper_id or paper.doi or key,
                title=paper.title,
                year=paper.year,
                doi=paper.doi,
                abstract=paper.abstract,
                overlap_count=overlap_count,
                citing_threads=citing_sorted,
                final_score=final_score,
                reasoning=reasoning,
                thread_overlap=round(thread_overlap, 4),
                semantic_relevance=round(semantic_relevance, 4),
                recency_score=round(recency_score, 4),
                pdf_url=paper.pdf_url,
                has_pdf=paper.has_pdf,
            )
            convergence_list.append(c_res)

        # 5. Rank by FinalScore descending
        convergence_list.sort(key=lambda item: item.final_score, reverse=True)
        return convergence_list[:max_hub_papers]

    async def discover_hub_papers(
        self,
        user_query: str,
        papers_per_thread: int = 2,
        include_forward: bool = True,
        max_hub_papers: int = 5,
    ) -> list[ConvergenceResult]:
        """Convenience method running the full end-to-end multi-thread snowballing workflow."""
        threads = await self.decompose_query_into_threads(user_query)
        threads = await self.retrieve_seed_papers(threads, papers_per_thread=papers_per_thread)
        return await self.snowball_and_score(
            user_query=user_query,
            threads=threads,
            include_forward=include_forward,
            max_hub_papers=max_hub_papers,
        )
