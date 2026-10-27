"""LLM orchestration engine with PaperQA-style evidence synthesis and streaming support."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
import logging
import math
from pathlib import Path
import re
from typing import Any, Callable

# litellm is imported lazily in _invoke_acompletion to avoid startup lag
from pydantic import BaseModel, Field

from scripcy.core.config import ScripcyConfig
from scripcy.core.crusher import MarkdownMinifier
from scripcy.core.models import ParsedDocument
from scripcy.core.skills import Skill, SkillsLoader
from scripcy.core.tracker import TokenTracker
from scripcy.exceptions import EngineError

logger = logging.getLogger("scripcy.core.engine")

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
    page_num: int = Field(default=1, description="Page number where evidence appears.")
    parent_text: str = Field(default="", description="Surrounding parent context block.")
    quote_snippet: str = Field(default="", description="Verbatim quote snippet for fine-grained citation.")


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

    def expand_query(self, query: str) -> list[str]:
        """Generate academic synonyms and English technical terminology expansions for cross-lingual hit rate."""
        clean = query.strip()
        expansions = [clean]
        lower_q = clean.lower()

        # Indonesian -> English Technical / Academic Mapping
        translations = [
            (r"\b(kebakaran\s+hutan|karhutla)\b", "wildfire forest fire hotspot detection"),
            (r"\b(cuaca|iklim)\b", "weather climate meteorology"),
            (r"\b(akurasi|ketelitian)\b", "accuracy precision benchmark evaluation"),
            (r"\b(metode|algoritma)\b", "method algorithm model architecture"),
            (r"\b(overfit|overfitting)\b", "regularization dropout weight decay generalization"),
            (r"\b(klasifikasi)\b", "classification categorical cross entropy"),
            (r"\b(prediksi|ramalan)\b", "prediction forecasting estimation"),
            (r"\b(kesehatan\s+mental)\b", "mental health psychological well-being depression"),
            (r"\b(citra|gambar|visual)\b", "image visual computer vision segmentation"),
        ]

        addon_terms = []
        for pat, trans in translations:
            if re.search(pat, lower_q):
                addon_terms.append(trans)

        if addon_terms:
            expansions.append(" ".join(addon_terms))
            # Synthesize compact HyDE-like query representation
            expansions.append(f"{clean} {' '.join(addon_terms)}")

        return expansions
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

    def compute_coverage_score(self, query: str, evidence: list[EvidenceChunk]) -> float:
        """Compute semantic/lexical coverage of query intent within retrieved evidence (0.0 to 1.0)."""
        if not evidence:
            return 0.0

        query_terms = [t for t in tokenize_query(query) if len(t) >= 3]
        if not query_terms:
            return 1.0

        combined_text = " ".join(c.text.lower() for c in evidence)
        matched_terms = 0
        for term in query_terms:
            if term in combined_text or (len(term) >= 4 and any(w.startswith(term[:4]) for w in combined_text.split())):
                matched_terms += 1

        coverage = matched_terms / len(query_terms)
        return round(coverage, 2)

    def retrieve_evidence(
        self,
        query: str,
        documents: list[str | Path | ParsedDocument],
        max_chunks: int | None = None,
        token_budget: int | None = None,
        vector_vault: Any = None,
    ) -> list[EvidenceChunk]:
        """Decompose documents, rank passages against query, and select within token limits.

        Supports both targeted query retrieval and corpus-wide multi-document balanced synthesis.
        """
        clean_q = query.strip().lower()
        corpus_patterns = [
            r"\b(rangkum|ringkas|sintesis|telaah|analisis|jelaskan|daftar|sebutkan|review)\s+(semua|seluruh|isi|setiap|koleksi)\b",
            r"\b(semua|seluruh|setiap)\s+(paper|jurnal|dokumen|artikel|literatur|file|berkas|topik|kajian)\b",
            r"\b(isi|koleksi|daftar)\s+(vault|proyek|project|katalog)\b",
            r"\b(summarize|synthesize|overview|review|list|analyze)\s+(all|every|the whole|entire)\b",
            r"\b(all|every|entire)\s+(papers?|documents?|journals?|articles?|vault|files?)\b",
            r"\bwhat\s+(papers?|documents?)\s+(are in|in the)\s+vault\b",
            r"\bapa\s+saja\s+(paper|jurnal|dokumen|isi|artikel)\b",
            r"\bada\s+paper\s+apa\s+saja\b",
        ]
        is_corpus_query = any(re.search(pat, clean_q) for pat in corpus_patterns)

        num_docs = len(documents)
        if is_corpus_query and num_docs > 0:
            limit_chunks = max(max_chunks or self.default_max_chunks, num_docs * 2)
            limit_tokens = max(token_budget or self.default_token_budget, num_docs * 1200)
        else:
            limit_chunks = max_chunks or self.default_max_chunks
            limit_tokens = token_budget or self.default_token_budget

        # 1. High-Performance Hybrid Search with Multi-Query Expansion & Reranking
        if vector_vault is not None:
            try:
                expanded_queries = self.expand_query(query)
                all_hybrid_matches: dict[str, Any] = {}
                for eq in expanded_queries:
                    res_q = vector_vault.hybrid_search_rrf(eq, top_k=limit_chunks)
                    for r in res_q:
                        if r.chunk_id not in all_hybrid_matches:
                            all_hybrid_matches[r.chunk_id] = r

                hybrid_res = list(all_hybrid_matches.values())
                if hybrid_res:
                    active_stems = set()
                    for d in documents:
                        if isinstance(d, Path):
                            active_stems.add(d.stem)
                        elif hasattr(d, "stem"):
                            active_stems.add(d.stem)
                        elif hasattr(d, "metadata") and getattr(d.metadata, "cite_key", None):
                            active_stems.add(d.metadata.cite_key)
                        elif isinstance(d, str):
                            active_stems.add(Path(d).stem)

                    matched_chunks = []
                    # Track seen parents to merge and prevent token explosion
                    seen_parents: set[str] = set()

                    for h in hybrid_res:
                        if active_stems and h.paper_id not in active_stems:
                            continue
                        
                        # Parent-child hierarchical selection: use parent context if not already covered
                        parent_key = f"{h.paper_id}_p{h.page_num}_{h.parent_content[:60]}"
                        if h.parent_content and parent_key not in seen_parents:
                            chunk_text = h.parent_content
                            seen_parents.add(parent_key)
                        else:
                            chunk_text = h.content

                        if h.tables:
                            for tbl in h.tables:
                                md_p = tbl.get("md_path")
                                tbl_id = tbl.get("id", "tbl")
                                if md_p and Path(md_p).exists():
                                    try:
                                        t_content = Path(md_p).read_text(encoding="utf-8")
                                        chunk_text += f"\n\n[[VISUAL:{tbl_id}]] [Tabel Terindeks: {tbl.get('caption', 'Tabel')}]:\n{t_content}"
                                    except Exception:
                                        pass
                        if h.images:
                            for img in h.images:
                                ai_desc = img.get("ai_analysis")
                                img_id = img.get("id", "img")
                                if ai_desc and "JUNK_IMAGE" not in ai_desc:
                                    chunk_text += f"\n\n[[VISUAL:{img_id}]] [Analisis Visual Gambar Halaman {h.page_num}: {img.get('caption', 'Gambar')}]:\n{ai_desc}"

                        # Extract first complete sentence as verbatim quote snippet
                        quote_snip = ""
                        sentences = re.split(r"(?<=[.!?])\s+", h.content.strip())
                        if sentences:
                            quote_snip = sentences[0][:140]

                        matched_chunks.append(EvidenceChunk(
                            chunk_id=h.chunk_id,
                            source=h.paper_id,
                            text=chunk_text,
                            token_estimate=estimate_tokens(chunk_text),
                            score=h.rerank_score if h.rerank_score is not None else h.rrf_score,
                            page_num=h.page_num,
                            parent_text=h.parent_content,
                            quote_snippet=quote_snip,
                        ))
                    if matched_chunks:
                        return matched_chunks[:limit_chunks]
            except Exception as v_err:
                logger.debug(f"Hybrid search fallback to lexical chunking: {v_err}")

        query_terms = tokenize_query(query)

        all_chunks: list[EvidenceChunk] = []
        doc_chunks_by_source: dict[str, list[EvidenceChunk]] = {}

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
            if doc_chunks:
                all_chunks.extend(doc_chunks)
                doc_chunks_by_source[source_name] = doc_chunks

        if not all_chunks:
            return []

        # Score chunks
        is_factual_q = any(
            w in clean_q for w in (
                "data", "angka", "berapa", "statistik", "jumlah", "luas", "tahun",
                "tabel", "bps", "kementerian", "resmi", "pemerintah", "nasional", "hektar", "ha"
            )
        )
        for chunk in all_chunks:
            chunk.score = self.score_chunk(query_terms, query, chunk.text)
            src_lower = chunk.source.lower()
            text_lower = chunk.text.lower()
            has_official_signal = (
                src_lower.startswith("web_")
                or "go_id" in src_lower
                or "gov" in src_lower
                or "bps" in src_lower
                or "menlhk" in src_lower
                or "klhk" in text_lower
                or "bps" in text_lower
                or ".go.id" in text_lower
                or "kementerian" in text_lower
                or "sipongi" in text_lower
            )
            if is_factual_q and has_official_signal:
                chunk.score = chunk.score * 1.6 + 2.0

            if is_corpus_query:
                # In corpus summary mode, chunk 0 (title/abstract/intro) is crucial for every document
                if chunk.chunk_id.endswith("_c0"):
                    chunk.score = max(chunk.score, 2.5)
                else:
                    chunk.score = max(chunk.score, 0.5)

        # Sort chunks within each document by score descending
        for src in doc_chunks_by_source:
            doc_chunks_by_source[src].sort(key=lambda c: c.score, reverse=True)

        selected: list[EvidenceChunk] = []
        selected_ids: set[str] = set()
        used_tokens = 0

        # Sort document sources by highest chunk score so official/most relevant sources are represented first
        sorted_sources = sorted(
            doc_chunks_by_source.keys(),
            key=lambda s: doc_chunks_by_source[s][0].score if doc_chunks_by_source[s] else 0.0,
            reverse=True,
        )

        # Phase 1: Stratified round-robin across documents to guarantee fair multi-doc representation
        max_rounds = 2 if is_corpus_query else 1
        for round_idx in range(max_rounds):
            for src in sorted_sources:
                doc_chunks = doc_chunks_by_source[src]
                if len(selected) >= limit_chunks:
                    break
                if round_idx < len(doc_chunks):
                    candidate = doc_chunks[round_idx]
                    if candidate.chunk_id in selected_ids:
                        continue
                    if candidate.score <= 0.0 and not is_corpus_query:
                        continue
                    if used_tokens + candidate.token_estimate > limit_tokens:
                        if not selected:
                            max_words = max(10, int(limit_tokens / 1.35))
                            truncated_words = candidate.text.split()[:max_words]
                            candidate.text = " ".join(truncated_words) + "..."
                            candidate.token_estimate = estimate_tokens(candidate.text)
                            selected.append(candidate)
                            selected_ids.add(candidate.chunk_id)
                            used_tokens += candidate.token_estimate
                        continue

                    selected.append(candidate)
                    selected_ids.add(candidate.chunk_id)
                    used_tokens += candidate.token_estimate

        # Phase 2: Fill remaining capacity with globally highest scoring chunks
        remaining_candidates = [
            c for c in all_chunks
            if c.chunk_id not in selected_ids and (c.score > 0.0 or is_corpus_query)
        ]
        remaining_candidates.sort(key=lambda c: c.score, reverse=True)

        for candidate in remaining_candidates:
            if len(selected) >= limit_chunks:
                break
            if used_tokens + candidate.token_estimate > limit_tokens:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)
            used_tokens += candidate.token_estimate

        selected.sort(key=lambda c: c.score, reverse=True)
        return selected

    def format_context_prompt(self, chunks: list[EvidenceChunk]) -> str:
        """Render evidence passages into a clean prompt context block with MarkdownMinifier compression."""
        if not chunks:
            return ""

        formatted_blocks = ["### RETRIEVED RESEARCH EVIDENCE CONTEXT:"]
        total_orig = sum(len(c.text) for c in chunks)
        total_compact = 0

        for idx, chunk in enumerate(chunks, 1):
            compact_text = MarkdownMinifier.minify(chunk.text)
            total_compact += len(compact_text)
            page_info = f", Halaman {chunk.page_num}" if chunk.page_num else ""
            quote_info = f"\nKutipan Kunci: \"{chunk.quote_snippet}\"" if chunk.quote_snippet else ""
            block = (
                f"[Evidence {idx}] (Source ID: {chunk.source}{page_info}, Score: {chunk.score:.2f}):{quote_info}\n"
                f"\"\"\"\n{compact_text}\n\"\"\""
            )
            formatted_blocks.append(block)

        # Footnote Citation Guidance
        formatted_blocks.append(
            "[PETUNJUK CITATION FOOTNOTE DIRECT MAPPING]:\n"
            "- Gunakan sitasi bernomor footnote [^1], [^2], dst. setiap kali Anda menyatakan fakta penting.\n"
            "- Nomor footnote harus berkorespondensi dengan nomor [Evidence 1], [Evidence 2] di atas.\n"
            "- Di akhir jawaban, Scripcy akan otomatis menyusun daftar rujukan kutipan dan link berkas halaman PDF."
        )

        has_visuals = any("[[VISUAL:" in c.text for c in chunks)
        if has_visuals:
            formatted_blocks.append(
                "[PETUNJUK TATA LETAK VISUAL]:\n"
                "- Jika analisis Anda merujuk pada tabel atau gambar di atas yang memuat tag [[VISUAL:id]], "
                "letakkan tag [[VISUAL:id]] tersebut pada baris baru tersendiri sebagai blok visual mandiri.\n"
                "- DILARANG menyisipkan tag [[VISUAL:id]] di tengah kalimat. Tulis kalimat pengantar di atasnya, "
                "letakkan tag [[VISUAL:id]] di baris baru, lalu berikan penjelasan atau analisis temuan visual tersebut di paragraf persis di bawahnya."
            )

        if total_orig > 0:
            saved_pct = ((total_orig - total_compact) / total_orig) * 100.0
            logger.debug(f"[SMARTCRUSHER] Minified evidence chunks: {total_orig}->{total_compact} chars ({saved_pct:.1f}% saved)")

        return "\n\n".join(formatted_blocks)


class LLMEngine:
    """Orchestrates LiteLLM streaming calls, skill persona injection, and evidence retrieval."""

    def __init__(
        self,
        config: ScripcyConfig | None = None,
        skills_loader: SkillsLoader | None = None,
        acompletion_fn: Callable[..., Any] | None = None,
        token_tracker: TokenTracker | None = None,
        project_name: str | None = None,
        project_path: str | Path | None = None,
    ) -> None:
        self.config = config or ScripcyConfig.load()
        self.skills_loader = skills_loader or SkillsLoader(skills_dir=self.config.skills_dir)
        self.evidence_synthesizer = EvidenceSynthesizer(
            default_max_chunks=self.config.citation.max_evidence_chunks,
            default_token_budget=self.config.citation.context_token_budget,
        )
        self._acompletion_fn = acompletion_fn
        self.token_tracker = token_tracker or TokenTracker()
        self.project_name = project_name
        self.project_path = str(Path(project_path).expanduser().resolve()) if project_path else None

    async def _invoke_acompletion(self, **kwargs: Any) -> Any:
        """Invoke litellm.acompletion or custom injected callable."""
        if self._acompletion_fn is not None:
            res = self._acompletion_fn(**kwargs)
            if hasattr(res, "__await__"):
                res = await res
            return res
        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.drop_params = True
        return await litellm.acompletion(**kwargs)

    async def stream_query(
        self,
        query: str,
        documents: list[str | Path | ParsedDocument] | None = None,
        skill_name: str = "scripcy_manual",
        model: str | None = None,
        temperature: float | None = None,
        max_evidence_chunks: int | None = None,
        project_name: str | None = None,
        project_path: str | Path | None = None,
        history: list[dict[str, str]] | None = None,
        vector_vault: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response tokens from LiteLLM using synthesized evidence."""
        if not query or not query.strip():
            raise EngineError("Query string cannot be empty.")

        if skill_name and skill_name != "scripcy_manual":
            try:
                skill = self.skills_loader.get_skill(skill_name)
                active_model = model or skill.model or self.config.default_model
                active_temp = temperature if temperature is not None else (skill.temperature if skill.temperature is not None else 0.2)
                sys_prompt = skill.system_prompt
            except Exception:
                from scripcy.core.skills import DEFAULT_SCRIPCY_MANUAL_PROMPT
                active_model = model or self.config.default_model
                active_temp = temperature if temperature is not None else 0.2
                sys_prompt = DEFAULT_SCRIPCY_MANUAL_PROMPT
        else:
            from scripcy.core.skills import DEFAULT_SCRIPCY_MANUAL_PROMPT
            active_model = model or self.config.default_model
            active_temp = temperature if temperature is not None else 0.2
            sys_prompt = DEFAULT_SCRIPCY_MANUAL_PROMPT
        api_key = self.config.get_api_key_for_model(active_model)
        api_base = self.config.get_api_base_for_model(active_model)

        # 9Router routing translation
        effective_model = active_model
        if active_model.lower().startswith("9router/"):
            submodel = active_model.split("/", 1)[1]
            effective_model = f"openai/{submodel}"
            api_base = api_base or "http://localhost:20128/v1"

        evidence_chunks = self.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=documents or [],
            max_chunks=max_evidence_chunks,
            vector_vault=vector_vault,
        )
        evidence_block = self.evidence_synthesizer.format_context_prompt(evidence_chunks)

        user_content = query
        if evidence_block:
            user_content = (
                f"{evidence_block}\n\n"
                f"### RESEARCH QUESTION / INSTRUCTION:\n{query}"
            )

        # Inspect if any COMPLEX visual is referenced in evidence chunks for Direct Multimodal Injection
        multimodal_image_payloads: list[dict[str, Any]] = []
        if vector_vault is not None and evidence_chunks:
            complex_vid_matches = []
            for ech in evidence_chunks:
                for m in re.finditer(r"\[\[VISUAL:([^:\]]+):COMPLEX\]\]", ech.text):
                    complex_vid_matches.append(m.group(1).strip())

            # De-duplicate while preserving retrieval order
            seen_vids = set()
            for vid in complex_vid_matches:
                if vid in seen_vids or len(multimodal_image_payloads) >= 2:
                    continue
                seen_vids.add(vid)
                try:
                    with vector_vault._get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT file_path FROM images WHERE id = ? LIMIT 1", (vid,))
                        row = cur.fetchone()
                        if row and row["file_path"]:
                            f_path = Path(row["file_path"])
                            if f_path.exists():
                                img_bytes = f_path.read_bytes()
                                ext = f_path.suffix.lstrip(".").lower()
                                if ext in ("jpg", "jpeg"):
                                    mime_type = "image/jpeg"
                                elif ext == "webp":
                                    mime_type = "image/webp"
                                else:
                                    mime_type = "image/png"
                                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                                multimodal_image_payloads.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{b64_str}"},
                                })
                except Exception as img_err:
                    logger.debug(f"Failed preparing multimodal image payload for {vid}: {img_err}")

        if "### SCRPCY ENVIRONMENT & SEAMLESS ACTION PROTOCOL:" not in sys_prompt:
            from scripcy.core.skills import SCRIPCY_AGENTIC_PROTOCOL
            sys_prompt = f"{sys_prompt}\n\n{SCRIPCY_AGENTIC_PROTOCOL}"

        messages = [
            {"role": "system", "content": sys_prompt},
        ]
        if history:
            messages.extend(history)

        # Formulate final user message (text-only or multimodal composite payload)
        if multimodal_image_payloads:
            user_msg_content: Any = [{"type": "text", "text": user_content}]
            user_msg_content.extend(multimodal_image_payloads)
        else:
            user_msg_content = user_content

        messages.append({"role": "user", "content": user_msg_content})

        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": active_temp,
            "stream": True,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        max_output_tokens = getattr(self.config, "max_output_tokens", 8192)
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens

        accumulated_output: list[str] = []
        input_tokens_est = estimate_tokens(user_content + sys_prompt)

        try:
            response_stream = await self._invoke_acompletion(**kwargs)
            async for chunk in response_stream:
                delta = None
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    delta = None

                if delta:
                    accumulated_output.append(delta)
                    yield delta
        except Exception as exc:
            err_msg = str(exc)
            if "9router" in active_model.lower() and ("AuthenticationError" in type(exc).__name__ or "invalid_api_key" in str(exc).lower()):
                err_msg += (
                    "\n[Hint] 9Router authentication failed. Set your 9Router API key using: "
                    "'models set-key 9router <key>' (or check http://localhost:20128)"
                )
            raise EngineError(f"Streaming query failed with model '{active_model}': {err_msg}") from exc
        finally:
            # Track tokens even if partially streamed or completed
            output_tokens_est = estimate_tokens("".join(accumulated_output))
            active_proj_name = project_name or self.project_name
            active_proj_path = project_path or self.project_path
            self.token_tracker.record_usage(
                model=active_model,
                input_tokens=input_tokens_est,
                output_tokens=output_tokens_est,
                project_name=active_proj_name,
                project_path=active_proj_path,
                skill=skill_name,
            )

    async def query(
        self,
        query: str,
        documents: list[str | Path | ParsedDocument] | None = None,
        skill_name: str = "scripcy_manual",
        model: str | None = None,
        temperature: float | None = None,
        max_evidence_chunks: int | None = None,
        project_name: str | None = None,
        project_path: str | Path | None = None,
        history: list[dict[str, str]] | None = None,
        vector_vault: Any = None,
    ) -> SynthesisResult:
        """Execute query, returning complete synthesis result and supporting evidence chunks."""
        evidence_chunks = self.evidence_synthesizer.retrieve_evidence(
            query=query,
            documents=documents or [],
            max_chunks=max_evidence_chunks,
            vector_vault=vector_vault,
        )

        accumulated: list[str] = []
        active_model = model or self.config.default_model
        resolved_skill_name = skill_name
        if skill_name and skill_name != "scripcy_manual":
            try:
                skill = self.skills_loader.get_skill(skill_name)
                active_model = model or skill.model or self.config.default_model
                resolved_skill_name = skill.name
            except Exception:
                pass

        async for token in self.stream_query(
            query=query,
            documents=documents,
            skill_name=skill_name,
            model=model,
            temperature=temperature,
            max_evidence_chunks=max_evidence_chunks,
            project_name=project_name or self.project_name,
            project_path=project_path or self.project_path,
            vector_vault=vector_vault,
        ):
            accumulated.append(token)

        full_answer = "".join(accumulated)
        return SynthesisResult(
            answer=full_answer,
            evidence_chunks=evidence_chunks,
            model=active_model,
            skill=resolved_skill_name,
        )
