# scrpcy

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Local--First](https://img.shields.io/badge/architecture-local--first-emerald.svg)](#introduction)
[![Audit: Two--Tier%20Contract%20Verification](https://img.shields.io/badge/verification-two--tier%20audit-violet.svg)](#key-features)

A local-first, autonomous academic research studio that bridges scholarly literature repositories, sovereign institutional portals, and multi-model AI synthesis into publication-grade manuscripts with verifiable epistemic rigor.

---

## Introduction

Modern Large Language Models (LLMs) often struggle with scholarly research: they hallucinate citations, generate superficial prose ("AI slop"), become overwhelmed by generic token matching, and prioritize high-level homepages over empirical data.

**Scripcy** is engineered to eliminate these vulnerabilities. Operating on a **local-first paradigm**, Scripcy stores all document vaults, caches, vector embeddings, chat sessions, and SQLite state machines directly inside isolated project directories on your machine (`projects/<name>/`).

Rather than relying on generic conversational drafting, Scripcy decomposes research queries into bounded **Small Contracts** executed by isolated Workers. Evidence is audited at the retrieval stage by a Devil's Advocate Critic, ensuring that final synthesis proceeds in a deterministic **One-Pass Golden Draft** verified by an empirical FactChecker.

### Core Design Principles
1. **Zero Hallucination Tolerance**: Every claim must be grounded in local vault documents. If empirical evidence is absent, the system explicitly declares an *Honest Data Gap Note* rather than fabricating statistics.
2. **Small Contracts & Worker Isolation**: Complex inquiries are decomposed into isolated sub-contracts. Workers operate within bounded token and document scopes without cross-worker context contamination, feeding structured findings into a shared project RAG vault.
3. **Two-Tier Epistemic Audit**: Evidence sufficiency is audited by Devil's Advocate at the Worker retrieval stage (Tier 2). Once evidence is cleared, final synthesis is generated in a single pass (*One-Pass Golden Drafting*) audited solely by FactChecker for grounding integrity.
4. **Autonomous Dual-Channel Scouting**: Searches global scholarly literature (OpenAlex, Semantic Scholar, CrossRef, Unpaywall) alongside sovereign government and institutional portals (`.go.id`, `.gov`, `.who.int`) when local vault evidence is insufficient.
5. **Transient Session Lifecycle**: Sessions remain active only when an interactive user is present or bound background tasks are executing. Exiting an idle session automatically transitions it to `closed`, eliminating zombie processes while preserving SQLite chat history.

---

## Key Features

### 1. Small Contracts & Isolated Worker Choreography
Scripcy departs from monolithic prompt engineering by employing an autonomous contract decomposition pattern:
- **Contractor Decomposition**: Deconstructs user research goals into distinct, typed sub-contracts (`DATA_FACTUAL`, `COMPARATIVE`, `EVALUATION_SYNTHESIS`).
- **Specialized Workers**: Deploys targeted workers (`VaultExtractor`, `GapExtractor`, `DialecticalExtractor`) with restricted context windows. Workers extract verified claim-level fact sheets into the shared project database without hallucination spillover.
- **Dynamic Follow-Up**: If initial evidence is deemed insufficient during post-scout audit, the Contractor selectively grants a bounded follow-up expansion to target missing empirical parameters.

### 2. Two-Tier Epistemic Audit & One-Pass Golden Drafting
To prevent infinite debate loops and redundant token burn:
- **Tier 1 (Planning Conformance)**: Contractor and Devil's Advocate review sub-contract boundaries and exclusion criteria prior to execution.
- **Tier 2 (Post-Scout Evidence Audit)**: Devil's Advocate evaluates candidate findings directly after Worker retrieval. If gaps are irremediable, an *Honest Data Gap Note* is registered.
- **One-Pass Golden Drafting**: With evidence verified in Tier 2, the final synthesis is generated deterministically in a single pass. FactChecker verifies claim grounding against vault chunks, omitting late-stage re-drafting cycles and reducing execution latency by over 50%.

### 3. Local-First Dense + Sparse Hybrid Search (RRF)
Scripcy embeds a zero-dependency hybrid retrieval engine directly inside each project's SQLite database (`vault/project_vault.db`):
- **Lexical Sparse Search (FTS5 BM25)**: Evaluates exact keyword matches, domain acronyms, author names, and model designations via SQLite FTS5 with Okapi BM25 scoring.
- **Semantic Dense Search (`sqlite-vec`)**: Computes cosine distance over dense vector embeddings (`text-embedding-004` or compatible local embedding models) natively in SQLite via SIMD C-extensions.
- **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense rankings using the Cormack, Clarke, and Büttcher algorithm ($k=60$):
  $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

### 4. Sovereign Web & Literature Scout with Relevance Gating
When local evidence is insufficient, Scripcy initiates autonomous discovery:
- **Literature Scout**: Explores 250M+ scholarly works across OpenAlex, Semantic Scholar, and CrossRef, resolving open-access PDFs, abstracts, and citation counts.
- **Web Scout**: Discovers sovereign government domains (`.go.id`, `.gov`, `.mil`), accredited academic domains (`.ac.id`, `.edu`), and intergovernmental portals (`who.int`, `ipcc.ch`).
- **Lexical Anchor Gate & Root Rejection**: Candidate search results must contain substantive topic entities in snippets and URL paths; shallow landing pages lacking article paths are rejected to prevent institutional homepage bias.

### 5. Multi-Thread Citation Snowballing
Adhering to systematic literature review methodology (Wohlin, 2014):
- **Backward Snowballing**: Parses reference lists of seed vault papers to capture foundational antecedents.
- **Forward Snowballing**: Traverses the Semantic Scholar Graph API to identify emerging studies citing vault papers.
- **Convergence Scoring**: Ranks candidate literature by cross-citation density across multiple independent discovery threads.

### 6. Transient Automatic Session Lifecycle
Scripcy manages conversation state machines without manual session termination:
- **Active Focus**: A session remains in `running` state while the user interacts within the terminal interface.
- **Detached Processing**: Exiting a session while background ingestion or literature scouting tasks are active marks the session as `detached`, allowing tasks to stream results into the session inbox.
- **Automatic Reaper**: When all bound background tasks complete, or when a session is exited with zero active tasks, Scripcy automatically marks it `closed`. Historical messages, artifacts, and token metrics remain fully intact in SQLite and can be resumed at any time.

### 7. Structured Visual Evidence & Publication Compilation
- **3-Line Visual Blocks**: Extracted figures, charts, and maps are presented with active OSC 8 clickable terminal hyperlinks (`file:///`), descriptive captions, and formal APA in-text source citations.
- **Publication Compilation**: Verified manuscripts can be compiled into structured Microsoft Word (`.docx`) documents with styled headings and integrated bibliographies via the Model Context Protocol (MCP) or native document formatters.

---

## References & Acknowledgements

The architecture of Scripcy has been informed and inspired by foundational research in information retrieval, hybrid search algorithms, autonomous multi-agent systems, and academic synthesis:

### Retrieval & Hybrid Search Algorithms
| Reference / Algorithm | Origin / Publication | Architectural Contribution to Scripcy |
| :--- | :--- | :--- |
| **[Reciprocal Rank Fusion (RRF)](https://dl.acm.org/doi/10.1145/1571941.1572114)** | Cormack, Clarke, and Büttcher (SIGIR 2009) | **Dense + Sparse Hybrid Aggregation**: Powers Scripcy's local hybrid retrieval in `VectorVault`, merging sparse FTS5 BM25 lexical rankings with dense cosine similarity vectors via $\sum \frac{1}{k + r_m(d)}$. |
| **[sqlite-vec](https://github.com/asg017/sqlite-vec)** | Alex Garcia | **In-Process Local Vector Search**: Provides zero-dependency, C-based SIMD vector search running natively inside per-project SQLite databases (`project_vault.db`) without external vector DB infrastructure. |
| **[Okapi BM25 (SQLite FTS5)](https://www.sqlite.org/fts5.html)** | Robertson & Zaragoza (2009) | **Lexical Relevance Scoring**: Powers Scripcy's exact lexical indexing for domain jargon, paper titles, author names, and visual captions across project vaults. |
| **[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** | RUC AI Box | **Cascaded Waterfall Relevance Gating**: Provided the mathematical foundation for Scripcy's gated search scoring, ensuring domain authority cannot override low-relevance results ($\text{FinalScore} = \text{Relevance} \times [1 + 0.5 \times \text{Authority}]$). |
| **[MindSearch](https://github.com/InternLM/MindSearch)** | InternLM | **Lexical Anchor Pre-Fetch Check & Root Landing Page Rejection**: Inspired Scripcy's pre-fetch gate verifying core entity occurrences in snippets before scraping, and the automatic rejection of shallow root homepages. |
| **[Perplexica](https://github.com/ItzCrazyK/Perplexica)** | ItzCrazyK | **Boolean Search Restructuring**: Informed Scripcy's search operator formulation and multi-engine fallback strategies across Bing, Yahoo, and DuckDuckGo. |

### Autonomous Multi-Agent Systems & Literature Discovery
| Project / Reference | Origin / Organization | Architectural Contribution to Scripcy |
| :--- | :--- | :--- |
| **[Stanford STORM](https://github.com/stanford-oval/storm)** | Stanford OVAL | **Perspective-Guided Querying & Meta-Word Stripping**: Informed Scripcy's `QueryDisambiguator`, removing conversational tokens and meta-words (`data`, `overview`) in favor of empirical metric expansion and two-phase portal discovery. |
| **[GPT-Researcher](https://github.com/assafelovic/gpt-researcher)** | Assaf Elovic | **Sub-Query Decomposition & Negative Filtering**: Informed Scripcy's query reformulation, decomposing complex user questions into targeted sub-queries with negative exclusion operators. |
| **[Citation Snowballing Guidelines](https://dl.acm.org/doi/10.1145/2601248.2601268)** | Claes Wohlin (EASE 2014) | **Systematic Citation Graph Traversal**: Governs Scripcy's 1-hop backward reference parsing and forward citation harvesting in `CitationSnowballer` to identify seminal cross-cutting hub literature. |
| **[OpenAlex](https://openalex.org/)** | Priem, Piwowar, & Orr (2022) | **Open Bibliometric Federation**: Primary metadata source for Scripcy's Literature Scout, indexing 250M+ global scholarly works, authors, institutions, and open-access PDF links. |
| **[Semantic Scholar Graph API](https://www.semanticscholar.org/product/api)** | Allen Institute for AI (Kinney et al., 2023) | **Citation Graph & TLDR Federation**: Powers citation count mapping, paper influence scoring, and forward citation tracking across literature pipelines. |
| **[Unpaywall](https://unpaywall.org/)** | OurResearch | **Automated Open-Access Harvesting**: Resolves legitimate open-access PDF endpoints directly from scholarly DOIs. |
| **[PaperQA / PaperQA2](https://github.com/Future-House/paper-qa)** | FutureHouse | **Evidence Synthesis & Fact-Grounding**: Informed Scripcy's chunk-level evidence ranking, context compression, and automated citation mapping to prevent LLM hallucinations. |
| **[Microsoft AutoGen](https://github.com/microsoft/autogen)** | Microsoft Research | **Multi-Agent Conversational Choreography**: Guided the isolated subagent architecture separating `Contractor`, `VaultExtractor`, `GapExtractor`, `FactChecker`, and `DevilsAdvocate`. |

### UI Craft & Ingestion Infrastructure
| Project / Reference | Origin / Organization | Architectural Contribution to Scripcy |
| :--- | :--- | :--- |
| **[Google Antigravity](https://deepmind.google/)** | Google DeepMind | **Terminal Studio Craft & UX Principles**: Informed Scripcy's duotone minimalist palette, outlier input framing, background task telemetry, and structured artifact generation. |
| **[PyMuPDF4LLM](https://github.com/pymupdf/pymupdf4llm)** | Artifex Software | **High-Fidelity Document Ingestion**: Powers Scripcy's local PDF parsing, converting academic tables and layout columns into clean Markdown. |
| **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** | Anthropic | **Extensible Tool Interoperability**: Utilized in Scripcy's `WordCraftClient` for dispatching manuscript authoring commands to WordCraft. |

---

## License

Scripcy is distributed under the [MIT License](LICENSE).
