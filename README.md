# scrpcy

A local-first academic research studio with autonomous literature scouting, multi-model AI synthesis, and publication-ready manuscript generation.

---

## Architecture & Modules

### Module 1: File Vault & Document Ingestion
Provides local-first ingestion and parsing of academic papers (PDF format):
- **Clean Markdown Conversion**: Extracts structure and body text via `pymupdf4llm` and `markitdown`.
- **Publication Metadata Extraction**: Extracts Title, Authors, Year, DOI, and Cite Keys using Gemini structured outputs (via Google GenAI SDK or LiteLLM).
- **Workspace Cache Management**: Stores parsed documents with YAML frontmatter headers for lightning-fast retrieval and caching.
- **Active Document Selection**: Selectively toggle papers into active context for focused literature review synthesis.

### Module 2: BYOK LLM Engine & Markdown Skills Loader
Flexible model orchestration and dynamic prompt management:
- **Local BYOK Configuration** (`~/.scrpcy/config.json`): Secure local storage of API keys, default models (e.g. `gemini/gemini-2.5-pro`, `openai/gpt-4o`, `ollama/llama3`), and citation preferences.
- **Dynamic Markdown Skills Loader** (`./skills/`): Load and modify system prompts (e.g. `academic_writer.md`, `literature_synthesizer.md`, `methodology_critic.md`) with YAML frontmatter and live hot-reloading.
- **PaperQA-Style Evidence Synthesis**: Filters and ranks document chunks using BM25-inspired lexical relevance and context token budgeting to minimize token usage.
- **Real-Time Streaming & Telemetry**: Full async streaming (`acompletion`) with token consumption telemetry (input, output, and total tokens).

### Module 3: Paper Search Fallback & PDF Retrieval
Global academic paper search and direct open-access PDF downloading:
- **Multi-API Fallback Search**: Queries Semantic Scholar Graph API with automatic fallback to ArXiv Atom API on rate limits (HTTP 429), server errors, or timeouts.
- **Clean Structured Metadata**: Standardizes Title, Authors, Abstract, Year, DOI, and direct PDF download links via `SearchResult`.
- **Workspace PDF Ingestion**: Downloads approved papers via streaming HTTP into the local workspace vault directory for direct ingestion by Module 1.

### Module 4: WordCraft MCP Client & Document Generator
Produces publication-ready `.docx` academic manuscripts:
- **Model Context Protocol (MCP) Integration**: Dispatches document authoring commands via JSON-RPC stdio to `wordcraft-cli mcp`.
- **Transparent Python-Docx Fallback**: Automatic, resilient error boundary falling back to direct `python-docx` generation if the MCP server is unreachable or fails.
- **Academic Citation Formatting**: Full formatting support for APA 7th edition (`(Vaswani et al., 2017)`) and IEEE (`[1]`) inline citations and bibliographies.
- **Standard Tool Interface**: Exposes `create_document`, `append_heading`, `add_paragraph_with_citation`, and `append_reference_list`.

### Module 5: Autonomous Literature Scout & Multi-Source Synthesis
Autonomous academic research assistant that discovers and curates relevant scholarly works:
- **Multi-Source Scraping**: Queries OpenAlex and Semantic Scholar/ArXiv in parallel to gather comprehensive literature candidates.
- **Natural Language Strategy Formulation**: Automatically extracts keywords, publication year ranges, and open-access constraints from free-form user descriptions.
- **AI-Driven Relevance Ranking**: Ranks candidates from highest relevance to lowest without artificial cutoffs, appending seminal groundbreaking literature recommendations.
- **Zero-Scroll Paginated Carousel**: Horizontal slide navigation (`←` / `→`) with explicit slide indicators (`[ Slide X/Y ]`) and cross-slide multi-paper selection (`download > 1, 4`).

### Module 6: Google Drive Vault Ingestion (Public Share & OAuth 2.0)
Seamless ingestion of remote research documents from Google Drive:
- **Public & Shared Links**: Ingests files directly via Google Drive shareable URLs or bare File IDs.
- **OAuth 2.0 Desktop Integration**: Secure login flow with personal `credentials.json` for private Google Drive access.
- **Automatic Conversion & Vault Sync**: Streams files to the project vault, converts them to clean Markdown via `MarkItDown`, and updates the active evidence index.
- **Non-Blocking Background Processing**: All Google Drive downloads run as background tasks in `TaskManager` to ensure the studio and chat never freeze.

### Module 7: Background Task Processing & Task Manager
Asynchronous execution engine keeping Scrpcy responsive during heavy operations:
- **ThreadPool Execution Queue**: Offloads literature scouting, PDF downloads, Google Drive ingests, and `.docx` compilations to background worker threads.
- **Live Timestamped Telemetry**: Detailed logs recorded per-task (`[HH:MM:SS] Progress message...`).
- **Dedicated Sub-Mode (`/processing`)**: View concise status summaries (running, completed, failed, cancelled) or inspect full task tables and logs.
- **Task Cancellation**: Gracefully cancel running background jobs via `/proc cancel <task_id>`.

### Module 8: Antigravity-Style Terminal Studio & UX Architecture
High-craft terminal user interface following Google Antigravity design principles:
- **Duotone Minimalist Palette**: Clean borderless interface using a single Cyan accent (`#06B6D4`) with neutral dark and bright white text (`#F3F4F6`).
- **Outlier Input Framing**: Borderless horizontal framing rules with live `/` slash command autocompletion.
- **Sub-Mode Isolation**: Contextual prompts for Project Root (`skripsi>`), Vault (`skripsi/vault>`), Processing (`skripsi/processing>`), and Chat Hub (`skripsi/chat>`).
- **Interactive Chat Hub & Multi-Session History**: Manage distinct SQLite chat sessions (`new`, `open`, `delete`, `list`) with beautiful LaTeX math rendering and academic Markdown formatting.
- **Dynamic Token Budget Warnings**: Automatically warns and prompts user confirmation when queries exceed high token thresholds (> 5 evidence chunks).

---

## Scrpcy CLI & Terminal Studio

### Starting Scrpcy Studio

```bash
# Launch interactive Antigravity CLI Studio
scrpcy

# Launch backend server and open the 3-panel Web Studio in browser
scrpcy start

# Start Web Studio on custom port without auto-opening browser
scrpcy start --port 8080 --dir ~/my_research_vault --no-browser
```

### Models & API Key Management

```bash
scrpcy models list                           # List configured models, API keys, 9Router status
scrpcy models use openai/gpt-4o              # Switch active default model
scrpcy models set-key openai YOUR_API_KEY   # Store API key per-provider (gemini, openai, anthropic, deepseek)
scrpcy models 9router                        # Interactive 1-step 9Router local gateway setup
scrpcy models 9router --port 20128 --key sk-9r --model 9router/claude-3-5-sonnet
scrpcy models tokens                         # Inspect global & per-project token consumption
```

### Project & Sub-Mode Navigation

Inside the Scrpcy Studio, navigate effortlessly between modes:

| Slash Command | Context | Deskripsi |
| :--- | :--- | :--- |
| `/vault` atau `/v` | Project Root | Masuk ke sub-mode perpustakaan berkas (`skripsi/vault>`) |
| `/processing` atau `/proc` | Project Root | Masuk ke sub-mode pemantauan antrean tugas background |
| `/chat` atau `/c` | Project Root | Masuk ke menu hub percakapan riset (`skripsi/chat>`) |
| `/scout <query>` | Project Root | Menjalankan pencarian & kurasi literatur otomatis |
| `/findpaper <topik>` | Project / Chat | Menjalankan pencarian paper di background via OpenAlex |
| `/gdrive <url>` | Project / Vault / Chat | Mengunduh berkas Google Drive langsung ke vault |
| `/docx [nama.docx]` | Project / Chat | Mengompilasi naskah Word lengkap dengan format sitasi |
| `/status` | Project / Chat | Melihat model aktif, level effort, dan konsumsi token project |
| `/skill` | Project / Chat | Membuka selektor interaktif persona asisten AI |
| `/back` | Sub-Mode / Chat | Kembali satu tingkat ke konteks sebelumnya |
| `?` | Semua Mode | Membuka daftar perintah interaktif yang tersedia |

---

## Quickstart (Python API)

### 1. Document Parsing & Vault Ingestion
```python
from scrpcy import DocumentParser

parser = DocumentParser()
result = parser.parse("path/to/paper.pdf")
print("Title:", result.metadata.title)
print("Cite Key:", result.metadata.cite_key)
print("Markdown Cache:", result.cache_path)
```

### 2. Evidence-Based Chat Synthesis & Streaming
```python
import asyncio
from scrpcy import LLMEngine

async def main():
    engine = LLMEngine()
    async for token in engine.stream_query(
        query="Bagaimana perbandingan efektivitas model Transformer vs CNN?",
        documents=["path/to/cached_paper.md"],
        skill_name="academic_writer",
    ):
        print(token, end="", flush=True)

asyncio.run(main())
```

### 3. Autonomous Literature Scouting
```python
import asyncio
from scrpcy.core.scout import LiteratureScout, ScoutQuery

async def main():
    scout = LiteratureScout()
    query = ScoutQuery(keywords="Transformer attention mechanism", year_start=2021)
    
    # Scrape multi-source across OpenAlex and Semantic Scholar
    candidates = await scout.search_multi_source(query, limit=20)
    
    # Curate and rank by relevance
    ranked = scout.curate_and_rank("Attention mechanisms", candidates)
    for idx, paper in enumerate(ranked[:5], 1):
        print(f"{idx}. {paper.title} ({paper.year}) - PDF: {paper.has_pdf}")

asyncio.run(main())
```

### 4. Compiling Word Manuscripts (.docx) via WordCraft
```python
from scrpcy import WordCraftClient, DocumentMetadata

with WordCraftClient() as client:
    client.create_document("Emerging Deep Learning Trends")
    client.append_heading("1. Attention Mechanisms", level=1)
    
    meta = DocumentMetadata(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        doi="10.48550/arXiv.1706.03762",
    )
    
    # Formatted APA citation: (Vaswani & Shazeer, 2017)
    client.add_paragraph_with_citation(
        "Transformers eliminate recurrence entirely.",
        citations=[meta],
        style="apa",
    )
    
    # Appends formatted references section
    client.append_reference_list([meta], style="apa")
    saved_path = client.save()
    print("Manuscript saved to:", saved_path)
```

---

## Testing

Run the full automated test suite:

```bash
pytest -v
```
