# scrpcy

A local-first academic research studio.

## Module 1: File Vault & Document Ingestion

Module 1 provides local-first ingestion of academic papers (PDF format), clean Markdown conversion via `pymupdf4llm`, publication metadata extraction using Gemini structured outputs (via Google GenAI SDK or LiteLLM), and workspace cache management with YAML frontmatter.

## Module 2: BYOK LLM Engine & Markdown Skills Loader

Module 2 provides flexible model orchestration and dynamic prompt management:
- **Local BYOK Configuration** (`~/.scrpcy/config.json`): Secure local storage of API keys, default models (e.g. `gemini/gemini-2.5-pro`, `openai/gpt-4o`, `ollama/llama3`), and citation preferences.
- **Dynamic Markdown Skills Loader** (`./skills/`): Load and modify system prompts (e.g. `academic_writer.md`, `literature_synthesizer.md`) with YAML frontmatter and hot-reloading without restarting.
- **PaperQA-Style Evidence Synthesis**: Filters and ranks document chunks using BM25-inspired lexical relevance and context token budgeting to minimize token usage.
- **LiteLLM Streaming Orchestration**: Full async streaming (`acompletion`) and non-streaming synthesis.

## Module 3: Paper Search Fallback & PDF Retrieval

Module 3 adds academic paper search and direct open-access PDF downloading:
- **Multi-API Fallback Search**: Queries Semantic Scholar Graph API with automatic fallback to ArXiv Atom API on rate limits (HTTP 429), server errors, or timeouts.
- **Clean Structured Metadata**: Standardizes Title, Authors, Abstract, Year, DOI, and direct PDF download links via [`SearchResult`](file:///c:/Users/mfthn/OneDrive/Documents/Pemograman/Ai/scrpcy/scrpcy/core/search.py#L32).
- **Workspace PDF Ingestion**: Downloads approved papers via streaming HTTP into the local workspace vault directory for direct ingestion by Module 1.

## Module 4: WordCraft MCP Client & Document Generator

Module 4 produces publication-ready `.docx` academic manuscripts:
- **Model Context Protocol (MCP) Integration**: Dispatches document authoring commands via JSON-RPC stdio to `wordcraft-cli mcp`.
- **Transparent Python-Docx Fallback**: Automatic, resilient error boundary falling back to direct `python-docx` generation if the MCP server is unreachable or fails.
- **Academic Citation Formatting**: Full formatting support for APA 7th edition (`(Vaswani et al., 2017)`) and IEEE (`[1]`) inline citations and bibliographies.
- **Standard Tool Interface**: Exposes `create_document`, `append_heading`, `add_paragraph_with_citation`, and `append_reference_list`.

---

## CLI & 3-Panel Research Studio (FastAPI + Typer)

### CLI Commands

The `scrpcy` command-line executable manages configuration and starts the local studio:

```bash
# Start backend server and automatically open the 3-panel studio in your default browser
scrpcy start

# Start on custom port and directory without auto-opening browser
scrpcy start --port 8080 --dir ~/my_research_vault --no-browser

# Set API keys locally (stored in ~/.scrpcy/config.json)
scrpcy config set-key gemini YOUR_GEMINI_API_KEY
scrpcy config set-key openai YOUR_OPENAI_API_KEY
scrpcy config set-key anthropic YOUR_ANTHROPIC_API_KEY

# Set default model (supports Gemini, OpenAI, Claude, Mistral, Groq, Ollama)
scrpcy config set-model gemini/gemini-2.5-pro

# Inspect current configuration
scrpcy config show

# Ingest and parse a local PDF from the command line
scrpcy parse path/to/paper.pdf

# Search global academic literature from the terminal
scrpcy search "Transformer Attention Mechanism" --limit 5
```

---

## 3-Panel Academic Studio Frontend Architecture

The studio provides a high-efficiency 3-panel workspace:

1. **Left Panel — Sources & Literature**:
   - **Vault / Library Tab**: Lists all ingested papers with title, authors, year, cite key badge, file size, and active checkboxes to select which papers are provided as context to the LLM. Includes an upload button for instant PDF parsing.
   - **Web Search Tab**: Search across Semantic Scholar and ArXiv with 1-click **"Download PDF"** buttons that fetch open-access preprints directly into your local vault.
2. **Center Panel — Academic Co-Pilot**:
   - Multi-turn conversation interface with streaming responses via Server-Sent Events (SSE).
   - Rendered inline citation pills (`[Vaswani2017]`) and an expandable **Retrieved Evidence Drawer** displaying exact quotes and relevance scores.
   - Dynamic skill persona switcher (`academic_writer`, `literature_synthesizer`) and citation style selector (`APA 7th`, `IEEE`).
3. **Right Panel — Studio & Artifacts**:
   - **Live Agent Tool Log**: Real-time telemetry log displaying timestamped agent events (`EvidenceSynthesizer`, `LLMEngine`, `WordCraft`).
   - **Manuscript Export**: 1-click **"Draft & Export to Word"** compiling your synthesized literature review into a formatted `.docx` manuscript with centered title, section headings, and formatted reference lists.

### Frontend Asset Building
The frontend is built as a zero-build, responsive Single Page Application (SPA) using Tailwind CSS and ES modules located at [`scrpcy/static/index.html`](file:///c:/Users/mfthn/OneDrive/Documents/Pemograman/Ai/scrpcy/scrpcy/static/index.html). It is served directly by the FastAPI backend on `http://127.0.0.1:8000/`. No Node.js or `npm install` steps are required.

---

### Running Tests

Run the complete test suite across all modules, server endpoints, and CLI commands:

```bash
pytest -v
```


### Quickstart

#### Parsing and Ingestion (Module 1)
```python
from scrpcy import DocumentParser

parser = DocumentParser()
result = parser.parse("path/to/paper.pdf")
print(result.metadata.title)
print(result.cache_path)
```

#### Evidence Synthesis & Streaming (Module 2)
```python
import asyncio
from scrpcy import LLMEngine

async def main():
    engine = LLMEngine()
    async for token in engine.stream_query(
        query="What attention mechanisms are used in Transformer models?",
        documents=["path/to/cached_paper.md"],
        skill_name="academic_writer",
    ):
        print(token, end="", flush=True)

asyncio.run(main())
```

#### Paper Search & PDF Retrieval (Module 3)
```python
import asyncio
from scrpcy import PaperSearcher

async def main():
    async with PaperSearcher() as searcher:
        # Search across Semantic Scholar with ArXiv fallback
        results = await searcher.search("Transformer Attention", limit=3)
        for paper in results:
            print(paper.title, paper.year, paper.has_pdf)
        
        # Download first paper with open-access PDF into workspace vault
        if results and results[0].has_pdf:
            pdf_path = await searcher.download_paper(results[0])
            print("Downloaded to:", pdf_path)

asyncio.run(main())
```

#### Document Generation & MCP Client (Module 4)
```python
from scrpcy import WordCraftClient, DocumentMetadata

with WordCraftClient() as client:
    # Creates document with centered title
    client.create_document("Emerging Deep Learning Trends")
    client.append_heading("1. Attention Mechanisms", level=1)
    
    meta = DocumentMetadata(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        doi="10.48550/arXiv.1706.03762",
    )
    
    # Adds paragraph with formatted APA citation: (Vaswani & Shazeer, 2017)
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

### Running Tests

```bash
pytest
```

