"""FastAPI application for scripcy local research studio with streaming and document tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripcy.core.config import ScripcyConfig
from scripcy.core.engine import LLMEngine
from scripcy.core.mcp_client import CitationItem, WordCraftClient
from scripcy.core.models import DocumentMetadata
from scripcy.core.parser import DocumentParser
from scripcy.core.search import PaperSearcher, SearchResult, sanitize_filename
from scripcy.core.skills import SkillsLoader
from scripcy.core.workspace import WorkspaceManager

STATIC_DIR = Path(__file__).parent / "static"


class ChatStreamRequest(BaseModel):
    """Payload for initiating a streaming academic research query."""

    query: str
    skill_name: str = "scripcy_manual"
    active_documents: list[str] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None


class GenerateDocumentSection(BaseModel):
    """A section of manuscript to append."""

    heading: str
    level: int = 1
    content: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)


class GenerateDocumentRequest(BaseModel):
    """Payload for generating an academic .docx manuscript."""

    title: str
    sections: list[GenerateDocumentSection] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    style: str = "apa"


def create_app(
    workspace: WorkspaceManager | None = None,
    config: ScripcyConfig | None = None,
    engine: LLMEngine | None = None,
    searcher: PaperSearcher | None = None,
    parser: DocumentParser | None = None,
    wordcraft: WordCraftClient | None = None,
) -> FastAPI:
    """Instantiate and configure the FastAPI server with scripcy dependencies."""
    app = FastAPI(
        title="scripcy",
        description="A local-first academic research studio backend",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws = workspace or WorkspaceManager()
    cfg = config or ScripcyConfig.load()
    skills = SkillsLoader(skills_dir=cfg.skills_dir)
    eng = engine or LLMEngine(config=cfg, skills_loader=skills)
    psr = parser or DocumentParser(workspace=ws)
    srch = searcher or PaperSearcher(workspace=ws)
    wc = wordcraft or WordCraftClient(workspace=ws)

    # Track active document selection in memory for quick retrieval
    active_doc_stems: set[str] = set()

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "app": "scripcy", "version": "0.1.0"}

    @app.get("/api/config")
    async def get_configuration() -> dict[str, Any]:
        return {
            "default_model": cfg.default_model,
            "skills_dir": str(cfg.skills_dir),
            "workspace_path": str(ws.root_dir),
            "citation": cfg.citation.model_dump(),
            "configured_providers": [k for k, v in cfg.api_keys.items() if v],
        }

    @app.get("/api/skills")
    async def list_available_skills() -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "model": s.model,
                "temperature": s.temperature,
            }
            for s in skills.list_skills()
        ]

    @app.get("/api/documents")
    async def list_documents() -> list[dict[str, Any]]:
        cached_paths = ws.list_cached_documents("*.md")
        results: list[dict[str, Any]] = []

        for p in cached_paths:
            try:
                content = p.read_text(encoding="utf-8")
                metadata, _ = DocumentMetadata.from_frontmatter(content)
                meta_dict = metadata.model_dump()
            except Exception:
                meta_dict = {"title": p.stem, "authors": [], "year": None, "doi": None, "cite_key": p.stem}

            stem = p.stem
            is_active = (stem in active_doc_stems) if active_doc_stems else True

            results.append({
                "filename": p.name,
                "stem": stem,
                "title": meta_dict.get("title") or p.stem,
                "authors": meta_dict.get("authors") or [],
                "year": meta_dict.get("year"),
                "doi": meta_dict.get("doi"),
                "cite_key": meta_dict.get("cite_key") or stem,
                "size_bytes": p.stat().st_size,
                "active": is_active,
            })
        return results

    @app.post("/api/documents/toggle")
    async def toggle_document_active(payload: dict[str, Any]) -> dict[str, Any]:
        stem = payload.get("stem")
        active = payload.get("active", True)
        if not stem:
            raise HTTPException(status_code=400, detail="Document stem is required.")
        if active:
            active_doc_stems.add(stem)
        else:
            active_doc_stems.discard(stem)
        return {"stem": stem, "active": stem in active_doc_stems}

    @app.post("/api/documents/upload")
    async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        ws.ensure_directories()
        dest_pdf = ws.vault_dir / file.filename
        content = await file.read()
        dest_pdf.write_bytes(content)

        try:
            parsed = psr.parse(dest_pdf, save_to_cache=True)
            active_doc_stems.add(dest_pdf.stem)
            return {
                "status": "success",
                "filename": file.filename,
                "cache_file": parsed.cache_path.name if parsed.cache_path else None,
                "metadata": parsed.metadata.model_dump(),
            }
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to ingest PDF: {exc}") from exc

    @app.get("/api/documents/{stem}")
    async def get_document_content(stem: str) -> dict[str, Any]:
        cache_file = ws.get_cache_path(f"{stem}.md")
        if not cache_file.is_file():
            raise HTTPException(status_code=404, detail="Document not found in cache.")

        content = cache_file.read_text(encoding="utf-8")
        try:
            metadata, body = DocumentMetadata.from_frontmatter(content)
            return {
                "stem": stem,
                "metadata": metadata.model_dump(),
                "body": body,
                "full_markdown": content,
            }
        except Exception:
            return {"stem": stem, "metadata": {}, "body": content, "full_markdown": content}

    @app.get("/api/search")
    async def search_papers(
        q: str = Query(..., min_length=1),
        limit: int = Query(10, ge=1, le=50),
        source: str = Query("auto"),
    ) -> list[dict[str, Any]]:
        try:
            results = await srch.search(q, limit=limit, source=source)  # type: ignore[arg-type]
            return [r.model_dump() for r in results]
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Search failed: {exc}") from exc

    @app.post("/api/search/download")
    async def download_search_result(paper_data: dict[str, Any]) -> dict[str, Any]:
        try:
            paper = SearchResult.model_validate(paper_data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid SearchResult payload: {exc}") from exc

        try:
            saved_pdf = await srch.download_paper(paper)
            parsed = psr.parse(saved_pdf, save_to_cache=True)
            active_doc_stems.add(saved_pdf.stem)
            return {
                "status": "success",
                "vault_path": str(saved_pdf),
                "cache_path": str(parsed.cache_path),
                "metadata": parsed.metadata.model_dump(),
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to download and parse paper: {exc}") from exc

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        active_files = request.active_documents
        if not active_files and active_doc_stems:
            active_files = [f"{s}.md" for s in active_doc_stems]

        doc_paths: list[Path] = []
        if active_files:
            for fname in active_files:
                name = fname if fname.endswith(".md") else f"{fname}.md"
                p = ws.get_cache_path(name)
                if p.is_file():
                    doc_paths.append(p)
        else:
            doc_paths = ws.list_cached_documents("*.md")

        async def sse_event_generator() -> AsyncGenerator[str, None]:
            try:
                # 1. Retrieve and send evidence context chunk event
                evidence_chunks = eng.evidence_synthesizer.retrieve_evidence(
                    query=request.query,
                    documents=doc_paths,
                )
                evidence_payload = [
                    {
                        "source": c.source,
                        "score": round(c.score, 2),
                        "snippet": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    }
                    for c in evidence_chunks
                ]
                yield f"event: evidence\ndata: {json.dumps(evidence_payload)}\n\n"

                # 2. Stream tokens from LLM
                stream = eng.stream_query(
                    query=request.query,
                    documents=doc_paths,
                    skill_name=request.skill_name,
                    model=request.model,
                    temperature=request.temperature,
                )
                async for delta in stream:
                    yield f"event: token\ndata: {json.dumps({'delta': delta})}\n\n"

                yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/document/generate")
    async def generate_manuscript(request: GenerateDocumentRequest) -> dict[str, Any]:
        dest_filename = f"{sanitize_filename(request.title)}.docx"
        target_path = ws.vault_dir / dest_filename

        try:
            wc.create_document(title=request.title, path=target_path)

            for sec in request.sections:
                wc.append_heading(sec.heading, level=sec.level)
                if sec.content:
                    citations = [CitationItem.from_any(c) for c in sec.citations]
                    wc.add_paragraph_with_citation(
                        text=sec.content,
                        citations=citations,
                        style=request.style,  # type: ignore[arg-type]
                    )

            if request.references:
                ref_items = [CitationItem.from_any(r) for r in request.references]
                wc.append_reference_list(ref_items, style=request.style)  # type: ignore[arg-type]

            saved = wc.save(target_path)
            return {
                "status": "success",
                "filename": dest_filename,
                "saved_path": str(saved),
                "download_url": f"/api/document/download/{dest_filename}",
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Manuscript generation failed: {exc}") from exc

    @app.get("/api/document/download/{filename}")
    async def download_manuscript(filename: str) -> FileResponse:
        file_path = ws.vault_dir / filename
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Requested manuscript file not found.")
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    @app.get("/api/manuscripts")
    async def list_manuscripts() -> list[dict[str, Any]]:
        if not ws.vault_dir.is_dir():
            return []
        files = sorted(ws.vault_dir.glob("*.docx"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "download_url": f"/api/document/download/{f.name}",
            }
            for f in files
        ]

    # Serve Single Page Application UI
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>scripcy Research Studio</h1><p>Frontend assets initializing...</p>")

    return app


app = create_app()
