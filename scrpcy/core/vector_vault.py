"""Local Per-Project Hybrid Vector Vault (sqlite-vec + FTS5 BM25 + PyMuPDF Multimodal)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import sqlite_vec
from fastembed import TextEmbedding
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global singleton embedder instance to prevent reload overhead
_GLOBAL_EMBEDDER: TextEmbedding | None = None


def get_shared_embedder() -> TextEmbedding:
    """Return shared FastEmbed BGE-small instance (dim: 384, ~130MB, CPU ONNX)."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _GLOBAL_EMBEDDER


def serialize_f32(vector: list[float]) -> bytes:
    """Serialize float32 vector into raw binary buffer for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


@dataclass
class HybridSearchResult:
    """Result item returned by RRF hybrid search."""

    chunk_id: str
    paper_id: str
    page_num: int
    content: str
    rrf_score: float
    bm25_rank: int | None = None
    vec_rank: int | None = None
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


class ProjectVectorVault:
    """Per-project SQLite database combining FTS5 lexical BM25 and sqlite-vec semantic search."""

    def __init__(self, db_path: Path, vault_dir: Path) -> None:
        self.db_path = db_path
        self.vault_dir = vault_dir
        self.papers_dir = vault_dir / "papers"
        self.images_dir = vault_dir / "images"
        self.tables_dir = vault_dir / "tables"

        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with sqlite-vec extension loaded."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _init_db(self) -> None:
        """Initialize database schema with relational tables, FTS5, and sqlite-vec virtual table."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Papers metadata table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
            """)

            # FTS5 full-text search table for BM25
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)

            # sqlite-vec virtual table (384 dimensions for bge-small)
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_rowid INTEGER PRIMARY KEY,
                    embedding float[384] distance_metric=cosine
                )
            """)

            # Filtered informative images table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    caption TEXT,
                    ai_analysis TEXT,
                    width INTEGER,
                    height INTEGER,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
            """)

            # Structured tables table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    csv_path TEXT,
                    md_path TEXT,
                    caption TEXT,
                    is_vision_fallback INTEGER DEFAULT 0,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def is_paper_indexed(self, paper_id: str) -> bool:
        """Check if a paper has already been indexed in this project vault."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM papers WHERE id = ?", (paper_id,))
            return cur.fetchone() is not None

    def ingest_paper_pdf(
        self,
        pdf_path: Path,
        paper_id: str,
        title: str,
        engine: Any = None,
        model_name: str | None = None,
    ) -> dict[str, int]:
        """Process academic PDF with PyMuPDF, extract informative images/tables, chunk and embed into SQLite."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        paper_images_dir = self.images_dir / paper_id
        paper_tables_dir = self.tables_dir / paper_id
        paper_images_dir.mkdir(parents=True, exist_ok=True)
        paper_tables_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        embedder = get_shared_embedder()

        raw_chunks: list[dict[str, Any]] = []
        raw_tables: list[dict[str, Any]] = []
        raw_images: list[dict[str, Any]] = []

        chunk_counter = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect_page = page.rect

            # 1. Text Extraction & Chunking (250 words per chunk with 40-word overlap)
            page_text = page.get_text("text").strip()
            if page_text:
                words = page_text.split()
                step = max(1, 250 - 40)
                for start in range(0, len(words), step):
                    chunk_words = words[start : start + 250]
                    chunk_str = " ".join(chunk_words)
                    if len(chunk_str.strip()) >= 20:
                        raw_chunks.append({
                            "id": f"{paper_id}_p{page_num}_c{chunk_counter}",
                            "paper_id": paper_id,
                            "page_num": page_num,
                            "chunk_index": chunk_counter,
                            "content": chunk_str,
                        })
                        chunk_counter += 1
                    if start + 250 >= len(words):
                        break

            # 2. Structured Table Extraction via PyMuPDF find_tables()
            try:
                tables = page.find_tables()
                for t_idx, tab in enumerate(tables):
                    df_rows = tab.extract()
                    if not df_rows or len(df_rows) < 2:
                        continue

                    total_cells = sum(len(r) for r in df_rows)
                    empty_cells = sum(1 for r in df_rows for c in r if not c or not str(c).strip())
                    empty_ratio = empty_cells / max(1, total_cells)

                    # Fallback check: empty ratio > 35% or ragged rows
                    needs_vision = empty_ratio > 0.35 or any(len(r) != len(df_rows[0]) for r in df_rows)

                    tbl_stem = f"tbl_p{page_num}_{t_idx + 1}"
                    csv_path = paper_tables_dir / f"{tbl_stem}.csv"
                    md_path = paper_tables_dir / f"{tbl_stem}.md"

                    if not needs_vision:
                        # Write clean Markdown and CSV
                        headers = [str(c or "").strip() for c in df_rows[0]]
                        md_lines = ["| " + " | ".join(headers) + " |"]
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        csv_lines = [",".join(f'"{c}"' for c in headers)]

                        for row in df_rows[1:]:
                            clean_cells = [str(c or "").replace("\n", " ").strip() for c in row]
                            md_lines.append("| " + " | ".join(clean_cells) + " |")
                            csv_lines.append(",".join(f'"{c}"' for c in clean_cells))

                        md_path.write_text("\n".join(md_lines), encoding="utf-8")
                        csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
                        raw_tables.append({
                            "id": f"{paper_id}_{tbl_stem}",
                            "paper_id": paper_id,
                            "page_num": page_num,
                            "csv_path": str(csv_path),
                            "md_path": str(md_path),
                            "caption": f"Tabel Halaman {page_num} #{t_idx + 1}",
                            "is_vision_fallback": 0,
                        })
                    else:
                        # Crop table bbox to image
                        try:
                            pix = page.get_pixmap(clip=tab.bbox, dpi=200)
                            crop_img_path = paper_tables_dir / f"{tbl_stem}.png"
                            pix.save(str(crop_img_path))

                            # Transcribe via Multimodal Vision API if engine available
                            vision_md = self._analyze_vision_table(crop_img_path, engine, model_name)
                            if vision_md:
                                md_path.write_text(vision_md, encoding="utf-8")

                            raw_tables.append({
                                "id": f"{paper_id}_{tbl_stem}",
                                "paper_id": paper_id,
                                "page_num": page_num,
                                "csv_path": str(crop_img_path),
                                "md_path": str(md_path) if vision_md else None,
                                "caption": f"Tabel Kompleks Halaman {page_num} #{t_idx + 1} (Vision Parse)",
                                "is_vision_fallback": 1,
                            })
                        except Exception as crop_err:
                            logger.warning(f"Failed cropping table on p{page_num}: {crop_err}")
            except Exception as tab_err:
                logger.debug(f"Table detection skipped on p{page_num}: {tab_err}")

            # 3. Informative Image Extraction with Junk Filter
            try:
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    width, height = base_image["width"], base_image["height"]

                    # Heuristic Rule 1: Min dimensions (>= 150x150)
                    if width < 150 or height < 150:
                        continue

                    # Heuristic Rule 2: Aspect ratio (0.2 to 5.0 - eliminate long border lines)
                    ratio = width / max(1, height)
                    if ratio < 0.2 or ratio > 5.0:
                        continue

                    # Heuristic Rule 3: Page boundary exclusion (skip top 6% header logos and bottom 6% copyright footers)
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = img_rects[0]
                        if bbox.y1 < rect_page.height * 0.06 or bbox.y0 > rect_page.height * 0.94:
                            continue

                    # Candidate image passed heuristics! Save temporary and verify with multimodal API
                    img_ext = base_image["ext"]
                    img_stem = f"fig_p{page_num}_{img_idx + 1}"
                    img_file = paper_images_dir / f"{img_stem}.{img_ext}"
                    img_file.write_bytes(base_image["image"])

                    # Multimodal Verification: discard if AI identifies it as decorative logo/watermark
                    analysis = self._analyze_vision_figure(img_file, engine, model_name)
                    if analysis is None or "JUNK_IMAGE" in analysis:
                        # Junk detected: remove file from vault
                        if img_file.exists():
                            img_file.unlink()
                        continue

                    # Genuine scientific diagram/chart
                    raw_images.append({
                        "id": f"{paper_id}_{img_stem}",
                        "paper_id": paper_id,
                        "page_num": page_num,
                        "file_path": str(img_file),
                        "caption": f"Gambar Halaman {page_num} #{img_idx + 1}",
                        "ai_analysis": analysis,
                        "width": width,
                        "height": height,
                    })
            except Exception as img_err:
                logger.debug(f"Image extraction skipped on p{page_num}: {img_err}")

        doc.close()

        # Batch compute dense embeddings for all chunks in this paper
        if raw_chunks:
            chunk_texts = [c["content"] for c in raw_chunks]
            embeddings = list(embedder.embed(chunk_texts))
        else:
            embeddings = []

        # Write to SQLite in one ACID transaction
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Insert paper
            cur.execute("""
                INSERT OR REPLACE INTO papers (id, title, filename, path)
                VALUES (?, ?, ?, ?)
            """, (paper_id, title, pdf_path.name, str(pdf_path)))

            # Delete old chunks if re-indexing
            cur.execute("SELECT rowid FROM chunks WHERE paper_id = ?", (paper_id,))
            old_rowids = [r[0] for r in cur.fetchall()]
            for r_id in old_rowids:
                cur.execute("DELETE FROM vec_chunks WHERE chunk_rowid = ?", (r_id,))
            cur.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            cur.execute("DELETE FROM tables WHERE paper_id = ?", (paper_id,))
            cur.execute("DELETE FROM images WHERE paper_id = ?", (paper_id,))

            # Insert chunks & vectors
            for c_info, emb in zip(raw_chunks, embeddings):
                cur.execute("""
                    INSERT INTO chunks (id, paper_id, page_num, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (c_info["id"], c_info["paper_id"], c_info["page_num"], c_info["chunk_index"], c_info["content"]))
                rowid = cur.lastrowid

                # Insert vector into vec_chunks
                blob = serialize_f32(list(emb))
                cur.execute("""
                    INSERT INTO vec_chunks (chunk_rowid, embedding)
                    VALUES (?, ?)
                """, (rowid, blob))

            # Insert tables
            for t_info in raw_tables:
                cur.execute("""
                    INSERT INTO tables (id, paper_id, page_num, csv_path, md_path, caption, is_vision_fallback)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    t_info["id"], t_info["paper_id"], t_info["page_num"],
                    t_info["csv_path"], t_info["md_path"], t_info["caption"],
                    t_info["is_vision_fallback"]
                ))

            # Insert images
            for i_info in raw_images:
                cur.execute("""
                    INSERT INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    i_info["id"], i_info["paper_id"], i_info["page_num"],
                    i_info["file_path"], i_info["caption"], i_info["ai_analysis"],
                    i_info["width"], i_info["height"]
                ))

            conn.commit()

        return {
            "chunks": len(raw_chunks),
            "tables": len(raw_tables),
            "images": len(raw_images),
        }

    def _analyze_vision_table(self, img_path: Path, engine: Any, model_name: str | None) -> str | None:
        """Call multimodal LLM to transcribe complex table crop into Markdown table."""
        if not engine or not img_path.exists():
            return None
        try:
            import base64
            import litellm

            b64_data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            effective_model = model_name or "gemini/gemini-2.5-flash"
            api_base = None
            if effective_model.startswith("9router/"):
                submodel = effective_model.split("/", 1)[1]
                effective_model = f"openai/{submodel}"
                api_base = "http://localhost:20128/v1"

            api_key = engine.config.get_api_key_for_model(model_name or "")
            kwargs = {
                "model": effective_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transkripsikan gambar tabel akademik ini menjadi tabel Markdown yang bersih dan terstruktur. Berikan HANYA format tabel markdown tanpa basa-basi."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "timeout": 15,
            }
            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                kwargs["api_base"] = api_base

            resp = litellm.completion(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    def _analyze_vision_figure(self, img_path: Path, engine: Any, model_name: str | None) -> str | None:
        """Call multimodal LLM to filter junk and transcribe academic diagrams/charts."""
        if not engine or not img_path.exists():
            # If no multimodal engine available, accept image based on heuristic filter
            return "Gambar ilmiah terindeks dari publikasi."
        try:
            import base64
            import litellm

            b64_data = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            effective_model = model_name or "gemini/gemini-2.5-flash"
            api_base = None
            if effective_model.startswith("9router/"):
                submodel = effective_model.split("/", 1)[1]
                effective_model = f"openai/{submodel}"
                api_base = "http://localhost:20128/v1"

            api_key = engine.config.get_api_key_for_model(model_name or "")
            prompt = (
                "Evaluasi gambar dari naskah akademik ini. Apakah gambar ini merupakan grafik empiris, plot statistik, diagram alur/arsitektur, atau peta tematik yang penting?\n"
                "Jika gambar ini HANYA logo penerbit/institusi, watermark, foto profil penulis, atau dekorasi tanpa data, balas HANYA: 'JUNK_IMAGE'.\n"
                "Jika gambar ini merupakan diagram/grafik ilmiah yang valid, berikan ringkasan 2-3 kalimat mengenai informasi visual yang disajikan."
            )
            kwargs = {
                "model": effective_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_data}"}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "timeout": 15,
            }
            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                kwargs["api_base"] = api_base

            resp = litellm.completion(**kwargs)
            out_txt = resp.choices[0].message.content.strip()
            return out_txt
        except Exception:
            return "Gambar ilmiah terindeks dari publikasi."

    def hybrid_search_rrf(self, query: str, top_k: int = 5, k: int = 60) -> list[HybridSearchResult]:
        """Perform Reciprocal Rank Fusion (RRF) search combining FTS5 BM25 and sqlite-vec dense embeddings."""
        clean_query = query.strip()
        if not clean_query:
            return []

        with self._get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT count(*) FROM chunks")
                row = cur.fetchone()
                if not row or row[0] == 0:
                    return []
            except Exception:
                return []

            try:
                embedder = get_shared_embedder()
                query_vec = list(embedder.embed([clean_query]))[0]
            except Exception as e_err:
                logger.warning(f"Vector embedder unavailable: {e_err}")
                query_vec = None

            # 1. Lexical BM25 Search via FTS5
            # Tokenize query for safe MATCH syntax
            fts_tokens = [re.sub(r'[^a-zA-Z0-9]', '', t) for t in clean_query.split()]
            fts_tokens = [t for t in fts_tokens if len(t) >= 2]
            bm25_matches: list[str] = []

            if fts_tokens:
                fts_query = " OR ".join(fts_tokens[:10])
                try:
                    cur.execute("""
                        SELECT chunks.id, rank
                        FROM fts_chunks
                        JOIN chunks ON chunks.rowid = fts_chunks.rowid
                        WHERE fts_chunks MATCH ?
                        ORDER BY rank
                        LIMIT 40
                    """, (fts_query,))
                    bm25_matches = [row["id"] for row in cur.fetchall()]
                except Exception:
                    bm25_matches = []

            # 2. Dense Vector Search via sqlite-vec
            vec_matches: list[str] = []
            if query_vec is not None:
                try:
                    vec_blob = serialize_f32(list(query_vec))
                    cur.execute("""
                        SELECT chunks.id, distance
                        FROM vec_chunks
                        JOIN chunks ON chunks.rowid = vec_chunks.chunk_rowid
                        WHERE embedding MATCH ? AND k = 40
                        ORDER BY distance
                    """, (vec_blob,))
                    vec_matches = [row["id"] for row in cur.fetchall()]
                except Exception:
                    vec_matches = []

            # 3. Reciprocal Rank Fusion (RRF)
            rrf_scores: dict[str, float] = {}
            bm25_rank_map: dict[str, int] = {}
            vec_rank_map: dict[str, int] = {}

            for rank, cid in enumerate(bm25_matches):
                bm25_rank_map[cid] = rank + 1
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

            for rank, cid in enumerate(vec_matches):
                vec_rank_map[cid] = rank + 1
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

            sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

            results: list[HybridSearchResult] = []
            for cid in sorted_ids:
                cur.execute("SELECT paper_id, page_num, content FROM chunks WHERE id = ?", (cid,))
                r = cur.fetchone()
                if not r:
                    continue

                p_id = r["paper_id"]
                page_n = r["page_num"]
                text_content = r["content"]

                # Pull any accompanying tables and figures on this page
                cur.execute("SELECT id, csv_path, md_path, caption FROM tables WHERE paper_id = ? AND page_num = ?", (p_id, page_n))
                matched_tables = [dict(t) for t in cur.fetchall()]

                cur.execute("SELECT id, file_path, caption, ai_analysis FROM images WHERE paper_id = ? AND page_num = ?", (p_id, page_n))
                matched_images = [dict(im) for t in cur.fetchall() for im in [t]]

                results.append(HybridSearchResult(
                    chunk_id=cid,
                    paper_id=p_id,
                    page_num=page_n,
                    content=text_content,
                    rrf_score=round(rrf_scores[cid], 4),
                    bm25_rank=bm25_rank_map.get(cid),
                    vec_rank=vec_rank_map.get(cid),
                    tables=matched_tables,
                    images=matched_images,
                ))

            return results

    def list_inventory(self) -> dict[str, list[dict[str, Any]]]:
        """Return categorized lists of indexed papers, tables, and images in this vault."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, title, filename, path, indexed_at FROM papers ORDER BY indexed_at DESC")
            papers = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT id, paper_id, page_num, md_path, csv_path, caption, is_vision_fallback FROM tables ORDER BY paper_id, page_num")
            tables = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT id, paper_id, page_num, file_path, caption, ai_analysis, width, height FROM images ORDER BY paper_id, page_num")
            images = [dict(r) for r in cur.fetchall()]

            return {
                "papers": papers,
                "tables": tables,
                "images": images,
            }
