"""Local Per-Project Hybrid Vector Vault (sqlite-vec + FTS5 BM25 + PyMuPDF Multimodal)."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import math
import re
import shutil
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
import pymupdf as fitz
import sqlite_vec
from pydantic import BaseModel, Field

from scripcy.core.classifier import VisualClassifier, VisualType

logger = logging.getLogger(__name__)

# Global singleton embedder instance to prevent reload overhead
_GLOBAL_EMBEDDER: Any | None = None
_GLOBAL_RERANKER: Any | None = None

FIGURE_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.|FIG\.|Gambar|Grafik|Diagram|Chart)\s*(\d+[a-zA-Z]?|[IVXLCDM]+)[\s.:—–-]+(.*)",
    re.IGNORECASE,
)

TABLE_CAPTION_RE = re.compile(
    r"^(?:Table|Tabel)\s*(\d+[a-zA-Z]?|[IVXLCDM]+)[\s.:—–-]+(.*)",
    re.IGNORECASE,
)


def evaluate_image_entropy_and_white(img_bytes: bytes) -> tuple[float, float, float]:
    """Calculate Shannon entropy, color variance, and white pixel ratio using PIL."""
    try:
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            rgba = pil_img.convert("RGBA")
            width, height = rgba.size
            total_pixels = width * height
            if total_pixels == 0:
                return 0.0, 0.0, 1.0

            data = rgba.getdata()
            white_or_alpha_count = 0
            gray_values = []
            hist = [0] * 256

            for r, g, b, a in data:
                if a < 15 or (r > 240 and g > 240 and b > 240):
                    white_or_alpha_count += 1
                lum = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray_values.append(lum)
                hist[lum] += 1

            white_ratio = white_or_alpha_count / float(total_pixels)

            # Shannon Entropy
            entropy = 0.0
            for count in hist:
                if count > 0:
                    p = count / float(total_pixels)
                    entropy -= p * math.log2(p)

            # Color Variance
            mean_lum = sum(gray_values) / float(total_pixels)
            variance = sum((x - mean_lum) ** 2 for x in gray_values) / float(total_pixels)

            return entropy, variance, white_ratio
    except Exception:
        return 4.0, 50.0, 0.0


def match_image_to_caption(page: fitz.Page, img_rect: fitz.Rect) -> tuple[str, str] | None:
    """Find the nearest valid academic figure caption below or above the image."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return None

    best_match: tuple[str, str] | None = None
    min_dist = float("inf")

    for b in blocks:
        if len(b) < 5 or b[6] != 0:
            continue
        text = b[4].strip()
        m = FIGURE_CAPTION_RE.match(text)
        if not m:
            continue

        cap_rect = fitz.Rect(b[0], b[1], b[2], b[3])
        # Check below image within 85 pt
        if cap_rect.y0 >= (img_rect.y1 - 15) and (cap_rect.y0 - img_rect.y1) <= 85:
            dist = cap_rect.y0 - img_rect.y1
            if dist < min_dist:
                min_dist = dist
                label = f"Gambar {m.group(1)}"
                clean_full = text.replace("\n", " ").strip()
                best_match = (label, clean_full)
        # Check above image within 85 pt
        elif cap_rect.y1 <= (img_rect.y0 + 15) and (img_rect.y0 - cap_rect.y1) <= 85:
            dist = img_rect.y0 - cap_rect.y1
            if dist < min_dist:
                min_dist = dist
                label = f"Gambar {m.group(1)}"
                clean_full = text.replace("\n", " ").strip()
                best_match = (label, clean_full)

    return best_match


def match_table_to_caption(page: fitz.Page, table_rect: fitz.Rect) -> tuple[str, str] | None:
    """Find the nearest valid academic table caption above or below the table."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return None

    best_match: tuple[str, str] | None = None
    min_dist = float("inf")

    for b in blocks:
        if len(b) < 5 or b[6] != 0:
            continue
        text = b[4].strip()
        m = TABLE_CAPTION_RE.match(text)
        if not m:
            continue

        cap_rect = fitz.Rect(b[0], b[1], b[2], b[3])
        # Check above table within 85 pt
        if cap_rect.y1 <= (table_rect.y0 + 15) and (table_rect.y0 - cap_rect.y1) <= 85:
            dist = table_rect.y0 - cap_rect.y1
            if dist < min_dist:
                min_dist = dist
                label = f"Tabel {m.group(1)}"
                clean_full = text.replace("\n", " ").strip()
                best_match = (label, clean_full)
        # Check below table within 85 pt
        elif cap_rect.y0 >= (table_rect.y1 - 15) and (cap_rect.y0 - table_rect.y1) <= 85:
            dist = cap_rect.y0 - table_rect.y1
            if dist < min_dist:
                min_dist = dist
                label = f"Tabel {m.group(1)}"
                clean_full = text.replace("\n", " ").strip()
                best_match = (label, clean_full)

    return best_match


def get_shared_embedder() -> Any:
    """Return shared FastEmbed BGE-small instance with optimized ONNX thread limits."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        from fastembed import TextEmbedding
        # Limit internal ONNX threads to 2 to prevent excessive memory arena expansion
        _GLOBAL_EMBEDDER = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=2)
    return _GLOBAL_EMBEDDER


def get_shared_reranker() -> Any:
    """Return shared FastEmbed TextCrossEncoder instance (dim: 80MB, MiniLM-L-6-v2 ONNX)."""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _GLOBAL_RERANKER = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.debug(f"Failed loading cross encoder reranker: {e}")
            _GLOBAL_RERANKER = None
    return _GLOBAL_RERANKER


def serialize_f32(vector: list[float]) -> bytes:
    """Serialize float32 vector into raw binary buffer for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


@dataclass
class HybridSearchResult:
    """Result item returned by RRF hybrid search and cross-encoder re-ranking."""

    chunk_id: str
    paper_id: str
    page_num: int
    content: str
    rrf_score: float
    parent_content: str = ""
    rerank_score: float | None = None
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

            # Chunks table (with parent-child hierarchy support)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    parent_content TEXT,
                    section_title TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
            """)

            # Ensure parent_content column exists if table was created in older schema
            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN parent_content TEXT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN section_title TEXT")
            except Exception:
                pass

            # FTS5 full-text search table for BM25
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)

            # Triggers to keep external-content FTS5 synchronized with chunks
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO fts_chunks(rowid, content) VALUES (new.rowid, new.content);
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.rowid, old.content);
                END;
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.rowid, old.content);
                    INSERT INTO fts_chunks(rowid, content) VALUES (new.rowid, new.content);
                END;
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
                    visual_type TEXT DEFAULT 'SIMPLE',
                    classification_reason TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
            """)

            # Ensure columns exist if table was created in older schema
            try:
                cur.execute("ALTER TABLE images ADD COLUMN visual_type TEXT DEFAULT 'SIMPLE'")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE images ADD COLUMN classification_reason TEXT")
            except Exception:
                pass

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

            # Ensure FTS5 index is fully synchronized with existing chunks on initialization
            try:
                cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
            except Exception:
                pass

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

        embedder = get_shared_embedder()
        raw_chunks: list[dict[str, Any]] = []
        raw_tables: list[dict[str, Any]] = []
        raw_images: list[dict[str, Any]] = []
        chunk_counter = 0

        # Strict resource-managed PDF extraction via context manager
        with fitz.open(str(pdf_path)) as doc:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                rect_page = page.rect

                # 1. Text Extraction & Parent-Child Chunking with Running Header/Footer Filtering
                # Filter out running heads (top margin < 35 pt) and running footers (bottom margin > height - 40 pt)
                try:
                    p_blocks = page.get_text("blocks")
                    p_height = rect_page.height
                    substantive_blocks = [
                        b[4].strip()
                        for b in p_blocks
                        if len(b) >= 5 and b[6] == 0 and b[1] >= 35 and b[3] <= (p_height - 35) and len(b[4].strip()) > 5
                    ]
                    page_text = "\n\n".join(substantive_blocks).strip() if substantive_blocks else page.get_text("text").strip()
                except Exception:
                    page_text = page.get_text("text").strip()

                if page_text:
                    words = page_text.split()
                    # Determine parent window (~400 words)
                    parent_content = " ".join(words[:450]) if words else ""
                    child_step = max(1, 110 - 25)
                    for start in range(0, len(words), child_step):
                        chunk_words = words[start : start + 110]
                        chunk_str = " ".join(chunk_words)
                        # Dynamic surrounding parent context (±150 words around child)
                        p_start = max(0, start - 80)
                        p_end = min(len(words), start + 110 + 80)
                        dynamic_parent = " ".join(words[p_start:p_end])

                        if len(chunk_str.strip()) >= 20:
                            raw_chunks.append({
                                "id": f"{paper_id}_p{page_num}_c{chunk_counter}",
                                "paper_id": paper_id,
                                "page_num": page_num,
                                "chunk_index": chunk_counter,
                                "content": chunk_str,
                                "parent_content": dynamic_parent or parent_content,
                            })
                            chunk_counter += 1
                        if start + 110 >= len(words):
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
                        tbl_cap_match = match_table_to_caption(page, tab.bbox)
                        tbl_caption = tbl_cap_match[1] if tbl_cap_match else f"Tabel Halaman {page_num} #{t_idx + 1}"

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
                                "caption": tbl_caption,
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
                                    "caption": tbl_caption,
                                    "is_vision_fallback": 1,
                                })
                            except Exception as crop_err:
                                logger.warning(f"Failed cropping table on p{page_num}: {crop_err}")
                except Exception as tab_err:
                    logger.debug(f"Table detection skipped on p{page_num}: {tab_err}")

                # 3. Informative Image Extraction with Multi-stage Academic Junk Filter
                try:
                    for img_idx, img_info in enumerate(page.get_images(full=True)):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        width, height = base_image["width"], base_image["height"]

                        # Heuristic Rule 1: Min dimensions (>= 120x120)
                        if width < 120 or height < 120:
                            continue

                        # Heuristic Rule 2: Aspect ratio (0.2 to 5.5 - eliminate long border lines)
                        ratio = width / max(1, height)
                        if ratio < 0.20 or ratio > 5.5:
                            continue

                        # Heuristic Rule 3: Page boundary exclusion (skip top 8% header logos and bottom 8% copyright footers)
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            continue
                        bbox = img_rects[0]
                        if bbox.y1 < rect_page.height * 0.08 or bbox.y0 > rect_page.height * 0.92:
                            continue

                        # Heuristic Rule 4: Physical page area check (skip icons / bullets < 2.5% of page)
                        page_area = max(1.0, rect_page.width * rect_page.height)
                        if (bbox.width * bbox.height) / page_area < 0.025:
                            continue

                        # Heuristic Rule 5: Color variance, Shannon entropy, and white pixel ratio
                        entropy, variance, white_ratio = evaluate_image_entropy_and_white(base_image["image"])
                        # Discard pure white masks, solid boxes, and zero-entropy bars
                        if white_ratio > 0.90 or variance < 4.5 or entropy < 2.5:
                            continue

                        # Heuristic Rule 6: Proximity Caption Matching
                        cap_match = match_image_to_caption(page, bbox)
                        if cap_match:
                            fig_label, caption_text = cap_match
                        else:
                            # If no academic figure caption is found, require high visual complexity
                            if entropy < 4.0:
                                continue
                            caption_text = f"Visual Diagram Halaman {page_num} #{img_idx + 1}"

                        # Candidate image passed all offline CV filters!
                        img_ext = base_image["ext"]
                        img_stem = f"fig_p{page_num}_{img_idx + 1}"
                        img_file = paper_images_dir / f"{img_stem}.{img_ext}"
                        img_file.write_bytes(base_image["image"])

                        # Visual Complexity Classification (PyMuPDF Vector stream inspection with Raster fallback)
                        try:
                            v_res = VisualClassifier.classify_vector_bbox(page, bbox, caption=caption_text)
                        except Exception:
                            v_res = VisualClassifier.classify_raster(img_file, caption=caption_text)

                        visual_type_str = v_res.visual_type.value
                        class_reasons = "; ".join(v_res.reasons[:3])

                        # Optional Multimodal Verification / AI Analysis if engine provided
                        analysis = None
                        if v_res.is_simple:
                            # For SIMPLE visual (flowchart/architecture), run AI Visual Analysis once ahead-of-time
                            if engine:
                                analysis = self._analyze_vision_figure(img_file, engine, model_name)
                                if analysis and "JUNK_IMAGE" in analysis:
                                    if img_file.exists():
                                        img_file.unlink()
                                    continue
                            if not analysis:
                                analysis = f"Bagan/diagram alir konseptual terindeks: {caption_text}."
                        else:
                            # For COMPLEX visual (charts, plots, heatmap, math), keep raw pixels for Direct Multimodal Injection
                            analysis = f"Visual data kompleks terindeks ({class_reasons}). Tersedia untuk Direct Multimodal Injection."

                        # Genuine scientific diagram/chart
                        raw_images.append({
                            "id": f"{paper_id}_{img_stem}",
                            "paper_id": paper_id,
                            "page_num": page_num,
                            "file_path": str(img_file),
                            "caption": caption_text,
                            "ai_analysis": analysis,
                            "width": width,
                            "height": height,
                            "visual_type": visual_type_str,
                            "classification_reason": class_reasons,
                        })
                except Exception as img_err:
                    logger.debug(f"Image extraction skipped on p{page_num}: {img_err}")

        # Batch compute dense embeddings for all chunks in this paper with streaming batching (batch_size=32)
        if raw_chunks and embedder is not None:
            chunk_texts = [c["content"] for c in raw_chunks]
            embeddings = list(embedder.embed(chunk_texts, batch_size=32))
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
                    INSERT INTO chunks (id, paper_id, page_num, chunk_index, content, parent_content)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    c_info["id"], c_info["paper_id"], c_info["page_num"],
                    c_info["chunk_index"], c_info["content"], c_info.get("parent_content", c_info["content"])
                ))
                rowid = cur.lastrowid

                # Insert vector into vec_chunks
                blob = serialize_f32(list(emb))
                cur.execute("""
                    INSERT INTO vec_chunks (chunk_rowid, embedding)
                    VALUES (?, ?)
                """, (rowid, blob))

            # Insert tables
            for t_idx, t_info in enumerate(raw_tables):
                cur.execute("""
                    INSERT INTO tables (id, paper_id, page_num, csv_path, md_path, caption, is_vision_fallback)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    t_info["id"], t_info["paper_id"], t_info["page_num"],
                    t_info["csv_path"], t_info["md_path"], t_info["caption"],
                    t_info["is_vision_fallback"]
                ))
                # Register visual table chunk into chunks and vec_chunks for hybrid BM25/dense retrieval
                tbl_id = t_info["id"]
                t_snippet = ""
                if t_info.get("md_path") and Path(t_info["md_path"]).exists():
                    try:
                        t_snippet = "\n" + Path(t_info["md_path"]).read_text(encoding="utf-8")[:400]
                    except Exception:
                        t_snippet = ""
                tbl_content = f"[[VISUAL:{tbl_id}]] [Tabel Halaman {t_info['page_num']}: {t_info['caption']}]{t_snippet}"
                cur.execute("""
                    INSERT INTO chunks (id, paper_id, page_num, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"tbl_chunk_{tbl_id}", paper_id, t_info["page_num"], 9000 + t_idx, tbl_content))
                rowid = cur.lastrowid
                if embedder is not None:
                    try:
                        t_emb = list(embedder.embed([tbl_content]))[0]
                        cur.execute("INSERT INTO vec_chunks (chunk_rowid, embedding) VALUES (?, ?)", (rowid, serialize_f32(list(t_emb))))
                    except Exception:
                        pass

            # Insert images
            for i_idx, i_info in enumerate(raw_images):
                v_type = i_info.get("visual_type", "SIMPLE")
                cur.execute("""
                    INSERT INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height, visual_type, classification_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    i_info["id"], i_info["paper_id"], i_info["page_num"],
                    i_info["file_path"], i_info["caption"], i_info["ai_analysis"],
                    i_info["width"], i_info["height"], v_type, i_info.get("classification_reason", "")
                ))
                # Register visual image chunk into chunks and vec_chunks for hybrid BM25/dense retrieval
                img_id = i_info["id"]
                visual_tag = f"[[VISUAL:{img_id}:COMPLEX]]" if v_type == "COMPLEX" else f"[[VISUAL:{img_id}]]"
                img_content = f"{visual_tag} [Gambar Halaman {i_info['page_num']}: {i_info['caption']}]\nKeterangan/Analisis: {i_info.get('ai_analysis', '')}"
                cur.execute("""
                    INSERT INTO chunks (id, paper_id, page_num, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"img_chunk_{img_id}", paper_id, i_info["page_num"], 9500 + i_idx, img_content))
                rowid = cur.lastrowid
                if embedder is not None:
                    try:
                        i_emb = list(embedder.embed([img_content]))[0]
                        cur.execute("INSERT INTO vec_chunks (chunk_rowid, embedding) VALUES (?, ?)", (rowid, serialize_f32(list(i_emb))))
                    except Exception:
                        pass

            # Ensure FTS5 index is fully synchronized with new chunks
            try:
                cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
            except Exception:
                pass

            conn.commit()

        # Free temporary chunk arrays and force immediate heap garbage collection
        c_count = len(raw_chunks)
        t_count = len(raw_tables)
        i_count = len(raw_images)
        raw_chunks.clear()
        raw_tables.clear()
        raw_images.clear()
        import gc
        gc.collect()

        return {
            "chunks": c_count,
            "tables": t_count,
            "images": i_count,
        }

    def ingest_web_document(
        self,
        paper_id: str,
        title: str,
        markdown_text: str,
        source_path: Path,
        images: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        """Ingest parsed official web document into SQLite vector vault."""
        embedder = get_shared_embedder()
        raw_chunks: list[dict[str, Any]] = []
        chunk_counter = 0

        words = markdown_text.split()
        child_step = max(1, 110 - 25)
        for start in range(0, len(words), child_step):
            chunk_words = words[start : start + 110]
            chunk_str = " ".join(chunk_words)
            p_start = max(0, start - 80)
            p_end = min(len(words), start + 110 + 80)
            dynamic_parent = " ".join(words[p_start:p_end])

            if len(chunk_str.strip()) >= 20:
                raw_chunks.append({
                    "id": f"{paper_id}_w1_c{chunk_counter}",
                    "paper_id": paper_id,
                    "page_num": 1,
                    "chunk_index": chunk_counter,
                    "content": chunk_str,
                    "parent_content": dynamic_parent or chunk_str,
                })
                chunk_counter += 1
            if start + 110 >= len(words):
                break

        if raw_chunks and embedder is not None:
            chunk_texts = [c["content"] for c in raw_chunks]
            embeddings = list(embedder.embed(chunk_texts, batch_size=32))
        else:
            embeddings = []

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO papers (id, title, filename, path)
                VALUES (?, ?, ?, ?)
            """, (paper_id, title, source_path.name, str(source_path)))

            cur.execute("SELECT rowid FROM chunks WHERE paper_id = ?", (paper_id,))
            old_rowids = [r[0] for r in cur.fetchall()]
            for r_id in old_rowids:
                cur.execute("DELETE FROM vec_chunks WHERE chunk_rowid = ?", (r_id,))
            cur.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            cur.execute("DELETE FROM images WHERE paper_id = ?", (paper_id,))

            for c_info, emb in zip(raw_chunks, embeddings):
                cur.execute("""
                    INSERT INTO chunks (id, paper_id, page_num, chunk_index, content, parent_content)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    c_info["id"], c_info["paper_id"], c_info["page_num"],
                    c_info["chunk_index"], c_info["content"], c_info.get("parent_content", c_info["content"])
                ))
                rowid = cur.lastrowid
                blob = serialize_f32(list(emb))
                cur.execute("""
                    INSERT INTO vec_chunks (chunk_rowid, embedding)
                    VALUES (?, ?)
                """, (rowid, blob))

            raw_images = images or []
            for i_idx, i_info in enumerate(raw_images):
                img_path_str = i_info.get("file_path", "")
                cap_str = i_info.get("caption", "") or ""
                # Classify web raster image
                v_res = VisualClassifier.classify_raster(img_path_str, caption=cap_str)
                v_type = v_res.visual_type.value
                reasons_str = "; ".join(v_res.reasons[:3])

                cur.execute("""
                    INSERT INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height, visual_type, classification_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    i_info["id"], i_info["paper_id"], i_info.get("page_num", 1),
                    img_path_str, cap_str, i_info.get("ai_analysis", "Visual resmi web terindeks."),
                    i_info.get("width", 0), i_info.get("height", 0), v_type, reasons_str
                ))
                # Register web image chunk into chunks and vec_chunks for hybrid BM25/dense retrieval
                img_id = i_info["id"]
                visual_tag = f"[[VISUAL:{img_id}:COMPLEX]]" if v_type == "COMPLEX" else f"[[VISUAL:{img_id}]]"
                img_cap = cap_str or "Bagan Resmi"
                img_desc = i_info.get("ai_analysis", "") or "Visual resmi web terindeks."
                img_content = f"{visual_tag} [Gambar/Infografis Resmi: {img_cap}]\nKeterangan/Analisis: {img_desc}"
                cur.execute("""
                    INSERT INTO chunks (id, paper_id, page_num, chunk_index, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"web_img_chunk_{img_id}", paper_id, i_info.get("page_num", 1), 9500 + i_idx, img_content))
                rowid = cur.lastrowid
                if embedder is not None:
                    try:
                        i_emb = list(embedder.embed([img_content]))[0]
                        cur.execute("INSERT INTO vec_chunks (chunk_rowid, embedding) VALUES (?, ?)", (rowid, serialize_f32(list(i_emb))))
                    except Exception:
                        pass

            try:
                cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
            except Exception:
                pass

            conn.commit()

        # Free temporary chunk arrays and force immediate heap garbage collection
        c_count = len(raw_chunks)
        i_count = len(images) if images else 0
        raw_chunks.clear()
        import gc
        gc.collect()

        return {
            "chunks": c_count,
            "tables": 0,
            "images": i_count,
        }

    def ingest_standalone_table(
        self,
        table_path: Path,
        caption: str = "",
        context: str = "",
        target_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Ingest a standalone user-uploaded table (.csv, .tsv, .xlsx, .md) into SQLite vector vault."""
        embedder = get_shared_embedder()
        dest_dir = target_dir or (self.tables_dir / "user")
        dest_dir.mkdir(parents=True, exist_ok=True)

        stem = table_path.stem
        paper_id = "[Unggahan Mandiri]"
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO papers (id, title, filename, path)
                VALUES (?, ?, ?, ?)
            """, (paper_id, "Unggahan Mandiri", "standalone", ""))
            conn.commit()

        dest_csv = dest_dir / f"{stem}.csv"
        dest_md = dest_dir / f"{stem}.md"

        md_content = ""
        csv_content = ""
        ext = table_path.suffix.lower()

        if ext in (".csv", ".tsv"):
            try:
                raw_text = table_path.read_text(encoding="utf-8", errors="replace")
                delimiter = "\t" if ext == ".tsv" else ("," if "," in raw_text[:500] else ";")
                reader = csv.reader(raw_text.splitlines(), delimiter=delimiter)
                rows = list(reader)
                if rows:
                    headers = [str(c or "").strip() for c in rows[0]]
                    md_lines = ["| " + " | ".join(headers) + " |"]
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for r in rows[1:]:
                        r_padded = r + [""] * (len(headers) - len(r))
                        md_lines.append("| " + " | ".join(str(c or "").strip() for c in r_padded[:len(headers)]) + " |")
                    md_content = "\n".join(md_lines)
                    csv_content = raw_text
            except Exception:
                pass
        elif ext in (".xlsx", ".xls"):
            try:
                import pandas as pd
                df = pd.read_excel(table_path)
                csv_content = df.to_csv(index=False)
                md_content = df.to_markdown(index=False)
            except Exception:
                pass

        if not md_content:
            try:
                md_content = table_path.read_text(encoding="utf-8", errors="replace")
                csv_content = md_content
            except Exception:
                md_content = f"# Tabel Mandiri: {table_path.name}"
                csv_content = md_content

        dest_md.write_text(md_content, encoding="utf-8")
        dest_csv.write_text(csv_content, encoding="utf-8")

        tbl_id = f"user_tbl_{stem}"
        tbl_caption = caption or f"Tabel Mandiri {stem}"
        if context:
            tbl_caption = f"{tbl_caption} ({context})"

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO tables (id, paper_id, page_num, csv_path, md_path, caption, is_vision_fallback)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (tbl_id, paper_id, 1, str(dest_csv), str(dest_md), tbl_caption, 0))

            tbl_chunk_content = f"[[VISUAL:{tbl_id}]] [Tabel Mandiri: {tbl_caption}]\n{md_content[:800]}"
            cur.execute("""
                INSERT INTO chunks (id, paper_id, page_num, chunk_index, content)
                VALUES (?, ?, ?, ?, ?)
            """, (f"tbl_chunk_{tbl_id}", paper_id, 1, 9900, tbl_chunk_content))
            rowid = cur.lastrowid
            if embedder is not None:
                try:
                    t_emb = list(embedder.embed([tbl_chunk_content]))[0]
                    cur.execute("INSERT INTO vec_chunks (chunk_rowid, embedding) VALUES (?, ?)", (rowid, serialize_f32(list(t_emb))))
                except Exception:
                    pass

            try:
                cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
            except Exception:
                pass
            conn.commit()

        # Free temporary chunk arrays and force immediate heap garbage collection
        raw_chunks.clear()
        raw_tables.clear()
        raw_images.clear()
        import gc
        gc.collect()

        return {
            "id": tbl_id,
            "md_path": str(dest_md),
            "csv_path": str(dest_csv),
            "caption": tbl_caption,
        }

    def ingest_standalone_image(
        self,
        image_path: Path,
        caption: str = "",
        context: str = "",
        ai_analysis: str = "",
        target_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Ingest a standalone user-uploaded image (.png, .jpg, .svg, etc.) into SQLite vector vault."""
        embedder = get_shared_embedder()
        dest_dir = target_dir or (self.images_dir / "user")
        dest_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        paper_id = "[Unggahan Mandiri]"
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO papers (id, title, filename, path)
                VALUES (?, ?, ?, ?)
            """, (paper_id, "Unggahan Mandiri", "standalone", ""))
            conn.commit()

        dest_img = dest_dir / image_path.name
        if image_path.resolve() != dest_img.resolve():
            shutil.copy2(str(image_path), str(dest_img))

        width = 0
        height = 0
        try:
            with Image.open(dest_img) as img:
                width, height = img.size
        except Exception:
            pass

        img_id = f"user_img_{stem}"
        img_caption = caption or f"Visual Mandiri {stem}"
        if context:
            img_caption = f"{img_caption} ({context})"
        desc = ai_analysis or context or "Visual mandiri terindeks pengguna."

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO images (id, paper_id, page_num, file_path, caption, ai_analysis, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (img_id, paper_id, 1, str(dest_img), img_caption, desc, width, height))

            img_chunk_content = f"[[VISUAL:{img_id}]] [Visual Mandiri: {img_caption}]\nKeterangan/Analisis: {desc}"
            cur.execute("""
                INSERT INTO chunks (id, paper_id, page_num, chunk_index, content)
                VALUES (?, ?, ?, ?, ?)
            """, (f"img_chunk_{img_id}", paper_id, 1, 9950, img_chunk_content))
            rowid = cur.lastrowid
            if embedder is not None:
                try:
                    i_emb = list(embedder.embed([img_chunk_content]))[0]
                    cur.execute("INSERT INTO vec_chunks (chunk_rowid, embedding) VALUES (?, ?)", (rowid, serialize_f32(list(i_emb))))
                except Exception:
                    pass

            try:
                cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
            except Exception:
                pass
            conn.commit()

        # Free temporary chunk arrays and force immediate heap garbage collection
        raw_chunks.clear()
        raw_tables.clear()
        raw_images.clear()
        import gc
        gc.collect()

        return {
            "id": img_id,
            "file_path": str(dest_img),
            "caption": img_caption,
            "width": width,
            "height": height,
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

            # Boost official website / institutional sources on factual & statistical queries
            lower_q = clean_query.lower()
            is_factual_q = any(
                w in lower_q for w in (
                    "data", "angka", "berapa", "statistik", "jumlah", "luas", "tahun",
                    "tabel", "bps", "kementerian", "resmi", "pemerintah", "nasional", "ha", "hektar"
                )
            )
            if is_factual_q and rrf_scores:
                cids_list = list(rrf_scores.keys())
                placeholders = ",".join("?" for _ in cids_list)
                try:
                    cur.execute(f"SELECT id, paper_id FROM chunks WHERE id IN ({placeholders})", cids_list)
                    for row in cur.fetchall():
                        cid = row["id"]
                        pid = str(row["paper_id"]).lower()
                        if pid.startswith("web_") or "go_id" in pid or "gov" in pid or "bps" in pid or "menlhk" in pid or "kemen" in pid:
                            rrf_scores[cid] = rrf_scores[cid] * 1.40
                except Exception:
                    pass

            # First stage candidates: take top 15 for cross-encoder reranking
            candidate_limit = max(top_k, 15)
            sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:candidate_limit]

            candidates: list[HybridSearchResult] = []
            for cid in sorted_ids:
                cur.execute("SELECT paper_id, page_num, content, parent_content FROM chunks WHERE id = ?", (cid,))
                r = cur.fetchone()
                if not r:
                    continue

                p_id = r["paper_id"]
                page_n = r["page_num"]
                text_content = r["content"]
                parent_text = r["parent_content"] or text_content

                # Pull any accompanying tables and figures on this page
                cur.execute("SELECT id, csv_path, md_path, caption FROM tables WHERE paper_id = ? AND page_num = ?", (p_id, page_n))
                matched_tables = [dict(t) for t in cur.fetchall()]

                cur.execute("SELECT id, file_path, caption, ai_analysis, visual_type, classification_reason FROM images WHERE paper_id = ? AND page_num = ?", (p_id, page_n))
                matched_images = [dict(im) for t in cur.fetchall() for im in [t]]

                candidates.append(HybridSearchResult(
                    chunk_id=cid,
                    paper_id=p_id,
                    page_num=page_n,
                    content=text_content,
                    parent_content=parent_text,
                    rrf_score=round(rrf_scores[cid], 4),
                    bm25_rank=bm25_rank_map.get(cid),
                    vec_rank=vec_rank_map.get(cid),
                    tables=matched_tables,
                    images=matched_images,
                ))

            # Second stage: Cross-Encoder Reranking
            reranker = get_shared_reranker()
            if reranker is not None and candidates:
                try:
                    doc_texts = [c.content for c in candidates]
                    rerank_scores = list(reranker.rerank(clean_query, doc_texts))
                    for c, r_score in zip(candidates, rerank_scores):
                        c.rerank_score = float(r_score)
                    # Sort primarily by cross-encoder score
                    candidates.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -999.0, reverse=True)
                except Exception as r_err:
                    logger.debug(f"Cross-encoder reranking failed, falling back to RRF: {r_err}")

            return candidates[:top_k]

    def list_inventory(self) -> dict[str, list[dict[str, Any]]]:
        """Return categorized lists of indexed papers, tables, and images in this vault."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, title, filename, path, indexed_at FROM papers ORDER BY indexed_at DESC")
            papers = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT id, paper_id, page_num, md_path, csv_path, caption, is_vision_fallback FROM tables ORDER BY paper_id, page_num")
            tables = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT id, paper_id, page_num, file_path, caption, ai_analysis, width, height, visual_type, classification_reason FROM images ORDER BY paper_id, page_num")
            images = [dict(r) for r in cur.fetchall()]

            return {
                "papers": papers,
                "tables": tables,
                "images": images,
            }

    def search_visual_assets(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[dict[str, Any]]:
        """Search and rank relevant academic figures and tables in the vault for a given query."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # Tokenize query for matching
        q_tokens = [
            t.lower() for t in re.findall(r"\w+", clean_query)
            if len(t) >= 3 and t.lower() not in (
                "yang", "pada", "dari", "untuk", "dengan", "dalam", "adalah", "bahwa",
                "tidak", "akan", "dapat", "oleh", "tentang", "bagaimana", "mengapa",
                "apakah", "apa", "sudah", "bisa", "jelaskan", "buat", "tulis"
            )
        ]

        visual_intent_keywords = {
            "gambar", "figure", "fig", "grafik", "plot", "diagram", "chart",
            "tabel", "table", "hasil", "akurasi", "perbandingan", "skema",
            "arsitektur", "flowchart", "benchmark", "matrix", "matriks", "kurva",
            "tren", "trend", "data", "peta", "sebaran", "luas", "jumlah", "statistik"
        }
        has_visual_intent = any(w in clean_query.lower() for w in visual_intent_keywords)

        candidates: list[dict[str, Any]] = []

        with self._get_connection() as conn:
            cur = conn.cursor()

            # 1. Search Images
            cur.execute("""
                SELECT images.id, images.paper_id, images.page_num, images.file_path,
                       images.caption, images.ai_analysis, images.width, images.height,
                       papers.title as paper_title
                FROM images
                LEFT JOIN papers ON papers.id = images.paper_id
            """)
            for row in cur.fetchall():
                im_dict = dict(row)
                score = 0.0
                searchable = f"{im_dict.get('caption', '')} {im_dict.get('ai_analysis', '')} {im_dict.get('paper_title', '')}".lower()

                for tok in q_tokens:
                    if tok in searchable:
                        score += 3.0
                    elif any(tok in w for w in searchable.split()):
                        score += 1.5

                if score > 0.0 or has_visual_intent:
                    base_score = score if score > 0.0 else 1.0
                    if has_visual_intent:
                        base_score *= 1.4
                    candidates.append({
                        "asset_type": "image",
                        "id": im_dict["id"],
                        "paper_id": im_dict["paper_id"],
                        "paper_title": im_dict["paper_title"] or im_dict["paper_id"],
                        "page_num": im_dict["page_num"],
                        "file_path": im_dict["file_path"],
                        "caption": im_dict["caption"],
                        "ai_analysis": im_dict["ai_analysis"],
                        "score": base_score,
                    })

            # 2. Search Tables
            cur.execute("""
                SELECT tables.id, tables.paper_id, tables.page_num, tables.csv_path,
                       tables.md_path, tables.caption, tables.is_vision_fallback,
                       papers.title as paper_title
                FROM tables
                LEFT JOIN papers ON papers.id = tables.paper_id
            """)
            for row in cur.fetchall():
                tbl_dict = dict(row)
                score = 0.0
                searchable = f"{tbl_dict.get('caption', '')} {tbl_dict.get('paper_title', '')}".lower()

                for tok in q_tokens:
                    if tok in searchable:
                        score += 3.0

                if score > 0.0 or has_visual_intent:
                    base_score = score if score > 0.0 else 1.0
                    if has_visual_intent:
                        base_score *= 1.4
                    candidates.append({
                        "asset_type": "table",
                        "id": tbl_dict["id"],
                        "paper_id": tbl_dict["paper_id"],
                        "paper_title": tbl_dict["paper_title"] or tbl_dict["paper_id"],
                        "page_num": tbl_dict["page_num"],
                        "csv_path": tbl_dict["csv_path"],
                        "md_path": tbl_dict["md_path"],
                        "caption": tbl_dict["caption"],
                        "score": base_score,
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def reindex_all_visuals(
        self,
        engine: Any = None,
        model_name: str | None = None,
    ) -> dict[str, int]:
        """Re-ingest all PDF documents in papers_dir, purging junk images and updating tables/images."""
        import shutil

        # Clean physical image and table folders
        if self.images_dir.exists():
            for item in self.images_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                elif item.is_file():
                    try:
                        item.unlink()
                    except Exception:
                        pass

        if self.tables_dir.exists():
            for item in self.tables_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                elif item.is_file():
                    try:
                        item.unlink()
                    except Exception:
                        pass

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM images")
            cur.execute("DELETE FROM tables")
            conn.commit()

        pdf_files = list(self.papers_dir.glob("*.pdf")) + list(self.vault_dir.glob("*.pdf"))
        seen_names = set()
        unique_pdfs = []
        for p in pdf_files:
            if p.name not in seen_names:
                seen_names.add(p.name)
                unique_pdfs.append(p)

        total_figures = 0
        total_tables = 0

        for pdf_p in unique_pdfs:
            p_id = pdf_p.stem
            try:
                res = self.ingest_paper_pdf(
                    pdf_path=pdf_p,
                    paper_id=p_id,
                    title=p_id.replace("_", " "),
                    engine=engine,
                    model_name=model_name,
                )
                total_figures += res.get("images", 0)
                total_tables += res.get("tables", 0)
            except Exception as e:
                logger.warning(f"Reindex failed for {pdf_p.name}: {e}")

        return {
            "papers": len(unique_pdfs),
            "images": total_figures,
            "tables": total_tables,
        }

    def _resolve_paper_citation(self, paper_id: str) -> str:
        """Resolve APA in-text citation like '(Melisa et al., 2022)' from cache frontmatter or paper id."""
        if not paper_id:
            return ""
        try:
            cache_p = self.vault_dir.parent / "cache" / f"{paper_id}.md"
            if cache_p.exists():
                from scripcy.core.models import FRONTMATTER_REGEX
                import yaml
                text = cache_p.read_text(encoding="utf-8", errors="replace")
                m = FRONTMATTER_REGEX.match(text)
                if m:
                    data = yaml.safe_load(m.group(1)) or {}
                    authors = data.get("authors") or []
                    year = data.get("year") or ""
                    if authors:
                        clean_authors = []
                        for a in authors:
                            clean_a = str(a).strip()
                            if "(" in clean_a and ")" in clean_a:
                                clean_authors.append(clean_a)
                            else:
                                surname = clean_a.split(",")[0].strip()
                                clean_authors.append(surname)
                        if len(clean_authors) == 1:
                            auth_str = clean_authors[0]
                        elif len(clean_authors) == 2:
                            auth_str = f"{clean_authors[0]} & {clean_authors[1]}"
                        else:
                            auth_str = f"{clean_authors[0]} et al."
                        return f"({auth_str}, {year})" if year else f"({auth_str})"
                    elif year:
                        return f"({paper_id}, {year})"
        except Exception:
            pass

        y_match = re.search(r"\b(19\d\d|20\d\d)\b", paper_id)
        if y_match:
            first_word = paper_id.split("_")[0].replace("-", " ").title()
            return f"({first_word} et al., {y_match.group(1)})"
        return f"({paper_id})"

    def resolve_visual_placeholders(self, text: str) -> str:
        """Resolve [[VISUAL:id]] placeholders into interactive visual blocks with clickable file links and citations."""
        if not text:
            return text

        pattern = re.compile(r"\[\[VISUAL:([^\]]+)\]\]", re.IGNORECASE)

        def _repl(match: re.Match) -> str:
            raw_vid = match.group(1).strip()
            # Strip optional :COMPLEX suffix if present
            v_id = raw_vid.split(":")[0].strip()
            with self._get_connection() as conn:
                cur = conn.cursor()
                # 1. Lookup in images
                cur.execute("SELECT file_path, caption, paper_id FROM images WHERE id = ? LIMIT 1", (v_id,))
                row = cur.fetchone()
                if row:
                    raw_fp = Path(row["file_path"]).resolve().as_posix()
                    if not raw_fp.startswith("/"):
                        raw_fp = f"/{raw_fp}"
                    cap = (row["caption"] or "Gambar").strip()
                    cit = self._resolve_paper_citation(row["paper_id"])
                    cit_line = f"  \n{cit}" if cit else ""
                    return f"\n\n[📊 Buka Gambar](file://{raw_fp})  \n{cap}{cit_line}\n\n"

                # 2. Lookup in tables
                cur.execute("SELECT csv_path, md_path, caption, paper_id FROM tables WHERE id = ? LIMIT 1", (v_id,))
                t_row = cur.fetchone()
                if t_row:
                    target_raw = t_row["md_path"] or t_row["csv_path"] or ""
                    target_p = Path(target_raw).resolve().as_posix()
                    if target_p and not target_p.startswith("/"):
                        target_p = f"/{target_p}"
                    cap = (t_row["caption"] or "Tabel").strip()
                    cit = self._resolve_paper_citation(t_row["paper_id"])
                    cit_line = f"  \n{cit}" if cit else ""
                    return f"\n\n[📋 Buka Tabel](file://{target_p})  \n{cap}{cit_line}\n\n"

                # 3. Fuzzy search by filename or partial id
                cur.execute("SELECT file_path, caption, paper_id FROM images WHERE file_path LIKE ? OR id LIKE ? LIMIT 1", (f"%{v_id}%", f"%{v_id}%"))
                f_row = cur.fetchone()
                if f_row:
                    raw_fp = Path(f_row["file_path"]).resolve().as_posix()
                    if not raw_fp.startswith("/"):
                        raw_fp = f"/{raw_fp}"
                    cap = (f_row["caption"] or "Gambar").strip()
                    cit = self._resolve_paper_citation(f_row["paper_id"])
                    cit_line = f"  \n{cit}" if cit else ""
                    return f"\n\n[📊 Buka Gambar](file://{raw_fp})  \n{cap}{cit_line}\n\n"

            return f"[📊 Visual: {v_id}]"

        return pattern.sub(_repl, text)
