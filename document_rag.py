#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
document_rag.py — Document RAG system for telegram bot file uploads

Based on examples/context.py, provides:
- Multi-format document parsing (PDF, DOCX, TXT, MD, HTML)
- PDF vision extraction using Ollama multimodal models
- Vector embeddings for semantic search
- Hybrid retrieval (vector + keyword when FTS5 available)
- Chunk-based storage with provenance tracking
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# External dependencies
try:
    import chardet
    import fitz  # PyMuPDF
    import ollama
    from bs4 import BeautifulSoup
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    chardet = None  # type: ignore
    fitz = None  # type: ignore
    ollama = None  # type: ignore
    BeautifulSoup = None  # type: ignore

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    docx = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_ts() -> int:
    return int(time.time())


def _new_id() -> str:
    return str(uuid.uuid4())


def _sha256_file(path: Path, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _to_blob_f32(vec: Sequence[float]) -> bytes:
    arr = array("f", (float(x) for x in vec))
    return arr.tobytes()


def _from_blob_f32(blob: bytes) -> List[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


def _l2_norm(vec: Sequence[float]) -> float:
    return sum(x * x for x in vec) ** 0.5


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_text(text: str, model: str) -> List[float]:
    """Generate embedding for a single text using Ollama"""
    if not ollama:
        return []
    try:
        resp = ollama.embeddings(model=model, prompt=text)
        return resp.get("embedding", [])
    except Exception:
        return []


def _embed_text_batch(texts: List[str], model: str) -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    return [_embed_text(t, model) for t in texts]


def _vision_extract_image_text(
    image_path: Path,
    *,
    vision_model: str,
    prompt: Optional[str] = None,
    max_chars: int = 6000
) -> str:
    """
    Use Ollama multimodal model to extract text from image.

    Based on examples/context.py _vision_extract_image_text
    """
    if not ollama:
        return ""

    system = "You are an OCR/IE assistant. Extract readable text and important tables as plain text. Keep order. If low quality, summarize key points."
    user = prompt or "Extract the text content of this page. Include lists and tables as text."

    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        pieces: List[str] = []

        for chunk in ollama.chat(
            model=vision_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [b64]},
            ],
            stream=True,
            options={"temperature": 0.0},
        ):
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                pieces.append(delta)

        content = "".join(pieces).strip()
        if len(content) > max_chars:
            content = content[:max_chars]
        return content

    except Exception as e:
        print(f"[vision] Failed to extract image text: {e}")
        return ""


def _render_pdf_page_to_png(pdf_path: Path, page_no: int, output_dir: Path) -> Optional[Path]:
    """Render a single PDF page to PNG image"""
    if not fitz:
        return None

    try:
        doc = fitz.open(pdf_path)
        if page_no < 0 or page_no >= len(doc):
            return None

        page = doc[page_no]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom for better OCR

        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / f"{pdf_path.stem}_page_{page_no + 1}.png"

        pix.save(str(img_path))
        doc.close()

        return img_path

    except Exception as e:
        print(f"[pdf] Failed to render page {page_no}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A chunk of document text with metadata"""
    id: str
    doc_id: str
    text: str
    page_no: Optional[int] = None
    chunk_index: int = 0
    score: Optional[float] = None
    modality: str = "text"  # "text" or "image"
    resource_path: Optional[str] = None  # Path to image if modality=image


@dataclass
class RetrievalResult:
    """Result of a document retrieval query"""
    query: str
    chunks: List[DocumentChunk]
    total_found: int
    unique_documents: int


# ─────────────────────────────────────────────────────────────────────────────
# Document RAG Store
# ─────────────────────────────────────────────────────────────────────────────

class DocumentRAGStore:
    """
    Document RAG system with vector embeddings and hybrid retrieval.

    Based on examples/context.py KnowledgeStore
    """

    def __init__(
        self,
        db_path: Path,
        embed_model: str = "nomic-embed-text",
        vision_model: str = "gemma3:4b",
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize the document store.

        Args:
            db_path: Path to SQLite database
            embed_model: Ollama model for embeddings
            vision_model: Ollama model for vision tasks
            data_dir: Directory to watch for new documents
        """
        self.db_path = db_path
        self.embed_model = embed_model
        self.vision_model = vision_model
        self.data_dir = data_dir or (db_path.parent / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Image storage directory
        self.images_dir = self.data_dir / ".page_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row

        self._init_schema()

    def _init_schema(self):
        """Initialize database schema"""
        with self._conn:
            # Documents table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    checksum TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    user_id INTEGER,
                    chat_id INTEGER,
                    message_id INTEGER,
                    added_ts INTEGER,
                    status TEXT DEFAULT 'pending'
                )
            """)

            # Chunks table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER,
                    text TEXT NOT NULL,
                    page_no INTEGER,
                    modality TEXT DEFAULT 'text',
                    resource_path TEXT,
                    ts INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            # Vector embeddings
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER,
                    norm REAL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
            """)

            # Try to create FTS5 table (optional, gracefully fails if not available)
            try:
                self._conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(chunk_id, content, tokenize='porter unicode61')
                """)
                self.fts_available = True
            except Exception:
                self.fts_available = False

            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id)")

    def add_document_from_telegram(
        self,
        file_path: Path,
        user_id: int,
        chat_id: int,
        message_id: int,
        *,
        use_vision: bool = False,
    ) -> Optional[str]:
        """
        Ingest a document uploaded via Telegram.

        Args:
            file_path: Path to the downloaded file
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            message_id: Telegram message ID
            use_vision: Whether to use vision for PDF pages

        Returns:
            Document ID if successful, None otherwise
        """
        if not DEPS_AVAILABLE:
            print("[rag] Dependencies not available (pymupdf, chardet, bs4)")
            return None

        doc_id = _new_id()
        now = _now_ts()
        checksum = _sha256_file(file_path)
        file_type = file_path.suffix.lower()

        # Check if already ingested
        existing = self._conn.execute(
            "SELECT id FROM documents WHERE checksum = ?",
            (checksum,)
        ).fetchone()

        if existing:
            print(f"[rag] Document already exists: {existing['id']}")
            return existing["id"]

        # Insert document record
        with self._conn:
            self._conn.execute(
                """INSERT INTO documents
                   (id, filename, file_path, checksum, file_type, file_size,
                    user_id, chat_id, message_id, added_ts, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing')""",
                (doc_id, file_path.name, str(file_path), checksum, file_type,
                 file_path.stat().st_size, user_id, chat_id, message_id, now)
            )

        # Extract text and create chunks
        try:
            chunks = self._extract_chunks(file_path, doc_id, use_vision=use_vision)

            if not chunks:
                self._conn.execute(
                    "UPDATE documents SET status = 'error' WHERE id = ?",
                    (doc_id,)
                )
                return None

            # Insert chunks and embeddings
            self._insert_chunks(chunks)

            # Mark as complete
            self._conn.execute(
                "UPDATE documents SET status = 'ready' WHERE id = ?",
                (doc_id,)
            )

            print(f"[rag] Ingested {file_path.name}: {len(chunks)} chunks")
            return doc_id

        except Exception as e:
            print(f"[rag] Failed to ingest {file_path.name}: {e}")
            self._conn.execute(
                "UPDATE documents SET status = 'error' WHERE id = ?",
                (doc_id,)
            )
            return None

    def _extract_chunks(
        self,
        file_path: Path,
        doc_id: str,
        *,
        use_vision: bool = False,
        chunk_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Extract text chunks from a document"""
        chunks: List[Dict[str, Any]] = []
        file_type = file_path.suffix.lower()

        # Extract raw text based on file type
        if file_type == ".pdf":
            text, page_texts = self._extract_pdf_text(file_path)
        elif file_type == ".txt":
            text = self._extract_txt_text(file_path)
            page_texts = {0: text}
        elif file_type in [".html", ".htm"]:
            text = self._extract_html_text(file_path)
            page_texts = {0: text}
        elif file_type == ".docx" and DOCX_AVAILABLE:
            text = self._extract_docx_text(file_path)
            page_texts = {0: text}
        elif file_type in [".md", ".markdown"]:
            text = file_path.read_text(encoding="utf-8")
            page_texts = {0: text}
        else:
            return []

        # Split into chunks
        chunk_idx = 0
        for page_no, page_text in page_texts.items():
            # Simple chunking by character count
            for i in range(0, len(page_text), chunk_size):
                chunk_text = page_text[i:i + chunk_size].strip()
                if chunk_text:
                    chunks.append({
                        "id": _new_id(),
                        "doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "text": chunk_text,
                        "page_no": page_no if page_no > 0 else None,
                        "modality": "text",
                        "resource_path": None,
                        "ts": _now_ts(),
                    })
                    chunk_idx += 1

        # Optional: PDF vision extraction
        if use_vision and file_type == ".pdf" and fitz:
            try:
                doc = fitz.open(file_path)
                for page_no in range(len(doc)):
                    img_path = _render_pdf_page_to_png(file_path, page_no, self.images_dir)
                    if img_path:
                        vision_text = _vision_extract_image_text(
                            img_path,
                            vision_model=self.vision_model
                        )
                        if vision_text:
                            chunks.append({
                                "id": _new_id(),
                                "doc_id": doc_id,
                                "chunk_index": chunk_idx,
                                "text": vision_text,
                                "page_no": page_no + 1,
                                "modality": "image",
                                "resource_path": str(img_path),
                                "ts": _now_ts(),
                            })
                            chunk_idx += 1
                doc.close()
            except Exception as e:
                print(f"[rag] Vision extraction failed: {e}")

        return chunks

    def _extract_pdf_text(self, path: Path) -> Tuple[str, Dict[int, str]]:
        """Extract text from PDF, returning (full_text, {page_no: text})"""
        if not fitz:
            return "", {}

        doc = fitz.open(path)
        page_texts = {}
        all_text_parts = []

        for i, page in enumerate(doc):
            text = page.get_text()
            page_texts[i + 1] = text
            all_text_parts.append(text)

        doc.close()
        return "\n\n".join(all_text_parts), page_texts

    def _extract_txt_text(self, path: Path) -> str:
        """Extract text from plain text file with encoding detection"""
        if not chardet:
            return path.read_text(encoding="utf-8", errors="ignore")

        raw = path.read_bytes()
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        return raw.decode(encoding, errors="ignore")

    def _extract_html_text(self, path: Path) -> str:
        """Extract text from HTML file"""
        if not BeautifulSoup:
            return ""

        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        return soup.get_text(separator="\n", strip=True)

    def _extract_docx_text(self, path: Path) -> str:
        """Extract text from DOCX file"""
        if not docx:
            return ""

        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n\n".join(paragraphs)

    def _insert_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert chunks and their embeddings into the database"""
        texts = [c["text"] for c in chunks]
        embeddings = _embed_text_batch(texts, self.embed_model)

        with self._conn:
            for chunk, embedding in zip(chunks, embeddings):
                # Insert chunk
                self._conn.execute(
                    """INSERT INTO chunks
                       (id, doc_id, chunk_index, text, page_no, modality, resource_path, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk["id"], chunk["doc_id"], chunk["chunk_index"],
                     chunk["text"], chunk["page_no"], chunk["modality"],
                     chunk["resource_path"], chunk["ts"])
                )

                # Insert embedding
                if embedding:
                    blob = _to_blob_f32(embedding)
                    norm = _l2_norm(embedding)
                    self._conn.execute(
                        """INSERT INTO chunk_vectors (chunk_id, embedding, dim, norm)
                           VALUES (?, ?, ?, ?)""",
                        (chunk["id"], blob, len(embedding), norm)
                    )

                # Insert into FTS if available
                if self.fts_available:
                    try:
                        self._conn.execute(
                            "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                            (chunk["id"], chunk["text"])
                        )
                    except Exception:
                        pass

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        user_id: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Search for relevant chunks using hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            user_id: Optional filter by user ID

        Returns:
            RetrievalResult with matched chunks
        """
        # Generate query embedding
        query_emb = _embed_text(query, self.embed_model)

        # Get vector similarity results
        vector_results = self._vector_search(query_emb, top_k * 2, user_id)

        # Get keyword search results if FTS available
        keyword_results = []
        if self.fts_available:
            keyword_results = self._keyword_search(query, top_k * 2, user_id)

        # Merge results using Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(vector_results, keyword_results, top_k)

        # Convert to DocumentChunk objects
        chunks = []
        unique_docs = set()
        for chunk_id, score in merged:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE id = ?",
                (chunk_id,)
            ).fetchone()

            if row:
                chunks.append(DocumentChunk(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    page_no=row["page_no"],
                    chunk_index=row["chunk_index"],
                    score=score,
                    modality=row["modality"] or "text",
                    resource_path=row["resource_path"],
                ))
                unique_docs.add(row["doc_id"])

        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_found=len(chunks),
            unique_documents=len(unique_docs),
        )

    def _vector_search(
        self,
        query_emb: List[float],
        top_k: int,
        user_id: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Vector similarity search"""
        if not query_emb:
            return []

        # Get all chunk vectors
        if user_id:
            rows = self._conn.execute("""
                SELECT cv.chunk_id, cv.embedding
                FROM chunk_vectors cv
                JOIN chunks c ON c.id = cv.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE d.user_id = ? AND d.status = 'ready'
            """, (user_id,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT cv.chunk_id, cv.embedding
                FROM chunk_vectors cv
                JOIN chunks c ON c.id = cv.chunk_id
                JOIN documents d ON d.id = c.doc_id
                WHERE d.status = 'ready'
            """).fetchall()

        # Calculate similarities
        results = []
        for row in rows:
            chunk_emb = _from_blob_f32(row["embedding"])
            sim = _cosine_sim(query_emb, chunk_emb)
            results.append((row["chunk_id"], sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        user_id: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Keyword search using FTS5"""
        if not self.fts_available:
            return []

        try:
            if user_id:
                rows = self._conn.execute("""
                    SELECT f.chunk_id, f.rank
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.chunk_id
                    JOIN documents d ON d.id = c.doc_id
                    WHERE chunks_fts MATCH ? AND d.user_id = ? AND d.status = 'ready'
                    ORDER BY rank
                    LIMIT ?
                """, (query, user_id, top_k)).fetchall()
            else:
                rows = self._conn.execute("""
                    SELECT f.chunk_id, f.rank
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.chunk_id
                    JOIN documents d ON d.id = c.doc_id
                    WHERE chunks_fts MATCH ? AND d.status = 'ready'
                    ORDER BY rank
                    LIMIT ?
                """, (query, top_k)).fetchall()

            # Convert FTS5 rank (negative) to positive score
            return [(r["chunk_id"], -float(r["rank"])) for r in rows]

        except Exception:
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
        top_k: int,
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each ranking
        """
        scores: Dict[str, float] = {}

        # Add vector scores
        for rank, (chunk_id, _) in enumerate(vector_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        # Add keyword scores
        for rank, (chunk_id, _) in enumerate(keyword_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def close(self):
        """Close the database connection"""
        if self._conn:
            self._conn.close()
