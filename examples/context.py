#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
context.py — Clause-aware, hybrid RAG with optional PDF page vision via Ollama.
Backward-compatible schema migrations (safe on existing knowledge.db).

What this provides
------------------
• KnowledgeStore:
  - SQLite: knowledge.db (WAL, foreign_keys=ON, busy_timeout)
  - Tables:
      documents(id, path, checksum, title, type, size, mtime, tags_json, added_ts, updated_ts, status, meta_json)
      doc_tags(doc_id, tag)
      clauses(id, doc_id, clause_code, title, text, page_start, page_end)
      chunks(id, doc_id, chunk_index, text, page_no, ts,
             clause_code, heading, chunk_idx, prev_chunk_id, next_chunk_id,
             page_start, page_end, modality, resource_path)
      chunk_vectors(chunk_id, embedding BLOB, dim, norm)
      chunks_fts(chunk_id, content, clause_code)  -- FTS5 (optional)
      invocations(id, session_id, query, ts, meta_json)
      invocation_chunks(invocation_id, chunk_id, relation)
  - Automatic schema migrations (ALTER TABLE / CREATE IF NOT EXISTS)
  - FTS5 is optional; if unavailable, retrieval gracefully degrades.

• Ingestion:
  - Scans ./data for PDFs, MD, TXT, HTML, DOCX
  - Robust text extraction (PyMuPDF for PDF; bs4; docx; chardet for TXT; MD front-matter)
  - Clause/header detection (e.g., "7.5.3 Control of documented information")
  - Clause-aware chunking with neighbor links (prev/next) and page ranges
  - Optional: Render each PDF page to PNG and (optionally) run LLM vision (e.g., llava) to extract text
    - Adds "image" modality chunks referencing the page image path
  - Force re-ingest support; zero-chunk safeguard (mark error so we retry later)

• Retrieval:
  - Vector similarity over chunks (cosine)
  - FTS5 full-text keyword search over chunk text (and clause_code) when available
  - search_with_bleed(): reciprocal-rank-fusion (RRF) of vector + FTS results + ±N neighbor expansion
  - Rich snippets with clause + page range provenance
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# External deps (installed by main.py)
import chardet  # type: ignore
import docx  # python-docx  type: ignore
import fitz  # PyMuPDF  type: ignore
import ollama  # type: ignore
import yaml  # PyYAML  type: ignore
from bs4 import BeautifulSoup  # type: ignore

from cognition import CognitionConfig

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
IMAGES_DIR = DATA_DIR / ".page_images"
DB_PATH_DEFAULT = SCRIPT_DIR / "knowledge.db"

ALLOWED_EXTS = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm", ".docx"}

CLAUSE_RE = re.compile(r"^\s*(?P<code>\d+(?:\.\d+)+)\s+(?P<title>[^\n]+?)\s*$")

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

def _cosine(a: Sequence[float], b: Sequence[float], norm_a: Optional[float] = None, norm_b: Optional[float] = None) -> float:
    if len(a) != len(b):
        return -1.0
    na = norm_a or _l2_norm(a)
    nb = norm_b or _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    return dot / (na * nb)

def _embed_text_batch(texts: List[str], model: str) -> List[List[float]]:
    """
    Batch embeddings via ollama. Some versions accept list prompts; otherwise fall back to sequential.
    """
    try:
        out = ollama.embeddings(model=model, prompt=texts)
        if "embeddings" in out and isinstance(out["embeddings"], list):
            return [list(map(float, e)) for e in out["embeddings"]]
    except Exception:
        pass
    vecs = []
    for t in texts:
        emb = ollama.embeddings(model=model, prompt=t.strip())
        if "embedding" in emb:
            vecs.append([float(x) for x in emb["embedding"]])
        elif "embeddings" in emb and emb["embeddings"]:
            vecs.append([float(x) for x in emb["embeddings"][0]])
        else:
            raise RuntimeError("Unexpected embeddings response from Ollama")
    return vecs

# ─────────────────────────────────────────────────────────────────────────────
# Parsing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return raw.decode(enc, errors="replace")

def _parse_markdown(path: Path) -> Tuple[str, Dict[str, Any]]:
    txt = _read_text_file(path)
    meta: Dict[str, Any] = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", txt, flags=re.DOTALL)
    if m:
        front = m.group(1)
        body = m.group(2)
        try:
            fm = yaml.safe_load(front) or {}
            if isinstance(fm, dict):
                meta.update(fm)
        except Exception:
            pass
        txt = body
    txt = re.sub(r"`{1,3}.*?`{1,3}", " ", txt, flags=re.DOTALL)
    txt = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", txt)
    txt = re.sub(r"\[[^\]]*\]\([^)]+\)", lambda m: m.group(0).split("]")[0][1:], txt)
    txt = re.sub(r"[#>*_~`]+", " ", txt)
    txt = re.sub(r"\s+\n", "\n", txt)
    return txt.strip(), meta

def _parse_pdf(path: Path) -> Tuple[str, Dict[str, Any], List[Tuple[int, str]]]:
    """
    Robust text extraction with PyMuPDF. Returns:
      - full_text
      - meta (title, keywords)
      - page_texts: list of (page_no (1-based), text)
    """
    doc = fitz.open(path)
    meta = doc.metadata or {}
    title = meta.get("title") or path.stem
    keywords = meta.get("keywords") or ""
    out_pages: List[Tuple[int, str]] = []
    full = []
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
            text = re.sub(r"\s+\n", "\n", text)
            out_pages.append((i, text.strip()))
            full.append(text)
        except Exception:
            out_pages.append((i, ""))
    doc.close()
    meta_norm = {
        "title": title,
        "keywords": keywords,
        "producer": meta.get("producer"),
        "author": meta.get("author"),
        "subject": meta.get("subject"),
        "creator": meta.get("creator"),
    }
    return "\n\n".join(full).strip(), meta_norm, out_pages

def _parse_docx(path: Path) -> Tuple[str, Dict[str, Any]]:
    d = docx.Document(str(path))
    lines = []
    for p in d.paragraphs:
        txt = p.text.strip()
        if txt:
            lines.append(txt)
    core = d.core_properties
    meta = {
        "title": core.title or path.stem,
        "author": core.author,
        "subject": core.subject,
        "keywords": core.keywords,
    }
    return "\n".join(lines), meta

def _parse_html(path: Path) -> Tuple[str, Dict[str, Any]]:
    raw = _read_text_file(path)
    soup = BeautifulSoup(raw, "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else path.stem)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    return text, {"title": title}

def _parse_any(path: Path) -> Tuple[str, Dict[str, Any], Optional[List[Tuple[int, str]]]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        full, meta, pages = _parse_pdf(path)
        return full, meta, pages
    elif ext in {".md", ".markdown"}:
        txt, meta = _parse_markdown(path)
        meta.setdefault("title", path.stem)
        return txt, meta, None
    elif ext == ".docx":
        txt, meta = _parse_docx(path)
        meta.setdefault("title", path.stem)
        return txt, meta, None
    elif ext in {".html", ".htm"}:
        txt, meta = _parse_html(path)
        meta.setdefault("title", path.stem)
        return txt, meta, None
    else:
        txt = _read_text_file(path)
        return txt, {"title": path.stem}, None

# ─────────────────────────────────────────────────────────────────────────────
# Clause detection & chunking
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]

def _chunk_text_by_chars(paragraphs: List[str], max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if cur + len(p) + 1 <= max_chars:
            buf.append(p)
            cur += len(p) + 1
        else:
            if buf:
                chunk = "\n".join(buf).strip()
                if chunk:
                    chunks.append(chunk)
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                buf = [tail, p]
                cur = len(tail) + 1 + len(p)
            else:
                buf = [p]
                cur = len(p)
    if buf:
        chunk = "\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def _detect_clauses(full_text: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (clause_code, title, clause_body_text).
    If no clauses found, returns a single pseudo-clause ("", "", full_text).
    """
    lines = [ln.rstrip() for ln in full_text.splitlines()]
    spans: List[Tuple[int, str, str]] = []  # (line_idx, code, title)
    for i, ln in enumerate(lines):
        m = CLAUSE_RE.match(ln)
        if m:
            code = m.group("code").strip()
            title = m.group("title").strip()
            spans.append((i, code, title))
    if not spans:
        return [("", "", full_text)]

    result: List[Tuple[str, str, str]] = []
    for j, (start_i, code, title) in enumerate(spans):
        end_i = (spans[j + 1][0] if j + 1 < len(spans) else len(lines))
        body = "\n".join(lines[start_i + 1 : end_i]).strip()
        header = lines[start_i].strip()
        body2 = (header + "\n" + body).strip()
        result.append((code, title, body2))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Vision (PDF page image rendering + LLM extraction)
# ─────────────────────────────────────────────────────────────────────────────

def _render_pdf_pages(path: Path) -> List[Tuple[int, Path]]:
    """
    Render pages to PNG files inside DATA_DIR/.page_images/<docstem>/p{n}.png
    Returns list of (page_no, image_path)
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = IMAGES_DIR / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    page_imgs: List[Tuple[int, Path]] = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        pm = page.get_pixmap(dpi=150)
        p = out_dir / f"p{i}.png"
        pm.save(str(p))
        page_imgs.append((i, p))
    doc.close()
    return page_imgs

def _vision_extract_image_text(image_path: Path, *, vision_model: str, prompt: Optional[str] = None, max_chars: int = 6000) -> str:
    """
    Use an Ollama multimodal model (e.g., 'llava') to extract page text / gist.
    """
    system = "You are an OCR/IE assistant. Extract readable text and important tables as plain text. Keep order. If low quality, summarize key points."
    user = prompt or "Extract the text content of this page. Include lists and tables as text."

    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        pieces: List[str] = []
        print(f"[vision:{vision_model}]", end=" ", flush=True)
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
                print(delta, end="", flush=True)
                pieces.append(delta)
        print("", flush=True)
        content = "".join(pieces).strip()
        if len(content) > max_chars:
            content = content[:max_chars]
        return content
    except Exception:
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkRow:
    id: str
    doc_id: str
    chunk_index: int
    text: str
    page_no: Optional[int]
    score: Optional[float] = None
    title: Optional[str] = None
    path: Optional[str] = None
    tags: Optional[List[str]] = None
    clause_code: Optional[str] = None
    heading: Optional[str] = None
    chunk_idx: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    modality: Optional[str] = None
    resource_path: Optional[str] = None


@dataclass
class RetrievalReport:
    """Human-readable and machine-friendly summary of a retrieval session."""

    query: str
    top_k: int
    bleed: int
    total_chunks: int
    unique_documents: int
    documents: List[Dict[str, Any]] = field(default_factory=list)
    top_chunks: List[Dict[str, Any]] = field(default_factory=list)
    ingestion: Dict[str, Any] = field(default_factory=dict)
    generated_ts: int = field(default_factory=lambda: int(time.time()))

    @classmethod
    def from_chunks(
        cls,
        query: str,
        chunks: Sequence[ChunkRow],
        *,
        top_k: int,
        bleed: int,
        ingestion: Optional[Dict[str, Any]] = None,
    ) -> "RetrievalReport":
        doc_map: Dict[str, Dict[str, Any]] = {}
        ordered_chunks = sorted(
            [c for c in chunks],
            key=lambda c: (c.score if c.score is not None else -1.0),
            reverse=True,
        )

        for chunk in ordered_chunks:
            doc_entry = doc_map.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title or "(untitled document)",
                    "path": chunk.path,
                    "chunk_count": 0,
                    "scores": [],
                    "top_chunk_id": chunk.id,
                    "top_chunk_score": chunk.score,
                    "sample_clause": chunk.clause_code,
                    "page": chunk.page_no or chunk.page_start,
                },
            )
            doc_entry["chunk_count"] += 1
            if chunk.score is not None:
                doc_entry["scores"].append(chunk.score)
                top_score = doc_entry.get("top_chunk_score")
                if top_score is None or chunk.score > top_score:
                    doc_entry["top_chunk_id"] = chunk.id
                    doc_entry["top_chunk_score"] = chunk.score
                    doc_entry["sample_clause"] = chunk.clause_code
                    doc_entry["page"] = chunk.page_no or chunk.page_start

        documents: List[Dict[str, Any]] = []
        for entry in doc_map.values():
            scores = entry.pop("scores", [])
            agg = entry
            if scores:
                agg["avg_score"] = sum(scores) / len(scores)
                agg["max_score"] = max(scores)
                agg["min_score"] = min(scores)
            else:
                agg["avg_score"] = None
                agg["max_score"] = None
                agg["min_score"] = None
            documents.append(agg)

        documents.sort(key=lambda d: d.get("max_score") or 0.0, reverse=True)

        top_chunks: List[Dict[str, Any]] = []
        for chunk in ordered_chunks[: min(len(ordered_chunks), 8)]:
            top_chunks.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.title or "(untitled document)",
                    "score": chunk.score,
                    "page": chunk.page_no or chunk.page_start,
                    "clause": chunk.clause_code,
                }
            )

        return cls(
            query=query,
            top_k=top_k,
            bleed=bleed,
            total_chunks=len(ordered_chunks),
            unique_documents=len(documents),
            documents=documents,
            top_chunks=top_chunks,
            ingestion=dict(ingestion or {}),
        )

    def render(self, *, max_docs: int = 5, max_chunks: int = 5) -> str:
        def _fmt(score: Optional[float]) -> str:
            return f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"

        lines = [
            f"Query: {self.query}",
            (
                f"Retrieved {self.total_chunks} chunks from {self.unique_documents} documents "
                f"(top_k={self.top_k}, bleed={self.bleed})"
            ),
        ]
        if self.ingestion:
            opts = ", ".join(f"{k}={v}" for k, v in sorted(self.ingestion.items()))
            lines.append(f"Ingest options: {opts}")

        if self.documents:
            lines.append("Top documents:")
            for doc in self.documents[:max_docs]:
                lines.append(
                    "  - "
                    + f"{doc.get('title')} (chunks={doc.get('chunk_count')}, max={_fmt(doc.get('max_score'))}, avg={_fmt(doc.get('avg_score'))})"
                )
        if self.top_chunks:
            lines.append("Representative chunks:")
            for chunk in self.top_chunks[:max_chunks]:
                lines.append(
                    "  - "
                    + f"#{chunk['chunk_id'][:8]} score={_fmt(chunk.get('score'))} "
                    + f"{chunk.get('doc_title')} (page={chunk.get('page')}, clause={chunk.get('clause') or 'n/a'})"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeStore
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeStore:
    def __init__(
        self,
        db_path: Path | str = DB_PATH_DEFAULT,
        data_dir: Path | str = DATA_DIR,
        *,
        embed_model: Optional[str] = None,
        busy_timeout_ms: int = 6000,
    ) -> None:
        self.db_path = str(db_path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        cfg = CognitionConfig.load()
        self.embed_model = (embed_model or cfg.embed_model or "nomic-embed-text")
        self.vision_model = getattr(cfg, "vision_model", None) or os.environ.get("OLLAMA_VISION_MODEL") or "gemma3:4b"

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
        self._fts_ok = False
        self._ensure_schema()

        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()

    # ── schema (with safe migrations) ─────────────────────
    def _ensure_schema(self) -> None:
        """
        Create base tables first; then add/migrate new columns; finally create
        indexes that reference new columns, and set up FTS5 (if available).
        This order guarantees compatibility with older DBs.
        """
        with self._conn:
            # 1) Base tables (no references to new columns)
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    checksum TEXT NOT NULL,
                    title TEXT NOT NULL,
                    type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime INTEGER NOT NULL,
                    tags_json TEXT,
                    added_ts INTEGER NOT NULL,
                    updated_ts INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    meta_json TEXT
                );
                CREATE TABLE IF NOT EXISTS doc_tags (
                    doc_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (doc_id, tag),
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS clauses (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    clause_code TEXT,
                    title TEXT,
                    text TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_no INTEGER,
                    ts INTEGER NOT NULL,
                    -- newer columns may be absent in older DBs (added via ALTER below)
                    clause_code TEXT,
                    heading TEXT,
                    chunk_idx INTEGER,
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    modality TEXT,
                    resource_path TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    norm REAL NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS invocations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    meta_json TEXT
                );
                CREATE TABLE IF NOT EXISTS invocation_chunks (
                    invocation_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    PRIMARY KEY (invocation_id, chunk_id, relation),
                    FOREIGN KEY (invocation_id) REFERENCES invocations(id) ON DELETE CASCADE,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_docs_title ON documents(title);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_idx ON chunks(doc_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_chunks_ts ON chunks(ts);
                """
            )

        # 2) Ensure new columns exist (older DBs won't have them)
        self._ensure_column("chunks", "clause_code", "TEXT")
        self._ensure_column("chunks", "heading", "TEXT")
        self._ensure_column("chunks", "chunk_idx", "INTEGER")
        self._ensure_column("chunks", "prev_chunk_id", "TEXT")
        self._ensure_column("chunks", "next_chunk_id", "TEXT")
        self._ensure_column("chunks", "page_start", "INTEGER")
        self._ensure_column("chunks", "page_end", "INTEGER")
        self._ensure_column("chunks", "modality", "TEXT")
        self._ensure_column("chunks", "resource_path", "TEXT")

        # 3) Indexes that reference new columns — create AFTER migration
        try:
            with self._conn:
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_clause ON chunks(doc_id, clause_code);")
        except sqlite3.OperationalError:
            # If clause_code still somehow missing, skip index silently
            pass

        # 4) FTS5 virtual table (optional; skip if not compiled)
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, content, clause_code, tokenize='porter');"
                )
            self._fts_ok = True
        except sqlite3.OperationalError:
            # FTS5 not available in this SQLite build — we’ll gracefully degrade
            self._fts_ok = False

    def _ensure_column(self, table: str, col: str, decl: str) -> None:
        cur = self._conn.execute(f"PRAGMA table_info({table});").fetchall()
        names = {r["name"] for r in cur}
        if col not in names:
            try:
                with self._conn:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl};")
            except sqlite3.OperationalError:
                # Older SQLite may complain for virtual tables etc.; safe to ignore here
                pass

    # ── ingestion orchestration ───────────────────────────
    def list_candidate_files(self) -> List[Path]:
        files: List[Path] = []
        for p in self.data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                files.append(p)
        return files

    def ingest_all_in_data(
        self,
        *,
        concurrent: bool = True,
        max_workers: int = 4,
        force: bool = False,
        vision: bool = False,
        vision_model: Optional[str] = None
    ) -> Dict[str, int]:
        files = self.list_candidate_files()
        stats = {"total": len(files), "ingested": 0, "skipped": 0, "failed": 0}
        if not files:
            return stats

        if not concurrent:
            for f in files:
                ok = self.ingest_path(f, force=force, vision=vision, vision_model=vision_model)
                if ok is True:
                    stats["ingested"] += 1
                elif ok is None:
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
            return stats

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.ingest_path, f, force, vision, vision_model): f for f in files}
            for fut in as_completed(futs):
                res = fut.result()
                if res is True:
                    stats["ingested"] += 1
                elif res is None:
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
        return stats

    def start_auto_ingest_watcher(
        self,
        interval_sec: int = 30,
        *,
        concurrent: bool = True,
        max_workers: int = 4,
        force: bool = False,
        vision: bool = False,
        vision_model: Optional[str] = None
    ) -> None:
        """
        Periodically rescan ./data and ingest new/changed files.
        """
        if self._watcher_thread and self._watcher_thread.is_alive():
            return

        self._watcher_stop.clear()

        def _loop():
            while not self._watcher_stop.is_set():
                try:
                    self.ingest_all_in_data(concurrent=concurrent, max_workers=max_workers, force=force, vision=vision, vision_model=vision_model)
                except Exception:
                    pass
                self._watcher_stop.wait(interval_sec)

        t = threading.Thread(target=_loop, daemon=True, name="KnowledgeStoreWatcher")
        t.start()
        self._watcher_thread = t

    def stop_auto_ingest_watcher(self) -> None:
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_stop.set()
            self._watcher_thread.join(timeout=2.0)

    # ── per-file ingestion ─────────────────────────────────
    def ingest_path(self, path: Path, force: bool = False, vision: bool = False, vision_model: Optional[str] = None) -> Optional[bool]:
        """
        Returns:
          True  -> ingested/updated
          None  -> skipped (unchanged)
          False -> failed
        """
        try:
            path = Path(path).resolve()
            if not path.is_file():
                return False
            if path.suffix.lower() not in ALLOWED_EXTS:
                return False

            checksum = _sha256_file(path)
            stat = path.stat()
            size = stat.st_size
            mtime = int(stat.st_mtime)
            ftype = path.suffix.lower().lstrip(".")

            # Check existing
            row = self._conn.execute("SELECT id, checksum FROM documents WHERE path = ?", (str(path),)).fetchone()
            if row and row["checksum"] == checksum and not force:
                return None  # unchanged -> skip

            # Parse content
            full_text, meta, page_texts = _parse_any(path)
            title = (meta.get("title") if isinstance(meta, dict) else None) or path.stem
            tags = set()
            tags.add(ftype)
            for seg in path.relative_to(self.data_dir).parts[:-1]:
                if seg:
                    tags.add(seg.lower())
            if isinstance(meta, dict):
                for k in ("keywords", "tags"):
                    v = meta.get(k)
                    if isinstance(v, str):
                        for t in re.split(r"[;,]\s*|\s+", v):
                            if t:
                                tags.add(t.lower())
                    elif isinstance(v, list):
                        for t in v:
                            if isinstance(t, str) and t.strip():
                                tags.add(t.strip().lower())

            # Upsert document record
            now = _now_ts()
            doc_id = (row["id"] if row else _new_id())
            with self._lock, self._conn:
                if row:
                    self._conn.execute(
                        """UPDATE documents
                           SET checksum=?, title=?, type=?, size=?, mtime=?, tags_json=?, updated_ts=?, status=?, meta_json=?
                           WHERE id=?""",
                        (checksum, title, ftype, size, mtime, json.dumps(sorted(tags)), now, "ok", json.dumps(meta or {}), doc_id),
                    )
                    # clear old chunks/vectors/clauses/fts
                    self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                    self._conn.execute("DELETE FROM chunk_vectors WHERE chunk_id NOT IN (SELECT id FROM chunks)")
                    self._conn.execute("DELETE FROM clauses WHERE doc_id = ?", (doc_id,))
                    try:
                        self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id NOT IN (SELECT id FROM chunks)")
                    except sqlite3.OperationalError:
                        pass  # FTS may not exist
                else:
                    self._conn.execute(
                        """INSERT INTO documents (id, path, checksum, title, type, size, mtime, tags_json, added_ts, updated_ts, status, meta_json)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (doc_id, str(path), checksum, title, ftype, size, mtime, json.dumps(sorted(tags)), now, now, "ok", json.dumps(meta or {})),
                    )
                # refresh tags table
                self._conn.execute("DELETE FROM doc_tags WHERE doc_id=?", (doc_id,))
                for t in sorted(tags):
                    self._conn.execute("INSERT OR IGNORE INTO doc_tags (doc_id, tag) VALUES (?,?)", (doc_id, t))

            # Clause detection
            clauses = _detect_clauses(full_text)
            clause_rows: List[Tuple[str, str, str]] = clauses  # (code, title, body)

            # Insert clauses
            with self._lock, self._conn:
                for code, ctitle, body in clause_rows:
                    cid = _new_id()
                    self._conn.execute(
                        "INSERT INTO clauses (id, doc_id, clause_code, title, text, page_start, page_end) VALUES (?,?,?,?,?,?,?)",
                        (cid, doc_id, code or None, ctitle or None, body, None, None),
                    )

            # Build text chunks (clause-aware when available)
            chunks_to_insert: List[Dict[str, Any]] = []
            global_idx = 0

            if clause_rows and not (len(clause_rows) == 1 and clause_rows[0][0] == "" and clause_rows[0][1] == ""):
                for code, ctitle, body in clause_rows:
                    paras = _split_into_paragraphs(body)
                    sub = _chunk_text_by_chars(paras, max_chars=1200, overlap=160)
                    for local_idx, s in enumerate(sub):
                        chunks_to_insert.append({
                            "id": _new_id(),
                            "doc_id": doc_id,
                            "chunk_index": global_idx,
                            "text": s,
                            "page_no": None,
                            "ts": now,
                            "clause_code": code or None,
                            "heading": ctitle or None,
                            "chunk_idx": local_idx,
                            "page_start": None,
                            "page_end": None,
                            "modality": "text",
                            "resource_path": None,
                        })
                        global_idx += 1
            else:
                if page_texts:
                    for page_no, ptxt in page_texts:
                        paras = _split_into_paragraphs(ptxt)
                        sub = _chunk_text_by_chars(paras, max_chars=1200, overlap=160)
                        for local_idx, s in enumerate(sub):
                            chunks_to_insert.append({
                                "id": _new_id(),
                                "doc_id": doc_id,
                                "chunk_index": global_idx,
                                "text": s,
                                "page_no": page_no,
                                "ts": now,
                                "clause_code": None,
                                "heading": None,
                                "chunk_idx": local_idx,
                                "page_start": page_no,
                                "page_end": page_no,
                                "modality": "text",
                                "resource_path": None,
                            })
                            global_idx += 1
                else:
                    paras = _split_into_paragraphs(full_text)
                    sub = _chunk_text_by_chars(paras, max_chars=1200, overlap=160)
                    for local_idx, s in enumerate(sub):
                        chunks_to_insert.append({
                            "id": _new_id(),
                            "doc_id": doc_id,
                            "chunk_index": global_idx,
                            "text": s,
                            "page_no": None,
                            "ts": now,
                            "clause_code": None,
                            "heading": None,
                            "chunk_idx": local_idx,
                            "page_start": None,
                            "page_end": None,
                            "modality": "text",
                            "resource_path": None,
                        })
                        global_idx += 1

            # Optional PDF page vision → add image-modality chunks
            if vision and path.suffix.lower() == ".pdf":
                try:
                    vmodel = vision_model or self.vision_model or "gemma3:4b"
                    page_imgs = _render_pdf_pages(path)
                    for page_no, img_path in page_imgs:
                        llm_text = _vision_extract_image_text(img_path, vision_model=vmodel)
                        if llm_text:
                            chunks_to_insert.append({
                                "id": _new_id(),
                                "doc_id": doc_id,
                                "chunk_index": global_idx,
                                "text": llm_text,
                                "page_no": page_no,
                                "ts": now,
                                "clause_code": None,
                                "heading": "Vision-Extracted Page Text",
                                "chunk_idx": 0,
                                "page_start": page_no,
                                "page_end": page_no,
                                "modality": "image",
                                "resource_path": str(img_path),
                            })
                            global_idx += 1
                except Exception:
                    pass  # best-effort

            if not chunks_to_insert:
                with self._lock, self._conn:
                    self._conn.execute("UPDATE documents SET status=?, updated_ts=? WHERE id=?", ("error", now, doc_id))
                return False

            # Insert chunks + embeddings + FTS; fill prev/next after insert
            texts = [c["text"] for c in chunks_to_insert]
            vecs = _embed_text_batch(texts, self.embed_model) if texts else []

            with self._lock, self._conn:
                for c, vec in zip(chunks_to_insert, vecs):
                    self._conn.execute(
                        """INSERT INTO chunks
                           (id, doc_id, chunk_index, text, page_no, ts, clause_code, heading, chunk_idx,
                            prev_chunk_id, next_chunk_id, page_start, page_end, modality, resource_path)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (c["id"], c["doc_id"], c["chunk_index"], c["text"], c["page_no"], c["ts"],
                         c["clause_code"], c["heading"], c["chunk_idx"], None, None,
                         c["page_start"], c["page_end"], c["modality"], c["resource_path"]),
                    )
                    blob = _to_blob_f32(vec)
                    norm = _l2_norm(vec)
                    self._conn.execute(
                        "INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding, dim, norm) VALUES (?,?,?,?)",
                        (c["id"], blob, len(vec), norm),
                    )
                    # FTS upsert (if available)
                    if self._fts_ok:
                        try:
                            self._conn.execute(
                                "INSERT INTO chunks_fts (chunk_id, content, clause_code) VALUES (?,?,?)",
                                (c["id"], c["text"], c["clause_code"] or "")
                            )
                        except sqlite3.OperationalError:
                            self._fts_ok = False  # disable further attempts
                            pass

            # prev/next linking by (doc_id, chunk_index) order
            with self._lock, self._conn:
                rows = self._conn.execute(
                    "SELECT id FROM chunks WHERE doc_id=? ORDER BY chunk_index ASC",
                    (doc_id,)
                ).fetchall()
                for i, r in enumerate(rows):
                    prev_id = rows[i - 1]["id"] if i > 0 else None
                    next_id = rows[i + 1]["id"] if i + 1 < len(rows) else None
                    self._conn.execute(
                        "UPDATE chunks SET prev_chunk_id=?, next_chunk_id=? WHERE id=?",
                        (prev_id, next_id, r["id"])
                    )

            return True

        except Exception:
            try:
                with self._lock, self._conn:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO documents (id, path, checksum, title, type, size, mtime, tags_json, added_ts, updated_ts, status, meta_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (_new_id(), str(path), "error", path.stem, path.suffix.lstrip("."), 0, 0, "[]",
                         _now_ts(), _now_ts(), "error", json.dumps({"error": "ingest failure"}))
                    )
            except Exception:
                pass
            return False

    # ── retrieval ───────────────────────────────────────────
    def _doc_meta(self, doc_id: str) -> Tuple[str, str, List[str]]:
        r = self._conn.execute("SELECT title, path, tags_json FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not r:
            return "", "", []
        return r["title"] or "", r["path"] or "", json.loads(r["tags_json"] or "[]")

    def search(
        self,
        *,
        query_text: str,
        k: int = 8,
        tags: Optional[Iterable[str]] = None,
    ) -> List[ChunkRow]:
        where = []
        params: List[Any] = []
        tag_join = ""
        if tags:
            tag_join = "JOIN doc_tags t ON t.doc_id = d.id"
            where.append("t.tag IN (%s)" % ",".join("?" for _ in tags))
            params.extend([t for t in tags])

        sql = f"""
            SELECT c.*, d.title AS dtitle, d.path AS dpath, d.tags_json AS dtags, v.embedding, v.dim, v.norm
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            JOIN chunk_vectors v ON v.chunk_id = c.id
            {tag_join}
            {"WHERE " + " AND ".join(where) if where else ""}
            ORDER BY c.ts DESC
            LIMIT ?
        """
        params.append(max(k * 12, 200))
        rows = self._conn.execute(sql, params).fetchall()

        qvec = _embed_text_batch([query_text], self.embed_model)[0]
        qnorm = _l2_norm(qvec)

        scored: List[ChunkRow] = []
        for r in rows:
            vec = _from_blob_f32(r["embedding"])
            score = _cosine(qvec, vec, qnorm, r["norm"])
            scored.append(
                ChunkRow(
                    id=r["id"],
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    page_no=r["page_no"],
                    score=score,
                    title=r["dtitle"],
                    path=r["dpath"],
                    tags=json.loads(r["dtags"] or "[]"),
                    clause_code=r["clause_code"],
                    heading=r["heading"],
                    chunk_idx=r["chunk_idx"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    modality=r["modality"],
                    resource_path=r["resource_path"],
                )
            )
        scored.sort(key=lambda x: (x.score if x.score is not None else -1e9, -x.chunk_index), reverse=True)
        return scored[:k]

    def search_with_bleed(
        self,
        query_text: str,
        *,
        k: int = 8,
        bleed: int = 1,
        tags: Optional[Iterable[str]] = None,
    ) -> List[ChunkRow]:
        """
        Hybrid retrieval:
          - Vector similarity (as above)
          - FTS5 keyword over chunks_fts (if available)
          - Reciprocal Rank Fusion
          - Expand ±bleed neighbors within same document
        """
        base = self.search(query_text=query_text, k=max(k, 8), tags=tags)  # vector
        base_ids = [b.id for b in base]

        # FTS (optional)
        fts_ids: List[str] = []
        if self._fts_ok:
            try:
                cur = self._conn.execute(
                    "SELECT rowid, chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
                    (query_text, max(k, 8))
                )
                fts_ids = [r["chunk_id"] for r in cur.fetchall()]
            except sqlite3.OperationalError:
                self._fts_ok = False
                fts_ids = []

        # RRF
        def rrf(rank: int) -> float:
            return 1.0 / (60.0 + rank)

        scores: Dict[str, float] = {}
        for i, cid in enumerate(base_ids, 1):
            scores[cid] = scores.get(cid, 0.0) + rrf(i)
        for i, cid in enumerate(fts_ids, 1):
            scores[cid] = scores.get(cid, 0.0) + rrf(i)

        best_ids = [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        expand_ids = set(best_ids)

        if bleed > 0 and best_ids:
            q = self._conn.execute(
                "SELECT id, prev_chunk_id, next_chunk_id FROM chunks WHERE id IN ({})".format(
                    ",".join("?" * len(best_ids))
                ), tuple(best_ids)
            )
            rows = [dict(r) for r in q.fetchall()]
            for r in rows:
                for direction in ("prev_chunk_id", "next_chunk_id"):
                    steps = 0
                    cur_id = r[direction]
                    while cur_id and steps < bleed:
                        expand_ids.add(cur_id)
                        row2 = self._conn.execute("SELECT prev_chunk_id, next_chunk_id FROM chunks WHERE id=?", (cur_id,)).fetchone()
                        cur_id = (row2[direction] if row2 else None)
                        steps += 1

        if not expand_ids:
            return base[:k]

        placeholders = ",".join("?" * len(expand_ids))
        rows = self._conn.execute(
            f"""SELECT c.*, d.title AS dtitle, d.path AS dpath, d.tags_json AS dtags,
                       v.embedding, v.dim, v.norm
                FROM chunks c
                JOIN documents d ON d.id=c.doc_id
                JOIN chunk_vectors v ON v.chunk_id=c.id
                WHERE c.id IN ({placeholders})""",
            tuple(expand_ids)
        ).fetchall()

        qvec = _embed_text_batch([query_text], self.embed_model)[0]
        qnorm = _l2_norm(qvec)

        items: List[Tuple[float, ChunkRow]] = []
        # clause co-hits for a tiny cohesion bonus
        best_clauses = set()
        if best_ids:
            r = self._conn.execute(
                "SELECT clause_code FROM chunks WHERE id IN ({})".format(",".join("?" * len(best_ids))),
                tuple(best_ids)
            ).fetchall()
            best_clauses = {x["clause_code"] for x in r if x["clause_code"]}

        for r in rows:
            vec = _from_blob_f32(r["embedding"])
            cos = _cosine(qvec, vec, qnorm, r["norm"])
            cid = r["id"]
            base_rrf = scores.get(cid, 0.0)
            clause_bonus = 0.2 if (r["clause_code"] and r["clause_code"] in best_clauses) else 0.0
            score = base_rrf + cos * 0.5 + clause_bonus
            items.append((
                score,
                ChunkRow(
                    id=r["id"],
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    text=r["text"],
                    page_no=r["page_no"],
                    score=score,
                    title=r["dtitle"],
                    path=r["dpath"],
                    tags=json.loads(r["dtags"] or "[]"),
                    clause_code=r["clause_code"],
                    heading=r["heading"],
                    chunk_idx=r["chunk_idx"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                    modality=r["modality"],
                    resource_path=r["resource_path"],
                )
            ))
        items.sort(key=lambda x: x[0], reverse=True)
        return [it[1] for it in items[:k]]

    def build_context_snippets(self, chunks: List[ChunkRow]) -> Tuple[List[str], List[str]]:
        """
        Turn chunk rows into human-usable context snippets with inline citations.
        Includes clause code, page range, modality, and short id.
        """
        snips: List[str] = []
        ids: List[str] = []
        for c in chunks:
            page_part = ""
            if c.page_start and c.page_end and c.page_start != c.page_end:
                page_part = f"p.{c.page_start}-{c.page_end}"
            elif c.page_no:
                page_part = f"p.{c.page_no}"
            elif c.page_start:
                page_part = f"p.{c.page_start}"
            clause_part = f" | clause {c.clause_code}" if c.clause_code else ""
            mod_part = f" | {c.modality}" if c.modality and c.modality != "text" else ""
            citation = f"[source: {c.title} {clause_part} | {page_part} | #{c.id[:8]} | score={c.score:.3f}{mod_part}]"
            body = c.text
            if c.resource_path and c.modality == "image":
                body = f"(image: {c.resource_path})\n" + body
            snips.append(f"{citation}\n{body}")
            ids.append(c.id)
        return snips, ids

    # ── invocation tracking ─────────────────────────────────
    def add_invocation(self, *, session_id: str, query: str, meta: Optional[Dict[str, Any]] = None) -> str:
        inv_id = _new_id()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO invocations (id, session_id, query, ts, meta_json) VALUES (?,?,?,?,?)",
                (inv_id, session_id, query, _now_ts(), json.dumps(meta or {}, ensure_ascii=False)),
            )
        return inv_id

    def link_invocation_chunk(self, invocation_id: str, chunk_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO invocation_chunks (invocation_id, chunk_id, relation) VALUES (?,?,?)",
                (invocation_id, chunk_id, "recalled"),
            )

    # ── misc ────────────────────────────────────────────────
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
