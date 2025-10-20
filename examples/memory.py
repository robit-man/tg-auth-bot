#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory.py — Hybrid memory store (traditional + vector) for agent cognition.

Goals
-----
• Store user/assistant turns and arbitrary notes as "memories"
  - memories(id, session_id, role, content, ts, source)
  - memory_tags(memory_id, tag)
  - memory_vectors(memory_id, embedding BLOB, dim, norm)
• Support hybrid queries:
  - Similarity (Ollama embeddings + cosine)
  - Timestamp ranges
  - Tag filters
  - Session scoping
• Link every cognition "invocation" (a single call/mode) to:
  - Recalled memories used as context (relation='recalled')
  - Produced memories created by that invocation (relation='produced')
• Write protection:
  - SQLite WAL + busy_timeout + foreign_keys
  - Thread lock around writes

Notes
-----
• Embedding model is taken from config.json ("embed_model"), auto-created if missing.
• Embeddings are stored as float32 BLOBs (little-endian) with a cached L2 norm.
• Vector search is implemented in Python for portability (no extra extensions).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# External deps (installed by main.py bootstrap)
import ollama  # type: ignore
from array import array

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = SCRIPT_DIR / "memory.db"
CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _now_ts() -> int:
    return int(time.time())


def _new_id() -> str:
    return str(uuid.uuid4())


def _ensure_embed_model_in_config() -> str:
    """
    Ensure config.json exists and contains an 'embed_model' key.
    Returns the effective embed_model (possibly after persisting to disk).
    """
    env_model = os.environ.get("OLLAMA_EMBED_MODEL")
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        if not isinstance(data.get("embed_model"), str) or not data["embed_model"].strip():
            data["embed_model"] = env_model.strip() if env_model and env_model.strip() else DEFAULT_EMBED_MODEL
            CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return data["embed_model"]
        if env_model and env_model.strip() and env_model.strip() != data["embed_model"]:
            data["embed_model"] = env_model.strip()
            CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data["embed_model"]
    else:
        data = {"embed_model": env_model.strip() if env_model and env_model.strip() else DEFAULT_EMBED_MODEL}
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data["embed_model"]


def _embed_text(text: str, model: str) -> List[float]:
    prompt = text.strip()
    out = ollama.embeddings(model=model, prompt=prompt)
    if "embedding" in out:
        return [float(x) for x in out["embedding"]]
    if "embeddings" in out and out["embeddings"]:
        return [float(x) for x in out["embeddings"][0]]
    raise RuntimeError("Unexpected embeddings response from Ollama")


def _to_blob_f32(vec: Sequence[float]) -> bytes:
    arr = array("f", (float(x) for x in vec))
    return arr.tobytes()


def _from_blob_f32(blob: bytes) -> List[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


def _l2_norm(vec: Sequence[float]) -> float:
    s = 0.0
    for x in vec:
        s += x * x
    return s ** 0.5


def _cosine(a: Sequence[float], b: Sequence[float], norm_a: Optional[float] = None, norm_b: Optional[float] = None) -> float:
    if len(a) != len(b):
        return -1.0
    if norm_a is None:
        norm_a = _l2_norm(a)
    if norm_b is None:
        norm_b = _l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    return dot / (norm_a * norm_b)


@dataclass
class MemoryRow:
    id: str
    session_id: str
    role: str
    content: str
    ts: int
    source: str
    tags: List[str]
    score: Optional[float] = None


class MemoryStore:
    def __init__(
        self,
        db_path: Path | str = DB_PATH_DEFAULT,
        *,
        embed_model: Optional[str] = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = str(db_path)
        # Resolve embed model: explicit arg > config (auto-ensured) > default
        self.embed_model = (embed_model.strip() if isinstance(embed_model, str) and embed_model.strip() else None) \
                           or _ensure_embed_model_in_config() \
                           or DEFAULT_EMBED_MODEL

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")

        self._ensure_schema()

    # ────────────────────────────────────────────────────────────────
    # Schema
    # ────────────────────────────────────────────────────────────────
    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,                  -- "user" | "assistant" | "system" | "note"
                    content TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    source TEXT NOT NULL                 -- "user_input" | "cog_output" | "ingest" | etc.
                );

                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (memory_id, tag),
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    norm REAL NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS invocations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    mode TEXT NOT NULL,                  -- e.g., "raw_chat", "structured_json", "system_message"
                    user_input TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    meta TEXT                             -- JSON string (nullable)
                );

                CREATE TABLE IF NOT EXISTS invocation_memory (
                    invocation_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    relation TEXT NOT NULL,              -- "recalled" | "produced" | "context"
                    PRIMARY KEY (invocation_id, memory_id, relation),
                    FOREIGN KEY (invocation_id) REFERENCES invocations(id) ON DELETE CASCADE,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories(ts);
                CREATE INDEX IF NOT EXISTS idx_mem_session_ts ON memories(session_id, ts);
                CREATE INDEX IF NOT EXISTS idx_tags_tag ON memory_tags(tag);
                CREATE INDEX IF NOT EXISTS idx_inv_session_ts ON invocations(session_id, ts);
                """
            )

    # ────────────────────────────────────────────────────────────────
    # CRUD: memories + invocations
    # ────────────────────────────────────────────────────────────────
    def add_invocation(
        self,
        *,
        session_id: str,
        mode: str,
        user_input: str,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[int] = None,
    ) -> str:
        inv_id = _new_id()
        ts = ts or _now_ts()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO invocations (id, session_id, mode, user_input, ts, meta) VALUES (?,?,?,?,?,?)",
                (inv_id, session_id, mode, user_input, ts, json.dumps(meta or {}, ensure_ascii=False)),
            )
        return inv_id

    def link_invocation_memory(self, invocation_id: str, memory_id: str, relation: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO invocation_memory (invocation_id, memory_id, relation) VALUES (?,?,?)",
                (invocation_id, memory_id, relation),
            )

    def add_memory(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        source: str,
        tags: Optional[Iterable[str]] = None,
        embed: bool = True,
        ts: Optional[int] = None,
    ) -> str:
        """
        Create a memory row (+vector row if embed=True). Returns memory_id.
        """
        memory_id = _new_id()
        ts = ts or _now_ts()
        tags = list({t.strip() for t in (tags or []) if t and t.strip()})

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO memories (id, session_id, role, content, ts, source) VALUES (?,?,?,?,?,?)",
                (memory_id, session_id, role, content, ts, source),
            )
            for t in tags:
                self._conn.execute(
                    "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?,?)",
                    (memory_id, t),
                )

        if embed:
            vec = _embed_text(content, self.embed_model)
            blob = _to_blob_f32(vec)
            norm = _l2_norm(vec)
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_vectors (memory_id, embedding, dim, norm) VALUES (?,?,?,?)",
                    (memory_id, blob, len(vec), norm),
                )

        return memory_id

    # ────────────────────────────────────────────────────────────────
    # Query / Recall
    # ────────────────────────────────────────────────────────────────
    def _fetch_candidates(
        self,
        *,
        session_id: Optional[str],
        tags: Optional[Iterable[str]],
        since_ts: Optional[int],
        until_ts: Optional[int],
        limit_hint: int = 2000,
    ) -> List[sqlite3.Row]:
        where = []
        params: List[Any] = []

        if session_id:
            where.append("m.session_id = ?")
            params.append(session_id)
        if since_ts:
            where.append("m.ts >= ?")
            params.append(since_ts)
        if until_ts:
            where.append("m.ts <= ?")
            params.append(until_ts)

        tag_join = ""
        if tags:
            tag_join = "JOIN memory_tags t ON t.memory_id = m.id"
            where.append("t.tag IN (%s)" % ",".join("?" for _ in tags))
            params.extend(list(tags))

        sql = f"""
            SELECT m.*, v.embedding, v.dim, v.norm
            FROM memories m
            LEFT JOIN memory_vectors v ON v.memory_id = m.id
            {tag_join}
            {"WHERE " + " AND ".join(where) if where else ""}
            ORDER BY m.ts DESC
            LIMIT ?
        """
        params.append(limit_hint)

        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    def search(
        self,
        *,
        query_text: Optional[str],
        session_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        since_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
        limit: int = 12,
    ) -> List[MemoryRow]:
        """
        Hybrid search. If query_text provided → vector ranking; else → SQL recent with filters.
        Always returns MemoryRow (score is cosine when vector used).
        """
        rows = self._fetch_candidates(
            session_id=session_id,
            tags=tags,
            since_ts=since_ts,
            until_ts=until_ts,
            limit_hint=max(limit * 8, 200),  # oversample for quality
        )

        results: List[MemoryRow] = []

        if query_text:
            qvec = _embed_text(query_text, self.embed_model)
            qnorm = _l2_norm(qvec)

            for r in rows:
                emb_blob = r["embedding"]
                if emb_blob is None:
                    score = -1e9  # prefer vectorized rows
                else:
                    vec = _from_blob_f32(emb_blob)
                    score = _cosine(qvec, vec, qnorm, r["norm"])
                results.append(
                    MemoryRow(
                        id=r["id"],
                        session_id=r["session_id"],
                        role=r["role"],
                        content=r["content"],
                        ts=r["ts"],
                        source=r["source"],
                        tags=self._tags_for_memory(r["id"]),
                        score=score,
                    )
                )
            # sort by score desc, ts desc tie-break
            results.sort(key=lambda x: (x.score if x.score is not None else -1e9, x.ts), reverse=True)
            return results[:limit]

        # No query_text → recent items by ts (already ordered desc)
        for r in rows[:limit]:
            results.append(
                MemoryRow(
                    id=r["id"],
                    session_id=r["session_id"],
                    role=r["role"],
                    content=r["content"],
                    ts=r["ts"],
                    source=r["source"],
                    tags=self._tags_for_memory(r["id"]),
                    score=None,
                )
            )
        return results

    def recall_for_context(
        self,
        *,
        session_id: Optional[str],
        query_text: str,
        tags: Optional[Iterable[str]] = None,
        k: int = 8,
    ) -> List[MemoryRow]:
        """Convenience: vector search primarily scoped to a session (if provided)."""
        return self.search(query_text=query_text, session_id=session_id, tags=tags, limit=k)

    def fetch_recent(
        self,
        *,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[MemoryRow]:
        """Return most recent memories, optionally scoped to a session."""
        where = []
        params: List[Any] = []
        if session_id:
            where.append("m.session_id = ?")
            params.append(session_id)

        sql = f"""
            SELECT m.id, m.session_id, m.role, m.content, m.ts, m.source,
                   GROUP_CONCAT(t.tag, '\u241f') AS tags
            FROM memories m
            LEFT JOIN memory_tags t ON t.memory_id = m.id
            {"WHERE " + " AND ".join(where) if where else ""}
            GROUP BY m.id
            ORDER BY m.ts DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        out: List[MemoryRow] = []
        for r in rows:
            tag_blob = r["tags"] or ""
            tags = [t for t in (tag_blob.split("\u241f") if tag_blob else []) if t]
            out.append(
                MemoryRow(
                    id=r["id"],
                    session_id=r["session_id"],
                    role=r["role"],
                    content=r["content"],
                    ts=int(r["ts"]),
                    source=r["source"],
                    tags=tags,
                    score=None,
                )
            )
        return out

    def _tags_for_memory(self, memory_id: str) -> List[str]:
        cur = self._conn.execute(
            "SELECT tag FROM memory_tags WHERE memory_id = ? ORDER BY tag ASC",
            (memory_id,),
        )
        return [row["tag"] for row in cur.fetchall()]

    # ────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
