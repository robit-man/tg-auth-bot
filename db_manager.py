#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Manager - SQLite schema management, connection pooling, and threaded write queue
"""

import sqlite3
import threading
import queue
import asyncio
import time
from pathlib import Path
from typing import Optional, Callable
from contextlib import closing


# Paths
def get_data_paths(root: Path) -> dict:
    """Initialize and return data directory paths"""
    data_dir = root / "data"
    users_dir = data_dir / "users"
    channels_dir = data_dir / "channels"

    for d in (data_dir, users_dir, channels_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        'data': data_dir,
        'users': users_dir,
        'channels': channels_dir
    }


def user_db_path(users_dir: Path, user_id: int) -> Path:
    """Get path to user-specific database shard"""
    return users_dir / f"user_{user_id}.db"


def channel_db_path(channels_dir: Path, chat_id: int, thread_id: Optional[int]) -> Path:
    """Get path to channel/thread-specific database shard"""
    return channels_dir / f"chat{chat_id}_topic{(thread_id or 0)}.db"


# Threaded DB Write Queue
class DBWriteQueue:
    """Serializes writes across all SQLite files using a dedicated worker thread."""

    def __init__(self):
        self.q: "queue.Queue[tuple[str, Callable[[sqlite3.Connection], object], asyncio.AbstractEventLoop, asyncio.Future]]" = queue.Queue()
        self.conns: dict[str, sqlite3.Connection] = {}
        self.thread: Optional[threading.Thread] = None
        self.stop = threading.Event()

    def _get_conn(self, path: str) -> sqlite3.Connection:
        con = self.conns.get(path)
        if con:
            return con
        con = sqlite3.connect(path, timeout=60, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=8000;")
        ensure_schema_on(con)  # safe for any db
        self.conns[path] = con
        return con

    def _worker(self):
        while not self.stop.is_set():
            try:
                db_path, fn, loop, fut = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                con = self._get_conn(db_path)
                delay = 0.05
                for _ in range(10):
                    try:
                        res = fn(con)  # NOTE: fn must be a regular def, not async def
                        loop.call_soon_threadsafe(fut.set_result, res)
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower() or "busy" in str(e).lower():
                            time.sleep(delay)
                            delay = min(delay * 1.8, 1.8)
                            continue
                        loop.call_soon_threadsafe(fut.set_exception, e)
                        break
                else:
                    loop.call_soon_threadsafe(fut.set_exception, sqlite3.OperationalError("database is locked (retry limit reached)"))
            except Exception as e:
                loop.call_soon_threadsafe(fut.set_exception, e)
            finally:
                self.q.task_done()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._worker, name="db-writer", daemon=True)
        self.thread.start()

    async def run(self, db_path: str, fn: Callable[[sqlite3.Connection], object]):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.q.put((db_path, fn, loop, fut))
        return await fut


# Schema management
def ensure_schema_on(con: sqlite3.Connection):
    """Create schema common to all database shards"""
    # Basic tables common to shards
    con.execute("""
      CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id INTEGER, chat_id INTEGER, chat_type TEXT, chat_title TEXT, thread_id INTEGER,
        reply_to_id INTEGER, user_id INTEGER, username TEXT, first_name TEXT, last_name TEXT,
        is_private INTEGER, date INTEGER, text TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS embeddings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT, ref_id INTEGER, chat_id INTEGER, thread_id INTEGER,
        user_id INTEGER, date INTEGER, dim INTEGER, vec TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS gleanings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_msg_id INTEGER, chat_id INTEGER, thread_id INTEGER,
        user_id INTEGER, date INTEGER, fact TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS kg_nodes(
        id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, node_key TEXT UNIQUE, label TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS kg_edges(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        src_id INTEGER, rel TEXT, dst_id INTEGER, weight REAL, ts INTEGER,
        chat_id INTEGER, thread_id INTEGER, message_row_id INTEGER, fact_id INTEGER
      );
    """)
    # Per-user shard profile table
    con.execute("""
      CREATE TABLE IF NOT EXISTS profile(
        user_id INTEGER PRIMARY KEY,
        version INTEGER,
        last_updated INTEGER,
        msg_count INTEGER,
        avg_len REAL,
        question_ratio REAL,
        emoji_ratio REAL,
        link_ratio REAL,
        code_ratio REAL,
        positivity REAL,
        style_notes TEXT,
        suggestions TEXT
      );
    """)


def create_main_db(db_path: str) -> sqlite3.Connection:
    """Create and initialize the main database with full schema"""
    con = sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=8000;")

    # Common schema
    ensure_schema_on(con)

    # Main database specific tables
    con.execute("""
      CREATE TABLE IF NOT EXISTS groups(
        chat_id INTEGER PRIMARY KEY,
        title TEXT,
        token TEXT UNIQUE
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS allowed(
        chat_id INTEGER, user_id INTEGER, username TEXT, first_name TEXT, last_name TEXT,
        PRIMARY KEY(chat_id, user_id)
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS settings(
        key TEXT PRIMARY KEY,
        value TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS user_profiles_idx(
        user_id INTEGER PRIMARY KEY,
        last_updated INTEGER DEFAULT 0,
        optout INTEGER DEFAULT 0
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS internal_state(
        scope TEXT,
        chat_id INTEGER,
        thread_id INTEGER,
        user_id INTEGER,
        state TEXT,
        updated INTEGER,
        PRIMARY KEY(scope, chat_id, thread_id, user_id)
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS memory_entries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT,
        chat_id INTEGER,
        thread_id INTEGER,
        user_id INTEGER,
        category TEXT,
        content TEXT,
        metadata TEXT,
        embedding TEXT,
        weight REAL DEFAULT 0.5,
        created_ts INTEGER,
        updated_ts INTEGER
      );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_memory_scope ON memory_entries(scope, chat_id, thread_id, user_id, category)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_ts)")
    con.execute("""
      CREATE TABLE IF NOT EXISTS memory_links(
        memory_id INTEGER,
        source_type TEXT,
        source_id INTEGER,
        weight REAL DEFAULT 0.5,
        metadata TEXT,
        PRIMARY KEY(memory_id, source_type, source_id),
        FOREIGN KEY(memory_id) REFERENCES memory_entries(id) ON DELETE CASCADE
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS memory_context_snapshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        thread_id INTEGER,
        created_ts INTEGER,
        complexity TEXT,
        summary TEXT,
        metadata TEXT
      );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_memory ON memory_links(memory_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_context_snapshots ON memory_context_snapshots(chat_id, thread_id, created_ts)")
    con.execute("""
      CREATE TABLE IF NOT EXISTS memory_summaries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT,
        chat_id INTEGER,
        thread_id INTEGER,
        user_id INTEGER,
        category TEXT,
        summary TEXT,
        metadata TEXT,
        created_ts INTEGER
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS self_state(
        scope TEXT PRIMARY KEY,
        mood REAL,
        tension REAL,
        last_feedback_ts INTEGER,
        data TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS breakout_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_id INTEGER,
        thread_id INTEGER,
        trigger TEXT,
        details TEXT,
        created_ts INTEGER,
        status TEXT
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS relationship_graph(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_user INTEGER,
        target_user INTEGER,
        chat_id INTEGER,
        thread_id INTEGER,
        relation TEXT,
        description TEXT,
        first_msg_id INTEGER,
        last_msg_id INTEGER,
        first_ts INTEGER,
        last_ts INTEGER,
        weight REAL DEFAULT 0.5,
        metadata TEXT
      );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_relationship_graph ON relationship_graph(source_user, target_user, chat_id, thread_id, relation)")

    # Migrations
    def has_col(table, col):
        return col in {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}

    if not has_col("embeddings", "thread_id"):
        con.execute("ALTER TABLE embeddings ADD COLUMN thread_id INTEGER;")
    if not has_col("gleanings", "thread_id"):
        con.execute("ALTER TABLE gleanings ADD COLUMN thread_id INTEGER;")
    if not has_col("allowed", "username"):
        con.execute("ALTER TABLE allowed ADD COLUMN username TEXT;")
    if not has_col("allowed", "first_name"):
        con.execute("ALTER TABLE allowed ADD COLUMN first_name TEXT;")
    if not has_col("allowed", "last_name"):
        con.execute("ALTER TABLE allowed ADD COLUMN last_name TEXT;")

    return con


# Settings helpers
async def settings_set_async(db_write_queue: DBWriteQueue, db_path: str, key: str, value: str):
    """Set a settings value asynchronously"""
    def _work(con: sqlite3.Connection):
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        con.commit()
    await db_write_queue.run(db_path, _work)


def settings_get(db_path: str, key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a settings value"""
    with closing(sqlite3.connect(db_path, timeout=60)) as con:
        con.execute("PRAGMA busy_timeout=8000;")
        r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r[0] if r else default
