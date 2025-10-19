#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gatekeeper Bot — gating, admin tools, inspector, Ollama chat, KG, semantic search,
gleanings, sharded DBs, auto system-prompt generation, robust SQLite write queue (threaded),
and privacy-aware per-user "interaction profiles" updated during idle time.

Python 3.10+
"""

import os, sys, subprocess, textwrap, sqlite3, base64, secrets, re, time, json, math, random, asyncio, socket, threading, queue, traceback
from pathlib import Path
from contextlib import closing
from functools import wraps
from typing import Optional, Tuple, List, Iterable, Dict, Callable
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────
# 0) Self-bootstrap venv + deps (with PTB job-queue extra)
# ──────────────────────────────────────────────────────────────
ROOT = Path.cwd()
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"

def _venv_python(p: Path) -> Path:
    return p / ("Scripts/python.exe" if IS_WIN else "bin/python")

def ensure_venv_and_deps():
    if os.environ.get("VIRTUAL_ENV") or str(sys.prefix).endswith(".venv"):
        return
    if not VENV.exists():
        print("[bootstrap] Creating .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    py = str(_venv_python(VENV))
    print("[bootstrap] Upgrading pip/setuptools/wheel ...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    reqs = [
        'python-telegram-bot[job-queue]>=20,<21',  # JobQueue included
        'APScheduler>=3.9,<4.0',
        "python-dotenv>=1.0,<2.0",
        "requests>=2.31,<3.0",
    ]
    print("[bootstrap] Installing requirements ...")
    subprocess.check_call([py, "-m", "pip", "install", *reqs])
    print("[bootstrap] Re-exec in .venv ...")
    os.execv(py, [py, *sys.argv])

ensure_venv_and_deps()

# Inside venv
import requests
from dotenv import load_dotenv, dotenv_values
from telegram import Update, ChatPermissions, InlineKeyboardMarkup, InlineKeyboardButton, MessageEntity
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

# ──────────────────────────────────────────────────────────────
# 1) .env + config (auto-detect Ollama models)
# ──────────────────────────────────────────────────────────────
ENV_PATH = ROOT / ".env"

def detect_ollama_models(url: str) -> tuple[str, str]:
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=2)
        r.raise_for_status()
        data = r.json() or {}
        models = [ (m.get("model") or m.get("name") or "").strip() for m in (data.get("models") or []) ]
        chat = models[0] if models else ""
        emb = ""
        for pref in ["nomic-embed","mxbai-embed","all-minilm","gte","bge","text-embedding","embed"]:
            for m in models:
                if pref in m.lower():
                    emb = m; break
            if emb: break
        return chat, emb
    except Exception:
        return "", ""

def write_env_template():
    chat, emb = detect_ollama_models("http://127.0.0.1:11434")
    ENV_PATH.write_text(textwrap.dedent(f"""\
        BOT_TOKEN=
        PRIMARY_CHAT_ID=
        ADMIN_WHITELIST=
        GATE_DB=gate.db

        OLLAMA_URL=http://127.0.0.1:11434
        OLLAMA_MODEL={chat}
        OLLAMA_EMBED_MODEL={emb}

        SYSTEM_PROMPT=
        AUTO_REPLY_MODE=smart

        MAX_CONTEXT_MESSAGES=300
        MAX_CONTEXT_CHARS=12000
        MAX_CONTEXT_USERS=60
        MAX_CONTEXT_GROUPS=30

        AUTO_SYSTEM_PROMPT_ENABLED=true
        AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=12

        PROFILES_ENABLED=true
        PROFILE_REFRESH_MINUTES=30
        """), encoding="utf-8")

def ensure_env_file() -> dict:
    if not ENV_PATH.exists():
        write_env_template()
        print("[env] Wrote .env (fill BOT_TOKEN; models auto-detected if possible).")
        sys.exit(0)
    load_dotenv(ENV_PATH)
    return dotenv_values(ENV_PATH)

def _write_env_var(key: str, value: str):
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"; found = True; break
    if not found: lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

CFG = ensure_env_file()
BOT_TOKEN = (os.getenv("BOT_TOKEN") or (CFG.get("BOT_TOKEN") or "")).strip()
PRIMARY_CHAT_ID = (os.getenv("PRIMARY_CHAT_ID") or (CFG.get("PRIMARY_CHAT_ID") or "")).strip()
DB_PATH = (os.getenv("GATE_DB") or (CFG.get("GATE_DB") or "gate.db")).strip()
ADMIN_WHITELIST_RAW = (os.getenv("ADMIN_WHITELIST") or (CFG.get("ADMIN_WHITELIST") or "")).strip()
OLLAMA_URL = (os.getenv("OLLAMA_URL") or (CFG.get("OLLAMA_URL") or "http://127.0.0.1:11434")).strip().rstrip("/")
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or (CFG.get("OLLAMA_MODEL") or "")).strip()
OLLAMA_EMBED_MODEL = (os.getenv("OLLAMA_EMBED_MODEL") or (CFG.get("OLLAMA_EMBED_MODEL") or "")).strip()
SYSTEM_PROMPT = (os.getenv("SYSTEM_PROMPT") or (CFG.get("SYSTEM_PROMPT") or "")).strip()
AUTO_REPLY_MODE = (os.getenv("AUTO_REPLY_MODE") or (CFG.get("AUTO_REPLY_MODE") or "smart")).strip().lower()
AUTO_SYS_ENABLED = ((os.getenv("AUTO_SYSTEM_PROMPT_ENABLED") or (CFG.get("AUTO_SYSTEM_PROMPT_ENABLED") or "true")).strip().lower()=="true")
AUTO_SYS_INTERVAL_HOURS = int((os.getenv("AUTO_SYSTEM_PROMPT_INTERVAL_HOURS") or (CFG.get("AUTO_SYSTEM_PROMPT_INTERVAL_HOURS") or "12")).strip() or "12")
PROFILES_ENABLED = ((os.getenv("PROFILES_ENABLED") or (CFG.get("PROFILES_ENABLED") or "true")).strip().lower()=="true")
PROFILE_REFRESH_MINUTES = int((os.getenv("PROFILE_REFRESH_MINUTES") or (CFG.get("PROFILE_REFRESH_MINUTES") or "30")).strip() or "30")

if not BOT_TOKEN:
    print("[env] BOT_TOKEN is empty. Set it and run again."); sys.exit(1)

if not OLLAMA_MODEL or not OLLAMA_EMBED_MODEL:
    chat, emb = detect_ollama_models(OLLAMA_URL)
    if not OLLAMA_MODEL and chat:
        OLLAMA_MODEL = chat; _write_env_var("OLLAMA_MODEL", OLLAMA_MODEL)
        print(f"[ollama] chat model: {OLLAMA_MODEL}")
    if not OLLAMA_EMBED_MODEL and emb:
        OLLAMA_EMBED_MODEL = emb; _write_env_var("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_MODEL)
        print(f"[ollama] embed model: {OLLAMA_EMBED_MODEL}")

def parse_admin_whitelist(raw: str) -> set[int]:
    ids = set()
    for tok in (raw or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok: continue
        if tok.startswith("@"): print(f"[env] Ignoring @{tok[1:]} in ADMIN_WHITELIST; use numeric IDs."); continue
        try: ids.add(int(tok))
        except ValueError: print(f"[env] Skipping non-numeric ADMIN_WHITELIST entry: {tok}")
    return ids

ADMIN_WHITELIST: set[int] = parse_admin_whitelist(ADMIN_WHITELIST_RAW)
def add_admin_ids(ids_to_add: Iterable[int]):
    global ADMIN_WHITELIST
    ADMIN_WHITELIST |= {int(i) for i in ids_to_add}
    _write_env_var("ADMIN_WHITELIST", ",".join(str(i) for i in sorted(ADMIN_WHITELIST)))

MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES") or 300)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS") or 12000)
MAX_CONTEXT_USERS = int(os.getenv("MAX_CONTEXT_USERS") or 60)
MAX_CONTEXT_GROUPS = int(os.getenv("MAX_CONTEXT_GROUPS") or 30)

# ──────────────────────────────────────────────────────────────
# 2) Paths + shard dirs
# ──────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
USERS_DIR = DATA_DIR / "users"
CHANNELS_DIR = DATA_DIR / "channels"
for d in (DATA_DIR, USERS_DIR, CHANNELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def user_db_path(user_id: int) -> Path:
    return USERS_DIR / f"user_{user_id}.db"

def channel_db_path(chat_id: int, thread_id: int | None) -> Path:
    return CHANNELS_DIR / f"chat{chat_id}_topic{(thread_id or 0)}.db"

# ──────────────────────────────────────────────────────────────
# 3) Threaded DB Write Queue (fixes "no loop" + lock contention)
# ──────────────────────────────────────────────────────────────
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

DBW: Optional[DBWriteQueue] = None

# ──────────────────────────────────────────────────────────────
# 4) Schema helpers (main + shards)
# ──────────────────────────────────────────────────────────────
def ensure_schema_on(con: sqlite3.Connection):
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

def db():
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=8000;")
    # main schema
    ensure_schema_on(con)
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
    # Index for scheduling user profile refreshes (+ optout)
    con.execute("""
      CREATE TABLE IF NOT EXISTS user_profiles_idx(
        user_id INTEGER PRIMARY KEY,
        last_updated INTEGER DEFAULT 0,
        optout INTEGER DEFAULT 0
      );
    """)
    # migrations
    def has_col(table, col):
        return col in {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if not has_col("embeddings","thread_id"): con.execute("ALTER TABLE embeddings ADD COLUMN thread_id INTEGER;")
    if not has_col("gleanings","thread_id"): con.execute("ALTER TABLE gleanings ADD COLUMN thread_id INTEGER;")
    if not has_col("allowed","username"): con.execute("ALTER TABLE allowed ADD COLUMN username TEXT;")
    if not has_col("allowed","first_name"): con.execute("ALTER TABLE allowed ADD COLUMN first_name TEXT;")
    if not has_col("allowed","last_name"): con.execute("ALTER TABLE allowed ADD COLUMN last_name TEXT;")
    return con

async def settings_set_async(key: str, value: str):
    def _work(con: sqlite3.Connection):
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        con.commit()
    await DBW.run(DB_PATH, _work)

def settings_get(key: str, default: Optional[str]=None) -> Optional[str]:
    with closing(db()) as con:
        r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r[0] if r else default

# ──────────────────────────────────────────────────────────────
# 5) Group token + allow-list (queued writes)
# ──────────────────────────────────────────────────────────────
async def get_or_create_group(chat_id: int, title: str) -> str:
    def _work(con: sqlite3.Connection):
        row = con.execute("SELECT token FROM groups WHERE chat_id=?", (chat_id,)).fetchone()
        if row: return row[0]
        token = base64.urlsafe_b64encode(secrets.token_bytes(12)).decode().rstrip("=")
        con.execute("INSERT INTO groups(chat_id,title,token) VALUES(?,?,?)",(chat_id,title or "",token))
        con.commit()
        return token
    return await DBW.run(DB_PATH, _work)

def token_to_chat_id(token: str) -> Optional[int]:
    with closing(db()) as con:
        r = con.execute("SELECT chat_id FROM groups WHERE token=?", (token,)).fetchone()
        return r[0] if r else None

async def mark_allowed_async(chat_id: int, user) -> None:
    def _work(con: sqlite3.Connection):
        con.execute("""INSERT OR REPLACE INTO allowed(chat_id,user_id,username,first_name,last_name)
                       VALUES(?,?,?,?,?)""",
                    (chat_id, user.id, (user.username or ""), (user.first_name or ""), (user.last_name or "")))
        con.commit()
    await DBW.run(DB_PATH, _work)

# ──────────────────────────────────────────────────────────────
# 6) Ollama helpers
# ──────────────────────────────────────────────────────────────
def embed_text(text: str) -> Optional[List[float]]:
    try:
        r = requests.post(OLLAMA_URL + "/api/embeddings", json={"model": OLLAMA_EMBED_MODEL, "prompt": text}, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        vec = data.get("embedding") or data.get("embeddings")
        if isinstance(vec, list) and vec and isinstance(vec[0], (int,float)):
            return [float(x) for x in vec]
    except Exception:
        return None
    return None

def ollama_chat(payload: dict) -> Optional[str]:
    try:
        r = requests.post(OLLAMA_URL + "/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json() or {}
        msg = data.get("message") or {}
        content = msg.get("content")
        if content: return content.strip()
        if "content" in data and isinstance(data["content"], str): return data["content"].strip()
    except Exception:
        return None
    return None

async def ai_generate_async(payload: dict) -> Optional[str]:
    return await asyncio.to_thread(ollama_chat, payload)

async def embed_text_async(text: str) -> Optional[List[float]]:
    return await asyncio.to_thread(embed_text, text)

async def with_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int, coro):
    stop = asyncio.Event()
    async def _typer():
        while not stop.is_set():
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            await asyncio.sleep(4)
    task = asyncio.create_task(_typer())
    try:
        return await coro
    finally:
        stop.set()

# ──────────────────────────────────────────────────────────────
# 7) KG (queued writes)
# ──────────────────────────────────────────────────────────────
async def kg_upsert_node_q(node_type: str, node_key: str, label: str) -> int:
    def _work(con: sqlite3.Connection):
        r = con.execute("SELECT id FROM kg_nodes WHERE node_key=?", (node_key,)).fetchone()
        if r: return r[0]
        cur = con.execute("INSERT INTO kg_nodes(type,node_key,label) VALUES(?,?,?)", (node_type, node_key, label))
        con.commit()
        return cur.lastrowid
    return await DBW.run(DB_PATH, _work)

async def kg_add_edge_q(src_id: int, rel: str, dst_id: int, weight: float, ts: int,
                        chat_id: int, thread_id: int, message_row_id: Optional[int]=None, fact_id: Optional[int]=None):
    def _work(con: sqlite3.Connection):
        con.execute("""INSERT INTO kg_edges(src_id,rel,dst_id,weight,ts,chat_id,thread_id,message_row_id,fact_id)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (src_id, rel, dst_id, float(weight), int(ts), chat_id, int(thread_id or 0), message_row_id, fact_id))
        con.commit()
    await DBW.run(DB_PATH, _work)

async def kg_ingest_message_signals_async(row_id: int):
    with closing(db()) as con:
        r = con.execute("SELECT chat_id, chat_title, thread_id, user_id, username, first_name, last_name, date, text FROM messages WHERE id=?", (row_id,)).fetchone()
    if not r: return
    chat_id, chat_title, thread_id, uid, uname, first, last, ts, text = r
    user_label = " ".join(p for p in (first,last) if p).strip() or (uname and f"@{uname}") or f"id {uid}"
    chat_node = await kg_upsert_node_q("chat", f"chat:{chat_id}", chat_title or str(chat_id))
    channel_node = await kg_upsert_node_q("channel", f"chat:{chat_id}:topic:{thread_id or 0}", f"topic:{thread_id or 0}")
    user_node = await kg_upsert_node_q("user", f"user:{uid}", user_label)
    await kg_add_edge_q(user_node, "in_chat", chat_node, 1.0, ts, chat_id, thread_id, message_row_id=row_id)
    await kg_add_edge_q(user_node, "in_channel", channel_node, 1.0, ts, chat_id, thread_id, message_row_id=row_id)
    for at in re.findall(r'@([A-Za-z0-9_]{5,})', text or "")[:10]:
        mnode = await kg_upsert_node_q("entity", f"handle:@{at}", f"@{at}")
        await kg_add_edge_q(user_node, "mentions", mnode, 0.7, ts, chat_id, thread_id, message_row_id=row_id)

def extract_triples_with_llm(snippet: str) -> List[dict]:
    prompt = textwrap.dedent(f"""
        Extract 1-6 triples SUBJECT|RELATION|OBJECT as JSON lines {{"s":"","r":"","o":""}}.
        Keep concrete and non-speculative.

        Snippet:
        {snippet}
    """).strip()
    try:
        r = requests.post(OLLAMA_URL + "/api/chat",
                          json={"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False},
                          timeout=40)
        r.raise_for_status()
        data = r.json() or {}
        content = (data.get("message") or {}).get("content") or ""
    except Exception:
        return []
    triples = []
    for line in content.splitlines():
        line=line.strip().strip(",; ")
        if not (line.startswith("{") and line.endswith("}")): continue
        try:
            obj=json.loads(line)
            s=(obj.get("s") or obj.get("subject") or "").strip()
            r_=(obj.get("r") or obj.get("relation") or "").strip()
            o=(obj.get("o") or obj.get("object") or "").strip()
            if s and r_ and o: triples.append({"s":s[:120],"r":r_[:60],"o":o[:120]})
        except Exception: pass
    return triples[:6]

async def kg_add_triples_for_message_async(row_id: int):
    with closing(db()) as con:
        r = con.execute("SELECT chat_id, chat_title, thread_id, user_id, username, first_name, last_name, date, text "
                        "FROM messages WHERE id=?", (row_id,)).fetchone()
    if not r: return
    chat_id, chat_title, thread_id, uid, uname, first, last, ts, text = r
    snippet = fetch_thread_context(chat_id, thread_id, limit=12)
    triples = extract_triples_with_llm(snippet + "\n\n" + (text or ""))
    if not triples: return
    chat_node = await kg_upsert_node_q("chat", f"chat:{chat_id}", chat_title or str(chat_id))
    channel_node = await kg_upsert_node_q("channel", f"chat:{chat_id}:topic:{thread_id or 0}", f"topic:{thread_id or 0}")
    async def node_for(label: str) -> int:
        if label.startswith("@"): return await kg_upsert_node_q("entity", f"handle:{label}", label)
        return await kg_upsert_node_q("entity", f"entity:{label.lower()}", label)
    for t in triples:
        s_id = await node_for(t["s"]); o_id = await node_for(t["o"])
        await kg_add_edge_q(s_id, t["r"], o_id, 1.0, ts, chat_id, thread_id, message_row_id=row_id, fact_id=None)
        await kg_add_edge_q(s_id, "in_channel", channel_node, 0.4, ts, chat_id, thread_id, message_row_id=row_id)
        await kg_add_edge_q(o_id, "in_channel", channel_node, 0.4, ts, chat_id, thread_id, message_row_id=row_id)
        await kg_add_edge_q(s_id, "in_chat", chat_node, 0.2, ts, chat_id, thread_id, message_row_id=row_id)
        await kg_add_edge_q(o_id, "in_chat", chat_node, 0.2, ts, chat_id, thread_id, message_row_id=row_id)

# ──────────────────────────────────────────────────────────────
# 8) Context/similarity (reads)
# ──────────────────────────────────────────────────────────────
def fetch_thread_context(chat_id: int, thread_id: int, limit: int = 24) -> str:
    with closing(db()) as con:
        rows = con.execute(
            """SELECT date, user_id, username, first_name, last_name, text
               FROM messages WHERE chat_id=? AND (thread_id=? OR ?=0)
               ORDER BY id DESC LIMIT ?""",
            (chat_id, thread_id or 0, thread_id or 0, limit)
        ).fetchall()
    lines = []
    for (ts, uid, uname, first, last, txt) in reversed(rows):
        name = " ".join(p for p in (first,last) if p).strip() or (uname or f"user#{uid}")
        lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] {name}: {txt}")
    return "\n".join(lines)

def fetch_context_messages(limit_msgs: int, limit_chars: int) -> str:
    with closing(db()) as con:
        rows = con.execute(
            "SELECT chat_type, chat_title, user_id, username, first_name, last_name, date, text "
            "FROM messages WHERE text IS NOT NULL AND TRIM(text)!='' ORDER BY date ASC LIMIT ?",
            (limit_msgs,)
        ).fetchall()
    buff, total = [], 0
    for (ctype, ctitle, uid, uname, first, last, ts, text) in rows:
        name = " ".join(p for p in (first,last) if p).strip() or (uname or f"user#{uid}")
        where = ctitle if ctype != "private" else "DM"
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}] ({where}) {name}: {text}"
        if total + len(line) + 1 > limit_chars: break
        buff.append(line); total += len(line) + 1
    return "\n".join(buff)

def fetch_metrics(max_users: int = 60, max_groups: int = 30) -> Dict[str,str]:
    with closing(db()) as con:
        glist = con.execute(
            "SELECT DISTINCT chat_id, COALESCE(NULLIF(chat_title,''), CAST(chat_id AS TEXT)), chat_type "
            "FROM messages WHERE chat_type!='private' ORDER BY chat_title ASC"
        ).fetchall()
        ulist = con.execute(
            "SELECT user_id, COALESCE(NULLIF(username,''),'(no-username)'), "
            "COALESCE(NULLIF(first_name||' '||last_name,''), '(no-name)'), MAX(date) AS last_seen "
            "FROM messages WHERE user_id IS NOT NULL GROUP BY user_id, username, first_name, last_name "
            "ORDER BY last_seen DESC"
        ).fetchall()
    servers_lines = [f"- {title} [{ctype}] (id {cid})" for (cid,title,ctype) in glist[:max_groups]]
    users_lines = [f"- {fname} [{'@'+uname if uname!='(no-username)' else '(no-username)'}] (id {uid})"
                   for (uid, uname, fname, _) in ulist[:max_users]]
    return {"servers_count": str(len(glist)), "users_count": str(len(ulist)),
            "servers_list": "\n".join(servers_lines), "users_list": "\n".join(users_lines)}

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a)!=len(b): return -1.0
    dot = sum(x*y for x,y in zip(a,b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    return (dot/(na*nb)) if na>0 and nb>0 else -1.0

async def store_embedding_q(db_path: str, kind: str, ref_id: int, chat_id: int, thread_id: int, user_id: Optional[int], ts: int, vec: List[float]):
    def _work(con: sqlite3.Connection):
        con.execute("INSERT INTO embeddings(kind,ref_id,chat_id,thread_id,user_id,date,dim,vec) VALUES(?,?,?,?,?,?,?,?)",
                    (kind, ref_id, chat_id, int(thread_id or 0), user_id or 0, ts, len(vec), json.dumps(vec)))
        con.commit()
    await DBW.run(db_path, _work)

async def similar_context(query: str, chat_id: int, thread_id: int, top_k: int = 8) -> Dict[str,str]:
    qv = await embed_text_async(query)
    if not qv: return {"channel":"", "server":"", "global":""}
    with closing(db()) as con:
        rows = con.execute("SELECT kind, ref_id, chat_id, thread_id, user_id, date, dim, vec FROM embeddings").fetchall()
    scored = []
    for (kind, ref_id, c_id, th_id, uid, ts, dim, vecjson) in rows:
        try:
            v = json.loads(vecjson)
            if isinstance(v, list) and len(v)==len(qv):
                s = cosine(qv, v)
                if c_id == chat_id and (th_id or 0) == (thread_id or 0): s *= 2.0
                elif c_id == chat_id: s *= 1.5
                scored.append((s, kind, ref_id, c_id, th_id, uid, ts))
        except Exception: continue
    scored.sort(key=lambda x: x[0], reverse=True)
    chan, serv, glob = [], [], []
    for (score, kind, ref_id, c_id, th_id, uid, ts) in scored[: max(top_k*3, 18)]:
        prefix = "[fact]" if kind == "fact" else "[msg]"
        if kind == "message":
            with closing(db()) as con:
                r = con.execute("SELECT text, chat_title, username, first_name, last_name FROM messages WHERE id=?", (ref_id,)).fetchone()
            if not r: continue
            text, ctitle, uname, first, last = r
            name = " ".join(p for p in (first,last) if p).strip() or (uname or f"user#{uid}")
            line = f"{prefix} s≈{score:.2f} ({ctitle}) {name}: {text}"
        else:
            with closing(db()) as con:
                r = con.execute("SELECT fact FROM gleanings WHERE id=?", (ref_id,)).fetchone()
            if not r: continue
            line = f"{prefix} s≈{score:.2f} {r[0]}"
        if c_id == chat_id and (th_id or 0) == (thread_id or 0): chan.append(line)
        elif c_id == chat_id: serv.append(line)
        else: glob.append(line)
        if len(chan)>=top_k and len(serv)>=top_k and len(glob)>=top_k: break
    return {"channel":"\n".join(chan[:top_k]), "server":"\n".join(serv[:top_k]), "global":"\n".join(glob[:top_k])}

# ──────────────────────────────────────────────────────────────
# 9) Sharded writes (queued)
# ──────────────────────────────────────────────────────────────
async def q_insert_message(db_path: str, m, chat, user, ts: int) -> int:
    txt = (m.text or m.caption or "")[:5000]
    thread_id = getattr(m,"message_thread_id",None) or 0
    reply_to = (m.reply_to_message.message_id if getattr(m,"reply_to_message",None) else None)
    def _work(con: sqlite3.Connection):
        cur = con.execute(
            """INSERT INTO messages(msg_id,chat_id,chat_type,chat_title,thread_id,reply_to_id,
                                    user_id,username,first_name,last_name,is_private,date,text)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (m.message_id, chat.id, chat.type, (getattr(chat,"title",None) or ""),
             thread_id, reply_to,
             (user.id if user else None), ((user.username or "") if user else ""),
             ((user.first_name or "") if user else ""), ((user.last_name or "") if user else ""),
             1 if chat.type=="private" else 0, ts, txt)
        )
        con.commit()
        return cur.lastrowid
    return await DBW.run(db_path, _work)

async def mirror_message_to_shards(update: Update) -> dict:
    m, chat, user = update.effective_message, update.effective_chat, update.effective_user
    ts = int(m.date.timestamp()) if m and m.date else int(time.time())
    global_row_id = await q_insert_message(DB_PATH, m, chat, user, ts)
    user_row_id = await q_insert_message(str(user_db_path(user.id)), m, chat, user, ts) if user else None
    thread_id = getattr(m,"message_thread_id",None) or 0
    channel_row_id = await q_insert_message(str(channel_db_path(chat.id, thread_id)), m, chat, user, ts)
    return {"global_row_id": global_row_id, "user_row_id": user_row_id,
            "channel_row_id": channel_row_id, "thread_id": thread_id, "ts": ts}

async def shard_embeddings_for_message(text: str, meta: dict, chat_id: int, user_id: Optional[int]):
    vec = await embed_text_async(text[:2000])
    if not vec: return
    await store_embedding_q(DB_PATH, "message", meta["global_row_id"], chat_id, meta["thread_id"], user_id, meta["ts"], vec)
    if user_id and meta.get("user_row_id"):
        await store_embedding_q(str(user_db_path(user_id)), "message", meta["user_row_id"], chat_id, meta["thread_id"], user_id, meta["ts"], vec)
    if meta.get("channel_row_id") is not None:
        await store_embedding_q(str(channel_db_path(chat_id, meta["thread_id"])), "message", meta["channel_row_id"], chat_id, meta["thread_id"], user_id, meta["ts"], vec)

async def mirror_fact_to_shards_async(fact_text: str, meta: dict, chat_id: int, user_id: Optional[int]):
    now = int(time.time())
    if user_id and meta.get("user_row_id"):
        def _user(con: sqlite3.Connection):
            cur = con.execute("INSERT INTO gleanings(source_msg_id,chat_id,thread_id,user_id,date,fact) VALUES(?,?,?,?,?,?)",
                              (meta["user_row_id"], chat_id, meta["thread_id"], user_id or 0, now, fact_text))
            gid = cur.lastrowid
            vec = embed_text(fact_text)
            if vec:
                con.execute("INSERT INTO embeddings(kind,ref_id,chat_id,thread_id,user_id,date,dim,vec) VALUES(?,?,?,?,?,?,?,?)",
                            ("fact", gid, chat_id, meta["thread_id"], user_id or 0, now, len(vec), json.dumps(vec)))
            con.commit()
        await DBW.run(str(user_db_path(user_id)), _user)
    def _chan(con: sqlite3.Connection):
        cur = con.execute("INSERT INTO gleanings(source_msg_id,chat_id,thread_id,user_id,date,fact) VALUES(?,?,?,?,?,?)",
                          (meta.get("channel_row_id") or 0, chat_id, meta["thread_id"], user_id or 0, now, fact_text))
        gid = cur.lastrowid
        vec = embed_text(fact_text)
        if vec:
            con.execute("INSERT INTO embeddings(kind,ref_id,chat_id,thread_id,user_id,date,dim,vec) VALUES(?,?,?,?,?,?,?,?)",
                        ("fact", gid, chat_id, meta["thread_id"], user_id or 0, now, len(vec), json.dumps(vec)))
        con.commit()
    await DBW.run(str(channel_db_path(chat_id, meta["thread_id"])), _chan)

# ──────────────────────────────────────────────────────────────
# 10) Facts gleaning (queued) + KG triples
# ──────────────────────────────────────────────────────────────
async def glean_facts_background(row_id: int):
    try:
        with closing(db()) as con:
            r = con.execute("SELECT chat_id, thread_id, user_id, text, date FROM messages WHERE id=?",(row_id,)).fetchone()
        if not r: return
        chat_id, thread_id, user_id, text_msg, ts = r
        if not text_msg or len(text_msg.strip()) < 40: return
        thread_text = fetch_thread_context(chat_id, thread_id, limit=12)
        prompt = textwrap.dedent(f"""
            Extract 1-4 short bullet-point facts about who said what to whom or intends to do.
            Keep concise and non-speculative.

            Context:
            {thread_text}

            New message:
            {text_msg}

            Bullets:
        """).strip()
        rr = requests.post(OLLAMA_URL + "/api/chat",
                           json={"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False},
                           timeout=40)
        rr.raise_for_status()
        data = rr.json() or {}
        content = (data.get("message") or {}).get("content") or ""
        if not content: return
        bullets = [b.strip(" -•\t") for b in content.strip().splitlines() if b.strip()]
        now = int(time.time())

        def _global(con: sqlite3.Connection):
            for b in bullets[:4]:
                cur = con.execute("INSERT INTO gleanings(source_msg_id,chat_id,thread_id,user_id,date,fact) VALUES(?,?,?,?,?,?)",
                                  (row_id, chat_id, thread_id or 0, user_id or 0, now, b))
                gid = cur.lastrowid
                vec = embed_text(b)
                if vec:
                    con.execute("INSERT INTO embeddings(kind,ref_id,chat_id,thread_id,user_id,date,dim,vec) VALUES(?,?,?,?,?,?,?,?)",
                                ("fact", gid, chat_id, thread_id or 0, user_id or 0, now, len(vec), json.dumps(vec)))
            con.commit()
        await DBW.run(DB_PATH, _global)

        meta = {"global_row_id": row_id, "user_row_id": None, "channel_row_id": None, "thread_id": thread_id or 0}
        for b in bullets[:4]:
            await mirror_fact_to_shards_async(b, meta, chat_id, user_id)

        await kg_add_triples_for_message_async(row_id)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 11) Commands catalog (omits /setadmin)
# ──────────────────────────────────────────────────────────────
def command_catalog() -> Dict[str, List[tuple]]:
    public = [
        ("/start", "Unlock chat (tap the pinned Start link in the group first)."),
        ("/inspect <link|@user|id>", "Show IDs/metadata for users/chats."),
        ("/commands", "Show available commands tailored to you."),
        ("/topic", "Show top entities & relations in this channel."),
        ("/profile", "Manage your interaction profile (DM me)."),
    ]
    admin = [
        ("/setup", "Make this group read-only & print Start link."),
        ("/ungate", "Restore permissive defaults."),
        ("/link", "Show Start link for this group."),
        ("/linkprimary", "Show Start link for primary group."),
        ("/setprimary", "Mark this group as the primary server."),
        ("/config", "Show config, DB counts, Ollama & graph status."),
        ("/users [here|primary] [page] [size] [q=...]", "List unlocked users."),
        ("/message <user_id|@username> <text...>", "DM a user via the bot."),
        ("/system <text>", "Set the system prompt."),
        ("/inspect <link|@user|id>", "Inspector."),
        ("/graph [here|server] [N]", "Show KG relations."),
        ("/autosystem [why]", "Regenerate the system prompt."),
    ]
    return {"public": public, "admin": admin}

def format_commands_for_user(is_admin: bool) -> str:
    cat = command_catalog()
    lines = ["Available commands:", "\nPublic:"]
    for cmd, help_ in cat["public"]: lines.append(f"  {cmd} — {help_}")
    if is_admin:
        lines.append("\nAdmin:")
        for cmd, help_ in cat["admin"]: lines.append(f"  {cmd} — {help_}")
    return "\n".join(lines)

async def commands_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    is_admin = uid in ADMIN_WHITELIST if uid else False
    await update.effective_message.reply_text(format_commands_for_user(is_admin))

# ──────────────────────────────────────────────────────────────
# 12) Admin gating, inspector, setadmin flow
# ──────────────────────────────────────────────────────────────
def admin_only(fn):
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat; user = update.effective_user
        if not chat or chat.type not in ("group","supergroup"):
            return await update.effective_message.reply_text("Run this in a group.")
        member = await context.bot.get_chat_member(chat.id, user.id)
        if member.status not in ("administrator","creator"):
            return await update.effective_message.reply_text("Admins only.")
        return await fn(update, context)
    return wrapper

def whitelist_only(fn):
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user and user.id in ADMIN_WHITELIST:
            return await fn(update, context)
        return await update.effective_message.reply_text("Not allowed. This command is restricted.")
    return wrapper

async def make_group_readonly(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    perms = ChatPermissions(
        can_send_messages=False, can_send_audios=False, can_send_documents=False, can_send_photos=False,
        can_send_videos=False, can_send_video_notes=False, can_send_voice_notes=False, can_send_polls=False,
        can_send_other_messages=False, can_add_web_page_previews=False, can_change_info=False,
        can_invite_users=False, can_pin_messages=False,
    )
    await context.bot.set_chat_permissions(chat_id, perms)

async def build_start_link(bot_username: str, chat_id: int, title: str | None) -> str:
    token = await get_or_create_group(chat_id, title or "")
    return f"https://t.me/{bot_username}?start=unlock_{token}"

@admin_only
async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await make_group_readonly(context, chat.id)
    bot_username = (await context.bot.get_me()).username
    start_link = await build_start_link(bot_username, chat.id, chat.title)
    global PRIMARY_CHAT_ID
    if not PRIMARY_CHAT_ID:
        PRIMARY_CHAT_ID = str(chat.id); _write_env_var("PRIMARY_CHAT_ID", PRIMARY_CHAT_ID)
    msg = textwrap.dedent(f"""
        ✅ Gating enabled for *{chat.title}*.

        1) Group default is now read-only.
        2) Pin this “Start” link so newcomers can unlock:
           {start_link}

        Primary server chat id: `{PRIMARY_CHAT_ID}`
    """).strip()
    await update.effective_message.reply_text(msg, disable_web_page_preview=True, parse_mode="Markdown")

@admin_only
async def ungate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    perms = ChatPermissions(
        can_send_messages=True, can_send_audios=True, can_send_documents=True, can_send_photos=True, can_send_videos=True,
        can_send_video_notes=True, can_send_voice_notes=True, can_send_polls=True, can_send_other_messages=True,
        can_add_web_page_previews=True, can_change_info=False, can_invite_users=True, can_pin_messages=False,
    )
    await context.bot.set_chat_permissions(chat.id, perms)
    await update.effective_message.reply_text("🚪 Gating disabled and defaults restored.")

@admin_only
async def link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    bot_username = (await context.bot.get_me()).username
    start_link = await build_start_link(bot_username, chat.id, chat.title)
    await update.effective_message.reply_text(f"🔗 Start link for this group:\n{start_link}", disable_web_page_preview=True)

async def linkprimary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not PRIMARY_CHAT_ID:
        return await update.effective_message.reply_text("PRIMARY_CHAT_ID is not set. Run /setprimary in your target group.")
    try:
        chat_id = int(PRIMARY_CHAT_ID); chat = await context.bot.get_chat(chat_id)
        bot_username = (await context.bot.get_me()).username
        start_link = await build_start_link(bot_username, chat.id, chat.title)
        await update.effective_message.reply_text(f"🔗 Primary group: <b>{chat.title}</b>\n{start_link}",
                                                  parse_mode="HTML", disable_web_page_preview=True)
    except Exception:
        await update.effective_message.reply_text(f"Could not fetch PRIMARY_CHAT_ID ({PRIMARY_CHAT_ID}). Try /setprimary in the group.")

@admin_only
async def setprimary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    _write_env_var("PRIMARY_CHAT_ID", str(chat.id))
    global PRIMARY_CHAT_ID; PRIMARY_CHAT_ID = str(chat.id)
    await update.effective_message.reply_text(f"✅ Primary server set to: {chat.title} ({chat.id}). Saved to .env as PRIMARY_CHAT_ID.")

@whitelist_only
async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args=context.args or []; scope,page,size,search=parse_users_args(args)
    if scope=="here":
        chat=update.effective_chat
        if not chat or chat.type not in ("group","supergroup"): scope="primary"
    if scope=="primary":
        if not PRIMARY_CHAT_ID: return await update.effective_message.reply_text("PRIMARY_CHAT_ID is not set.")
        try: chat_id=int(PRIMARY_CHAT_ID); chat=await context.bot.get_chat(chat_id)
        except Exception: return await update.effective_message.reply_text(f"Invalid PRIMARY_CHAT_ID: {PRIMARY_CHAT_ID}")
    else:
        chat_id=update.effective_chat.id; chat=update.effective_chat
    total, rows = list_allowed(chat_id, limit=size, offset=(page-1)*size, search=search)
    if total==0: return await update.effective_message.reply_text("No unlocked users found for this group.")
    parts=[f"<b>Unlocked users for {chat.title}</b> (total {total}, page {page}, size {size}{', q='+search if search else ''})"]
    for uid, uname, first, last in rows:
        name=" ".join(p for p in (first,last) if p).strip() or (uname or f"id {uid}")
        handle=f"@{uname}" if uname else f"id {uid}"
        parts.append(f"• <a href=\"tg://user?id={uid}\">{name}</a> ({handle})")
    if page*size<total: parts.append(f"\nType: /users {scope} {page+1} size={size}" + (f" q={search}" if search else ""))
    await update.effective_message.reply_html("\n".join(parts), disable_web_page_preview=True)

def parse_users_args(args: list[str]) -> tuple[str, int, int, Optional[str]]:
    """
    Returns (scope, page, size, search)
      scope: "here" | "primary"
      page: 1-based
      size: items per page (<= 200)
      search: optional string from q=...
    """
    scope = "here"
    page = 1
    size = 50
    search = None
    for a in args:
        if a in ("here", "primary"):
            scope = a
        elif a.startswith("q="):
            search = a[2:].strip()
        elif re.fullmatch(r"\d+", a):
            page = int(a)
        elif re.fullmatch(r"\d+x\d+", a):
            # "pagexsize" like 2x100
            p, s = a.split("x", 1)
            page, size = int(p), int(s)
        elif a.startswith("size="):
            try:
                size = int(a.split("=",1)[1])
            except: pass
    size = max(1, min(size, 200))
    page = max(1, page)
    return scope, page, size, search

@whitelist_only
async def message_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args)<2: return await update.effective_message.reply_text("Usage: /message <user_id|@username> <text...>")
    target=context.args[0]; text_to=" ".join(context.args[1:]).strip()
    if not text_to: return await update.effective_message.reply_text("Message text cannot be empty.")
    uid=None
    if target.startswith("@"):
        uid = resolve_user_by_username(None, target)
    else:
        try: uid=int(target)
        except ValueError: return await update.effective_message.reply_text("First arg must be numeric user_id or @username.")
    if uid is None: return await update.effective_message.reply_text(f"Could not resolve {target}.")
    try: await context.bot.send_message(chat_id=uid, text=text_to); await update.effective_message.reply_text(f"✅ Sent to {uid}.")
    except Exception: await update.effective_message.reply_text(f"Failed to DM {uid}. The user must have started the bot at least once.")


@whitelist_only
async def config_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    me = await context.bot.get_me()
    here = update.effective_chat
    with closing(db()) as con:
        gcount = con.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
        acount = con.execute("SELECT COUNT(*) FROM allowed").fetchone()[0]
        mcount = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        servers_seen = con.execute("SELECT COUNT(DISTINCT chat_id) FROM messages WHERE chat_type!='private'").fetchone()[0]
        users_seen = con.execute("SELECT COUNT(DISTINCT user_id) FROM messages WHERE user_id IS NOT NULL").fetchone()[0]
        facts_n = con.execute("SELECT COUNT(*) FROM gleanings").fetchone()[0]
        emb_n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        nodes_n = con.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        edges_n = con.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        prof_idx = con.execute("SELECT COUNT(*) FROM user_profiles_idx").fetchone()[0]
    meta = settings_get("AUTO_SYS_META","")
    lines = [
        f"<b>Bot</b>: @{me.username} (id={me.id})",
        f"<b>DB</b>: {DB_PATH} — groups:{gcount}, allowed:{acount}, msgs:{mcount}, facts:{facts_n}, embeds:{emb_n}, profiles_idx:{prof_idx}",
        f"<b>KG</b>: nodes:{nodes_n}, edges:{edges_n}",
        f"<b>Seen</b>: servers={servers_seen}, users={users_seen}",
        f"<b>Admin whitelist</b>: {', '.join(str(i) for i in sorted(ADMIN_WHITELIST)) or '(empty)'}",
        f"<b>Ollama</b>: url={OLLAMA_URL} model={OLLAMA_MODEL or '(unset)'} embed={OLLAMA_EMBED_MODEL or '(unset)'}",
        f"<b>Auto reply</b>: {AUTO_REPLY_MODE}",
        f"<b>Auto System Prompt</b>: {'enabled' if AUTO_SYS_ENABLED else 'disabled'} every {AUTO_SYS_INTERVAL_HOURS}h",
        f"<b>Profiles</b>: {'enabled' if PROFILES_ENABLED else 'disabled'}, refresh ~{PROFILE_REFRESH_MINUTES}m",
    ]
    if meta: lines.append(f"<b>Prompt meta</b>: {meta}")
    if PRIMARY_CHAT_ID:
        try: pcid=int(PRIMARY_CHAT_ID); pchat=await context.bot.get_chat(pcid); lines.append(f"<b>Primary</b>: {pchat.title} (id={pcid})")
        except Exception: lines.append(f"<b>Primary</b>: (invalid) {PRIMARY_CHAT_ID}")
    if here and here.type in ("group","supergroup"): lines.append(f"<b>Here</b>: {here.title} (id={here.id})")
    await update.effective_message.reply_html("\n".join(lines), disable_web_page_preview=True)

# /setadmin bootstrap & approvals (unchanged)
async def setadmin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    requester=update.effective_user
    if requester is None: return await update.effective_message.reply_text("No user found.")
    rid=requester.id
    if not ADMIN_WHITELIST:
        add_admin_ids([rid]); return await update.effective_message.reply_text("✅ You are now an admin (bootstrap).")
    if rid in ADMIN_WHITELIST: return await update.effective_message.reply_text("You are already an admin.")
    kb=InlineKeyboardMarkup([[InlineKeyboardButton("✅ Approve", callback_data=f"adminreq:approve:{rid}"),
                              InlineKeyboardButton("✖️ Deny",   callback_data=f"adminreq:deny:{rid}")]])
    name=" ".join(p for p in ((requester.first_name or ""), (requester.last_name or "")) if p) or (requester.username or f"id {rid}")
    handle=f"@{requester.username}" if requester.username else f"id {rid}"
    text=f"🔐 Admin request received.\n\nRequester: {name} ({handle})\nUser ID: <code>{rid}</code>\n\nApprove or deny this admin request."
    sent=0
    for aid in list(ADMIN_WHITELIST):
        try: await context.bot.send_message(aid, text, parse_mode="HTML", reply_markup=kb); sent+=1
        except Exception: pass
    if sent==0: return await update.effective_message.reply_text("No admins reachable. Ask an admin to DM me once, then try again.")
    return await update.effective_message.reply_text(f"Request sent to {sent} admin(s).")

async def adminreq_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query; await q.answer()
    m=re.match(r"^adminreq:(approve|deny):(\d+)$", q.data or "")
    if not m: return
    action, uid_str=m.group(1), m.group(2); actor=q.from_user
    if actor.id not in ADMIN_WHITELIST: return await q.answer("Only admins can do this.", show_alert=True)
    uid=int(uid_str)
    if action=="approve":
        if uid in ADMIN_WHITELIST:
            try: await q.edit_message_text(f"Already admin: <code>{uid}</code>", parse_mode="HTML")
            except Exception: pass
            return
        add_admin_ids([uid])
        try: await context.bot.send_message(uid, "✅ Your admin request has been approved.")
        except Exception: pass
        try: await q.edit_message_text(f"✅ Approved admin: <code>{uid}</code>", parse_mode="HTML")
        except Exception: pass
    else:
        try: await context.bot.send_message(uid, "❌ Your admin request was denied.")
        except Exception: pass
        try: await q.edit_message_text(f"❌ Denied admin request: <code>{uid}</code>", parse_mode="HTML")
        except Exception: pass

# Inspector
INSPECT_PATTERNS = [
    re.compile(r'(?:https?://)?t\.me/c/(\d+)(?:/\d+)?', re.I),
    re.compile(r'(?:https?://)?t\.me/(?:joinchat/|\+)([A-Za-z0-9_-]+)', re.I),
    re.compile(r'(?:https?://)?t\.me/([A-Za-z0-9_]{5,})', re.I),
    re.compile(r'tg://user\?id=(\d+)', re.I),
    re.compile(r'^@([A-Za-z0-9_]{5,})$'),
    re.compile(r'^(?:-?\d{7,})$'),
]

async def do_inspect(update: Update, context: ContextTypes.DEFAULT_TYPE, target: str):
    target=target.strip(); msg=update.effective_message; bot=context.bot
    m=INSPECT_PATTERNS[0].search(target)
    if m:
        try:
            cid=int(m.group(1)); chat_id=int(f"-100{cid}"); chat=await bot.get_chat(chat_id)
            return await msg.reply_html(f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\nusername: @{getattr(chat,'username','') or '(none)'}")
        except Exception: return await msg.reply_text("Couldn’t resolve that private /c/ link (bot must be a member).")
    m=INSPECT_PATTERNS[1].search(target)
    if m: return await msg.reply_text("Invite links can’t be resolved unless the bot joins that chat.")
    m=INSPECT_PATTERNS[2].search(target)
    if m:
        uname=m.group(1)
        try:
            chat=await bot.get_chat(f"@{uname}")
            return await msg.reply_html(f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\nusername: @{getattr(chat,'username','') or '(none)'}")
        except Exception:
            uid=resolve_user_by_username(None, uname)
            if uid: return await msg.reply_html(f"<b>User</b>\nid: <code>{uid}</code>\nusername: @{uname}")
            return await msg.reply_text("Couldn’t resolve that username.")
    m=INSPECT_PATTERNS[3].search(target)
    if m: uid=int(m.group(1)); return await msg.reply_html(f"<b>User</b>\nid: <code>{uid}</code>")
    m=INSPECT_PATTERNS[4].search(target)
    if m:
        uname=m.group(1); uid=resolve_user_by_username(None, uname)
        if uid: return await msg.reply_html(f"<b>User</b>\nid: <code>{uid}</code>\nusername: @{uname}")
        try:
            chat=await bot.get_chat(f"@{uname}")
            return await msg.reply_html(f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\nusername: @{getattr(chat,'username','') or '(none)'}")
        except Exception: return await msg.reply_text("Couldn’t resolve that handle.")
    m=INSPECT_PATTERNS[5].search(target)
    if m:
        try:
            ident=int(m.group(0))
            try:
                chat=await bot.get_chat(ident)
                return await msg.reply_html(f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\nusername: @{getattr(chat,'username','') or '(none)'}")
            except Exception:
                return await msg.reply_html(f"<b>User</b>\nid: <code>{ident}</code>")
        except Exception: pass
    return await msg.reply_text("Send a t.me / tg:// link, @username, or a numeric id to inspect.")

@whitelist_only
async def inspect_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: return await update.effective_message.reply_text("Usage: /inspect <link|@username|id>")
    return await do_inspect(update, context, " ".join(context.args))

# Topic/Graph
def kg_top_relations(chat_id: int, thread_id: int, limit: int = 12) -> List[str]:
    with closing(db()) as con:
        rows = con.execute(
            """SELECT e.rel, ns.label, nd.label, SUM(e.weight) AS w
               FROM kg_edges e
               JOIN kg_nodes ns ON ns.id=e.src_id
               JOIN kg_nodes nd ON nd.id=e.dst_id
               WHERE e.chat_id=? AND e.thread_id=?
               GROUP BY e.rel, ns.label, nd.label
               ORDER BY w DESC, e.ts DESC
               LIMIT ?""",
            (chat_id, int(thread_id or 0), limit)
        ).fetchall()
    return [f"{s} —{rel}→ {o}" for (rel, s, o, _w) in rows]

def kg_top_entities(chat_id: int, thread_id: int, limit: int = 12) -> List[str]:
    with closing(db()) as con:
        rows = con.execute(
            """SELECT n.label, SUM(e.weight) AS w
               FROM kg_nodes n
               JOIN kg_edges e ON (e.src_id=n.id OR e.dst_id=n.id)
               WHERE e.chat_id=? AND e.thread_id=?
               GROUP BY n.id
               ORDER BY w DESC
               LIMIT ?""",
            (chat_id, int(thread_id or 0), limit)
        ).fetchall()
    return [f"{lab}" for (lab, _w) in rows]

async def topic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat; m = update.effective_message
    if not chat or chat.type not in ("group","supergroup","channel"):
        return await m.reply_text("Run this in a group/channel.")
    thread_id = getattr(m, "message_thread_id", None) or 0
    ents = kg_top_entities(chat.id, thread_id, limit=10)
    rels = kg_top_relations(chat.id, thread_id, limit=10)
    if not ents and not rels: return await m.reply_text("No graph signals here yet.")
    text = "<b>Channel snapshot</b>\n"
    if ents: text += "Top entities:\n• " + "\n• ".join(ents) + "\n"
    if rels: text += "\nTop relations:\n• " + "\n• ".join(rels)
    await m.reply_html(text)

@whitelist_only
async def graph_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    scope = "here"; topn = 12
    for a in args:
        if a in ("here","server"): scope=a
        elif re.fullmatch(r"\d+", a): topn = max(3, min(40, int(a)))
    chat = update.effective_chat
    if not chat or chat.type not in ("group","supergroup","channel"):
        return await update.effective_message.reply_text("Run this in a group/channel.")
    if scope == "here":
        thread_id = getattr(update.effective_message, "message_thread_id", None) or 0
        ents = kg_top_entities(chat.id, thread_id, limit=topn)
        rels = kg_top_relations(chat.id, thread_id, limit=topn)
        hdr = f"<b>Graph here</b> (chat {chat.id}, topic {thread_id})"
    else:
        with closing(db()) as con:
            rows = con.execute(
                """SELECT e.rel, ns.label, nd.label, SUM(e.weight) AS w
                   FROM kg_edges e
                   JOIN kg_nodes ns ON ns.id=e.src_id
                   JOIN kg_nodes nd ON nd.id=e.dst_id
                   WHERE e.chat_id=?
                   GROUP BY e.rel, ns.label, nd.label
                   ORDER BY w DESC LIMIT ?""",
                (chat.id, topn)
            ).fetchall()
            ents_rows = con.execute(
                """SELECT n.label, SUM(e.weight) AS w
                   FROM kg_nodes n
                   JOIN kg_edges e ON (e.src_id=n.id OR e.dst_id=n.id)
                   WHERE e.chat_id=?
                   GROUP BY n.id ORDER BY w DESC LIMIT ?""",
                (chat.id, topn)
            ).fetchall()
        rels = [f"{s} —{rel}→ {o}" for (rel, s, o, _w) in rows]
        ents = [f"{lab}" for (lab,_w) in ents_rows]
        hdr = f"<b>Graph server-wide</b> (chat {chat.id})"
    if not ents and not rels: return await update.effective_message.reply_text("No graph yet.")
    text = hdr + "\n"
    if ents: text += "Top entities:\n• " + "\n• ".join(ents) + "\n"
    if rels: text += "\nTop relations:\n• " + "\n".join(rels)
    await update.effective_message.reply_html(text)

# ──────────────────────────────────────────────────────────────
# 13) Auto System Prompt (unchanged behavior; queued settings writes)
# ──────────────────────────────────────────────────────────────
def _top_global_entities(limit=10) -> List[str]:
    with closing(db()) as con:
        rows = con.execute(
            """SELECT n.label, SUM(e.weight) AS w
               FROM kg_nodes n JOIN kg_edges e ON (e.src_id=n.id OR e.dst_id=n.id)
               GROUP BY n.id ORDER BY w DESC LIMIT ?""", (limit,)
        ).fetchall()
    return [r[0] for r in rows]

def _top_server_entities(chat_id: int, limit=10) -> List[str]:
    with closing(db()) as con:
        rows = con.execute(
            """SELECT n.label, SUM(e.weight) AS w
               FROM kg_nodes n JOIN kg_edges e ON (e.src_id=n.id OR e.dst_id=n.id)
               WHERE e.chat_id=? GROUP BY n.id ORDER BY w DESC LIMIT ?""", (chat_id, limit)
        ).fetchall()
    return [r[0] for r in rows]

def _commands_list_text() -> str:
    cat = command_catalog()
    pub = "\n".join([f"- {c} — {h}" for c,h in cat["public"]])
    adm = "\n".join([f"- {c} — {h}" for c,h in cat["admin"]])
    return f"Public:\n{pub}\n\nAdmin:\n{adm}"

async def build_environment_digest(context: ContextTypes.DEFAULT_TYPE) -> str:
    me = await context.bot.get_me()
    host = socket.gethostname()
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    metrics = fetch_metrics(MAX_CONTEXT_USERS, MAX_CONTEXT_GROUPS)
    primary_title = ""; primary_id = None
    try:
        if PRIMARY_CHAT_ID:
            primary_id = int(PRIMARY_CHAT_ID)
            pchat = await context.bot.get_chat(primary_id)
            primary_title = pchat.title or str(primary_id)
    except Exception:
        primary_title = "(unavailable)"
    top_global = _top_global_entities(12)
    top_primary = _top_server_entities(primary_id, 12) if primary_id else []
    user_shards = len(list(USERS_DIR.glob("user_*.db")))
    chan_shards = len(list(CHANNELS_DIR.glob("chat*_topic*.db")))
    admins_txt = ", ".join(str(i) for i in sorted(ADMIN_WHITELIST)) or "(none)"
    cmds = _commands_list_text()
    return textwrap.dedent(f"""
        DIGEST_VERSION: 1
        NOW_LOCAL: {now_local}
        NOW_UTC: {now_utc}
        HOST: {host}
        BOT: @{me.username} (id={me.id})
        OLLAMA: url={OLLAMA_URL}, chat_model={OLLAMA_MODEL}, embed_model={OLLAMA_EMBED_MODEL}
        AUTO_REPLY_MODE: {AUTO_REPLY_MODE}
        PRIMARY_CHAT: {primary_title} ({PRIMARY_CHAT_ID or '(unset)'})
        ADMINS_WHITELIST_IDS: {admins_txt}

        METRICS:
          servers_seen={metrics['servers_count']}
          users_seen={metrics['users_count']}
          shard_users={user_shards}
          shard_channels={chan_shards}

        TOP_ENTITIES_GLOBAL: {', '.join(top_global) or '(none)'}
        TOP_ENTITIES_PRIMARY: {', '.join(top_primary) or '(none)'}

        COMMANDS (omit /setadmin to non-admins):
        {cmds}

        POLICY HINTS:
          - Replies concise & channel-aware; use KG & similarity context; no speculation.
          - Reveal admin commands only to admins (never surface /setadmin to non-admins).
          - Mentions must be answered; otherwise SMART heuristics.
          - Guide newcomers to Start link when gated.
    """).strip()

def _sanitize_system_prompt(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n", "", t); t = re.sub(r"\n```$", "", t)
    return t.strip()

async def autosys_meta_update(source: str, status: str, chars: int):
    await settings_set_async("AUTO_SYS_META", f"{source} @ {datetime.now().isoformat(timespec='seconds')} — {status}, {chars} chars")

async def auto_generate_system_prompt(context: ContextTypes.DEFAULT_TYPE, include_digest_in_meta: bool=False) -> Optional[str]:
    digest = await build_environment_digest(context)
    designer = textwrap.dedent(f"""
        Write a SYSTEM PROMPT for "Gatekeeper Bot" tailored to the environment below.
        Keep under ~1200 words. Include: Decision Rules; When to say no; Tone & style; Quick reminders.
        Show only commands the user can actually use (omit /setadmin unless admin). Plain text only.

        ENVIRONMENT DIGEST:
        {digest}
    """).strip()
    payload = {"model": OLLAMA_MODEL, "messages":[{"role":"user","content":designer}], "stream": False}
    new_prompt = await ai_generate_async(payload)
    if not new_prompt:
        await autosys_meta_update("auto", "ollama-failed", 0); return None
    new_prompt = _sanitize_system_prompt(new_prompt)
    await autosys_meta_update("auto", "ok" + ("+why" if include_digest_in_meta else ""), len(new_prompt))
    return new_prompt

async def set_system_prompt_and_persist(new_prompt: str):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = new_prompt.strip()
    _write_env_var("SYSTEM_PROMPT", SYSTEM_PROMPT.replace("\n","\\n"))


@whitelist_only
async def system_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SYSTEM_PROMPT
    msg=update.effective_message
    if not context.args:
        preview=(SYSTEM_PROMPT[:500]+"…") if len(SYSTEM_PROMPT)>500 else SYSTEM_PROMPT
        meta=settings_get("AUTO_SYS_META","")
        return await msg.reply_text(f"Current SYSTEM_PROMPT ({len(SYSTEM_PROMPT)} chars):\n{preview or '(empty)'}\n\nMeta: {meta or '(none)'}")
    SYSTEM_PROMPT=" ".join(context.args).strip()
    _write_env_var("SYSTEM_PROMPT", SYSTEM_PROMPT.replace("\n","\\n"))
    await settings_set_async("AUTO_SYS_META", f"manual-set at {datetime.now().isoformat(timespec='seconds')}")
    await msg.reply_text("✅ SYSTEM_PROMPT updated.")
    
@whitelist_only
async def autosystem_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update.effective_chat else None
    show_why = (context.args and any(a.lower()=="why" for a in context.args))
    async def _work():
        p = await auto_generate_system_prompt(context, include_digest_in_meta=show_why)
        if not p: return None
        await set_system_prompt_and_persist(p); return p
    prompt = await with_typing(context, chat_id, _work()) if chat_id is not None else await _work()
    if not prompt: return await update.effective_message.reply_text("❌ Auto-prompt generation failed (Ollama not responding).")
    prev = (prompt[:500]+"…") if len(prompt)>500 else prompt
    await update.effective_message.reply_text(f"✅ System prompt regenerated ({len(prompt)} chars).\n\nPreview:\n{prev}")

async def autosystem_job(context: ContextTypes.DEFAULT_TYPE):
    if not AUTO_SYS_ENABLED: return
    try:
        p = await auto_generate_system_prompt(context, include_digest_in_meta=False)
        if p: await set_system_prompt_and_persist(p)
    except Exception: pass

# ──────────────────────────────────────────────────────────────
# 14) “Interaction profiles” (privacy-aware)
# ──────────────────────────────────────────────────────────────
def _emoji_count(s: str) -> int:
    return len(re.findall(r'[\U0001F300-\U0001FAFF]', s))

async def profiles_idx_get(user_id: int) -> Tuple[int,int]:
    with closing(db()) as con:
        r = con.execute("SELECT COALESCE(last_updated,0), COALESCE(optout,0) FROM user_profiles_idx WHERE user_id=?", (user_id,)).fetchone()
    if not r: return 0, 0
    return int(r[0] or 0), int(r[1] or 0)

async def profiles_idx_set(user_id: int, last_updated: int = None, optout: Optional[bool] = None):
    def _work(con: sqlite3.Connection):
        row = con.execute("SELECT user_id FROM user_profiles_idx WHERE user_id=?", (user_id,)).fetchone()
        if row:
            if last_updated is not None:
                con.execute("UPDATE user_profiles_idx SET last_updated=? WHERE user_id=?", (int(last_updated), user_id))
            if optout is not None:
                con.execute("UPDATE user_profiles_idx SET optout=? WHERE user_id=?", (1 if optout else 0, user_id))
        else:
            con.execute("INSERT INTO user_profiles_idx(user_id,last_updated,optout) VALUES(?,?,?)",
                        (user_id, int(last_updated or 0), (1 if (optout or False) else 0)))
        con.commit()
    await DBW.run(DB_PATH, _work)

async def update_profile_for_user(user_id: int):
    if not PROFILES_ENABLED: return
    last_upd, optout = await profiles_idx_get(user_id)
    if optout: return
    # pull last N messages from this user across all chats
    with closing(db()) as con:
        rows = con.execute(
            "SELECT text FROM messages WHERE user_id=? AND text IS NOT NULL AND TRIM(text)!='' ORDER BY date DESC LIMIT 300",
            (user_id,)
        ).fetchall()
    if not rows: 
        await profiles_idx_set(user_id, int(time.time()))
        return
    texts = [r[0] for r in rows]
    msg_count = len(texts)
    lens = [len(t) for t in texts]
    avg_len = (sum(lens)/msg_count) if msg_count else 0.0
    question_ratio = sum(1 for t in texts if "?" in t)/msg_count
    emoji_ratio = sum(1 for t in texts if _emoji_count(t)>0)/msg_count
    link_ratio = sum(1 for t in texts if "http://" in t or "https://" in t)/msg_count
    code_ratio = sum(1 for t in texts if "```" in t or "`" in t or re.search(r"\bclass\b|\bdef\b|\bfunction\b|{.*}", t))/msg_count

    # LLM summary (privacy-aware, no diagnosis, no protected attributes)
    sample = "\n".join(list(reversed(texts[:25])))[-4000:]
    prompt = textwrap.dedent(f"""
        From the chat excerpts below, derive a brief, neutral "interaction summary" and a short list of
        "communication tips" for how a bot should best communicate with this user.
        Constraints:
          - DO NOT infer or mention health status, politics, religion, ethnicity, sexuality, or sensitive traits.
          - DO NOT psychoanalyze or diagnose.
          - Focus on observable communication patterns only (e.g., brevity, formality, emoji use, technical depth, patience).
          - Keep it helpful and respectful.
        Format:
          SUMMARY: <3-6 compact sentences focusing on communication behavior>
          TIPS:
          - <tip 1>
          - <tip 2>
          - <tip 3>

        EXCERPTS (chronological snippet):
        {sample}
    """).strip()
    payload = {"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False}
    content = await ai_generate_async(payload)
    if not content: content = "SUMMARY: (not enough data)\nTIPS:\n- Keep replies clear and concise.\n- Ask clarifying questions when needed.\n- Provide examples when explaining."

    # positivity (very rough, bounded 0..1) — privacy-safe tone score
    pos_prompt = textwrap.dedent(f"""
        Rate the overall tone of the excerpts on a 0..1 scale (0 very negative, 1 very positive).
        Output ONLY the number.
        Excerpts:
        {sample}
    """).strip()
    pos_raw = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":pos_prompt}], "stream": False}) or "0.5"
    try: positivity = max(0.0, min(1.0, float(re.findall(r"[0-1](?:\.\d+)?", pos_raw)[0])))
    except Exception: positivity = 0.5

    # persist into per-user shard
    path = str(user_db_path(user_id))
    now = int(time.time())
    def _work(con: sqlite3.Connection):
        con.execute("""INSERT INTO profile(user_id,version,last_updated,msg_count,avg_len,question_ratio,emoji_ratio,link_ratio,code_ratio,positivity,style_notes,suggestions)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(user_id) DO UPDATE SET
                         version=excluded.version,
                         last_updated=excluded.last_updated,
                         msg_count=excluded.msg_count,
                         avg_len=excluded.avg_len,
                         question_ratio=excluded.question_ratio,
                         emoji_ratio=excluded.emoji_ratio,
                         link_ratio=excluded.link_ratio,
                         code_ratio=excluded.code_ratio,
                         positivity=excluded.positivity,
                         style_notes=excluded.style_notes,
                         suggestions=excluded.suggestions
        """,
        (user_id, 1, now, msg_count, avg_len, question_ratio, emoji_ratio, link_ratio, code_ratio, positivity, content[:3000], ""))
        con.commit()
    await DBW.run(path, _work)
    await profiles_idx_set(user_id, last_updated=now)

async def profile_background_job(context: ContextTypes.DEFAULT_TYPE):
    if not PROFILES_ENABLED: return
    try:
        # candidates: most recently active users
        with closing(db()) as con:
            cands = con.execute("""
                SELECT user_id, MAX(date) as last_seen
                FROM messages
                WHERE user_id IS NOT NULL
                GROUP BY user_id
                ORDER BY last_seen DESC
                LIMIT 50
            """).fetchall()
        updated = 0
        for (uid, _last_seen) in cands:
            last_upd, optout = await profiles_idx_get(int(uid))
            if optout: continue
            if (int(time.time()) - last_upd) >= PROFILE_REFRESH_MINUTES*60:
                await update_profile_for_user(int(uid))
                updated += 1
                if updated >= 3: break
    except Exception:
        # swallow background errors
        pass

# /profile command (DM)
async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.effective_message; chat = update.effective_chat; user = update.effective_user
    if chat.type != "private":
        return await m.reply_text("DM me to manage your interaction profile: open chat and send /profile.")
    args = [a.lower() for a in (context.args or [])]
    if not args or args[0] in ("help","?"):
        return await m.reply_text(textwrap.dedent(f"""
            Your interaction profile helps me tailor responses (length, clarity, examples, etc.).
            I avoid sensitive traits and never diagnose you.

            Commands:
              /profile show        — Display your current profile
              /profile now         — Refresh it right away
              /profile erase       — Delete it from my storage
              /profile optout on   — Stop building/updating your profile
              /profile optout off  — Resume profile updates
        """).strip())
    if args[0] == "show":
        path = str(user_db_path(user.id))
        with closing(sqlite3.connect(path)) as con:
            ensure_schema_on(con)
            row = con.execute("SELECT last_updated,msg_count,avg_len,question_ratio,emoji_ratio,link_ratio,code_ratio,positivity,style_notes FROM profile WHERE user_id=?", (user.id,)).fetchone()
        if not row:
            return await m.reply_text("No profile yet. Use /profile now to build it.")
        last_updated,msg_count,avg_len,qr,er,lr,cr,pos,notes = row
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_updated or 0))
        txt = textwrap.dedent(f"""
            Profile last updated: {ts}
            Messages analyzed: {msg_count}
            Avg chars: {avg_len:.0f}
            ?-ratio: {qr:.2f} | emoji-ratio: {er:.2f} | link-ratio: {lr:.2f} | code-ratio: {cr:.2f}
            Positivity (0..1): {pos:.2f}

            {notes}
        """).strip()
        return await m.reply_text(txt)
    if args[0] == "now":
        await with_typing(context, chat.id, update_profile_for_user(user.id))
        return await m.reply_text("✅ Profile refreshed (or queued). Try /profile show.")
    if args[0] == "erase":
        path = str(user_db_path(user.id))
        def _wipe(con: sqlite3.Connection):
            con.execute("DELETE FROM profile WHERE user_id=?", (user.id,))
            con.commit()
        await DBW.run(path, _wipe)
        await profiles_idx_set(user.id, last_updated=0)
        return await m.reply_text("🗑️ Deleted your profile.")
    if args[0] == "optout":
        if len(args) < 2 or args[1] not in ("on","off"):
            return await m.reply_text("Usage: /profile optout on|off")
        await profiles_idx_set(user.id, optout=(args[1]=="on"))
        return await m.reply_text(f"✅ Opt-out {'enabled' if args[1]=='on' else 'disabled'}.")
    return await m.reply_text("Unknown subcommand. Try /profile help.")

# ──────────────────────────────────────────────────────────────
# 15) AI prompt builder + heuristics
# ──────────────────────────────────────────────────────────────
def ai_available() -> bool:
    return bool(OLLAMA_MODEL and OLLAMA_URL)

def build_ai_prompt(user_text: str, current_user, chat_id: int, thread_id: int,
                    thread_ctx: str, similar_blend: Dict[str,str], kg_snapshot: Dict[str,List[str]]) -> dict:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S (local)")
    metrics = fetch_metrics(MAX_CONTEXT_USERS, MAX_CONTEXT_GROUPS)
    cat = command_catalog()
    public_block = "\n".join([f"- {c} — {h}" for c,h in cat["public"]])
    admin_block = "\n".join([f"- {c} — {h}" for c,h in cat["admin"]])
    uid = current_user.id if current_user else None
    uhandle = f"@{current_user.username}" if (current_user and current_user.username) else "(no-username)"
    uname = " ".join(p for p in ((getattr(current_user,"first_name","") or ""), (getattr(current_user,"last_name","") or "")) if p) or uhandle
    is_admin = (uid in ADMIN_WHITELIST) if uid else False
    chron_global = fetch_context_messages(MAX_CONTEXT_MESSAGES, MAX_CONTEXT_CHARS)

    sys_prompt = SYSTEM_PROMPT or (
        "You are Gatekeeper Bot. Be succinct and accurate. Prefer channel-local context, then server, then global. "
        "Use the GRAPH and similarities. Show only commands the user can use; don't mention /setadmin to non-admins."
    )
    graph_here = "\n".join(kg_snapshot.get("rels_here", [])[:10]) or "(none)"
    ents_here = "\n".join(kg_snapshot.get("ents_here", [])[:10]) or "(none)"
    blended = f"SIMILAR_CHANNEL:\n{similar_blend.get('channel') or '(none)'}\n\nSIMILAR_SERVER:\n{similar_blend.get('server') or '(none)'}\n\nSIMILAR_GLOBAL:\n{similar_blend.get('global') or '(none)'}"

    preface = textwrap.dedent(f"""
        NOW_UTC: {now_utc}
        NOW_LOCAL: {now_local}

        METRICS:
          servers_seen: {metrics['servers_count']}
          users_seen: {metrics['users_count']}

        COMMAND_CATALOG (omitting /setadmin):
          public:
        {public_block or '(none)'}
          admin:
        {admin_block or '(none)'}

        CURRENT_USER:
          id: {uid}
          handle: {uhandle}
          display_name: {uname}
          is_admin: {is_admin}

        GRAPH_CONTEXT (server={chat_id}, channel={thread_id or 0}):
          Top entities:
        {ents_here}
          Top relations:
        {graph_here}

        THREAD_CONTEXT (recent here):
        {thread_ctx or '(none)'}

        BLENDED_SIMILAR (channel > server > global):
        {blended}

        GLOBAL_CHRONOLOGY (trimmed):
        {chron_global or '(none)'}
    """).strip()

    messages = [{"role":"system","content":sys_prompt},
                {"role":"user","content":preface + "\n\nUser says:\n" + user_text}]
    return {"model": OLLAMA_MODEL, "messages": messages, "stream": False}

BOT_CACHE = {"username": None, "id": None}
async def ensure_me(context: ContextTypes.DEFAULT_TYPE):
    if BOT_CACHE["username"] is None or BOT_CACHE["id"] is None:
        me = await context.bot.get_me()
        BOT_CACHE["username"] = me.username; BOT_CACHE["id"] = me.id

def mentions_bot(m) -> bool:
    uname = BOT_CACHE["username"]; 
    if not uname: return False
    text = (m.text or "").lower()
    if f"@{uname.lower()}" in text: return True
    if m.entities:
        for e in m.entities:
            if e.type == MessageEntity.MENTION:
                try: ent = (m.text or "")[e.offset:e.offset+e.length]
                except Exception: ent = ""
                if ent.lower() == f"@{uname.lower()}": return True
            if e.type == MessageEntity.TEXT_MENTION and getattr(e,"user",None) and e.user.id == BOT_CACHE["id"]:
                return True
    return False

def should_auto_reply(m, thread_ctx_text: str) -> bool:
    if AUTO_REPLY_MODE != "smart": return False
    text = (m.text or "").strip()
    if not text: return False
    recent_has_bot = (BOT_CACHE["username"] and BOT_CACHE["username"].lower() in (thread_ctx_text or "").lower())
    is_question = "?" in text
    starts_with = text.lower().startswith(("bot","hey bot","gatekeeper","help","how do i","why "))
    return (is_question and recent_has_bot) or starts_with

# ──────────────────────────────────────────────────────────────
# 16) Text handler
# ──────────────────────────────────────────────────────────────
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await ensure_me(context)
    m = update.effective_message; chat = update.effective_chat; user = update.effective_user
    if not m or not chat: return

    meta = await mirror_message_to_shards(update)
    if meta["global_row_id"] is not None:
        await kg_ingest_message_signals_async(meta["global_row_id"])

    if m.text or m.caption:
        asyncio.create_task(shard_embeddings_for_message((m.text or m.caption), meta, chat.id, user.id if user else None))

    if meta["global_row_id"] is not None and random.random() < 0.25:
        asyncio.create_task(glean_facts_background(meta["global_row_id"]))

    # schedule/refresh profile for this user opportunistically
    if PROFILES_ENABLED and user:
        asyncio.create_task(update_profile_for_user(user.id))

    # DM: always reply
    if chat.type == "private":
        text = (m.text or "").strip()
        if not text: return
        thread_id = 0
        thread_ctx = fetch_thread_context(chat.id, thread_id, limit=24)
        sim = await similar_context(text, chat.id, thread_id, top_k=8)
        kg_snap = {"ents_here": kg_top_entities(chat.id, thread_id, 10),
                   "rels_here": kg_top_relations(chat.id, thread_id, 10)}
        if ai_available():
            payload = build_ai_prompt(text, user, chat.id, thread_id, thread_ctx, sim, kg_snap)
            resp = await with_typing(context, chat.id, ai_generate_async(payload))
            return await m.reply_text(resp or "AI is unavailable right now (Ollama not responding).")
        else:
            return await m.reply_text("AI is disabled (set OLLAMA_URL and OLLAMA_MODEL/OLLAMA_EMBED_MODEL in .env).")

    # Groups/channels: respond on @mention/reply, else maybe (smart)
    thread_id = getattr(m, "message_thread_id", None) or 0
    must = mentions_bot(m) or (getattr(m,"reply_to_message",None) and m.reply_to_message.from_user and m.reply_to_message.from_user.id == BOT_CACHE["id"])
    smart = should_auto_reply(m, fetch_thread_context(chat.id, thread_id, limit=16))
    if not (must or smart): return
    text = (m.text or "").strip()
    if not text: return
    thread_ctx = fetch_thread_context(chat.id, thread_id, limit=24)
    sim = await similar_context(text, chat.id, thread_id, top_k=8)
    kg_snap = {"ents_here": kg_top_entities(chat.id, thread_id, 10),
               "rels_here": kg_top_relations(chat.id, thread_id, 10)}
    if ai_available():
        payload = build_ai_prompt(text, user, chat.id, thread_id, thread_ctx, sim, kg_snap)
        resp = await with_typing(context, chat.id, ai_generate_async(payload))
        if resp:
            try: return await m.reply_text(resp)
            except Exception: return await context.bot.send_message(chat_id=chat.id, text=resp)

# ──────────────────────────────────────────────────────────────
# 17) Unlock flow + greetings
# ──────────────────────────────────────────────────────────────
async def allow_user(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user) -> None:
    perms = ChatPermissions(
        can_send_messages=True, can_send_audios=True, can_send_documents=True,
        can_send_photos=True, can_send_videos=True, can_send_video_notes=True,
        can_send_voice_notes=True, can_send_polls=True, can_send_other_messages=True, can_add_web_page_previews=True,
    )
    await context.bot.restrict_chat_member(chat_id=chat_id, user_id=user.id, permissions=perms,
                                           use_independent_chat_permissions=True)
    await mark_allowed_async(chat_id, user)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args=(context.args or []); token=None
    if args and args[0].startswith("unlock_"): token=args[0].split("unlock_",1)[1]
    if not token:
        return await update.message.reply_text("Hi! To chat in a gated group, tap the pinned Start link in that group.\nAdmins: run /setup in the group to enable gating.")
    chat_id=token_to_chat_id(token)
    if not chat_id: return await update.message.reply_text("That link is invalid or expired. Ask an admin for a fresh one.")
    kb=InlineKeyboardMarkup([[InlineKeyboardButton("✅ I agree — unlock me", callback_data=f"agree:{chat_id}")]])
    await update.message.reply_text("Before we unlock you, please confirm you’ve read the group rules.", reply_markup=kb)

async def agree_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q=update.callback_query; await q.answer()
    data=q.data or ""; 
    if not data.startswith("agree:"): return
    chat_id=int(data.split("agree:",1)[1]); user=q.from_user
    try:
        member=await context.bot.get_chat_member(chat_id, user.id)
        if member.status in ("left","kicked"):
            return await q.edit_message_text("Join the group first, then tap the Start link again.")
    except Exception:
        return await q.edit_message_text("I couldn’t check your membership. Ask an admin to verify the setup.")
    try:
        await allow_user(context, chat_id, user)
    except Exception:
        return await q.edit_message_text("Couldn’t unlock you (bot needs admin rights with 'Manage Members').")
    await q.edit_message_text("✅ You’re unlocked. You can now receive messages from me.")
    try: await context.bot.send_message(chat_id, f"👋 {user.mention_html()} has been onboarded and unlocked.", parse_mode="HTML")
    except Exception: pass

async def greet_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat=update.effective_chat
    if not update.message or not update.message.new_chat_members: return
    bot_username=(await context.bot.get_me()).username
    start_link=await build_start_link(bot_username, chat.id, chat.title)
    names=", ".join(m.mention_html() for m in update.message.new_chat_members)
    text=f"Welcome {names}! I would love to be able to converse with you.\n\nTap this link to allow me to DM you: {start_link}"
    await update.message.reply_html(text, disable_web_page_preview=True)

# ──────────────────────────────────────────────────────────────
# 18) Resolve helpers
# ──────────────────────────────────────────────────────────────
def resolve_user_by_username(chat_id: Optional[int], username: str) -> Optional[int]:
    uname=username.lstrip("@").lower()
    with closing(db()) as con:
        q="SELECT user_id FROM allowed WHERE LOWER(username)=?"; args=(uname,)
        if chat_id is not None: q+=" AND chat_id=?"; args=(uname, chat_id)
        r=con.execute(q, args).fetchall()
        if r: return r[0][0]
        r2=con.execute("SELECT DISTINCT user_id FROM messages WHERE LOWER(username)=? ORDER BY date DESC",(uname,)).fetchall()
        return r2[0][0] if r2 else None

# ──────────────────────────────────────────────────────────────
# 19) /profile + other public/admin handlers wiring
# ──────────────────────────────────────────────────────────────
async def commands_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    is_admin = uid in ADMIN_WHITELIST if uid else False
    await update.effective_message.reply_text(format_commands_for_user(is_admin))

# (re-register commands function defined earlier)
# ──────────────────────────────────────────────────────────────
# 20) Error handler
# ──────────────────────────────────────────────────────────────
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("Exception in handler:", file=sys.stderr)
        traceback.print_exception(type(context.error), context.error, context.error.__traceback__)
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ An internal error occurred. Logged.")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 21) Wire up + run (safe JobQueue + DBW start)
# ──────────────────────────────────────────────────────────────
def main():
    # Ensure base schema present
    with closing(db()): pass

    # Start global write-queue (threaded; no loop required)
    global DBW
    DBW = DBWriteQueue()
    DBW.start()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Gating
    app.add_handler(CommandHandler("setup", setup))
    app.add_handler(CommandHandler("ungate", ungate))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(CommandHandler("linkprimary", linkprimary))
    app.add_handler(CommandHandler("setprimary", setprimary))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(agree_cb, pattern=r"^agree:\-?\d+$"))
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, greet_new_members))

    # Admin tools
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("users", users_cmd))
    app.add_handler(CommandHandler("message", message_cmd))
    app.add_handler(CommandHandler("system", system_cmd))
    app.add_handler(CommandHandler("inspect", inspect_cmd))
    app.add_handler(CommandHandler("autosystem", autosystem_cmd))

    # Profiles (DM)
    app.add_handler(CommandHandler("profile", profile_cmd))

    # /setadmin + approvals
    app.add_handler(CommandHandler("setadmin", setadmin_cmd))
    app.add_handler(CallbackQueryHandler(adminreq_cb, pattern=r"^adminreq:(approve|deny):\d+$"))

    # Topic / Graph
    app.add_handler(CommandHandler("commands", commands_cmd))
    app.add_handler(CommandHandler("topic", topic_cmd))
    app.add_handler(CommandHandler("graph", graph_cmd))

    # Text logger + reply logic
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # Error handler
    app.add_error_handler(on_error)

    # JobQueue (PTB 20) — ensure exists
    jq = getattr(app, "job_queue", None)
    if jq is None:
        from telegram.ext import JobQueue
        jq = JobQueue(); jq.set_application(app); jq.start()

    # Schedule auto-system-prompt + profile background jobs
    if AUTO_SYS_ENABLED and jq:
        jq.run_once(autosystem_job, when=5)
        jq.run_repeating(autosystem_job, interval=AUTO_SYS_INTERVAL_HOURS*3600, first=AUTO_SYS_INTERVAL_HOURS*3600)
    if PROFILES_ENABLED and jq:
        jq.run_repeating(profile_background_job, interval=PROFILE_REFRESH_MINUTES*60, first=60)

    print("Bot running…")
    app.run_polling()

if __name__ == "__main__":
    main()
