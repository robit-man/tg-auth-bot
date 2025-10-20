#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gatekeeper Bot — gating, admin tools, inspector, Ollama chat, KG, semantic search,
gleanings, sharded DBs, auto system-prompt generation, robust SQLite write queue (threaded),
and privacy-aware per-user "interaction profiles" updated during idle time.

Python 3.10+
"""

# ──────────────────────────────────────────────────────────────
# BOOTSTRAP: STDLIB IMPORTS ONLY
# ──────────────────────────────────────────────────────────────
import os, sys, subprocess, textwrap, sqlite3, base64, secrets, re, time, json, math, random, asyncio, socket, threading, queue, traceback, glob, html
from collections import defaultdict
from pathlib import Path
from contextlib import closing
from functools import wraps, lru_cache
from typing import Optional, Tuple, List, Iterable, Dict, Callable, Any
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────
# VISUALIZER/SLEEP IMPORTS HAPPEN AFTER VENV BOOTSTRAP
# They are imported later in the file after ensure_venv_and_deps()
# ──────────────────────────────────────────────────────────────

# Sleep cycle (optional - graceful degradation if import fails)
try:
    from sleep_cycle import init_sleep_cycle, sleep_cycle_tick, get_sleep_cycle, is_sleeping, get_sleep_state
    SLEEP_CYCLE_AVAILABLE = True
except ImportError:
    SLEEP_CYCLE_AVAILABLE = False
    def init_sleep_cycle(*args, **kwargs): pass
    async def sleep_cycle_tick(*args, **kwargs): pass
    def get_sleep_cycle(): return None
    def is_sleeping(): return False
    def get_sleep_state(): return {'state': 'awake'}

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

    force_bootstrap = os.environ.get("FORCE_BOOTSTRAP") == "1"
    boot_flag = VENV / ".bootstrapped"

    if VENV.exists() and boot_flag.exists() and not force_bootstrap:
        py = str(_venv_python(VENV))
        os.execv(py, [py, *sys.argv])

    if not VENV.exists():
        print("[bootstrap] Creating .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])

    py = str(_venv_python(VENV))
    print("[bootstrap] Upgrading pip/setuptools/wheel ...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Core requirements - MUST install
    reqs = [
        'python-telegram-bot[job-queue]>=20,<21',  # JobQueue included
        'APScheduler>=3.9,<4.0',
        "python-dotenv>=1.0,<2.0",
        "requests>=2.31,<3.0",
    ]

    # Tool requirements - for search_internet and other tools
    tool_reqs = [
        "selenium>=4.17,<5.0",
        "webdriver-manager>=4.0,<5.0",
        "beautifulsoup4>=4.12,<5.0",
        "lxml>=4.9,<6.0",
        "ollama>=0.1.0",  # For LLM operations
    ]

    print("[bootstrap] Installing core requirements ...")
    core_failed = False
    try:
        subprocess.check_call([py, "-m", "pip", "install", *reqs])
        print("[bootstrap]   ✓ Core requirements installed")
    except subprocess.CalledProcessError as exc:
        core_failed = True
        print(f"[bootstrap]   ✗ Core requirements FAILED: {exc}")
        print("[bootstrap]   Will retry on next run")

    print("[bootstrap] Installing tool requirements ...")
    failed = []
    for pkg in tool_reqs:
        pkg_name = pkg.split('>=')[0]
        try:
            print(f"[bootstrap]   Installing {pkg_name}...", end=" ", flush=True)
            result = subprocess.run(
                [py, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("✓")
            else:
                print("✗ FAILED")
                if result.stderr:
                    print(f"[bootstrap]     Error: {result.stderr[:200]}")
                failed.append(pkg_name)
        except Exception as exc:
            print("✗ ERROR")
            print(f"[bootstrap]     Exception: {exc}")
            failed.append(pkg_name)

    # Only create bootstrap flag if ALL packages installed successfully
    if not core_failed and not failed:
        try:
            boot_flag.write_text(f"bootstrapped at {datetime.now().isoformat()}\n")
            print("[bootstrap] ✓ All dependencies installed successfully!")
        except Exception:
            pass
    else:
        if failed:
            print(f"[bootstrap] ⚠ {len(failed)} packages failed: {', '.join(failed)}")
        print("[bootstrap] ⚠ Bootstrap incomplete - will retry on next run")

    print("[bootstrap] Re-exec in .venv ...")
    os.execv(py, [py, *sys.argv])

ensure_venv_and_deps()

# ──────────────────────────────────────────────────────────────
# NOW IN VENV - IMPORT THIRD-PARTY MODULES
# ──────────────────────────────────────────────────────────────

# Memory visualizer (optional - graceful degradation if import fails)
try:
    from memory_visualizer import start_visualizer, log_recall, log_operation, update_stats, log_autonomous, update_sleep_state
    VISUALIZER_AVAILABLE = True
    print("[bootstrap] ✓ Memory visualizer imported successfully")
except ImportError as e:
    VISUALIZER_AVAILABLE = False
    _fallback_visualizer_started = False
    print(f"[bootstrap] ✗ Memory visualizer import failed: {e}")
    print("[bootstrap]   Using fallback console logging")

    def _mv_fallback_log(kind: str, message: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[visualizer:{kind}]"
        if message:
            print(f"{prefix} {timestamp} {message}")
        else:
            print(f"{prefix} {timestamp}")

    def start_visualizer(force_curses=False):
        global _fallback_visualizer_started
        if not _fallback_visualizer_started:
            _fallback_visualizer_started = True
            _mv_fallback_log("fallback", "curses unavailable; logging to console only")

    def log_recall(scope="?", category="?", content="", weight=0.0, metadata=None):
        if _fallback_visualizer_started:
            _mv_fallback_log("recall", f"{scope}::{category} w={weight:.2f} {content[:80]}")

    def log_operation(operation: str, **details):
        if _fallback_visualizer_started:
            _mv_fallback_log("op", f"{operation} {details}")

    def update_stats(**kwargs):
        if _fallback_visualizer_started:
            _mv_fallback_log("stats", str(kwargs))

    def log_autonomous(*args, **kwargs):
        if _fallback_visualizer_started:
            payload = kwargs if kwargs else args
            _mv_fallback_log("auto", str(payload))

    def update_sleep_state(*args, **kwargs):
        if _fallback_visualizer_started:
            payload = kwargs if kwargs else args
            _mv_fallback_log("sleep", str(payload))

# Inside venv
import requests
from dotenv import load_dotenv, dotenv_values
from telegram import Update, ChatPermissions, InlineKeyboardMarkup, InlineKeyboardButton, MessageEntity
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

# Duplicate code removed - bootstrap runs at top of file

try:
    from ai_tool_bridge import (
        TOOL_INTEGRATION_AVAILABLE,
        TOOLS_AVAILABLE as BRIDGE_TOOLS_AVAILABLE,
        EnhancedPromptBuilder,
        ToolExecutionCoordinator,
    )
    if not BRIDGE_TOOLS_AVAILABLE:
        print("[bootstrap] Tools unavailable - missing dependencies")
        print("[bootstrap] Delete .venv/.bootstrapped and restart to retry install")
except ImportError as e:
    TOOL_INTEGRATION_AVAILABLE = False
    BRIDGE_TOOLS_AVAILABLE = False
    EnhancedPromptBuilder = None  # type: ignore
    ToolExecutionCoordinator = None  # type: ignore
    print(f"[bootstrap] Tool bridge import failed: {e}")
    print("[bootstrap] Delete .venv/.bootstrapped and restart to retry install")

try:
    from tool_integration import ToolInspector, build_tool_context_for_prompt
    from tool_schema import ToolSchemaGenerator, ToolFormatter
    TOOL_SUMMARY_AVAILABLE = True
    TOOL_SCHEMA_AVAILABLE = True
except ImportError:
    ToolInspector = None  # type: ignore
    build_tool_context_for_prompt = None  # type: ignore
    ToolSchemaGenerator = None  # type: ignore
    ToolFormatter = None  # type: ignore
    TOOL_SUMMARY_AVAILABLE = False
    TOOL_SCHEMA_AVAILABLE = False

# Intelligent tool routing, RAG, and vision
try:
    from tool_router import IntelligentToolRouter, RouteType, RouteDecision
    from document_rag import DocumentRAGStore
    from vision_handler import VisionHandler
    INTELLIGENT_ROUTING_AVAILABLE = True
except ImportError:
    IntelligentToolRouter = None  # type: ignore
    RouteType = None  # type: ignore
    RouteDecision = None  # type: ignore
    DocumentRAGStore = None  # type: ignore
    VisionHandler = None  # type: ignore
    INTELLIGENT_ROUTING_AVAILABLE = False

# Real tool execution with UI
try:
    from tool_executor_bridge import (
        ToolDecisionMaker,
        RealToolExecutor,
        ToolDecision as RealToolDecision,
        ToolDecisionType,
        decide_and_execute_tool,
    )
    from tool_telegram_ui import (
        ToolTelegramUI,
        handle_tool_confirmation_callback,
        handle_rating_callback,
    )
    REAL_TOOL_EXECUTION_AVAILABLE = True
except ImportError:
    ToolDecisionMaker = None  # type: ignore
    RealToolExecutor = None  # type: ignore
    RealToolDecision = None  # type: ignore
    ToolDecisionType = None  # type: ignore
    decide_and_execute_tool = None  # type: ignore
    ToolTelegramUI = None  # type: ignore
    handle_tool_confirmation_callback = None  # type: ignore
    handle_rating_callback = None  # type: ignore
    REAL_TOOL_EXECUTION_AVAILABLE = False

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

        MEMORY_VISUALIZER_ENABLED=true
        MAX_CONTEXT_GROUPS=30

        AUTO_SYSTEM_PROMPT_ENABLED=true
        AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=12

        PROFILES_ENABLED=true
        PROFILE_REFRESH_MINUTES=30

        AUTO_RELOAD=true
        AUTO_RELOAD_PATHS=bot_server.py
        AUTO_RELOAD_INTERVAL=1.0

        INTERNAL_REFLECTION_ENABLED=true
        INTERNAL_REFLECTION_MAX_CHARS=1200

        MEMORY_BUDGET_CHARS=600
        MEMORY_THREAD_LIMIT=40
        MEMORY_USER_LIMIT=60
        MEMORY_GLOBAL_LIMIT=80
        MEMORY_SUMMARY_BATCH=10

        SUMMARY_ROLLUP_INTERVAL_SECONDS=900

        SLEEP_CYCLE_ENABLED=true
        SLEEP_CYCLE_TICK_SECONDS=60
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

AUTO_RELOAD_ENABLED = ((os.getenv("AUTO_RELOAD") or (CFG.get("AUTO_RELOAD") or "false")).strip().lower() == "true")
AUTO_RELOAD_INTERVAL = float((os.getenv("AUTO_RELOAD_INTERVAL") or (CFG.get("AUTO_RELOAD_INTERVAL") or "1.0")).strip() or "1.0")
_raw_reload_paths = (os.getenv("AUTO_RELOAD_PATHS") or (CFG.get("AUTO_RELOAD_PATHS") or "")).strip()
AUTO_RELOAD_PATHS = [p.strip() for p in re.split(r"[;,]", _raw_reload_paths) if p.strip()] or ["bot_server.py"]

INTERNAL_REFLECTION_ENABLED = ((os.getenv("INTERNAL_REFLECTION_ENABLED") or (CFG.get("INTERNAL_REFLECTION_ENABLED") or "true")).strip().lower() == "true")
INTERNAL_REFLECTION_MAX_CHARS = int((os.getenv("INTERNAL_REFLECTION_MAX_CHARS") or (CFG.get("INTERNAL_REFLECTION_MAX_CHARS") or "1200")).strip() or "1200")
MEMORY_BUDGET_CHARS = int((os.getenv("MEMORY_BUDGET_CHARS") or (CFG.get("MEMORY_BUDGET_CHARS") or "600")).strip() or "600")
MEMORY_THREAD_LIMIT = int((os.getenv("MEMORY_THREAD_LIMIT") or (CFG.get("MEMORY_THREAD_LIMIT") or "40")).strip() or "40")
MEMORY_USER_LIMIT = int((os.getenv("MEMORY_USER_LIMIT") or (CFG.get("MEMORY_USER_LIMIT") or "60")).strip() or "60")
MEMORY_GLOBAL_LIMIT = int((os.getenv("MEMORY_GLOBAL_LIMIT") or (CFG.get("MEMORY_GLOBAL_LIMIT") or "80")).strip() or "80")
MEMORY_SUMMARY_BATCH = int((os.getenv("MEMORY_SUMMARY_BATCH") or (CFG.get("MEMORY_SUMMARY_BATCH") or "10")).strip() or "10")
MEMORY_VISUALIZER_ENABLED = ((os.getenv("MEMORY_VISUALIZER_ENABLED") or (CFG.get("MEMORY_VISUALIZER_ENABLED") or "true")).strip().lower() == "true")
FORCE_CURSES = ((os.getenv("FORCE_CURSES") or (CFG.get("FORCE_CURSES") or "false")).strip().lower() == "true")

SLEEP_CYCLE_ENABLED = ((os.getenv("SLEEP_CYCLE_ENABLED") or (CFG.get("SLEEP_CYCLE_ENABLED") or "true")).strip().lower() == "true")
SLEEP_CYCLE_TICK_SECONDS = int((os.getenv("SLEEP_CYCLE_TICK_SECONDS") or (CFG.get("SLEEP_CYCLE_TICK_SECONDS") or "60")).strip() or "60")

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

# ──────────────────────────────────────────────────────────────
# Intelligent routing, RAG, and vision components
# ──────────────────────────────────────────────────────────────
TOOL_ROUTER: Optional[Any] = None
RAG_STORE: Optional[Any] = None
VISION_HANDLER: Optional[Any] = None

def init_intelligent_components():
    """Initialize intelligent routing, RAG, and vision components"""
    global TOOL_ROUTER, RAG_STORE, VISION_HANDLER

    if not INTELLIGENT_ROUTING_AVAILABLE:
        print("[intelligent] Components not available (missing tool_router, document_rag, or vision_handler)")
        return

    # Initialize tool router
    if OLLAMA_MODEL and IntelligentToolRouter:
        try:
            TOOL_ROUTER = IntelligentToolRouter(
                ollama_model=OLLAMA_MODEL,
                enable_llm_routing=True,
            )
            print(f"[intelligent] Tool router initialized with model: {OLLAMA_MODEL}")
        except Exception as e:
            print(f"[intelligent] Failed to initialize tool router: {e}")

    # Initialize RAG store
    if OLLAMA_EMBED_MODEL and DocumentRAGStore:
        try:
            rag_db_path = ROOT / "data" / "documents.db"
            rag_data_dir = ROOT / "data"
            RAG_STORE = DocumentRAGStore(
                db_path=rag_db_path,
                embed_model=OLLAMA_EMBED_MODEL,
                vision_model=os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b"),
                data_dir=rag_data_dir,
            )
            print(f"[intelligent] RAG store initialized at {rag_db_path}")
        except Exception as e:
            print(f"[intelligent] Failed to initialize RAG store: {e}")

    # Initialize vision handler
    if VisionHandler:
        try:
            vision_model = os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b")
            VISION_HANDLER = VisionHandler(vision_model=vision_model)
            print(f"[intelligent] Vision handler initialized with model: {vision_model}")
        except Exception as e:
            print(f"[intelligent] Failed to initialize vision handler: {e}")

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

MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES") or 300)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS") or 12000)
MAX_CONTEXT_USERS = int(os.getenv("MAX_CONTEXT_USERS") or 60)
MAX_CONTEXT_GROUPS = int(os.getenv("MAX_CONTEXT_GROUPS") or 30)

DEBUG_FLAGS: Dict[int, bool] = defaultdict(bool)
DEBUG_CONTEXTS: Dict[tuple[int, int], Dict[str, str]] = {}

# ──────────────────────────────────────────────────────────────
# 2) Paths + shard dirs
# ──────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
ERROR_LOG_PATH = DATA_DIR / "handler_errors.log"
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


def _tool_bridge_ready() -> bool:
    return bool(
        TOOL_INTEGRATION_AVAILABLE
        and BRIDGE_TOOLS_AVAILABLE
        and EnhancedPromptBuilder is not None
        and ToolExecutionCoordinator is not None
    )


async def generate_agentic_reply(
    payload: dict,
    user_id: Optional[int],
    *,
    include_examples: bool = True,
    auto_execute: bool = True,
    max_tools: int = 5,
) -> Tuple[Optional[str], List[Any]]:
    """
    Run the main LLM call, injecting tool context and executing requested tools for admins.
    Returns the (possibly augmented) reply text and execution metadata.
    """
    enhanced_payload = payload
    executions: List[Any] = []
    tool_access = _tool_bridge_ready()

    if tool_access:
        try:
            enhanced_payload = EnhancedPromptBuilder.inject_tool_context(
                payload,
                user_id,  # type: ignore[arg-type]
                ADMIN_WHITELIST,
                include_examples=include_examples,
            )
        except Exception as exc:
            print(f"[tools] Failed to inject tool context: {exc}")
            enhanced_payload = payload

    response = await ai_generate_async(enhanced_payload)
    if not response or not tool_access:
        return response, executions

    try:
        coordinator = ToolExecutionCoordinator(user_id, ADMIN_WHITELIST)  # type: ignore[call-arg]
        modified_response, execution_results = await coordinator.process_ai_response(
            response,
            auto_execute=auto_execute,
            max_tools=max_tools,
            max_dag_nodes=max(4, max_tools * 2),
            max_dags=2,
        )
        response = modified_response
        executions = execution_results
    except Exception as exc:
        print(f"[tools] Failed to process tool executions: {exc}")

    return response, executions

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


def _truncate_text(text: str, limit: int = 1800) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_dump(obj: Any, limit: int = 1800) -> str:
    try:
        rendered = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        rendered = str(obj)
    return _truncate_text(rendered, limit)


def describe_available_tools(max_tools: int = 12) -> str:
    if not TOOL_SUMMARY_AVAILABLE or not ToolInspector:
        return "(tool catalog unavailable)"
    summaries: List[str] = []
    try:
        for meta in ToolInspector.get_all_tools():
            entry = f"{meta.name}{meta.signature}"
            if meta.is_async:
                entry += " [async]"
            summaries.append(entry)
    except Exception:
        return "(tool catalog unavailable)"
    if not summaries:
        return "(no tools discovered)"
    return "\n".join(summaries[:max_tools])

def list_tool_signatures(max_tools: int = 20) -> str:
    if not TOOL_SUMMARY_AVAILABLE or not ToolInspector:
        return "(tool catalog unavailable)"
    try:
        tools = ToolInspector.get_all_tools()
    except Exception:
        return "(tool catalog unavailable)"
    if not tools:
        return "(no tools registered)"
    tools.sort(key=lambda t: t.name.lower())
    lines: List[str] = []
    for tool in tools[:max_tools]:
        signature = tool.signature or "()"
        lines.append(f"- {tool.name}{signature}" + (" [async]" if tool.is_async else ""))
        params = tool.parameters or []
        required = [p["name"] for p in params if p.get("required")]
        optional = [p["name"] for p in params if not p.get("required")]
        if required:
            lines.append(f"    required: {', '.join(required)}")
        if optional:
            lines.append(f"    optional: {', '.join(optional)}")
    if len(tools) > max_tools:
        lines.append(f"... ({len(tools) - max_tools} more)")
    return "\n".join(lines)


def build_debug_context_text(
    user_text: str,
    payload: Optional[dict],
    context_bundle: Dict[str, str],
    complexity_meta: Dict[str, object],
    internal_state: Dict[str, object],
    executed_actions: List[Any],
    tool_runs: List[Any],
    decision_meta: Dict[str, object],
) -> str:
    lines: List[str] = []
    lines.append("INPUT MESSAGE:")
    lines.append(_truncate_text(user_text or "(empty)", 600))

    preface = ""
    if payload and isinstance(payload.get("messages"), list) and len(payload["messages"]) >= 2:
        preface = payload["messages"][1].get("content", "")
    if preface:
        lines.append("\nMODEL CONTEXT PREFACE:")
        lines.append(_truncate_text(preface, 2000))

    lines.append("\nCOMPLEXITY META:")
    lines.append(_json_dump(complexity_meta, 900))

    if decision_meta:
        lines.append("\nDECISION META:")
        lines.append(_json_dump(decision_meta, 600))

    if context_bundle:
        lines.append("\nCONTEXT BUNDLE:")
        lines.append(_json_dump(context_bundle, 900))

    snippet_count = len(internal_state.get("snippets") or []) if isinstance(internal_state, dict) else 0
    lines.append(f"\nMEMORY STATE: snippets={snippet_count}, narrative={bool(internal_state.get('narrative')) if isinstance(internal_state, dict) else False}")

    lines.append("\nAVAILABLE TOOL SUMMARIES:")
    lines.append(describe_available_tools(15))

    if executed_actions:
        lines.append("\nEXECUTED INTERNAL ACTIONS:")
        for name, result in executed_actions[:6]:
            lines.append(f"- {name}: {_truncate_text(str(result), 160)}")
    else:
        lines.append("\nEXECUTED INTERNAL ACTIONS: (none)")

    if tool_runs:
        lines.append("\nTOOL EXECUTIONS:")
        for run in tool_runs[:8]:
            state = getattr(run, "state", None)
            state_val = getattr(state, "value", state) if state is not None else "unknown"
            result_preview = getattr(run, "result", None)
            lines.append(f"- {getattr(run, 'tool_name', '(unknown)')} [{state_val}]: {_truncate_text(str(result_preview), 200)}")
    else:
        lines.append("\nTOOL EXECUTIONS: (none)")

    return _truncate_text("\n".join(lines), 3500)


def compose_debug_display(response_text: str, context_text: str, show: bool) -> str:
    base = _truncate_text(response_text or "", 3500)
    if not show:
        return base
    combined = f"{base}\n\n--- DEBUG CONTEXT ---\n{context_text}"
    return _truncate_text(combined, 3900)

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
        ("/tools", "List available tool functions exposed to the AI."),
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

@whitelist_only
async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not chat:
        return
    chat_id = chat.id
    args = [a.lower() for a in (context.args or [])]
    if not args:
        state = DEBUG_FLAGS.get(chat_id, False)
        return await update.effective_message.reply_text(
            f"Debug mode is {'ON' if state else 'OFF'} for this chat. Use /debug on or /debug off."
        )
    val = args[0]
    if val in ("on", "true", "1", "enable", "enabled"):
        DEBUG_FLAGS[chat_id] = True
        await update.effective_message.reply_text(
            "🔍 Debug mode enabled for this chat. Future responses will include a show/hide context toggle."
        )
    elif val in ("off", "false", "0", "disable", "disabled"):
        DEBUG_FLAGS[chat_id] = False
        await update.effective_message.reply_text("🔇 Debug mode disabled for this chat.")
    else:
        await update.effective_message.reply_text("Usage: /debug on|off")

async def commands_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    is_admin = uid in ADMIN_WHITELIST if uid else False
    await update.effective_message.reply_text(format_commands_for_user(is_admin))


async def debug_toggle_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.message:
        return
    data = (query.data or "").split(":")
    if len(data) != 2:
        await query.answer()
        return
    _, action = data
    chat = query.message.chat
    if not chat:
        await query.answer()
        return
    key = (chat.id, query.message.message_id)
    entry = DEBUG_CONTEXTS.get(key)
    if not entry:
        await query.answer("No debug context available.", show_alert=True)
        return
    show = action == "show"
    new_text = compose_debug_display(entry.get("response", ""), entry.get("context", ""), show)
    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Hide context", callback_data="debug:hide")]]
        if show
        else [[InlineKeyboardButton("Show context", callback_data="debug:show")]]
    )
    try:
        await query.edit_message_text(new_text, reply_markup=markup)
    except Exception:
        await query.answer("Unable to update message.", show_alert=True)
        return
    await query.answer()

# ──────────────────────────────────────────────────────────────
# 12) Admin gating, inspector, setadmin flow
# ──────────────────────────────────────────────────────────────

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

@admin_only
async def tools_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /tools - Show available tools for admin users
    Usage:
      /tools           - Show all tools with full details
      /tools compact   - Show compact tool list
      /tools [category] - Show tools in specific category (e.g., web, browser, filesystem)
    """
    if not TOOL_SUMMARY_AVAILABLE or not ToolInspector:
        msg = (
            "⚠️ Tool catalog unavailable.\n\n"
            "Tool integration modules are not loaded. This usually means:\n"
            "• tools.py is missing required dependencies (bs4, selenium, etc.)\n"
            "• tool_integration.py or tool_schema.py are not in the correct location\n\n"
            "To enable tools, ensure all dependencies are installed:\n"
            "pip install beautifulsoup4 selenium duckduckgo_search"
        )
        await update.effective_message.reply_text(msg)
        return

    user_id = update.effective_user.id if update.effective_user else 0
    args = context.args or []

    # Determine mode
    compact = "compact" in args
    category = None
    for arg in args:
        if arg.lower() in ["web", "browser", "filesystem", "system", "ai", "general"]:
            category = arg.lower()
            break

    # Use new schema-based formatting if available
    if TOOL_SCHEMA_AVAILABLE and ToolSchemaGenerator and ToolFormatter:
        try:
            schemas = ToolSchemaGenerator.generate_all_schemas()

            if not schemas:
                await update.effective_message.reply_text("No tools registered in tools.py.")
                return

            # Filter by category if specified
            categories = [category] if category else None

            if compact:
                catalog = ToolFormatter.format_compact(schemas, max_per_category=10)
            else:
                catalog = ToolFormatter.format_for_prompt(
                    schemas,
                    categories=categories,
                    max_tools=50 if not category else None
                )

            # Send in chunks if too long
            if len(catalog) > 4000:
                chunks = [catalog[i:i+3900] for i in range(0, len(catalog), 3900)]
                for i, chunk in enumerate(chunks[:3]):  # Max 3 messages
                    header = f"📚 Tools ({i+1}/{min(len(chunks), 3)})\n\n" if len(chunks) > 1 else ""
                    await update.effective_message.reply_text(f"{header}{chunk}")
            else:
                await update.effective_message.reply_text(catalog)
            return

        except Exception as e:
            # Fall back to simple listing
            pass

    # Fallback to simple tool signatures
    catalog = list_tool_signatures(max_tools=40)
    if not catalog or catalog.startswith("(no tools"):
        await update.effective_message.reply_text("No tools registered in tools.py.")
        return
    if len(catalog) > 3500:
        catalog = catalog[:3497] + "..."
    await update.effective_message.reply_text(f"Available tools:\n{catalog}")

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
        log_autonomous('assessment', scope='global', details='Auto-generating system prompt')
        p = await auto_generate_system_prompt(context, include_digest_in_meta=False)
        if p:
            await set_system_prompt_and_persist(p)
            log_autonomous('assessment', scope='global', count=1,
                          details='System prompt auto-generated and persisted')
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
        log_autonomous('assessment', scope='profile', details='Starting profile background update')

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
                log_autonomous('assessment', scope='user', count=1,
                              details=f'Updated profile for user {uid}')
                if updated >= 3: break

        if updated > 0:
            log_autonomous('assessment', scope='profile', count=updated,
                          details=f'Completed profile updates for {updated} users')
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
                    thread_ctx: str, similar_blend: Dict[str, str], kg_snapshot: Dict[str, List[str]],
                    internal_state: Optional[Dict[str, object]] = None,
                    context_bundle: Optional[Dict[str, str]] = None,
                    complexity_meta: Optional[Dict[str, object]] = None) -> dict:
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
    state_blob = internal_state or {}
    narrative = (state_blob.get("narrative") or "").strip()
    bias_text = (state_blob.get("bias") or "").strip()
    snippet_lines = []
    for snip in state_blob.get("snippets", []) or []:
        if not isinstance(snip, dict):
            continue
        label = f"{snip.get('scope','')}::{snip.get('category','')}".strip(":")
        tags = snip.get("metadata", {}).get("tags") if isinstance(snip.get("metadata"), dict) else []
        tag_txt = f" [{' '.join(tags)}]" if tags else ""
        body = (snip.get("content") or "").strip()
        if not body:
            continue
        snippet_lines.append(f"- {label}{tag_txt}: {body}")
    memory_snippets = "\n".join(snippet_lines) or "(none)"
    bundle = context_bundle or {}
    complexity = complexity_meta or {}

    if is_admin:
        # Use full schema-based tool exposure for admin users
        if TOOL_SCHEMA_AVAILABLE and ToolSchemaGenerator and ToolFormatter:
            try:
                schemas = ToolSchemaGenerator.generate_all_schemas()
                if schemas:
                    # Use compact format to save context space, but include all essential info
                    tool_catalog = ToolFormatter.format_for_prompt(
                        schemas,
                        categories=None,  # Show all categories
                        max_tools=30,  # Limit to prevent context overflow
                    )
                else:
                    # Fallback to simple signatures
                    tool_catalog = list_tool_signatures(18)
            except Exception as e:
                print(f"[tools] Schema-based formatting failed: {e}")
                tool_catalog = list_tool_signatures(18)
        else:
            # Fallback to simple tool listing
            tool_catalog = list_tool_signatures(18)

        if not tool_catalog or tool_catalog.startswith("(tool catalog") or tool_catalog.startswith("(no tools"):
            tool_catalog = "(tool catalog unavailable)"
    else:
        tool_catalog = "(restricted — admin only)"

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

        TOOL FUNCTIONS (admin visibility):
        {tool_catalog}

        TOOL USAGE INSTRUCTIONS:

        1. WHEN TO USE TOOLS:
           - User explicitly asks you to search, fetch, or gather information
           - You need current/external information not in context
           - User asks about documents they've uploaded (use knowledge query)
           - You need to perform file operations, system commands, etc.

        2. HOW TO CALL TOOLS:
           Simple tool call:
             Tools.tool_name(arg1, arg2, kwarg1=value)

           Async tool call (for async tools marked [async]):
             await Tools.tool_name(arg1, arg2)

           Example: Tools.search_internet('Python 3.12 features', num_results=5)

        3. MULTI-TOOL WORKFLOWS (DAG):
           For complex tasks needing multiple tools in sequence:

           <<TOOL_DAG>>
           summary: Search and analyze Python features
           nodes:
             - id: search1
               tool: search_internet
               args:
                 topic: "Python 3.12 new features"
                 num_results: 3

             - id: fetch1
               tool: fetch_webpage
               args:
                 url: "{{search1.results[0].url}}"
               depends_on: [search1]
           <<END_DAG>>

        4. BEST PRACTICES:
           - Check tool parameter requirements from the tool catalog above
           - Use quoted strings for text parameters: 'like this'
           - For numbers use: num_results=5 (no quotes)
           - Summarize tool outputs naturally, don't paste raw output
           - Chain tools with DAGs when one output feeds another
           - Skip tools if answer is already clear from context

        5. AFTER TOOL EXECUTION:
           - Tool results will be injected into context automatically
           - Synthesize a natural language response incorporating the results
           - Cite sources when appropriate (URLs, page numbers, etc.)

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

        INTERNAL_MONOLOGUE (KEEP HIDDEN FROM USERS):
          Narrative:
        {narrative or '(none)'}
          Bias cues:
        {bias_text or '(none)'}
          Memory snippets:
        {memory_snippets}

        CONTEXT BUNDLE:
          Thread brief:
        {bundle.get('thread_brief') or '(none)'}
          Relationship signals:
        {bundle.get('relationship_signals') or '(none)'}
          User signals:
        {bundle.get('user_signals') or '(none)'}
          Global themes:
        {bundle.get('global_themes') or '(none)'}
          Emotional cues:
        {bundle.get('emotional_state') or '(none)'}
          Summary rollup:
        {bundle.get('summary_rollup') or '(none)'}
          Global rollup:
        {bundle.get('global_rollup') or '(none)'}

        COMPLEXITY GAUGE:
          Level: {complexity.get('complexity', 'low')}
          Confidence: {complexity.get('confidence', 0.0)}
          Planned tools: {complexity.get('actions', [])}
          Notes: {complexity.get('notes', '')}

        RESPONSE_RULES:
          - Treat this metadata and INTERNAL_MONOLOGUE as private planning only; never mention them in replies.
          - Answer the user directly in 1-3 sentences (under ~120 words) unless they explicitly ask for a longer format.
          - Avoid headings, section breaks, or restating these instructions unless the user requests structure.
          - Do not expose thought processes, internal reflections, or mention that you are hiding them.
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

def _basic_auto_reply_signal(m, thread_ctx_text: str) -> bool:
    text = (m.text or "").strip()
    if not text: return False
    recent_has_bot = (BOT_CACHE["username"] and BOT_CACHE["username"].lower() in (thread_ctx_text or "").lower())
    is_question = "?" in text
    starts_with = text.lower().startswith(("bot","hey bot","gatekeeper","help","how do i","why "))
    contains_you = text.lower().startswith(("can you","could you","will you","would you","please")) or "tell me" in text.lower()
    return (is_question and recent_has_bot) or starts_with or contains_you

def _parse_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

async def should_auto_reply_async(message, chat, user, thread_ctx_text: str,
                                  internal_state: Dict[str, object],
                                  must: bool) -> tuple[bool, Dict[str, object]]:
    text = (message.text or "").strip()
    if not text:
        return False, {"reason": "empty"}
    reply_to_bot = bool(getattr(message, "reply_to_message", None) and
                        getattr(message.reply_to_message, "from_user", None) and
                        message.reply_to_message.from_user.id == BOT_CACHE.get("id"))

    if AUTO_REPLY_MODE and AUTO_REPLY_MODE.lower() in ("off", "disable", "disabled"):
        return must, {"reason": "mode-off"}
    if must or reply_to_bot:
        return True, {"reason": "direct-mention-or-reply", "priority": "high"}
    mode = (AUTO_REPLY_MODE or "smart").lower()
    basic_signal = _basic_auto_reply_signal(message, thread_ctx_text)
    if mode in ("always", "all"):
        return True, {"reason": "mode-always"}
    if mode not in ("smart", "auto", "hybrid"):
        return basic_signal, {"reason": "basic-signal"}
    if basic_signal:
        return True, {"reason": "heuristic-signal"}
    if not ai_available():
        return False, {"reason": "no-ai"}

    mem_snippets = []
    for snip in (internal_state or {}).get("snippets", [])[:4]:
        if not isinstance(snip, dict):
            continue
        label = f"{snip.get('scope','')}::{snip.get('category','')}".strip(":")
        body = (snip.get("content") or "").strip()
        if not body:
            continue
        participants = ""
        meta = snip.get("metadata")
        if isinstance(meta, dict) and meta.get("participants"):
            participants = f" participants={','.join(meta['participants'])}"
        mem_snippets.append(f"{label}{participants}: {body}")
    narrative = (internal_state or {}).get("narrative") or ""
    bias = (internal_state or {}).get("bias") or ""
    feature_lines = [
        f"question={bool('?' in text)}",
        f"length={len(text)}",
        f"contains_link={bool(re.search(r'https?://', text))}",
        f"emoji={bool(re.search(r'[\\u2600-\\u26FF\\U0001F300-\\U0001FAFF]', text))}",
        f"all_caps={text.isupper()}",
        f"reply_to_bot={reply_to_bot}",
        f"user_id={getattr(user, 'id', 0) if user else 0}",
        f"chat_type={getattr(chat, 'type', '(unknown)')}",
    ]
    decision_prompt = textwrap.dedent(f"""
        Decide if Gatekeeper Bot should respond to the latest user message.
        Output compact JSON: {{"reply": true|false, "priority": "low|medium|high", "reason": "..."}}.
        Consider user intent, usefulness, and avoid redundant replies.

        Message: {text}
        Features: {', '.join(feature_lines)}
        Narrative cues: {narrative}
        Bias cues: {bias}
        Memory snippets:
        {mem_snippets or ['(none)']}

        Recent thread excerpt:
        {textwrap.shorten(thread_ctx_text or '(empty)', width=480, placeholder='…')}
    """).strip()
    raw = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":decision_prompt}], "stream": False})
    decision = _parse_json_object(_sanitize_internal_blob(raw or ""))
    if not decision:
        return False, {"reason": "decider-no-json"}
    reply_flag = bool(decision.get("reply"))
    reason = (decision.get("reason") or "").strip() or "decider"
    priority = (decision.get("priority") or "medium").strip().lower()
    return reply_flag, {"reason": reason, "priority": priority}

# ──────────────────────────────────────────────────────────────
# 16a) Internal reflection & narrative bias
# ──────────────────────────────────────────────────────────────
MEMORY_SCOPE_WEIGHTS = {
    ("thread", "thread_note"): 1.25,
    ("thread", "thread_summary"): 1.05,
    ("thread", "relationship"): 1.3,
    ("thread", "bias"): 1.4,
    ("thread", "emotion_profile"): 1.2,
    ("user", "user_trait"): 1.2,
    ("user", "user_summary"): 1.0,
    ("user", "relationship"): 1.25,
    ("user", "bias"): 1.35,
    ("user", "emotion_profile"): 1.3,
    ("global", "global_theme"): 0.95,
    ("global", "global_summary"): 0.9,
    ("global", "relationship"): 1.05,
    ("global", "self_reflection"): 1.1,
    ("global", "bias"): 1.0,
}

INTERNAL_TOOLS = {
    "deep_thread_summary": {
        "description": "Condense the active thread and highlight actionable next steps when conversation density rises."
    },
    "relationship_mapper": {
        "description": "Identify and summarize relationships or roles between participants mentioned in the current exchange."
    },
    "emotion_scan": {
        "description": "Gauge the emotional tone and intent of the requesting user to adapt style and empathy."
    },
}

SELF_STATE_CACHE: Dict[str, Dict[str, object]] = {"global": {"ts": 0.0, "state": {"mood": 0.0, "tension": 0.15, "last_feedback_ts": 0, "data": {}}}}
BREAKOUT_COOLDOWN_SECONDS = 3600
SUMMARY_ROLLUP_THRESHOLD = 6
SUMMARY_ROLLUP_KEEP = 4
SUMMARY_ROLLUP_INTERVAL_SECONDS = 900

def _memory_limit_for(scope: str) -> int:
    if scope == "thread":
        return max(5, MEMORY_THREAD_LIMIT)
    if scope == "user":
        return max(5, MEMORY_USER_LIMIT)
    return max(5, MEMORY_GLOBAL_LIMIT)

def _memory_scope_identifiers(chat, thread_id: int, user) -> Dict[str, tuple[int, int, int]]:
    chat_id = getattr(chat, "id", 0) if chat else 0
    thread_key = int(thread_id or 0)
    user_id = getattr(user, "id", 0) if user else 0
    scopes = {"thread": (chat_id, thread_key, 0)}
    if user_id:
        scopes["user"] = (0, 0, user_id)
    scopes["global"] = (0, 0, 0)
    return scopes

def _sanitize_internal_blob(text: str) -> str:
    blob = (text or "").strip()
    if not blob:
        return ""
    blob = re.sub(r"^```[a-zA-Z]*\n", "", blob)
    blob = re.sub(r"\n```$", "", blob)
    return blob.strip()

def _memory_rows_for_scope(scope: str, chat_id: int, thread_id: int, user_id: int, limit: int) -> list[tuple]:
    with closing(db()) as con:
        return con.execute(
            """SELECT id, category, content, metadata, embedding, weight, created_ts
               FROM memory_entries
               WHERE scope=? AND chat_id=? AND thread_id=? AND user_id=?
               ORDER BY created_ts DESC
               LIMIT ?""",
            (scope, chat_id, thread_id, user_id, limit)
        ).fetchall()

def _normalize_sources(sources: Optional[List[dict]]) -> List[dict]:
    normed = []
    if not sources:
        return normed
    for item in sources:
        if not isinstance(item, dict):
            continue
        s_type = (item.get("type") or "").strip().lower()
        s_id = item.get("id")
        if not s_type or s_id is None:
            continue
        weight = float(item.get("weight", 0.6))
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        normed.append({"type": s_type, "id": int(s_id), "weight": weight, "metadata": meta})
    return normed

async def _upsert_memory_links_async(memory_id: int, sources: Optional[List[dict]]):
    sources = _normalize_sources(sources)
    if not memory_id or not sources:
        return
    def _work(con: sqlite3.Connection):
        for src in sources:
            metadata_json = json.dumps(src["metadata"], ensure_ascii=False)
            con.execute(
                """INSERT INTO memory_links(memory_id, source_type, source_id, weight, metadata)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(memory_id, source_type, source_id)
                   DO UPDATE SET weight=excluded.weight, metadata=excluded.metadata""",
                (memory_id, src["type"], src["id"], src["weight"], metadata_json)
            )
        con.commit()
    if DBW:
        await DBW.run(DB_PATH, _work)
    else:
        with closing(db()) as con:
            _work(con)

def get_self_state(scope: str = "global") -> Dict[str, object]:
    with closing(db()) as con:
        row = con.execute(
            "SELECT mood, tension, last_feedback_ts, data FROM self_state WHERE scope=?",
            (scope,)
        ).fetchone()
    if row:
        mood, tension, last_feedback_ts, data_json = row
        try:
            data = json.loads(data_json or "{}")
        except Exception:
            data = {}
        return {"mood": float(mood or 0.0), "tension": float(tension or 0.0),
                "last_feedback_ts": int(last_feedback_ts or 0), "data": data}
    return {"mood": 0.0, "tension": 0.15, "last_feedback_ts": 0, "data": {}}

def _set_self_state_cache(scope: str, state: Dict[str, object]):
    SELF_STATE_CACHE[scope] = {"ts": time.time(), "state": state}

def get_cached_self_state(scope: str = "global", max_age: float = 120.0) -> Dict[str, object]:
    cache = SELF_STATE_CACHE.get(scope)
    now = time.time()
    if cache and now - float(cache.get("ts", 0)) < max_age:
        return cache.get("state", {})
    state = get_self_state(scope)
    _set_self_state_cache(scope, state)
    return state

async def save_self_state_async(scope: str, mood: float, tension: float, last_feedback_ts: int, data: Dict[str, object]):
    data_json = json.dumps(data or {}, ensure_ascii=False)
    def _work(con: sqlite3.Connection):
        con.execute(
            """INSERT INTO self_state(scope, mood, tension, last_feedback_ts, data)
               VALUES(?,?,?,?,?)
               ON CONFLICT(scope) DO UPDATE SET
                 mood=excluded.mood,
                 tension=excluded.tension,
                 last_feedback_ts=excluded.last_feedback_ts,
                 data=excluded.data""",
            (scope, float(mood), float(tension), int(last_feedback_ts), data_json)
        )
        con.commit()
    if DBW:
        await DBW.run(DB_PATH, _work)
    else:
        with closing(db()) as con:
            _work(con)
    _set_self_state_cache(scope, {"mood": float(mood), "tension": float(tension),
                                  "last_feedback_ts": int(last_feedback_ts), "data": data})

def _emotion_delta(emotion: str) -> float:
    mapping = {
        "joy": 0.35, "happy": 0.3, "positive": 0.25, "grateful": 0.3,
        "neutral": 0.0, "curious": 0.1,
        "confused": -0.1, "sad": -0.25, "frustrated": -0.35,
        "angry": -0.4, "negative": -0.3, "disappointed": -0.3,
    }
    if not emotion:
        return 0.0
    return mapping.get(emotion.strip().lower(), 0.0)

def _tone_modifier(tone: str) -> float:
    if not tone:
        return 0.0
    tone_lower = tone.lower()
    if any(k in tone_lower for k in ("supportive", "encouraging", "appreciative")):
        return 0.2
    if any(k in tone_lower for k in ("frustrated", "upset", "critical", "worried")):
        return -0.25
    if "neutral" in tone_lower:
        return 0.0
    return 0.0

async def apply_feedback_to_mood(emotion: str, tone: str, intent: str, confidence: float,
                                 chat_id: int, user_id: Optional[int], metadata: Dict[str, object]) -> Dict[str, object]:
    state = get_cached_self_state()
    base = _emotion_delta(emotion)
    base += _tone_modifier(tone)
    if intent:
        intent_lower = intent.lower()
        if any(k in intent_lower for k in ("complaint", "escalate", "issue")):
            base -= 0.25
        if any(k in intent_lower for k in ("thank", "praise", "appreciate")):
            base += 0.2
    confidence = max(0.0, min(float(confidence or 0.5), 1.0))
    mood = float(state.get("mood", 0.0))
    tension = float(state.get("tension", 0.0))
    delta = base * confidence
    new_mood = max(-1.0, min(1.0, mood * 0.92 + delta * 0.35))
    new_tension = max(0.0, min(1.0, tension * 0.90 + abs(delta) * 0.4))

    data = state.get("data", {}) or {}
    feedback_log = data.get("recent_feedback")
    if not isinstance(feedback_log, list):
        feedback_log = []
    feedback_log.append({
        "ts": int(time.time()),
        "emotion": emotion,
        "tone": tone,
        "intent": intent,
        "confidence": confidence,
        "chat_id": chat_id,
        "user_id": user_id,
    })
    if len(feedback_log) > 25:
        feedback_log = feedback_log[-25:]
    data["recent_feedback"] = feedback_log
    await save_self_state_async("global", new_mood, new_tension, int(time.time()), data)
    return {"mood": new_mood, "tension": new_tension, "data": data}

def _resolve_participant_to_user_id(participant, chat) -> Optional[int]:
    if participant is None:
        return None
    if isinstance(participant, dict):
        for key in ("id", "user_id"):
            if key in participant:
                try:
                    return int(participant[key])
                except (TypeError, ValueError):
                    continue
    if isinstance(participant, int):
        return participant
    if isinstance(participant, str):
        cleaned = participant.strip()
        if not cleaned:
            return None
        if cleaned.startswith("@"):
            chat_id = getattr(chat, "id", None) if chat else None
            resolved = resolve_user_by_username(chat_id, cleaned)
            return resolved
        m = re.search(r"\d{5,}", cleaned)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass
    return None

def _merge_relationship_metadata(existing_json: Optional[str], new_meta: dict, description: str) -> str:
    try:
        existing = json.loads(existing_json or "{}")
    except Exception:
        existing = {}
    desc_list = existing.get("descriptions")
    if not isinstance(desc_list, list):
        desc_list = []
    if description:
        desc_list.append(description)
        if len(desc_list) > 10:
            desc_list = desc_list[-10:]
    merged = {**existing, **(new_meta or {})}
    merged["descriptions"] = desc_list
    return json.dumps(merged, ensure_ascii=False)

async def update_relationship_graph_async(source_user: Optional[int], target_user: Optional[int],
                                          chat_id: int, thread_id: int, relation: str,
                                          description: str, msg_id: Optional[int],
                                          timestamp: int, weight: float,
                                          metadata: Optional[dict] = None):
    if not source_user or not target_user or source_user == target_user:
        return
    relation = (relation or "").strip() or "relationship"
    description = (description or "").strip()
    weight = max(0.05, float(weight or 0.5))
    timestamp = int(timestamp or time.time())
    def _work(con: sqlite3.Connection):
        row = con.execute(
            """SELECT id, weight, metadata, first_msg_id, first_ts, description FROM relationship_graph
               WHERE source_user=? AND target_user=? AND chat_id=? AND thread_id=? AND relation=?""",
            (source_user, target_user, chat_id, thread_id, relation)
        ).fetchone()
        meta_json = _merge_relationship_metadata(row[2] if row else "{}", metadata or {}, description)
        if row:
            current_weight = row[1] or 0.0
            new_weight = min(current_weight + weight, 10.0)
            first_msg_id = row[3] if row[3] else msg_id
            first_ts = row[4] if row[4] else timestamp
            con.execute(
                """UPDATE relationship_graph
                   SET description=?, last_msg_id=?, last_ts=?, weight=?, metadata=?, first_msg_id=?, first_ts=?
                   WHERE id=?""",
                (description or row[5] or "", msg_id or row[3], timestamp, new_weight, meta_json,
                 first_msg_id, first_ts, row[0])
            )
        else:
            con.execute(
                """INSERT INTO relationship_graph(source_user, target_user, chat_id, thread_id,
                                                  relation, description, first_msg_id, last_msg_id,
                                                  first_ts, last_ts, weight, metadata)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (source_user, target_user, chat_id, thread_id, relation, description,
                 msg_id, msg_id, timestamp, timestamp, weight, meta_json)
            )
        con.commit()
    if DBW:
        await DBW.run(DB_PATH, _work)
    else:
        with closing(db()) as con:
            _work(con)

async def _update_graph_from_relationship_entries(stored_entries: List[dict], chat, fallback_thread_id: int):
    for stored in stored_entries:
        if stored.get("category") != "relationship":
            continue
        metadata = stored.get("metadata") or {}
        participants = metadata.get("participants") if isinstance(metadata, dict) else None
        if not isinstance(participants, (list, tuple)):
            continue
        participant_ids: List[int] = []
        for participant in participants:
            uid = _resolve_participant_to_user_id(participant, chat)
            if uid and uid not in participant_ids:
                participant_ids.append(uid)
        if len(participant_ids) < 2:
            continue
        description = stored.get("content") or ""
        relation = "relationship"
        tags = metadata.get("tags") if isinstance(metadata, dict) else None
        if isinstance(tags, list) and tags:
            relation = tags[0]
        timestamp = metadata.get("generated") if isinstance(metadata, dict) else None
        entry_id = stored.get("_entry_id")
        weight = stored.get("weight", 0.6)
        chat_id = stored.get("chat_id", getattr(chat, "id", 0))
        thread_key = stored.get("thread_id", fallback_thread_id)
        for i, source_id in enumerate(participant_ids):
            for target_id in participant_ids[i+1:]:
                meta_extra = {**(metadata or {}), "memory_entry_id": entry_id}
                await update_relationship_graph_async(
                    source_id, target_id,
                    chat_id, thread_key,
                    relation,
                    description,
                    entry_id,
                    timestamp or int(time.time()),
                    weight,
                    meta_extra
                )
                await update_relationship_graph_async(
                    target_id, source_id,
                    chat_id, thread_key,
                    relation,
                    description,
                    entry_id,
                    timestamp or int(time.time()),
                    weight,
                    meta_extra
                )

def _last_breakout_timestamp(user_id: int, trigger: str) -> int:
    with closing(db()) as con:
        row = con.execute(
            """SELECT MAX(created_ts) FROM breakout_events
               WHERE user_id=? AND trigger=?""",
            (user_id, trigger)
        ).fetchone()
    return int(row[0] or 0)

async def record_breakout_event_async(user_id: int, chat_id: int, thread_id: int,
                                      trigger: str, details: str, status: str):
    now = int(time.time())
    def _work(con: sqlite3.Connection):
        con.execute(
            """INSERT INTO breakout_events(user_id, chat_id, thread_id, trigger, details, created_ts, status)
               VALUES(?,?,?,?,?,?,?)""",
            (user_id, chat_id, thread_id, trigger, details, now, status)
        )
        con.commit()
    if DBW:
        await DBW.run(DB_PATH, _work)
    else:
        with closing(db()) as con:
            _work(con)

async def maybe_trigger_breakout_event(context: ContextTypes.DEFAULT_TYPE, chat, user, thread_id: int,
                                       decision_meta: Dict[str, object],
                                       complexity_meta: Dict[str, object],
                                       mood_state: Dict[str, object],
                                       thread_ctx: str,
                                       message_text: str):
    if not user or not getattr(user, "id", None):
        return
    user_id = user.id
    chat_id = getattr(chat, "id", 0)
    mood = float((mood_state or {}).get("mood", 0.0))
    tension = float((mood_state or {}).get("tension", 0.0))
    complexity_level = (complexity_meta or {}).get("complexity", "low")
    reason = (decision_meta or {}).get("reason", "")

    trigger = None
    notes = []
    if mood < -0.4:
        trigger = "mood_check"
        notes.append(f"mood={mood:.2f}")
    elif tension > 0.65:
        trigger = "tension_check"
        notes.append(f"tension={tension:.2f}")
    elif complexity_level == "high":
        trigger = "complexity_followup"
        notes.append("complexity=high")
    elif isinstance(reason, str) and "escalate" in reason.lower():
        trigger = "decision_escalate"
        notes.append(f"reason={reason}")

    if not trigger:
        return
    last_ts = _last_breakout_timestamp(user_id, trigger)
    if last_ts and time.time() - last_ts < BREAKOUT_COOLDOWN_SECONDS:
        return

    greeting = user.first_name or user.full_name or "there"
    reason_txt = ", ".join(notes) if notes else ""
    message = textwrap.dedent(f"""
        Hi {greeting}! I wanted to follow up proactively to make sure everything is on track.
        If you have additional details, feedback, or requests, feel free to share them with me any time.
        {"(Reason: " + reason_txt + ")" if reason_txt else ""}
    """).strip()
    try:
        await context.bot.send_message(chat_id=user_id, text=message)
        status = "sent"
    except Exception:
        status = "failed"
    details = json.dumps({
        "notes": notes,
        "thread_excerpt": textwrap.shorten(thread_ctx or "", width=280, placeholder="…"),
        "message_excerpt": textwrap.shorten(message_text or "", width=180, placeholder="…"),
        "mood": mood,
        "tension": tension,
        "complexity": complexity_level,
        "decision": decision_meta,
    }, ensure_ascii=False)
    await record_breakout_event_async(user_id, chat_id, thread_id or 0, trigger, details, status)

def fetch_latest_summary(scope: str, chat_id: int, thread_id: int, user_id: int = 0, category: Optional[str] = None) -> str:
    query = """SELECT summary FROM memory_summaries
               WHERE scope=? AND chat_id=? AND thread_id=? AND user_id=?
               {cat_clause}
               ORDER BY created_ts DESC LIMIT 1"""
    cat_clause = ""
    params: List[object] = [scope, chat_id, thread_id, user_id]
    if category:
        cat_clause = "AND category=?"
        params.append(category)
    query = query.format(cat_clause=cat_clause)
    with closing(db()) as con:
        row = con.execute(query, params).fetchone()
    return row[0] if row else ""

async def hierarchical_memory_rollup():
    rollup_created = False
    try:
        log_autonomous('consolidation', scope='all', details='Scanning for rollup candidates')

        with closing(db()) as con:
            rows = con.execute(
                """SELECT id, scope, chat_id, thread_id, user_id, category, content, metadata, weight, created_ts
                   FROM memory_entries
                   WHERE category LIKE '%_summary'
                   ORDER BY created_ts DESC
                   LIMIT 400"""
            ).fetchall()
        groups: Dict[tuple, List[dict]] = {}
        for row in rows:
            (mid, scope, chat_id, thread_id, user_id,
             category, content, metadata_json, weight, created_ts) = row
            base_cat = category[:-8] if category.endswith("_summary") else category
            key = (scope, chat_id, thread_id, user_id, base_cat)
            try:
                metadata = json.loads(metadata_json or "{}")
            except Exception:
                metadata = {}
            groups.setdefault(key, []).append({
                "id": mid,
                "scope": scope,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "user_id": user_id,
                "category": category,
                "base_category": base_cat,
                "content": content,
                "metadata": metadata,
                "weight": float(weight or metadata.get("confidence", 0.5)),
                "created_ts": int(created_ts or 0),
            })
        now = int(time.time())
        for key, entries in groups.items():
            if len(entries) < SUMMARY_ROLLUP_THRESHOLD:
                continue
            scope, chat_id, thread_id, user_id, base_category = key
            entries.sort(key=lambda e: (e["weight"], e["created_ts"]), reverse=True)
            top_entries = entries[: min(len(entries), SUMMARY_ROLLUP_THRESHOLD + 2)]
            prompt_payload = []
            for e in top_entries:
                prompt_payload.append({
                    "content": e["content"],
                    "weight": e["weight"],
                    "metadata": e.get("metadata", {}),
                })
            prompt = textwrap.dedent(f"""
                Create a concise hierarchical summary capturing the key patterns, relationships, and signals
                from the following memory summaries. Structure it as:
                - OVERVIEW: <short paragraph>
                - HIGHLIGHTS: <2-4 bullet points>
                - NEXT_STEPS: <1-2 suggestions if applicable, else omit>

                Focus on what's actionable or distinctive. Preserve any critical participants or tags mentioned.

                SUMMARIES:
                {json.dumps(prompt_payload, ensure_ascii=False)}
            """).strip()
            rollup = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
            rollup = _sanitize_internal_blob(rollup or "")
            if not rollup:
                continue
            metadata = {
                "source_summary_ids": [e["id"] for e in top_entries],
                "base_category": base_category,
                "generated": now,
                "count": len(top_entries),
            }
            def _store_summary(con: sqlite3.Connection):
                con.execute(
                    """INSERT INTO memory_summaries(scope,chat_id,thread_id,user_id,category,summary,metadata,created_ts)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (scope, chat_id, thread_id, user_id, f"{base_category}_rollup", rollup,
                     json.dumps(metadata, ensure_ascii=False), now)
                )
                con.commit()
            if DBW:
                await DBW.run(DB_PATH, _store_summary)
            else:
                with closing(db()) as con:
                    _store_summary(con)

            rollup_created = True

            log_autonomous('consolidation', scope=scope, count=len(top_entries),
                          details=f'Rolled up {base_category}')

            # prune older summaries beyond keep count
            if len(entries) > SUMMARY_ROLLUP_KEEP:
                to_delete = [e["id"] for e in entries[SUMMARY_ROLLUP_KEEP:]]
                await _delete_memory_entries_async(to_delete)
                log_autonomous('decay', scope=scope, count=len(to_delete),
                              details=f'Pruned old {base_category} summaries')
    except Exception:
        pass
    if rollup_created:
        await maybe_update_system_prompt_from_reflection(force=True)

async def hierarchical_rollup_job(context: ContextTypes.DEFAULT_TYPE):
    log_autonomous('rollup', scope='hierarchical', details='Starting hierarchical memory rollup')
    await hierarchical_memory_rollup()
    log_autonomous('rollup', scope='hierarchical', details='Completed hierarchical memory rollup')

async def sleep_cycle_job(context: ContextTypes.DEFAULT_TYPE):
    """Autonomous sleep cycle tick - manages sleep/wake states and deep memory processing"""
    if not SLEEP_CYCLE_ENABLED or not SLEEP_CYCLE_AVAILABLE:
        return

    try:
        # Tick the sleep cycle
        await sleep_cycle_tick(ai_generate_fn=ai_generate_async)

        # Update visualizer with current sleep state
        state_info = get_sleep_state()
        update_sleep_state(
            state=state_info['state'],
            time_in_state=state_info['time_in_state'],
            cycle_count=state_info['cycle_count'],
            discoveries=state_info['discoveries']
        )

    except Exception as e:
        # Swallow errors in background job
        pass

def assemble_context_bundle(chat, thread_id: int, user, message_text: str,
                            thread_ctx: str, memory_state: Dict[str, object]) -> Dict[str, str]:
    snippets = memory_state.get("snippets") or []
    relationship_lines, user_lines, global_lines, emotion_lines = [], [], [], []
    for snip in snippets:
        if not isinstance(snip, dict):
            continue
        category = snip.get("category") or ""
        content = (snip.get("content") or "").strip()
        if not content:
            continue
        meta = snip.get("metadata") if isinstance(snip.get("metadata"), dict) else {}
        participants = meta.get("participants") if isinstance(meta, list) else meta.get("participants")
        tag_txt = ""
        tags = meta.get("tags") if isinstance(meta, dict) else None
        if isinstance(tags, list) and tags:
            tag_txt = f" [{' '.join(tags[:3])}]"
        if category == "relationship":
            part_txt = ""
            if isinstance(participants, list) and participants:
                part_txt = f" ({', '.join(participants[:3])})"
            relationship_lines.append(f"- {content}{part_txt}{tag_txt}")
        elif category in ("user_trait",):
            user_lines.append(f"- {content}{tag_txt}")
        elif category == "emotion_profile":
            emotion_lines.append(f"- {content}{tag_txt}")
        elif category in ("global_theme", "global_summary", "self_reflection"):
            global_lines.append(f"- {content}{tag_txt}")
        elif category == "thread_note":
            # thread notes are folded into user_lines for immediate guidance
            user_lines.append(f"- {content}{tag_txt}")
    thread_brief = textwrap.shorten(thread_ctx or "(empty)", width=600, placeholder=" …")
    message_excerpt = textwrap.shorten(message_text or "", width=320, placeholder=" …")
    mood_state = get_cached_self_state()
    return {
        "thread_brief": thread_brief,
        "message_excerpt": message_excerpt,
        "relationship_signals": "\n".join(relationship_lines[:6]) or "(none)",
        "user_signals": "\n".join(user_lines[:6]) or "(none)",
        "global_themes": "\n".join(global_lines[:5]) or "(none)",
        "emotional_state": "\n".join(emotion_lines[:4]) or "(none)",
        "self_mood": f"{mood_state.get('mood', 0.0):+.2f}",
        "self_tension": f"{mood_state.get('tension', 0.0):.2f}",
        "summary_rollup": fetch_latest_summary("thread", getattr(chat, "id", 0), thread_id) or "(none)",
        "global_rollup": fetch_latest_summary("global", 0, 0) or "(none)",
    }

async def assess_discussion_complexity(message_text: str,
                                       thread_ctx: str,
                                       memory_state: Dict[str, object],
                                       context_bundle: Dict[str, str],
                                       chat,
                                       user) -> Dict[str, object]:
    if not ai_available():
        return {"complexity": "low", "confidence": 0.4, "actions": []}
    snippet_preview = []
    for snip in (memory_state.get("snippets") or [])[:5]:
        if not isinstance(snip, dict):
            continue
        label = f"{snip.get('scope','')}::{snip.get('category','')}".strip(":")
        snippet_preview.append(f"{label}: {textwrap.shorten((snip.get('content') or '').strip(), width=160, placeholder='…')}")
    features = {
        "message_length": len(message_text or ""),
        "thread_length": len(thread_ctx or ""),
        "relationships_found": context_bundle.get("relationship_signals", "(none)") != "(none)",
        "user_signals": context_bundle.get("user_signals", "(none)") != "(none)",
        "global_themes": context_bundle.get("global_themes", "(none)") != "(none)",
    }
    tool_summary_text = describe_available_tools(15)
    prompt = textwrap.dedent(f"""
        You are an orchestration planner for Gatekeeper Bot.
        Decide if the latest interaction requires additional internal tools.
        Respond with JSON: {{"complexity": "low|medium|high", "confidence": 0.0-1.0, "actions": ["tool_slug", ...], "notes": "short text"}}
        Available tool slugs: {', '.join(INTERNAL_TOOLS.keys())}. Only include a slug if it is truly needed.

        Message: {message_text}
        Thread glimpse: {textwrap.shorten(thread_ctx or '', width=420, placeholder='…')}
        Memory snippets: {snippet_preview or '(none)'}
        Context bundle: {context_bundle}
        Features: {features}
        Tool summaries:
        {tool_summary_text}

        Choose at most 2 actions. Prefer none when conversation is simple.
    """).strip()
    raw = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
    decision = _parse_json_object(_sanitize_internal_blob(raw or ""))
    if not decision:
        return {"complexity": "low", "confidence": 0.4, "actions": []}
    actions = [a for a in (decision.get("actions") or []) if a in INTERNAL_TOOLS]
    return {
        "complexity": (decision.get("complexity") or "low").lower(),
        "confidence": float(decision.get("confidence") or 0.5),
        "actions": actions,
        "notes": decision.get("notes") or "",
    }

async def _generate_thread_summary_async(meta: dict, chat, thread_id: int, thread_ctx: str, message_text: str):
    if not ai_available():
        return None
    prompt = textwrap.dedent(f"""
        Summarize the recent thread for Gatekeeper Bot.
        Return plain text under 4 sentences plus a list of up to 3 bullet next-steps for the bot.

        THREAD:
        {thread_ctx or '(empty)'}

        LATEST MESSAGE:
        {message_text}
    """).strip()
    summary = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
    summary = _sanitize_internal_blob(summary or "")
    if not summary:
        return None
    source_msg_id = meta.get("global_row_id") if isinstance(meta, dict) else None
    metadata = {
        "source_msg_id": source_msg_id,
        "tags": ["auto", "thread_summary"],
        "generated": int(time.time())
    }
    sources = []
    if source_msg_id:
        sources.append({"type": "message", "id": int(source_msg_id), "weight": 0.75, "metadata": {"scope": "thread", "category": "thread_summary"}})
    await _store_memory_entry_async("thread", getattr(chat, "id", 0), int(thread_id or 0), 0,
                                    "thread_summary", summary[:480], metadata, 0.75, sources)
    return summary

async def _extract_relationships_async(meta: dict, chat, thread_id: int, thread_ctx: str, message_text: str):
    if not ai_available():
        return
    prompt = textwrap.dedent(f"""
        Extract up to 3 relationships or roles between participants mentioned in the thread.
        Respond with JSON list of items: [{{"description": "...", "participants": ["name or id"], "confidence": 0.0-1.0, "tags": ["..."]}}]
        Focus on durable or meaningful ties only.

        THREAD:
        {thread_ctx or '(empty)'}

        LATEST MESSAGE:
        {message_text}
    """).strip()
    raw = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
    items = _parse_json_object(_sanitize_internal_blob(raw or ""))
    if not isinstance(items, list):
        return
    source_msg_id = meta.get("global_row_id") if isinstance(meta, dict) else None
    scopes = _memory_scope_identifiers(chat, thread_id, None)
    records = []
    for item in items[:3]:
        description = (item.get("description") or "").strip()
        if not description:
            continue
        confidence = float(item.get("confidence", 0.6))
        tags = item.get("tags") or ["relationship"]
        participants = item.get("participants") or []
        metadata = {
            "source_msg_id": source_msg_id,
            "tags": tags,
            "participants": participants,
            "generated": int(time.time())
        }
        sources = []
        if source_msg_id:
            sources.append({"type": "message", "id": int(source_msg_id), "weight": confidence,
                            "metadata": {"scope": "thread", "category": "relationship"}})
        records.append({
            "scope": "thread",
            "chat_id": getattr(chat, "id", 0),
            "thread_id": int(thread_id or 0),
            "user_id": 0,
            "category": "relationship",
            "content": description[:240],
            "metadata": metadata,
            "weight": confidence,
            "sources": sources
        })
    if records:
        stored_records = await _store_memories_with_decay(records)
        await _update_graph_from_relationship_entries(stored_records, chat, thread_id)

async def run_agentic_expansions(actions: List[str], meta: dict, chat, thread_id: int,
                                 thread_ctx: str, message_text: str, user):
    executed = []
    for action in actions:
        try:
            if action == "deep_thread_summary":
                summary = await _generate_thread_summary_async(meta, chat, thread_id, thread_ctx, message_text)
                if summary:
                    executed.append(("deep_thread_summary", summary))
            elif action == "relationship_mapper":
                await _extract_relationships_async(meta, chat, thread_id, thread_ctx, message_text)
                executed.append(("relationship_mapper", "ok"))
            elif action == "emotion_scan":
                mood_state = await analyze_user_affect_intent(meta, chat, thread_id, user, message_text)
                executed.append(("emotion_scan", mood_state or "ok"))
        except Exception:
            continue
    return executed

async def analyze_user_affect_intent(meta: dict, chat, thread_id: int, user, message_text: str):
    if not ai_available() or not user:
        return
    user_id = getattr(user, "id", None)
    if not user_id:
        return
    prompt = textwrap.dedent(f"""
        Analyse the user's emotional tone and intent.
        Respond with compact JSON:
        {{"emotion": "...", "intent": "...", "confidence": 0.0-1.0, "tone": "...", "tags": ["..."]}}

        User message:
        {message_text}
    """).strip()
    raw = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
    data = _parse_json_object(_sanitize_internal_blob(raw or ""))
    if not isinstance(data, dict):
        return
    emotion = (data.get("emotion") or "").strip()
    intent = (data.get("intent") or "").strip()
    tone = (data.get("tone") or "").strip()
    if not emotion and not intent and not tone:
        return
    confidence = float(data.get("confidence", 0.6))
    tags = data.get("tags") or []
    source_msg_id = meta.get("global_row_id") if isinstance(meta, dict) else None
    metadata = {
        "source_msg_id": source_msg_id,
        "emotion": emotion,
        "intent": intent,
        "tone": tone,
        "confidence": confidence,
        "tags": tags,
        "generated": int(time.time())
    }
    content_parts = [p for p in (emotion, intent, tone) if p]
    content = " | ".join(content_parts) or "emotional update"
    sources = []
    if source_msg_id:
        sources.append({"type": "message", "id": int(source_msg_id), "weight": confidence,
                        "metadata": {"scope": "user", "category": "emotion_profile"}})
    await _store_memory_entry_async("user", 0, 0, int(user_id),
                                    "emotion_profile", content[:240], metadata, confidence, sources)
    mood_state = await apply_feedback_to_mood(emotion, tone, intent, confidence,
                                              getattr(chat, "id", 0), int(user_id), metadata)
    log_emotion_state(
        emotion=emotion,
        intent=intent,
        tone=tone,
        confidence=confidence,
        user_id=int(user_id),
        chat_id=getattr(chat, "id", 0),
        message_excerpt=message_text[:160],
    )
    return mood_state

async def maybe_update_system_prompt_from_reflection(force: bool = False):
    if not ai_available():
        return
    now = int(time.time())
    last_raw = settings_get("REFLECTIVE_PROMPT_TS")
    last_ts = int(last_raw) if last_raw and last_raw.isdigit() else 0
    if not force and now - last_ts < 3600:
        return
    with closing(db()) as con:
        reflection_rows = con.execute(
            """SELECT content, metadata FROM memory_entries
               WHERE scope='global' AND category='self_reflection'
               ORDER BY created_ts DESC LIMIT 12"""
        ).fetchall()
        summary_rows = con.execute(
            """SELECT summary, metadata, category, created_ts FROM memory_summaries
               WHERE scope='global' AND category LIKE '%_rollup'
               ORDER BY created_ts DESC LIMIT 5"""
        ).fetchall()
    if not reflection_rows and not summary_rows:
        return
    reflections = []
    for content, metadata_json in reflection_rows:
        meta = {}
        try:
            meta = json.loads(metadata_json or "{}")
        except Exception:
            pass
        ts = meta.get("generated")
        stamp = datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else ""
        reflections.append(f"[{stamp}] {content}")
    summaries = []
    for summary, metadata_json, category, created_ts in summary_rows:
        meta = {}
        try:
            meta = json.loads(metadata_json or "{}")
        except Exception:
            pass
        ts = meta.get("generated") or created_ts
        stamp = datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else ""
        snippet = textwrap.shorten((summary or "").strip(), width=320, placeholder="…")
        label = category or "rollup"
        summaries.append(f"[{stamp}] ({label}) {snippet}")
    prompt = textwrap.dedent(f"""
        Use the latest global summaries and self-reflections to refine Gatekeeper Bot's internal system prompt.
        Keep it under 600 words, emphasise tone, decision rules, memory usage, and safety guardrails.
        Return plain text (no JSON).

        GLOBAL SUMMARIES:
        {('\\n'.join(summaries)) if summaries else '(none)'}

        SELF-REFLECTIONS:
        {('\\n'.join(reflections)) if reflections else '(none)'}
    """).strip()
    new_prompt = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
    new_prompt = _sanitize_internal_blob(new_prompt or "")
    if not new_prompt:
        return
    await set_system_prompt_and_persist(new_prompt)
    await settings_set_async("REFLECTIVE_PROMPT_TS", str(now))

async def _store_memory_entry_async(scope: str, chat_id: int, thread_id: int, user_id: int,
                                    category: str, content: str, metadata: dict,
                                    weight: float, sources: Optional[List[dict]] = None):
    content = (content or "").strip()
    if not content:
        return
    metadata = metadata or {}
    now = int(time.time())
    vec = await embed_text_async(content[:768]) if ai_available() else None
    embedding_json = json.dumps(vec) if vec else None
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    weight = float(weight or metadata.get("confidence") or 0.5)
    def _work(con: sqlite3.Connection):
        cur = con.execute(
            """INSERT INTO memory_entries(scope,chat_id,thread_id,user_id,category,content,metadata,embedding,weight,created_ts,updated_ts)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (scope, chat_id, thread_id, user_id, category, content, metadata_json, embedding_json, weight, now, now)
        )
        con.commit()
        return cur.lastrowid
    if DBW:
        entry_id = await DBW.run(DB_PATH, _work)
    else:
        with closing(db()) as con:
            entry_id = _work(con)
    if entry_id and sources:
        await _upsert_memory_links_async(entry_id, sources)
    return entry_id

async def _delete_memory_entries_async(ids: List[int]):
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    def _work(con: sqlite3.Connection):
        con.execute(f"DELETE FROM memory_entries WHERE id IN ({placeholders})", ids)
        con.commit()
    if DBW:
        await DBW.run(DB_PATH, _work)
    else:
        with closing(db()) as con:
            con.execute(f"DELETE FROM memory_entries WHERE id IN ({placeholders})", ids)
            con.commit()

async def _maybe_decay_memories_async(scope: str, chat_id: int, thread_id: int, user_id: int, category: str):
    limit = _memory_limit_for(scope)
    with closing(db()) as con:
        count = con.execute(
            """SELECT COUNT(*) FROM memory_entries
               WHERE scope=? AND chat_id=? AND thread_id=? AND user_id=? AND category=?""",
            (scope, chat_id, thread_id, user_id, category)
        ).fetchone()[0]
    if count <= limit:
        return

    log_autonomous('decay', scope=scope, count=count-limit,
                  details=f'Memory limit exceeded for {category}')

    batch = max(3, MEMORY_SUMMARY_BATCH)
    with closing(db()) as con:
        rows = con.execute(
            """SELECT id, content, metadata FROM memory_entries
               WHERE scope=? AND chat_id=? AND thread_id=? AND user_id=? AND category=?
               ORDER BY created_ts ASC
               LIMIT ?""",
            (scope, chat_id, thread_id, user_id, category, batch)
        ).fetchall()
    if not rows:
        return
    snippets = []
    aggregated_ids = []
    tags = set()
    confidences = []
    for rid, content, metadata_json in rows:
        aggregated_ids.append(rid)
        try:
            meta = json.loads(metadata_json or "{}")
        except Exception:
            meta = {}
        confidences.append(float(meta.get("confidence", 0.5)))
        tags.update(meta.get("tags", []))
        snippets.append(textwrap.shorten(content or "", width=240, placeholder="…"))
    prompt = textwrap.dedent(f"""
        Summarize the following internal memories into <=3 concise sentences and provide 1-3 bullet biases.
        Ensure tone remains professional, avoid protected attribute references, and keep total under {INTERNAL_REFLECTION_MAX_CHARS} characters.

        MEMORIES:
        {json.dumps(snippets, ensure_ascii=False)}
    """).strip()
    summary = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False}) if ai_available() else None
    summary = _sanitize_internal_blob(summary or "")
    if not summary:
        summary = "; ".join(snippets[:3])
    metadata = {
        "collapsed_ids": aggregated_ids,
        "confidence": float(sum(confidences)/len(confidences)) if confidences else 0.5,
        "tags": sorted(tags),
        "kind": "summary"
    }
    sources = [{"type": "memory", "id": int(mid), "weight": 0.7, "metadata": {"role": "collapsed_component"}} for mid in aggregated_ids]
    await _store_memory_entry_async(scope, chat_id, thread_id, user_id, f"{category}_summary", summary, metadata, metadata["confidence"], sources)
    await _delete_memory_entries_async(aggregated_ids)

    log_autonomous('consolidation', scope=scope, count=len(aggregated_ids),
                  details=f'Consolidated {len(aggregated_ids)} {category} memories into summary')

async def _store_memories_with_decay(entries: List[dict]) -> List[dict]:
    scope_groups: Dict[tuple, list[dict]] = {}
    stored_entries: List[dict] = []
    for entry in entries:
        scope = entry["scope"]
        key = (scope, entry.get("chat_id", 0), entry.get("thread_id", 0), entry.get("user_id", 0))
        scope_groups.setdefault(key, []).append(entry)
        entry_id = await _store_memory_entry_async(scope, entry.get("chat_id", 0), entry.get("thread_id", 0),
                                                   entry.get("user_id", 0), entry.get("category", "note"),
                                                   entry.get("content", ""), entry.get("metadata", {}),
                                                   entry.get("weight", 0.6), entry.get("sources"))
        stored_entry = {**entry}
        stored_entry["_entry_id"] = entry_id
        stored_entries.append(stored_entry)

        # Log memory storage operation to visualizer
        log_operation(
            "store",
            scope=scope,
            category=entry.get("category", "note"),
            content=entry.get("content", "")[:80],
            weight=entry.get("weight", 0.6)
        )

    for (scope, chat_id, thread_id, user_id), group in scope_groups.items():
        for entry in group:
            await _maybe_decay_memories_async(scope, chat_id, thread_id, user_id, entry.get("category", "note"))
    return stored_entries

async def recall_memories_async(user_text: str, chat, thread_id: int, user) -> Dict[str, object]:
    if not INTERNAL_REFLECTION_ENABLED:
        return {"narrative": "", "bias": "", "snippets": []}
    scopes = _memory_scope_identifiers(chat, thread_id, user)
    budget = max(200, MEMORY_BUDGET_CHARS)
    query_vec = await embed_text_async(user_text[:768]) if ai_available() else None
    candidates = []
    mood_snapshot = get_cached_self_state()
    mood_value = float(mood_snapshot.get("mood", 0.0))
    tension_value = float(mood_snapshot.get("tension", 0.0))
    for scope, ids in scopes.items():
        chat_id, thread_key, user_id = ids
        rows = _memory_rows_for_scope(scope, chat_id, thread_key, user_id, limit=80)
        for row in rows:
            mem_id, category, content, metadata_json, embedding_json, weight, created_ts = row
            try:
                metadata = json.loads(metadata_json or "{}")
            except Exception:
                metadata = {}
            try:
                vec = json.loads(embedding_json) if embedding_json else None
            except Exception:
                vec = None
            score = float(weight or metadata.get("confidence", 0.5))
            if query_vec and vec and len(query_vec) == len(vec):
                sim = cosine(query_vec, vec)
                score = 0.6 * score + 0.4 * sim
            scope_weight = MEMORY_SCOPE_WEIGHTS.get((scope, category), MEMORY_SCOPE_WEIGHTS.get((scope, "bias"), 1.0))
            mood_factor = 1.0
            if scope in ("thread", "user"):
                mood_factor += 0.25 * mood_value
            else:
                mood_factor += 0.15 * mood_value
            if category == "emotion_profile":
                mood_factor += 0.35 * tension_value
            elif category in ("thread_note", "thread_summary") and tension_value > 0.6:
                mood_factor += 0.2 * tension_value
            scope_weight *= max(0.2, min(1.8, mood_factor))
            recency_bonus = max(0.85, min(1.15, 1 + (max(time.time() - created_ts, 0) * -1e-6)))
            final_score = score * scope_weight * recency_bonus
            candidates.append({
                "id": mem_id,
                "scope": scope,
                "category": category,
                "content": content,
                "metadata": metadata,
                "weight": final_score,
                "raw_weight": score,
            })
    candidates.sort(key=lambda x: x["weight"], reverse=True)
    selected = []
    used_chars = 0
    narrative = ""
    bias_line = ""
    for cand in candidates:
        text_body = cand["content"].strip()
        if not text_body:
            continue
        snippet_len = len(text_body)
        if used_chars + snippet_len > budget:
            continue
        used_chars += snippet_len + 20
        selected.append(cand)

        # Log to visualizer
        log_recall(
            scope=cand["scope"],
            category=cand["category"],
            content=text_body,
            weight=cand["weight"],
            metadata=cand["metadata"]
        )

        if not bias_line and "bias" in cand["category"]:
            bias_line = text_body
        if not narrative and cand["category"] in ("thread_note", "user_trait", "global_theme", "thread_summary", "relationship", "self_reflection"):
            narrative = text_body
        if used_chars >= budget:
            break
    narrative = narrative or (selected[0]["content"] if selected else "")

    # Update visualizer stats
    scope_counts = {}
    for s in selected:
        scope = s["scope"]
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
    update_stats(
        thread=scope_counts.get('thread', 0),
        user=scope_counts.get('user', 0),
        global_scope=scope_counts.get('global', 0),
        total_recalls=len(selected),
        last_recall_time=time.time()
    )

    return {
        "narrative": narrative.strip(),
        "bias": bias_line.strip(),
        "snippets": selected
    }

async def process_memory_update_async(meta: dict, chat, thread_id: int, user,
                                      thread_ctx: str, user_text: str, bot_reply: str):
    if not INTERNAL_REFLECTION_ENABLED or not ai_available():
        return
    existing = await recall_memories_async(user_text, chat, thread_id, user)
    existing_lines = []
    for snip in existing.get("snippets", [])[:5]:
        tags = ", ".join(snip["metadata"].get("tags", [])) if isinstance(snip["metadata"], dict) else ""
        label = f"{snip['scope']}::{snip['category']}"
        existing_lines.append(f"{label}: {snip['content']} ({tags})")
    prior_summary = "\n".join(existing_lines) if existing_lines else "(none)"
    prompt = textwrap.dedent(f"""
        Produce a structured reflection after the latest exchange.
        Respond with compact JSON matching:
        {{
          "thread_notes": [{{"content": "...", "confidence": 0.0-1.0, "tags": ["tag"]}}],
          "user_traits": [{{"content": "...", "confidence": 0.0-1.0, "tags": ["tag"]}}],
          "global_themes": [{{"content": "...", "confidence": 0.0-1.0, "tags": ["tag"]}}],
          "relationships": [
            {{"scope": "thread|user|global", "participants": ["nameA","nameB"], "description": "...", "confidence": 0.0-1.0, "tags": ["tag"]}}
          ],
          "bias_overrides": {{"narrative": "...", "bias": "..."}},
          "self_reflection": {{"content": "...", "confidence": 0.0-1.0, "tags": ["tag"]}}
        }}

        Guidelines:
          - Each string <= 200 chars, constructive, policy-compliant, and free of sensitive trait speculation.
          - Use empty arrays/strings when no update applies.
          - Prefer "thread" scope relationships when specific to the current conversation, "user" for durable traits, "global" for broad narratives.
          - Self_reflection should summarize the bot's role/performance; omit if redundant.

        PRIOR MEMORIES (context only):
        {prior_summary}

        THREAD CONTEXT:
        {thread_ctx or '(empty)'}

        USER MESSAGE:
        {user_text}

        BOT REPLY:
        {bot_reply or '(none)'}
    """).strip()
    raw = await ai_generate_async({"model": OLLAMA_MODEL, "messages":[{"role":"user","content":prompt}], "stream": False})
    if not raw:
        return
    cleaned = _sanitize_internal_blob(raw)
    try:
        data = json.loads(cleaned)
    except Exception:
        try:
            data = json.loads(raw)
        except Exception:
            return
    entries = []
    timestamp = int(time.time())
    source_msg_id = meta.get("global_row_id") if isinstance(meta, dict) else None
    base_metadata = {
        "source_msg_id": source_msg_id,
        "user_excerpt": textwrap.shorten(user_text or "", width=260, placeholder="…"),
        "bot_excerpt": textwrap.shorten(bot_reply or "", width=260, placeholder="…"),
        "updated": timestamp,
    }
    scopes = _memory_scope_identifiers(chat, thread_id, user)

    def _prepare_entries(items, scope_key, category_label):
        for item in items or []:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            confidence = float(item.get("confidence", 0.5))
            tags = item.get("tags") or []
            metadata = {**base_metadata, "confidence": confidence, "tags": tags}
            chat_id, thread_key, user_id = scopes.get(scope_key, (0, 0, 0))
            sources = []
            if source_msg_id:
                sources.append({
                    "type": "message",
                    "id": int(source_msg_id),
                    "weight": confidence,
                    "metadata": {"scope": scope_key, "category": category_label}
                })
            entries.append({
                "scope": scope_key,
                "chat_id": chat_id,
                "thread_id": thread_key,
                "user_id": user_id,
                "category": category_label,
                "content": content,
                "metadata": metadata,
                "weight": confidence,
                "sources": sources
            })

    _prepare_entries(data.get("thread_notes", []), "thread", "thread_note")
    if user and scopes.get("user"):
        _prepare_entries(data.get("user_traits", []), "user", "user_trait")
    _prepare_entries(data.get("global_themes", []), "global", "global_theme")

    bias_data = data.get("bias_overrides") or {}
    for scope_key, category_label in (("thread", "bias"), ("user", "bias"), ("global", "bias")):
        bias_text = (bias_data.get("bias") or "").strip()
        narrative_text = (bias_data.get("narrative") or "").strip()
        if not bias_text and not narrative_text:
            continue
        chat_id, thread_key, user_id = scopes.get(scope_key, (0, 0, 0))
        merged = "; ".join(filter(None, [narrative_text, bias_text]))
        if not merged:
            continue
        metadata = {**base_metadata, "confidence": 0.8, "tags": ["bias"]}
        sources = []
        if source_msg_id:
            sources.append({
                "type": "message",
                "id": int(source_msg_id),
                "weight": 0.8,
                "metadata": {"scope": scope_key, "category": category_label}
            })
        entries.append({
            "scope": scope_key,
            "chat_id": chat_id,
            "thread_id": thread_key,
            "user_id": user_id,
            "category": category_label,
            "content": merged[:240],
            "metadata": metadata,
            "weight": 0.8,
            "sources": sources
        })
        break

    for rel in data.get("relationships", []) or []:
        if not isinstance(rel, dict):
            continue
        scope_key = (rel.get("scope") or "thread").strip().lower()
        if scope_key not in ("thread", "user", "global"):
            scope_key = "thread"
        description = (rel.get("description") or "").strip()
        if not description:
            continue
        confidence = float(rel.get("confidence", 0.6))
        tags = rel.get("tags") or []
        participants = rel.get("participants") or []
        metadata = {
            **base_metadata,
            "confidence": confidence,
            "tags": tags,
            "participants": participants
        }
        chat_id, thread_key, user_id = scopes.get(scope_key, (0, 0, 0))
        sources = []
        if source_msg_id:
            sources.append({
                "type": "message",
                "id": int(source_msg_id),
                "weight": confidence,
                "metadata": {"scope": scope_key, "category": "relationship"}
            })
        entries.append({
            "scope": scope_key,
            "chat_id": chat_id,
            "thread_id": thread_key,
            "user_id": user_id,
            "category": "relationship",
            "content": description[:240],
            "metadata": metadata,
            "weight": confidence,
            "sources": sources
        })

    self_ref = data.get("self_reflection") or {}
    if isinstance(self_ref, dict):
        self_text = (self_ref.get("content") or "").strip()
        if self_text:
            confidence = float(self_ref.get("confidence", 0.7))
            tags = self_ref.get("tags") or ["self"]
            metadata = {**base_metadata, "confidence": confidence, "tags": tags}
            sources = []
            if source_msg_id:
                sources.append({
                    "type": "message",
                    "id": int(source_msg_id),
                    "weight": confidence,
                    "metadata": {"scope": "global", "category": "self_reflection"}
                })
            entries.append({
                "scope": "global",
                "chat_id": 0,
                "thread_id": 0,
                "user_id": 0,
                "category": "self_reflection",
                "content": self_text[:240],
                "metadata": metadata,
                "weight": confidence,
                "sources": sources
            })

    if entries:
        stored_entries = await _store_memories_with_decay(entries)
        await _update_graph_from_relationship_entries(stored_entries, chat, thread_id)

# ──────────────────────────────────────────────────────────────
# 16b) Dev utilities — auto-reload on file change
# ──────────────────────────────────────────────────────────────
AUTO_RELOAD_THREAD: Optional[threading.Thread] = None
AUTO_RELOAD_EVENT = threading.Event()

def _candidate_watch_files() -> Dict[Path, float]:
    files: Dict[Path, float] = {}
    for raw in AUTO_RELOAD_PATHS:
        if not raw:
            continue
        is_glob = any(ch in raw for ch in "*?[]")
        base = Path(raw)
        if not base.is_absolute():
            base = ROOT / raw
        candidates: Iterable[Path]
        if is_glob:
            candidates = [Path(p) for p in glob.glob(str(base), recursive=True)]
        elif base.is_dir():
            candidates = base.rglob("*.py")
        else:
            candidates = [base]
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                files[path.resolve()] = path.stat().st_mtime
            except FileNotFoundError:
                continue
    return files

def _watch_loop():
    baseline = _candidate_watch_files()
    if not baseline:
        print("[reload] No valid files to watch; auto-reload disabled.")
        return
    interval = max(0.2, float(AUTO_RELOAD_INTERVAL or 1.0))
    print(f"[reload] Watching {len(baseline)} file(s) for changes every {interval:.2f}s …")
    while not AUTO_RELOAD_EVENT.is_set():
        time.sleep(interval)
        current = _candidate_watch_files()
        changed = False
        if baseline.keys() != current.keys():
            changed = True
        else:
            for path, ts in current.items():
                if ts != baseline.get(path):
                    changed = True
                    break
        if changed:
            print(f"[reload] Change detected; restarting… ({datetime.now().isoformat(timespec='seconds')})")
            try:
                AUTO_RELOAD_EVENT.set()
                time.sleep(0.2)
                os.execv(sys.executable, [sys.executable, *sys.argv])
            except Exception as exc:
                print(f"[reload] Restart failed: {exc}", file=sys.stderr)
                os._exit(1)
        baseline = current

def start_auto_reloader():
    global AUTO_RELOAD_THREAD
    if not AUTO_RELOAD_ENABLED:
        return
    if AUTO_RELOAD_THREAD and AUTO_RELOAD_THREAD.is_alive():
        return
    AUTO_RELOAD_THREAD = threading.Thread(target=_watch_loop, name="auto-reload", daemon=True)
    AUTO_RELOAD_THREAD.start()

# ──────────────────────────────────────────────────────────────
# 15.5) Document and Image Handlers (RAG + Vision)
# ──────────────────────────────────────────────────────────────

async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads for RAG ingestion"""
    if not RAG_STORE:
        return

    m = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if not m or not m.document or not user:
        return

    # Download the file
    try:
        file = await context.bot.get_file(m.document.file_id)
        file_path = ROOT / "data" / "uploads" / f"{user.id}_{m.document.file_name}"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        await file.download_to_drive(str(file_path))

        # Notify user
        await m.reply_text(f"📄 Processing document: {m.document.file_name}...")

        # Ingest into RAG
        use_vision = chat.type == "private" or (user.id in ADMIN_WHITELIST)  # Vision only for DMs or admins
        doc_id = RAG_STORE.add_document_from_telegram(
            file_path=file_path,
            user_id=user.id,
            chat_id=chat.id,
            message_id=m.message_id,
            use_vision=use_vision,
        )

        if doc_id:
            await m.reply_text(
                f"✅ Document ingested successfully!\n"
                f"Document ID: {doc_id[:8]}\n"
                f"You can now ask questions about this document."
            )
        else:
            await m.reply_text("❌ Failed to process document. Please check the file format.")

    except Exception as e:
        print(f"[rag] Document upload failed: {e}")
        await m.reply_text(f"❌ Error processing document: {str(e)}")


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads for vision analysis"""
    if not VISION_HANDLER:
        return

    m = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if not m or not m.photo or not user:
        return

    # Get the largest photo
    photo = m.photo[-1]

    # Download the photo
    try:
        file = await context.bot.get_file(photo.file_id)
        file_path = ROOT / "data" / "images" / f"{user.id}_{photo.file_id}.jpg"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        await file.download_to_drive(str(file_path))

        # Analyze with vision model
        caption = m.caption or "Describe this image in detail"

        await m.reply_text("🔍 Analyzing image...")

        result = VISION_HANDLER.analyze_image(
            image_path=file_path,
            user_prompt=caption,
        )

        if result.success:
            response = f"🖼️ Image Analysis:\n\n{result.description}"

            # If there's ongoing conversation context, offer to continue
            if chat.type == "private":
                response += "\n\n💬 Feel free to ask questions about this image!"

            await m.reply_text(response)
        else:
            await m.reply_text(f"❌ Vision analysis failed: {result.error}")

    except Exception as e:
        print(f"[vision] Photo analysis failed: {e}")
        await m.reply_text(f"❌ Error analyzing image: {str(e)}")


# ──────────────────────────────────────────────────────────────
# 15.9) Intelligent routing helper
# ──────────────────────────────────────────────────────────────

async def handle_real_tool_execution(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    user: Any,
    chat: Any,
) -> Optional[bool]:
    """
    Use REAL tool execution with LLM decision making and Telegram UI.

    Returns True if tool was executed, False/None otherwise.
    """
    if not REAL_TOOL_EXECUTION_AVAILABLE or not user:
        return None

    # Only for admin users
    if user.id not in ADMIN_WHITELIST:
        return None

    try:
        # Step 1: Use LLM to decide if tool is needed
        decision = await ToolDecisionMaker.decide_tool_from_message(
            text,
            ollama_model=OLLAMA_MODEL,
            ollama_url=OLLAMA_URL,
        )

        # If no tool needed, return None to proceed with normal flow
        if decision.decision_type == ToolDecisionType.DIRECT_REPLY:
            return None

        # Step 2: Send confirmation message with buttons
        if decision.needs_confirmation:
            confirmation_msg = await ToolTelegramUI.send_tool_confirmation(
                update,
                context,
                decision,
            )

            # For now, auto-execute (later we'll wait for button press)
            # TODO: Store decision in context.user_data and wait for callback
            await asyncio.sleep(1)  # Brief pause to show confirmation

        # Step 3: Execute tool with progress UI
        result = await ToolTelegramUI.execute_tool_with_progress_ui(
            update,
            context,
            decision,
            progress_message=confirmation_msg if decision.needs_confirmation else None,
        )

        # Tool was executed (result will be shown by UI)
        return True

    except Exception as e:
        print(f"[real_tool] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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

        # Try real tool execution first (for admin users)
        if REAL_TOOL_EXECUTION_AVAILABLE and user and user.id in ADMIN_WHITELIST:
            tool_executed = await handle_real_tool_execution(update, context, text, user, chat)
            if tool_executed:
                return  # Tool was executed, UI handled response
        reply_to = getattr(m, "reply_to_message", None)
        if reply_to and getattr(reply_to, "from_user", None) and user and meta:
            src_id = getattr(user, "id", None)
            tgt_id = getattr(reply_to.from_user, "id", None)
            if src_id and tgt_id:
                rel_meta = {
                    "kind": "reply",
                    "message_id": m.message_id,
                    "reply_to_message_id": getattr(reply_to, "message_id", None)
                }
                asyncio.create_task(update_relationship_graph_async(
                    src_id, tgt_id,
                    getattr(chat, "id", 0), thread_id,
                    "replied_to",
                    text[:200] if text else "(reply)",
                    m.message_id,
                    meta.get("ts", int(time.time())),
                    0.6,
                    rel_meta
                ))
                asyncio.create_task(update_relationship_graph_async(
                    tgt_id, src_id,
                    getattr(chat, "id", 0), thread_id,
                    "replied_by",
                    text[:200] if text else "(reply)",
                    m.message_id,
                    meta.get("ts", int(time.time())),
                    0.5,
                    rel_meta
                ))
        thread_ctx = fetch_thread_context(chat.id, thread_id, limit=24)
        sim = await similar_context(text, chat.id, thread_id, top_k=8)
        kg_snap = {"ents_here": kg_top_entities(chat.id, thread_id, 10),
                   "rels_here": kg_top_relations(chat.id, thread_id, 10)}
        if ai_available():
            internal_state = await recall_memories_async(text, chat, thread_id, user)
            context_bundle = assemble_context_bundle(chat, thread_id, user, text, thread_ctx, internal_state)
            complexity_meta = await assess_discussion_complexity(text, thread_ctx, internal_state, context_bundle, chat, user)
            decision_meta = {"reason": "direct-dm", "priority": "high"}
            executed_actions = []
            if complexity_meta.get("actions"):
                executed_actions = await run_agentic_expansions(complexity_meta["actions"], meta, chat, thread_id, thread_ctx, text, user)
                if executed_actions:
                    internal_state = await recall_memories_async(text, chat, thread_id, user)
                    context_bundle = assemble_context_bundle(chat, thread_id, user, text, thread_ctx, internal_state)
            if executed_actions:
                complexity_meta = {**complexity_meta, "actions_executed": executed_actions}
            if decision_meta:
                complexity_meta = {**complexity_meta, "auto_reply_reason": decision_meta}
            payload = build_ai_prompt(text, user, chat.id, thread_id, thread_ctx, sim, kg_snap,
                                      internal_state=internal_state,
                                      context_bundle=context_bundle,
                                      complexity_meta=complexity_meta)
            # Use more tools for admin users who have full tool access
            user_id = getattr(user, "id", None)
            is_admin_user = user_id in ADMIN_WHITELIST if user_id else False
            max_tools_allowed = 8 if is_admin_user else 3

            resp, tool_runs = await with_typing(
                context,
                chat.id,
                generate_agentic_reply(
                    payload,
                    user_id,
                    include_examples=True,
                    auto_execute=True,  # Execute tools automatically
                    max_tools=max_tools_allowed,
                ),
            )
            if resp:
                if tool_runs:
                    tool_summary = []
                    for run in tool_runs:
                        state_obj = getattr(run, "state", None)
                        state_val = getattr(state_obj, "value", state_obj)
                        result_preview = getattr(run, "result", None)
                        if result_preview is None:
                            preview_txt = "(None)"
                        else:
                            preview_txt = str(result_preview)
                        tool_summary.append(
                            {
                                "tool": getattr(run, "tool_name", ""),
                                "state": state_val,
                                "result_preview": preview_txt[:200],
                            }
                        )
                    complexity_meta = {**complexity_meta, "tool_executions": tool_summary}
                debug_enabled = DEBUG_FLAGS.get(chat.id, False)
                debug_context_text = ""
                reply_markup = None
                if debug_enabled:
                    debug_context_text = build_debug_context_text(
                        text,
                        payload,
                        context_bundle,
                        complexity_meta,
                        internal_state,
                        executed_actions,
                        tool_runs,
                        decision_meta,
                    )
                    reply_markup = InlineKeyboardMarkup(
                        [[InlineKeyboardButton("Show context", callback_data="debug:show")]]
                    )
                reply_msg = await m.reply_text(resp, reply_markup=reply_markup)
                if debug_enabled and reply_msg:
                    DEBUG_CONTEXTS[(reply_msg.chat_id, reply_msg.message_id)] = {
                        "response": resp,
                        "context": debug_context_text,
                    }
                asyncio.create_task(process_memory_update_async(meta, chat, thread_id, user,
                                                                thread_ctx, text, resp))
                mood_state = await analyze_user_affect_intent(meta, chat, thread_id, user, text)
                if not mood_state:
                    mood_state = get_cached_self_state()
                asyncio.create_task(maybe_update_system_prompt_from_reflection())
                asyncio.create_task(maybe_trigger_breakout_event(context, chat, user, thread_id,
                                                                 decision_meta, complexity_meta,
                                                                 mood_state, thread_ctx, text))
            else:
                await m.reply_text("AI is unavailable right now (Ollama not responding).")
            return
        else:
            return await m.reply_text("AI is disabled (set OLLAMA_URL and OLLAMA_MODEL/OLLAMA_EMBED_MODEL in .env).")

    # Groups/channels: respond on @mention/reply, else maybe (smart)
    text = (m.text or "").strip()
    if not text: return
    thread_id = getattr(m, "message_thread_id", None) or 0
    reply_to = getattr(m, "reply_to_message", None)
    if reply_to and getattr(reply_to, "from_user", None) and user and meta:
        src_id = getattr(user, "id", None)
        tgt_id = getattr(reply_to.from_user, "id", None)
        if src_id and tgt_id:
            rel_meta = {
                "kind": "reply",
                "message_id": m.message_id,
                "reply_to_message_id": getattr(reply_to, "message_id", None)
            }
            asyncio.create_task(update_relationship_graph_async(
                src_id, tgt_id,
                getattr(chat, "id", 0), thread_id,
                "replied_to",
                text[:200] if text else "(reply)",
                m.message_id,
                meta.get("ts", int(time.time())),
                0.6,
                rel_meta
            ))
            asyncio.create_task(update_relationship_graph_async(
                tgt_id, src_id,
                getattr(chat, "id", 0), thread_id,
                "replied_by",
                text[:200] if text else "(reply)",
                m.message_id,
                meta.get("ts", int(time.time())),
                0.5,
                rel_meta
            ))
    thread_ctx = fetch_thread_context(chat.id, thread_id, limit=24)
    internal_state = await recall_memories_async(text, chat, thread_id, user)
    must = mentions_bot(m) or (getattr(m,"reply_to_message",None) and m.reply_to_message.from_user and m.reply_to_message.from_user.id == BOT_CACHE["id"])
    context_bundle = assemble_context_bundle(chat, thread_id, user, text, thread_ctx, internal_state)
    complexity_meta = await assess_discussion_complexity(text, thread_ctx, internal_state, context_bundle, chat, user)
    should_reply, decision_meta = await should_auto_reply_async(m, chat, user, thread_ctx, internal_state, must)
    if not should_reply:
        return
    if decision_meta:
        complexity_meta = {**complexity_meta, "auto_reply_reason": decision_meta}
    executed_actions = []
    if complexity_meta.get("actions"):
        executed_actions = await run_agentic_expansions(complexity_meta["actions"], meta, chat, thread_id, thread_ctx, text, user)
        if executed_actions:
            internal_state = await recall_memories_async(text, chat, thread_id, user)
            context_bundle = assemble_context_bundle(chat, thread_id, user, text, thread_ctx, internal_state)
    if executed_actions:
        complexity_meta = {**complexity_meta, "actions_executed": executed_actions}
    sim = await similar_context(text, chat.id, thread_id, top_k=8)
    kg_snap = {"ents_here": kg_top_entities(chat.id, thread_id, 10),
               "rels_here": kg_top_relations(chat.id, thread_id, 10)}
    if ai_available():
        payload = build_ai_prompt(text, user, chat.id, thread_id, thread_ctx, sim, kg_snap,
                                  internal_state=internal_state,
                                  context_bundle=context_bundle,
                                  complexity_meta=complexity_meta)
        # Use more tools for admin users who have full tool access
        user_id = getattr(user, "id", None)
        is_admin_user = user_id in ADMIN_WHITELIST if user_id else False
        max_tools_allowed = 8 if is_admin_user else 3

        resp, tool_runs = await with_typing(
            context,
            chat.id,
            generate_agentic_reply(
                payload,
                user_id,
                include_examples=True,
                auto_execute=True,
                max_tools=max_tools_allowed,
            ),
        )
        if resp:
            if tool_runs:
                tool_summary = []
                for run in tool_runs:
                    state_obj = getattr(run, "state", None)
                    state_val = getattr(state_obj, "value", state_obj)
                    result_preview = getattr(run, "result", None)
                    if result_preview is None:
                        preview_txt = "(None)"
                    else:
                        preview_txt = str(result_preview)
                    tool_summary.append(
                        {
                            "tool": getattr(run, "tool_name", ""),
                            "state": state_val,
                            "result_preview": preview_txt[:200],
                        }
                    )
                complexity_meta = {**complexity_meta, "tool_executions": tool_summary}
            debug_enabled = DEBUG_FLAGS.get(chat.id, False)
            debug_context_text = ""
            reply_markup = None
            if debug_enabled:
                debug_context_text = build_debug_context_text(
                    text,
                    payload,
                    context_bundle,
                    complexity_meta,
                    internal_state,
                    executed_actions,
                    tool_runs,
                    decision_meta,
                )
                reply_markup = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Show context", callback_data="debug:show")]]
                )
            reply_msg = None
            try:
                reply_msg = await m.reply_text(resp, reply_markup=reply_markup)
            except Exception:
                reply_msg = await context.bot.send_message(chat_id=chat.id, text=resp, reply_markup=reply_markup)
            if debug_enabled and reply_msg:
                DEBUG_CONTEXTS[(reply_msg.chat_id, reply_msg.message_id)] = {
                    "response": resp,
                    "context": debug_context_text,
                }
            asyncio.create_task(process_memory_update_async(meta, chat, thread_id, user,
                                                            thread_ctx, text, resp))
            mood_state = await analyze_user_affect_intent(meta, chat, thread_id, user, text)
            if not mood_state:
                mood_state = get_cached_self_state()
            asyncio.create_task(maybe_update_system_prompt_from_reflection())
            asyncio.create_task(maybe_trigger_breakout_event(context, chat, user, thread_id,
                                                             decision_meta, complexity_meta,
                                                             mood_state, thread_ctx, text))
            return

# ──────────────────────────────────────────────────────────────
# 17) Unlock flow + greetings
# ──────────────────────────────────────────────────────────────
WELCOME_MESSAGES: Dict[tuple[int, int], set[int]] = defaultdict(set)
WELCOME_JOBS: Dict[tuple[int, int, int], object] = {}

def _get_job_queue(context: ContextTypes.DEFAULT_TYPE):
    jq = getattr(context, "job_queue", None)
    if jq:
        return jq
    app = getattr(context, "application", None)
    return getattr(app, "job_queue", None) if app else None

def _register_welcome_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, message_id: int):
    key = (chat_id, user_id)
    WELCOME_MESSAGES[key].add(message_id)
    job_queue = _get_job_queue(context)
    if job_queue:
        job = job_queue.run_once(
            _auto_delete_welcome_message_job,
            when=600,
            data={"chat_id": chat_id, "user_id": user_id, "message_id": message_id},
        )
        WELCOME_JOBS[(chat_id, user_id, message_id)] = job

async def _delete_single_welcome_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    message_id: int,
    executing_job=None,
):
    job_key = (chat_id, user_id, message_id)
    job = WELCOME_JOBS.pop(job_key, None)
    if job and job is not executing_job:
        try:
            job.schedule_removal()
        except Exception:
            pass
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass
    key = (chat_id, user_id)
    msgs = WELCOME_MESSAGES.get(key)
    if msgs:
        msgs.discard(message_id)
        if not msgs:
            WELCOME_MESSAGES.pop(key, None)

async def _auto_delete_welcome_message_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data or {}
    chat_id = data.get("chat_id")
    user_id = data.get("user_id")
    message_id = data.get("message_id")
    if chat_id is None or user_id is None or message_id is None:
        return
    await _delete_single_welcome_message(
        context, chat_id, user_id, message_id, executing_job=context.job
    )

async def cleanup_welcome_messages(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int):
    key = (chat_id, user_id)
    message_ids = list(WELCOME_MESSAGES.get(key, set()))
    for mid in message_ids:
        await _delete_single_welcome_message(context, chat_id, user_id, mid)

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
    await cleanup_welcome_messages(context, chat_id, user.id)
    await q.edit_message_text("✅ You’re unlocked. You can now receive messages from me.")
    try: await context.bot.send_message(chat_id, f"👋 {user.mention_html()} has been onboarded and unlocked.", parse_mode="HTML")
    except Exception: pass

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
def _summarize_update_for_error(update: object) -> str:
    if not isinstance(update, Update):
        return repr(update)

    parts: list[str] = []
    user = update.effective_user
    chat = update.effective_chat
    msg = update.effective_message

    if user:
        parts.append(f"user_id={user.id}")
        if user.username:
            parts.append(f"username=@{user.username}")
    if chat:
        parts.append(f"chat_id={chat.id}")
        parts.append(f"chat_type={chat.type}")
    if msg:
        snippet = (msg.text or msg.caption or "").replace("\n", " ").strip()
        if snippet:
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            parts.append(f"text={snippet}")

    return " ".join(parts) if parts else "(no update context)"


def _append_handler_error_log(update_info: str, trace_text: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = [
        "=" * 60,
        f"{timestamp}",
        update_info or "(no update context)",
        (trace_text or "No traceback captured.").rstrip(),
        "",
    ]
    try:
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(entry))
    except Exception as log_exc:
        print(f"[error-log] Failed to persist handler error: {log_exc}", file=sys.stderr)


def _tail_traceback_text(trace_text: str, *, max_chars: int = 3500, max_lines: int = 12) -> tuple[str, bool]:
    stripped = (trace_text or "").strip()
    if not stripped:
        return "", False

    lines = stripped.splitlines()
    truncated = False
    if max_lines and len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated = True

    snippet = "\n".join(lines).strip()
    if max_chars and len(snippet) > max_chars:
        snippet = snippet[-max_chars:]
        truncated = True

    return snippet, truncated


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    try:
        exc = getattr(context, "error", None)
        trace_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ) if exc else "No exception information available."

        update_info = _summarize_update_for_error(update)

        print("Exception in handler:", file=sys.stderr)
        if exc:
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        else:
            print("No exception object available.", file=sys.stderr)

        _append_handler_error_log(update_info, trace_text)

        notified_ids: set[int] = set()

        if isinstance(update, Update) and update.effective_message:
            snippet, truncated = _tail_traceback_text(trace_text)
            snippet_html = html.escape(snippet) if snippet else ""

            context_snippet = update_info
            if len(context_snippet) > 500:
                context_snippet = context_snippet[:497] + "..."

            lines = [
                "⚠️ Handler error encountered.",
                "Full traceback saved to <code>data/handler_errors.log</code>.",
            ]
            if context_snippet:
                lines.append(f"Context: <code>{html.escape(context_snippet)}</code>")
            if snippet_html:
                label = "Last traceback lines"
                if truncated:
                    label += " (truncated)"
                lines.append(f"{label}:<br><pre>{snippet_html}</pre>")

            message_body = "<br>".join(lines)
            try:
                await update.effective_message.reply_text(message_body, parse_mode="HTML")
                if update.effective_user:
                    notified_ids.add(update.effective_user.id)
            except Exception:
                pass

        bot = getattr(context, "bot", None)
        if bot and ADMIN_WHITELIST:
            admin_snippet, admin_truncated = _tail_traceback_text(
                trace_text, max_chars=3500, max_lines=20
            )
            admin_snippet_html = html.escape(admin_snippet) if admin_snippet else ""

            admin_lines = [
                "⚠️ Handler error detected.",
                f"Context: <code>{html.escape(update_info[:800])}</code>" if update_info else "Context: (none)",
                "Full traceback saved to <code>data/handler_errors.log</code>.",
            ]
            if admin_snippet_html:
                label = "Traceback tail"
                if admin_truncated:
                    label += " (truncated)"
                admin_lines.append(f"{label}:<br><pre>{admin_snippet_html}</pre>")

            admin_message = "<br>".join(admin_lines)

            for admin_id in ADMIN_WHITELIST:
                if admin_id in notified_ids:
                    continue
                try:
                    await bot.send_message(admin_id, admin_message, parse_mode="HTML")
                except Exception:
                    continue
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# 21) Wire up + run (safe JobQueue + DBW start)
# ──────────────────────────────────────────────────────────────
def main():
    start_auto_reloader()

    # Start memory visualizer if available and enabled
    if MEMORY_VISUALIZER_ENABLED:
        if VISUALIZER_AVAILABLE:
            print("[visualizer] Starting memory consciousness visualizer...")
            start_visualizer(force_curses=FORCE_CURSES)
            if FORCE_CURSES:
                print("[visualizer] Curses mode forced. Make sure you're running in tmux/screen!")
            print("[visualizer] Visualizer running. Press 'q' in the visualizer to close it.")
        else:
            start_visualizer()

    # Ensure base schema present
    with closing(db()): pass

    # Start global write-queue (threaded; no loop required)
    global DBW
    DBW = DBWriteQueue()
    DBW.start()

    # Initialize sleep cycle if enabled
    if SLEEP_CYCLE_ENABLED and SLEEP_CYCLE_AVAILABLE:
        print("[sleep] Initializing autonomous sleep cycle...")
        init_sleep_cycle(DB_PATH, visualizer_log_fn=log_autonomous)
        print(f"[sleep] Sleep cycle enabled. Tick interval: {SLEEP_CYCLE_TICK_SECONDS}s")
    elif SLEEP_CYCLE_ENABLED and not SLEEP_CYCLE_AVAILABLE:
        print("[sleep] Sleep cycle enabled but module not available.")

    # Initialize intelligent routing, RAG, and vision
    init_intelligent_components()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Gating
    app.add_handler(CommandHandler("setup", setup))
    app.add_handler(CommandHandler("ungate", ungate))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(CommandHandler("linkprimary", linkprimary))
    app.add_handler(CommandHandler("setprimary", setprimary))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(agree_cb, pattern=r"^agree:\-?\d+$"))

    # Admin tools
    app.add_handler(CommandHandler("config", config_cmd))
    app.add_handler(CommandHandler("users", users_cmd))
    app.add_handler(CommandHandler("message", message_cmd))
    app.add_handler(CommandHandler("system", system_cmd))
    app.add_handler(CommandHandler("inspect", inspect_cmd))
    app.add_handler(CommandHandler("autosystem", autosystem_cmd))
    app.add_handler(CommandHandler("debug", debug_cmd))
    app.add_handler(CommandHandler("tools", tools_cmd))

    # Profiles (DM)
    app.add_handler(CommandHandler("profile", profile_cmd))

    # /setadmin + approvals
    app.add_handler(CommandHandler("setadmin", setadmin_cmd))
    app.add_handler(CallbackQueryHandler(adminreq_cb, pattern=r"^adminreq:(approve|deny):\d+$"))
    app.add_handler(CallbackQueryHandler(debug_toggle_cb, pattern=r"^debug:(show|hide)$"))

    # Tool execution callbacks (confirmation and rating)
    if REAL_TOOL_EXECUTION_AVAILABLE:
        app.add_handler(CallbackQueryHandler(handle_tool_confirmation_callback, pattern=r"^tool_confirm:"))
        app.add_handler(CallbackQueryHandler(handle_rating_callback, pattern=r"^tool_rating:"))

    # Topic / Graph
    app.add_handler(CommandHandler("commands", commands_cmd))
    app.add_handler(CommandHandler("topic", topic_cmd))
    app.add_handler(CommandHandler("graph", graph_cmd))

    # Document and photo handlers (RAG + Vision)
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))

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
    if jq:
        jq.run_repeating(hierarchical_rollup_job, interval=SUMMARY_ROLLUP_INTERVAL_SECONDS, first=SUMMARY_ROLLUP_INTERVAL_SECONDS)

    # Schedule sleep cycle job
    if SLEEP_CYCLE_ENABLED and SLEEP_CYCLE_AVAILABLE and jq:
        jq.run_repeating(sleep_cycle_job, interval=SLEEP_CYCLE_TICK_SECONDS, first=SLEEP_CYCLE_TICK_SECONDS)
        print(f"[sleep] Sleep cycle job scheduled (every {SLEEP_CYCLE_TICK_SECONDS}s)")

    print("Bot running…")
    app.run_polling()

if __name__ == "__main__":
    main()
