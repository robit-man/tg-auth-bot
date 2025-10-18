#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Gatekeeper Bot — self-bootstrapping, env-scaffolded, per-user unlock.

Features
- Auto-creates .venv and installs requirements (python-telegram-bot + python-dotenv).
- Generates .env with BOT_TOKEN and PRIMARY_CHAT_ID (blank) on first run.
- /setup makes group read-only by default and prints a Start link pinned for onboarding.
- DM /start unlock flow (agree button) → bot grants send perms in the group.
- /setprimary (run in a group) stores that group's chat_id into .env as PRIMARY_CHAT_ID.
- /link shows the Start link for the current group; /linkprimary shows the primary's link.

Run
  python gatekeeper_bot.py
Then edit .env (BOT_TOKEN=...) and run again.

Requires Python 3.10+.
"""

import os, sys, subprocess, textwrap, sqlite3, base64, secrets, asyncio
from pathlib import Path
from contextlib import closing
from functools import wraps
from typing import Optional

# ──────────────────────────────────────────────────────────────
# 0) Self-bootstrap a local venv and deps, then re-exec in it
# ──────────────────────────────────────────────────────────────
ROOT = Path.cwd()
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"

def _venv_python(p: Path) -> Path:
    return p / ("Scripts/python.exe" if IS_WIN else "bin/python")

def ensure_venv_and_deps():
    # already inside a venv?
    if os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(".venv"):
        return
    # create venv if missing
    if not VENV.exists():
        print("[bootstrap] Creating .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    py = str(_venv_python(VENV))
    # upgrade pip tooling
    print("[bootstrap] Upgrading pip/setuptools/wheel ...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # install runtime deps
    reqs = [
        # Pinned to the stable PTB v20 branch
        "python-telegram-bot>=20,<21",
        "python-dotenv>=1.0,<2.0",
    ]
    print("[bootstrap] Installing requirements ...")
    subprocess.check_call([py, "-m", "pip", "install", *reqs])
    # Re-exec the script within the venv
    print("[bootstrap] Re-exec in .venv ...")
    os.execv(py, [py, *sys.argv])

ensure_venv_and_deps()

# From here on, we are inside the venv with deps available.
from dotenv import load_dotenv, dotenv_values
from telegram import (
    Update, ChatPermissions, InlineKeyboardMarkup, InlineKeyboardButton
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

# ──────────────────────────────────────────────────────────────
# 1) .env scaffold + config
# ──────────────────────────────────────────────────────────────
ENV_PATH = ROOT / ".env"

ENV_TEMPLATE = textwrap.dedent("""\
    # Telegram Gatekeeper Bot — environment
    # Paste your bot token from @BotFather:
    BOT_TOKEN=

    # (Optional) Primary server (group) chat id, e.g. -1001234567890
    # You can set this by running /setprimary in the target group, too.
    PRIMARY_CHAT_ID=
    # Optional database file name:
    GATE_DB=gate.db
    """)

def ensure_env_file() -> dict:
    # Create .env if missing
    if not ENV_PATH.exists():
        ENV_PATH.write_text(ENV_TEMPLATE, encoding="utf-8")
        print("[env] Wrote .env (fill in BOT_TOKEN, then run again).")
        sys.exit(0)
    load_dotenv(ENV_PATH)
    cfg = dotenv_values(ENV_PATH)
    return cfg

def _write_env_var(key: str, value: str):
    # Safe append/replace behavior
    lines = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

CFG = ensure_env_file()
BOT_TOKEN = os.getenv("BOT_TOKEN") or (CFG.get("BOT_TOKEN") or "").strip()
PRIMARY_CHAT_ID = os.getenv("PRIMARY_CHAT_ID") or (CFG.get("PRIMARY_CHAT_ID") or "").strip()
DB_PATH = os.getenv("GATE_DB") or (CFG.get("GATE_DB") or "gate.db")

if not BOT_TOKEN:
    print("[env] BOT_TOKEN is empty in .env. Set it and run again.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# 2) Minimal SQLite state (groups, tokens, allowlist)
# ──────────────────────────────────────────────────────────────
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""
      CREATE TABLE IF NOT EXISTS groups(
        chat_id INTEGER PRIMARY KEY,
        title TEXT,
        token TEXT UNIQUE
      );
    """)
    con.execute("""
      CREATE TABLE IF NOT EXISTS allowed(
        chat_id INTEGER,
        user_id INTEGER,
        PRIMARY KEY(chat_id, user_id)
      );
    """)
    return con

def get_or_create_group(chat_id: int, title: str) -> str:
    with closing(db()) as con:
        cur = con.execute("SELECT token FROM groups WHERE chat_id=?", (chat_id,))
        row = cur.fetchone()
        if row:
            return row[0]
        token = base64.urlsafe_b64encode(secrets.token_bytes(12)).decode().rstrip("=")
        con.execute("INSERT INTO groups(chat_id,title,token) VALUES(?,?,?)",
                    (chat_id, title or "", token))
        con.commit()
        return token

def token_to_chat_id(token: str) -> Optional[int]:
    with closing(db()) as con:
        cur = con.execute("SELECT chat_id FROM groups WHERE token=?", (token,))
        r = cur.fetchone()
        return r[0] if r else None

def mark_allowed(chat_id: int, user_id: int):
    with closing(db()) as con:
        con.execute("INSERT OR IGNORE INTO allowed(chat_id,user_id) VALUES(?,?)",
                    (chat_id, user_id))
        con.commit()

def is_allowed(chat_id: int, user_id: int) -> bool:
    with closing(db()) as con:
        cur = con.execute("SELECT 1 FROM allowed WHERE chat_id=? AND user_id=?",
                          (chat_id, user_id))
        return cur.fetchone() is not None

# ──────────────────────────────────────────────────────────────
# 3) Helpers / decorators
# ──────────────────────────────────────────────────────────────
def admin_only(fn):
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        user = update.effective_user
        if not chat or chat.type not in ("group", "supergroup"):
            return await update.effective_message.reply_text("Run this in a group.")
        member = await context.bot.get_chat_member(chat.id, user.id)
        if member.status not in ("administrator", "creator"):
            return await update.effective_message.reply_text("Admins only.")
        return await fn(update, context)
    return wrapper

async def make_group_readonly(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    perms = ChatPermissions(
        can_send_messages=False,
        can_send_audios=False,
        can_send_documents=False,
        can_send_photos=False,
        can_send_videos=False,
        can_send_video_notes=False,
        can_send_voice_notes=False,
        can_send_polls=False,
        can_send_other_messages=False,
        can_add_web_page_previews=False,
        can_change_info=False,
        can_invite_users=False,
        can_pin_messages=False,
    )
    await context.bot.set_chat_permissions(chat_id, perms)

async def allow_user(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int):
    perms = ChatPermissions(
        can_send_messages=True,
        can_send_audios=True,
        can_send_documents=True,
        can_send_photos=True,
        can_send_videos=True,
        can_send_video_notes=True,
        can_send_voice_notes=True,
        can_send_polls=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True,
    )
    await context.bot.restrict_chat_member(
        chat_id=chat_id,
        user_id=user_id,
        permissions=perms,
        use_independent_chat_permissions=True
    )
    mark_allowed(chat_id, user_id)

# ──────────────────────────────────────────────────────────────
# 4) Commands
# ──────────────────────────────────────────────────────────────
@admin_only
async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await make_group_readonly(context, chat.id)
    token = get_or_create_group(chat.id, chat.title or "")
    bot_username = (await context.bot.get_me()).username
    start_link = f"https://t.me/{bot_username}?start=unlock_{token}"

    # If PRIMARY_CHAT_ID is blank, set it to this group.
    global PRIMARY_CHAT_ID
    if not PRIMARY_CHAT_ID:
        PRIMARY_CHAT_ID = str(chat.id)
        _write_env_var("PRIMARY_CHAT_ID", PRIMARY_CHAT_ID)

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
        can_send_messages=True,
        can_send_audios=True,
        can_send_documents=True,
        can_send_photos=True,
        can_send_videos=True,
        can_send_video_notes=True,
        can_send_voice_notes=True,
        can_send_polls=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True,
        can_change_info=False,
        can_invite_users=True,
        can_pin_messages=False,
    )
    await context.bot.set_chat_permissions(chat.id, perms)
    await update.effective_message.reply_text("🚪 Gating disabled and defaults restored.")

@admin_only
async def link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    token = get_or_create_group(chat.id, chat.title or "")
    bot_username = (await context.bot.get_me()).username
    start_link = f"https://t.me/{bot_username}?start=unlock_{token}"
    await update.effective_message.reply_text(
        f"🔗 Start link for this group:\n{start_link}",
        disable_web_page_preview=True
    )

async def linkprimary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not PRIMARY_CHAT_ID:
        return await update.effective_message.reply_text(
            "PRIMARY_CHAT_ID is not set. Run /setprimary in your target group."
        )
    try:
        chat_id = int(PRIMARY_CHAT_ID)
        chat = await context.bot.get_chat(chat_id)
        token = get_or_create_group(chat.id, chat.title or "")
        bot_username = (await context.bot.get_me()).username
        start_link = f"https://t.me/{bot_username}?start=unlock_{token}"
        await update.effective_message.reply_text(
            f"🔗 Primary group: <b>{chat.title}</b>\n{start_link}",
            parse_mode="HTML",
            disable_web_page_preview=True
        )
    except Exception as e:
        await update.effective_message.reply_text(
            f"Could not fetch PRIMARY_CHAT_ID ({PRIMARY_CHAT_ID}). Try /setprimary in the group."
        )

@admin_only
async def setprimary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    _write_env_var("PRIMARY_CHAT_ID", str(chat.id))
    global PRIMARY_CHAT_ID
    PRIMARY_CHAT_ID = str(chat.id)
    await update.effective_message.reply_text(
        f"✅ Primary server set to: {chat.title} ({chat.id}). Saved to .env as PRIMARY_CHAT_ID."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    args = (context.args or [])
    token = None
    if args and args[0].startswith("unlock_"):
        token = args[0].split("unlock_", 1)[1]

    if not token:
        return await update.message.reply_text(
            "Hi! To chat in a gated group, tap the pinned Start link in that group.\n"
            "Admins: run /setup in the group to enable gating."
        )

    chat_id = token_to_chat_id(token)
    if not chat_id:
        return await update.message.reply_text("That link is invalid or expired. Ask an admin for a fresh one.")

    rules = "Before we unlock you, please confirm you’ve read the group rules."
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ I agree — unlock me", callback_data=f"agree:{chat_id}")]])
    await update.message.reply_text(rules, reply_markup=kb)

async def agree_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if not data.startswith("agree:"):
        return
    chat_id = int(data.split("agree:", 1)[1])
    user = query.from_user

    try:
        member = await context.bot.get_chat_member(chat_id, user.id)
        if member.status in ("left", "kicked"):
            await query.edit_message_text("Join the group first, then tap the Start link again.")
            return
    except Exception:
        await query.edit_message_text("I couldn’t check your membership. Ask an admin to verify the setup.")
        return

    try:
        await allow_user(context, chat_id, user.id)
    except Exception:
        await query.edit_message_text("Couldn’t unlock you (bot needs admin rights with 'Manage Members').")
        return

    await query.edit_message_text("✅ You’re unlocked. You can now chat in the group.")
    try:
        await context.bot.send_message(chat_id, f"👋 {user.mention_html()} has been onboarded and unlocked.", parse_mode="HTML")
    except Exception:
        pass

async def greet_new_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    if not update.message or not update.message.new_chat_members:
        return
    token = get_or_create_group(chat.id, chat.title or "")
    bot_username = (await context.bot.get_me()).username
    start_link = f"https://t.me/{bot_username}?start=unlock_{token}"
    names = ", ".join(m.mention_html() for m in update.message.new_chat_members)
    text = (
        f"Welcome {names}! 🔒 This group is gated.\n\n"
        f"Tap this link to DM the bot and unlock chat: {start_link}"
    )
    await update.message.reply_html(text, disable_web_page_preview=True)

# ──────────────────────────────────────────────────────────────
# 5) Main
# ──────────────────────────────────────────────────────────────
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("setup", setup))
    app.add_handler(CommandHandler("ungate", ungate))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(CommandHandler("linkprimary", linkprimary))
    app.add_handler(CommandHandler("setprimary", setprimary))
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(agree_cb, pattern=r"^agree:\-?\d+$"))
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, greet_new_members))

    print("Bot running…")
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    asyncio.run(main())
