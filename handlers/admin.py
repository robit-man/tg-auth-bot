#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Admin Handlers - Commands for administrators and whitelisted users
"""

import re
import sqlite3
import textwrap
from contextlib import closing
from typing import Optional, List
from datetime import datetime
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes

# Import decorators and utilities
from telegram_utils import whitelist_only

# Import from bot_server temporarily (will be refactored)
import bot_server


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def db():
    """Get database connection"""
    con = sqlite3.connect(bot_server.DB_PATH, timeout=60)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def _write_env_var(key: str, value: str):
    """Write or update environment variable in .env file"""
    from pathlib import Path
    env_path = Path(bot_server.ROOT) / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def settings_get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get setting from database"""
    with closing(db()) as con:
        r = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r[0] if r else default


async def settings_set_async(key: str, value: str):
    """Set setting in database"""
    def _work(con: sqlite3.Connection):
        con.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )
        con.commit()
    await bot_server.DBW.run(bot_server.DB_PATH, _work)


def list_allowed(chat_id: int, limit: int = 50, offset: int = 0, search: Optional[str] = None) -> tuple[int, List[tuple]]:
    """
    List allowed users for a chat with pagination and search
    Returns (total_count, rows) where rows are (user_id, username, first_name, last_name)
    """
    with closing(db()) as con:
        if search:
            search_pattern = f"%{search}%"
            total = con.execute(
                """SELECT COUNT(*) FROM allowed
                   WHERE chat_id=? AND (
                       username LIKE ? OR
                       first_name LIKE ? OR
                       last_name LIKE ? OR
                       CAST(user_id AS TEXT) LIKE ?
                   )""",
                (chat_id, search_pattern, search_pattern, search_pattern, search_pattern)
            ).fetchone()[0]
            rows = con.execute(
                """SELECT user_id, username, first_name, last_name
                   FROM allowed
                   WHERE chat_id=? AND (
                       username LIKE ? OR
                       first_name LIKE ? OR
                       last_name LIKE ? OR
                       CAST(user_id AS TEXT) LIKE ?
                   )
                   ORDER BY user_id
                   LIMIT ? OFFSET ?""",
                (chat_id, search_pattern, search_pattern, search_pattern, search_pattern, limit, offset)
            ).fetchall()
        else:
            total = con.execute("SELECT COUNT(*) FROM allowed WHERE chat_id=?", (chat_id,)).fetchone()[0]
            rows = con.execute(
                "SELECT user_id, username, first_name, last_name FROM allowed WHERE chat_id=? ORDER BY user_id LIMIT ? OFFSET ?",
                (chat_id, limit, offset)
            ).fetchall()
    return total, rows


def resolve_user_by_username(chat_id: Optional[int], username: str) -> Optional[int]:
    """Resolve username to user_id"""
    uname = username.lstrip("@").lower()
    with closing(db()) as con:
        q = "SELECT user_id FROM allowed WHERE LOWER(username)=?"
        args = (uname,)
        if chat_id is not None:
            q += " AND chat_id=?"
            args = (uname, chat_id)
        r = con.execute(q, args).fetchall()
        if r:
            return r[0][0]
        r2 = con.execute(
            "SELECT DISTINCT user_id FROM messages WHERE LOWER(username)=? ORDER BY date DESC",
            (uname,)
        ).fetchall()
        return r2[0][0] if r2 else None


def add_admin_ids(ids_to_add):
    """Add user IDs to admin whitelist"""
    bot_server.ADMIN_WHITELIST |= {int(i) for i in ids_to_add}
    _write_env_var("ADMIN_WHITELIST", ",".join(str(i) for i in sorted(bot_server.ADMIN_WHITELIST)))


def parse_users_args(args: List[str]) -> tuple[str, int, int, Optional[str]]:
    """
    Parse arguments for /users command
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
                size = int(a.split("=", 1)[1])
            except:
                pass
    size = max(1, min(size, 200))
    page = max(1, page)
    return scope, page, size, search


# ──────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────

@whitelist_only
async def users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /users - List unlocked users for a group with pagination and search
    Usage: /users [here|primary] [page] [size=N] [q=search]
    """
    args = context.args or []
    scope, page, size, search = parse_users_args(args)

    if scope == "here":
        chat = update.effective_chat
        if not chat or chat.type not in ("group", "supergroup"):
            scope = "primary"

    if scope == "primary":
        if not bot_server.PRIMARY_CHAT_ID:
            return await update.effective_message.reply_text("PRIMARY_CHAT_ID is not set.")
        try:
            chat_id = int(bot_server.PRIMARY_CHAT_ID)
            chat = await context.bot.get_chat(chat_id)
        except Exception:
            return await update.effective_message.reply_text(f"Invalid PRIMARY_CHAT_ID: {bot_server.PRIMARY_CHAT_ID}")
    else:
        chat_id = update.effective_chat.id
        chat = update.effective_chat

    total, rows = list_allowed(chat_id, limit=size, offset=(page - 1) * size, search=search)
    if total == 0:
        return await update.effective_message.reply_text("No unlocked users found for this group.")

    parts = [f"<b>Unlocked users for {chat.title}</b> (total {total}, page {page}, size {size}{', q=' + search if search else ''})"]
    for uid, uname, first, last in rows:
        name = " ".join(p for p in (first, last) if p).strip() or (uname or f"id {uid}")
        handle = f"@{uname}" if uname else f"id {uid}"
        parts.append(f"• <a href=\"tg://user?id={uid}\">{name}</a> ({handle})")

    if page * size < total:
        parts.append(f"\nType: /users {scope} {page + 1} size={size}" + (f" q={search}" if search else ""))

    await update.effective_message.reply_html("\n".join(parts), disable_web_page_preview=True)


@whitelist_only
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /message - Send a DM to a user by ID or username
    Usage: /message <user_id|@username> <text...>
    """
    if not context.args or len(context.args) < 2:
        return await update.effective_message.reply_text("Usage: /message <user_id|@username> <text...>")

    target = context.args[0]
    text_to = " ".join(context.args[1:]).strip()
    if not text_to:
        return await update.effective_message.reply_text("Message text cannot be empty.")

    uid = None
    if target.startswith("@"):
        uid = resolve_user_by_username(None, target)
    else:
        try:
            uid = int(target)
        except ValueError:
            return await update.effective_message.reply_text("First arg must be numeric user_id or @username.")

    if uid is None:
        return await update.effective_message.reply_text(f"Could not resolve {target}.")

    try:
        await context.bot.send_message(chat_id=uid, text=text_to)
        await update.effective_message.reply_text(f"✅ Sent to {uid}.")
    except Exception:
        await update.effective_message.reply_text(
            f"Failed to DM {uid}. The user must have started the bot at least once."
        )


@whitelist_only
async def config_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /config - Display bot configuration and statistics
    """
    me = await context.bot.get_me()
    here = update.effective_chat

    with closing(db()) as con:
        gcount = con.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
        acount = con.execute("SELECT COUNT(*) FROM allowed").fetchone()[0]
        mcount = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        servers_seen = con.execute(
            "SELECT COUNT(DISTINCT chat_id) FROM messages WHERE chat_type!='private'"
        ).fetchone()[0]
        users_seen = con.execute(
            "SELECT COUNT(DISTINCT user_id) FROM messages WHERE user_id IS NOT NULL"
        ).fetchone()[0]
        facts_n = con.execute("SELECT COUNT(*) FROM gleanings").fetchone()[0]
        emb_n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        nodes_n = con.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        edges_n = con.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        prof_idx = con.execute("SELECT COUNT(*) FROM user_profiles_idx").fetchone()[0]

    meta = settings_get("AUTO_SYS_META", "")
    lines = [
        f"<b>Bot</b>: @{me.username} (id={me.id})",
        f"<b>DB</b>: {bot_server.DB_PATH} — groups:{gcount}, allowed:{acount}, msgs:{mcount}, facts:{facts_n}, embeds:{emb_n}, profiles_idx:{prof_idx}",
        f"<b>KG</b>: nodes:{nodes_n}, edges:{edges_n}",
        f"<b>Seen</b>: servers={servers_seen}, users={users_seen}",
        f"<b>Admin whitelist</b>: {', '.join(str(i) for i in sorted(bot_server.ADMIN_WHITELIST)) or '(empty)'}",
        f"<b>Ollama</b>: url={bot_server.OLLAMA_URL} model={bot_server.OLLAMA_MODEL or '(unset)'} embed={bot_server.OLLAMA_EMBED_MODEL or '(unset)'}",
        f"<b>Auto reply</b>: {bot_server.AUTO_REPLY_MODE}",
        f"<b>Auto System Prompt</b>: {'enabled' if bot_server.AUTO_SYS_ENABLED else 'disabled'} every {bot_server.AUTO_SYS_INTERVAL_HOURS}h",
        f"<b>Profiles</b>: {'enabled' if bot_server.PROFILES_ENABLED else 'disabled'}, refresh ~{bot_server.PROFILE_REFRESH_MINUTES}m",
    ]
    if meta:
        lines.append(f"<b>Prompt meta</b>: {meta}")
    if bot_server.PRIMARY_CHAT_ID:
        try:
            pcid = int(bot_server.PRIMARY_CHAT_ID)
            pchat = await context.bot.get_chat(pcid)
            lines.append(f"<b>Primary</b>: {pchat.title} (id={pcid})")
        except Exception:
            lines.append(f"<b>Primary</b>: (invalid) {bot_server.PRIMARY_CHAT_ID}")
    if here and here.type in ("group", "supergroup"):
        lines.append(f"<b>Here</b>: {here.title} (id={here.id})")

    await update.effective_message.reply_html("\n".join(lines), disable_web_page_preview=True)


@whitelist_only
async def system_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /system - View or manually set the system prompt
    Usage: /system [new prompt text...]
    """
    msg = update.effective_message
    if not context.args:
        preview = (bot_server.SYSTEM_PROMPT[:500] + "…") if len(bot_server.SYSTEM_PROMPT) > 500 else bot_server.SYSTEM_PROMPT
        meta = settings_get("AUTO_SYS_META", "")
        return await msg.reply_text(
            f"Current SYSTEM_PROMPT ({len(bot_server.SYSTEM_PROMPT)} chars):\n{preview or '(empty)'}\n\nMeta: {meta or '(none)'}"
        )

    bot_server.SYSTEM_PROMPT = " ".join(context.args).strip()
    _write_env_var("SYSTEM_PROMPT", bot_server.SYSTEM_PROMPT.replace("\n", "\\n"))
    await settings_set_async("AUTO_SYS_META", f"manual-set at {datetime.now().isoformat(timespec='seconds')}")
    await msg.reply_text("✅ SYSTEM_PROMPT updated.")


@whitelist_only
async def autosystem_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /autosystem - Regenerate system prompt automatically using AI
    Usage: /autosystem [why]  (include 'why' to see reasoning in metadata)
    """
    chat_id = update.effective_chat.id if update.effective_chat else None
    show_why = (context.args and any(a.lower() == "why" for a in context.args))

    async def _work():
        p = await bot_server.auto_generate_system_prompt(context, include_digest_in_meta=show_why)
        if not p:
            return None
        await bot_server.set_system_prompt_and_persist(p)
        return p

    if chat_id is not None:
        prompt = await bot_server.with_typing(context, chat_id, _work())
    else:
        prompt = await _work()

    if not prompt:
        return await update.effective_message.reply_text(
            "❌ Auto-prompt generation failed (Ollama not responding)."
        )

    prev = (prompt[:500] + "…") if len(prompt) > 500 else prompt
    await update.effective_message.reply_text(
        f"✅ System prompt regenerated ({len(prompt)} chars).\n\nPreview:\n{prev}"
    )


@whitelist_only
async def inspect_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /inspect - Inspect Telegram links, usernames, or IDs
    Usage: /inspect <link|@username|id>
    """
    if not context.args:
        return await update.effective_message.reply_text("Usage: /inspect <link|@username|id>")
    return await do_inspect(update, context, " ".join(context.args))


# Inspector patterns
INSPECT_PATTERNS = [
    re.compile(r'(?:https?://)?t\.me/c/(\d+)(?:/\d+)?', re.I),
    re.compile(r'(?:https?://)?t\.me/(?:joinchat/|\+)([A-Za-z0-9_-]+)', re.I),
    re.compile(r'(?:https?://)?t\.me/([A-Za-z0-9_]{5,})', re.I),
    re.compile(r'tg://user\?id=(\d+)', re.I),
    re.compile(r'^@([A-Za-z0-9_]{5,})$'),
    re.compile(r'^(?:-?\d{7,})$'),
]


async def do_inspect(update: Update, context: ContextTypes.DEFAULT_TYPE, target: str):
    """Perform inspection of Telegram entity"""
    target = target.strip()
    msg = update.effective_message
    bot = context.bot

    # Try /c/ link pattern
    m = INSPECT_PATTERNS[0].search(target)
    if m:
        try:
            cid = int(m.group(1))
            chat_id = int(f"-100{cid}")
            chat = await bot.get_chat(chat_id)
            return await msg.reply_html(
                f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\n"
                f"username: @{getattr(chat, 'username', '') or '(none)'}"
            )
        except Exception:
            return await msg.reply_text("Couldn't resolve that private /c/ link (bot must be a member).")

    # Try invite link pattern
    m = INSPECT_PATTERNS[1].search(target)
    if m:
        return await msg.reply_text("Invite links can't be resolved unless the bot joins that chat.")

    # Try t.me/username pattern
    m = INSPECT_PATTERNS[2].search(target)
    if m:
        uname = m.group(1)
        try:
            chat = await bot.get_chat(f"@{uname}")
            return await msg.reply_html(
                f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\n"
                f"username: @{getattr(chat, 'username', '') or '(none)'}"
            )
        except Exception:
            uid = resolve_user_by_username(None, uname)
            if uid:
                return await msg.reply_html(f"<b>User</b>\nid: <code>{uid}</code>\nusername: @{uname}")
            return await msg.reply_text("Couldn't resolve that username.")

    # Try tg://user?id= pattern
    m = INSPECT_PATTERNS[3].search(target)
    if m:
        uid = int(m.group(1))
        return await msg.reply_html(f"<b>User</b>\nid: <code>{uid}</code>")

    # Try @username pattern
    m = INSPECT_PATTERNS[4].search(target)
    if m:
        uname = m.group(1)
        uid = resolve_user_by_username(None, uname)
        if uid:
            return await msg.reply_html(f"<b>User</b>\nid: <code>{uid}</code>\nusername: @{uname}")
        try:
            chat = await bot.get_chat(f"@{uname}")
            return await msg.reply_html(
                f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\n"
                f"username: @{getattr(chat, 'username', '') or '(none)'}"
            )
        except Exception:
            return await msg.reply_text("Couldn't resolve that handle.")

    # Try numeric ID pattern
    m = INSPECT_PATTERNS[5].search(target)
    if m:
        try:
            ident = int(m.group(0))
            try:
                chat = await bot.get_chat(ident)
                return await msg.reply_html(
                    f"<b>Chat</b>\nid: <code>{chat.id}</code>\ntype: {chat.type}\ntitle: {chat.title or ''}\n"
                    f"username: @{getattr(chat, 'username', '') or '(none)'}"
                )
            except Exception:
                return await msg.reply_html(f"<b>User</b>\nid: <code>{ident}</code>")
        except Exception:
            pass

    return await msg.reply_text("Send a t.me / tg:// link, @username, or a numeric id to inspect.")


async def setadmin_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /setadmin - Request admin privileges
    First user bootstraps, subsequent requests require approval
    """
    requester = update.effective_user
    if requester is None:
        return await update.effective_message.reply_text("No user found.")

    rid = requester.id
    if not bot_server.ADMIN_WHITELIST:
        add_admin_ids([rid])
        return await update.effective_message.reply_text("✅ You are now an admin (bootstrap).")

    if rid in bot_server.ADMIN_WHITELIST:
        return await update.effective_message.reply_text("You are already an admin.")

    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"adminreq:approve:{rid}"),
            InlineKeyboardButton("✖️ Deny", callback_data=f"adminreq:deny:{rid}")
        ]
    ])

    name = " ".join(p for p in ((requester.first_name or ""), (requester.last_name or "")) if p) or (
        requester.username or f"id {rid}")
    handle = f"@{requester.username}" if requester.username else f"id {rid}"
    text = f"🔐 Admin request received.\n\nRequester: {name} ({handle})\nUser ID: <code>{rid}</code>\n\nApprove or deny this admin request."

    sent = 0
    for aid in list(bot_server.ADMIN_WHITELIST):
        try:
            await context.bot.send_message(aid, text, parse_mode="HTML", reply_markup=kb)
            sent += 1
        except Exception:
            pass

    if sent == 0:
        return await update.effective_message.reply_text(
            "No admins reachable. Ask an admin to DM me once, then try again."
        )
    return await update.effective_message.reply_text(f"Request sent to {sent} admin(s).")


async def adminreq_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin request approval/denial callbacks"""
    q = update.callback_query
    await q.answer()
    m = re.match(r"^adminreq:(approve|deny):(\d+)$", q.data or "")
    if not m:
        return

    action, uid_str = m.group(1), m.group(2)
    actor = q.from_user
    if actor.id not in bot_server.ADMIN_WHITELIST:
        return await q.answer("Only admins can do this.", show_alert=True)

    uid = int(uid_str)
    if action == "approve":
        if uid in bot_server.ADMIN_WHITELIST:
            try:
                await q.edit_message_text(f"Already admin: <code>{uid}</code>", parse_mode="HTML")
            except Exception:
                pass
            return
        add_admin_ids([uid])
        try:
            await context.bot.send_message(uid, "✅ Your admin request has been approved.")
        except Exception:
            pass
        try:
            await q.edit_message_text(f"✅ Approved admin: <code>{uid}</code>", parse_mode="HTML")
        except Exception:
            pass
    else:
        try:
            await context.bot.send_message(uid, "❌ Your admin request was denied.")
        except Exception:
            pass
        try:
            await q.edit_message_text(f"❌ Denied admin request: <code>{uid}</code>", parse_mode="HTML")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

__all__ = [
    'users_handler',
    'message_handler',
    'config_handler',
    'system_handler',
    'autosystem_handler',
    'inspect_handler',
    'setadmin_handler',
    'adminreq_callback_handler',
]
