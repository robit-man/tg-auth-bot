#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Handlers - Public commands for users and groups
"""

import re
import time
import sqlite3
import textwrap
from contextlib import closing
from typing import Dict, List, Optional
from telegram import Update
from telegram.ext import ContextTypes

# Import decorators
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


def kg_top_entities(chat_id: int, thread_id: int, limit: int = 12) -> List[str]:
    """Get top entities from knowledge graph for a chat/thread"""
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


def kg_top_relations(chat_id: int, thread_id: int, limit: int = 12) -> List[str]:
    """Get top relations from knowledge graph for a chat/thread"""
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


def command_catalog() -> Dict[str, List[tuple]]:
    """Return catalog of available commands"""
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
    """Format command list for display to user"""
    cat = command_catalog()
    lines = ["Available commands:", "\nPublic:"]
    for cmd, help_ in cat["public"]:
        lines.append(f"  {cmd} — {help_}")
    if is_admin:
        lines.append("\nAdmin:")
        for cmd, help_ in cat["admin"]:
            lines.append(f"  {cmd} — {help_}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────

async def commands_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /commands - Show available commands tailored to the user
    """
    uid = update.effective_user.id if update.effective_user else None
    is_admin = uid in bot_server.ADMIN_WHITELIST if uid else False
    await update.effective_message.reply_text(format_commands_for_user(is_admin))


async def topic_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /topic - Show top entities and relations for the current channel/thread
    """
    chat = update.effective_chat
    m = update.effective_message
    if not chat or chat.type not in ("group", "supergroup", "channel"):
        return await m.reply_text("Run this in a group/channel.")

    thread_id = getattr(m, "message_thread_id", None) or 0
    ents = kg_top_entities(chat.id, thread_id, limit=10)
    rels = kg_top_relations(chat.id, thread_id, limit=10)

    if not ents and not rels:
        return await m.reply_text("No graph signals here yet.")

    text = "<b>Channel snapshot</b>\n"
    if ents:
        text += "Top entities:\n• " + "\n• ".join(ents) + "\n"
    if rels:
        text += "\nTop relations:\n• " + "\n• ".join(rels)
    await m.reply_html(text)


@whitelist_only
async def graph_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /graph - Show knowledge graph entities and relations
    Usage: /graph [here|server] [N]
    """
    args = context.args or []
    scope = "here"
    topn = 12

    for a in args:
        if a in ("here", "server"):
            scope = a
        elif re.fullmatch(r"\d+", a):
            topn = max(3, min(40, int(a)))

    chat = update.effective_chat
    if not chat or chat.type not in ("group", "supergroup", "channel"):
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
        ents = [f"{lab}" for (lab, _w) in ents_rows]
        hdr = f"<b>Graph server-wide</b> (chat {chat.id})"

    if not ents and not rels:
        return await update.effective_message.reply_text("No graph yet.")

    text = hdr + "\n"
    if ents:
        text += "Top entities:\n• " + "\n• ".join(ents) + "\n"
    if rels:
        text += "\nTop relations:\n• " + "\n".join(rels)
    await update.effective_message.reply_html(text)


async def profile_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /profile - Manage user interaction profile
    Usage: /profile [show|now|erase|optout on/off|help]
    """
    m = update.effective_message
    chat = update.effective_chat
    user = update.effective_user

    if chat.type != "private":
        return await m.reply_text("DM me to manage your interaction profile: open chat and send /profile.")

    args = [a.lower() for a in (context.args or [])]
    if not args or args[0] in ("help", "?"):
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
        path = str(bot_server.user_db_path(user.id))
        with closing(sqlite3.connect(path)) as con:
            bot_server.ensure_schema_on(con)
            row = con.execute(
                "SELECT last_updated,msg_count,avg_len,question_ratio,emoji_ratio,link_ratio,code_ratio,positivity,style_notes FROM profile WHERE user_id=?",
                (user.id,)
            ).fetchone()
        if not row:
            return await m.reply_text("No profile yet. Use /profile now to build it.")
        last_updated, msg_count, avg_len, qr, er, lr, cr, pos, notes = row
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_updated or 0))
        txt = textwrap.dedent(f"""
            Profile last updated: {ts}
            Messages analyzed: {msg_count}
            Avg chars: {avg_len:.0f}
            ?-ratio: {qr:.2f} | emoji-ratio: {er:.2f} | link-ratio: {lr:.2f} | code-ratio: {cr:.2f}
            Positivity: {pos:.2f}

            {notes or '(no style notes)'}
        """).strip()
        return await m.reply_text(txt)

    if args[0] == "now":
        built = await bot_server.maybe_build_profile(user.id, context, force=True)
        if not built:
            return await m.reply_text("Profile refresh failed (check AI availability).")
        return await m.reply_text("✅ Profile refreshed. Use /profile show to view it.")

    if args[0] == "erase":
        path = str(bot_server.user_db_path(user.id))
        with closing(sqlite3.connect(path)) as con:
            bot_server.ensure_schema_on(con)
            con.execute("DELETE FROM profile WHERE user_id=?", (user.id,))
            con.commit()
        return await m.reply_text("✅ Your profile has been erased from local storage.")

    if args[0] == "optout":
        if len(args) < 2:
            return await m.reply_text("Usage: /profile optout on|off")
        val = args[1] in ("on", "true", "1", "yes")
        path = str(bot_server.user_db_path(user.id))
        with closing(sqlite3.connect(path)) as con:
            bot_server.ensure_schema_on(con)
            con.execute(
                "INSERT INTO profile(user_id,opted_out) VALUES(?,?) ON CONFLICT(user_id) DO UPDATE SET opted_out=excluded.opted_out",
                (user.id, 1 if val else 0)
            )
            con.commit()
        status = "enabled" if val else "disabled"
        return await m.reply_text(f"✅ Profile opt-out {status}.")

    return await m.reply_text("Unknown subcommand. Use /profile help for usage.")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

__all__ = [
    'commands_handler',
    'topic_handler',
    'graph_handler',
    'profile_handler',
]
