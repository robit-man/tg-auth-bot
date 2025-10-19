#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gating Handlers - Commands for setting up and managing group gating
"""

import base64
import textwrap
import sqlite3
from contextlib import closing
from typing import Optional
from collections import defaultdict
from telegram import Update, ChatPermissions, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes

# Import decorators and utilities
from telegram_utils import admin_only, whitelist_only, build_start_link, make_group_readonly


# ──────────────────────────────────────────────────────────────
# Configuration and Globals
# ──────────────────────────────────────────────────────────────
# These will be imported from bot_server or config module
# For now, import from bot_server to maintain functionality
import bot_server

# Track welcome messages for cleanup
WELCOME_MESSAGES: dict[tuple[int, int], set[int]] = defaultdict(set)
WELCOME_JOBS: dict[tuple[int, int, int], object] = {}


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


def token_to_chat_id(token: str) -> Optional[int]:
    """Convert token to chat_id by looking up in groups table"""
    with closing(db()) as con:
        r = con.execute("SELECT chat_id FROM groups WHERE token=?", (token,)).fetchone()
        return r[0] if r else None


async def mark_allowed_async(chat_id: int, user) -> None:
    """Mark user as allowed in the database"""
    def _work(con: sqlite3.Connection):
        con.execute("""INSERT OR REPLACE INTO allowed(chat_id,user_id,username,first_name,last_name)
                       VALUES(?,?,?,?,?)""",
                    (chat_id, user.id, (user.username or ""), (user.first_name or ""), (user.last_name or "")))
        con.commit()
    await bot_server.DBW.run(bot_server.DB_PATH, _work)


async def _delete_single_welcome_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    message_id: int,
    executing_job=None,
):
    """Delete a single welcome message"""
    job_key = (chat_id, user_id, message_id)
    job = WELCOME_JOBS.pop(job_key, None)
    if executing_job and job and job != executing_job:
        return  # Another job is handling this
    if executing_job is None and job:
        job.schedule_removal()
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass
    key = (chat_id, user_id)
    if key in WELCOME_MESSAGES:
        WELCOME_MESSAGES[key].discard(message_id)
        if not WELCOME_MESSAGES[key]:
            del WELCOME_MESSAGES[key]


async def cleanup_welcome_messages(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int):
    """Clean up all welcome messages for a user"""
    key = (chat_id, user_id)
    message_ids = list(WELCOME_MESSAGES.get(key, set()))
    for mid in message_ids:
        await _delete_single_welcome_message(context, chat_id, user_id, mid)


async def allow_user(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user) -> None:
    """Grant user permissions to post in the group"""
    perms = ChatPermissions(
        can_send_messages=True, can_send_audios=True, can_send_documents=True,
        can_send_photos=True, can_send_videos=True, can_send_video_notes=True,
        can_send_voice_notes=True, can_send_polls=True, can_send_other_messages=True,
        can_add_web_page_previews=True,
    )
    await context.bot.restrict_chat_member(chat_id=chat_id, user_id=user.id, permissions=perms,
                                           use_independent_chat_permissions=True)
    await mark_allowed_async(chat_id, user)


# ──────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────

@admin_only
async def setup_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /setup - Initialize group gating
    Sets the group to read-only and provides a start link for new members
    """
    chat = update.effective_chat
    await make_group_readonly(context, chat.id)
    bot_username = (await context.bot.get_me()).username
    start_link = await build_start_link(bot_username, chat.id, chat.title)

    # Set as primary chat if not already set
    if not bot_server.PRIMARY_CHAT_ID:
        bot_server.PRIMARY_CHAT_ID = str(chat.id)
        _write_env_var("PRIMARY_CHAT_ID", bot_server.PRIMARY_CHAT_ID)

    msg = textwrap.dedent(f"""
        ✅ Gating enabled for *{chat.title}*.

        1) Group default is now read-only.
        2) Pin this "Start" link so newcomers can unlock:
           {start_link}

        Primary server chat id: `{bot_server.PRIMARY_CHAT_ID}`
    """).strip()
    await update.effective_message.reply_text(msg, disable_web_page_preview=True, parse_mode="Markdown")


@admin_only
async def ungate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /ungate - Disable gating and restore full group permissions
    """
    chat = update.effective_chat
    perms = ChatPermissions(
        can_send_messages=True, can_send_audios=True, can_send_documents=True,
        can_send_photos=True, can_send_videos=True,
        can_send_video_notes=True, can_send_voice_notes=True, can_send_polls=True,
        can_send_other_messages=True,
        can_add_web_page_previews=True, can_change_info=False, can_invite_users=True,
        can_pin_messages=False,
    )
    await context.bot.set_chat_permissions(chat.id, perms)
    await update.effective_message.reply_text("🚪 Gating disabled and defaults restored.")


@admin_only
async def link_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /link - Get the start link for the current group
    """
    chat = update.effective_chat
    bot_username = (await context.bot.get_me()).username
    start_link = await build_start_link(bot_username, chat.id, chat.title)
    await update.effective_message.reply_text(
        f"🔗 Start link for this group:\n{start_link}",
        disable_web_page_preview=True
    )


async def linkprimary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /linkprimary - Get the start link for the primary group
    """
    if not bot_server.PRIMARY_CHAT_ID:
        return await update.effective_message.reply_text(
            "PRIMARY_CHAT_ID is not set. Run /setprimary in your target group."
        )
    try:
        chat_id = int(bot_server.PRIMARY_CHAT_ID)
        chat = await context.bot.get_chat(chat_id)
        bot_username = (await context.bot.get_me()).username
        start_link = await build_start_link(bot_username, chat.id, chat.title)
        await update.effective_message.reply_text(
            f"🔗 Primary group: <b>{chat.title}</b>\n{start_link}",
            parse_mode="HTML",
            disable_web_page_preview=True
        )
    except Exception:
        await update.effective_message.reply_text(
            f"Could not fetch PRIMARY_CHAT_ID ({bot_server.PRIMARY_CHAT_ID}). Try /setprimary in the group."
        )


@admin_only
async def setprimary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /setprimary - Set the current group as the primary group
    """
    chat = update.effective_chat
    _write_env_var("PRIMARY_CHAT_ID", str(chat.id))
    bot_server.PRIMARY_CHAT_ID = str(chat.id)
    await update.effective_message.reply_text(
        f"✅ Primary server set to: {chat.title} ({chat.id}). Saved to .env as PRIMARY_CHAT_ID."
    )


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start - Handle DM start command (for gating unlock flow)
    """
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
        return await update.message.reply_text(
            "That link is invalid or expired. Ask an admin for a fresh one."
        )

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ I agree — unlock me", callback_data=f"agree:{chat_id}")]
    ])
    await update.message.reply_text(
        "Before we unlock you, please confirm you've read the group rules.",
        reply_markup=kb
    )


async def agree_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle the agreement callback when user agrees to group rules
    """
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if not data.startswith("agree:"):
        return

    chat_id = int(data.split("agree:", 1)[1])
    user = q.from_user

    # Check if user is a member of the group
    try:
        member = await context.bot.get_chat_member(chat_id, user.id)
        if member.status in ("left", "kicked"):
            return await q.edit_message_text(
                "Join the group first, then tap the Start link again."
            )
    except Exception:
        return await q.edit_message_text(
            "I couldn't check your membership. Ask an admin to verify the setup."
        )

    # Grant permissions
    try:
        await allow_user(context, chat_id, user)
    except Exception:
        return await q.edit_message_text(
            "Couldn't unlock you (bot needs admin rights with 'Manage Members')."
        )

    # Clean up welcome messages
    await cleanup_welcome_messages(context, chat_id, user.id)

    await q.edit_message_text("✅ You're unlocked. You can now receive messages from me.")

    # Announce in group
    try:
        await context.bot.send_message(
            chat_id,
            f"👋 {user.mention_html()} has been onboarded and unlocked.",
            parse_mode="HTML"
        )
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

__all__ = [
    'setup_handler',
    'ungate_handler',
    'link_handler',
    'linkprimary_handler',
    'setprimary_handler',
    'start_handler',
    'agree_callback_handler',
    'WELCOME_MESSAGES',
    'WELCOME_JOBS',
]
