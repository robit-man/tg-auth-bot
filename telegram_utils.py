#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Utilities - Decorators, permissions, and helper functions for Telegram bot
"""

import base64
from functools import wraps
from telegram import Update, ChatPermissions
from telegram.ext import ContextTypes
from typing import Callable


# Decorator for admin-only commands
def admin_only(fn: Callable):
    """Decorator to restrict commands to admin whitelist"""
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        # Import here to avoid circular dependency
        from bot_server import ADMIN_WHITELIST
        if user.id not in ADMIN_WHITELIST:
            await update.effective_message.reply_text("⛔ Admin-only command.")
            return
        return await fn(update, context)
    return wrapper


def whitelist_only(fn: Callable):
    """Decorator to restrict commands to whitelisted users"""
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        # Import here to avoid circular dependency
        from bot_server import ADMIN_WHITELIST
        if user.id not in ADMIN_WHITELIST:
            await update.effective_message.reply_text("⛔ This command requires whitelist access.")
            return
        return await fn(update, context)
    return wrapper


async def make_group_readonly(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Set group to read-only (no posting)"""
    perms = ChatPermissions(
        can_send_messages=False, can_send_audios=False, can_send_documents=False,
        can_send_photos=False, can_send_videos=False, can_send_video_notes=False,
        can_send_voice_notes=False, can_send_polls=False, can_send_other_messages=False,
        can_add_web_page_previews=False, can_change_info=False, can_invite_users=False, can_pin_messages=False
    )
    await context.bot.set_chat_permissions(chat_id, perms)


async def make_group_writable(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Restore full group permissions"""
    perms = ChatPermissions(
        can_send_messages=True, can_send_audios=True, can_send_documents=True, can_send_photos=True, can_send_videos=True,
        can_send_video_notes=True, can_send_voice_notes=True, can_send_polls=True, can_send_other_messages=True,
        can_add_web_page_previews=True, can_change_info=False, can_invite_users=True, can_pin_messages=False,
    )
    await context.bot.set_chat_permissions(chat_id, perms)


async def build_start_link(bot_username: str, chat_id: int, title: str | None) -> str:
    """Build a deep-link start URL for a group"""
    payload = base64.urlsafe_b64encode(f"{chat_id}".encode()).decode().rstrip("=")
    return f"https://t.me/{bot_username}?start={payload}"


def format_user_display(user_id: int, username: str = None, first_name: str = None, last_name: str = None) -> str:
    """Format user info for display"""
    name_parts = [p for p in [first_name, last_name] if p]
    name = " ".join(name_parts) if name_parts else None

    if name and username:
        return f"{name} (@{username})"
    elif name:
        return name
    elif username:
        return f"@{username}"
    else:
        return f"User #{user_id}"
