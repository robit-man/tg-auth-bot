#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message Handlers - Handle text messages and errors

Note: The on_text handler is complex and tightly integrated with many systems
(memory, KG, AI, embeddings, profiles, etc.). For now, we re-export it from
bot_server. Future refactoring should break it into smaller, testable pieces.
"""

from telegram import Update
from telegram.ext import ContextTypes

# Import from bot_server temporarily
# TODO: Refactor on_text into smaller, composable functions
import bot_server


# ──────────────────────────────────────────────────────────────
# Message Handlers
# ──────────────────────────────────────────────────────────────

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle all text messages (DMs and groups)

    This handler orchestrates:
    - Message mirroring to database shards
    - Knowledge graph ingestion
    - Embedding generation
    - Fact gleaning
    - Profile updates
    - Relationship graph updates
    - Memory recall and storage
    - AI response generation (DMs always, groups on mention/reply/smart detection)
    - Agentic action execution
    - System prompt reflection
    """
    # For now, delegate to the complex handler in bot_server
    # TODO: Break this into smaller, testable components:
    #   - message_ingestion(update) -> meta
    #   - relationship_tracking(update, meta)
    #   - should_respond(update, meta) -> bool, reason
    #   - generate_response(update, meta, context)
    #   - post_response_tasks(update, response, meta)
    await bot_server.on_text(update, context)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle errors that occur during message processing
    """
    await bot_server.on_error(update, context)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

__all__ = [
    'text_message_handler',
    'error_handler',
]
