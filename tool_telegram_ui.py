#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tool_telegram_ui.py — Telegram UI for tool confirmation, progress, and feedback

Provides:
- Tool confirmation buttons
- Progress updates via message editing
- Rating buttons (0-5) for feedback
"""

import asyncio
from typing import Any, Optional
import html
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import ContextTypes

from tool_executor_bridge import (
    ToolDecision,
    ToolDecisionType,
    ToolExecutionResult,
    RealToolExecutor,
)


class ToolTelegramUI:
    """Telegram UI for tool interactions"""

    @staticmethod
    def create_tool_confirmation_keyboard(
        tool_decision: ToolDecision,
        callback_prefix: str = "tool_confirm",
    ) -> InlineKeyboardMarkup:
        """
        Create confirmation keyboard for tool execution.

        Shows:
        [✅ Execute Tool] [❌ Cancel] [📝 Direct Reply]
        """
        tool_display = tool_decision.tool_name or "Unknown"

        buttons = [
            [
                InlineKeyboardButton(
                    f"✅ Execute {tool_display}",
                    callback_data=f"{callback_prefix}:execute:{tool_decision.tool_name}"
                ),
                InlineKeyboardButton(
                    "❌ Cancel",
                    callback_data=f"{callback_prefix}:cancel"
                ),
            ],
            [
                InlineKeyboardButton(
                    "📝 Direct Reply Instead",
                    callback_data=f"{callback_prefix}:direct"
                ),
            ]
        ]

        return InlineKeyboardMarkup(buttons)

    @staticmethod
    def create_rating_keyboard(callback_prefix: str = "tool_rating") -> InlineKeyboardMarkup:
        """
        Create rating keyboard (0-5 stars).

        Shows:
        [0] [1] [2] [3] [4] [5]
        """
        buttons = [
            [
                InlineKeyboardButton(
                    f"{'⭐' * i if i > 0 else '0'}",
                    callback_data=f"{callback_prefix}:{i}"
                )
                for i in range(6)
            ]
        ]

        return InlineKeyboardMarkup(buttons)

    @staticmethod
    async def send_tool_confirmation(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        tool_decision: ToolDecision,
    ) -> Message:
        """
        Send message asking user to confirm tool execution.

        Returns:
            Sent message (for later editing)
        """
        # Format tool info
        tool_name = tool_decision.tool_name or "Unknown"
        args_preview = ", ".join(
            f"{k}={repr(v)[:50]}" for k, v in tool_decision.tool_args.items()
        ) or "(no arguments)"
        reasoning_text = tool_decision.reasoning or "(no reasoning provided)"

        message_text = (
            "🤖 <b>Tool Decision</b>\n\n"
            f"<b>Tool:</b> <code>{html.escape(tool_name)}</code>\n"
            f"<b>Arguments:</b> {html.escape(args_preview) or 'None'}\n"
            f"<b>Confidence:</b> {html.escape(f'{tool_decision.confidence:.1%}')}\n"
            f"<b>Reasoning:</b> {html.escape(reasoning_text)}\n\n"
            "Execute this tool?"
        )

        keyboard = ToolTelegramUI.create_tool_confirmation_keyboard(tool_decision)

        return await update.effective_message.reply_text(
            message_text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )

    @staticmethod
    async def update_message_progress(
        message: Message,
        stage: str,
        details: str = "",
    ) -> None:
        """
        Update message to show progress.

        Stages:
        - "executing"
        - "completed"
        - "failed"
        """
        stage_icons = {
            "executing": "⏳",
            "completed": "✅",
            "failed": "❌",
        }

        icon = stage_icons.get(stage, "🔄")

        try:
            await message.edit_text(
                f"{icon} <b>{html.escape(stage.title())}</b>\n\n{html.escape(details or '')}",
                parse_mode="HTML",
            )
        except Exception:
            # Ignore edit failures (message might be too old)
            pass

    @staticmethod
    async def show_tool_result_with_rating(
        message: Message,
        tool_result: ToolExecutionResult,
        original_question: str = "",
    ) -> None:
        """
        Edit message to show tool result and add rating buttons.
        """
        if tool_result.success:
            result_payload = tool_result.formatted_output or tool_result.output
            result_body = str(result_payload)
            if len(result_body) > 1500:
                result_body = result_body[:1497] + "..."
            result_body_escaped = html.escape(result_body)
            result_text = (
                "✅ <b>Tool Executed Successfully</b>\n\n"
                f"<b>Tool:</b> <code>{html.escape(tool_result.tool_name)}</code>\n"
                f"<b>Time:</b> {html.escape(f'{tool_result.execution_time:.2f}s')}\n\n"
                f"<b>Result:</b>\n<pre>{result_body_escaped}</pre>\n\n"
                "Rate this response:"
            )
        else:
            error_text = tool_result.error or "Unknown error"
            if len(error_text) > 1500:
                error_text = error_text[:1497] + "..."
            result_text = (
                "❌ <b>Tool Execution Failed</b>\n\n"
                f"<b>Tool:</b> <code>{html.escape(tool_result.tool_name)}</code>\n"
                f"<b>Time:</b> {html.escape(f'{tool_result.execution_time:.2f}s')}\n"
                f"<b>Error:</b> <pre>{html.escape(error_text)}</pre>\n\n"
                "Rate this response:"
            )

        rating_keyboard = ToolTelegramUI.create_rating_keyboard()

        try:
            await message.edit_text(
                result_text,
                reply_markup=rating_keyboard,
                parse_mode="HTML",
            )
        except Exception as e:
            # If can't edit (message too long), send new message
            try:
                await message.reply_text(
                    result_text,
                    reply_markup=rating_keyboard,
                    parse_mode="Markdown",
                )
            except Exception:
                pass

    @staticmethod
    async def execute_tool_with_progress_ui(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        tool_decision: ToolDecision,
        progress_message: Optional[Message] = None,
    ) -> ToolExecutionResult:
        """
        Execute tool with live progress updates in Telegram.

        Args:
            update: Telegram update
            context: Telegram context
            tool_decision: Tool to execute
            progress_message: Optional message to edit (if None, creates new)

        Returns:
            ToolExecutionResult
        """
        # Create or use provided progress message
        if progress_message is None:
            progress_message = await update.effective_message.reply_text(
                "⏳ Preparing to execute tool...",
            )

        # Update: Executing
        await ToolTelegramUI.update_message_progress(
            progress_message,
            "executing",
            f"Running `{tool_decision.tool_name}`...",
        )

        # Create progress callback to stream updates to Telegram
        progress_log = []
        last_update_time = [0.0]  # Use list for mutable reference

        def sync_progress_callback(message: str):
            """Synchronous callback that accumulates progress messages"""
            import time
            progress_log.append(message)

            # Rate limit updates to avoid Telegram API limits (max 1 per second)
            current_time = time.time()
            if current_time - last_update_time[0] >= 1.0:
                last_update_time[0] = current_time

                # Update message with accumulated progress
                full_progress = "\n".join(progress_log[-20:])  # Last 20 messages
                try:
                    # Use asyncio to schedule the update
                    import asyncio
                    loop = asyncio.get_event_loop()
                    loop.create_task(
                        progress_message.edit_text(
                            f"⏳ **{tool_decision.tool_name}**\n\n{full_progress}",
                            parse_mode="Markdown"
                        )
                    )
                except Exception:
                    pass  # Ignore edit failures

        # Execute the tool with progress callback
        result = await RealToolExecutor.execute_tool(
            tool_decision.tool_name,
            tool_decision.tool_args,
            progress_callback=sync_progress_callback,
        )

        # Update: Result
        if result.success:
            await ToolTelegramUI.update_message_progress(
                progress_message,
                "completed",
                f"Tool completed in {result.execution_time:.2f}s",
            )
        else:
            await ToolTelegramUI.update_message_progress(
                progress_message,
                "failed",
                f"Error: {result.error}",
            )

        # Wait a moment so user sees the status
        await asyncio.sleep(0.5)

        # Show result with rating buttons
        await ToolTelegramUI.show_tool_result_with_rating(
            progress_message,
            result,
        )

        return result


# ──────────────────────────────────────────────────────────────
# Callback handlers for buttons
# ──────────────────────────────────────────────────────────────

async def handle_tool_confirmation_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle tool confirmation button presses.

    Callback format: tool_confirm:action:tool_name
    """
    query = update.callback_query
    await query.answer()

    callback_data = query.data
    parts = callback_data.split(":")

    if len(parts) < 2:
        return

    action = parts[1]

    if action == "execute":
        # User confirmed - execute the tool
        # Need to retrieve the tool decision from context
        # For now, just acknowledge
        await query.edit_message_text(
            "✅ Executing tool...",
        )

    elif action == "cancel":
        # User cancelled
        await query.edit_message_text(
            "❌ Tool execution cancelled.",
        )

    elif action == "direct":
        # User wants direct reply instead
        await query.edit_message_text(
            "📝 Generating direct reply instead...",
        )


async def handle_rating_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle rating button presses.

    Callback format: tool_rating:0-5
    """
    query = update.callback_query
    await query.answer()

    callback_data = query.data
    parts = callback_data.split(":")

    if len(parts) < 2:
        return

    try:
        rating = int(parts[1])
    except ValueError:
        return

    # Store rating (you can save this to database)
    print(f"[rating] User rated response: {rating}/5")

    # Update message to show rating was recorded
    current_text = query.message.text or ""

    # Remove rating keyboard
    try:
        await query.edit_message_text(
            current_text + f"\n\n⭐ **Rating recorded:** {rating}/5\nThank you for your feedback!",
            parse_mode="Markdown",
        )
    except Exception:
        await query.answer(f"Rating recorded: {rating}/5. Thank you!")
