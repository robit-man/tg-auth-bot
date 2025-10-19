#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Integration Example - How to integrate tool system into bot_server.py

This file shows the minimal code changes needed to add tool capabilities
to your Telegram bot for admin users.
"""

# ============================================================================
# STEP 1: Add imports to bot_server.py (top of file, after other imports)
# ============================================================================

# Tool integration imports
try:
    from tool_integration import (
        TOOLS_AVAILABLE,
        ToolInspector,
        ToolExecutor,
        DAGExecutor,
    )
    from ai_tool_bridge import (
        TOOL_INTEGRATION_AVAILABLE,
        EnhancedPromptBuilder,
        ToolExecutionCoordinator,
        ToolRequestParser,
    )
except ImportError:
    TOOLS_AVAILABLE = False
    TOOL_INTEGRATION_AVAILABLE = False
    print("[WARNING] Tool integration modules not found. Tool capabilities disabled.")


# ============================================================================
# STEP 2: Add configuration (near other config in bot_server.py)
# ============================================================================

# Tool integration configuration
TOOL_INTEGRATION_ENABLED = os.getenv("TOOL_INTEGRATION_ENABLED", "true").lower() == "true"
MAX_TOOLS_PER_RESPONSE = int(os.getenv("MAX_TOOLS_PER_RESPONSE", "5"))
AUTO_EXECUTE_TOOLS = os.getenv("AUTO_EXECUTE_TOOLS", "true").lower() == "true"
INCLUDE_TOOL_EXAMPLES = os.getenv("INCLUDE_TOOL_EXAMPLES", "true").lower() == "true"


# ============================================================================
# STEP 3: Create helper function for enhanced AI generation
# ============================================================================

async def generate_ai_response_with_tools(
    base_payload: dict,
    user_id: int,
    admin_whitelist: set
) -> str:
    """
    Generate AI response with tool context injection for admin users

    Args:
        base_payload: The base Ollama payload (model, messages, etc.)
        user_id: ID of the requesting user
        admin_whitelist: Set of admin user IDs

    Returns:
        AI response text (with tool execution results if applicable)
    """
    if not TOOL_INTEGRATION_AVAILABLE or not TOOL_INTEGRATION_ENABLED:
        # Tool integration disabled - use normal AI generation
        return await ai_generate_async(base_payload)

    # Inject tool context for admin users
    enhanced_payload = base_payload
    if user_id in admin_whitelist:
        enhanced_payload = EnhancedPromptBuilder.inject_tool_context(
            base_payload,
            user_id=user_id,
            admin_whitelist=admin_whitelist,
            include_examples=INCLUDE_TOOL_EXAMPLES
        )
        print(f"[tools] Injected tool context for admin user {user_id}")

    # Generate AI response
    ai_response = await ai_generate_async(enhanced_payload)

    # Execute tools if auto-execute is enabled and user is admin
    if AUTO_EXECUTE_TOOLS and user_id in admin_whitelist:
        coordinator = ToolExecutionCoordinator(user_id, admin_whitelist)
        modified_response, results = await coordinator.process_ai_response(
            ai_response,
            auto_execute=True,
            max_tools=MAX_TOOLS_PER_RESPONSE
        )

        if results:
            print(f"[tools] Executed {len(results)} tools for user {user_id}")
            for result in results:
                print(f"  - {result.tool_name}: {result.state.value}")

        return modified_response

    return ai_response


# ============================================================================
# STEP 4: Modify the on_text handler (or wherever AI responses are generated)
# ============================================================================

# FIND THIS CODE IN bot_server.py (around line 3377 for DMs, 3430 for groups):
#
#     payload = build_ai_prompt(text, user, chat.id, thread_id, thread_ctx, sim, kg_snap,
#                               internal_state=internal_state,
#                               context_bundle=context_bundle,
#                               complexity_meta=complexity_meta)
#     resp = await with_typing(context, chat.id, ai_generate_async(payload))
#     if resp:
#         await m.reply_text(resp)
#
# REPLACE WITH:
#
#     payload = build_ai_prompt(text, user, chat.id, thread_id, thread_ctx, sim, kg_snap,
#                               internal_state=internal_state,
#                               context_bundle=context_bundle,
#                               complexity_meta=complexity_meta)
#
#     # Use enhanced AI generation with tool support
#     resp = await with_typing(
#         context,
#         chat.id,
#         generate_ai_response_with_tools(payload, user.id, ADMIN_WHITELIST)
#     )
#
#     if resp:
#         await m.reply_text(resp)


# ============================================================================
# STEP 5: (Optional) Add /tools command for admins to list available tools
# ============================================================================

@whitelist_only
async def tools_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /tools - Show available tools for admin users
    Usage: /tools [category]
    """
    if not TOOL_INTEGRATION_AVAILABLE:
        return await update.effective_message.reply_text(
            "Tool integration not available (modules not installed)."
        )

    args = context.args or []
    category = args[0] if args else None

    if category:
        # Show tools in specific category
        tools = ToolInspector.get_tools_by_category(category)
        if not tools:
            return await update.effective_message.reply_text(
                f"No tools found in category '{category}'.\n\n"
                "Available categories: browser, web, filesystem, system, ai, general"
            )

        lines = [f"<b>{category.upper()} TOOLS</b>", "=" * 40, ""]
        for tool in tools:
            lines.append(f"<code>{tool.name}{tool.signature}</code>")
            doc_preview = tool.docstring.split("\n")[0][:100]
            lines.append(f"  {doc_preview}")
            lines.append("")

        await update.effective_message.reply_html("\n".join(lines))
    else:
        # Show tool summary
        tools = ToolInspector.get_all_tools()
        by_category = {}
        for tool in tools:
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)

        lines = [
            "<b>AVAILABLE TOOLS</b>",
            "=" * 40,
            f"Total: {len(tools)} tools",
            ""
        ]

        for cat in sorted(by_category.keys()):
            count = len(by_category[cat])
            lines.append(f"{cat}: {count} tools")

        lines.append("")
        lines.append("Use <code>/tools [category]</code> to see tools in a category")
        lines.append("")
        lines.append("Categories: " + ", ".join(sorted(by_category.keys())))

        await update.effective_message.reply_html("\n".join(lines))


# ============================================================================
# STEP 6: Register the /tools command (in main() function)
# ============================================================================

# Add this line where other command handlers are registered:
#     app.add_handler(CommandHandler("tools", tools_cmd))


# ============================================================================
# STEP 7: Update .env file with tool configuration
# ============================================================================

# Add these lines to your .env file:
#
# # Tool Integration
# TOOL_INTEGRATION_ENABLED=true
# MAX_TOOLS_PER_RESPONSE=5
# AUTO_EXECUTE_TOOLS=true
# INCLUDE_TOOL_EXAMPLES=true


# ============================================================================
# COMPLETE EXAMPLE: Modified on_text for DMs with tools
# ============================================================================

async def on_text_with_tools_example(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Example of how on_text should look with tool integration.
    This is a simplified version showing the key changes.
    """
    await ensure_me(context)
    m = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not m or not chat:
        return

    # ... [Mirror message, KG ingestion, etc. - unchanged] ...

    # DM: always reply
    if chat.type == "private":
        text = (m.text or "").strip()
        if not text:
            return

        thread_id = 0
        # ... [Relationship tracking - unchanged] ...

        thread_ctx = fetch_thread_context(chat.id, thread_id, limit=24)
        sim = await similar_context(text, chat.id, thread_id, top_k=8)
        kg_snap = {
            "ents_here": kg_top_entities(chat.id, thread_id, 10),
            "rels_here": kg_top_relations(chat.id, thread_id, 10)
        }

        if ai_available():
            internal_state = await recall_memories_async(text, chat, thread_id, user)
            context_bundle = assemble_context_bundle(chat, thread_id, user, text, thread_ctx, internal_state)
            complexity_meta = await assess_discussion_complexity(text, thread_ctx, internal_state, context_bundle, chat, user)

            # ... [Agentic actions - unchanged] ...

            # Build prompt
            payload = build_ai_prompt(
                text, user, chat.id, thread_id, thread_ctx, sim, kg_snap,
                internal_state=internal_state,
                context_bundle=context_bundle,
                complexity_meta=complexity_meta
            )

            # ===== KEY CHANGE: Use tool-enhanced AI generation =====
            resp = await with_typing(
                context,
                chat.id,
                generate_ai_response_with_tools(payload, user.id, ADMIN_WHITELIST)
            )
            # ======================================================

            if resp:
                await m.reply_text(resp)
                # ... [Memory update, affect analysis - unchanged] ...
            else:
                await m.reply_text("AI is unavailable right now.")
            return
        else:
            return await m.reply_text("AI is disabled.")

    # Groups/channels: [unchanged]
    # ...


# ============================================================================
# TESTING THE INTEGRATION
# ============================================================================

"""
Test Checklist:

1. As Admin User (in DM):
   - Send: "Search for Python 3.12 features"
   - Expect: AI uses Tools.search_internet() and shows results
   - Verify: Tool execution results appear in response

2. As Admin User (in DM):
   - Send: "Create a file called test.txt with 'Hello World'"
   - Expect: AI uses Tools.create_file()
   - Verify: File created in workspace

3. As Admin User:
   - Send: "/tools"
   - Expect: List of tool categories
   - Send: "/tools web"
   - Expect: List of web tools with signatures

4. As Non-Admin User:
   - Send: "Search for Python 3.12 features"
   - Expect: AI responds from knowledge base (no tools used)
   - Verify: No tool context in prompt, no execution

5. Check Logs:
   - Look for: "[tools] Injected tool context for admin user X"
   - Look for: "[tools] Executed N tools for user X"

6. Error Handling:
   - Send invalid tool request
   - Expect: Graceful error message
   - Verify: Bot doesn't crash
"""


# ============================================================================
# MINIMAL WORKING EXAMPLE (copy-paste ready)
# ============================================================================

"""
# At top of bot_server.py:
try:
    from ai_tool_bridge import EnhancedPromptBuilder, ToolExecutionCoordinator
    TOOL_INTEGRATION_AVAILABLE = True
except ImportError:
    TOOL_INTEGRATION_AVAILABLE = False

# Configuration:
AUTO_EXECUTE_TOOLS = True

# Helper function:
async def ai_with_tools(payload, user_id):
    if TOOL_INTEGRATION_AVAILABLE and user_id in ADMIN_WHITELIST:
        payload = EnhancedPromptBuilder.inject_tool_context(payload, user_id, ADMIN_WHITELIST)
        resp = await ai_generate_async(payload)
        if AUTO_EXECUTE_TOOLS:
            coord = ToolExecutionCoordinator(user_id, ADMIN_WHITELIST)
            resp, _ = await coord.process_ai_response(resp, auto_execute=True)
        return resp
    return await ai_generate_async(payload)

# In on_text (replace ai_generate_async call):
resp = await with_typing(context, chat.id, ai_with_tools(payload, user.id))

# That's it! Tools now available for admin users.
"""
