# Integration Guide - Using Extracted Handler Modules

## Overview

The bot has been partially refactored to extract command handlers into dedicated modules. This guide explains how to integrate these modules into `bot_server.py`.

## What's Been Extracted

### ✅ Complete Modules

1. **handlers/gating.py** (320 lines)
   - `setup_handler` - Initialize group gating
   - `ungate_handler` - Disable gating
   - `link_handler` - Get start link
   - `linkprimary_handler` - Primary group link
   - `setprimary_handler` - Set primary group
   - `start_handler` - DM start command
   - `agree_callback_handler` - Agreement callback

2. **handlers/admin.py** (480 lines)
   - `users_handler` - List unlocked users with pagination/search
   - `message_handler` - DM a user by ID/username
   - `config_handler` - Display bot configuration
   - `system_handler` - View/set system prompt
   - `autosystem_handler` - Auto-generate system prompt
   - `inspect_handler` - Inspect Telegram entities
   - `setadmin_handler` - Request admin privileges
   - `adminreq_callback_handler` - Handle admin approvals/denials

3. **handlers/chat.py** (280 lines)
   - `commands_handler` - Show available commands
   - `topic_handler` - Show channel snapshot
   - `graph_handler` - Show KG relations
   - `profile_handler` - Manage user profile

4. **handlers/messages.py** (75 lines)
   - `text_message_handler` - Handle all text messages (delegates to bot_server.on_text)
   - `error_handler` - Handle errors

5. **telegram_utils.py** (90 lines)
   - `@admin_only` decorator
   - `@whitelist_only` decorator
   - `make_group_readonly()` function
   - `make_group_writable()` function
   - `build_start_link()` function
   - `format_user_display()` function

## Integration Steps

### Step 1: Update Imports in bot_server.py

Add these imports near the top of `bot_server.py` (after the Telegram imports):

```python
# Handler modules
from handlers.gating import (
    setup_handler, ungate_handler, link_handler,
    linkprimary_handler, setprimary_handler,
    start_handler, agree_callback_handler
)
from handlers.admin import (
    users_handler, message_handler as admin_message_handler,
    config_handler, system_handler, autosystem_handler,
    inspect_handler, setadmin_handler, adminreq_callback_handler
)
from handlers.chat import (
    commands_handler, topic_handler, graph_handler, profile_handler
)
from handlers.messages import (
    text_message_handler, error_handler
)
```

### Step 2: Replace Handler Registrations

Find the section in `bot_server.py` where handlers are registered (around line 3650+) and update:

**Before:**
```python
app.add_handler(CommandHandler("setup", setup))
app.add_handler(CommandHandler("ungate", ungate))
app.add_handler(CommandHandler("link", link))
# ... etc
```

**After:**
```python
# Gating handlers
app.add_handler(CommandHandler("setup", setup_handler))
app.add_handler(CommandHandler("ungate", ungate_handler))
app.add_handler(CommandHandler("link", link_handler))
app.add_handler(CommandHandler("linkprimary", linkprimary_handler))
app.add_handler(CommandHandler("setprimary", setprimary_handler))
app.add_handler(CommandHandler("start", start_handler))
app.add_handler(CallbackQueryHandler(agree_callback_handler, pattern="^agree:"))

# Admin handlers
app.add_handler(CommandHandler("users", users_handler))
app.add_handler(CommandHandler("message", admin_message_handler))
app.add_handler(CommandHandler("config", config_handler))
app.add_handler(CommandHandler("system", system_handler))
app.add_handler(CommandHandler("autosystem", autosystem_handler))
app.add_handler(CommandHandler("inspect", inspect_handler))
app.add_handler(CommandHandler("setadmin", setadmin_handler))
app.add_handler(CallbackQueryHandler(adminreq_callback_handler, pattern="^adminreq:"))

# Chat handlers
app.add_handler(CommandHandler("commands", commands_handler))
app.add_handler(CommandHandler("topic", topic_handler))
app.add_handler(CommandHandler("graph", graph_handler))
app.add_handler(CommandHandler("profile", profile_handler))

# Message handlers
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
app.add_error_handler(error_handler)
```

### Step 3: Remove Duplicate Functions

After integrating the handlers, you can **optionally** remove the old handler definitions from `bot_server.py`:

**Functions to remove** (once integration is confirmed working):
- `setup`, `ungate`, `link`, `linkprimary`, `setprimary` (lines ~994-1048)
- `start`, `agree_cb` (lines ~3549-3577)
- `users_cmd`, `message_cmd`, `config_cmd`, `system_cmd`, `autosystem_cmd` (lines ~1051-1473)
- `inspect_cmd`, `do_inspect`, `INSPECT_PATTERNS` (lines ~1196-1251)
- `setadmin_cmd`, `adminreq_cb` (lines ~1154-1194)
- `topic_cmd`, `graph_cmd` (lines ~1283-1338)
- `profile_cmd` (lines ~1627-1700+)
- `commands_cmd`, `command_catalog`, `format_commands_for_user` (lines ~918-954)

**Keep these in bot_server.py for now** (still needed by extracted handlers):
- `admin_only`, `whitelist_only` decorators (will move to telegram_utils eventually)
- `kg_top_entities`, `kg_top_relations` (used by chat.py)
- `list_allowed` (now defined in handlers/admin.py but could be shared)
- All AI, memory, KG, profile functions (not yet extracted)

## Important Notes

### Circular Dependency Handling

The extracted handlers currently import `bot_server` to access:
- Configuration variables (DB_PATH, ADMIN_WHITELIST, etc.)
- Database functions (DBW, db(), etc.)
- AI functions (auto_generate_system_prompt, set_system_prompt_and_persist, etc.)
- Helper functions (user_db_path, ensure_schema_on, maybe_build_profile, etc.)

**This is temporary**. Future refactoring should:
1. Extract `config.py` for all configuration
2. Extract `ai_interface.py` for Ollama functions
3. Extract `memory_system.py` for memory operations
4. Pass dependencies via function parameters instead of importing

### Missing Function - list_allowed

The `list_allowed` function was **not defined** in the original `bot_server.py` but was being called. It has been **implemented** in `handlers/admin.py` with this signature:

```python
def list_allowed(chat_id: int, limit: int = 50, offset: int = 0,
                 search: Optional[str] = None) -> tuple[int, List[tuple]]:
    """
    List allowed users for a chat with pagination and search
    Returns (total_count, rows) where rows are (user_id, username, first_name, last_name)
    """
```

### Testing Checklist

After integration, test these commands:

**Gating**:
- [ ] `/setup` in a group (requires admin)
- [ ] `/ungate` in a group (requires admin)
- [ ] `/link` to get start link
- [ ] `/linkprimary` to get primary link
- [ ] `/setprimary` to set primary chat
- [ ] `/start` in DM with unlock token
- [ ] Agree button callback

**Admin**:
- [ ] `/users` to list allowed users
- [ ] `/users primary` to list primary group users
- [ ] `/users q=searchterm` to search users
- [ ] `/message @username text` to DM a user
- [ ] `/config` to view configuration
- [ ] `/system` to view system prompt
- [ ] `/system new prompt` to set prompt
- [ ] `/autosystem` to regenerate prompt
- [ ] `/inspect @username` to inspect entity
- [ ] `/setadmin` to request admin access
- [ ] Admin approval/denial buttons

**Chat**:
- [ ] `/commands` to see command list
- [ ] `/topic` in a channel to see entities/relations
- [ ] `/graph here` to see thread graph
- [ ] `/graph server` to see server-wide graph
- [ ] `/profile` in DM to manage profile
- [ ] `/profile show` to view profile
- [ ] `/profile now` to refresh profile
- [ ] `/profile erase` to delete profile

**Messages**:
- [ ] Send text in DM (should get AI response)
- [ ] Send text in group (should respond on mention/reply)
- [ ] Trigger an error (should log and show error message)

## Rollback Plan

If integration causes issues:

1. **Revert handler registrations** in bot_server.py to use old function names
2. **Comment out handler imports** from the new modules
3. **Keep extracted modules** for future use
4. **Report issues** so they can be fixed

## Next Steps

After confirming the handler integration works:

1. **Extract config.py** - Centralize all configuration loading
2. **Extract ai_interface.py** - Move Ollama functions
3. **Extract knowledge_graph.py** - Move KG operations
4. **Extract memory_system.py** - Move memory management
5. **Extract profile_manager.py** - Move profile logic
6. **Extract utils/** - Move text and context utilities
7. **Break circular dependencies** - Pass dependencies as parameters
8. **Remove duplicates** - Delete old handler code from bot_server.py
9. **Add tests** - Create unit tests for extracted modules
10. **Final cleanup** - Reduce bot_server.py to ~500 lines

## File Structure

```
tg-auth-bot/
├── bot_server.py           # Main entry point (~3700 lines → target ~500)
├── db_manager.py           # ✅ Database management (350 lines)
├── sleep_cycle.py          # ✅ Sleep/wake autonomous operations (600 lines)
├── memory_visualizer.py    # ✅ Curses UI for memory visualization (750 lines)
├── telegram_utils.py       # ✅ Telegram decorators and utilities (90 lines)
├── handlers/
│   ├── __init__.py         # ✅ Package exports (70 lines)
│   ├── gating.py           # ✅ Gating commands (320 lines)
│   ├── admin.py            # ✅ Admin commands (480 lines)
│   ├── chat.py             # ✅ Chat commands (280 lines)
│   └── messages.py         # ✅ Message handlers (75 lines)
├── config.py               # ⏳ Pending - Configuration loading
├── ai_interface.py         # ⏳ Pending - Ollama integration
├── knowledge_graph.py      # ⏳ Pending - KG operations
├── memory_system.py        # ⏳ Pending - Memory management
├── profile_manager.py      # ⏳ Pending - Profile generation
└── utils/
    ├── text_utils.py       # ⏳ Pending - Text processing
    └── context_utils.py    # ⏳ Pending - Context building
```

## Benefits of Current Refactoring

1. **Separation of Concerns** - Each handler module has a clear purpose
2. **Testability** - Handlers can be tested independently
3. **Readability** - Easier to find and understand specific commands
4. **Maintainability** - Changes to gating logic don't affect admin logic
5. **Reusability** - Handlers can be reused in other bots
6. **Documentation** - Each module is self-documenting with docstrings
7. **Type Safety** - Easier to add type hints to smaller modules

## Conclusion

You now have **9 out of 15 modules complete (60%)**, with approximately **3,015 lines extracted** into dedicated, focused modules. The handlers are ready to be integrated into `bot_server.py` following the steps above.

The remaining work (config, AI, KG, memory, profiles, utils) can be done incrementally without disrupting the current functionality.
