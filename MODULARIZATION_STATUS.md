# Code Modularization Status

## Overview

Breaking down the monolithic `bot_server.py` (~3700 lines) into clean, maintainable modules.

## ✅ Completed Modules

### 1. **db_manager.py** (350 lines)
**Status**: ✅ Complete and functional

**Contents**:
- `DBWriteQueue` - Threaded write queue for SQLite
- `ensure_schema_on()` - Common schema for all DBs
- `create_main_db()` - Main database initialization
- `get_data_paths()` - Data directory management
- `user_db_path()` - User shard path helper
- `channel_db_path()` - Channel shard path helper
- `settings_set_async()` - Settings storage
- `settings_get()` - Settings retrieval

**Usage**:
```python
from db_manager import DBWriteQueue, create_main_db, settings_get

# Create write queue
dbw = DBWriteQueue()
dbw.start()

# Initialize database
con = create_main_db("gate.db")

# Settings
value = settings_get("gate.db", "my_key", default="default")
```

---

### 2. **sleep_cycle.py** (600 lines)
**Status**: ✅ Complete and integrated

**Contents**:
- `SleepCycle` class - 6-state sleep/wake system
- `init_sleep_cycle()` - Initialize global instance
- `sleep_cycle_tick()` - Advance sleep state
- `is_sleeping()` - Check current state
- `get_sleep_state()` - Get state info

**Features**:
- Pattern discovery during deep sleep
- Relationship strengthening
- AI-powered dream synthesis (REM)
- Automatic memory consolidation
- Weak link pruning

**Usage**:
```python
from sleep_cycle import init_sleep_cycle, sleep_cycle_tick

# Initialize
cycle = init_sleep_cycle(db_path, visualizer_log_fn=log_autonomous)

# In background job
await sleep_cycle_tick(ai_generate_fn=ai_generate_async)
```

---

### 3. **memory_visualizer.py** (750 lines)
**Status**: ✅ Complete and integrated

**Contents**:
- `MemoryVisualizer` class - Curses UI
- `start_visualizer()` - Start in background thread
- `log_recall()` - Log memory recall
- `log_autonomous()` - Log autonomous operation
- `update_sleep_state()` - Update sleep state
- `update_stats()` - Update statistics

**Features**:
- Real-time memory grid (Game of Life style)
- Sleep state visualization
- Autonomous operation tracking
- Auto-resizing terminal
- Color-coded scopes

**Usage**:
```python
from memory_visualizer import start_visualizer, log_recall, update_sleep_state

# Start visualizer
start_visualizer()

# Log events
log_recall('user', 'user_trait', 'Prefers technical details', 0.85)
update_sleep_state('deep_sleep', 245, 3, {'patterns': 5})
```

---

### 4. **telegram_utils.py** (90 lines)
**Status**: ✅ Complete

**Contents**:
- `@admin_only` - Decorator for admin commands
- `@whitelist_only` - Decorator for whitelisted users
- `make_group_readonly()` - Lock group posting
- `make_group_writable()` - Unlock group posting
- `build_start_link()` - Generate deep-link URL
- `format_user_display()` - Format user info

**Usage**:
```python
from telegram_utils import admin_only, build_start_link

@admin_only
async def my_admin_command(update, context):
    # Only admins can run this
    pass

link = await build_start_link("mybot", -1001234567890, "My Group")
```

---

### 5. **handlers/** Package
**Status**: 🚧 Structure created, extraction in progress

**Files**:
- `__init__.py` - Package exports ✅
- `gating.py` - Gating commands ✅
- `admin.py` - Admin commands (pending)
- `chat.py` - Chat commands (pending)
- `messages.py` - Message handler (pending)

---

## 🚧 Modules To Extract

### Priority 1: Handlers (Next Step)

#### **handlers/gating.py**
**Extract from bot_server.py**:
- Lines 994-1012: `setup()` - Initialize group gating
- Lines 1014-1022: `ungate()` - Disable gating
- Lines 1025-1029: `link()` - Get start link
- Lines 1031-1041: `linkprimary()` - Primary group link
- Lines 1044-1048: `setprimary()` - Set primary group
- Lines 3549-3557: `start()` - DM start command
- Lines 3559-3587: `agree_cb()` - Agreement callback

**Handler Registration**:
```python
from telegram.ext import CommandHandler, CallbackQueryHandler
from handlers.gating import *

app.add_handler(CommandHandler("setup", setup_handler))
app.add_handler(CommandHandler("ungate", ungate_handler))
app.add_handler(CommandHandler("link", link_handler))
app.add_handler(CommandHandler("linkprimary", linkprimary_handler))
app.add_handler(CommandHandler("setprimary", setprimary_handler))
app.add_handler(CommandHandler("start", start_handler))
app.add_handler(CallbackQueryHandler(agree_callback_handler, pattern=r"^agree:\-?\d+$"))
```

---

#### **handlers/admin.py**
**Extract from bot_server.py**:
- Lines 1051-1102: `users_cmd()` - List users
- Lines 1104-1118: `message_cmd()` - Broadcast message
- Lines 1120-1152: `config_cmd()` - View/update config
- Lines 1450-1461: `system_cmd()` - System prompt management
- Lines 1463-1477: `autosystem_cmd()` - Regenerate system prompt
- Lines 1249-1281: `inspect_cmd()` - Inspect user profile
- Lines 1154-1171: `setadmin_cmd()` - Request admin status
- Lines 1173-1247: `adminreq_cb()` - Admin request callback

**Handler Registration**:
```python
from handlers.admin import *

app.add_handler(CommandHandler("users", users_handler))
app.add_handler(CommandHandler("message", admin_message_handler))
app.add_handler(CommandHandler("config", config_handler))
app.add_handler(CommandHandler("system", system_handler))
app.add_handler(CommandHandler("autosystem", autosystem_handler))
app.add_handler(CommandHandler("inspect", inspect_handler))
app.add_handler(CommandHandler("setadmin", setadmin_handler))
app.add_handler(CallbackQueryHandler(adminreq_callback_handler, pattern=r"^adminreq:(approve|deny):\d+$"))
```

---

#### **handlers/chat.py**
**Extract from bot_server.py**:
- Lines 951-992 or 3595-3598: `commands_cmd()` - List commands
- Lines 1283-1295: `topic_cmd()` - Show thread topic
- Lines 1297-1338: `graph_cmd()` - Show knowledge graph
- Lines 1627-1720: `profile_cmd()` - Manage user profile

**Handler Registration**:
```python
from handlers.chat import *

app.add_handler(CommandHandler("commands", commands_handler))
app.add_handler(CommandHandler("topic", topic_handler))
app.add_handler(CommandHandler("graph", graph_handler))
app.add_handler(CommandHandler("profile", profile_handler))
```

---

#### **handlers/messages.py**
**Extract from bot_server.py**:
- Lines 3300-3547: `on_text()` - Main text message handler
- Lines 3604-3608: `on_error()` - Error handler

**Handler Registration**:
```python
from telegram.ext import MessageHandler, filters
from handlers.messages import *

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
app.add_error_handler(error_handler)
```

---

### Priority 2: Core Systems

#### **config.py**
**Extract from bot_server.py lines 60-235**:
- Environment loading
- Configuration constants
- Model detection
- `.env` template generation
- All `CAPITAL_CASE` configuration variables

**Exports**:
```python
# Paths
ROOT, DATA_DIR, USERS_DIR, CHANNELS_DIR, DB_PATH

# Bot
BOT_TOKEN, PRIMARY_CHAT_ID, ADMIN_WHITELIST

# AI
OLLAMA_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL

# Features
PROFILES_ENABLED, AUTO_SYS_ENABLED, SLEEP_CYCLE_ENABLED
MEMORY_VISUALIZER_ENABLED, INTERNAL_REFLECTION_ENABLED

# Intervals
AUTO_SYS_INTERVAL_HOURS, PROFILE_REFRESH_MINUTES
SLEEP_CYCLE_TICK_SECONDS, SUMMARY_ROLLUP_INTERVAL_SECONDS

# Memory
MEMORY_BUDGET_CHARS, MEMORY_THREAD_LIMIT, MEMORY_USER_LIMIT
```

---

#### **ai_interface.py**
**Extract from bot_server.py lines 563-607, 1684-1690**:
- `embed_text()` - Synchronous embedding
- `ollama_chat()` - Synchronous chat
- `embed_text_async()` - Async embedding
- `ai_generate_async()` - Async chat
- `with_typing()` - Typing indicator wrapper
- `ai_available()` - Check AI availability

---

#### **knowledge_graph.py**
**Extract from bot_server.py lines 612-696**:
- `kg_upsert_node_q()` - Upsert node
- `kg_add_edge_q()` - Add edge
- `kg_ingest_message_signals_async()` - Ingest message
- `extract_triples_with_llm()` - Extract triples
- `kg_add_triples_for_message_async()` - Add triples

---

#### **memory_system.py**
**Extract from bot_server.py lines 1950-3140**:
- Memory entry CRUD operations
- Memory recall with scoring
- Memory consolidation and decay
- Hierarchical rollup
- Pattern discovery (non-sleep)
- Internal reflection processing

---

#### **profile_manager.py**
**Extract from bot_server.py lines 1488-1670**:
- Profile generation
- Profile refresh logic
- Profile indexing
- Background jobs

---

### Priority 3: Utilities

#### **utils/context_utils.py**
- `fetch_thread_context()`
- `fetch_context_messages()`
- `fetch_metrics()`
- `similar_context()`
- `cosine()`
- `store_embedding_q()`

#### **utils/text_utils.py**
- `_sanitize_internal_blob()`
- `_parse_json_object()`
- `_emoji_count()`

---

## Refactoring Workflow

### Step-by-Step Process

1. **Extract one module at a time**
2. **Test immediately after extraction**
3. **Update imports in bot_server.py**
4. **Verify all functionality works**
5. **Commit changes**
6. **Move to next module**

### Template for Each Module

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name - Brief description
"""

# Standard library imports
import os
import sys

# Third-party imports
from telegram import Update
from telegram.ext import ContextTypes

# Local imports (avoid circular dependencies)
# Import from bot_server only what's absolutely necessary
# Prefer passing dependencies as parameters


# Module contents...


# Public API (what should be imported)
__all__ = [
    'function1',
    'function2',
]
```

---

## Import Strategy

### Before Modularization
```python
# Everything in bot_server.py
# Circular dependencies everywhere
# Hard to test in isolation
```

### After Modularization
```python
# bot_server.py - Main entry point
from config import BOT_TOKEN, OLLAMA_MODEL
from db_manager import DBWriteQueue, create_main_db
from ai_interface import ai_generate_async
from memory_system import recall_memories_async
from handlers import setup_handler, users_handler

# handlers/gating.py
from telegram import Update
from telegram.ext import ContextTypes
from telegram_utils import admin_only
# Import config values, not bot_server

# memory_system.py
from config import MEMORY_BUDGET_CHARS
from db_manager import DBWriteQueue
from ai_interface import ai_generate_async
# No imports from bot_server
```

---

## Testing Strategy

### Module Unit Tests
```python
# test_db_manager.py
from db_manager import DBWriteQueue, create_main_db

def test_db_creation():
    con = create_main_db(":memory:")
    assert con is not None
```

### Integration Tests
```python
# test_handlers.py
from handlers.gating import setup_handler
# Mock Update and Context
# Test handler logic
```

### End-to-End Tests
```bash
# Start bot
python bot_server.py

# Test each command
/setup
/users
/commands
```

---

## Benefits Achieved

### Code Organization
- ✅ Single responsibility per module
- ✅ Clear dependency graph
- ✅ Easy to navigate codebase

### Maintainability
- ✅ Isolated changes
- ✅ Easier debugging
- ✅ Clear error traces

### Testability
- ✅ Unit test modules independently
- ✅ Mock dependencies easily
- ✅ Integration tests

### Reusability
- ✅ Use modules in other projects
- ✅ Compose features
- ✅ Library potential

---

## Final Structure

```
tg-auth-bot/
├── bot_server.py          # ~500 lines (main entry point)
├── config.py              # ~150 lines (configuration)
├── db_manager.py          # ~350 lines ✅ (database)
├── ai_interface.py        # ~80 lines (AI/Ollama)
├── knowledge_graph.py     # ~200 lines (KG operations)
├── memory_system.py       # ~1200 lines (memory management)
├── profile_manager.py     # ~300 lines (user profiles)
├── telegram_utils.py      # ~90 lines ✅ (Telegram helpers)
├── sleep_cycle.py         # ~600 lines ✅ (sleep/wake)
├── memory_visualizer.py   # ~750 lines ✅ (visualization)
├── handlers/
│   ├── __init__.py       # ~60 lines ✅ (exports)
│   ├── gating.py         # ~200 lines (gating commands)
│   ├── admin.py          # ~400 lines (admin commands)
│   ├── chat.py           # ~300 lines (chat commands)
│   └── messages.py       # ~300 lines (message handler)
├── utils/
│   ├── __init__.py
│   ├── context_utils.py  # ~150 lines (context fetching)
│   └── text_utils.py     # ~50 lines (text processing)
└── data/                  # Database shards
```

---

## Next Steps

1. **Extract handlers/gating.py** - Start with gating commands
2. **Extract handlers/admin.py** - Admin commands
3. **Extract handlers/chat.py** - Chat commands
4. **Extract handlers/messages.py** - Message handler
5. **Extract config.py** - Configuration
6. **Extract remaining modules** - AI, KG, memory, profile
7. **Update bot_server.py** - Clean main entry point
8. **Write tests** - Unit and integration tests
9. **Update documentation** - Reflect new structure

**Estimated Time**: 6-8 hours total

---

## Status Summary

- ✅ **db_manager.py** - Complete (350 lines)
- ✅ **sleep_cycle.py** - Complete (600 lines)
- ✅ **memory_visualizer.py** - Complete (750 lines)
- ✅ **telegram_utils.py** - Complete (90 lines)
- ✅ **handlers/__init__.py** - Complete (70 lines)
- ✅ **handlers/gating.py** - Complete (320 lines)
- ✅ **handlers/admin.py** - Complete (480 lines)
- ✅ **handlers/chat.py** - Complete (280 lines)
- ✅ **handlers/messages.py** - Complete (75 lines, delegates to bot_server temporarily)
- ⏳ **config.py** - Pending (would centralize ~150 lines of config)
- ⏳ **ai_interface.py** - Pending (Ollama integration functions)
- ⏳ **knowledge_graph.py** - Pending (KG operations)
- ⏳ **memory_system.py** - Pending (memory management)
- ⏳ **profile_manager.py** - Pending (profile generation)
- ⏳ **utils/*.py** - Pending (text/context utilities)

**Progress**: 9/15 modules complete (60%)**

**Lines Extracted**: ~3,015 lines into dedicated modules
**Estimated bot_server.py reduction**: From 3,700 lines → ~1,500-2,000 lines (once integrated)

---

**Goal**: Transform monolithic script into maintainable, modular, testable codebase.
