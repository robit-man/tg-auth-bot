# Bot Server Refactoring Plan

## Overview

The `bot_server.py` file is currently ~3700 lines. This plan outlines the modular extraction to improve maintainability, testability, and code organization.

## Current Status

### ✅ **Completed Modules**

1. **[db_manager.py](db_manager.py)** - Database management
   - DBWriteQueue (threaded write queue)
   - Schema creation and migrations
   - Connection pooling
   - Settings helpers
   - Path utilities

2. **[memory_visualizer.py](memory_visualizer.py)** - Memory visualization
   - Curses-based UI
   - Sleep state visualization
   - Autonomous operation tracking

3. **[sleep_cycle.py](sleep_cycle.py)** - Autonomous sleep/wake cycles
   - 6-state sleep machine
   - Pattern discovery
   - Relationship strengthening
   - Dream synthesis

## Recommended Module Structure

```
tg-auth-bot/
├── bot_server.py              # Main entry point (~500 lines after refactor)
├── config.py                  # Configuration and environment
├── db_manager.py             # ✅ Database management
├── ai_interface.py           # AI/Ollama integration
├── knowledge_graph.py        # Knowledge graph operations
├── memory_system.py          # Memory entries, recall, consolidation
├── profile_manager.py        # User interaction profiles
├── sleep_cycle.py            # ✅ Sleep/wake autonomous processing
├── memory_visualizer.py      # ✅ Curses visualization
├── handlers/
│   ├── __init__.py
│   ├── admin_commands.py     # /config, /users, /system, etc.
│   ├── gating_commands.py    # /setup, /ungate, /link
│   ├── chat_commands.py      # /topic, /graph, /commands
│   └── message_handler.py    # Text message processing
└── utils/
    ├── __init__.py
    ├── context_utils.py      # Context fetching and similarity
    └── text_utils.py         # Text processing helpers
```

## Module Breakdown

### 1. **config.py** (NEW)
**Extract from bot_server.py lines ~60-200**

**Contents:**
- Environment variable loading
- Configuration constants
- Model detection
- `.env` template generation

**Exports:**
```python
# Paths
ROOT, DATA_DIR, USERS_DIR, CHANNELS_DIR, DB_PATH

# Bot config
BOT_TOKEN, PRIMARY_CHAT_ID, ADMIN_WHITELIST

# AI config
OLLAMA_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL

# Feature flags
PROFILES_ENABLED, AUTO_SYS_ENABLED, SLEEP_CYCLE_ENABLED
MEMORY_VISUALIZER_ENABLED, INTERNAL_REFLECTION_ENABLED

# Intervals
AUTO_SYS_INTERVAL_HOURS, PROFILE_REFRESH_MINUTES
SLEEP_CYCLE_TICK_SECONDS, SUMMARY_ROLLUP_INTERVAL_SECONDS

# Memory limits
MEMORY_BUDGET_CHARS, MEMORY_THREAD_LIMIT, MEMORY_USER_LIMIT
MEMORY_GLOBAL_LIMIT, MEMORY_SUMMARY_BATCH
```

### 2. **ai_interface.py** (NEW)
**Extract from bot_server.py lines ~563-607, 1684-1690**

**Contents:**
- Ollama API integration
- Text embedding
- Chat generation
- AI availability check
- Typing action wrapper

**Exports:**
```python
def embed_text(text: str) -> Optional[List[float]]
def ollama_chat(payload: dict) -> Optional[str]
async def embed_text_async(text: str) -> Optional[List[float]]
async def ai_generate_async(payload: dict) -> Optional[str]
async def with_typing(context, chat_id, coro)
def ai_available() -> bool
```

### 3. **knowledge_graph.py** (NEW)
**Extract from bot_server.py lines ~612-696**

**Contents:**
- Node and edge management
- Graph ingestion from messages
- Triple extraction with LLM
- KG queries and traversal

**Exports:**
```python
async def kg_upsert_node_q(node_type: str, node_key: str, label: str) -> int
async def kg_add_edge_q(src_id, rel, dst_id, weight, ts, chat_id, thread_id, ...)
async def kg_ingest_message_signals_async(row_id: int)
def extract_triples_with_llm(snippet: str) -> List[dict]
async def kg_add_triples_for_message_async(row_id: int)
```

### 4. **memory_system.py** (NEW)
**Extract from bot_server.py lines ~1950-3140**

**Contents:**
- Memory entry storage and retrieval
- Memory recall with scoring
- Pattern discovery (basic, non-sleep)
- Memory consolidation and decay
- Hierarchical rollup
- Memory scope management
- Internal reflection processing

**Exports:**
```python
# Memory operations
async def _store_memory_entry_async(scope, chat_id, thread_id, user_id, category, content, metadata, weight, sources)
async def _delete_memory_entries_async(ids: List[int])
def _memory_rows_for_scope(scope, chat_id, thread_id, user_id, limit) -> list[tuple]

# Memory recall
async def recall_memories_async(user_text, chat, thread_id, user) -> Dict
def _memory_scope_identifiers(chat, thread_id, user) -> Dict
def _memory_limit_for(scope: str) -> int

# Memory processing
async def process_memory_update_async(meta, chat, thread_id, user, thread_ctx, user_text, bot_reply)
async def _maybe_decay_memories_async(scope, chat_id, thread_id, user_id, category)
async def _store_memories_with_decay(entries: List[dict]) -> List[dict]

# Hierarchical processing
async def hierarchical_memory_rollup()
def fetch_latest_summary(scope, chat_id, thread_id, user_id, category) -> str

# Constants
MEMORY_SCOPE_WEIGHTS
MEMORY_SCOPE_PRIORITIES
```

### 5. **profile_manager.py** (NEW)
**Extract from bot_server.py lines ~1488-1670**

**Contents:**
- User interaction profile generation
- Profile refresh logic
- Profile indexing
- Style analysis

**Exports:**
```python
async def profiles_idx_get(user_id: int) -> Tuple[int, int]
async def profiles_idx_set(user_id: int, last_updated: int, optout: Optional[bool])
async def update_profile_for_user(user_id: int)
async def profile_background_job(context)
def _emoji_count(s: str) -> int
```

### 6. **handlers/** Package (NEW)

#### **handlers/admin_commands.py**
Commands: `/config`, `/users`, `/message`, `/system`, `/inspect`, `/autosystem`, `/setadmin`

#### **handlers/gating_commands.py**
Commands: `/setup`, `/ungate`, `/link`, `/linkprimary`, `/setprimary`, `/start`

#### **handlers/chat_commands.py**
Commands: `/topic`, `/graph`, `/commands`, `/profile`

#### **handlers/message_handler.py**
Text message processing, auto-reply logic, context assembly

### 7. **utils/** Package (NEW)

#### **utils/context_utils.py**
```python
def fetch_thread_context(chat_id, thread_id, limit) -> str
def fetch_context_messages(limit_msgs, limit_chars) -> str
def fetch_metrics(max_users, max_groups) -> Dict
async def similar_context(query, chat_id, thread_id, top_k) -> Dict
def cosine(a, b) -> float
async def store_embedding_q(db_path, kind, ref_id, chat_id, thread_id, user_id, ts, vec)
```

#### **utils/text_utils.py**
```python
def _sanitize_internal_blob(raw: str) -> str
def _parse_json_object(raw: str) -> Optional[dict]
def _emoji_count(text: str) -> int
```

## Refactoring Steps

### Phase 1: Configuration (30 min)
1. Create `config.py`
2. Move all environment loading and constants
3. Update imports in `bot_server.py`
4. Test that bot starts

### Phase 2: AI Interface (20 min)
1. Create `ai_interface.py`
2. Move Ollama functions
3. Update imports
4. Test chat and embeddings

### Phase 3: Knowledge Graph (30 min)
1. Create `knowledge_graph.py`
2. Move KG functions
3. Update imports
4. Test KG ingestion

### Phase 4: Memory System (60 min)
1. Create `memory_system.py`
2. Move memory functions (largest module)
3. Carefully update imports
4. Test memory recall and storage

### Phase 5: Profile Manager (30 min)
1. Create `profile_manager.py`
2. Move profile functions
3. Update imports
4. Test profile generation

### Phase 6: Handlers (60 min)
1. Create `handlers/` package
2. Create handler files
3. Move command handlers
4. Update bot registration
5. Test all commands

### Phase 7: Utils (20 min)
1. Create `utils/` package
2. Move utility functions
3. Update imports
4. Test utilities

### Phase 8: Testing & Validation (30 min)
1. Run full bot test
2. Test each command
3. Test autonomous operations
4. Test sleep cycle
5. Test visualizer

**Total Time: ~4.5 hours**

## Benefits

### Code Organization
- **Single Responsibility**: Each module has one clear purpose
- **Easier Navigation**: Find code by logical grouping
- **Reduced Cognitive Load**: Work on one system at a time

### Maintainability
- **Isolated Changes**: Modify memory system without touching handlers
- **Clear Dependencies**: Explicit imports show relationships
- **Easier Debugging**: Smaller files, clearer stack traces

### Testability
- **Unit Testing**: Test modules independently
- **Mocking**: Mock dependencies easily
- **Integration Testing**: Test module interactions

### Reusability
- **Import Anywhere**: Use memory system in other projects
- **Compose Features**: Mix and match modules
- **Library Potential**: Modules could become standalone packages

## Migration Strategy

### Backward Compatibility
Keep `bot_server.py` as the main entry point. Other code can still import from it during transition.

### Gradual Migration
1. Extract modules one at a time
2. Test after each extraction
3. Keep git history clean with atomic commits

### Import Patterns

**Before:**
```python
# Everything in bot_server.py
from bot_server import ai_generate_async, recall_memories_async
```

**After:**
```python
# Specific imports
from ai_interface import ai_generate_async
from memory_system import recall_memories_async
from config import OLLAMA_MODEL, MEMORY_BUDGET_CHARS
```

## File Size Reduction

### Current
- `bot_server.py`: ~3700 lines

### After Refactoring
- `bot_server.py`: ~500 lines (main entry, bot setup)
- `config.py`: ~150 lines
- `db_manager.py`: ~350 lines ✅
- `ai_interface.py`: ~80 lines
- `knowledge_graph.py`: ~200 lines
- `memory_system.py`: ~1200 lines
- `profile_manager.py`: ~300 lines
- `handlers/*.py`: ~800 lines total
- `utils/*.py`: ~200 lines total
- `sleep_cycle.py`: ~600 lines ✅
- `memory_visualizer.py`: ~700 lines ✅

**Total lines unchanged, but spread across 15+ focused files**

## Next Steps

1. **Create config.py** - Start with configuration extraction
2. **Test thoroughly** - Ensure nothing breaks
3. **One module at a time** - Don't rush
4. **Update documentation** - Document each module
5. **Add type hints** - Improve type safety during refactor
6. **Write unit tests** - Test modules in isolation

## Notes

- Keep db_manager.py, sleep_cycle.py, and memory_visualizer.py as-is (already well-structured)
- bot_server.py should only contain:
  - Main entry point
  - Bot application setup
  - Handler registration
  - Job scheduling
- Each module should have clear inputs/outputs
- Minimize circular dependencies
- Use dependency injection where possible

---

**The goal: Transform a monolithic script into a maintainable, modular codebase.**
