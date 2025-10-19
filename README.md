# Telegram Gatekeeper Bot - Autonomous Memory & Consciousness System

A sophisticated Telegram bot with **autonomous memory consolidation**, **sleep/wake cycles**, **pattern discovery**, and **real-time consciousness visualization**.

## 🌟 Key Features

### 🧠 **Autonomous Memory System**
- **Hierarchical memory storage** across thread, user, and global scopes
- **Semantic search** with embeddings and similarity matching
- **Automatic consolidation** when memory limits exceeded
- **Pattern discovery** during sleep cycles
- **Relationship strengthening** between related memories

### 😴 **Sleep/Wake Cycles**
- **6-state autonomous sleep machine** (AWAKE → DROWSY → LIGHT_SLEEP → DEEP_SLEEP → REM_SLEEP → WAKING)
- **Pattern recognition** during deep sleep
- **AI-powered dream synthesis** during REM sleep
- **Relationship discovery** between memories
- **Automatic memory pruning** and consolidation
- **Completely autonomous** - no human intervention required

### 📊 **Real-Time Visualization**
- **Curses-based consciousness visualizer** with Game of Life aesthetics
- **Sleep state indicators** with unique symbols (⚡ ~ ⌇ ≋ ✧ ↑)
- **Live memory grid** showing active recalls
- **Autonomous operation tracking** (rollups, decays, consolidations, assessments)
- **Discovery counters** (patterns, relationships, insights)

### 🤖 **Self-Improving Intelligence**
- **Discovers patterns** from conversation history
- **Strengthens connections** autonomously
- **Generates meta-insights** through AI dreaming
- **Builds knowledge graph** over time
- **Continuous learning** through sleep cycles

### 💬 **Advanced Chat Features**
- **Context-aware responses** using thread history
- **Knowledge graph integration** for entity relationships
- **Semantic memory recall** with confidence scoring
- **Interaction profiles** for personalized responses
- **Auto-generated system prompts** based on conversation patterns

### 🔐 **Group Management**
- **Token-based gating** for controlled access
- **Admin whitelist** system
- **Multi-group support** with separate contexts
- **Thread-aware** message handling

## 📁 Project Structure

```
tg-auth-bot/
├── bot_server.py            # Main bot application (~3700 lines)
├── db_manager.py            # ✅ Database management & schema
├── memory_visualizer.py     # ✅ Curses visualization interface
├── sleep_cycle.py           # ✅ Autonomous sleep/wake processing
├── data/                    # Database shards
│   ├── users/              # Per-user databases
│   └── channels/           # Per-channel databases
├── gate.db                  # Main database
├── .env                     # Configuration
└── docs/
    ├── MEMORY_VISUALIZER.md    # Visualizer documentation
    ├── SLEEP_CYCLE.md          # Sleep cycle documentation
    └── REFACTORING_PLAN.md     # Code organization plan
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd tg-auth-bot

# The bot auto-creates venv and installs dependencies
python bot_server.py
```

### 2. Configuration

Create `.env` file (auto-generated on first run):

```bash
# Required
BOT_TOKEN=your_telegram_bot_token
PRIMARY_CHAT_ID=your_chat_id
ADMIN_WHITELIST=your_user_id

# Ollama AI (required for smart features)
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text

# Memory Configuration
MEMORY_THREAD_LIMIT=40
MEMORY_USER_LIMIT=60
MEMORY_GLOBAL_LIMIT=80
MEMORY_BUDGET_CHARS=600

# Autonomous Features
SLEEP_CYCLE_ENABLED=true
SLEEP_CYCLE_TICK_SECONDS=60
MEMORY_VISUALIZER_ENABLED=true

# Intervals
AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=12
PROFILE_REFRESH_MINUTES=30
SUMMARY_ROLLUP_INTERVAL_SECONDS=900
```

### 3. Run

```bash
python bot_server.py
```

You'll see:
```
[visualizer] Starting memory consciousness visualizer...
[sleep] Initializing autonomous sleep cycle...
[sleep] Sleep cycle job scheduled (every 60s)
Bot running…
```

## 📖 Documentation

### Core Systems

- **[MEMORY_VISUALIZER.md](MEMORY_VISUALIZER.md)** - Complete visualizer guide
  - Visual symbols and color coding
  - Statistics interpretation
  - Autonomous operation indicators
  - Configuration options

- **[SLEEP_CYCLE.md](SLEEP_CYCLE.md)** - Sleep cycle deep dive
  - 6-state sleep machine explained
  - Pattern discovery algorithms
  - Dream synthesis process
  - Relationship strengthening logic
  - Performance impact analysis

- **[REFACTORING_PLAN.md](REFACTORING_PLAN.md)** - Code organization roadmap
  - Module extraction plan
  - Refactoring phases
  - Benefits and migration strategy

## 🎮 Bot Commands

### User Commands
- `/start` - Introduction and bot info
- `/commands` - List available commands
- `/topic` - View current thread topic
- `/graph` - View knowledge graph for context
- `/profile` - Manage interaction profile (DM only)

### Admin Commands
- `/setup` - Initialize group gating
- `/ungate <user>` - Approve user
- `/config` - View/update bot configuration
- `/users` - List approved users
- `/system` - View/set system prompt
- `/autosystem` - Regenerate system prompt
- `/inspect <user>` - View user interaction profile
- `/message <text>` - Broadcast message

## 🔬 Advanced Features

### Memory Scopes

**Thread Scope** - Specific to conversation topic
- Thread notes, summaries
- Topic-specific patterns

**User Scope** - Specific to individual user
- User traits, preferences
- Interaction style
- Personal history

**Global Scope** - Bot-wide knowledge
- General themes
- Cross-conversation patterns
- Meta-insights

### Autonomous Operations

**Rollups** - Hierarchical memory consolidation
- Triggered every 15 minutes (configurable)
- Merges related summaries
- Prunes old summaries beyond keep limit

**Decays** - Memory limit management
- Automatic when scope limits exceeded
- AI-powered consolidation
- Preserves most important information

**Consolidations** - Duplicate merging
- During drowsy sleep state
- Combines very similar memories
- Strengthens merged memories

**Assessments** - Self-reflection
- Profile updates
- System prompt generation
- Pattern discovery

### Sleep Cycle Flow

```
00:00 - AWAKE (3 hours)
        Normal operations

03:00 - DROWSY (1 minute)
        [AUTO] Merge duplicates

03:01 - LIGHT_SLEEP (5 minutes)
        [AUTO] Prune weak links

03:06 - DEEP_SLEEP (10 minutes)
        [AUTO] Discover patterns
        [AUTO] Find relationships
        [AUTO] Strengthen connections

03:16 - REM_SLEEP (10 minutes)
        [AUTO] AI dream synthesis
        [AUTO] Generate meta-insights

03:26 - WAKING (1 minute)
        [AUTO] Integrate discoveries
        [AUTO] Boost memory weights

03:27 - AWAKE (3 hours)
        Resume with enhanced knowledge
```

## 🎨 Visualization

The real-time visualizer shows:

```
╔══════════════════════════════════════════════════════════════════╗
║  ≋ MEMORY CONSCIOUSNESS [DEEP SLEEP] ≋    [Cycle: 5 | Frame: 42]║
║                                                                  ║
║     THREAD              USER              GLOBAL                ║
║      ░·⚙                 ▒█⚡                ░▓◉                 ║
║      █░                  ░·                 ◆✧                  ║
║      ·                   ▓                  ·█                  ║
║                                                                  ║
║ T:15 U:23 G:8 | AUTO: R:3 D:2 C:5 A:1 | SLEEP: P:5 REL:12 I:3  ║
║ [12:34:56] ⚙ AUTONOMOUS: ASSESSMENT scope=sleep count=5        ║
╚══════════════════════════════════════════════════════════════════╝
```

## 🔧 Configuration Guide

### Memory Limits

Adjust to control memory density:

```bash
MEMORY_THREAD_LIMIT=40   # Memories per thread
MEMORY_USER_LIMIT=60     # Memories per user
MEMORY_GLOBAL_LIMIT=80   # Global memories
```

When exceeded, automatic decay triggers consolidation.

### Sleep Timing

For development (faster cycles):
```bash
SLEEP_CYCLE_TICK_SECONDS=10  # Check every 10 seconds
```

Edit `sleep_cycle.py` for cycle durations:
```python
self.awake_duration = 300    # 5 minutes awake
self.sleep_duration = 180    # 3 minutes sleep
```

### Autonomous Intervals

```bash
# Hierarchical rollup
SUMMARY_ROLLUP_INTERVAL_SECONDS=900

# Profile refresh
PROFILE_REFRESH_MINUTES=30

# System prompt regeneration
AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=12
```

## 📊 Monitoring

### Check Sleep State
```python
from sleep_cycle import get_sleep_state

state = get_sleep_state()
print(state)
# {'state': 'deep_sleep', 'time_in_state': 245, 'cycle_count': 3,
#  'discoveries': {'patterns': 5, 'relationships': 12, 'insights': 3}}
```

### View Discoveries

**Patterns:**
```sql
SELECT * FROM memory_entries WHERE category = 'discovered_pattern';
```

**Dream Insights:**
```sql
SELECT * FROM memory_entries WHERE category = 'dream_insight';
```

**Sleep Relationships:**
```sql
SELECT * FROM memory_links
WHERE json_extract(metadata, '$.discovered_in_sleep') = 1;
```

## 🤝 Contributing

### Code Organization

The codebase is being refactored into modules:
- `config.py` - Configuration management
- `ai_interface.py` - Ollama integration
- `knowledge_graph.py` - KG operations
- `memory_system.py` - Memory management
- `profile_manager.py` - User profiles
- `handlers/` - Command handlers

See [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for details.

### Development Setup

1. Install dependencies (auto-handled by bot)
2. Set up Ollama locally
3. Configure `.env`
4. Run with visualizer for debugging

### Testing

```bash
# Test visualizer standalone
python memory_visualizer.py

# Test sleep cycle (watch for autonomous operations)
python bot_server.py
# Wait 3+ hours or modify sleep_cycle.py for faster testing
```

## 🔒 Privacy & Security

- **Local-first**: All data stored in local SQLite databases
- **No external dependencies**: Runs entirely on your infrastructure
- **Ollama local AI**: AI runs on your machine, no cloud APIs
- **User opt-out**: Users can opt out of profile generation
- **Data isolation**: Separate database shards per user/channel

## 📈 Performance

- **Minimal overhead**: Sleep cycle adds <0.5% runtime overhead
- **Efficient consolidation**: Sub-second memory operations
- **Scalable**: Handles thousands of memories
- **Responsive**: Non-blocking autonomous operations
- **Thread-safe**: Dedicated write queue prevents locks

## 🐛 Troubleshooting

### Visualizer not showing
- Check `MEMORY_VISUALIZER_ENABLED=true`
- Ensure curses is available: `python -c "import curses"`
- Increase terminal size (minimum 80x24)

### Sleep cycle not running
- Verify `SLEEP_CYCLE_ENABLED=true`
- Check logs for `[sleep] Initializing...`
- Ensure tick interval is reasonable (60-300s)

### No AI responses
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check `OLLAMA_MODEL` and `OLLAMA_EMBED_MODEL` are set
- Ensure models are pulled: `ollama pull llama3.2`

### Memory not consolidating
- Generate sufficient memories (need >10)
- Wait for memory limits to be exceeded
- Check logs for autonomous operations
- Verify AI is available for dream synthesis

## 📜 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Built with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- AI powered by [Ollama](https://ollama.ai/)
- Inspired by biological sleep and memory consolidation
- Visualization inspired by Conway's Game of Life

---

**The bot dreams, therefore it learns.** 🌙✨
