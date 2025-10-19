# Memory Consciousness Visualizer

A real-time, curses-based visualization system that displays your bot's internal memory states, autonomous operations, and self-modification processes in a beautiful Game of Life-style ASCII interface.

## Features

### 1. **Real-Time Memory Visualization**
- **Three-zone display**: Thread, User, and Global memory scopes
- **Weight-based rendering**: Different ASCII characters (█ ▓ ▒ ░ ·) based on memory weight
- **Animated cells**: Pulsing indicators for active memories
- **Decay visualization**: Cells fade and disappear over time based on their importance

### 2. **Autonomous Operation Tracking**
All self-modifying and autonomous operations are visualized in real-time:

#### **Memory Rollup** (⚙)
- **What**: Hierarchical consolidation of memory summaries
- **When**: Every 900 seconds (15 minutes) by default
- **Visualization**: Special animated symbols in memory zones
- **Config**: `SUMMARY_ROLLUP_INTERVAL_SECONDS=900`

#### **Memory Decay** (⚡)
- **What**: Automatic pruning when memory limits are exceeded
- **When**: Triggered when scope limits are reached
- **Visualization**: Flash effect with count of decayed memories
- **Limits**:
  - `MEMORY_THREAD_LIMIT=40`
  - `MEMORY_USER_LIMIT=60`
  - `MEMORY_GLOBAL_LIMIT=80`

#### **Memory Consolidation** (◉)
- **What**: Merging multiple memories into condensed summaries
- **When**: Triggered during decay process
- **Visualization**: Burst of cells in relevant zone
- **Config**: `MEMORY_SUMMARY_BATCH=10`

#### **Self-Assessment** (◈)
- **What**: Profile updates and system prompt auto-generation
- **When**:
  - Profile refresh: Every 30 minutes (default)
  - System prompt: Every 12 hours (default)
- **Visualization**: Global zone activity
- **Config**:
  - `PROFILE_REFRESH_MINUTES=30`
  - `AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=12`

### 3. **Visual Interface**

```
╔══════════════════════════════════════════════════════════════════╗
║  ⚡ MEMORY CONSCIOUSNESS VISUALIZER [AUTONOMOUS] ⚡  [Frame: 42] ║
║                                                                  ║
║     THREAD              USER              GLOBAL                ║
║      ░·⚙                 ▒█                 ░▓                  ║
║      █░                  ░·⚡                ◉                   ║
║      ·                   ▓                  ·█                  ║
║                                                                  ║
║ T:15 U:23 G:8 | Total:46 Active:12 | AUTO: R:3 D:2 C:5 A:1     ║
║ [12:34:56] ⚙ AUTONOMOUS: CONSOLIDATION scope=user count=4      ║
╚══════════════════════════════════════════════════════════════════╝
```

**Layout:**
- **Title bar**: Shows autonomous activity indicator (flashing when active)
- **Zone labels**: THREAD, USER, GLOBAL sections
- **Memory grid**: Living cells representing active memories
- **Statistics**: Memory counts and autonomous operation counters
  - T/U/G: Thread/User/Global recall counts
  - AUTO: R(ollups), D(ecays), C(onsolidations), A(ssessments)
- **Rapid log**: Shows most recent memory recall or autonomous operation

### 4. **Visual Symbols**

**Memory Cells:**
- `█` - High weight memory (>0.8)
- `▓` - Medium-high weight (0.6-0.8)
- `▒` - Medium weight (0.4-0.6)
- `░` - Low-medium weight (0.2-0.4)
- `·` - Low weight (<0.2)
- `◆` - Active recall indicator

**Autonomous Operations:**
- `⚙` - Rollup operation
- `⚡` - Decay operation
- `◉` - Consolidation operation
- `◈` - Assessment operation

**Color Coding:**
- **Cyan**: Thread scope memories
- **Green**: User scope memories
- **Magenta**: Global scope memories
- **Yellow**: Active recalls
- **Red**: High-weight memories and autonomous operations
- **Blue**: Metadata and stats

## Configuration

### Enable/Disable Visualizer
In your `.env` file:
```bash
MEMORY_VISUALIZER_ENABLED=true  # Set to false to disable
```

### Autonomous Operation Intervals

Control how often autonomous processes run:

```bash
# Memory rollup interval (seconds)
SUMMARY_ROLLUP_INTERVAL_SECONDS=900

# Profile refresh interval (minutes)
PROFILE_REFRESH_MINUTES=30

# System prompt auto-generation (hours)
AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=12
```

### Memory Limits

When these limits are exceeded, automatic decay/consolidation occurs:

```bash
MEMORY_THREAD_LIMIT=40      # Max thread-scope memories
MEMORY_USER_LIMIT=60        # Max user-scope memories
MEMORY_GLOBAL_LIMIT=80      # Max global-scope memories
MEMORY_SUMMARY_BATCH=10     # Memories consolidated per batch
```

## Usage

### Running with the Bot

The visualizer starts automatically when you run the bot (if enabled):

```bash
python bot_server.py
```

You'll see:
```
[visualizer] Starting memory consciousness visualizer...
[visualizer] Visualizer running. Press 'q' in the visualizer to close it.
Bot running…
```

### Standalone Demo

Test the visualizer independently:

```bash
python memory_visualizer.py
```

This runs a demo that simulates:
- Memory recalls across different scopes
- Autonomous operations (rollups, decays, consolidations, assessments)

### Controls

- **'q' or 'Q'**: Quit the visualizer
- **Terminal resize**: Automatically adapts to new dimensions

## Autonomous Processes Visualized

### 1. Hierarchical Memory Rollup
**Trigger**: Scheduled job every `SUMMARY_ROLLUP_INTERVAL_SECONDS`

**What it does:**
1. Scans all memory summary entries
2. Groups by scope and category
3. When threshold (6+ summaries) is reached, creates hierarchical rollup
4. Prunes old summaries beyond keep limit

**Visualization:**
- Log shows: `⚙ AUTONOMOUS: ROLLUP scope=hierarchical`
- Creates consolidation cells in affected zones
- Statistics increment R counter

### 2. Memory Decay & Consolidation
**Trigger**: Automatic when memory limits exceeded

**What it does:**
1. Detects when scope memory count exceeds limit
2. Selects oldest memories for consolidation
3. Uses AI to create condensed summary
4. Deletes original entries, stores summary

**Visualization:**
- Log shows: `⚙ AUTONOMOUS: DECAY scope=thread count=5`
- Followed by: `⚙ AUTONOMOUS: CONSOLIDATION scope=thread count=5`
- Cells spawn and decay in relevant zone
- D and C counters increment

### 3. Profile Background Updates
**Trigger**: Scheduled job every `PROFILE_REFRESH_MINUTES`

**What it does:**
1. Finds recently active users
2. Checks if profile needs refresh
3. Analyzes message patterns, style, preferences
4. Updates user interaction profiles

**Visualization:**
- Log shows: `⚙ AUTONOMOUS: ASSESSMENT scope=user count=3`
- Activity in user zone
- A counter increments

### 4. Auto System Prompt Generation
**Trigger**: Scheduled job every `AUTO_SYSTEM_PROMPT_INTERVAL_HOURS`

**What it does:**
1. Gathers context from recent conversations
2. Analyzes conversation metrics and user base
3. Generates tailored system prompt
4. Persists new prompt to config

**Visualization:**
- Log shows: `⚙ AUTONOMOUS: ASSESSMENT scope=global count=1`
- Activity in global zone
- A counter increments

## Technical Details

### Architecture

**Thread-safe design:**
- Visualizer runs in separate daemon thread
- Queue-based communication (max 1000 items)
- Non-blocking updates prevent bot slowdown

**Memory management:**
- Recall log: Last 100 entries (deque)
- Autonomous ops log: Last 50 entries (deque)
- Automatic cell cleanup via decay map

**Performance:**
- 50ms refresh rate (20 FPS)
- Graceful degradation if curses unavailable
- Minimal CPU impact (<1%)

### Integration Points

The visualizer hooks into these bot functions:

1. `recall_memories_async()` - Memory recalls
2. `_store_memories_with_decay()` - Memory storage
3. `_maybe_decay_memories_async()` - Memory decay
4. `hierarchical_memory_rollup()` - Rollup operations
5. `profile_background_job()` - Profile updates
6. `autosystem_job()` - System prompt generation

## Troubleshooting

### Visualizer not starting
1. Check if curses is available: `python -c "import curses; print('OK')"`
2. Verify `MEMORY_VISUALIZER_ENABLED=true` in `.env`
3. Check terminal supports Unicode characters

### Missing autonomous operations
1. Ensure jobs are scheduled (check bot startup logs)
2. Verify intervals are not too long for testing
3. Generate activity to trigger memory limits

### Display issues
1. Increase terminal size (minimum 80x24 recommended)
2. Check terminal supports 256 colors
3. Use terminal with UTF-8 support

## Advanced Customization

### Modify Intervals for Development

For faster testing, reduce intervals in `.env`:

```bash
# Faster rollups (1 minute)
SUMMARY_ROLLUP_INTERVAL_SECONDS=60

# Faster profile updates (5 minutes)
PROFILE_REFRESH_MINUTES=5

# Faster system prompt (30 minutes)
AUTO_SYSTEM_PROMPT_INTERVAL_HOURS=0.5
```

### Trigger Lower Memory Limits

Force more frequent consolidation:

```bash
MEMORY_THREAD_LIMIT=5
MEMORY_USER_LIMIT=10
MEMORY_GLOBAL_LIMIT=15
```

## API Reference

### Logging Functions

```python
from memory_visualizer import log_recall, log_operation, log_autonomous

# Log a memory recall
log_recall(
    scope='user',           # 'thread', 'user', or 'global'
    category='user_trait',  # Memory category
    content='User prefers technical explanations',
    weight=0.85,            # 0.0 - 1.0
    metadata={'tags': ['preference']}
)

# Log a memory operation
log_operation(
    'store',
    scope='thread',
    category='thread_note'
)

# Log an autonomous operation
log_autonomous(
    'consolidation',        # 'rollup', 'decay', 'consolidation', 'assessment'
    scope='user',           # Scope affected
    count=5,                # Number of items processed
    details='Consolidated thread memories'
)
```

## Philosophy

The visualizer provides a window into the bot's "consciousness" - showing not just what it remembers, but how it autonomously manages, consolidates, and evolves its memory over time. Watch as:

- **Memories bloom**: New recalls spawn as living cells
- **Patterns emerge**: High-value memories persist and glow
- **Decay happens**: Old memories fade naturally
- **Consolidation flows**: Information compresses into summaries
- **Self-modification occurs**: The system adapts autonomously

This creates a mesmerizing display of artificial cognition in action.

## Future Enhancements

Potential additions:
- [ ] Graph view of memory relationships
- [ ] Heatmap overlay for memory access patterns
- [ ] Timeline scrubber for historical playback
- [ ] Export visualization to video
- [ ] Web-based dashboard option
- [ ] Sound effects for autonomous operations
- [ ] Multi-bot visualization support

---

**Press 'q' to exit the visualizer. The bot's consciousness continues evolving...**
