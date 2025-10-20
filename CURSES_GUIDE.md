# Memory Visualizer Curses Interface Guide

## Quick Fix: Enable Curses Mode NOW

Run the bot in tmux with curses enabled:

```bash
./run_bot_with_curses.sh
```

That's it! The curses visualizer will start automatically.

---

## Why isn't the curses interface working?

The memory visualizer has two modes:

1. **Console Mode** (fallback) - Simple text logging to stdout
2. **Curses Mode** - Full-screen TUI with Game of Life-style animation

By default, when `bot_server.py` runs, it detects that stdout is not a TTY (terminal) and uses console mode. This is expected behavior for server processes.

## How to Enable Curses Mode

### Option 1: Run in tmux or screen (Recommended)

The curses interface needs a proper terminal. Use tmux or screen:

```bash
# Using tmux
tmux new-session -s bot
python3 bot_server.py

# Detach: Ctrl+b then d
# Reattach: tmux attach -t bot
```

```bash
# Using screen
screen -S bot
python3 bot_server.py

# Detach: Ctrl+a then d
# Reattach: screen -r bot
```

Then the visualizer will automatically use curses mode!

### Option 2: Force Curses Mode

Set the `FORCE_CURSES` environment variable or config option:

**Environment variable:**
```bash
FORCE_CURSES=true python3 bot_server.py
```

**Or in .config file:**
```ini
FORCE_CURSES=true
```

**Warning:** Only use `FORCE_CURSES=true` when running inside tmux/screen. Otherwise, it will crash!

### Option 3: Test Standalone

Run the visualizer standalone to see the curses interface:

```bash
# Direct run
python3 memory_visualizer.py

# Or using the test script
./test_visualizer_curses.sh
```

Press `q` to quit.

## Curses Interface Features

When running in curses mode, you'll see:

- **Title Bar**: Shows sleep state and animation frame
- **Memory Grid**: Game of Life-style visualization with zones:
  - Left: THREAD scope memories
  - Center: USER scope memories
  - Right: GLOBAL scope memories
- **Stats Panel**: Memory counts and autonomous operation counts
- **Recall Log**: Most recent memory recall or autonomous operation
- **Visual Effects**:
  - `#` = High-weight memories
  - `=` `-` `.` = Medium to low weight
  - `*` = Active cells
  - `*#@%&` = Autonomous operations (animated)

## Controls

- `q` or `Q` - Quit the visualizer
- Window resizing is automatically handled

## Troubleshooting

**"stdout is not a TTY"**
- Solution: Run in tmux/screen, or set `FORCE_CURSES=true`

**Garbled display**
- Your terminal might not support curses properly
- Try: `export TERM=xterm-256color`
- Falls back to console mode automatically if curses fails

**No visualization at all**
- Check: `MEMORY_VISUALIZER_ENABLED=true` in config
- Check: curses module is installed (it should be by default)

## Architecture

The visualizer runs in a separate daemon thread and doesn't block the main bot. Updates are sent via a queue, so memory operations are logged asynchronously. If the visualizer crashes, the bot continues running normally.
