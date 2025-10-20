# Quick Start: Curses Visualizer

## The Problem

You're seeing:
```
[visualizer:fallback] curses unavailable; logging to console only
```

This means the visualizer is using console fallback mode instead of the full curses TUI.

## The Solution (30 seconds)

**Step 1**: Install tmux (if not already installed)
```bash
# Ubuntu/Debian
sudo apt-get install tmux

# Fedora/RHEL
sudo dnf install tmux

# macOS
brew install tmux
```

**Step 2**: Run the bot with curses
```bash
./run_bot_with_curses.sh
```

**Done!** The curses visualizer will launch in a tmux session.

## What Changed

I've already configured everything for you:

✓ Added `FORCE_CURSES=true` to .env
✓ Added `MEMORY_VISUALIZER_ENABLED=true` to .env
✓ Created [run_bot_with_curses.sh](run_bot_with_curses.sh) launcher
✓ Fixed all curses rendering bugs (Unicode → ASCII)

## Using Tmux

**Detach from session** (leave bot running in background):
```
Ctrl+b then d
```

**Reattach to session**:
```bash
tmux attach -t tg-bot
```

**Kill session** (stop bot):
```bash
tmux kill-session -t tg-bot
```

**List sessions**:
```bash
tmux ls
```

## Visualizer Controls

Once in curses mode:
- **q** - Quit visualizer (keeps bot running)
- Window resizing is auto-detected

## What You'll See

```
+----------------------------------------------------------+
| * MEMORY CONSCIOUSNESS VISUALIZER *     [Frame: 1234]   |
|                                                          |
|     THREAD           USER            GLOBAL              |
|       *                #                =                |
|      #=               -*-              ...               |
|     .-,               ===              =#-               |
|                                                          |
| T:42 U:18 G:7 | Total:67 Active:15 | AUTO: R:3 D:1 ...  |
| [12:34:56] thread::note w=0.85 | Memory about context   |
+----------------------------------------------------------+
```

Features:
- Game of Life-style memory visualization
- Real-time memory recalls and autonomous operations
- ASCII art (works in any terminal)
- Color-coded by scope (thread/user/global)

## Troubleshooting

**Still seeing fallback mode?**
- Make sure you ran `./run_bot_with_curses.sh` (not `python3 bot_server.py` directly)
- Verify .env has `FORCE_CURSES=true`
- Check you're in a tmux session: `echo $TMUX` (should not be empty)

**Bot won't start?**
- Check logs in tmux: `tmux attach -t tg-bot`
- Or run directly: `python3 bot_server.py` (console mode)

**"tmux: command not found"**
- Install tmux (see Step 1 above)

## Alternative: Manual tmux

If you prefer to run manually:

```bash
# Start tmux
tmux

# Run bot (curses will auto-enable)
python3 bot_server.py

# Detach: Ctrl+b then d
# Reattach: tmux attach
```

That's it! Enjoy your curses visualizer!
