#!/bin/bash
# Launch bot with curses visualizer in tmux

SESSION_NAME="tg-bot"

echo "=========================================="
echo "Telegram Bot with Curses Visualizer"
echo "=========================================="
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed"
    echo ""
    echo "Install tmux:"
    echo "  Ubuntu/Debian: sudo apt-get install tmux"
    echo "  Fedora/RHEL:   sudo dnf install tmux"
    echo "  macOS:         brew install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠  Session '$SESSION_NAME' already exists"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo ""
    read -p "Kill existing session and start new? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "✓ Killed existing session"
    else
        echo "Attaching to existing session..."
        tmux attach -t "$SESSION_NAME"
        exit 0
    fi
fi

echo "Starting bot in tmux session '$SESSION_NAME'..."
echo ""
echo "Tmux commands:"
echo "  Detach:  Ctrl+b then d"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Kill:    tmux kill-session -t $SESSION_NAME"
echo ""
echo "Visualizer commands:"
echo "  Quit:    Press 'q'"
echo ""

# Start tmux session with bot
tmux new-session -s "$SESSION_NAME" -d "python3 bot_server.py"

# Wait a moment for the session to start
sleep 1

# Check if the session is running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✓ Bot started in tmux session '$SESSION_NAME'"
    echo ""
    echo "Attaching to session now..."
    sleep 1
    tmux attach -t "$SESSION_NAME"
else
    echo "❌ Failed to start bot"
    exit 1
fi
