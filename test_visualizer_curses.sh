#!/bin/bash
# Test script to run the memory visualizer in curses mode
# This ensures proper terminal handling

# Check if running in a terminal
if [ -t 0 ]; then
    echo "Running memory visualizer in curses mode..."
    echo "Press 'q' to quit"
    echo ""
    python3 memory_visualizer.py
else
    echo "ERROR: Not running in a terminal!"
    echo "Please run this script directly in a terminal:"
    echo "  bash test_visualizer_curses.sh"
    exit 1
fi
