#!/bin/bash
# Test the embedded bootstrap auto-installer

echo "=========================================="
echo "Testing Bootstrap Auto-Installer"
echo "=========================================="
echo ""

# Remove marker to trigger fresh install
if [ -f ".deps_installed" ]; then
    echo "Removing .deps_installed marker to trigger auto-install..."
    rm .deps_installed
fi

echo ""
echo "Testing dependency check (will auto-install missing packages)..."
echo ""

# Just import bot_server - the bootstrap will run
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# This will trigger auto_install_dependencies() on import
import bot_server

print("\n✓ Bootstrap completed successfully!")
EOF

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
