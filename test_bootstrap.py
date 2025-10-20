#!/usr/bin/env python3
"""Test that bot_server auto-generates requirements-tools.txt"""

import sys
from pathlib import Path

# Remove existing file to test generation
req_file = Path("requirements-tools.txt")
if req_file.exists():
    print(f"Removing existing {req_file.name}...")
    req_file.unlink()

# Import bot_server (should trigger auto-generation)
print("Importing bot_server...")
try:
    import bot_server
    print("✓ bot_server imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Check if file was created
if req_file.exists():
    print(f"✓ {req_file.name} was auto-generated!")
    print(f"\nContent preview:")
    print(req_file.read_text()[:200] + "...")
else:
    print(f"✗ {req_file.name} was NOT generated")
    sys.exit(1)
