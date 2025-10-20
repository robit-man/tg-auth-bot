# Bootstrap Auto-Install - Complete Implementation

## What Was Fixed

The bootstrap system now **automatically installs ALL dependencies** on first run, embedded directly in `bot_server.py` with no external files needed.

## Key Changes

### 1. Import Order Fixed
**Before**: Third-party imports happened BEFORE dependency check
**After**: Bootstrap runs FIRST, then imports

```python
# bot_server.py structure:
1. Shebang and docstring
2. Import ONLY stdlib (no third-party)
3. Define DEPENDENCIES dict (single source of truth)
4. Define auto_install_dependencies() function
5. RUN auto_install_dependencies() ← RUNS NOW
6. Generate requirements.txt (optional reference file)
7. NOW import third-party modules (telegram, requests, etc.)
```

### 2. Single Source of Truth

All dependencies defined in ONE place:

```python
DEPENDENCIES = {
    'beautifulsoup4': 'bs4',
    'lxml': 'lxml',
    'requests': 'requests',
    'selenium': 'selenium',
    'webdriver-manager': 'webdriver_manager',
    'ollama': 'ollama',
    'python-telegram-bot': 'telegram',
    'python-dotenv': 'dotenv',
}
```

### 3. Auto-Install Logic

```python
def auto_install_dependencies():
    # Skip if .deps_installed marker exists
    # Check each package in DEPENDENCIES
    # Install missing ones via pip
    # Create marker file when done
```

## How It Works

### First Run
```
1. User runs: python3 bot_server.py
2. Bootstrap detects no .deps_installed marker
3. Checks each dependency
4. Installs missing packages via pip
5. Creates .deps_installed marker
6. Continues with bot startup
```

### Subsequent Runs
```
1. User runs: python3 bot_server.py
2. Bootstrap finds .deps_installed marker
3. Skips dependency check (instant)
4. Continues with bot startup
```

## File Structure

### bot_server.py
```
Lines 1-10:   Shebang, docstring
Lines 11-20:  Stdlib imports only
Lines 22-34:  DEPENDENCIES dict
Lines 36-106: auto_install_dependencies() function
Line 109:     RUN bootstrap
Lines 112-118: Generate requirements.txt
Line 120+:    Third-party imports
```

### No External Dependencies
- ❌ No requirements-tools.txt needed (auto-generated for reference only)
- ❌ No auto_install_deps.py needed
- ❌ No install_tool_deps.sh needed
- ✅ Everything embedded in bot_server.py

## What Gets Installed

All packages in DEPENDENCIES dict:
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser
- `requests` - HTTP library
- `selenium` - Browser automation
- `webdriver-manager` - ChromeDriver management
- `ollama` - LLM client
- `python-telegram-bot` - Telegram bot framework
- `python-dotenv` - Environment variables

## Marker File

`.deps_installed` - Created after first successful run
- Prevents re-checking on every startup
- Delete to force re-check: `rm .deps_installed`

## Output Example

### First Run
```
======================================================================
[bootstrap] FIRST RUN - Checking dependencies...
======================================================================
[bootstrap] ✗ beautifulsoup4 - MISSING
[bootstrap] ✗ lxml - MISSING
[bootstrap] ✓ requests
[bootstrap] ✗ selenium - MISSING
...

[bootstrap] Installing missing packages...
[bootstrap] This will take a few minutes...

[bootstrap]   Installing beautifulsoup4...
[bootstrap]   ✓ beautifulsoup4 installed
[bootstrap]   Installing lxml...
[bootstrap]   ✓ lxml installed
...

======================================================================
[bootstrap] All dependencies installed successfully!
[bootstrap] ✓ You can now use all bot features
======================================================================
```

### Second Run
```
(no output - marker exists, bootstrap skipped)
```

## Benefits

✅ **Zero manual setup** - Just run the bot
✅ **Self-contained** - No external files needed
✅ **Idempotent** - Safe to run multiple times
✅ **Fast** - Only checks/installs on first run
✅ **Maintainable** - Single source of truth
✅ **Transparent** - Clear output showing what's happening

## Error Handling

### If pip install fails
```
[bootstrap]   ✗ package FAILED
[bootstrap]      Error message here...
[bootstrap] Installed 6/8 packages
[bootstrap] Some packages failed - check errors above
```

Bot will still start but missing features won't work.

### Manual fix
```bash
# Remove marker and try again
rm .deps_installed
python3 bot_server.py

# Or install manually
pip install package-name
```

## Maintenance

### Adding a new dependency
1. Edit DEPENDENCIES dict in bot_server.py
2. Add: `'package-name': 'import_name',`
3. Done - will auto-install on next run (if marker deleted)

### Removing a dependency
1. Remove from DEPENDENCIES dict
2. Optionally uninstall: `pip uninstall package-name`

### Force re-check
```bash
rm .deps_installed
python3 bot_server.py
```

## Cross-File Context

### tools.py
Requires: beautifulsoup4, lxml, requests, selenium, webdriver-manager, ollama
Status: ✅ All auto-installed by bootstrap

### ai_tool_bridge.py
Imports: tools.py
Status: ✅ Will work after bootstrap

### memory_visualizer.py
Requires: curses (stdlib)
Status: ✅ No external deps needed

## Testing

### Test bootstrap
```bash
# Remove marker
rm .deps_installed

# Run bot - should auto-install
python3 bot_server.py
```

### Verify tools work
```bash
python3 -c "from tools import Tools; print('✓ Tools work')"
```

### Check all deps
```bash
python3 << 'EOF'
import bs4, lxml, requests, selenium, ollama, telegram, dotenv
print("✓ All dependencies available")
EOF
```

## Summary

**The entire dependency installation process is now embedded in bot_server.py bootstrap section.**

No external files. No manual steps. Just run the bot.

**That's it.**
