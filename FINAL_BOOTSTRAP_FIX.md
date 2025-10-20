# Final Bootstrap Fix - Retry Until All Deps Install

## Problem
The `.bootstrapped` flag was created even when packages failed to install, so it would never retry on subsequent runs.

## Solution
**Only create `.bootstrapped` flag when ALL packages install successfully.**

## Changes Made

### bot_server.py - `ensure_venv_and_deps()` function (lines 129-163)

**Before:**
```python
subprocess.check_call([py, "-m", "pip", "install", *reqs])
# Always creates flag even if install fails
boot_flag.write_text(...)
```

**After:**
```python
# Track failures
core_failed = False
failed = []

# Install core
try:
    subprocess.check_call([py, "-m", "pip", "install", *reqs])
except:
    core_failed = True

# Install tools
for pkg in tool_reqs:
    try:
        subprocess.check_call([py, "-m", "pip", "install", pkg])
    except:
        failed.append(pkg_name)

# Only create flag if EVERYTHING succeeded
if not core_failed and not failed:
    boot_flag.write_text(...)
    print("✓ All dependencies installed successfully!")
else:
    print("⚠ Bootstrap incomplete - will retry on next run")
```

## How It Works Now

### Run 1 - Some Packages Fail
```bash
python3 bot_server.py

[bootstrap] Creating .venv ...
[bootstrap] Installing core requirements ...
[bootstrap]   ✓ Core requirements installed
[bootstrap] Installing tool requirements ...
[bootstrap]   ✓ selenium installed
[bootstrap]   ✗ ollama failed: <error>
[bootstrap]   ✓ beautifulsoup4 installed
[bootstrap] ⚠ 1 packages failed: ollama
[bootstrap] ⚠ Bootstrap incomplete - will retry on next run
[bootstrap] Re-exec in .venv ...

# Bot starts but tools partially available
```

### Run 2 - Retries Failed Packages
```bash
python3 bot_server.py

[bootstrap] Installing core requirements ...
[bootstrap]   ✓ Core requirements installed
[bootstrap] Installing tool requirements ...
[bootstrap]   ✓ selenium installed
[bootstrap]   ✓ ollama installed  # ← Retried and succeeded
[bootstrap]   ✓ beautifulsoup4 installed
[bootstrap] ✓ All dependencies installed successfully!
[bootstrap] Re-exec in .venv ...

# Now .bootstrapped flag is created
# Future runs skip bootstrap
```

### Run 3+ - Skip Bootstrap
```bash
python3 bot_server.py

# .bootstrapped exists, skips directly to re-exec
[bootstrap] Re-exec in .venv ...

# Fast startup, all deps available
```

## Key Features

✅ **Retries on every run** until all packages install
✅ **Tracks both core and tool failures** separately
✅ **Clear status messages** showing what failed
✅ **Only creates flag on success** - ensures complete installation
✅ **Idempotent** - safe to run multiple times

## Testing

### Clean start test
```bash
rm -rf .venv
python3 bot_server.py
# Should install everything or show clear errors
```

### Force retry test
```bash
rm .venv/.bootstrapped
python3 bot_server.py
# Will re-run bootstrap even if venv exists
```

### Simulate failure recovery
```bash
# Manually break something
rm .venv/.bootstrapped

# Run - will detect missing and retry
python3 bot_server.py
```

## Status Messages

### Success
```
[bootstrap] ✓ All dependencies installed successfully!
```

### Partial Failure
```
[bootstrap] ⚠ 2 packages failed: ollama, selenium
[bootstrap] ⚠ Bootstrap incomplete - will retry on next run
```

### Core Failure
```
[bootstrap]   ✗ Core requirements FAILED: <error>
[bootstrap]   Will retry on next run
```

## Summary

**The bootstrap now retries on every run until ALL dependencies are successfully installed.**

No more one-shot failures. It will keep trying until everything works.
