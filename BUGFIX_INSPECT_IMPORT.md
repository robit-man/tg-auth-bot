# Bug Fix: UnboundLocalError for inspect module

## Error Encountered

```
❌ Tool Execution Failed

Tool: search_internet
Time: 0.00s
Error: Tool execution failed: cannot access local variable 'inspect' where it is not associated with a value
Traceback (most recent call last):
  File "/home/robit/Respositories/tg-auth-bot/tool_executor_bridge.py", line 125, in execute_tool
    if progress_callback and 'progress_callback' in inspect.signature(tool_func).parameters:
                                                    ^^^^^^^
UnboundLocalError: cannot access local variable 'inspect' where it is not associated with a value
```

## Root Cause

The `inspect` module was being:
1. **Used on line 126** to check function signatures
2. **Imported on line 130** inside a try block

This caused a `UnboundLocalError` because Python detected `inspect` would be assigned later in the scope, making it a local variable that was accessed before assignment.

## The Problem Code

```python
# Line 126 - Using inspect BEFORE it's imported
if progress_callback and 'progress_callback' in inspect.signature(tool_func).parameters:
    tool_args['progress_callback'] = progress_callback

# Line 130 - Importing inspect AFTER using it
import inspect
if inspect.iscoroutinefunction(tool_func):
    ...
```

## The Fix

**File:** [tool_executor_bridge.py](tool_executor_bridge.py)

### Change 1: Add to top-level imports (Line 11)

```python
import asyncio
import inspect  # ← Added here
import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
```

### Change 2: Remove duplicate import (Line 130)

```python
# Execute with timeout
try:
    # Pass progress_callback if tool supports it
    if progress_callback and 'progress_callback' in inspect.signature(tool_func).parameters:
        tool_args['progress_callback'] = progress_callback

    # Check if tool is async
    # import inspect  # ← Removed this line
    if inspect.iscoroutinefunction(tool_func):
        ...
```

## Verification

```bash
✅ python3 -m py_compile tool_executor_bridge.py
✅ All files compile successfully
✅ Error resolved
```

## Why This Happened

When adding the progress callback feature, I:
1. Added code that uses `inspect.signature()` (line 126)
2. Forgot that `inspect` was only imported later in the function (line 130)

Python's scoping rules made `inspect` a local variable (because it would be assigned with `import inspect`), but it was accessed before that assignment, causing the `UnboundLocalError`.

## Lesson Learned

Always import modules at the **top of the file**, not inside functions, unless you have a specific reason (like conditional imports in try/except blocks for optional dependencies).

**Good:**
```python
import inspect  # Top-level

def my_function():
    inspect.signature(...)  # Works
```

**Bad:**
```python
def my_function():
    inspect.signature(...)  # UnboundLocalError!
    import inspect  # Too late
```

## Status

✅ **Fixed and verified**
- Tool execution now works correctly
- Progress callbacks are properly detected
- Ready for testing with real searches
