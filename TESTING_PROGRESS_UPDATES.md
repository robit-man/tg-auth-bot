# Testing Guide: Real-Time Progress Updates

## Quick Start

### 1. Start the Bot

```bash
cd /home/robit/Respositories/tg-auth-bot
python3 bot_server.py
```

### 2. Send Test Message

**In Telegram (as admin user):**
```
Search for Python asyncio tutorials
```

### 3. Expected Behavior

#### Step 1: Tool Decision
You should see confirmation buttons:
```
🤖 Tool Decision

Tool: search_internet
Arguments: topic='Python asyncio tutorials', num_results=5
Confidence: 95.0%

Execute this tool?
[✅ Execute search_internet] [❌ Cancel] [📝 Direct Reply Instead]
```

#### Step 2: Click "Execute"

The message should start updating in real-time (every ~1 second):

```
⏳ search_internet

🌐 Opening browser (GUI mode)...
```

#### Step 3: Real-Time Updates Continue

```
⏳ search_internet

🌐 Opening browser (GUI mode)...
🔍 Navigating to DuckDuckGo...
✓ DuckDuckGo loaded
⌨️ Typing search query: 'Python asyncio tutorials'
🔎 Submitting search...
✓ Search results received
📋 Found 5 results, processing each...
```

#### Step 4: Browser Window Appears

You should see a **Chrome/Chromium browser window** open on your screen showing:
- DuckDuckGo homepage
- Search being typed
- Results loading
- Each result page opening in new tabs

#### Step 5: Progress Updates for Each Result

```
⏳ search_internet

📄 Result 1/5: Real Python - Async IO in Python...
🌐 Opening page in new tab: https://realpython.com...
⏳ Page loading...
⏱️ Waiting for page to stabilize (JavaScript execution)...
📸 Capturing page DOM and extracting text...
✓ Captured 45231 characters from page
🔙 Returned to search results
🤖 Generating AI summary of page content...
✓ Summary generated (523 chars)
🔍 Extracting key information for topic: 'Python asyncio tutorials'...
✓ Extracted key information (847 chars)
✅ Result 1 complete: Real Python - Async IO in Python...

📄 Result 2/5: Python.org Official Tutorial...
[... continues ...]
```

#### Step 6: Completion

```
⏳ search_internet

✅ Result 5 complete: Medium - Python Asyncio for Beginners

✨ Search complete! Processed 5/5 results successfully
🔒 Closing browser...
```

#### Step 7: Final Results

```
✅ Tool Executed Successfully

Tool: search_internet
Time: 45.2s

Result:
🔍 Search Results:

1. Real Python - Async IO in Python: A Complete Walkthrough
   Comprehensive guide to Python's asyncio library...
   https://realpython.com/async-io-python/

2. Python.org - Coroutines and Tasks
   Official Python documentation...
   https://docs.python.org/3/library/asyncio-task.html

[... 3 more results ...]

Rate this response:
[0] [1⭐] [2⭐⭐] [3⭐⭐⭐] [4⭐⭐⭐⭐] [5⭐⭐⭐⭐⭐]
```

## What to Watch For

### ✅ Success Indicators

1. **Browser Window Opens** - Visible Chrome/Chromium window
2. **Message Updates** - Telegram message edits every ~1 second
3. **Progress Emojis** - Clear indicators for each step
4. **Character Counts** - Shows how much content was captured
5. **Timing Information** - See how long each step takes
6. **Real URLs** - Actual website links in results
7. **Browser Closes** - Clean shutdown at the end

### ❌ Failure Indicators

1. **No browser window** - Check headless mode settings
2. **No message updates** - Progress callback not working
3. **"Tool not found" error** - Tools.py import issue
4. **Timeout errors** - Network or page load issues
5. **Empty results** - Check if DuckDuckGo is blocked

## Test Cases

### Test 1: Basic Search (Quick)

**Message:**
```
Search for Python
```

**Expected:**
- 1 result by default
- ~10-15 progress updates
- Completes in ~10 seconds
- Browser opens and closes

### Test 2: Multi-Result Search (Standard)

**Message:**
```
Search for machine learning tutorials and get 3 results
```

**Expected:**
- 3 results
- ~20-30 progress updates
- Completes in ~30-40 seconds
- Browser visible throughout

### Test 3: Complex Search (Comprehensive)

**Message:**
```
Search for "advanced Python asyncio patterns" and give me 5 detailed results
```

**Expected:**
- 5 results with full extraction
- ~35-50 progress updates
- Completes in ~60-90 seconds
- Each result fully processed

### Test 4: Error Handling

**Message:**
```
Search for asdfghjklqwertyuiopzxcvbnm12345
```

**Expected:**
- Progress updates still work
- Shows "Found 0 results" or minimal results
- No crashes
- Clean error messages

### Test 5: Direct Reply (No Tool)

**Message:**
```
What is 2 + 2?
```

**Expected:**
- No tool decision (direct reply)
- No browser opens
- Normal chat response
- No progress updates

## Debugging

### Check Bot Logs

```bash
# Watch for progress callback messages
grep "progress" bot.log

# Watch for tool execution
grep "search_internet" bot.log

# Check for errors
grep "ERROR" bot.log
```

### Enable Debug Logging

Add to bot_server.py:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Progress Callback Directly

```python
from tools import Tools

messages = []

def test_callback(msg):
    messages.append(msg)
    print(f"[{len(messages):2d}] {msg}")

results = Tools.search_internet(
    topic="Python",
    num_results=1,
    headless=False,
    progress_callback=test_callback
)

print(f"\nTotal messages: {len(messages)}")
print(f"Results: {len(results)}")
```

**Expected Output:**
```
[ 1] 🌐 Opening browser (GUI mode)...
[ 2] 🔍 Navigating to DuckDuckGo...
[ 3] ✓ DuckDuckGo loaded
[ 4] ⌨️ Typing search query: 'Python'
[ 5] 🔎 Submitting search...
[ 6] ✓ Search results received
[ 7] 📋 Found 1 results, processing each...
[ 8] 📄 Result 1/1: Python.org
[ 9] 🌐 Opening page in new tab...
[10] ⏳ Page loading...
[11] ⏱️ Waiting for page to stabilize...
[12] 📸 Capturing page DOM and extracting text...
[13] ✓ Captured 23456 characters from page
[14] 🔙 Returned to search results
[15] 🤖 Generating AI summary of page content...
[16] ✓ Summary generated (412 chars)
[17] 🔍 Extracting key information for topic: 'Python'...
[18] ✓ Extracted key information (734 chars)
[19] ✅ Result 1 complete: Python.org
[20] ✨ Search complete! Processed 1/1 results successfully
[21] 🔒 Closing browser...

Total messages: 21
Results: 1
```

## Common Issues

### Issue 1: Browser Doesn't Open

**Symptoms:**
- No browser window visible
- Progress shows "Opening browser..." but nothing happens

**Causes:**
- Running on headless server (no display)
- Missing Chrome/Chromium installation
- Display not available

**Fix:**
```bash
# Check if Chrome is installed
which google-chrome
which chromium-browser

# Check if display is available
echo $DISPLAY

# Install Chrome if missing
sudo apt-get install chromium-browser
```

### Issue 2: No Progress Updates

**Symptoms:**
- Message shows "Executing..." but no updates
- Browser works but Telegram message doesn't update

**Causes:**
- Progress callback not being passed
- Telegram message editing failing
- Rate limiting too aggressive

**Fix:**
```python
# Check in bot logs for:
# "[PROGRESS]" messages should appear

# If not, verify in tool_telegram_ui.py:
def sync_progress_callback(message: str):
    print(f"[PROGRESS] {message}")  # Add this for debugging
    progress_log.append(message)
    ...
```

### Issue 3: Updates Too Slow

**Symptoms:**
- Updates appear but delayed by several seconds

**Cause:**
- Rate limiting set to 1 second

**Fix:**
Reduce rate limit in tool_telegram_ui.py (line 236):
```python
# Change from 1.0 to 0.5 for faster updates
if current_time - last_update_time[0] >= 0.5:
```

### Issue 4: Message Too Long Error

**Symptoms:**
- Updates stop partway through
- Telegram error about message length

**Cause:**
- Too many progress messages accumulated

**Fix:**
Reduce message history in tool_telegram_ui.py (line 240):
```python
# Show last 10 instead of 20
full_progress = "\n".join(progress_log[-10:])
```

## Performance Metrics

### Expected Timing

| Operation | Time | Progress Updates |
|-----------|------|-----------------|
| Browser open | 2-3s | 2-3 |
| Search submission | 1-2s | 2-3 |
| Per result (basic) | 5-8s | 4-6 |
| Per result (with AI) | 10-15s | 8-12 |
| Total (1 result) | 10-15s | 10-15 |
| Total (5 results) | 50-75s | 40-60 |

### Update Rate

- **Max rate:** 1 update per second (Telegram limit)
- **Typical:** 1-2 updates per second
- **Per search:** 10-60 updates depending on num_results

### Network Usage

- **Message edits:** ~0.5 KB each
- **Total per search:** 5-30 KB (negligible)

## Success Criteria

✅ The implementation is successful if:

1. **Browser is visible** - Window opens on screen
2. **Progress updates in real-time** - Message edits every ~1 second
3. **All stages visible** - Can see navigation, capture, AI processing
4. **Character counts shown** - See how much content captured
5. **Errors are visible** - Warnings and errors clearly marked
6. **Results are real** - Actual URLs and content from web
7. **Browser closes cleanly** - No hanging processes

## Visual Checklist

While testing, you should see:

- [ ] Telegram message with confirmation buttons
- [ ] Message editing with progress emojis
- [ ] Browser window opening
- [ ] DuckDuckGo homepage loading
- [ ] Search query being typed
- [ ] Results page appearing
- [ ] Each result opening in new tab
- [ ] Content being captured
- [ ] Tab closing after each result
- [ ] AI summary progress indicators
- [ ] Browser closing at the end
- [ ] Final results with real URLs
- [ ] Rating buttons at bottom

If you see all of these, the implementation is **fully working**! ✅

## Next Steps After Testing

Once verified working:

1. **Adjust timing** - Tune rate limits and timeouts
2. **Add more tools** - Apply progress callbacks to other tools
3. **Enhance messages** - Add more detailed progress info
4. **Screenshots** - Capture and send screenshots of pages
5. **Interactive controls** - Add pause/resume buttons
6. **Analytics** - Track which steps take longest
7. **Optimization** - Speed up slow operations

## Documentation

See also:
- [SEARCH_TOOL_GUI_AND_PROGRESS.md](SEARCH_TOOL_GUI_AND_PROGRESS.md) - Full technical details
- [EXAMPLE_SEARCH_OUTPUT.md](EXAMPLE_SEARCH_OUTPUT.md) - Example output
- [PROGRESS_UPDATES_SUMMARY.md](PROGRESS_UPDATES_SUMMARY.md) - Implementation summary
- [BUGFIX_INSPECT_IMPORT.md](BUGFIX_INSPECT_IMPORT.md) - Bug fix documentation

Happy testing! 🚀
