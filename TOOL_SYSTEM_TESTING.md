# Tool System Testing Guide

## Overview

This guide explains how to test the tool exposure system that was integrated into the bot. The system allows admin users to view and use tools from `tools.py` through the AI model.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool System Architecture                  │
└─────────────────────────────────────────────────────────────┘

User sends /tools command (admin only)
        ↓
bot_server.py: tools_cmd handler
        ↓
tool_schema.py: ToolSchemaGenerator.generate_all_schemas()
        ├── Imports tools.py → Tools class
        ├── Scans all methods using inspect
        ├── Extracts docstrings with DocstringParser
        ├── Generates JSON schemas with type hints
        └── Returns List[ToolSchema]
        ↓
tool_schema.py: ToolFormatter.format_for_prompt()
        ├── Groups schemas by category
        ├── Formats with parameters and descriptions
        └── Returns formatted string
        ↓
User receives formatted tool catalog
```

## Files Modified/Created

### 1. **tool_schema.py** (NEW - 411 lines)
Core schema generation system modeled after Claude Code's architecture.

**Key Classes:**
- `DocstringParser` - Parses Google/NumPy style docstrings
- `ToolSchemaGenerator` - Generates JSON schemas from Python functions
- `ToolFormatter` - Formats tools for AI prompts (Claude Code style)
- `ParsedDocstring` - Structured docstring data
- `ToolSchema` - Complete tool metadata

**Key Functions:**
```python
# Parse function docstrings
DocstringParser.parse(docstring) -> ParsedDocstring

# Generate schema from function
ToolSchemaGenerator.generate_schema(func, category) -> ToolSchema

# Generate all tool schemas
ToolSchemaGenerator.generate_all_schemas() -> List[ToolSchema]

# Format for AI prompt
ToolFormatter.format_for_prompt(schemas, categories, max_tools) -> str

# Compact format
ToolFormatter.format_compact(schemas, max_per_category) -> str
```

### 2. **tool_integration.py** (MODIFIED)
Added integration with schema system.

**Changes:**
```python
# Added imports
from tool_schema import ToolSchemaGenerator, ToolFormatter

# Enhanced function
def build_tool_context_for_prompt(
    user_id: int,
    admin_whitelist: set,
    include_examples: bool = True,
    compact: bool = False,
    max_tools: Optional[int] = None
) -> str:
    # Check admin status
    if user_id not in admin_whitelist:
        return ""

    # Use schema-based formatting
    if TOOL_SCHEMA_AVAILABLE:
        schemas = ToolSchemaGenerator.generate_all_schemas()
        return ToolFormatter.format_for_prompt(schemas, ...)
```

### 3. **bot_server.py** (MODIFIED)
Enhanced `/tools` command with schema system.

**Changes:**
```python
# Lines 113-124: Added imports
from tool_integration import ToolInspector, build_tool_context_for_prompt
from tool_schema import ToolSchemaGenerator, ToolFormatter

# Lines 1438-1505: Enhanced /tools command
@admin_only
async def tools_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Supports multiple modes:
    # /tools          - Full tool catalog
    # /tools compact  - Compact listing
    # /tools web      - Only web tools
    # /tools browser  - Only browser tools
    # etc.
```

## Prerequisites

### Required Dependencies

The tools system requires these Python packages:

```bash
pip install beautifulsoup4 selenium duckduckgo-search
```

**Why these are needed:**
- `beautifulsoup4` (bs4) - HTML parsing in web scraping tools
- `selenium` - Browser automation tools
- `duckduckgo-search` - Web search functionality

### Verify tools.py Can Be Imported

```bash
python3 << 'EOF'
try:
    from tools import Tools
    print(f"✓ Tools class imported successfully")
    print(f"✓ Available tool methods: {len([a for a in dir(Tools) if not a.startswith('_')])}")
except ImportError as e:
    print(f"✗ Failed to import Tools: {e}")
EOF
```

## Testing the /tools Command

### Test 1: Basic Tool Listing (Admin Only)

**As an admin user in Telegram:**
```
/tools
```

**Expected Output:**
```
================================================================================
AVAILABLE TOOLS
================================================================================

You have access to 50 tools across 5 categories.
To use a tool, emit a function call in this format:

  Tools.tool_name(arg1='value1', arg2='value2')

For async tools, the system will automatically await them.

━━━ BROWSER TOOLS (8) ━━━

Tool: open_browser  [async]
Description: Open a headless Chrome browser instance
Parameters:
  • headless: boolean (optional)
    Run browser in headless mode
  • proxy: string (optional)
    Proxy server URL

Tool: navigate  [async]
Description: Navigate to a URL in the browser
Parameters:
  • url: string (required)
    The URL to navigate to

[... more tools ...]

━━━ WEB TOOLS (12) ━━━

Tool: search_internet  [async]
Description: Search the internet using DuckDuckGo
Parameters:
  • query: string (required)
    Search query
  • num_results: integer (optional)
    Number of results to return (default 5)

[... etc ...]

================================================================================
USAGE NOTES:
• Always check parameter requirements before calling
• String parameters should be quoted: 'value' or "value"
• For file operations, paths are relative to workspace unless absolute
• Web searches automatically extract and summarize content
• Browser tools require Chrome/Chromium to be installed
================================================================================
```

### Test 2: Compact Listing

**Command:**
```
/tools compact
```

**Expected Output:**
```
AVAILABLE TOOLS (compact view):

BROWSER:
  • open_browser([headless], [proxy])
  • navigate(url)
  • click(selector)
  • screenshot([filename])
  • close_browser()

WEB:
  • search_internet(query, [num_results])
  • fetch_webpage(url)
  • scrape_links(url)

[... etc ...]
```

### Test 3: Category Filtering

**Command:**
```
/tools web
```

**Expected Output:**
Only shows tools in the "web" category with full details.

**Valid Categories:**
- `web` - Web search and fetching
- `browser` - Browser automation
- `filesystem` - File operations
- `system` - System utilities
- `ai` - AI inference helpers
- `general` - Uncategorized tools

### Test 4: Non-Admin User

**As a non-admin user:**
```
/tools
```

**Expected Output:**
```
🚫 This command is restricted to administrators.
```

### Test 5: Missing Dependencies

**If tools.py dependencies aren't installed:**
```
/tools
```

**Expected Output:**
```
⚠️ Tool catalog unavailable.

Tool integration modules are not loaded. This usually means:
• tools.py is missing required dependencies (bs4, selenium, etc.)
• tool_integration.py or tool_schema.py are not in the correct location

To enable tools, ensure all dependencies are installed:
pip install beautifulsoup4 selenium duckduckgo_search
```

## Testing Tool Schema Generation

### Programmatic Test

Create a test script:

```python
#!/usr/bin/env python3
"""Test tool schema generation"""

from tool_schema import ToolSchemaGenerator, ToolFormatter, DocstringParser

# Test 1: Generate all schemas
print("=" * 80)
print("TEST 1: Generate All Tool Schemas")
print("=" * 80)

schemas = ToolSchemaGenerator.generate_all_schemas()
print(f"\n✓ Generated {len(schemas)} tool schemas")

if schemas:
    # Group by category
    by_cat = {}
    for s in schemas:
        by_cat.setdefault(s.category, []).append(s)

    print(f"✓ Categories found: {sorted(by_cat.keys())}")
    for cat, tools in sorted(by_cat.items()):
        print(f"  • {cat}: {len(tools)} tools")

    # Test 2: Format for prompt
    print("\n" + "=" * 80)
    print("TEST 2: Format for AI Prompt (Full)")
    print("=" * 80)

    full_format = ToolFormatter.format_for_prompt(schemas[:5])  # First 5 tools
    print(full_format)

    # Test 3: Compact format
    print("\n" + "=" * 80)
    print("TEST 3: Compact Format")
    print("=" * 80)

    compact_format = ToolFormatter.format_compact(schemas, max_per_category=3)
    print(compact_format)

    # Test 4: Category filtering
    print("\n" + "=" * 80)
    print("TEST 4: Category Filtering (Web Tools Only)")
    print("=" * 80)

    web_format = ToolFormatter.format_for_prompt(schemas, categories=["web"])
    print(web_format)

    # Test 5: Docstring parsing
    print("\n" + "=" * 80)
    print("TEST 5: Docstring Parsing")
    print("=" * 80)

    sample_docstring = """
    Search the internet using DuckDuckGo

    This function performs a web search and returns structured results.

    Args:
        query (str): The search query to execute
        num_results (int): Number of results to return (default 5)

    Returns:
        List of search results with titles and URLs

    Examples:
        Tools.search_internet('Python asyncio tutorial', num_results=3)
    """

    parsed = DocstringParser.parse(sample_docstring)
    print(f"Summary: {parsed.summary}")
    print(f"Args: {parsed.args}")
    print(f"Returns: {parsed.returns}")
    print(f"Examples: {parsed.examples}")

else:
    print("✗ No tools found. Check if tools.py can be imported.")
    print("\nTry:")
    print("  pip install beautifulsoup4 selenium duckduckgo-search")

print("\n" + "=" * 80)
print("TESTS COMPLETE")
print("=" * 80)
```

**Run the test:**
```bash
python3 test_tool_schema.py
```

## Testing Tool Context Injection (AI Prompts)

The tool system should inject tool context into AI prompts for admin users.

### Test Script

```python
#!/usr/bin/env python3
"""Test tool context injection"""

from tool_integration import build_tool_context_for_prompt

# Admin whitelist (replace with actual admin IDs)
ADMIN_WHITELIST = {123456789, 987654321}

# Test 1: Admin user (should get tools)
print("TEST 1: Admin User")
print("=" * 80)
admin_context = build_tool_context_for_prompt(
    user_id=123456789,
    admin_whitelist=ADMIN_WHITELIST,
    compact=False,
    max_tools=10
)
print(f"Length: {len(admin_context)} characters")
print(admin_context[:500] if admin_context else "(empty)")

# Test 2: Non-admin user (should get nothing)
print("\n\nTEST 2: Non-Admin User")
print("=" * 80)
non_admin_context = build_tool_context_for_prompt(
    user_id=111111111,  # Not in whitelist
    admin_whitelist=ADMIN_WHITELIST
)
print(f"Length: {len(non_admin_context)} characters")
print("Result:", repr(non_admin_context))
assert non_admin_context == "", "Non-admin should get empty context!"
print("✓ Correctly blocked non-admin")

# Test 3: Compact mode
print("\n\nTEST 3: Compact Mode")
print("=" * 80)
compact_context = build_tool_context_for_prompt(
    user_id=123456789,
    admin_whitelist=ADMIN_WHITELIST,
    compact=True
)
print(compact_context)
```

## Integration with AI Message Handler

The tool context should be injected when admin users send messages to the AI.

### Check on_text Handler

Look for this pattern in [bot_server.py:3646](bot_server.py#L3646):

```python
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... existing code ...

    # Build prompt
    user_id = update.effective_user.id

    # Check if tool context should be included
    if BRIDGE_TOOLS_AVAILABLE and EnhancedPromptBuilder:
        prompt = EnhancedPromptBuilder.inject_tool_context(
            base_prompt,
            user_id=user_id,
            admin_whitelist=ADMIN_WHITELIST
        )
    else:
        prompt = base_prompt

    # Send to Ollama
    # ...
```

**To test:**
1. Send a message as an admin user
2. The AI should be aware of available tools
3. Ask the AI: "What tools do you have access to?"
4. The AI should list the tools from the injected context

## Troubleshooting

### Issue: "Tool catalog unavailable"

**Cause:** Dependencies missing or import failed.

**Solution:**
```bash
pip install beautifulsoup4 selenium duckduckgo-search
python3 -c "from tools import Tools; print('OK')"
```

### Issue: "No tools registered"

**Cause:** Tools class exists but has no methods.

**Solution:** Verify tools.py has the Tools class with methods:
```bash
python3 << 'EOF'
from tools import Tools
print(f"Tools found: {[m for m in dir(Tools) if not m.startswith('_')]}")
EOF
```

### Issue: Tools show but have no descriptions

**Cause:** Docstrings missing or not in Google/NumPy style.

**Solution:** Add proper docstrings to tools.py functions:
```python
def search_internet(query: str, num_results: int = 5):
    """
    Search the internet using DuckDuckGo

    Args:
        query: The search query to execute
        num_results: Number of results to return (default 5)

    Returns:
        List of search results with titles, URLs, and snippets
    """
```

### Issue: Non-admin users see tools

**Cause:** Admin decorator not applied.

**Solution:** Verify `@admin_only` decorator is on tools_cmd:
```python
@admin_only
async def tools_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ...
```

## Expected Behavior Summary

| User Type | Command | Result |
|-----------|---------|--------|
| Admin | `/tools` | Full tool catalog with descriptions |
| Admin | `/tools compact` | Compact tool list |
| Admin | `/tools web` | Only web category tools |
| Non-Admin | `/tools` | "Restricted to administrators" |
| Any | `/tools` (deps missing) | Installation instructions |

## Next Steps

After verifying the `/tools` command works:

1. **Test tool execution**: Verify tools can actually be called by the AI
2. **Test error handling**: Try invalid tool calls to check retry logic
3. **Test DAG workflows**: Create multi-tool workflows
4. **Add logging**: Track which tools are called and by whom
5. **Add usage analytics**: Monitor tool usage patterns

## Related Files

- [tool_schema.py](tool_schema.py) - Schema generation system
- [tool_integration.py](tool_integration.py) - Tool discovery and execution
- [ai_tool_bridge.py](ai_tool_bridge.py) - AI prompt building and tool parsing
- [tools.py](tools.py) - Actual tool implementations
- [TOOL_INTEGRATION_GUIDE.md](TOOL_INTEGRATION_GUIDE.md) - Integration documentation
- [TOOL_SYSTEM_SUMMARY.md](TOOL_SYSTEM_SUMMARY.md) - Implementation summary

## Success Criteria

✅ **Phase 1 Complete (Current):**
- [x] tool_schema.py created with docstring parsing
- [x] ToolSchemaGenerator generates proper JSON schemas
- [x] ToolFormatter formats in Claude Code style
- [x] /tools command shows full tool catalog for admins
- [x] /tools compact shows abbreviated listing
- [x] Category filtering works (/tools web, etc.)
- [x] Non-admins are blocked from seeing tools
- [x] Graceful handling of missing dependencies

⏳ **Phase 2 (Pending):**
- [ ] Tool context injected into AI prompts
- [ ] AI can request tool execution
- [ ] Tools are actually executed with results
- [ ] Retry logic works for failed tools
- [ ] DAG workflows function correctly

⏳ **Phase 3 (Future):**
- [ ] Tool usage logging
- [ ] Tool execution analytics
- [ ] Per-tool permissions
- [ ] Tool execution sandboxing
- [ ] Tool result caching
