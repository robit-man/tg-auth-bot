# Tool Integration Guide - AI-Powered Tool Execution

## Overview

The bot now exposes powerful tools from `tools.py` to the AI when invoked by **admin users only**. This enables the AI to autonomously execute web searches, file operations, browser automation, and more through natural language requests.

## Architecture

```
User (Admin) → Telegram Message
    ↓
Bot Message Handler
    ↓
Enhanced Prompt Builder (ai_tool_bridge.py)
    ↓
AI with Tool Context (sees all available tools)
    ↓
AI Response (may include tool calls)
    ↓
Tool Request Parser
    ↓
Tool Executor (tool_integration.py)
    ↓
Tools (tools.py)
    ↓
Result → Formatted Response → User
```

## Key Components

### 1. Tool Integration (`tool_integration.py`)

**Core Classes:**
- `ToolInspector` - Discovers and introspects all available tools
- `ToolExecutor` - Executes tools with retry logic and error handling
- `DAGExecutor` - Orchestrates complex multi-tool workflows
- `ToolMetadata` - Rich metadata about each tool (params, docs, signatures)

**Features:**
- Automatic tool discovery from `Tools` class
- Runtime tool registration support
- Categorization (browser, web, filesystem, system, ai)
- Parameter introspection with type hints
- Async/sync tool detection

### 2. AI Tool Bridge (`ai_tool_bridge.py`)

**Core Classes:**
- `EnhancedPromptBuilder` - Injects tool context into AI prompts (admin only)
- `ToolRequestParser` - Extracts tool calls from AI responses
- `ToolExecutionCoordinator` - Manages tool execution lifecycle

**Features:**
- Admin-only tool exposure
- Natural language tool request parsing
- Safe execution with sandboxing
- Result formatting for AI consumption
- Execution history tracking

### 3. Tools (`tools.py`)

**Categories:**

**Browser Tools:**
- `open_browser(headless=False)` - Launch Chrome/Chromium
- `close_browser()` - Close browser
- `navigate(url)` - Navigate to URL
- `click(selector)` - Click element
- `input(selector, text)` - Type into element
- `screenshot(filename)` - Capture screenshot
- `get_html()` - Get page HTML
- `scroll(amount)` - Scroll page

**Web Search Tools:**
- `search_internet(topic, num_results=1, deep_scrape=True, summarize=True)` - DuckDuckGo search with content extraction
- `fetch_webpage(url, summarize=True)` - Fetch and summarize webpage
- `complex_search_agent(objective, success_criteria, max_iterations=5)` - Autonomous multi-step research

**Filesystem Tools:**
- `read_file(filepath, base_dir=WORKSPACE_DIR)` - Read file
- `write_file(filepath, content, base_dir=WORKSPACE_DIR)` - Write file
- `append_file(filename, content)` - Append to file
- `delete_file(filename)` - Delete file
- `create_file(filename, content)` - Create file
- `find_files(pattern, path='.')` - Recursive file search
- `list_files(path, pattern='*')` - List files
- `list_dir(path)` - List directory
- `rename_file(old, new)` - Rename file
- `copy_file(src, dst)` - Copy file
- `file_exists(filename)` - Check existence
- `file_info(filename)` - Get metadata

**System Tools:**
- `get_cwd()` - Get current working directory
- `get_current_location()` - Get location from IP
- `get_system_utilization()` - CPU/memory/disk usage
- `find_file(filename, search_path='.')` - Find file

**AI Tools:**
- `auxiliary_inference(prompt, system, model, temperature, stream=True)` - Run auxiliary LLM call
- `describe_image(path, model, prompt)` - Vision model image description

## Integration into bot_server.py

### Step 1: Import Modules

Add these imports to `bot_server.py`:

```python
# Tool integration
try:
    from tool_integration import TOOLS_AVAILABLE, ToolInspector
    from ai_tool_bridge import (
        TOOL_INTEGRATION_AVAILABLE,
        EnhancedPromptBuilder,
        ToolExecutionCoordinator
    )
except ImportError:
    TOOLS_AVAILABLE = False
    TOOL_INTEGRATION_AVAILABLE = False
```

### Step 2: Enhance Prompt Building

Modify the `build_ai_prompt` function or the point where prompts are sent to AI:

```python
async def enhanced_ai_generate(payload: dict, user_id: int) -> str:
    """Generate AI response with tool context for admin users"""

    # Inject tool context if user is admin
    if TOOL_INTEGRATION_AVAILABLE:
        payload = EnhancedPromptBuilder.inject_tool_context(
            payload,
            user_id=user_id,
            admin_whitelist=ADMIN_WHITELIST,
            include_examples=True
        )

    # Call AI
    response = await ai_generate_async(payload)

    return response
```

### Step 3: Process AI Responses with Tool Execution

```python
async def process_ai_response_with_tools(
    ai_response: str,
    user_id: int,
    auto_execute: bool = True
) -> str:
    """Process AI response and execute any tool requests"""

    if not TOOL_INTEGRATION_AVAILABLE:
        return ai_response

    coordinator = ToolExecutionCoordinator(user_id, ADMIN_WHITELIST)

    modified_response, results = await coordinator.process_ai_response(
        ai_response,
        auto_execute=auto_execute,
        max_tools=5
    )

    return modified_response
```

### Step 4: Update Message Handler

In `on_text` or wherever AI responses are generated:

```python
# Generate AI response
ai_response = await enhanced_ai_generate(payload, user.id)

# Execute tools if admin requested them
if user.id in ADMIN_WHITELIST:
    ai_response = await process_ai_response_with_tools(
        ai_response,
        user.id,
        auto_execute=True
    )

await m.reply_text(ai_response)
```

## Usage Examples

### For Admin Users

**Example 1: Web Search**
```
User: Search for the latest Python 3.12 features and summarize them

AI Response:
I'll search for that information.

[Executes: Tools.search_internet('Python 3.12 new features', num_results=3, summarize=True)]

Based on the search results, here are the key features in Python 3.12:
1. Type Parameter Syntax (PEP 695)
2. Override Decorator (PEP 698)
3. F-String Improvements
...

[TOOL EXECUTION RESULTS]
================================================================
1. search_internet
   Status: completed
   Time: 8.45s
   Result: [3 results with summaries...]
================================================================
```

**Example 2: File Operations**
```
User: Create a file called notes.txt with "Hello World" and then read it back

AI Response:
I'll create the file and read it.

[Executes: Tools.create_file('notes.txt', 'Hello World')]
[Executes: Tools.read_file('notes.txt')]

Created file successfully. Contents: "Hello World"

[TOOL EXECUTION RESULTS]
================================================================
1. create_file
   Status: completed
   Time: 0.01s
   Result: Created file: /path/to/notes.txt

2. read_file
   Status: completed
   Time: 0.01s
   Result: Hello World
================================================================
```

**Example 3: Complex Research**
```
User: Research the latest breakthroughs in quantum computing and create a summary report

AI Response:
I'll conduct a comprehensive search and create a report.

[Executes: Tools.complex_search_agent(
    objective='Latest quantum computing breakthroughs',
    success_criteria='Find at least 3 recent breakthroughs with technical details',
    max_iterations=3
)]
[Executes: Tools.write_file('quantum_report.md', '<generated report>')]

I've completed the research and created a detailed report in quantum_report.md.
Key findings:
1. IBM's 1000+ qubit processor
2. Google's error correction milestone
3. IonQ's commercial applications

[TOOL EXECUTION RESULTS]
================================================================
...
================================================================
```

### For Non-Admin Users

Tool context is **not injected** into prompts. The AI responds normally without tool awareness.

```
User: Search for Python 3.12 features

AI Response:
Python 3.12 was released in October 2023. Key features include:
- Type parameter syntax (PEP 695)
- Override decorator (PEP 698)
...

[No tool execution - responds from knowledge base]
```

## Security Considerations

### Access Control
- ✅ Tools **ONLY** exposed to admin users in `ADMIN_WHITELIST`
- ✅ Non-admin users cannot see or execute tools
- ✅ Tool context completely absent from non-admin prompts

### Execution Safety
- ✅ All file operations constrained to `WORKSPACE_DIR`
- ✅ Path traversal protection (`..` blocked)
- ✅ Retry limits prevent infinite loops
- ✅ Execution timeouts
- ✅ Error handling and graceful degradation

### Resource Limits
- ✅ Maximum 5 tools per response
- ✅ Maximum 2 retries per tool
- ✅ Truncated results for large outputs
- ✅ Browser sessions properly closed

## Configuration

### Environment Variables

```env
# Enable/disable tool integration
TOOL_INTEGRATION_ENABLED=true

# Max tools per AI response
MAX_TOOLS_PER_RESPONSE=5

# Tool execution timeout (seconds)
TOOL_EXECUTION_TIMEOUT=30

# Auto-execute tools or just detect them
AUTO_EXECUTE_TOOLS=true

# Include tool usage examples in prompt
INCLUDE_TOOL_EXAMPLES=true
```

### In bot_server.py

```python
TOOL_INTEGRATION_ENABLED = os.getenv("TOOL_INTEGRATION_ENABLED", "true").lower() == "true"
MAX_TOOLS_PER_RESPONSE = int(os.getenv("MAX_TOOLS_PER_RESPONSE", "5"))
AUTO_EXECUTE_TOOLS = os.getenv("AUTO_EXECUTE_TOOLS", "true").lower() == "true"
```

## DAG-Based Tool Orchestration

For complex multi-step workflows with dependencies:

```python
from tool_integration import ToolNode, DAGExecutor

# Define nodes
nodes = [
    ToolNode(
        id="search",
        tool_name="search_internet",
        args=["quantum computing 2024"],
        kwargs={"num_results": 3},
        dependencies=[]  # No dependencies
    ),
    ToolNode(
        id="write_report",
        tool_name="write_file",
        args=["report.md"],
        kwargs={"content": "<results from search>"},
        dependencies=["search"]  # Depends on search completing
    )
]

# Execute DAG
executor = DAGExecutor(nodes)
results = await executor.execute()
```

## Testing

### Test Tool Discovery

```python
from tool_integration import ToolInspector

# Get all tools
tools = ToolInspector.get_all_tools()
print(f"Found {len(tools)} tools")

# Get by category
web_tools = ToolInspector.get_tools_by_category("web")
print(f"Web tools: {[t.name for t in web_tools]}")

# Format for prompt
prompt_text = ToolInspector.format_tools_for_prompt()
print(prompt_text)
```

### Test Tool Execution

```python
from tool_integration import ToolExecutor

# Execute a tool
result = await ToolExecutor.execute_tool(
    "search_internet",
    "Python 3.12",
    num_results=2,
    max_retries=2
)

print(f"Status: {result.state}")
print(f"Result: {result.result}")
```

### Test Prompt Enhancement

```python
from ai_tool_bridge import EnhancedPromptBuilder

base_prompt = {
    "model": "llama3.2",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Search for AI news"}
    ]
}

enhanced = EnhancedPromptBuilder.inject_tool_context(
    base_prompt,
    user_id=12345,  # Admin user ID
    admin_whitelist={12345, 67890},
    include_examples=True
)

# Check that tool context was added
system_msg = enhanced["messages"][0]["content"]
assert "TOOL CAPABILITIES AVAILABLE" in system_msg
```

## Troubleshooting

### Tools Not Appearing in Prompt

**Check:**
1. User is in `ADMIN_WHITELIST`
2. `TOOLS_AVAILABLE = True` (tools.py imported)
3. `TOOL_INTEGRATION_AVAILABLE = True` (modules imported)
4. Prompt enhancement is being called

### Tool Execution Fails

**Check:**
1. Tool name is correct (case-sensitive)
2. Required parameters provided
3. File paths are within `WORKSPACE_DIR`
4. Browser is installed (for browser tools)
5. Network connectivity (for web tools)

### AI Doesn't Use Tools

**Possible Reasons:**
1. Query doesn't require tools
2. Tool context not clear enough
3. AI model doesn't understand tool syntax
4. Add explicit instruction: "Use tools to search for this"

## Best Practices

### For Bot Developers

1. **Always validate admin status** before injecting tool context
2. **Set reasonable limits** on tool execution (max tools, timeouts)
3. **Log tool usage** for audit trail
4. **Monitor resource usage** (browser sessions, file I/O)
5. **Provide clear error messages** to users

### For Bot Admins

1. **Test tools in DMs first** before using in groups
2. **Be explicit** in requests: "Search the web for X"
3. **Review results** before trusting automated searches
4. **Clean up files** created during testing
5. **Close browsers** if left open: `/close_browser`

## Future Enhancements

### Planned Features

1. **Tool Chaining** - Allow tools to pass results to other tools
2. **Conditional Execution** - Execute tools based on conditions
3. **Parallel Execution** - Run independent tools concurrently
4. **Tool Permissions** - Granular per-tool access control
5. **Usage Quotas** - Limit tool usage per user/time period
6. **Result Caching** - Cache tool results for repeated queries
7. **Custom Tools** - Allow admins to register custom tools
8. **Tool Templates** - Predefined tool workflows

### Extensibility

**Register Custom Tools:**

```python
from tools import Tools

def my_custom_tool(param1: str, param2: int = 10) -> str:
    """My custom tool that does something useful"""
    return f"Processed {param1} with {param2}"

# Register
Tools.register_runtime_tool(
    "my_custom_tool",
    my_custom_tool,
    doc="Custom tool for special processing"
)
```

**Custom Tool Categories:**

```python
from tool_integration import ToolInspector

# Override categorization
def custom_categorize(name: str, doc: str) -> str:
    if "blockchain" in name.lower():
        return "blockchain"
    return ToolInspector._categorize_tool(name, doc)

ToolInspector._categorize_tool = custom_categorize
```

## Conclusion

The tool integration system transforms your Telegram bot into a powerful AI agent capable of:

- 🔍 Autonomous web research
- 📁 File management
- 🌐 Browser automation
- 💻 System operations
- 🤖 Multi-step workflows

All exposed securely **only to admin users** through natural language interaction!

---

**Created**: 2025-10-18
**Version**: 1.0
**Status**: Ready for integration
