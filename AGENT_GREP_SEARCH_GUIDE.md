# Agent Grep & Search Command Reference

## Purpose

This guide provides the telegram agent with grep commands and search patterns for code analysis, information retrieval, and codebase navigation.

## Core Grep Commands

### 1. Find Function Definitions

**Pattern:** `def function_name`

**Use when:** User asks "where is function X defined?" or "show me the function X"

**Example:**
```bash
grep -rn "def search_internet" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Where is search_internet defined?"
Execute: grep -rn "def search_internet" .
Parse the output for filename and line number
Report: "search_internet is defined in tools.py at line 852"
```

### 2. Find Class Definitions

**Pattern:** `class ClassName`

**Use when:** User asks about a specific class

**Example:**
```bash
grep -rn "class Tools" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Show me the Tools class"
Execute: grep -rn "class Tools" .
Report the file and line number
Optionally: Read that section of the file to show the class
```

### 3. Find Imports

**Pattern:** `from module import` or `import module`

**Use when:** User asks "what uses X?" or "where is X imported?"

**Example:**
```bash
grep -rn "from tools import" /home/robit/Respositories/tg-auth-bot/
grep -rn "import asyncio" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "What files use the Tools class?"
Execute: grep -rn "from tools import Tools" .
List all files that import it
```

### 4. Find Variable Usage

**Pattern:** Variable name (whole word)

**Use when:** User asks "where is X used?" or "what uses X?"

**Example:**
```bash
grep -rnw "OLLAMA_MODEL" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Where is OLLAMA_MODEL used?"
Execute: grep -rnw "OLLAMA_MODEL" .
Note: -w flag ensures whole word match (not OLLAMA_MODEL_NAME)
List all occurrences with context
```

### 5. Find String Literals

**Pattern:** String in quotes

**Use when:** User asks about specific messages, errors, or text

**Example:**
```bash
grep -rn "Tool execution failed" /home/robit/Respositories/tg-auth-bot/
grep -rn '"search_internet"' /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Where does it say 'Tool execution failed'?"
Execute: grep -rn "Tool execution failed" .
Report where this error message appears
```

### 6. Find Comments

**Pattern:** `# comment text` or `"""docstring"""`

**Use when:** User asks "are there any TODOs?" or "what does the code say about X?"

**Example:**
```bash
grep -rn "# TODO" /home/robit/Respositories/tg-auth-bot/
grep -rn "# FIXME" /home/robit/Respositories/tg-auth-bot/
grep -rn "# NOTE" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Are there any TODOs in the code?"
Execute: grep -rn "# TODO" .
List all TODO comments with file and line
```

### 7. Find Configuration Values

**Pattern:** Key names in config/env

**Use when:** User asks about settings or configuration

**Example:**
```bash
grep -rn "TELEGRAM_TOKEN" /home/robit/Respositories/tg-auth-bot/
grep -rn "ADMIN_WHITELIST" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "How is the Telegram token configured?"
Execute: grep -rn "TELEGRAM_TOKEN" .
Show where it's defined and used
```

### 8. Find Error Handling

**Pattern:** `except`, `raise`, `try`

**Use when:** User asks about error handling or exceptions

**Example:**
```bash
grep -rn "except Exception" /home/robit/Respositories/tg-auth-bot/
grep -rn "raise ValueError" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "How does it handle errors?"
Execute: grep -rn "except" . | head -20
Show main error handling patterns
```

### 9. Find Async Functions

**Pattern:** `async def`

**Use when:** User asks about asynchronous code

**Example:**
```bash
grep -rn "async def" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "What async functions are there?"
Execute: grep -rn "async def" .
List all async function definitions
```

### 10. Find Tool Definitions

**Pattern:** `@staticmethod` or method names

**Use when:** User asks "what tools are available?"

**Example:**
```bash
grep -rn "@staticmethod" /home/robit/Respositories/tg-auth-bot/tools.py
grep -rn "def [a-z_]*(" /home/robit/Respositories/tg-auth-bot/tools.py
```

**Agent Instruction:**
```
When user asks: "What tools can the agent use?"
Execute: grep -A1 "@staticmethod" tools.py
Show all @staticmethod followed by their function definitions
```

## Advanced Grep Patterns

### 11. Case-Insensitive Search

**Flag:** `-i`

**Use when:** User's query might have different capitalization

**Example:**
```bash
grep -rni "ollama" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Find anything about Ollama"
Execute: grep -rni "ollama" .
Note: Matches OLLAMA, Ollama, ollama, etc.
```

### 12. Search with Context

**Flags:** `-A N` (after), `-B N` (before), `-C N` (both)

**Use when:** User needs to see surrounding code

**Example:**
```bash
grep -rn -A5 "def search_internet" /home/robit/Respositories/tg-auth-bot/
grep -rn -B3 "raise Exception" /home/robit/Respositories/tg-auth-bot/
grep -rn -C2 "ADMIN_WHITELIST" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Show me the search_internet function signature"
Execute: grep -A10 "def search_internet" tools.py
Shows function definition plus 10 lines (parameters, docstring)
```

### 13. Exclude Patterns

**Flag:** `--exclude` or `--exclude-dir`

**Use when:** Want to skip certain files/directories

**Example:**
```bash
grep -rn "search" --exclude="*.md" /home/robit/Respositories/tg-auth-bot/
grep -rn "import" --exclude-dir=".venv" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When searching code, exclude documentation:
Execute: grep -rn "pattern" --exclude="*.md" --exclude-dir=".venv" .
```

### 14. Count Matches

**Flag:** `-c`

**Use when:** User asks "how many X are there?"

**Example:**
```bash
grep -rc "def " /home/robit/Respositories/tg-auth-bot/tools.py
grep -rc "TODO" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "How many functions are in tools.py?"
Execute: grep -c "def " tools.py
Report: "There are N function definitions"
```

### 15. List Files Only

**Flag:** `-l`

**Use when:** User wants to know which files contain something

**Example:**
```bash
grep -rl "search_internet" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Which files use search_internet?"
Execute: grep -rl "search_internet" .
List the filenames only, not line numbers
```

### 16. Regular Expressions

**Flag:** `-E` (extended regex)

**Use when:** Need complex pattern matching

**Example:**
```bash
grep -rn -E "def (search|read|write)_" /home/robit/Respositories/tg-auth-bot/
grep -rn -E "http[s]?://[^\s]+" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Find all tool functions that start with search, read, or write"
Execute: grep -E "def (search|read|write)_[a-z_]+\(" tools.py
Use regex for complex patterns
```

### 17. Invert Match

**Flag:** `-v`

**Use when:** Want lines that DON'T match

**Example:**
```bash
grep -v "^#" config.py  # Lines not starting with #
grep -v "^$" script.py   # Non-empty lines
```

**Agent Instruction:**
```
When user asks: "Show config without comments"
Execute: grep -v "^#" config_file | grep -v "^$"
Removes comment lines and empty lines
```

### 18. Show Only Matches

**Flag:** `-o`

**Use when:** Want only the matched text, not the whole line

**Example:**
```bash
grep -o "http[s]://[^\"' ]*" /home/robit/Respositories/tg-auth-bot/tools.py
```

**Agent Instruction:**
```
When user asks: "Extract all URLs from the code"
Execute: grep -oh "http[s]://[^\"' ]*" .
Shows just the URLs, not full lines
```

## File-Specific Searches

### 19. Search Python Files Only

**Pattern:** `--include="*.py"`

**Example:**
```bash
grep -rn "async def" --include="*.py" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When searching code, focus on Python files:
Execute: grep -rn "pattern" --include="*.py" .
```

### 20. Search Documentation Only

**Pattern:** `--include="*.md"`

**Example:**
```bash
grep -rn "installation" --include="*.md" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks about documentation:
Execute: grep -rn "keyword" --include="*.md" .
Search markdown files only
```

## Combination Patterns

### 21. Find Function Calls

**Use when:** User asks "where is function X called?"

**Example:**
```bash
grep -rn "search_internet(" /home/robit/Respositories/tg-auth-bot/ | grep -v "def search_internet"
```

**Agent Instruction:**
```
When user asks: "Where is search_internet called?"
Execute: grep -rn "search_internet(" . | grep -v "def search_internet"
Note: Second grep removes the definition line
Show all call sites
```

### 22. Find and Count by File

**Use when:** Need statistics per file

**Example:**
```bash
grep -rc "progress_callback" /home/robit/Respositories/tg-auth-bot/*.py
```

**Agent Instruction:**
```
When user asks: "Which files use progress_callback the most?"
Execute: grep -rc "progress_callback" *.py | sort -t: -k2 -nr
Shows count per file, sorted descending
```

### 23. Multi-Pattern Search

**Flag:** `-e pattern1 -e pattern2`

**Use when:** Multiple patterns needed

**Example:**
```bash
grep -rn -e "ERROR" -e "WARNING" -e "CRITICAL" /home/robit/Respositories/tg-auth-bot/
```

**Agent Instruction:**
```
When user asks: "Find all log levels"
Execute: grep -rn -e "ERROR" -e "WARNING" -e "INFO" .
Search for multiple patterns at once
```

## Agent Decision Tree

### When to Use Grep

```
User question type → Grep command

"Where is X defined?" → grep -rn "def X\|class X" .
"What uses X?" → grep -rnw "X" .
"Show me X" → grep -A20 "def X\|class X" specific_file.py
"How many X?" → grep -rc "X" .
"Which files have X?" → grep -rl "X" .
"Find X in docs" → grep -rn "X" --include="*.md" .
"Any TODOs?" → grep -rn "# TODO" .
"What are the errors?" → grep -rn "except\|raise" .
"List all functions" → grep -rn "^def " file.py
"Find URLs" → grep -oh "http[s]://[^\"' ]*" .
```

## Agent Workflow

### Standard Search Process

1. **Identify Intent**
   - What is the user looking for?
   - Code? Documentation? Configuration?

2. **Choose Pattern**
   - Function/class? → `def name` or `class Name`
   - Usage? → just the name with `-w` flag
   - String literal? → exact text in quotes

3. **Select Scope**
   - Specific file? → grep file.py
   - All Python? → grep --include="*.py"
   - Everything? → grep -r .

4. **Add Modifiers**
   - Need context? → Add -A/-B/-C
   - Case insensitive? → Add -i
   - Count only? → Add -c
   - Files only? → Add -l

5. **Execute and Parse**
   - Run the grep command
   - Parse output (file:line:content)
   - Present results to user

6. **Follow Up**
   - If results found: Show them clearly
   - If too many results: Offer to narrow down
   - If no results: Try variations (case-insensitive, broader pattern)

## Example Agent Conversations

### Example 1: Find Function

**User:** "Where is the search_internet function?"

**Agent Internal:**
```
Intent: Find function definition
Pattern: "def search_internet"
Scope: All Python files
Command: grep -rn "def search_internet" --include="*.py" .
```

**Agent Execute:**
```bash
grep -rn "def search_internet" --include="*.py" /home/robit/Respositories/tg-auth-bot/
```

**Agent Response:**
"The `search_internet` function is defined in [tools.py:852](tools.py#L852)"

### Example 2: What Uses X?

**User:** "What files use the progress_callback?"

**Agent Internal:**
```
Intent: Find usage across files
Pattern: "progress_callback" (whole word)
Scope: All Python files
Command: grep -rlw "progress_callback" --include="*.py" .
```

**Agent Execute:**
```bash
grep -rlw "progress_callback" --include="*.py" /home/robit/Respositories/tg-auth-bot/
```

**Agent Response:**
"The progress_callback is used in:
- [tools.py](tools.py) - Function parameter and calls
- [tool_executor_bridge.py](tool_executor_bridge.py) - Passed to tools
- [tool_telegram_ui.py](tool_telegram_ui.py) - Callback implementation"

### Example 3: Show Me Code

**User:** "Show me how progress_callback is used in tools.py"

**Agent Internal:**
```
Intent: Show code with context
Pattern: "progress_callback"
Scope: Specific file
Command: grep -n -A3 -B3 "progress_callback" tools.py | head -30
```

**Agent Execute:**
```bash
grep -n -A3 -B3 "progress_callback" tools.py | head -30
```

**Agent Response:**
"Here's how progress_callback is used in tools.py:

Line 864: Function parameter
```python
progress_callback: Optional[callable] = None,
```

Line 905-910: Helper function
```python
def _progress(message: str):
    if progress_callback:
        try:
            progress_callback(message)
```

[Show more...]"

### Example 4: Count Occurrences

**User:** "How many times is progress_callback mentioned?"

**Agent Internal:**
```
Intent: Count matches
Pattern: "progress_callback"
Scope: All Python files
Command: grep -rc "progress_callback" --include="*.py" .
```

**Agent Execute:**
```bash
grep -rc "progress_callback" --include="*.py" /home/robit/Respositories/tg-auth-bot/ | grep -v ":0"
```

**Agent Response:**
"progress_callback appears:
- tools.py: 23 times
- tool_executor_bridge.py: 5 times
- tool_telegram_ui.py: 8 times
Total: 36 occurrences"

## Common Grep Flags Reference

| Flag | Meaning | Use Case |
|------|---------|----------|
| `-r` | Recursive | Search all files in directory |
| `-n` | Line numbers | Show where matches occur |
| `-i` | Ignore case | Case-insensitive search |
| `-w` | Whole word | Match whole words only |
| `-l` | List files | Show filenames only |
| `-c` | Count | Count matches per file |
| `-v` | Invert | Show non-matching lines |
| `-A N` | After | Show N lines after match |
| `-B N` | Before | Show N lines before match |
| `-C N` | Context | Show N lines before and after |
| `-o` | Only match | Show matched text only |
| `-E` | Extended regex | Use extended regex patterns |
| `-h` | No filename | Hide filenames in output |
| `--include` | File pattern | Only search matching files |
| `--exclude` | File pattern | Skip matching files |
| `--exclude-dir` | Directory | Skip directory |

## Regex Patterns Reference

| Pattern | Meaning | Example |
|---------|---------|---------|
| `.` | Any character | `a.c` matches "abc", "a1c" |
| `*` | Zero or more | `ab*c` matches "ac", "abc", "abbc" |
| `+` | One or more | `ab+c` matches "abc", "abbc" (not "ac") |
| `?` | Zero or one | `ab?c` matches "ac", "abc" |
| `^` | Start of line | `^def` matches lines starting with "def" |
| `$` | End of line | `;$` matches lines ending with ";" |
| `\|` | OR | `cat\|dog` matches "cat" or "dog" |
| `[abc]` | Character class | `[aeiou]` matches any vowel |
| `[^abc]` | Not in class | `[^0-9]` matches non-digits |
| `[a-z]` | Range | `[a-z]` matches lowercase letters |
| `\w` | Word character | `\w+` matches words |
| `\s` | Whitespace | `\s+` matches spaces/tabs |
| `\d` | Digit | `\d+` matches numbers |
| `()` | Group | `(ab)+` matches "ab", "abab" |

## Practical Examples for Agent

### Find All Available Tools

```bash
grep -n "^\s*def [a-z_]*(" tools.py | grep -v "^\s*def _"
```
Shows all public methods (not starting with _)

### Find Configuration Options

```bash
grep -rn "^[A-Z_]* =" . --include="*.py"
```
Finds CONSTANT_NAMES = value patterns

### Find All Async Functions

```bash
grep -rn "async def" --include="*.py" .
```

### Find All Error Messages

```bash
grep -rn '".*[Ee]rror.*"' --include="*.py" .
grep -rn "'.*[Ee]rror.*'" --include="*.py" .
```

### Find Telegram Message Sends

```bash
grep -rn "\.send_message\|\.reply_text\|\.edit_text" --include="*.py" .
```

### Find Database Operations

```bash
grep -rn "execute\|commit\|fetchall\|fetchone" --include="*.py" .
```

### Find Environment Variables

```bash
grep -rn "os\.getenv\|os\.environ" --include="*.py" .
```

### Find Logging Statements

```bash
grep -rn "log_message\|logger\.\|print(" --include="*.py" .
```

## Agent Learning Process

### Phase 1: Direct Execution

User asks → Agent greps → Agent reports results

**Example:**
- User: "Find search_internet"
- Agent: `grep -rn "def search_internet" .`
- Agent: "Found at tools.py:852"

### Phase 2: Context Enhancement

User asks → Agent greps with context → Agent shows code

**Example:**
- User: "Show me search_internet signature"
- Agent: `grep -A15 "def search_internet" tools.py`
- Agent: Shows function definition with parameters

### Phase 3: Multi-Step Analysis

User asks → Agent chains multiple greps → Agent synthesizes

**Example:**
- User: "How is progress_callback used?"
- Agent:
  1. `grep -rl "progress_callback" .` (find files)
  2. `grep -c "progress_callback" file1 file2 file3` (count uses)
  3. `grep -A5 "progress_callback" files` (show examples)
- Agent: "Used in 3 files, 36 total occurrences, here are key examples..."

### Phase 4: Pattern Recognition

User asks vague question → Agent infers intent → Agent executes appropriate grep

**Example:**
- User: "What can this bot do?"
- Agent: Infers need to find tools
- Agent: `grep -n "^\s*def [a-z_]*(" tools.py | grep -v "_"`
- Agent: Lists all public tool methods

## Error Handling

### No Results Found

**Response Pattern:**
```
I couldn't find [X] in the codebase. Let me try:
1. Case-insensitive search
2. Broader pattern
3. Search in documentation
```

**Example:**
```bash
# First attempt
grep -rn "searchInternet" .

# No results, try case-insensitive
grep -rni "searchinternet" .

# Still nothing, try partial match
grep -rni "search.*internet" .
```

### Too Many Results

**Response Pattern:**
```
Found many matches for [X]. Let me narrow it down:
1. Which file? (bot_server.py, tools.py, etc.)
2. Definitions only?
3. Most recent occurrences?
```

**Example:**
```bash
# Too many results
grep -rn "search" .  # Returns 500+ matches

# Narrow to function definitions
grep -rn "def.*search" .  # Returns 20 matches

# Further narrow to specific file
grep -n "def.*search" tools.py  # Returns 5 matches
```

### Ambiguous Patterns

**Response Pattern:**
```
"[X]" could mean:
1. Function definition → def [X]
2. Class usage → [X]()
3. String literal → "[X]"
Which did you mean?
```

## Integration with Tool System

### When to Use search_internet Tool vs Grep

**Use grep when:**
- Searching within codebase
- Finding definitions
- Analyzing code structure
- Counting occurrences

**Use search_internet when:**
- Need external information
- Documentation not in codebase
- Looking for tutorials/examples
- Researching libraries/APIs

### Combining Both

**Example:**
```
User: "How do I use asyncio in Python?"

Agent:
1. First check codebase: grep -rn "asyncio" .
2. If found: Show local examples
3. If need more: search_internet("Python asyncio tutorial")
4. Combine: "Here's how we use it locally, and here are external resources"
```

## Best Practices for Agent

### 1. Always Quote Patterns

```bash
# Good
grep -rn "def search" .

# Bad (may break with special chars)
grep -rn def search .
```

### 2. Use Appropriate Flags

```bash
# Finding definitions
grep -rn "def function_name"

# Finding usage
grep -rnw "variable_name"

# Counting
grep -rc "pattern"

# Just filenames
grep -rl "pattern"
```

### 3. Limit Output When Needed

```bash
# Show first 10 matches
grep -rn "pattern" . | head -10

# Show last 5 matches
grep -rn "pattern" . | tail -5
```

### 4. Combine with Other Tools

```bash
# Sort by file
grep -rn "pattern" . | sort

# Count total matches
grep -rn "pattern" . | wc -l

# Unique files
grep -rl "pattern" . | wc -l
```

### 5. Readable Output

Present results in markdown format:

```markdown
Found `search_internet` in:

1. [tools.py:852](tools.py#L852) - Function definition
2. [bot_server.py:4060](bot_server.py#L4060) - Function call
3. [tool_executor_bridge.py:185](tool_executor_bridge.py#L185) - Usage example
```

## Summary

This guide provides the telegram agent with:

✅ **20+ grep command patterns** for common searches
✅ **Decision tree** for choosing right grep command
✅ **Workflow process** for executing searches
✅ **Example conversations** showing agent behavior
✅ **Error handling** strategies
✅ **Integration** with tool system
✅ **Best practices** for reliable results

The agent should now be able to:
- Search codebase effectively
- Find functions, classes, variables
- Analyze code structure
- Count and list occurrences
- Present results clearly to users
- Combine grep with other tools
- Handle edge cases and errors

**Key Principle:** Always parse grep output and present it in a user-friendly format, not raw grep output.
