# Tool Dependencies Setup Guide

## Problem: "Tools not available - check dependencies"

This error occurs when trying to use tools like `search_internet` because the required Python packages are not installed.

## Auto-Generated Requirements

The `requirements-tools.txt` file is **automatically generated** by `bot_server.py` on startup from a baked-in dependency list. This ensures the requirements file is always in sync with the code.

If the file is missing, it will be recreated automatically the next time you run the bot.

## Quick Fix

Install the tool dependencies:

```bash
pip install -r requirements-tools.txt
```

Or use the automated installer:

```bash
./install_tool_deps.sh
```

Or install manually:

```bash
pip install beautifulsoup4 lxml requests selenium webdriver-manager ollama
```

## Required Dependencies

The tools system requires these packages:

### Core Dependencies (Required)
- **beautifulsoup4** - HTML parsing and web scraping
- **lxml** - Fast XML/HTML parser (used by BeautifulSoup)
- **requests** - HTTP library for web requests
- **selenium** - Browser automation for JavaScript-heavy sites
- **webdriver-manager** - Automatic ChromeDriver management

### Optional Dependencies
- **ollama** - LLM summarization of tool results (optional, but recommended)

## System Dependencies

For Selenium to work, you also need Chrome/Chromium installed:

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install chromium-browser chromium-chromedriver
```

### Fedora/RHEL
```bash
sudo dnf install chromium chromium-chromedriver
```

### macOS
```bash
brew install chromium
```

## Verification

Test if tools are working:

```bash
python3 -c "from tools import Tools; print('✓ Tools loaded successfully')"
```

If successful, you should see: `✓ Tools loaded successfully`

## Troubleshooting

### Error: "No module named 'bs4'"
```bash
pip install beautifulsoup4
```

### Error: "No module named 'selenium'"
```bash
pip install selenium webdriver-manager
```

### Error: "Chrome/ChromeDriver not found"
```bash
# Ubuntu/Debian
sudo apt-get install chromium-browser chromium-chromedriver

# Or let webdriver-manager download it automatically
python3 -c "from webdriver_manager.chrome import ChromeDriverManager; ChromeDriverManager().install()"
```

### Error: "No module named 'ollama'"
```bash
pip install ollama
```

Note: ollama is optional. If you don't want to use it, you can modify tools.py to make it optional.

## Architecture

The tool system has graceful degradation:

1. **Tools Available**: If all dependencies are installed, tools work normally
2. **Tools Unavailable**: If dependencies are missing, the bot returns:
   - Error message: "Tools not available - check dependencies"
   - The bot continues to work for non-tool commands

The check happens in `tool_executor_bridge.py`:

```python
try:
    from tools import Tools
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    Tools = None
```

## Available Tools

Once dependencies are installed, these tools become available:

- `search_internet` - Web search with content extraction
- `execute_bash` - Run shell commands
- `read_file` - Read file contents
- `write_file` - Write/create files
- `list_directory` - List directory contents
- `search_files` - Find files by pattern
- And more...

See [tools.py](tools.py) for the complete list.

## Optional: Making ollama Optional

If you don't want to install ollama, you can make it optional by editing tools.py:

```python
# Change line 50 from:
import ollama  # type: ignore

# To:
try:
    import ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None
```

Then add checks before using ollama in the code.
