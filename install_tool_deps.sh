#!/bin/bash
# Quick installer for tool dependencies

set -e

echo "=========================================="
echo "Tool Dependencies Installer"
echo "=========================================="
echo ""

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "❌ Error: pip is not available"
    echo "Install pip first:"
    echo "  sudo apt-get install python3-pip  # Ubuntu/Debian"
    echo "  sudo dnf install python3-pip      # Fedora/RHEL"
    exit 1
fi

echo "Step 1: Installing Python packages..."
echo "--------------------------------------"
if [ -f "requirements-tools.txt" ]; then
    python3 -m pip install -r requirements-tools.txt
else
    echo "requirements-tools.txt not found, installing manually..."
    python3 -m pip install beautifulsoup4 lxml requests selenium webdriver-manager ollama
fi

echo ""
echo "Step 2: Checking for Chrome/Chromium..."
echo "--------------------------------------"
if command -v chromium-browser &> /dev/null || command -v chromium &> /dev/null || command -v google-chrome &> /dev/null; then
    echo "✓ Chrome/Chromium found"
else
    echo "⚠ Chrome/Chromium not found"
    echo ""
    echo "To install Chrome/Chromium:"
    echo "  Ubuntu/Debian: sudo apt-get install chromium-browser chromium-chromedriver"
    echo "  Fedora/RHEL:   sudo dnf install chromium chromium-chromedriver"
    echo "  macOS:         brew install chromium"
    echo ""
    echo "Or webdriver-manager will download ChromeDriver automatically."
fi

echo ""
echo "Step 3: Verifying installation..."
echo "--------------------------------------"
if python3 -c "from tools import Tools; print('✓ Tools loaded successfully')" 2>/dev/null; then
    echo "✓ All dependencies installed successfully!"
    echo ""
    echo "You can now use tools like search_internet in the bot."
else
    echo "❌ Verification failed"
    echo ""
    echo "Try running manually:"
    echo "  python3 -c \"from tools import Tools\""
    echo ""
    echo "This will show you the specific error."
    exit 1
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
