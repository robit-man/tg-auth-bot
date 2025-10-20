#!/usr/bin/env python3
"""
Auto-installer for bot dependencies
Runs on first startup to install all required packages
"""

import sys
import subprocess
import os
from pathlib import Path


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package):
    """Install a package using pip"""
    try:
        print(f"  Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install {package}: {e}")
        return False


def auto_install_dependencies():
    """Auto-install missing dependencies on first run"""

    # Check for marker file
    marker_file = Path(__file__).parent / ".deps_installed"

    if marker_file.exists():
        # Dependencies already checked/installed
        return

    print("=" * 60)
    print("FIRST RUN: Checking and installing dependencies...")
    print("=" * 60)
    print()

    # Core bot dependencies (should already be installed)
    core_deps = [
        ("python-telegram-bot", "telegram"),
        ("ollama", "ollama"),
    ]

    # Tool dependencies (often missing)
    tool_deps = [
        ("beautifulsoup4", "bs4"),
        ("lxml", "lxml"),
        ("requests", "requests"),
        ("selenium", "selenium"),
        ("webdriver-manager", "webdriver_manager"),
    ]

    all_installed = True

    print("Checking core dependencies...")
    for package, import_name in core_deps:
        if not check_package(package, import_name):
            print(f"  ✗ Missing: {package}")
            if not install_package(package):
                all_installed = False
        else:
            print(f"  ✓ {package}")

    print()
    print("Checking tool dependencies...")
    tools_needed = False
    for package, import_name in tool_deps:
        if not check_package(package, import_name):
            print(f"  ✗ Missing: {package}")
            tools_needed = True
            if not install_package(package):
                all_installed = False
        else:
            print(f"  ✓ {package}")

    print()

    if tools_needed:
        print("=" * 60)
        print("TOOL DEPENDENCIES INSTALLED")
        print("=" * 60)
        print()
        print("Tools like search_internet will now work!")
        print()

    if all_installed:
        # Create marker file
        marker_file.write_text("Dependencies checked and installed\n")
        print("✓ All dependencies installed successfully!")
    else:
        print("⚠ Some dependencies failed to install.")
        print("  You may need to install them manually:")
        print("  pip install -r requirements-tools.txt")

    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    auto_install_dependencies()
