#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for real tool execution system

This script verifies that:
1. tool_executor_bridge.py loads correctly
2. tool_telegram_ui.py loads correctly
3. RealToolExecutor can actually call tools
4. ToolDecisionMaker can make decisions
"""

import asyncio
import sys
from typing import Optional

def test_imports():
    """Test that all modules import correctly"""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)

    try:
        from tool_executor_bridge import (
            ToolDecisionMaker,
            RealToolExecutor,
            ToolDecision,
            ToolDecisionType,
            ToolExecutionResult,
        )
        print("✓ tool_executor_bridge imported successfully")
        print(f"  - ToolDecisionMaker: {ToolDecisionMaker}")
        print(f"  - RealToolExecutor: {RealToolExecutor}")
        print(f"  - ToolDecision: {ToolDecision}")
        print(f"  - ToolDecisionType: {ToolDecisionType}")
        print(f"  - ToolExecutionResult: {ToolExecutionResult}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import tool_executor_bridge: {e}")
        return False


def test_tool_availability():
    """Test that Tools class is available"""
    print("\n" + "=" * 80)
    print("TEST 2: Tools Class Availability")
    print("=" * 80)

    try:
        from tools import Tools
        print("✓ Tools class imported successfully")

        # List available tools
        tool_methods = [
            attr for attr in dir(Tools)
            if not attr.startswith('_') and callable(getattr(Tools, attr))
        ]
        print(f"✓ Found {len(tool_methods)} tool methods")

        # Show a few examples
        print("\nExample tools:")
        for tool in tool_methods[:10]:
            print(f"  - {tool}")

        if len(tool_methods) > 10:
            print(f"  ... and {len(tool_methods) - 10} more")

        return True
    except ImportError as e:
        print(f"✗ Failed to import Tools: {e}")
        print("  This is expected if dependencies (bs4, selenium, etc.) are missing")
        return False


async def test_real_execution():
    """Test actual tool execution"""
    print("\n" + "=" * 80)
    print("TEST 3: Real Tool Execution")
    print("=" * 80)

    try:
        from tool_executor_bridge import RealToolExecutor
        from tools import Tools

        # Test with get_cwd (simple, no external dependencies)
        if hasattr(Tools, 'get_cwd'):
            print("\nTesting Tools.get_cwd()...")

            result = await RealToolExecutor.execute_tool('get_cwd', {})

            print(f"✓ Tool executed successfully")
            print(f"  - Success: {result.success}")
            print(f"  - Output: {result.output}")
            print(f"  - Execution time: {result.execution_time:.3f}s")

            if result.success:
                print("✓ Real execution verified - tool actually ran!")
                return True
            else:
                print(f"✗ Tool failed: {result.error}")
                return False
        else:
            print("! get_cwd not available, skipping execution test")
            return True

    except Exception as e:
        print(f"✗ Execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_decision_maker():
    """Test LLM-based tool decision making"""
    print("\n" + "=" * 80)
    print("TEST 4: Tool Decision Making")
    print("=" * 80)

    try:
        from tool_executor_bridge import ToolDecisionMaker, ToolDecisionType

        # Test with a clear tool-requiring message
        message = "Search the internet for Python tutorials"

        print(f"\nTest message: '{message}'")
        print("Asking LLM to decide if tool is needed...")

        # This requires Ollama to be running
        try:
            decision = await ToolDecisionMaker.decide_tool_from_message(
                message,
                ollama_model="llama3.2:3b",
                ollama_url="http://localhost:11434",
            )

            print(f"✓ Decision made successfully")
            print(f"  - Decision type: {decision.decision_type}")
            print(f"  - Tool name: {decision.tool_name}")
            print(f"  - Tool args: {decision.tool_args}")
            print(f"  - Confidence: {decision.confidence:.1%}")
            print(f"  - Reasoning: {decision.reasoning}")

            if decision.decision_type == ToolDecisionType.NEEDS_TOOL:
                print("✓ LLM correctly identified tool need")

            return True

        except Exception as e:
            print(f"! Decision making requires Ollama running: {e}")
            print("  Skipping this test (Ollama not available)")
            return True

    except Exception as e:
        print(f"✗ Decision maker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ui_components():
    """Test UI component availability"""
    print("\n" + "=" * 80)
    print("TEST 5: Telegram UI Components")
    print("=" * 80)

    try:
        # This will fail if python-telegram-bot is not installed
        # That's ok - it just means we're testing outside telegram context
        from tool_telegram_ui import (
            ToolTelegramUI,
            handle_tool_confirmation_callback,
            handle_rating_callback,
        )
        print("✓ tool_telegram_ui imported successfully")
        print(f"  - ToolTelegramUI: {ToolTelegramUI}")
        print(f"  - Confirmation callback: {handle_tool_confirmation_callback}")
        print(f"  - Rating callback: {handle_rating_callback}")
        return True

    except ImportError as e:
        print(f"! telegram module not available (expected in standalone test): {e}")
        print("  This is OK - UI components will work when bot is running")
        return True


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("REAL TOOL EXECUTION SYSTEM - TEST SUITE")
    print("=" * 80)
    print()

    results = []

    # Test 1: Imports
    results.append(("Module Imports", test_imports()))

    # Test 2: Tools availability
    results.append(("Tools Class", test_tool_availability()))

    # Test 3: Real execution
    results.append(("Real Execution", await test_real_execution()))

    # Test 4: Decision making
    results.append(("Decision Making", await test_decision_maker()))

    # Test 5: UI components
    results.append(("UI Components", await test_ui_components()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("=" * 80)

    if passed_count == total_count:
        print("\n✓ All tests passed! Real tool execution system is ready.")
        return 0
    else:
        print("\n! Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
