#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tool_executor_bridge.py — Actual tool execution with confirmation and progress tracking

This module provides REAL tool execution by calling Tools.* methods,
with Telegram UI for confirmation, progress updates, and feedback.
"""

import asyncio
import inspect
import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

try:
    from tools import Tools
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    Tools = None  # type: ignore


class ToolDecisionType(Enum):
    """Type of tool decision"""
    DIRECT_REPLY = "direct_reply"
    WEB_SEARCH = "web_search"
    FILE_ACTION = "file_action"
    BROWSER_ACTION = "browser_action"
    SYSTEM_ACTION = "system_action"


@dataclass
class ToolDecision:
    """Decision about which tool to use"""
    decision_type: ToolDecisionType
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    confidence: float
    reasoning: str
    needs_confirmation: bool = True


@dataclass
class ToolExecutionResult:
    """Result of actual tool execution"""
    success: bool
    tool_name: str
    execution_time: float
    output: Any
    error: Optional[str] = None
    formatted_output: Optional[str] = None


class RealToolExecutor:
    """
    Execute tools from tools.py with actual function calls.

    This replaces the hallucination issue by ACTUALLY calling the tools.
    """

    @staticmethod
    async def execute_tool(
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        timeout: float = 180.0,
        progress_callback: Optional[callable] = None,
    ) -> ToolExecutionResult:
        """
        Actually execute a tool from tools.py

        Args:
            tool_name: Name of the tool method
            tool_args: Arguments to pass to the tool
            timeout: Maximum execution time

        Returns:
            ToolExecutionResult with actual output
        """
        if not TOOLS_AVAILABLE or not Tools:
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                execution_time=0.0,
                output=None,
                error="Tools not available - check dependencies",
            )

        start_time = time.time()

        try:
            # Get the actual method from Tools class
            if not hasattr(Tools, tool_name):
                # Check runtime tools
                runtime_tools = Tools.runtime_tool_functions()
                if tool_name in runtime_tools:
                    tool_func = runtime_tools[tool_name]
                else:
                    return ToolExecutionResult(
                        success=False,
                        tool_name=tool_name,
                        execution_time=time.time() - start_time,
                        output=None,
                        error=f"Tool '{tool_name}' not found in Tools class",
                    )
            else:
                tool_func = getattr(Tools, tool_name)

            # Check if it's callable
            if not callable(tool_func):
                return ToolExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    execution_time=time.time() - start_time,
                    output=None,
                    error=f"'{tool_name}' is not callable",
                )

            # Execute with timeout
            try:
                # Pass progress_callback if tool supports it
                if progress_callback and 'progress_callback' in inspect.signature(tool_func).parameters:
                    tool_args['progress_callback'] = progress_callback

                # Check if tool is async
                if inspect.iscoroutinefunction(tool_func):
                    # Async tool
                    output = await asyncio.wait_for(
                        tool_func(**tool_args),
                        timeout=timeout
                    )
                else:
                    # Sync tool - run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    output = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool_func(**tool_args)),
                        timeout=timeout
                    )
            except asyncio.TimeoutError:
                return ToolExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    execution_time=timeout,
                    output=None,
                    error=f"Tool execution timed out after {timeout}s",
                )
            except Exception as e:
                return ToolExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    execution_time=time.time() - start_time,
                    output=None,
                    error=f"Tool execution failed: {str(e)}\n{traceback.format_exc()}",
                )

            execution_time = time.time() - start_time

            # Format output for display
            formatted = RealToolExecutor._format_tool_output(tool_name, output)

            return ToolExecutionResult(
                success=True,
                tool_name=tool_name,
                execution_time=execution_time,
                output=output,
                formatted_output=formatted,
            )

        except Exception as e:
            return ToolExecutionResult(
                success=False,
                tool_name=tool_name,
                execution_time=time.time() - start_time,
                output=None,
                error=f"Unexpected error: {str(e)}\n{traceback.format_exc()}",
            )

    @staticmethod
    def _format_tool_output(tool_name: str, output: Any) -> str:
        """Format tool output for human-readable display"""
        if output is None:
            return "(No output)"

        # Special formatting for common tool types
        if tool_name == "search_internet":
            if isinstance(output, list):
                lines = ["🔍 Search Results:\n"]
                for i, result in enumerate(output[:5], 1):
                    if isinstance(result, dict):
                        title = result.get("title", "No title")
                        url = result.get("url", "")
                        snippet = result.get("snippet", "")
                        lines.append(f"{i}. {title}")
                        if snippet:
                            lines.append(f"   {snippet[:150]}...")
                        if url:
                            lines.append(f"   {url}")
                        lines.append("")
                return "\n".join(lines)

        elif tool_name == "read_file":
            if isinstance(output, str):
                if len(output) > 500:
                    return f"📄 File contents ({len(output)} chars):\n\n{output[:500]}...\n\n(truncated)"
                return f"📄 File contents:\n\n{output}"

        elif tool_name == "list_dir":
            if isinstance(output, (list, tuple)):
                return f"📁 Directory contents ({len(output)} items):\n" + "\n".join(f"  • {item}" for item in output[:20])

        # Default: convert to string
        output_str = str(output)
        if len(output_str) > 1000:
            return output_str[:1000] + "\n\n...(truncated)"
        return output_str


class ToolDecisionMaker:
    """
    Decide which tool to use based on user message using LLM.

    Uses Ollama with structured JSON output.
    """

    @staticmethod
    async def decide_tool_from_message(
        user_message: str,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
    ) -> ToolDecision:
        """
        Use LLM to decide if a tool is needed and which one.

        Returns:
            ToolDecision with tool selection and parameters
        """
        import requests

        system_prompt = """You are a tool selection assistant. Analyze the user's message and decide:

1. Does this require a tool? Or can you answer directly?
2. If tool needed, which one? (search_internet, read_file, write_file, list_dir, get_cwd, etc.)
3. What are the arguments?

Return JSON:
{
  "needs_tool": boolean,
  "tool_name": string or null,
  "tool_args": object,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Examples:
- "Search for Python tutorials" → {"needs_tool": true, "tool_name": "search_internet", "tool_args": {"topic": "Python tutorials", "num_results": 5}}
- "What's 2+2?" → {"needs_tool": false, "tool_name": null, "tool_args": {}}
- "Read config.json" → {"needs_tool": true, "tool_name": "read_file", "tool_args": {"file_path": "config.json"}}
"""

        user_prompt = f"User message: {user_message}\n\nDecide tool usage:"

        try:
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("message", {}).get("content", "{}")
            decision_data = json.loads(content)

            # Debug logging
            print(f"[tool_decision] LLM response: {content[:200]}")
            print(f"[tool_decision] Parsed tool_args: {decision_data.get('tool_args', {})}")

            # Parse decision
            needs_tool = decision_data.get("needs_tool", False)
            tool_name = decision_data.get("tool_name")
            tool_args = decision_data.get("tool_args", {})
            confidence = decision_data.get("confidence", 0.5)
            reasoning = decision_data.get("reasoning", "No reasoning provided")

            if not needs_tool or not tool_name:
                return ToolDecision(
                    decision_type=ToolDecisionType.DIRECT_REPLY,
                    tool_name=None,
                    tool_args={},
                    confidence=confidence,
                    reasoning=reasoning,
                    needs_confirmation=False,
                )

            # Determine tool type
            if "search" in tool_name.lower():
                decision_type = ToolDecisionType.WEB_SEARCH
            elif "file" in tool_name.lower() or "write" in tool_name.lower() or "read" in tool_name.lower():
                decision_type = ToolDecisionType.FILE_ACTION
            elif "browser" in tool_name.lower() or "navigate" in tool_name.lower():
                decision_type = ToolDecisionType.BROWSER_ACTION
            else:
                decision_type = ToolDecisionType.SYSTEM_ACTION

            return ToolDecision(
                decision_type=decision_type,
                tool_name=tool_name,
                tool_args=tool_args,
                confidence=confidence,
                reasoning=reasoning,
                needs_confirmation=True,
            )

        except Exception as e:
            print(f"[tool_decision] Failed: {e}")
            # Default to direct reply on error
            return ToolDecision(
                decision_type=ToolDecisionType.DIRECT_REPLY,
                tool_name=None,
                tool_args={},
                confidence=0.0,
                reasoning=f"Error in decision making: {str(e)}",
                needs_confirmation=False,
            )


# Convenience functions
async def execute_tool_with_decision(
    decision: ToolDecision,
) -> Optional[ToolExecutionResult]:
    """Execute a tool based on a decision"""
    if decision.decision_type == ToolDecisionType.DIRECT_REPLY:
        return None

    if not decision.tool_name:
        return None

    return await RealToolExecutor.execute_tool(
        decision.tool_name,
        decision.tool_args,
    )


async def decide_and_execute_tool(
    user_message: str,
    *,
    ollama_model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
) -> Tuple[ToolDecision, Optional[ToolExecutionResult]]:
    """
    Complete flow: decide tool → execute if needed

    Returns:
        (decision, result or None)
    """
    decision = await ToolDecisionMaker.decide_tool_from_message(
        user_message,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )

    if decision.decision_type == ToolDecisionType.DIRECT_REPLY:
        return decision, None

    result = await execute_tool_with_decision(decision)
    return decision, result
