#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Integration System - Expose tools.py capabilities to AI for admin users

This module provides a DAG-based instruction system that allows the AI to discover,
inspect, and execute tools from tools.py when invoked by whitelisted admin users.

Features:
- Tool discovery and introspection
- Admin-only access control
- Error handling with retry logic
- Execution tracking and context awareness
- DAG-based tool orchestration
"""

import json
import inspect
import traceback
import asyncio
import os
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import Tools class
try:
    from tools import Tools
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    Tools = None

_CATEGORY_ENV = os.getenv("TOOL_CONTEXT_CATEGORIES", "")
_ALLOWED_TOOL_CATEGORIES: Optional[Set[str]] = {
    cat.strip().lower()
    for cat in _CATEGORY_ENV.split(",")
    if cat.strip()
} if _CATEGORY_ENV else None


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


_MAX_DOC_LINES = _int_env("TOOL_CONTEXT_DOC_LINES", 2)
_MAX_DOC_CHARS = _int_env("TOOL_CONTEXT_DOC_CHARS", 220)
_INCLUDE_RUNTIME_TOOLS = os.getenv("TOOL_CONTEXT_INCLUDE_RUNTIME", "true").strip().lower() != "false"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution States
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionState(Enum):
    """States for tool execution in DAG"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolMetadata:
    """Metadata about a tool function"""
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    is_async: bool
    category: str = "general"


@dataclass
class ToolExecutionResult:
    """Result of executing a tool"""
    tool_name: str
    state: ExecutionState
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolNode:
    """Node in the tool execution DAG"""
    id: str
    tool_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # IDs of nodes that must complete first
    state: ExecutionState = ExecutionState.PENDING
    result: Optional[ToolExecutionResult] = None
    max_retries: int = 2


@dataclass
class ToolDAGRequest:
    """Represents a parsed DAG execution request"""
    raw_block: str
    nodes: List[ToolNode]
    summary_label: str = "Untitled DAG"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Inspector
# ─────────────────────────────────────────────────────────────────────────────

class ToolInspector:
    """Provides introspection capabilities for available tools"""

    @staticmethod
    def get_all_tools() -> List[ToolMetadata]:
        """Discover all available tools from Tools class"""
        if not TOOLS_AVAILABLE or not Tools:
            return []

        tools = []

        # Get static methods and class methods from Tools
        for attr_name in dir(Tools):
            if attr_name.startswith("_"):
                continue

            attr = getattr(Tools, attr_name)
            if not callable(attr):
                continue

            try:
                sig = inspect.signature(attr)
                doc = inspect.getdoc(attr) or "No documentation available"

                # Extract parameters
                params = []
                for param_name, param in sig.parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
                    params.append(param_info)

                # Determine return type
                return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else None

                # Check if async
                is_async = inspect.iscoroutinefunction(attr)

                # Categorize tool
                category = ToolInspector._categorize_tool(attr_name, doc)

                tools.append(ToolMetadata(
                    name=attr_name,
                    signature=str(sig),
                    docstring=doc,
                    parameters=params,
                    return_type=return_type,
                    is_async=is_async,
                    category=category
                ))
            except Exception as e:
                # Skip tools that can't be inspected
                continue

        # Get runtime registered tools
        runtime_tools = Tools.runtime_tool_functions() if Tools else {}
        for name, fn in runtime_tools.items():
            try:
                sig = inspect.signature(fn)
                doc = Tools._runtime_docs.get(name, inspect.getdoc(fn) or "Runtime tool")

                params = []
                for param_name, param in sig.parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
                    params.append(param_info)

                tools.append(ToolMetadata(
                    name=name,
                    signature=str(sig),
                    docstring=doc,
                    parameters=params,
                    return_type=None,
                    is_async=inspect.iscoroutinefunction(fn),
                    category="runtime"
                ))
            except Exception:
                continue

        return tools

    @staticmethod
    def _categorize_tool(name: str, doc: str) -> str:
        """Categorize tool based on name and documentation"""
        name_lower = name.lower()
        doc_lower = doc.lower()

        if any(kw in name_lower for kw in ["browser", "navigate", "click", "screenshot"]):
            return "browser"
        elif any(kw in name_lower for kw in ["search", "fetch", "scrape", "web"]):
            return "web"
        elif any(kw in name_lower for kw in ["file", "read", "write", "list", "find"]):
            return "filesystem"
        elif any(kw in name_lower for kw in ["system", "location", "utilization", "cwd"]):
            return "system"
        elif "llm" in doc_lower or "inference" in doc_lower or "chat" in doc_lower:
            return "ai"
        else:
            return "general"

    @staticmethod
    def get_tool_by_name(name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool"""
        for tool in ToolInspector.get_all_tools():
            if tool.name == name:
                return tool
        return None

    @staticmethod
    def get_tools_by_category(category: str) -> List[ToolMetadata]:
        """Get all tools in a category"""
        return [t for t in ToolInspector.get_all_tools() if t.category == category]

    @staticmethod
    def format_tools_for_prompt(
        allowed_categories: Optional[Set[str]] = None,
        *,
        max_doc_lines: int = 3,
        max_doc_chars: int = 240,
        include_runtime: bool = True,
    ) -> str:
        """Format tools for AI prompt with optional filtering and truncation."""
        tools = ToolInspector.get_all_tools()

        if not tools:
            return "No tools available."

        allowed = {c.lower() for c in allowed_categories} if allowed_categories else None

        # Group by category
        by_category: Dict[str, List[ToolMetadata]] = {}
        for tool in tools:
            category_key = tool.category
            if allowed and category_key.lower() not in allowed:
                continue
            if category_key == "runtime" and not include_runtime:
                continue
            by_category.setdefault(category_key, []).append(tool)

        if not by_category:
            return "No tools available for the current policy."

        lines = ["AVAILABLE TOOLS", "=" * 80, ""]

        for category in sorted(by_category.keys()):
            lines.append(f"\n{category.upper()} TOOLS:")
            lines.append("-" * 80)

            for tool in sorted(by_category[category], key=lambda t: t.name):
                lines.append(f"\n• {tool.name}{tool.signature}")

                doc_lines: List[str] = []
                for raw_line in (tool.docstring or "").splitlines():
                    stripped = raw_line.strip()
                    if stripped:
                        doc_lines.append(stripped)
                    if len(doc_lines) >= max_doc_lines:
                        break
                doc_preview = " ".join(doc_lines)
                if doc_preview and len(doc_preview) > max_doc_chars:
                    doc_preview = doc_preview[:max_doc_chars].rstrip() + "..."
                if doc_preview:
                    lines.append(f"  {doc_preview}")

                for param_line in ToolInspector._format_param_lines(tool):
                    lines.append(param_line)

        lines.append("\n" + "=" * 80)
        lines.append("\nTo use a tool, call it with: Tools.tool_name(arg1, arg2, kwarg1=value1)")
        lines.append("For async tools, use: await Tools.tool_name(...)")
        lines.append("")
        lines.append("For multi-step orchestration, emit a DAG block between <<TOOL_DAG>> and <<END_DAG>> with JSON:")
        lines.append("<<TOOL_DAG>>")
        lines.append("{")
        lines.append('  "label": "short description",')
        lines.append('  "nodes": [')
        lines.append('    {"id": "step1", "tool": "search_internet", "args": ["query"], "deps": []}')
        lines.append("  ]")
        lines.append("}")
        lines.append("<<END_DAG>>")

        return "\n".join(lines)

    @staticmethod
    def _format_param_lines(tool: ToolMetadata) -> List[str]:
        lines: List[str] = []
        required_params = [p for p in tool.parameters if p["required"]]
        optional_params = [p for p in tool.parameters if not p["required"]]
        if required_params:
            lines.append(f"  Required: {', '.join(p['name'] for p in required_params)}")
        if optional_params:
            lines.append(f"  Optional: {', '.join(p['name'] for p in optional_params)}")
        if tool.is_async:
            lines.append("  [ASYNC]")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Tool Executor
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """Executes tools with error handling and retry logic"""

    @staticmethod
    async def execute_tool(
        tool_name: str,
        *args,
        max_retries: int = 2,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ToolExecutionResult:
        """Execute a tool with retry logic"""

        if not TOOLS_AVAILABLE or not Tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                state=ExecutionState.FAILED,
                error="Tools module not available"
            )

        tool_meta = ToolInspector.get_tool_by_name(tool_name)
        if not tool_meta:
            return ToolExecutionResult(
                tool_name=tool_name,
                state=ExecutionState.FAILED,
                error=f"Tool '{tool_name}' not found"
            )

        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                import time
                start_time = time.time()

                # Get the tool function
                if hasattr(Tools, tool_name):
                    tool_fn = getattr(Tools, tool_name)
                else:
                    runtime_tools = Tools.runtime_tool_functions()
                    if tool_name in runtime_tools:
                        tool_fn = runtime_tools[tool_name]
                    else:
                        return ToolExecutionResult(
                            tool_name=tool_name,
                            state=ExecutionState.FAILED,
                            error=f"Tool '{tool_name}' not found"
                        )

                # Execute tool (handle async)
                if tool_meta.is_async:
                    result = await tool_fn(*args, **kwargs)
                else:
                    result = tool_fn(*args, **kwargs)

                execution_time = time.time() - start_time

                return ToolExecutionResult(
                    tool_name=tool_name,
                    state=ExecutionState.COMPLETED,
                    result=result,
                    execution_time=execution_time,
                    retry_count=retry_count,
                    context=context or {}
                )

            except Exception as e:
                last_error = str(e)
                retry_count += 1

                if retry_count <= max_retries:
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                else:
                    # Max retries reached
                    return ToolExecutionResult(
                        tool_name=tool_name,
                        state=ExecutionState.FAILED,
                        error=f"{last_error}\n{traceback.format_exc()}",
                        retry_count=retry_count,
                        context=context or {}
                    )

        return ToolExecutionResult(
            tool_name=tool_name,
            state=ExecutionState.FAILED,
            error="Unknown error",
            context=context or {}
        )


# ─────────────────────────────────────────────────────────────────────────────
# DAG Executor
# ─────────────────────────────────────────────────────────────────────────────

class DAGExecutor:
    """Executes a DAG of tool operations"""

    def __init__(self, nodes: List[ToolNode]):
        self.nodes = {node.id: node for node in nodes}
        self.results: Dict[str, ToolExecutionResult] = {}

    async def execute(self) -> Dict[str, ToolExecutionResult]:
        """Execute all nodes in topological order"""

        # Build dependency graph
        in_degree = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for dep_id in node.dependencies:
                in_degree[node.id] += 1

        # Find nodes with no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        completed = set()

        while queue:
            # Execute nodes in parallel if possible
            current_batch = queue[:]
            queue = []

            tasks = []
            for node_id in current_batch:
                node = self.nodes[node_id]

                # Check if all dependencies are satisfied
                deps_met = all(dep_id in completed for dep_id in node.dependencies)

                if deps_met:
                    tasks.append(self._execute_node(node))
                else:
                    # Re-queue
                    queue.append(node_id)

            # Execute batch
            if tasks:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, node_id in enumerate(current_batch):
                    if i < len(batch_results):
                        result = batch_results[i]
                        if isinstance(result, ToolExecutionResult):
                            self.results[node_id] = result
                            if result.state == ExecutionState.COMPLETED:
                                completed.add(node_id)

                                # Reduce in-degree for dependent nodes
                                for other_node in self.nodes.values():
                                    if node_id in other_node.dependencies:
                                        in_degree[other_node.id] -= 1
                                        if in_degree[other_node.id] == 0 and other_node.id not in completed:
                                            queue.append(other_node.id)

            # Prevent infinite loop
            if not tasks and not any(in_degree[nid] == 0 and nid not in completed for nid in self.nodes):
                break

        return self.results

    async def _execute_node(self, node: ToolNode) -> ToolExecutionResult:
        """Execute a single node"""
        node.state = ExecutionState.RUNNING

        result = await ToolExecutor.execute_tool(
            node.tool_name,
            *node.args,
            max_retries=node.max_retries,
            context={"node_id": node.id, "dependencies": node.dependencies},
            **node.kwargs
        )

        node.state = result.state
        node.result = result

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool Context Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tool_context_for_prompt(user_id: int, admin_whitelist: set, include_examples: bool = True) -> str:
    """Build tool context to include in AI prompt for admin users"""

    # Check if user is admin
    if user_id not in admin_whitelist:
        return ""

    if not TOOLS_AVAILABLE:
        return "\n[TOOLS: Not available - tools.py not found]\n"

    context_parts = [
        "\n" + "=" * 80,
        "TOOL CAPABILITIES AVAILABLE",
        "=" * 80,
        "",
        "You have access to powerful tools for web search, file operations, browser automation, and more.",
        "These tools are ONLY available when invoked by admin users.",
        "",
        ToolInspector.format_tools_for_prompt(
            _ALLOWED_TOOL_CATEGORIES,
            max_doc_lines=_MAX_DOC_LINES,
            max_doc_chars=_MAX_DOC_CHARS,
            include_runtime=_INCLUDE_RUNTIME_TOOLS,
        ),
    ]

    if include_examples:
        context_parts.extend([
            "",
            "USAGE EXAMPLES:",
            "-" * 80,
            "",
            "1. Search the web:",
            "   results = Tools.search_internet('latest AI news', num_results=3)",
            "",
            "2. Read a file:",
            "   content = Tools.read_file('report.txt')",
            "",
            "3. List files:",
            "   files = Tools.list_files('.', '*.py')",
            "",
            "4. Get system info:",
            "   location = Tools.get_current_location()",
            "",
            "5. Complex multi-step search:",
            "   result = Tools.complex_search_agent(",
            "       objective='Find latest Python 3.12 features',",
            "       success_criteria='List at least 5 new features with examples',",
            "       max_iterations=3",
            "   )",
            "",
            "6. Multi-step orchestration:",
            "   <<TOOL_DAG>>",
            "   {",
            "     \"label\": \"scan repo for policies\",",
            "     \"nodes\": [",
            "       {\"id\": \"list\", \"tool\": \"list_files\", \"args\": [\".\"], \"kwargs\": {\"pattern\": \"*.md\"}},",
            "       {\"id\": \"open\", \"tool\": \"read_file\", \"args\": [\"README.md\"], \"deps\": [\"list\"]}",
            "     ]",
            "   }",
            "   <<END_DAG>>",
            "",
            "=" * 80,
        ])

    return "\n".join(context_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    'TOOLS_AVAILABLE',
    'ExecutionState',
    'ToolMetadata',
    'ToolExecutionResult',
    'ToolNode',
    'ToolInspector',
    'ToolExecutor',
    'DAGExecutor',
    'build_tool_context_for_prompt',
    'ToolDAGRequest',
]
