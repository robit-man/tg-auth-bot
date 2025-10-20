#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
AI Tool Bridge - Connects AI with tool execution capabilities

This module enhances the AI prompt building process to include tool awareness
and provides mechanisms for the AI to request and execute tools through
natural language.

Features:
- Automatic tool context injection for admin users
- Tool request parsing from AI responses
- Safe tool execution with sandboxing
- Result formatting for AI consumption
"""

import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from tool_integration import (
        TOOLS_AVAILABLE,
        ToolInspector,
        ToolExecutor,
        ToolNode,
        DAGExecutor,
        build_tool_context_for_prompt,
        ExecutionState,
        ToolExecutionResult,
        ToolDAGRequest,
    )
    TOOL_INTEGRATION_AVAILABLE = True
except ImportError:
    TOOL_INTEGRATION_AVAILABLE = False
    TOOLS_AVAILABLE = False
    ToolInspector = None  # type: ignore
    ToolExecutor = None  # type: ignore
    ToolNode = None  # type: ignore
    DAGExecutor = None  # type: ignore
    ExecutionState = None  # type: ignore
    ToolExecutionResult = None  # type: ignore
    ToolDAGRequest = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Tool Request Parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolRequest:
    """Parsed tool request from AI response"""
    tool_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    original_text: str


class ToolRequestParser:
    """Parse tool requests from AI-generated text"""

    # Pattern to match tool calls like: Tools.search_internet('query', num_results=5)
    TOOL_CALL_PATTERN = re.compile(
        r'Tools\.(\w+)\((.*?)\)',
        re.DOTALL
    )

    # Pattern to match await statements: await Tools.tool_name(...)
    ASYNC_TOOL_PATTERN = re.compile(
        r'await\s+Tools\.(\w+)\((.*?)\)',
        re.DOTALL
    )

    DAG_BLOCK_PATTERN = re.compile(
        r'<<TOOL_DAG>>(.*?)<<END_DAG>>',
        re.DOTALL | re.IGNORECASE
    )

    @staticmethod
    def parse_tool_requests(text: str) -> List[ToolRequest]:
        """Extract tool requests from text"""
        requests = []

        # Find all tool calls (both sync and async)
        for pattern in [ToolRequestParser.ASYNC_TOOL_PATTERN, ToolRequestParser.TOOL_CALL_PATTERN]:
            for match in pattern.finditer(text):
                tool_name = match.group(1)
                args_text = match.group(2).strip()

                try:
                    # Parse arguments (simple approach - handles strings, numbers, booleans)
                    args, kwargs = ToolRequestParser._parse_arguments(args_text)

                    requests.append(ToolRequest(
                        tool_name=tool_name,
                        args=args,
                        kwargs=kwargs,
                        original_text=match.group(0)
                    ))
                except Exception as e:
                    # Skip malformed requests
                    continue

        return requests

    @staticmethod
    def _parse_arguments(args_text: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Parse function arguments from text.
        This is a simplified parser - for production use, consider ast.literal_eval
        """
        if not args_text:
            return [], {}

        args = []
        kwargs = {}

        # Split by commas (simple approach - doesn't handle nested structures)
        parts = []
        current = []
        paren_depth = 0
        quote_char = None

        for char in args_text + ',':
            if char in ('"', "'") and (not quote_char or quote_char == char):
                quote_char = char if not quote_char else None
            elif not quote_char:
                if char in ('(', '[', '{'):
                    paren_depth += 1
                elif char in (')', ']', '}'):
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    parts.append(''.join(current).strip())
                    current = []
                    continue
            current.append(char)

        # Process each part
        for part in parts:
            if not part:
                continue

            # Check if it's a keyword argument
            if '=' in part and not part.startswith('"') and not part.startswith("'"):
                key, value = part.split('=', 1)
                kwargs[key.strip()] = ToolRequestParser._parse_value(value.strip())
            else:
                args.append(ToolRequestParser._parse_value(part))

        return args, kwargs

    @staticmethod
    def _parse_value(value_str: str) -> Any:
        """Parse a single value from string"""
        value_str = value_str.strip()

        # String
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        # Boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # None
        if value_str.lower() == 'none':
            return None

        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # List (simple)
        if value_str.startswith('[') and value_str.endswith(']'):
            return value_str  # Return as string for now

        # Dict (simple)
        if value_str.startswith('{') and value_str.endswith('}'):
            return value_str  # Return as string for now

        # Default: return as string
        return value_str

    @staticmethod
    def parse_dag_requests(text: str, *, max_nodes: int = 12) -> List[ToolDAGRequest]:
        """Extract DAG execution requests embedded between <<TOOL_DAG>> markers."""
        dag_requests: List[ToolDAGRequest] = []
        for match in ToolRequestParser.DAG_BLOCK_PATTERN.finditer(text):
            raw_block = match.group(0)
            payload = (match.group(1) or "").strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except Exception:
                continue
            nodes_payload = data.get("nodes") if isinstance(data, dict) else None
            if not isinstance(nodes_payload, list):
                continue
            label = str(data.get("label") or data.get("name") or "Tool DAG")
            nodes: List[ToolNode] = []
            for idx, node_spec in enumerate(nodes_payload):
                if not isinstance(node_spec, dict):
                    continue
                tool_name = node_spec.get("tool") or node_spec.get("tool_name")
                node_id = node_spec.get("id") or f"step_{idx+1}"
                if not tool_name:
                    continue
                args = node_spec.get("args", [])
                if not isinstance(args, list):
                    args = [args]
                kwargs = node_spec.get("kwargs", {})
                if not isinstance(kwargs, dict):
                    kwargs = {}
                deps = node_spec.get("deps", node_spec.get("dependencies", []))
                if isinstance(deps, (list, tuple, set)):
                    dependencies = [str(d) for d in deps]
                elif deps:
                    dependencies = [str(deps)]
                else:
                    dependencies = []
                try:
                    max_retries = int(node_spec.get("max_retries", 2))
                except (TypeError, ValueError):
                    max_retries = 2
                nodes.append(
                    ToolNode(
                        id=str(node_id),
                        tool_name=str(tool_name),
                        args=args,
                        kwargs=kwargs,
                        dependencies=dependencies,
                        max_retries=max(0, max_retries),
                    )
                )
                if len(nodes) >= max_nodes:
                    break
            if nodes:
                dag_requests.append(ToolDAGRequest(raw_block=raw_block, nodes=nodes, summary_label=label))
        return dag_requests


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedPromptBuilder:
    """Builds AI prompts with tool context for admin users"""

    @staticmethod
    def inject_tool_context(
        base_prompt: dict,
        user_id: int,
        admin_whitelist: set,
        include_examples: bool = True
    ) -> dict:
        """
        Inject tool context into base AI prompt for admin users

        Args:
            base_prompt: The base prompt dict (with 'model', 'messages', etc.)
            user_id: ID of the user making the request
            admin_whitelist: Set of admin user IDs
            include_examples: Whether to include usage examples

        Returns:
            Enhanced prompt dict with tool context
        """
        if not TOOL_INTEGRATION_AVAILABLE or not TOOLS_AVAILABLE:
            return base_prompt

        # Build tool context
        tool_context = build_tool_context_for_prompt(user_id, admin_whitelist, include_examples)

        if not tool_context:
            return base_prompt

        # Inject into system message
        enhanced_prompt = base_prompt.copy()

        if 'messages' in enhanced_prompt:
            messages = enhanced_prompt['messages'][:]

            # Find system message
            system_idx = None
            for i, msg in enumerate(messages):
                if msg.get('role') == 'system':
                    system_idx = i
                    break

            if system_idx is not None:
                # Append to existing system message
                current_system = messages[system_idx]['content']
                messages[system_idx] = {
                    'role': 'system',
                    'content': current_system + "\n\n" + tool_context
                }
            else:
                # Prepend new system message
                messages.insert(0, {
                    'role': 'system',
                    'content': tool_context
                })

            enhanced_prompt['messages'] = messages

        return enhanced_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution Coordinator
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutionCoordinator:
    """Coordinates tool execution from AI responses"""

    def __init__(self, user_id: int, admin_whitelist: set):
        self.user_id = user_id
        self.admin_whitelist = admin_whitelist
        self.execution_history: List[ToolExecutionResult] = []

    async def process_ai_response(
        self,
        ai_response: str,
        auto_execute: bool = False,
        max_tools: int = 5,
        max_dag_nodes: int = 8,
        max_dags: int = 2,
    ) -> Tuple[str, List[ToolExecutionResult]]:
        """
        Process AI response, extract tool requests, and optionally execute them

        Args:
            ai_response: The AI's response text
            auto_execute: Whether to automatically execute discovered tools
            max_tools: Maximum number of tools to execute in one response

        Returns:
            Tuple of (modified_response, execution_results)
        """
        if not TOOL_INTEGRATION_AVAILABLE or not TOOLS_AVAILABLE:
            return ai_response, []

        modified_response = ai_response
        all_results: List[ToolExecutionResult] = []

        dag_requests = ToolRequestParser.parse_dag_requests(modified_response, max_nodes=max_dag_nodes)[:max_dags]
        simple_requests = ToolRequestParser.parse_tool_requests(modified_response)[:max_tools]

        if not auto_execute:
            if not dag_requests and not simple_requests:
                return ai_response, []
            summary_lines = ["[TOOLS DETECTED - Not executed]"]
            for dag in dag_requests:
                tool_names = ", ".join(node.tool_name for node in dag.nodes)
                summary_lines.append(f"• DAG[{dag.summary_label}] → {tool_names}")
            for req in simple_requests:
                arg_preview = ", ".join(map(str, req.args))
                if req.kwargs:
                    kw_preview = ", ".join(f"{k}={v}" for k, v in req.kwargs.items())
                    arg_preview = ", ".join(filter(None, [arg_preview, kw_preview]))
                summary_lines.append(f"• {req.tool_name}({arg_preview})")
            return ai_response + "\n\n" + "\n".join(summary_lines), []

        # Execute DAG requests first
        for dag in dag_requests:
            if len(dag.nodes) > max_dag_nodes:
                replacement = f"[TOOL DAG SKIPPED] {dag.summary_label}: too many nodes ({len(dag.nodes)} > {max_dag_nodes})"
                modified_response = modified_response.replace(dag.raw_block, replacement, 1)
                continue
            missing = [
                node.tool_name for node in dag.nodes
                if ToolInspector.get_tool_by_name(node.tool_name) is None
            ]
            if missing:
                replacement = f"[TOOL DAG ERROR] {dag.summary_label}: unknown tools {', '.join(sorted(set(missing)))}"
                modified_response = modified_response.replace(dag.raw_block, replacement, 1)
                continue
            executor = DAGExecutor(dag.nodes)
            dag_results_map = await executor.execute()
            summary_lines = [
                "[TOOL DAG EXECUTED]",
                f"Label: {dag.summary_label}",
                "-" * 50,
            ]
            for node in dag.nodes:
                result = dag_results_map.get(node.id)
                if isinstance(result, ToolExecutionResult):
                    all_results.append(result)
                    self.execution_history.append(result)
                    summary_lines.append(
                        f"- {node.tool_name}: {result.state.value.upper()} → {self._format_result_value(result.result)}"
                    )
                else:
                    summary_lines.append(f"- {node.tool_name}: no result")
            summary_lines.append("-" * 50)
            replacement = "\n".join(summary_lines)
            modified_response = modified_response.replace(dag.raw_block, replacement, 1)

        # Execute simple Tools.* invocations
        for req in simple_requests:
            result = await ToolExecutor.execute_tool(
                req.tool_name,
                *req.args,
                **req.kwargs
            )
            all_results.append(result)
            self.execution_history.append(result)
            replacement = f"[EXECUTED: {req.original_text}]\nResult: {self._format_single_result(result)}"
            modified_response = modified_response.replace(req.original_text, replacement, 1)

        results_text = self._format_execution_results(all_results)
        if results_text:
            modified_response = modified_response.rstrip() + "\n\n" + results_text

        return modified_response, all_results

    def _format_execution_results(self, results: List[ToolExecutionResult]) -> str:
        """Format execution results for display"""
        if not results:
            return ""

        lines = ["\n[TOOL EXECUTION RESULTS]", "=" * 60]

        for i, result in enumerate(results, 1):
            lines.append(f"\n{i}. {result.tool_name}")
            lines.append(f"   Status: {result.state.value}")

            if result.state == ExecutionState.COMPLETED:
                lines.append(f"   Time: {result.execution_time:.2f}s")
                lines.append(f"   Result: {self._format_result_value(result.result)}")
            elif result.state == ExecutionState.FAILED:
                lines.append(f"   Error: {result.error}")

            if result.retry_count > 0:
                lines.append(f"   Retries: {result.retry_count}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_single_result(self, result: ToolExecutionResult) -> str:
        """Format a single result concisely"""
        if result.state == ExecutionState.COMPLETED:
            return self._format_result_value(result.result)
        elif result.state == ExecutionState.FAILED:
            return f"ERROR: {result.error}"
        else:
            return f"STATUS: {result.state.value}"

    def _format_result_value(self, value: Any) -> str:
        """Format result value for display"""
        if value is None:
            return "(None)"

        # Truncate long strings
        value_str = str(value)
        if len(value_str) > 500:
            return value_str[:500] + "... (truncated)"

        return value_str


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    'TOOL_INTEGRATION_AVAILABLE',
    'TOOLS_AVAILABLE',
    'ToolRequest',
    'ToolRequestParser',
    'EnhancedPromptBuilder',
    'ToolExecutionCoordinator',
]
