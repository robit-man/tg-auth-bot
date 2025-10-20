#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool Schema System - Claude Code style tool definitions and execution

This module provides proper tool schema generation from Python functions,
modeled after Claude Code's own tool calling architecture with XML-style
function definitions and structured execution.
"""

import json
import inspect
import re
from typing import Dict, List, Any, Optional, Callable, get_type_hints
from dataclasses import dataclass, field
from enum import Enum

try:
    from tools import Tools
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    Tools = None


# ─────────────────────────────────────────────────────────────────────────────
# Type Mapping
# ─────────────────────────────────────────────────────────────────────────────

def python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert Python type to JSON schema type"""
    type_str = str(py_type)

    if py_type == str or 'str' in type_str:
        return {"type": "string"}
    elif py_type == int or 'int' in type_str:
        return {"type": "integer"}
    elif py_type == float or 'float' in type_str:
        return {"type": "number"}
    elif py_type == bool or 'bool' in type_str:
        return {"type": "boolean"}
    elif py_type == list or 'List' in type_str or 'list' in type_str:
        return {"type": "array"}
    elif py_type == dict or 'Dict' in type_str or 'dict' in type_str:
        return {"type": "object"}
    elif 'Optional' in type_str:
        # Extract inner type
        inner_match = re.search(r'Optional\[(.*?)\]', type_str)
        if inner_match:
            inner_type = inner_match.group(1)
            if 'str' in inner_type:
                return {"type": "string", "nullable": True}
            elif 'int' in inner_type:
                return {"type": "integer", "nullable": True}
        return {"type": "string", "nullable": True}
    else:
        return {"type": "string"}


# ─────────────────────────────────────────────────────────────────────────────
# Docstring Parser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedDocstring:
    """Parsed components of a function docstring"""
    summary: str
    description: str
    args: Dict[str, str]  # param_name -> description
    returns: str
    examples: List[str]


class DocstringParser:
    """Parse Google-style and NumPy-style docstrings"""

    @staticmethod
    def parse(docstring: Optional[str]) -> ParsedDocstring:
        """Parse a docstring into structured components"""
        if not docstring:
            return ParsedDocstring("", "", {}, "", [])

        lines = docstring.strip().split('\n')

        summary = ""
        description = []
        args = {}
        returns = ""
        examples = []

        current_section = "summary"
        current_param = None

        for line in lines:
            stripped = line.strip()

            # Detect sections
            if stripped.lower().startswith(('args:', 'arguments:', 'parameters:', 'params:')):
                current_section = "args"
                continue
            elif stripped.lower().startswith(('returns:', 'return:')):
                current_section = "returns"
                continue
            elif stripped.lower().startswith(('example:', 'examples:')):
                current_section = "examples"
                continue
            elif stripped.lower().startswith(('raises:', 'note:', 'notes:', 'warning:')):
                current_section = "other"
                continue

            # Process based on section
            if current_section == "summary" and not summary:
                if stripped:
                    summary = stripped
                    current_section = "description"
            elif current_section == "description":
                if stripped:
                    description.append(stripped)
            elif current_section == "args":
                # Look for parameter definitions like "param_name: description" or "param_name (type): description"
                param_match = re.match(r'(\w+)\s*(?:\([^)]+\))?\s*:\s*(.+)', stripped)
                if param_match:
                    current_param = param_match.group(1)
                    args[current_param] = param_match.group(2)
                elif current_param and stripped:
                    # Continuation of previous parameter description
                    args[current_param] += " " + stripped
            elif current_section == "returns":
                if stripped:
                    returns += " " + stripped if returns else stripped
            elif current_section == "examples":
                if stripped:
                    examples.append(stripped)

        return ParsedDocstring(
            summary=summary,
            description=" ".join(description),
            args=args,
            returns=returns.strip(),
            examples=examples
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Schema Generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolSchema:
    """Complete schema for a tool function"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    returns: Optional[str]
    examples: List[str]
    is_async: bool
    category: str


class ToolSchemaGenerator:
    """Generate proper tool schemas from Python functions"""

    @staticmethod
    def generate_schema(func: Callable, category: str = "general") -> ToolSchema:
        """Generate a complete tool schema from a function"""

        # Get function metadata
        func_name = func.__name__
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func)
        parsed_doc = DocstringParser.parse(docstring)

        # Try to get type hints
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        # Build parameters schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            # Skip self/cls
            if param_name in ('self', 'cls'):
                continue

            # Get type from hints or annotation
            param_type = type_hints.get(param_name, param.annotation)
            if param_type == inspect.Parameter.empty:
                param_type = str  # Default to string

            # Convert to JSON schema
            schema_type = python_type_to_json_schema(param_type)

            # Add description from docstring
            param_desc = parsed_doc.args.get(param_name, f"Parameter {param_name}")
            schema_type["description"] = param_desc

            parameters["properties"][param_name] = schema_type

            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        # Build full schema
        return ToolSchema(
            name=func_name,
            description=parsed_doc.summary or parsed_doc.description or "No description available",
            parameters=parameters,
            returns=parsed_doc.returns,
            examples=parsed_doc.examples,
            is_async=inspect.iscoroutinefunction(func),
            category=category
        )

    @staticmethod
    def generate_all_schemas() -> List[ToolSchema]:
        """Generate schemas for all available tools"""
        if not TOOLS_AVAILABLE or not Tools:
            return []

        schemas = []

        # Get all tool methods from Tools class
        for attr_name in dir(Tools):
            if attr_name.startswith('_'):
                continue

            attr = getattr(Tools, attr_name)
            if not callable(attr):
                continue

            try:
                # Categorize
                doc = inspect.getdoc(attr) or ""
                category = ToolSchemaGenerator._categorize(attr_name, doc)

                # Generate schema
                schema = ToolSchemaGenerator.generate_schema(attr, category)
                schemas.append(schema)
            except Exception as e:
                # Skip tools that can't be processed
                continue

        return schemas

    @staticmethod
    def _categorize(name: str, doc: str) -> str:
        """Categorize a tool"""
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


# ─────────────────────────────────────────────────────────────────────────────
# Tool Formatter (Claude Code Style)
# ─────────────────────────────────────────────────────────────────────────────

class ToolFormatter:
    """Format tools for AI prompts in Claude Code style"""

    @staticmethod
    def format_for_prompt(
        schemas: List[ToolSchema],
        categories: Optional[List[str]] = None,
        max_tools: Optional[int] = None
    ) -> str:
        """
        Format tool schemas for inclusion in AI prompt

        Uses a clear, structured format similar to Claude Code's tool definitions
        """
        if not schemas:
            return "No tools available."

        # Filter by categories
        if categories:
            category_set = {c.lower() for c in categories}
            schemas = [s for s in schemas if s.category.lower() in category_set]

        # Limit number of tools
        if max_tools and max_tools > 0:
            schemas = schemas[:max_tools]

        # Group by category
        by_category: Dict[str, List[ToolSchema]] = {}
        for schema in schemas:
            by_category.setdefault(schema.category, []).append(schema)

        # Build formatted output
        lines = [
            "=" * 80,
            "AVAILABLE TOOLS",
            "=" * 80,
            "",
            f"You have access to {len(schemas)} tools across {len(by_category)} categories.",
            "To use a tool, emit a function call in this format:",
            "",
            "  Tools.tool_name(arg1='value1', arg2='value2')",
            "",
            "For async tools, the system will automatically await them.",
            "",
        ]

        for category in sorted(by_category.keys()):
            tools = by_category[category]

            lines.append("")
            lines.append(f"━━━ {category.upper()} TOOLS ({len(tools)}) ━━━")
            lines.append("")

            for schema in sorted(tools, key=lambda s: s.name):
                lines.append(f"Tool: {schema.name}{'  [async]' if schema.is_async else ''}")
                lines.append(f"Description: {schema.description}")

                # Parameters
                if schema.parameters.get("properties"):
                    lines.append("Parameters:")
                    required = set(schema.parameters.get("required", []))

                    for param_name, param_schema in schema.parameters["properties"].items():
                        param_type = param_schema.get("type", "any")
                        param_desc = param_schema.get("description", "")
                        req_marker = " (required)" if param_name in required else " (optional)"

                        lines.append(f"  • {param_name}: {param_type}{req_marker}")
                        if param_desc:
                            lines.append(f"    {param_desc}")

                # Examples
                if schema.examples:
                    lines.append("Examples:")
                    for example in schema.examples[:2]:  # Max 2 examples
                        lines.append(f"  {example}")

                lines.append("")  # Blank line between tools

        lines.extend([
            "=" * 80,
            "",
            "USAGE NOTES:",
            "• Always check parameter requirements before calling",
            "• String parameters should be quoted: 'value' or \"value\"",
            "• For file operations, paths are relative to workspace unless absolute",
            "• Web searches automatically extract and summarize content",
            "• Browser tools require Chrome/Chromium to be installed",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    @staticmethod
    def format_compact(schemas: List[ToolSchema], max_per_category: int = 5) -> str:
        """Compact format showing tool signatures only"""
        by_category: Dict[str, List[ToolSchema]] = {}
        for schema in schemas:
            by_category.setdefault(schema.category, []).append(schema)

        lines = ["AVAILABLE TOOLS (compact view):", ""]

        for category in sorted(by_category.keys()):
            tools = by_category[category][:max_per_category]
            lines.append(f"{category.upper()}:")
            for schema in tools:
                # Build signature
                params = []
                if schema.parameters.get("properties"):
                    required = set(schema.parameters.get("required", []))
                    for pname in schema.parameters["properties"].keys():
                        if pname in required:
                            params.append(f"{pname}")
                        else:
                            params.append(f"[{pname}]")

                sig = f"{schema.name}({', '.join(params)})"
                lines.append(f"  • {sig}")

            lines.append("")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    'ToolSchema',
    'ToolSchemaGenerator',
    'ToolFormatter',
    'DocstringParser',
    'ParsedDocstring',
]
