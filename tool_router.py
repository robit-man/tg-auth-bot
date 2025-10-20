#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tool_router.py — Automatic tool selection and routing for telegram bot

Based on the examples/main.py routing logic, this module provides intelligent
tool selection based on user intent, with support for:
- LLM-based route selection
- Heuristic keyword matching
- Automatic tool instantiation and execution
- RAG (document) queries
- Web search
- File operations

Routes:
  - direct_answer: Simple Q&A with memory
  - web_search: Search internet and synthesize
  - knowledge_query: RAG over uploaded documents
  - file_operation: File/system tools
  - image_analysis: Vision-based image analysis
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

# Import tool components
try:
    from tool_integration import ToolInspector, ToolExecutor, ToolMetadata
    from tool_schema import ToolSchemaGenerator, ToolFormatter
    TOOL_INTEGRATION_AVAILABLE = True
except ImportError:
    TOOL_INTEGRATION_AVAILABLE = False
    ToolInspector = None  # type: ignore
    ToolExecutor = None  # type: ignore


class RouteType(Enum):
    """Available routing options"""
    DIRECT_ANSWER = "direct_answer"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_QUERY = "knowledge_query"
    FILE_OPERATION = "file_operation"
    IMAGE_ANALYSIS = "image_analysis"
    TOOL_EXECUTION = "tool_execution"


@dataclass
class RouteDecision:
    """Result of route selection"""
    route: RouteType
    confidence: float  # 0.0 to 1.0
    reason: str
    tools_needed: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    success: bool
    tool_name: str
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0


def _summarize_plan_output(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    success = bool(data.get("success", True))
    summary = data.get("summary") or {}
    objective = data.get("objective") or (data.get("plan") or {}).get("objective")
    steps = (data.get("plan") or {}).get("steps") or []
    deliverables = data.get("deliverables")

    total_steps = summary.get("total_steps") if isinstance(summary, dict) else None
    successes = summary.get("successes") if isinstance(summary, dict) else None
    failures = summary.get("failures") if isinstance(summary, dict) else []

    status_icon = "✅" if success else "⚠️"
    parts: List[str] = [f"{status_icon} planner {'completed' if success else 'needs review'}"]
    if objective:
        parts.append(f"objective: {textwrap.shorten(str(objective), width=80, placeholder='…')}")
    if total_steps is not None and successes is not None:
        parts.append(f"steps {successes}/{total_steps}")
    if failures:
        fail_labels = ", ".join(str(f) for f in failures[:3])
        parts.append(f"failed: {fail_labels}")
    if deliverables:
        parts.append(f"deliverables: {textwrap.shorten(str(deliverables), width=80, placeholder='…')}")
    step_titles = [
        textwrap.shorten(str(step.get("title") or step.get("id") or ""), width=60, placeholder="…")
        for step in steps[:3] if isinstance(step, dict)
    ]
    if step_titles:
        parts.append("key steps: " + "; ".join(step_titles))
    return " | ".join(parts)


class IntelligentToolRouter:
    """
    Intelligent tool router that decides what tools to use based on user intent.

    Modeled after examples/main.py auto_route_selection logic:
    1. Analyze user intent with LLM (if available)
    2. Apply heuristic keyword matching
    3. Aggregate scores
    4. Select best route with confidence
    5. Determine required tools
    6. Execute tools with proper error handling
    """

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        enable_llm_routing: bool = True,
        min_confidence_threshold: float = 0.3,
    ):
        """
        Initialize the router

        Args:
            ollama_base_url: Ollama API endpoint
            ollama_model: Model for LLM-based routing decisions
            enable_llm_routing: Whether to use LLM for route selection (vs heuristics only)
            min_confidence_threshold: Minimum confidence to accept a route
        """
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.enable_llm_routing = enable_llm_routing
        self.min_confidence_threshold = min_confidence_threshold

        # Load available tools
        self.available_tools: Dict[str, ToolMetadata] = {}
        if TOOL_INTEGRATION_AVAILABLE and ToolInspector:
            tools = ToolInspector.get_all_tools()
            self.available_tools = {t.name: t for t in tools}

    def analyze_intent(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> RouteDecision:
        """
        Analyze user intent and decide which route to take.

        Args:
            user_message: The user's input message
            context: Optional context (chat history, user info, etc.)

        Returns:
            RouteDecision with selected route and confidence
        """
        # Calculate heuristic scores for each route
        heuristic_scores = self._calculate_heuristic_scores(user_message)

        # If LLM routing is enabled, get LLM scores
        llm_scores: Dict[RouteType, float] = {}
        if self.enable_llm_routing:
            try:
                llm_scores = self._get_llm_route_scores(user_message, context)
            except Exception as e:
                print(f"[router] LLM routing failed, using heuristics only: {e}")

        # Aggregate scores
        aggregated = self._aggregate_scores(heuristic_scores, llm_scores)

        # Select best route
        best_route = max(aggregated, key=aggregated.get)
        best_score = aggregated[best_route]

        # Calculate confidence (similar to examples/main.py logic)
        scores_sorted = sorted(aggregated.values(), reverse=True)
        if len(scores_sorted) > 1:
            margin = scores_sorted[0] - scores_sorted[1]
            denominator = scores_sorted[0] + scores_sorted[1] + 1e-6
            confidence = max(0.1, min(1.0, 0.5 + margin / denominator))
        else:
            confidence = 0.5

        # Determine which tools are needed
        tools_needed = self._determine_tools_for_route(best_route, user_message)

        # Build reason
        reason_parts = []
        if llm_scores and best_route in llm_scores:
            reason_parts.append("LLM recommendation")
        if heuristic_scores.get(best_route, 0) > 0:
            reason_parts.append("keyword match")
        if not reason_parts:
            reason_parts.append("default fallback")
        reason = ", ".join(reason_parts)

        return RouteDecision(
            route=best_route,
            confidence=confidence,
            reason=reason,
            tools_needed=tools_needed,
            parameters={},
            details={
                "heuristic_scores": {r.value: s for r, s in heuristic_scores.items()},
                "llm_scores": {r.value: s for r, s in llm_scores.items()} if llm_scores else {},
                "aggregated": {r.value: s for r, s in aggregated.items()},
            }
        )

    def _calculate_heuristic_scores(self, user_message: str) -> Dict[RouteType, float]:
        """
        Calculate heuristic scores based on keyword matching.

        Similar to _heuristic_route_bias in examples/main.py
        """
        msg_lower = user_message.lower()
        scores: Dict[RouteType, float] = {r: 0.0 for r in RouteType}

        # Web search keywords
        if any(kw in msg_lower for kw in [
            "search", "look up", "find out", "what is", "who is", "where is",
            "google", "web", "internet", "latest", "recent", "news", "current"
        ]):
            scores[RouteType.WEB_SEARCH] += 1.5

        # Knowledge/document query keywords
        if any(kw in msg_lower for kw in [
            "document", "pdf", "file", "uploaded", "in the ", "according to",
            "clause", "section", "page", "reference", "citation"
        ]):
            scores[RouteType.KNOWLEDGE_QUERY] += 1.8

        # File operation keywords
        if any(kw in msg_lower for kw in [
            "read file", "write file", "save", "create file", "delete file",
            "list files", "find file", "directory", "folder"
        ]):
            scores[RouteType.FILE_OPERATION] += 1.5

        # Image analysis keywords
        if any(kw in msg_lower for kw in [
            "image", "picture", "photo", "screenshot", "describe image",
            "what's in", "analyze image", "visual"
        ]):
            scores[RouteType.IMAGE_ANALYSIS] += 1.8

        # Tool execution keywords (explicit tool calls)
        if any(kw in msg_lower for kw in [
            "use tool", "call", "execute", "run", "tools."
        ]):
            scores[RouteType.TOOL_EXECUTION] += 1.5

        planning_triggers = [
            "plan", "multi-step", "multi step", "roadmap", "strategy",
            "workflow", "sequence", "pipeline", "project plan", "long task",
            "break down", "step by step", "architecture", "design", "blueprint",
            "script", "build", "implement"
        ]
        if any(kw in msg_lower for kw in planning_triggers) or len(user_message) > 160:
            scores[RouteType.TOOL_EXECUTION] += 2.0

        # Default: direct answer gets small boost if nothing else matches
        if sum(scores.values()) < 0.5:
            scores[RouteType.DIRECT_ANSWER] = 0.8

        return scores

    def _get_llm_route_scores(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[RouteType, float]:
        """
        Use LLM to score different routes.

        Similar to examples/main.py decide_from_options approach.
        """
        try:
            import ollama
        except ImportError:
            return {}

        routes_desc = {
            RouteType.DIRECT_ANSWER: "Direct conversational answer (no external tools needed)",
            RouteType.WEB_SEARCH: "Search the internet for current information",
            RouteType.KNOWLEDGE_QUERY: "Query uploaded documents and knowledge base",
            RouteType.FILE_OPERATION: "File system operations (read, write, list files)",
            RouteType.IMAGE_ANALYSIS: "Analyze images using vision models",
            RouteType.TOOL_EXECUTION: "Execute specific tools or utilities",
        }

        system_prompt = """You are a routing assistant. Analyze the user's request and determine which approach is most appropriate.
Score each option from 0.0 to 1.0 based on relevance. Return ONLY valid JSON."""

        user_prompt = f"""User request: "{user_message}"

Available routes:
{chr(10).join(f"- {r.value}: {desc}" for r, desc in routes_desc.items())}

Return JSON with scores for each route (0.0 to 1.0):
{{
  "direct_answer": 0.0,
  "web_search": 0.0,
  "knowledge_query": 0.0,
  "file_operation": 0.0,
  "image_analysis": 0.0,
  "tool_execution": 0.0
}}"""

        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": 0.2},
            )

            result = json.loads(response["message"]["content"])

            # Convert to RouteType keys
            scores: Dict[RouteType, float] = {}
            for route in RouteType:
                score = result.get(route.value, 0.0)
                scores[route] = max(0.0, min(1.0, float(score)))

            return scores

        except Exception as e:
            print(f"[router] LLM scoring failed: {e}")
            return {}

    def _aggregate_scores(
        self,
        heuristic: Dict[RouteType, float],
        llm: Dict[RouteType, float],
    ) -> Dict[RouteType, float]:
        """
        Aggregate heuristic and LLM scores.

        Similar to examples/main.py aggregation logic:
        - Base score from heuristics
        - Bonus for LLM agreement
        """
        aggregated: Dict[RouteType, float] = {}

        for route in RouteType:
            score = heuristic.get(route, 0.0)

            # Add LLM score if available
            if llm and route in llm:
                score += llm[route] * 2.0  # Weight LLM scores higher

            aggregated[route] = score

        return aggregated

    def _determine_tools_for_route(
        self,
        route: RouteType,
        user_message: str
    ) -> List[str]:
        """
        Determine which specific tools are needed for the selected route.
        """
        if not TOOL_INTEGRATION_AVAILABLE or not self.available_tools:
            return []

        tools_needed = []
        msg_lower = user_message.lower()

        if route == RouteType.WEB_SEARCH:
            if "search_internet" in self.available_tools:
                tools_needed.append("search_internet")
            if "fetch_webpage" in self.available_tools and ("url" in msg_lower or "website" in msg_lower):
                tools_needed.append("fetch_webpage")

        elif route == RouteType.FILE_OPERATION:
            # Check for specific file operations
            if "read" in msg_lower and "read_file" in self.available_tools:
                tools_needed.append("read_file")
            if any(kw in msg_lower for kw in ["write", "save", "create"]) and "write_file" in self.available_tools:
                tools_needed.append("write_file")
            if "list" in msg_lower and "list_dir" in self.available_tools:
                tools_needed.append("list_dir")
            if "find" in msg_lower and "find_files" in self.available_tools:
                tools_needed.append("find_files")

        elif route == RouteType.IMAGE_ANALYSIS:
            if "describe_image" in self.available_tools:
                tools_needed.append("describe_image")

        elif route == RouteType.TOOL_EXECUTION:
            if "plan_complex_task" in self.available_tools:
                plan_keywords = [
                    "plan", "roadmap", "strategy", "multi-step", "multi step",
                    "workflow", "project plan", "long project", "break down", "step by step",
                    "architecture", "design", "script", "build", "implement"
                ]
                if any(kw in msg_lower for kw in plan_keywords) or len(user_message) > 160:
                    tools_needed.append("plan_complex_task")
            # fall back to read/write heuristics to allow combined execution
            if "read_file" in self.available_tools and "read" in msg_lower:
                tools_needed.append("read_file")

        return tools_needed

    async def execute_route(
        self,
        decision: RouteDecision,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[ToolExecutionResult]]:
        """
        Execute the selected route with appropriate tools.

        Args:
            decision: The routing decision
            user_message: Original user message
            context: Optional context

        Returns:
            (response_text, list of tool execution results)
        """
        results: List[ToolExecutionResult] = []

        if not TOOL_INTEGRATION_AVAILABLE or not ToolExecutor:
            return "Tool execution not available", results

        # Execute each required tool
        for tool_name in decision.tools_needed:
            try:
                # Extract parameters from user message (simplified)
                params = self._extract_tool_parameters(tool_name, user_message, context)

                # Execute tool
                result = await ToolExecutor.execute_tool(tool_name, **params)

                results.append(ToolExecutionResult(
                    success=result.success,
                    tool_name=tool_name,
                    output=result.output,
                    error=result.error,
                    execution_time=result.execution_time,
                ))

            except Exception as e:
                results.append(ToolExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    output=None,
                    error=str(e),
                ))

        # Build response from tool results
        response = self._build_response_from_results(decision, results, user_message)

        return response, results

    def _extract_tool_parameters(
        self,
        tool_name: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract parameters for a specific tool from user message.

        This is a simplified version - in production, you'd use LLM to extract parameters.
        """
        params: Dict[str, Any] = {}

        if tool_name == "search_internet":
            # Extract search query
            params["topic"] = user_message
            params["num_results"] = 5

        elif tool_name == "read_file":
            # Extract file path
            match = re.search(r'["\']([^"\']+)["\']', user_message)
            if match:
                params["file_path"] = match.group(1)

        elif tool_name == "write_file":
            # Extract file path and content
            match = re.search(r'["\']([^"\']+)["\']', user_message)
            if match:
                params["file_path"] = match.group(1)
            # Content would need to be extracted separately

        elif tool_name == "plan_complex_task":
            params["objective"] = user_message.strip()
            params["constraints"] = None
            params["deliverables"] = None

        return params

    def _build_response_from_results(
        self,
        decision: RouteDecision,
        results: List[ToolExecutionResult],
        user_message: str,
    ) -> str:
        """
        Build a user-facing response from tool execution results.
        """
        if not results:
            return f"Selected route: {decision.route.value}, but no tools were executed."

        # Check for failures
        failures = [r for r in results if not r.success]
        if failures:
            error_msgs = [f"- {r.tool_name}: {r.error}" for r in failures]
            return f"Tool execution failed:\n" + "\n".join(error_msgs)

        # Format successful results
        response_parts = []
        for result in results:
            if result.output:
                if result.tool_name == "plan_complex_task":
                    digest = _summarize_plan_output(result.output)
                    if digest:
                        response_parts.append(f"[{result.tool_name}]\n{digest}")
                        continue
                response_parts.append(f"[{result.tool_name}]\n{result.output}")

        return "\n\n".join(response_parts) if response_parts else "Tools executed successfully (no output)"


def create_router(
    ollama_model: str = "llama3.2",
    enable_llm: bool = True,
) -> IntelligentToolRouter:
    """
    Factory function to create a router instance.

    Args:
        ollama_model: Model to use for LLM routing
        enable_llm: Whether to enable LLM-based routing

    Returns:
        Configured IntelligentToolRouter instance
    """
    return IntelligentToolRouter(
        ollama_model=ollama_model,
        enable_llm_routing=enable_llm,
    )
