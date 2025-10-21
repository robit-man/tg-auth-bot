#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_safety.py — lightweight safety, validation, and policy helpers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


BANNED_TERMS = {
    "disallowed": [
        r"(?i)terrorist",
        r"(?i)bioweapon",
    ]
}


def load_policy(path: Optional[str]) -> Dict[str, Iterable[str]]:
    if not path:
        return BANNED_TERMS
    try:
        data = json.loads(Path(path).read_text())
        return data
    except Exception:
        return BANNED_TERMS


def validate_tool_output(tool_name: str, output: Any):
    """Basic schema/policy validation for tool results."""
    if tool_name == "search_internet":
        if not isinstance(output, list):
            raise ValueError("search_internet expected list output")
        for item in output:
            if not isinstance(item, dict):
                raise ValueError("search_internet items must be dicts")
            if "url" not in item:
                raise ValueError("search_internet result missing url")


def validate_final_reply(reply: str, policy: Optional[Dict[str, Iterable[str]]] = None):
    if not reply:
        raise ValueError("reply is empty")
    if len(reply) > 4000:
        raise ValueError("reply exceeds length limit")
    policy = policy or BANNED_TERMS
    for _, patterns in policy.items():
        for pattern in patterns:
            if re.search(pattern, reply):
                raise ValueError(f"reply violates policy pattern: {pattern}")


def validate_plan_json(raw: str) -> str:
    """
    Extract JSON code block or raw JSON from LLM plan response.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
            if raw.strip().startswith(("json", "JSON")):
                raw = parts[2]
    raw = raw.strip()
    # ensure valid JSON
    json.loads(raw)
    return raw
