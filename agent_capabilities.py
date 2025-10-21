#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_capabilities.py — Tool metadata registry, embeddings, and discovery helpers.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from tools import Tools  # type: ignore
except Exception:
    Tools = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

_EMBEDDER = None
_TOOL_INDEX: List[Tuple[str, str]] = []
_TOOL_CACHE: Dict[str, Callable[..., Any]] = {}


def _lazy_embedder():
    global _EMBEDDER
    if _EMBEDDER is None and SentenceTransformer:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def build_tool_index():
    """Populate tool metadata for later lookup."""
    global _TOOL_INDEX, _TOOL_CACHE
    if Tools is None:
        return
    entries: List[Tuple[str, str]] = []
    for name in dir(Tools):
        if name.startswith("_"):
            continue
        attr = getattr(Tools, name)
        if not callable(attr):
            continue
        doc = inspect.getdoc(attr) or ""
        sig = ""
        try:
            sig = str(inspect.signature(attr))
        except Exception:
            sig = "(...)"
        description = f"{name}{sig}\n{doc}"
        entries.append((name, description))
        _TOOL_CACHE[name] = attr
    _TOOL_INDEX = entries


def resolve_tool_callable(name: str) -> Optional[Callable[..., Any]]:
    if not _TOOL_CACHE:
        build_tool_index()
    return _TOOL_CACHE.get(name)


def suggest_tools(query: str, recent_tools: Optional[str] = None, top_k: int = 4) -> List[str]:
    if not _TOOL_INDEX:
        build_tool_index()
    embedder = _lazy_embedder()
    if not embedder:
        return [name for (name, _) in _TOOL_INDEX[:top_k]]

    corpus = [desc for (_, desc) in _TOOL_INDEX]
    query_text = query + ("\n" + recent_tools if recent_tools else "")
    query_vec = embedder.encode(query_text)
    tool_vecs = embedder.encode(corpus)

    ranked: List[Tuple[float, str]] = []
    for (name, _), vec in zip(_TOOL_INDEX, tool_vecs):
        denom = _norm(query_vec) * _norm(vec)
        if denom == 0:
            score = 0.0
        else:
            score = float(sum(a * b for a, b in zip(query_vec, vec)) / denom)
        ranked.append((score, name))
    ranked.sort(reverse=True)
    return [name for (_, name) in ranked[:top_k]]
