#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_context.py — Maintains a lightweight knowledge graph and thread workspace
for global agent context.

Responsibilities
----------------
• Persist a graph of entities, relationships, and conversational threads
• Surface thread matches for new user goals to enable automatic route hints
• Update the graph after each invocation using an LLM-generated delta
• Provide context snippets summarising past threads when relevant
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cognition import CognitionChat, CognitionConfig
from context import RetrievalReport
from memory import MemoryRow, MemoryStore

GRAPH_SCHEMA_VERSION = 1
GLOBAL_CONTEXT_FILENAME = "global_context.json"

# JSON schema used to validate/repair LLM-produced graph deltas.
GRAPH_UPDATE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["entities", "threads"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "summary"],
                "properties": {
                    "name": {"type": "string", "minLength": 2},
                    "type": {"type": "string"},
                    "summary": {"type": "string", "minLength": 5},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "supporting_memories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "supporting_docs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "related_to": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["target", "relation"],
                            "properties": {
                                "target": {"type": "string", "minLength": 2},
                                "relation": {"type": "string", "minLength": 2},
                                "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "rationale": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                        "maxItems": 12,
                    },
                },
                "additionalProperties": False,
            },
        },
        "threads": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic"],
                "properties": {
                    "topic": {"type": "string", "minLength": 3},
                    "summary": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "primary_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "preferred_route": {"type": "string"},
                    "status": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

GRAPH_SYSTEM_PROMPT = (
    "You maintain a compact knowledge graph for an operations assistant.\n"
    "Given recent conversation snippets, summarize key entities, relationships, and threads.\n"
    "Always return JSON that matches the provided schema. Prefer stable entity names.\n"
    "Use memory ids (mem#...) or chunk ids (#abcd1234) whenever you cite evidence."
)

PROACTIVE_SYSTEM_PROMPT = (
    "You craft the first, human-inspired message for an ongoing assistant conversation.\n"
    "Given recent threads, key entities, and memory snippets, write a short, friendly opener.\n"
    "Encourage continuity with past work, invite clarification, and keep it under 120 words."
)


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _truncate(text: str, max_len: int = 600) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _normalize(text: str) -> str:
    return text.strip().lower()


@dataclass
class ThreadMatch:
    thread_id: str
    topic: str
    summary: str
    preferred_route: Optional[str] = None
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    node_summaries: List[str] = field(default_factory=list)

    def as_context_snippets(self) -> List[str]:
        lines = [f"[global-thread {self.thread_id}] Topic: {self.topic}"]
        if self.summary:
            lines.append(self.summary)
        for node_line in self.node_summaries:
            lines.append(node_line)
        return ["\n".join(lines)]


class GlobalContextWorkspace:
    """Persistent knowledge graph + thread workspace (LLM-assisted)."""

    def __init__(
        self,
        *,
        path: Optional[Path] = None,
        config: Optional[CognitionConfig] = None,
    ) -> None:
        self.path = path or Path(__file__).resolve().parent / GLOBAL_CONTEXT_FILENAME
        self.config = config or CognitionConfig.load()
        self.state = self._load()
        self._refresh_indexes()

    # ── public API -----------------------------------------------------------
    def match_goal(self, goal: str) -> Optional[ThreadMatch]:
        goal_norm = _normalize(goal)
        if not goal_norm:
            return None

        best: Optional[Tuple[float, ThreadMatch]] = None
        for thread_id, thread in self.state.get("threads", {}).items():
            topic = thread.get("topic", "")
            summary = thread.get("summary", "")
            kw_list = thread.get("keywords", []) or []
            score = 0.0
            matched_keywords: List[str] = []

            if topic and _normalize(topic) in goal_norm:
                score += 2.5

            for kw in kw_list:
                if kw and _normalize(kw) in goal_norm:
                    matched_keywords.append(kw)
                    score += 1.5

            node_summaries: List[str] = []
            for node_id in thread.get("node_ids", []) or []:
                node = self.state.get("nodes", {}).get(node_id)
                if not node:
                    continue
                name = node.get("name", "")
                summary_line = node.get("summary", "")
                if name and _normalize(name) in goal_norm:
                    score += 0.75
                for kw in node.get("keywords", []) or []:
                    if kw and _normalize(kw) in goal_norm:
                        matched_keywords.append(kw)
                        score += 0.35
                node_summaries.append(f"[global-node {name}] {summary_line}")

            if score <= 0.0:
                continue

            confidence = min(1.0, score / 5.0)
            match = ThreadMatch(
                thread_id=thread_id,
                topic=topic,
                summary=summary,
                preferred_route=thread.get("preferred_route"),
                confidence=confidence,
                keywords=sorted(set(matched_keywords)),
                node_summaries=node_summaries[:3],
            )

            if not best or score > best[0]:
                best = (score, match)

        return best[1] if best else None

    def infer_route_bias(self, goal: str) -> Optional[Dict[str, Any]]:
        match = self.match_goal(goal)
        if not match or not match.preferred_route:
            return None
        return {
            "route": match.preferred_route,
            "thread_match": match,
            "reason": f"Continuing thread '{match.topic}'",
            "confidence": match.confidence,
        }

    def update_after_invocation(
        self,
        *,
        session_id: Optional[str],
        user_goal: str,
        invoked_route: str,
        success: bool,
        user_input: Optional[str],
        user_memory_id: Optional[str],
        assistant_memory_id: Optional[str],
        thread_match: Optional[ThreadMatch],
        retrieval_report: Optional[RetrievalReport],
        memory_store: MemoryStore,
    ) -> None:
        try:
            recent_memories = self._serialize_memories(
                memory_store.fetch_recent(session_id=session_id, limit=14)
            )
        except Exception:
            recent_memories = []

        payload: Dict[str, Any] = {
            "user_goal": user_goal,
            "invoked_route": invoked_route,
            "success": bool(success),
            "user_input": user_input,
            "thread_context": {
                "thread_id": getattr(thread_match, "thread_id", None),
                "topic": getattr(thread_match, "topic", None),
                "preferred_route": getattr(thread_match, "preferred_route", None),
            }
            if thread_match
            else None,
            "recent_memories": recent_memories,
            "retrieval_report": retrieval_report.to_dict() if retrieval_report else None,
            "existing_threads": self._compact_threads(),
        }

        llm = CognitionChat(self.config)
        llm.set_system(GRAPH_SYSTEM_PROMPT)
        try:
            update = llm.structured_json(
                user_message=json.dumps(payload, ensure_ascii=False),
                json_schema=GRAPH_UPDATE_SCHEMA,
                stream=False,
            )
            if not isinstance(update, dict):
                update = json.loads(update)
        except Exception:
            return

        self._merge_update(
            update,
            session_id=session_id,
            invoked_route=invoked_route,
            success=success,
            user_memory_id=user_memory_id,
            assistant_memory_id=assistant_memory_id,
            retrieval_report=retrieval_report,
        )
        self.state["last_updated"] = _now()
        self._save()

    def describe_thread(self, thread_id: str) -> Optional[str]:
        thread = self.state.get("threads", {}).get(thread_id)
        if not thread:
            return None
        lines = [f"Thread: {thread.get('topic', '(unknown)')}"]
        if thread.get("summary"):
            lines.append(thread["summary"])
        if thread.get("preferred_route"):
            lines.append(f"Preferred route: {thread['preferred_route']}")
        if thread.get("keywords"):
            lines.append("Keywords: " + ", ".join(thread["keywords"]))
        return "\n".join(lines)

    def active_threads(self, *, limit: int = 5) -> List[Dict[str, Any]]:
        threads = list(self.state.get("threads", {}).items())
        threads.sort(key=lambda kv: kv[1].get("last_updated", 0), reverse=True)
        out: List[Dict[str, Any]] = []
        for thread_id, thread in threads[:limit]:
            out.append({"thread_id": thread_id, **thread})
        return out

    def propose_proactive_message(
        self,
        *,
        cognition: Optional[CognitionChat] = None,
        memory_store: Optional[MemoryStore] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        threads = self.active_threads(limit=4)
        if not threads:
            return None

        payload: Dict[str, Any] = {"threads": []}
        for thread in threads:
            payload["threads"].append(
                {
                    "thread_id": thread.get("thread_id"),
                    "topic": thread.get("topic"),
                    "summary": thread.get("summary"),
                    "keywords": thread.get("keywords"),
                    "preferred_route": thread.get("preferred_route"),
                    "status": thread.get("status"),
                }
            )

        if memory_store:
            try:
                payload["recent_memories"] = [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": _truncate(m.content, 300),
                        "ts": m.ts,
                        "tags": list(m.tags or []),
                    }
                    for m in memory_store.fetch_recent(session_id=session_id, limit=6)
                ]
            except Exception:
                payload["recent_memories"] = []

        payload["meta"] = {"generated_ts": _now(), "thread_count": len(payload["threads"]) }

        llm = cognition or CognitionChat(CognitionConfig.load())
        llm.set_system(PROACTIVE_SYSTEM_PROMPT)
        try:
            opener = llm.raw_chat(json.dumps(payload, ensure_ascii=False), stream=False)
            if isinstance(opener, str):
                return opener.strip()
        except Exception:
            return None
        return None

    # ── internals -----------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "nodes": {},
                "edges": [],
                "threads": {},
                "last_updated": _now(),
            }
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("global_context.json corrupted")
            return data
        except Exception:
            return {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "nodes": {},
                "edges": [],
                "threads": {},
                "last_updated": _now(),
            }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _refresh_indexes(self) -> None:
        nodes = self.state.setdefault("nodes", {})
        threads = self.state.setdefault("threads", {})
        self._name_index: Dict[str, str] = {}
        for node_id, node in nodes.items():
            name = node.get("name")
            if name:
                self._name_index[_normalize(name)] = node_id
        self._thread_index: Dict[str, str] = {}
        for thread_id, thread in threads.items():
            topic = thread.get("topic")
            if topic:
                self._thread_index[_normalize(topic)] = thread_id

    def _compact_threads(self) -> List[Dict[str, Any]]:
        out = []
        for thread_id, thread in self.state.get("threads", {}).items():
            out.append(
                {
                    "thread_id": thread_id,
                    "topic": thread.get("topic"),
                    "summary": thread.get("summary"),
                    "keywords": thread.get("keywords", []),
                    "preferred_route": thread.get("preferred_route"),
                    "status": thread.get("status"),
                }
            )
        return out

    def _serialize_memories(self, memories: Sequence[MemoryRow], limit: int = 14) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for m in memories[:limit]:
            rows.append(
                {
                    "id": m.id,
                    "role": m.role,
                    "content": _truncate(m.content, 600),
                    "ts": m.ts,
                    "tags": list(m.tags or []),
                }
            )
        return rows

    def _merge_update(
        self,
        update: Dict[str, Any],
        *,
        session_id: Optional[str],
        invoked_route: str,
        success: bool,
        user_memory_id: Optional[str],
        assistant_memory_id: Optional[str],
        retrieval_report: Optional[RetrievalReport],
    ) -> None:
        entities = update.get("entities") or []
        for entity in entities:
            name = entity.get("name")
            summary = entity.get("summary")
            if not name or not summary:
                continue
            node_id = self._find_or_create_node(name, entity.get("type"), summary)
            node = self.state["nodes"][node_id]
            node["summary"] = summary
            node["type"] = entity.get("type") or node.get("type")
            node.setdefault("keywords", [])
            node["keywords"] = sorted(
                set(node["keywords"] + (entity.get("keywords") or []))
            )[:24]
            node.setdefault("memories", [])
            for mem_id in entity.get("supporting_memories") or []:
                if mem_id and mem_id not in node["memories"]:
                    node["memories"].append(mem_id)
            node.setdefault("documents", [])
            for doc_id in entity.get("supporting_docs") or []:
                if doc_id and doc_id not in node["documents"]:
                    node["documents"].append(doc_id)
            node.setdefault("route_counts", {})
            node["route_counts"][invoked_route] = node["route_counts"].get(invoked_route, 0) + 1
            node["last_seen"] = _now()

            for rel in entity.get("related_to") or []:
                target_name = rel.get("target")
                relation = rel.get("relation")
                if not target_name or not relation:
                    continue
                target_id = self._find_or_create_node(target_name, None, None)
                self._upsert_edge(node_id, target_id, relation, rel, invoked_route)

        threads = update.get("threads") or []
        for thread_desc in threads:
            topic = thread_desc.get("topic")
            if not topic:
                continue
            thread_id = self._find_or_create_thread(topic)
            thread = self.state["threads"].setdefault(thread_id, {})
            thread["topic"] = topic
            if thread_desc.get("summary"):
                thread["summary"] = thread_desc["summary"]
            thread.setdefault("keywords", [])
            thread["keywords"] = sorted(
                set(thread["keywords"] + (thread_desc.get("keywords") or []))
            )[:24]
            node_ids: List[str] = []
            for name in thread_desc.get("primary_entities") or []:
                node_ids.append(self._find_or_create_node(name, None, None))
            if node_ids:
                merged = set(thread.get("node_ids", []))
                merged.update(node_ids)
                thread["node_ids"] = list(merged)
            pref_route = thread_desc.get("preferred_route") or thread.get("preferred_route")
            thread["preferred_route"] = pref_route or thread.get("preferred_route") or invoked_route
            thread["status"] = thread_desc.get("status") or thread.get("status") or ("stable" if success else "open")
            thread["last_route"] = invoked_route
            thread["last_updated"] = _now()
            if user_memory_id:
                thread.setdefault("recent_memories", [])
                if user_memory_id not in thread["recent_memories"]:
                    thread["recent_memories"].append(user_memory_id)
                    thread["recent_memories"] = thread["recent_memories"][-10:]
            if assistant_memory_id:
                thread.setdefault("recent_outputs", [])
                if assistant_memory_id not in thread["recent_outputs"]:
                    thread["recent_outputs"].append(assistant_memory_id)
                    thread["recent_outputs"] = thread["recent_outputs"][-10:]
            if retrieval_report:
                doc_titles = [doc.get("title") for doc in retrieval_report.documents[:4]]
                thread.setdefault("recent_docs", [])
                for title in doc_titles:
                    if title and title not in thread["recent_docs"]:
                        thread["recent_docs"].append(title)
                thread["recent_docs"] = thread["recent_docs"][-12:]

        self._refresh_indexes()

    def _find_or_create_node(self, name: str, kind: Optional[str], summary: Optional[str]) -> str:
        norm = _normalize(name)
        node_id = self._name_index.get(norm)
        if node_id:
            node = self.state["nodes"].setdefault(node_id, {})
            node.setdefault("name", name)
            if kind and not node.get("type"):
                node["type"] = kind
            if summary and not node.get("summary"):
                node["summary"] = summary
            return node_id
        node_id = _new_id("node")
        self.state.setdefault("nodes", {})[node_id] = {
            "name": name,
            "type": kind,
            "summary": summary or "",
            "keywords": [],
            "memories": [],
            "documents": [],
            "route_counts": {},
            "first_seen": _now(),
            "last_seen": _now(),
        }
        self._name_index[norm] = node_id
        return node_id

    def _find_or_create_thread(self, topic: str) -> str:
        norm = _normalize(topic)
        thread_id = self._thread_index.get(norm)
        if thread_id:
            return thread_id
        thread_id = _new_id("thread")
        self.state.setdefault("threads", {})[thread_id] = {
            "topic": topic,
            "keywords": [],
            "node_ids": [],
            "preferred_route": None,
            "status": "active",
            "created_ts": _now(),
            "last_updated": _now(),
        }
        self._thread_index[norm] = thread_id
        return thread_id

    def _upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        rel_payload: Dict[str, Any],
        invoked_route: str,
    ) -> None:
        edges = self.state.setdefault("edges", [])
        relation_norm = _normalize(relation)
        for edge in edges:
            if (
                edge.get("source") == source_id
                and edge.get("target") == target_id
                and _normalize(edge.get("relation", "")) == relation_norm
            ):
                weight = rel_payload.get("weight")
                if isinstance(weight, (int, float)):
                    edge["weight"] = max(edge.get("weight", 0.1), float(weight))
                edge.setdefault("rationale", rel_payload.get("rationale"))
                edge.setdefault("route_counts", {})
                edge["route_counts"][invoked_route] = edge["route_counts"].get(invoked_route, 0) + 1
                edge["last_seen"] = _now()
                return
        edge_entry = {
            "edge_id": _new_id("edge"),
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "weight": float(rel_payload.get("weight", 0.5)) if isinstance(rel_payload.get("weight"), (int, float)) else 0.5,
            "rationale": rel_payload.get("rationale"),
            "route_counts": {invoked_route: 1},
            "first_seen": _now(),
            "last_seen": _now(),
        }
        edges.append(edge_entry)
