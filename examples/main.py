#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Interactive demos with HITL path correction, prompt A/B, and system config.
Now upgraded to include a Planning (DAG) path that invokes the planning engine to
plan → delegate → execute → evaluate arbitrarily complex tasks.

Features:
• Policy learning (LinUCB routing + param bandits)
• HITL path correction using natural language → LLM normalization to exact route
• System prompts & schemas in system.json + overrides in system_modified.json
• Prompt A/B when heavy failure detected
• Knowledge RAG (hybrid-ready: neighbor bleed, vision; backward compatible)
• Tools web search
• Memory chat
• NEW: Planning: Plan & Execute (DAG) using planning.py + planning.json

Files:
  system.json, system_modified.json, policy.db, knowledge.db, memory.db, planning.json
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
from datetime import datetime, timezone
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List, Callable

# ─────────────────────────────────────────────────────────────────────────────
# 1) venv bootstrap
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv"
PYTHON_BIN = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PIP_BIN = VENV_DIR / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
CONFIG_PATH = SCRIPT_DIR / "config.json"
DATA_DIR = SCRIPT_DIR / "data"
SYSTEM_BASE = SCRIPT_DIR / "system.json"
SYSTEM_MOD = SCRIPT_DIR / "system_modified.json"
PLANNING_JSON = SCRIPT_DIR / "planning.json"

DEPS = [
    "ollama>=0.3.0",
    "jsonschema>=4.22.0",
    "selenium>=4.22.0",
    "webdriver-manager>=4.0.2",
    "beautifulsoup4>=4.12.3",
    "lxml>=5.2.2",
    "requests>=2.32.3",
    "pymupdf>=1.24.8",
    "python-docx>=1.1.2",
    "chardet>=5.2.0",
    "pyyaml>=6.0.2",
    "numpy>=1.26.0",
    # (optional) planning engines may not require extras; keep lean for compatibility
]

def ensure_venv_and_reexec():
    if os.environ.get("INSIDE_VENV") == "1":
        return
    if not VENV_DIR.exists():
        print(f"[BOOT] Creating venv at {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True, clear=False).create(str(VENV_DIR))
    print("[BOOT] Upgrading pip/setuptools/wheel ...")
    subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    print("[BOOT] Ensuring dependencies are installed ...")
    subprocess.check_call([str(PIP_BIN), "install", "--upgrade"] + DEPS)
    env = dict(os.environ)
    env["INSIDE_VENV"] = "1"
    cmd = [str(PYTHON_BIN), __file__]
    print("[BOOT] Re-executing inside venv ...")
    os.execvpe(str(PYTHON_BIN), cmd, env)

ensure_venv_and_reexec()

# ─────────────────────────────────────────────────────────────────────────────
# 2) Imports (inside venv)
# ─────────────────────────────────────────────────────────────────────────────
from cognition import CognitionChat, CognitionConfig  # noqa: E402
from memory import MemoryStore  # noqa: E402
from tools import Tools, log_message  # noqa: E402
from context import KnowledgeStore, RetrievalReport  # noqa: E402
from policy import PolicyManager  # noqa: E402
from global_context import GlobalContextWorkspace, ThreadMatch  # noqa: E402

# Planning is optional but recommended; keep a safe import.
try:
    from planning import PlanningEngine  # type: ignore
    HAS_PLANNING = True
except Exception as _e:
    HAS_PLANNING = False

# ─────────────────────────────────────────────────────────────────────────────
# 3) System config (prompts + schemas)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM = {
    "system_messages": {
        "selector": "You map intents to one of the provided options. Output a single choice.",
        "route_normalizer": (
            "You are a route normalizer. Given the user's free-text correction describing the desired route, "
            "map it to ONE EXACT option from the provided list. "
            "Return ONLY valid JSON matching the schema; do not add prose."
        ),
        "raw_chat": "Be succinct. If referencing past, cite the memory id short hash like #deadbeef when relevant.",
        "structured_json": "Return valid JSON for the schema. Use prior context only when helpful.",
        "system_message": "Generate a single production-ready system message. No fences.",
        "tool_planner": "You are a tool-calling planner. Produce ONLY a JSON object that matches the given JSON Schema. Pick sensible defaults. No prose.",
        "knowledge_rag": (
            "You answer strictly based on the provided library excerpts.\n"
            "• When you cite, include the bracketed source tag exactly as shown: [source: Title | p.X | #abcdef12 | score=0.XXX]\n"
            "• If something is not covered by the excerpts, explicitly say you don't have evidence in the provided sources.\n"
            "• Be accurate, concise, and cite after each relevant sentence or paragraph."
        ),
        # NEW: Orchestrator prompt for planning; engine may use its own planning.json,
        # but we expose a message so A/B and HITL can tune it if the engine supports overrides.
        "planner_orchestrator": (
            "You are an operations planner. Expand a short user goal into an actionable plan (DAG). "
            "Break down tasks, order them with dependencies, assign tools, define success criteria for each step, "
            "and specify evaluation checks. Prefer small, verifiable steps; retry with refined instructions on failure."
        ),
    },
    "schemas": {
        "todo_item": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "TodoItem",
            "type": "object",
            "required": ["title", "priority", "tags"],
            "properties": {
                "title": {"type": "string", "minLength": 3},
                "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True},
                "due": {"type": "string", "format": "date"}
            },
            "additionalProperties": False
        },
        "internet_search_args": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "InternetSearchArgs",
            "type": "object",
            "required": ["topic"],
            "properties": {
                "topic": {"type": "string", "minLength": 3},
                "num_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "wait_sec": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
                "deep_scrape": {"type": "boolean", "default": True},
                "summarize": {"type": "boolean", "default": True},
                "bs4_verbose": {"type": "boolean", "default": False},
                "headless": {"type": "boolean", "default": False}
            },
            "additionalProperties": False
        }
    }
}

class SystemConfig:
    def __init__(self, base_path: Path, mod_path: Path):
        self.base_path = base_path
        self.mod_path = mod_path
        self.base = {}
        self.mod = {}
        self.load()

    def load(self):
        if not self.base_path.exists():
            self.base_path.write_text(json.dumps(DEFAULT_SYSTEM, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            self.base = json.loads(self.base_path.read_text(encoding="utf-8"))
        except Exception:
            self.base = DEFAULT_SYSTEM
        if self.mod_path.exists():
            try:
                self.mod = json.loads(self.mod_path.read_text(encoding="utf-8"))
            except Exception:
                self.mod = {}
        else:
            self.mod = {}

    def save_modified(self):
        self.mod_path.write_text(json.dumps(self.mod, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_message(self, key: str, variant: str = "best") -> str:
        base_msg = (self.base.get("system_messages", {}) or {}).get(key, "")
        mod_msg = (self.mod.get("system_messages", {}) or {}).get(key, None)
        if variant == "base":
            return base_msg
        if variant == "modified" and mod_msg:
            return mod_msg
        return mod_msg or base_msg

    def set_modified_message(self, key: str, text: str):
        self.mod.setdefault("system_messages", {})
        self.mod["system_messages"][key] = text
        self.save_modified()

    def get_schema(self, name: str) -> dict:
        base_s = (self.base.get("schemas", {}) or {}).get(name, {})
        mod_s = (self.mod.get("schemas", {}) or {}).get(name, None)
        return mod_s or base_s

    def set_modified_schema(self, name: str, schema: dict):
        self.mod.setdefault("schemas", {})
        self.mod["schemas"][name] = schema
        self.save_modified()

# ─────────────────────────────────────────────────────────────────────────────
# 4) UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_hr(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Demo catalog
# ─────────────────────────────────────────────────────────────────────────────

DEMO_OPTIONS = [
    "Raw chat",
    "Structured JSON (TodoItem)",
    "Produce system message",
    "Tool: Search Internet",
    "Knowledge: Ask the library (RAG)",
    "Planning: Plan & Execute (DAG)",  # NEW
    "Quit",
]

ACTION_MAP = {
    "Raw chat": "raw_chat",
    "Structured JSON (TodoItem)": "structured_json",
    "Produce system message": "system_message",
    "Tool: Search Internet": "tool_search_internet",
    "Knowledge: Ask the library (RAG)": "knowledge_rag",
    "Planning: Plan & Execute (DAG)": "planning_dag",  # NEW
}


@dataclass
class RunOutcome:
    success: bool
    sysmsg_arm_token: Optional[str]
    user_memory_id: Optional[str] = None
    assistant_memory_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    option: str
    action_id: str
    selector_arm: str
    confidence: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    thread_match: Optional[ThreadMatch] = None

# ─────────────────────────────────────────────────────────────────────────────
# 6) Config + policy utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_embed_model_in_config(cfg: CognitionConfig) -> CognitionConfig:
    changed = False
    env_embed = os.environ.get("OLLAMA_EMBED_MODEL")
    if not cfg.embed_model or not isinstance(cfg.embed_model, str) or not cfg.embed_model.strip():
        cfg.embed_model = env_embed.strip() if env_embed and env_embed.strip() else "nomic-embed-text"
        changed = True
    elif env_embed and env_embed.strip() and env_embed.strip() != cfg.embed_model:
        cfg.embed_model = env_embed.strip()
        changed = True
    if changed:
        cfg.save()
    if not CONFIG_PATH.exists():
        cfg.save()
    else:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        if raw.get("embed_model") != cfg.embed_model:
            raw["embed_model"] = cfg.embed_model
            CONFIG_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# 7) Decision + HITL + prompt variant selection
# ─────────────────────────────────────────────────────────────────────────────

def choose_sysmsg_variant(policy: PolicyManager, syscfg: SystemConfig, key: str) -> Tuple[str, str]:
    """
    Returns (system_message_text, chosen_arm) where arm in {"base","modified"}.
    Uses a small param bandit to pick between base vs modified.
    """
    arms: List[str] = ["base"]
    if syscfg.get_message(key, "modified") != syscfg.get_message(key, "base"):
        arms.append("modified")
    bandit = f"sysmsg.{key}"
    arm = policy.params.select(bandit, arms)
    text = syscfg.get_message(key, arm if arm in ("base", "modified") else "base")
    return text, arm


def _heuristic_route_bias(intent: str) -> Dict[str, float]:
    intent_l = intent.lower()
    bias: Dict[str, float] = {}

    def add(action: str, weight: float) -> None:
        bias[action] = bias.get(action, 0.0) + weight

    if any(kw in intent_l for kw in ("plan", "roadmap", "workflow", "sequence")):
        add("planning_dag", 1.4)
    if any(kw in intent_l for kw in ("document", "policy", "pdf", "manual", "spec")):
        add("knowledge_rag", 1.6)
    if any(kw in intent_l for kw in ("search", "web", "internet", "news")):
        add("tool_search_internet", 1.2)
    if any(kw in intent_l for kw in ("system prompt", "persona", "system message")):
        add("system_message", 1.0)
    if any(kw in intent_l for kw in ("json", "schema", "structure", "fields")):
        add("structured_json", 0.9)
    if any(kw in intent_l for kw in ("chat", "talk", "conversation", "catch up")):
        add("raw_chat", 0.5)
    return bias


def auto_route_selection(
    *,
    user_intent: str,
    model_selector: CognitionChat,
    policy: PolicyManager,
    syscfg: SystemConfig,
    workspace: GlobalContextWorkspace,
) -> Tuple[RouteDecision, Any, Dict[str, float]]:
    selector_sys, selector_arm = choose_sysmsg_variant(policy, syscfg, "selector")
    model_selector.set_system(selector_sys)

    llm_choice = model_selector.decide_from_options(
        question=user_intent,
        options=DEMO_OPTIONS,
        return_index=False,
        stream=False,
    )
    if not isinstance(llm_choice, str):
        llm_choice = "Raw chat"
    llm_choice = next((opt for opt in DEMO_OPTIONS if opt.lower() in llm_choice.lower()), "Raw chat")
    llm_action = ACTION_MAP.get(llm_choice, "raw_chat")

    available_actions = [ACTION_MAP[o] for o in DEMO_OPTIONS if o != "Quit"]
    feats = policy.features.from_intent(user_intent)
    pol_action, policy_scores = policy.policy.select(available_actions, feats)

    heuristics = _heuristic_route_bias(user_intent)
    thread_match = workspace.match_goal(user_intent)

    aggregated: Dict[str, float] = {}
    for option in DEMO_OPTIONS:
        if option == "Quit":
            continue
        action = ACTION_MAP[option]
        base = float(policy_scores.get(action, 0.0))
        agg = base
        if action == llm_action:
            agg += 1.0
        if action == pol_action:
            agg += 0.6
        agg += heuristics.get(action, 0.0)
        if thread_match and thread_match.preferred_route and action == thread_match.preferred_route:
            agg += 2.2
        aggregated[action] = agg

    fallback_action = llm_action or pol_action or "raw_chat"
    best_action = max(aggregated, key=aggregated.get, default=fallback_action)
    chosen_option = next((opt for opt, act in ACTION_MAP.items() if act == best_action), "Raw chat")

    scores_sorted = sorted(aggregated.values(), reverse=True)
    if scores_sorted:
        top_score = scores_sorted[0]
        second_score = scores_sorted[1] if len(scores_sorted) > 1 else (top_score - 0.5)
        margin = top_score - second_score
        denominator = abs(top_score) + abs(second_score) + 1e-6
        confidence = max(0.1, min(1.0, 0.5 + margin / (denominator)))
    else:
        confidence = 0.5

    reason_bits: List[str] = []
    if thread_match and thread_match.preferred_route == best_action:
        reason_bits.append(f"continuing thread '{thread_match.topic}'")
    if llm_action == best_action:
        reason_bits.append("selector LLM alignment")
    if pol_action == best_action:
        reason_bits.append("policy exploit")
    heur = heuristics.get(best_action)
    if heur:
        reason_bits.append("heuristic keyword match")
    if not reason_bits:
        reason_bits.append("default safety")
    reason = ", ".join(reason_bits)

    details = {
        "llm_choice": llm_choice,
        "llm_action": llm_action,
        "policy_action": pol_action,
        "policy_scores": policy_scores,
        "heuristics": heuristics,
        "aggregated": aggregated,
        "selector_arm": selector_arm,
    }

    decision = RouteDecision(
        option=chosen_option,
        action_id=best_action,
        selector_arm=selector_arm,
        confidence=confidence,
        reason=reason,
        details=details,
        thread_match=thread_match,
    )

    return decision, feats, policy_scores

def normalize_route_with_llm(
    free_text: str,
    options: List[str],
    base_cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager
) -> Tuple[Optional[str], Optional[str]]:
    """
    Use Cognition structured_json with a dynamic schema (enum of options)
    to map free_text → exact menu option.

    Returns (normalized_option_or_None, sysmsg_arm_token_or_None)
    """
    if not free_text.strip():
        return None, None

    sys_text, sys_arm = choose_sysmsg_variant(policy, syscfg, "route_normalizer")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RouteSelection",
        "type": "object",
        "required": ["choice"],
        "additionalProperties": False,
        "properties": {
            "choice": {
                "type": "string",
                "enum": options
            }
        }
    }

    normalizer = CognitionChat(CognitionConfig(
        model=base_cfg.model,
        temperature=0.0,
        stream=False,
        global_system_prelude="",
        embed_model=base_cfg.embed_model,
    ))
    normalizer.set_system(sys_text)

    user_msg = (
        "User correction describing desired route: "
        f"«{free_text}»\n\n"
        "Map to ONE EXACT option from this list:\n"
        + "\n".join(f"- {opt}" for opt in options)
    )
    out = normalizer.structured_json(
        user_message=user_msg,
        json_schema=schema,
        stream=False
    )

    choice = None
    if isinstance(out, dict):
        choice = out.get("choice")
    else:
        try:
            parsed = json.loads(out)
            choice = parsed.get("choice")
        except Exception:
            choice = None

    if choice in options:
        return choice, f"sysmsg.route_normalizer:{sys_arm}"
    return None, f"sysmsg.route_normalizer:{sys_arm}"

def decision_stage(
    *,
    model_selector: CognitionChat,
    policy: PolicyManager,
    syscfg: SystemConfig,
    workspace: GlobalContextWorkspace,
    base_cfg: CognitionConfig,
) -> Tuple[Optional[RouteDecision], Optional[str], Optional[Any], Optional[Dict[str, float]], Optional[str]]:
    print_hr("What do you want to tackle?")
    user_intent = input("Describe your goal (or 'q' to quit): ").strip()
    if user_intent.lower() in {"q", "quit", "exit"}:
        return None, user_intent, None, None, None

    decision, feats, policy_scores = auto_route_selection(
        user_intent=user_intent,
        model_selector=model_selector,
        policy=policy,
        syscfg=syscfg,
        workspace=workspace,
    )

    thread_line = ""
    if decision.thread_match:
        thread = decision.thread_match
        thread_line = (
            f"Thread '{thread.topic}' ({thread.confidence:.0%})"
            + (f" → prefers {thread.preferred_route}" if thread.preferred_route else "")
        )

    print_hr("Routing Engine")
    print(
        f"Auto-selected route: {decision.option} [{decision.action_id}] "
        f"(confidence {decision.confidence:.0%})"
    )
    if thread_line:
        print(f"Context: {thread_line}")
    print(f"Reason: {decision.reason}")
    print(
        f"Selector LLM → {decision.details.get('llm_choice')} | "
        f"Policy → {decision.details.get('policy_action')}"
    )

    override = input("Override? (Enter keeps, or describe desired route): ").strip()
    route_norm_arm: Optional[str] = None

    if override:
        normalized = None
        # Direct match first for quick overrides
        for opt in DEMO_OPTIONS:
            if opt.lower() == override.lower():
                normalized = opt
                break
        if not normalized:
            normalized, route_norm_arm = normalize_route_with_llm(
                override,
                DEMO_OPTIONS,
                base_cfg,
                syscfg,
                policy,
            )

        if normalized and normalized != decision.option:
            new_action = ACTION_MAP.get(normalized, decision.action_id)
            decision = RouteDecision(
                option=normalized,
                action_id=new_action,
                selector_arm=decision.selector_arm,
                confidence=0.35,
                reason="HITL override",
                details={**decision.details, "override": override, "normalized": normalized},
                thread_match=decision.thread_match,
            )
            print(f"[HITL] Switched to: {normalized}")
        elif normalized is None:
            print("[HITL] Could not normalize override; keeping auto-selected route.")

    return decision, user_intent, feats, policy_scores, route_norm_arm

def prompt_editor(syscfg: SystemConfig):
    print_hr("System Config Editor")
    print("You can edit a system message or a schema.")
    choice = input("Type 'msg' for message, 'schema' for schema, or Enter to cancel: ").strip().lower()
    if choice == "msg":
        keys = list((syscfg.base.get("system_messages", {}) or {}).keys())
        print("Available message keys:", ", ".join(keys))
        key = input("Key to edit: ").strip()
        if key not in keys:
            print("Unknown key.")
            return
        base = syscfg.get_message(key, "base")
        mod = syscfg.get_message(key, "modified")
        print("\n--- BASE ---\n", base, "\n--- MODIFIED (current) ---\n", mod)
        print("\nPaste the NEW modified message. End input with a single line containing ONLY: END")
        lines = []
        while True:
            ln = input()
            if ln.strip() == "END":
                break
            lines.append(ln)
        new_text = "\n".join(lines).strip()
        if new_text:
            syscfg.set_modified_message(key, new_text)
            print("[editor] Modified message saved.")
        else:
            print("[editor] Empty input; nothing changed.")
    elif choice == "schema":
        keys = list((syscfg.base.get("schemas", {}) or {}).keys())
        print("Available schema keys:", ", ".join(keys))
        key = input("Key to edit: ").strip()
        if key not in keys:
            print("Unknown key.")
            return
        base = syscfg.get_schema(key)
        mod = (syscfg.mod.get("schemas", {}) or {}).get(key, {})
        print("\n--- BASE ---\n", json.dumps(base, indent=2))
        print("\n--- MODIFIED (current) ---\n", json.dumps(mod or {}, indent=2))
        print("\nPaste the NEW modified JSON schema. End input with a single line containing ONLY: END")
        lines = []
        while True:
            ln = input()
            if ln.strip() == "END":
                break
            lines.append(ln)
        try:
            new_schema = json.loads("\n".join(lines))
            syscfg.set_modified_schema(key, new_schema)
            print("[editor] Modified schema saved.")
        except Exception as e:
            print(f"[editor] Invalid JSON: {e}")
    else:
        print("Editor cancelled.")

def ab_test_prompt_for_action(action_id: str, policy: PolicyManager, syscfg: SystemConfig):
    action_to_key = {
        "raw_chat": "raw_chat",
        "structured_json": "structured_json",
        "system_message": "system_message",
        "tool_search_internet": "tool_planner",
        "knowledge_rag": "knowledge_rag",
        "planning_dag": "planner_orchestrator",  # NEW
    }
    key = action_to_key.get(action_id, "raw_chat")

    base_text = syscfg.get_message(key, "base")
    mod_text = syscfg.get_message(key, "modified")

    if not mod_text or mod_text == base_text:
        print_hr("A/B setup — create a modified prompt")
        print("BASE prompt:\n", base_text)
        print("\nEnter a MODIFIED prompt (END to finish):")
        lines = []
        while True:
            ln = input()
            if ln.strip() == "END":
                break
            lines.append(ln)
        new_text = "\n".join(lines).strip()
        if new_text:
            syscfg.set_modified_message(key, new_text)
            mod_text = new_text
        else:
            print("[A/B] No modified prompt entered; aborting A/B.")
            return

    print_hr(f"A/B Test for '{key}'")
    print("A) BASE prompt:\n", base_text)
    print("\nB) MODIFIED prompt:\n", mod_text)
    pick = input("\nWhich performed better for your intent? (A/B) ").strip().lower()
    bandit = f"sysmsg.{key}"
    if pick == "a":
        policy.params.update(bandit, "base", 1.0)
        policy.params.update(bandit, "modified", 0.0)
        print("[A/B] Logged preference: BASE")
    elif pick == "b":
        policy.params.update(bandit, "modified", 1.0)
        policy.params.update(bandit, "base", 0.0)
        print("[A/B] Logged preference: MODIFIED")
    else:
        print("[A/B] Skipped.")

# ─────────────────────────────────────────────────────────────────────────────
# 8) Runners (use system-configured prompts + param-bandit variants)
# ─────────────────────────────────────────────────────────────────────────────

from tools import Tools  # noqa: E402

def build_context_from_recall(store: MemoryStore, session_id: str, user_input: str, k: int = 8):
    recalled = store.recall_for_context(session_id=session_id, query_text=user_input, k=k)
    ctx_snips = []
    recalled_ids = []
    for m in recalled:
        tag_str = (",".join(m.tags)) if m.tags else ""
        prefix = f"[{m.role} {_iso(m.ts)} #{m.id[:8]}{(' '+tag_str) if tag_str else ''}]"
        ctx_snips.append(f"{prefix}\n{m.content}")
        recalled_ids.append(m.id)
    return ctx_snips, recalled_ids

def run_raw_chat(
    mem: MemoryStore,
    session_id: str,
    cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager,
) -> RunOutcome:
    sys_text, sys_arm = choose_sysmsg_variant(policy, syscfg, "raw_chat")
    user_text = input("You: ").strip()
    if not user_text:
        print("Empty input; returning to menu.")
        return RunOutcome(False, None)
    ctx_snips, recalled_ids = build_context_from_recall(mem, session_id, user_text, k=8)
    cog = CognitionChat(cfg)
    cog.set_system(sys_text)
    if ctx_snips:
        cog.add_context(ctx_snips + [{"recalled_memory_ids": recalled_ids}])
    inv_id = mem.add_invocation(session_id=session_id, mode="raw_chat", user_input=user_text, meta={"recalled_ids": recalled_ids})
    print_hr("Model")
    out = cog.raw_chat(user_text, stream=False)
    print(out)
    user_mem = mem.add_memory(session_id=session_id, role="user", content=user_text, source="user_input",
                              tags=["mode:raw_chat"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")
    asst_mem = mem.add_memory(session_id=session_id, role="assistant", content=out, source="cog_output",
                              tags=["mode:raw_chat"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")
    for mid in recalled_ids:
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=mid, relation="recalled")
    return RunOutcome(
        success=bool(out and out.strip()),
        sysmsg_arm_token=f"sysmsg.raw_chat:{sys_arm}",
        user_memory_id=user_mem,
        assistant_memory_id=asst_mem,
        extra={
            "invocation_id": inv_id,
            "recalled_ids": recalled_ids,
            "user_input": user_text,
        },
    )

def run_structured_json(
    mem: MemoryStore,
    session_id: str,
    cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager,
) -> RunOutcome:
    sys_text, sys_arm = choose_sysmsg_variant(policy, syscfg, "structured_json")
    prompt = input("Describe the todo you want (title, rough priority 1-5, tags CSV): ").strip()
    if not prompt:
        print("Empty input; returning to menu.")
        return RunOutcome(False, None)
    ctx_snips, recalled_ids = build_context_from_recall(mem, session_id, prompt, k=8)
    cog = CognitionChat(cfg)
    cog.set_system(sys_text)
    if ctx_snips:
        cog.add_context(ctx_snips + [{"recalled_memory_ids": recalled_ids}])
    inv_id = mem.add_invocation(session_id=session_id, mode="structured_json", user_input=prompt, meta={"recalled_ids": recalled_ids})
    print_hr("Structured JSON")
    schema = syscfg.get_schema("todo_item")
    out = cog.structured_json(prompt, json_schema=schema, stream=False)
    print(out)
    user_mem = mem.add_memory(session_id=session_id, role="user", content=prompt, source="user_input",
                              tags=["mode:structured_json"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")
    out_str = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=2)
    asst_mem = mem.add_memory(session_id=session_id, role="assistant", content=out_str, source="cog_output",
                              tags=["mode:structured_json"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")
    for mid in recalled_ids:
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=mid, relation="recalled")
    ok = isinstance(out, dict) or (isinstance(out, str) and "{" in out)
    return RunOutcome(
        success=ok,
        sysmsg_arm_token=f"sysmsg.structured_json:{sys_arm}",
        user_memory_id=user_mem,
        assistant_memory_id=asst_mem,
        extra={
            "invocation_id": inv_id,
            "recalled_ids": recalled_ids,
            "prompt": prompt,
        },
    )

def run_system_message(
    mem: MemoryStore,
    session_id: str,
    cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager,
) -> RunOutcome:
    sys_text, sys_arm = choose_sysmsg_variant(policy, syscfg, "system_message")
    req = input("What system behavior do you want? ").strip()
    if not req:
        print("Empty input; returning to menu.")
        return RunOutcome(False, None)
    ctx_snips, recalled_ids = build_context_from_recall(mem, session_id, req, k=8)
    cog = CognitionChat(cfg)
    cog.set_system(sys_text)
    if ctx_snips:
        cog.add_context(ctx_snips + [{"recalled_memory_ids": recalled_ids}])
    inv_id = mem.add_invocation(session_id=session_id, mode="system_message", user_input=req, meta={"recalled_ids": recalled_ids})
    print_hr("System Message")
    out = cog.produce_system_message(req, stream=False)
    print(out)
    user_mem = mem.add_memory(session_id=session_id, role="user", content=req, source="user_input",
                              tags=["mode:system_message"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")
    asst_mem = mem.add_memory(session_id=session_id, role="assistant", content=out, source="cog_output",
                              tags=["mode:system_message"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")
    for mid in recalled_ids:
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=mid, relation="recalled")
    return RunOutcome(
        success=bool(out and out.strip()),
        sysmsg_arm_token=f"sysmsg.system_message:{sys_arm}",
        user_memory_id=user_mem,
        assistant_memory_id=asst_mem,
        extra={
            "invocation_id": inv_id,
            "recalled_ids": recalled_ids,
            "prompt": req,
            "output": out,
        },
    )

def run_tool_search_internet(
    mem: MemoryStore,
    session_id: str,
    cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager,
) -> RunOutcome:
    intent = input("What would you like to research on the web? ").strip()
    if not intent:
        print("Empty input; returning to menu.")
        return RunOutcome(False, None)
    ctx_snips, recalled_ids = build_context_from_recall(mem, session_id, intent, k=8)

    use_agent = input("Run advanced multi-step search agent? (y/N): ").strip().lower() in {"y", "yes", "true", "1"}

    if use_agent:
        success_criteria = input("Success criteria or format expectations? (Enter to skip) ").strip()
        inv_id = mem.add_invocation(
            session_id=session_id,
            mode="tool_search_internet/agent",
            user_input=intent,
            meta={
                "recalled_ids": recalled_ids,
                "success_criteria": success_criteria,
            },
        )

        agent_result = Tools.complex_search_agent(
            objective=intent,
            success_criteria=success_criteria or None,
            background=[snip[:400] for snip in ctx_snips[:3]] if ctx_snips else None,
            max_iterations=6,
            max_results=6,
            headless=True,
            deep_scrape=True,
        )

        print_hr("Search Agent Summary")
        summary_text = agent_result.get("summary") or "(no summary generated)"
        print(summary_text)
        sources = agent_result.get("sources") or []
        if sources:
            print("\nSources:")
            for idx, src in enumerate(sources, start=1):
                print(f"[{idx}] {src.get('title') or src.get('url')} — {src.get('url')}")

        issues = agent_result.get("issues") or []
        if issues:
            print("\nIssues encountered:")
            for note in issues[-6:]:
                print(f"- {note}")

        if agent_result.get("history"):
            print_hr("Agent Steps")
            for step in agent_result["history"][-4:]:
                label = f"{step.get('step_id')} ({step.get('action')})"
                findings = step.get("key_findings") or step.get("summary") or ""
                print(f"- {label}: {findings[:400]}")

        user_mem = mem.add_memory(
            session_id=session_id,
            role="user",
            content=intent,
            source="user_input",
            tags=["mode:tool_search_internet", "mode:tool_search_agent"],
        )
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")

        asst_payload = {
            "tool": "complex_search_agent",
            "objective": intent,
            "success_criteria": success_criteria,
            "result": agent_result,
        }
        asst_mem = mem.add_memory(
            session_id=session_id,
            role="assistant",
            content=json.dumps(asst_payload, ensure_ascii=False, indent=2)[:8000],
            source="cog_output",
            tags=["mode:tool_search_internet", "mode:tool_search_agent"],
        )
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")
        for mid in recalled_ids:
            mem.link_invocation_memory(invocation_id=inv_id, memory_id=mid, relation="recalled")

        return RunOutcome(
            success=bool(agent_result.get("success")),
            sysmsg_arm_token=None,
            user_memory_id=user_mem,
            assistant_memory_id=asst_mem,
            extra={
                "invocation_id": inv_id,
                "agent_result": agent_result,
                "recalled_ids": recalled_ids,
                "success_criteria": success_criteria,
            },
        )

    planner_sys, planner_arm = choose_sysmsg_variant(policy, syscfg, "tool_planner")
    cog = CognitionChat(cfg)
    cog.set_system(planner_sys)
    if ctx_snips:
        cog.add_context(ctx_snips + [{"recalled_memory_ids": recalled_ids}])
    inv_id = mem.add_invocation(session_id=session_id, mode="tool_search_internet/plan", user_input=intent, meta={"recalled_ids": recalled_ids})
    print_hr("Planning tool call (structured JSON)")
    schema = syscfg.get_schema("internet_search_args")
    plan = cog.structured_json(user_message=f"Create tool args to research: {intent}",
                               json_schema=schema, stream=False)
    print(plan)
    if isinstance(plan, str):
        try:
            plan_obj = json.loads(plan)
        except Exception:
            print("[ERROR] Tool planning did not return valid JSON.")
            return RunOutcome(False, f"sysmsg.tool_planner:{planner_arm}")
    else:
        plan_obj = plan
    print_hr("Executing Tools.search_internet")
    try:
        results = Tools.search_internet(**plan_obj)
    except TypeError as e:
        print(f"[ERROR] Invalid tool args: {e}")
        return RunOutcome(False, f"sysmsg.tool_planner:{planner_arm}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.get('title','(no title)')}")
        print(r.get("url", ""))
        if r.get("snippet"):
            print("  snippet:", r["snippet"])
        if r.get("aux_summary"):
            print("  summary:", r["aux_summary"][:500].replace('\n', ' '))
    user_mem = mem.add_memory(session_id=session_id, role="user", content=intent, source="user_input",
                              tags=["mode:tool_search_internet"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")
    compact = {"tool": "search_internet", "args": plan_obj,
               "results": [{"title": r.get("title"), "url": r.get("url"), "summary": r.get("aux_summary", "")[:500]} for r in results]}
    asst_mem = mem.add_memory(session_id=session_id, role="assistant", content=json.dumps(compact, ensure_ascii=False, indent=2),
                              source="cog_output", tags=["mode:tool_search_internet"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")
    for mid in recalled_ids:
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=mid, relation="recalled")
    return RunOutcome(
        success=len(results) > 0,
        sysmsg_arm_token=f"sysmsg.tool_planner:{planner_arm}",
        user_memory_id=user_mem,
        assistant_memory_id=asst_mem,
        extra={
            "invocation_id": inv_id,
            "plan": plan_obj,
            "results": results,
            "recalled_ids": recalled_ids,
        },
    )

def run_knowledge_rag(
    mem: MemoryStore,
    kstore: KnowledgeStore,
    session_id: str,
    cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager,
    *,
    top_k: Optional[int] = None,
) -> RunOutcome:
    """
    RAG runner (backward compatible):
      - optional one-shot ingest with force + vision if supported by context.py
      - hybrid search with neighbor bleed if supported, else vector-only search
    """
    rag_sys, rag_arm = choose_sysmsg_variant(policy, syscfg, "knowledge_rag")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[RAG] Using data folder: {DATA_DIR}")

    # Ask for ingest preferences (defaults: no force, no vision)
    use_force = input("Force re-ingest unchanged files? (y/N): ").strip().lower() in {"y", "yes"}
    default_vision_model = os.environ.get("OLLAMA_VISION_MODEL", "llava")
    use_vision = input(f"Enable page vision extraction (PNG + {default_vision_model})? (y/N): ").strip().lower() in {"y", "yes"}
    if use_vision:
        print(f"[RAG] Vision model: {default_vision_model}")

    # One-shot ingest pass (compatible with old/new signatures)
    def _ingest_now():
        try:
            stats = kstore.ingest_all_in_data(concurrent=True, max_workers=4, force=use_force, vision=use_vision, vision_model=default_vision_model)  # type: ignore
        except TypeError:
            # Older context.py versions without extra args
            stats = kstore.ingest_all_in_data(concurrent=True, max_workers=4)
        print(f"[RAG] Ingest pass — total:{stats['total']}  ingested:{stats['ingested']}  skipped:{stats['skipped']}  failed:{stats['failed']}")
    threading.Thread(target=_ingest_now, daemon=True).start()

    query = input("Ask the library a question: ").strip()
    if not query:
        print("Empty input; returning to menu.")
        return RunOutcome(False, None)

    # Bandit-chosen retrieval params
    arms_k = ["5", "8", "12"]
    chosen_k_arm = policy.params.select("rag.top_k", arms_k)
    k_val = int(chosen_k_arm) if chosen_k_arm else (int(top_k) if top_k else 8)

    arms_bleed = ["0", "1", "2"]
    chosen_bleed_arm = policy.params.select("rag.bleed", arms_bleed)
    bleed_val = int(chosen_bleed_arm or "1")

    # Retrieval (compatible with/without bleed)
    try:
        top_chunks = kstore.search_with_bleed(query_text=query, k=k_val, bleed=bleed_val, tags=None)  # type: ignore
    except AttributeError:
        top_chunks = kstore.search(query_text=query, k=k_val, tags=None)

    doc_snips, chunk_ids = kstore.build_context_snippets(top_chunks)

    retrieval_report = None
    if top_chunks:
        ingestion_meta = {"force": use_force, "vision": use_vision, "top_k": k_val, "bleed": bleed_val}
        retrieval_report = RetrievalReport.from_chunks(
            query=query,
            chunks=top_chunks,
            top_k=k_val,
            bleed=bleed_val,
            ingestion=ingestion_meta,
        )
        print_hr("Retrieval Report")
        print(retrieval_report.render())

    cog = CognitionChat(cfg)
    cog.set_system(rag_sys)
    if doc_snips:
        cog.add_context(doc_snips)

    kinv_id = kstore.add_invocation(session_id=session_id, query=query, meta={"k": k_val, "bleed": bleed_val, "vision": use_vision, "force": use_force})
    for cid in chunk_ids:
        kstore.link_invocation_chunk(invocation_id=kinv_id, chunk_id=cid)

    print_hr("Knowledge RAG — Model")
    answer = cog.raw_chat(query, stream=False)
    print(answer)

    user_mem = mem.add_memory(session_id=session_id, role="user", content=query, source="user_input",
                              tags=["mode:knowledge_rag"])
    inv_id = mem.add_invocation(session_id=session_id, mode="knowledge_rag", user_input=query,
                                meta={"k_invocation": kinv_id, "k": k_val, "bleed": bleed_val})
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")
    asst_mem = mem.add_memory(session_id=session_id, role="assistant", content=answer, source="cog_output",
                              tags=["mode:knowledge_rag"])
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")

    # Update tuned params
    success = bool(answer and answer.strip())
    policy.params.update("rag.top_k", chosen_k_arm, 1.0 if success else 0.3)
    policy.params.update("rag.bleed", chosen_bleed_arm, 1.0 if success else 0.3)

    return RunOutcome(
        success=success,
        sysmsg_arm_token=f"sysmsg.knowledge_rag:{rag_arm}",
        user_memory_id=user_mem,
        assistant_memory_id=asst_mem,
        extra={
            "invocation_id": inv_id,
            "k_invocation_id": kinv_id,
            "chunk_ids": chunk_ids,
            "retrieval_report": retrieval_report,
            "rag_top_k_arm": chosen_k_arm,
            "rag_bleed_arm": chosen_bleed_arm,
            "query": query,
            "recalled_ids": [],
        },
    )

# === NEW: Planning runner =====================================================

def run_planning(
    mem: MemoryStore,
    kstore: KnowledgeStore,
    session_id: str,
    cfg: CognitionConfig,
    syscfg: SystemConfig,
    policy: PolicyManager,
) -> RunOutcome:
    """
    Planning path:
      - Takes a high-level objective
      - Uses PlanningEngine (DAG) to plan, delegate, execute, evaluate
      - Registers full Tools set (including new file/location/system utils)
      - Logs artifacts to memory + updates policy
      - Backward compatible if planning.py or methods are absent
    """
    if not HAS_PLANNING:
        print_hr("Planning not available")
        print("planning.py not found or failed to import. Please add planning.py before using this path.")
        return RunOutcome(False, None)

    planner_sys, planner_arm = choose_sysmsg_variant(policy, syscfg, "planner_orchestrator")

    print_hr("Planning: Plan & Execute (DAG)")
    goal = input("Describe the end goal: ").strip()
    if not goal:
        print("Empty input; returning to menu.")
        return RunOutcome(False, None)

    # Optional constraints & resources (simple prompt; engine may ignore if unsupported)
    constraints = input("Any constraints (time, cost, tools, risk)? (Enter to skip) ").strip()
    deliverables = input("Expected deliverables (files, reports, metrics)? (Enter to skip) ").strip()

    # Tool registry (explicit mapping)
    tool_registry = {
        # Web
        "search_internet": Tools.search_internet,

        # Filesystem / data
        "find_file": Tools.find_file,
        "find_files": Tools.find_files,   # returns JSON string
        "list_dir": Tools.list_dir,       # returns JSON string
        "list_files": Tools.list_files,   # returns list[dict]
        "read_file": Tools.read_file,
        "read_files": Tools.read_files,
        "write_file": Tools.write_file,
        "rename_file": Tools.rename_file,
        "copy_file": Tools.copy_file,
        "create_file": Tools.create_file,
        "append_file": Tools.append_file,
        "delete_file": Tools.delete_file,
        "file_exists": Tools.file_exists,
        "file_info": Tools.file_info,
        "get_cwd": Tools.get_cwd,

        # System / env
        "get_current_location": Tools.get_current_location,
        "get_system_utilization": Tools.get_system_utilization,

        # Knowledge (thin wrappers to keep signatures simple)
        "rag_search": lambda query, k=8: [
            {"id": c.id, "title": c.title, "path": c.path, "page": c.page_no, "score": c.score, "text": c.text}
            for c in (getattr(kstore, "search_with_bleed", kstore.search)(query_text=query, k=int(k), bleed=1)  # type: ignore
                      if hasattr(kstore, "search_with_bleed") else kstore.search(query_text=query, k=int(k)))
        ],
        "fetch_webpage": Tools.fetch_webpage,
        "search_agent": lambda objective, success_criteria=None, background=None, max_iterations=5, max_results=6, headless=True, deep_scrape=True: Tools.complex_search_agent(
            objective=objective,
            success_criteria=success_criteria,
            background=background,
            max_iterations=max_iterations,
            max_results=max_results,
            headless=headless,
            deep_scrape=deep_scrape,
        ),
    }

    # Memory context for planning (optional)
    ctx_snips, recalled_ids = build_context_from_recall(mem, session_id, goal, k=8)

    # Instantiate engine (duck-typed to be backward compatible with your planning.py)
    try:
        engine = PlanningEngine(
            policy=policy,
            memory=mem,
            knowledge=kstore,
            tools=tool_registry,
            config_path=str(PLANNING_JSON),
            model=cfg.model,
            system_override=planner_sys  # some engines may accept this; safe to ignore otherwise
        )
    except TypeError:
        # Older constructor
        engine = PlanningEngine(policy, mem, kstore, tool_registry)

    # Execute the plan; try several method signatures for compatibility
    result = None
    ok = False
    error_txt = None
    try:
        if hasattr(engine, "plan_and_execute"):
            try:
                result = engine.plan_and_execute(
                    objective=goal,
                    session_id=session_id,
                    constraints=constraints or None,
                    deliverables=deliverables or None,
                )
            except TypeError:
                result = engine.plan_and_execute(goal)
        elif hasattr(engine, "run"):
            try:
                result = engine.run(
                    goal=goal,
                    session_id=session_id,
                    constraints=constraints or None,
                    deliverables=deliverables or None,
                )
            except TypeError:
                result = engine.run(goal)
        elif hasattr(engine, "execute"):
            try:
                result = engine.execute(goal)
            except TypeError:
                result = engine.execute(goal=goal)
        else:
            error_txt = "PlanningEngine has no plan_and_execute/run/execute methods."
    except Exception as e:
        error_txt = f"{e}"

    # Inspect outcome
    if isinstance(result, dict):
        ok = bool(result.get("success", True))
    elif result is None and error_txt is None:
        ok = True  # treat None as success if engine didn't return a dict
    else:
        ok = False

    # Print a compact report
    print_hr("Planning — Report")
    if isinstance(result, dict):
        print(json.dumps({k: v for k, v in result.items() if k not in {"artifacts", "blobs"}}, ensure_ascii=False, indent=2))
    elif error_txt:
        print(f"[error] {error_txt}")
    else:
        print(str(result))

    # Persist to memory
    inv_meta = {
        "route": "planning_dag",
        "constraints": constraints,
        "deliverables": deliverables,
        "ok": ok
    }
    user_mem = mem.add_memory(session_id=session_id, role="user", content=goal, source="user_input",
                              tags=["mode:planning"])
    inv_id = mem.add_invocation(session_id=session_id, mode="planning", user_input=goal, meta=inv_meta)
    mem.link_invocation_memory(invocation_id=inv_id, memory_id=user_mem, relation="produced")
    if result is not None:
        asst_mem = mem.add_memory(
            session_id=session_id,
            role="assistant",
            content=json.dumps(result, ensure_ascii=False, indent=2)[:8000],
            source="planning_output",
            tags=["mode:planning"],
        )
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=asst_mem, relation="produced")
    else:
        asst_mem = None

    # Optional: save recalled ids linkage (context used to seed planning)
    for mid in recalled_ids:
        mem.link_invocation_memory(invocation_id=inv_id, memory_id=mid, relation="recalled")

    return RunOutcome(
        success=ok,
        sysmsg_arm_token=f"sysmsg.planner_orchestrator:{planner_arm}",
        user_memory_id=user_mem,
        assistant_memory_id=asst_mem,
        extra={
            "invocation_id": inv_id,
            "constraints": constraints,
            "deliverables": deliverables,
            "result": result,
            "error": error_txt,
        },
    )

# ─────────────────────────────────────────────────────────────────────────────
# 9) Reward capture
# ─────────────────────────────────────────────────────────────────────────────

ROUTE_RUNNERS: Dict[str, Callable[..., RunOutcome]] = {
    "raw_chat": run_raw_chat,
    "structured_json": run_structured_json,
    "system_message": run_system_message,
    "tool_search_internet": run_tool_search_internet,
    "knowledge_rag": run_knowledge_rag,
    "planning_dag": run_planning,
}


def get_reward_from_user() -> Optional[float]:
    print("\nPlease rate the helpfulness (1–5). Press Enter to skip.")
    s = input("Rating: ").strip()
    if not s:
        return None
    try:
        v = int(s)
        if 1 <= v <= 5:
            return (v - 1) / 4.0
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 10) Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Cognition config + embed model
    base_cfg = CognitionConfig.load()
    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model and env_model.strip() and env_model.strip() != base_cfg.model:
        base_cfg.model = env_model.strip()
        base_cfg.save()
    base_cfg = ensure_embed_model_in_config(base_cfg)

    # Stores & policy
    mem_store = MemoryStore(db_path=SCRIPT_DIR / "memory.db", embed_model=base_cfg.embed_model)
    know_store = KnowledgeStore(db_path=SCRIPT_DIR / "knowledge.db", data_dir=DATA_DIR, embed_model=base_cfg.embed_model)
    policy = PolicyManager(db_path=SCRIPT_DIR / "policy.db")

    # System config
    syscfg = SystemConfig(SYSTEM_BASE, SYSTEM_MOD)

    # Global context workspace (knowledge graph + thread memory)
    workspace = GlobalContextWorkspace(config=base_cfg)

    # Session
    session_id = f"demo-session-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    # Selector instance (system prompt set per decision)
    selector = CognitionChat(CognitionConfig(
        model=base_cfg.model,
        temperature=0.2,
        stream=False,
        global_system_prelude="",
        embed_model=base_cfg.embed_model,
    ))

    print_hr("Interactive Agent Demos (Memory + Tools + Knowledge RAG + Planning + Learning + HITL)")
    print(f"chat model   = {base_cfg.model}")
    print(f"embed model  = {base_cfg.embed_model}")
    print(f"vision model = {os.environ.get('OLLAMA_VISION_MODEL', 'gemma3:4b')} (set OLLAMA_VISION_MODEL to change)")
    print(f"session      = {session_id}")
    print(f"data dir     = {DATA_DIR} (drop PDFs / MD / TXT / HTML / DOCX here)")
    print("Type 'q' at the decision prompt to quit.\n")

    if base_cfg.proactive:
        proactive_chat = CognitionChat(
            CognitionConfig(
                model=base_cfg.model,
                temperature=base_cfg.temperature,
                stream=False,
                global_system_prelude=base_cfg.global_system_prelude,
                ollama_options=dict(base_cfg.ollama_options),
                embed_model=base_cfg.embed_model,
                proactive=base_cfg.proactive,
                proactive_prompt=base_cfg.proactive_prompt,
            )
        )
        opener = workspace.propose_proactive_message(
            cognition=proactive_chat,
            memory_store=mem_store,
            session_id=session_id,
        )
        if opener:
            print_hr("Proactive")
            print(opener)

    # Background watcher (compat: try new signature, else old)
    try:
        know_store.start_auto_ingest_watcher(interval_sec=30, concurrent=True, max_workers=4, force=False, vision=False)  # type: ignore
    except TypeError:
        know_store.start_auto_ingest_watcher(interval_sec=30, concurrent=True, max_workers=4)

    while True:
        decision, user_goal, feats, _, route_norm_arm = decision_stage(
            model_selector=selector,
            policy=policy,
            syscfg=syscfg,
            workspace=workspace,
            base_cfg=base_cfg,
        )
        if decision is None:
            print("Bye!")
            break

        chosen_action = decision.action_id
        chosen_option = decision.option

        print_hr(f"Selected: {chosen_option}")
        print(
            f"[routing] confidence={decision.confidence:.0%} | reason={decision.reason}"
        )
        agg_scores = decision.details.get("aggregated")
        if isinstance(agg_scores, dict):
            debug_line = ", ".join(
                f"{key}:{val:.2f}" for key, val in sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)
            )
            print(f"[scores] {debug_line}")

        runner = ROUTE_RUNNERS.get(chosen_action)
        if runner is run_knowledge_rag:
            outcome = runner(mem_store, know_store, session_id, base_cfg, syscfg, policy)
        elif runner is run_planning:
            outcome = runner(mem_store, know_store, session_id, base_cfg, syscfg, policy)
        elif runner is not None:
            outcome = runner(mem_store, session_id, base_cfg, syscfg, policy)
        else:
            outcome = run_raw_chat(mem_store, session_id, base_cfg, syscfg, policy)

        success = outcome.success
        reward = get_reward_from_user()
        if reward is None:
            reward = 0.7 if success else 0.2

        if feats is not None:
            policy.policy.update(chosen_action, feats, reward)
            policy.feedback.log(chosen_action, reward)

        policy.params.update("sysmsg.selector", decision.selector_arm, reward)

        if route_norm_arm:
            try:
                bandit, arm = route_norm_arm.split(":")
                policy.params.update(bandit, arm, reward)
            except Exception:
                pass

        if outcome.sysmsg_arm_token:
            try:
                bandit, arm = outcome.sysmsg_arm_token.split(":")
                policy.params.update(bandit, arm, reward)
            except Exception:
                pass

        extra = outcome.extra or {}
        if extra.get("rag_top_k_arm"):
            policy.params.update("rag.top_k", extra["rag_top_k_arm"], reward)
        if extra.get("rag_bleed_arm"):
            policy.params.update("rag.bleed", extra["rag_bleed_arm"], reward)

        retrieval_report = extra.get("retrieval_report") if isinstance(extra, dict) else None
        user_memory_id = outcome.user_memory_id
        assistant_memory_id = outcome.assistant_memory_id

        try:
            workspace.update_after_invocation(
                session_id=session_id,
                user_goal=user_goal or "",
                invoked_route=chosen_action,
                success=success,
                user_input=extra.get("user_input") if isinstance(extra, dict) else user_goal,
                user_memory_id=user_memory_id,
                assistant_memory_id=assistant_memory_id,
                thread_match=decision.thread_match,
                retrieval_report=retrieval_report,
                memory_store=mem_store,
            )
        except Exception as exc:
            log_message("warn", f"Global context update failed: {exc}")

        if policy.feedback.heavy_failure(chosen_action, n=5, threshold=0.25, min_count=3):
            print_hr("Heavy failure detected — entering A/B prompt tuning")
            ab_test_prompt_for_action(chosen_action, policy, syscfg)

if __name__ == "__main__":
    main()
