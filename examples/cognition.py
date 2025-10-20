#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cognition.py — Thin, opinionated wrapper around the Ollama Python chat API
for agent-style context handling and multiple response modalities.

Features
--------
• Modalities via explicit methods:
    - raw_chat(...)                → free-form model response
    - decide_from_options(...)     → choose exactly 1 item from a list
    - structured_json(...)         → emit JSON conforming to a JSON Schema
    - produce_system_message(...)  → generate a crisp system message

• Robust context handling:
    - set_system(...) to inject a per-call system message
    - add_context([...]) accepts str/dict/list; non-str are injected as fenced JSON
    - global_system_prelude in config prepends every call

• Validation-first ergonomics:
    - If required args are missing or invalid, the class does NOT throw.
      Instead it calls the model with a DEBUG system message that lists:
        provided arguments + precise validation errors,
      so orchestration loops can auto-correct.

• Streaming:
    - Stream by default (configured). Pass stream=False to get a single string.
    - When streaming, returns a generator of text deltas.

• Persistent configuration (config.json):
    - Auto-created on first import with sensible defaults.
    - Load/Save via CognitionConfig.load()/save()
    - Includes "embed_model" so memory.py can share a single source of truth.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import re

# External deps (installed by main.py bootstrap)
import ollama  # type: ignore
from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Types & helpers
# ─────────────────────────────────────────────────────────────────────────────

RoleMsg = Dict[str, str]
JSONable = Union[dict, list, str, int, float, bool, None]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

_RUNAWAY_RE = re.compile(r"(.{1,12})\1{6,}", re.IGNORECASE)


def _as_fenced(obj: JSONable) -> str:
    """Render arbitrary JSON-able content as a deterministic fenced block."""
    if isinstance(obj, (dict, list)):
        return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return "```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"
    return "```json\n" + json.dumps(str(obj), ensure_ascii=False) + "\n```"


def _extract_first_json(s: str) -> Optional[JSONable]:
    """
    Extract the first JSON object/array (or primitive) from an LLM response.
    Tries, in order:
      • whole-string json
      • first fenced code block ```json ... ```
      • first {...} or [...]
    """
    s = s.strip()
    # Direct full-string
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fenced
    import re
    fence = re.search(r"```json\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # First {...} or [...]
    first_obj = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    first_arr = re.search(r"(\[.*\])", s, flags=re.DOTALL)
    for grp in (first_obj, first_arr):
        if grp:
            try:
                return json.loads(grp.group(1))
            except Exception:
                continue
    return None


def _detect_runaway(text: str) -> bool:
    if not text:
        return False
    return bool(_RUNAWAY_RE.search(text))


# ─────────────────────────────────────────────────────────────────────────────
# Config (includes embed_model)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitionConfig:
    """
    Persistent configuration for CognitionChat. Saved/loaded from config.json.
    """
    model: str = "llama3.1"
    temperature: float = 0.2
    stream: bool = True
    global_system_prelude: Optional[str] = None
    # Advanced generation options for chat calls
    ollama_options: Dict[str, Any] = field(default_factory=dict)
    # Embedding model (used by memory.py; kept here for a single source of truth)
    embed_model: str = "nomic-embed-text"
    # Interaction posture
    proactive: bool = False
    proactive_prompt: Optional[str] = None

    @classmethod
    def load(cls) -> "CognitionConfig":
        """Load from config.json, creating/normalizing with defaults (incl. embed_model)."""
        if not CONFIG_PATH.exists():
            cfg = cls()
            cfg.save()
            return cfg
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Filter unexpected keys but preserve known ones (tolerant loads)
            fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
            filtered = {k: v for k, v in raw.items() if k in fields}
            # Ensure embed_model exists (auto-augment)
            if "embed_model" not in filtered or not filtered.get("embed_model"):
                filtered["embed_model"] = cls.embed_model
            cfg = cls(**filtered)
            cfg.save()  # normalize file to include any new defaults/keys
            return cfg
        except Exception:
            cfg = cls()
            cfg.save()
            return cfg

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitionChat:
    """
    High-level wrapper with explicit modalities and validation-first behavior.

    Typical use:
        cog = CognitionChat()  # loads config.json
        cog.set_system("You are concise.")
        cog.add_context(["facts...", {"k": "v"}])
        txt = cog.raw_chat("Hi there", stream=False)
    """
    config: CognitionConfig = field(default_factory=CognitionConfig.load)
    _system_message: Optional[str] = None
    _context_snippets: List[str] = field(default_factory=list)

    # Config convenience
    def update_config(self, *, persist: bool = False, **kwargs: Any) -> "CognitionChat":
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        if persist:
            self.config.save()
        return self

    # Context management
    def set_system(self, system_message: Optional[str]) -> "CognitionChat":
        self._system_message = system_message
        return self

    def add_context(self, entries: Iterable[Union[str, dict, list]]) -> "CognitionChat":
        for e in entries:
            if isinstance(e, str):
                s = e.strip()
                if s:
                    self._context_snippets.append(s)
            else:
                self._context_snippets.append(_as_fenced(e))
        return self

    def clear_context(self) -> "CognitionChat":
        self._context_snippets.clear()
        return self

    # Modalities
    def raw_chat(self, user_message: str, *, stream: Optional[bool] = None) -> Union[str, Iterable[str]]:
        msgs = self._assemble_messages(user=user_message)
        return self._call_ollama(msgs, stream=stream)

    def decide_from_options(
        self,
        question: str,
        options: Sequence[str],
        *,
        return_index: bool = False,
        stream: Optional[bool] = None,
    ) -> Union[str, int, Tuple[str, int], Iterable[str]]:
        errors = []
        if not options or len(options) < 2:
            errors.append("`options` must contain at least 2 strings.")
        if any(not isinstance(x, str) or not x.strip() for x in options):
            errors.append("`options` must be non-empty strings.")
        if errors:
            debug_msgs = self._build_debug_messages(
                mode="decide_from_options", user=question,
                provided={"options": options, "return_index": return_index}, errors=errors)
            return self._call_ollama(debug_msgs, stream=stream)

        numbered = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options, start=1)])
        sys_msg = textwrap.dedent("""
            You are a decision selector. Pick ONE best answer from a numbered list.
            Output:
            ```json
            {"index": <1-based>, "value": "<exact text>"}
            ```
        """).strip()
        msgs = self._assemble_messages(extra_system=sys_msg,
                                       user=f"Question: {question.strip()}\n\nOptions (numbered):\n{numbered}")
        result = self._call_ollama(msgs, stream=False if stream is None else stream)
        if isinstance(result, str):
            parsed = _extract_first_json(result) or {}
            idx, value = parsed.get("index"), parsed.get("value")
            if isinstance(value, str) and value in options:
                i = options.index(value)
                return (value, i) if return_index else value
            if isinstance(idx, int) and 1 <= idx <= len(options):
                chosen = options[idx - 1]
                return (chosen, idx - 1) if return_index else chosen
            for i, opt in enumerate(options):
                if opt.lower() in result.lower():
                    return (opt, i) if return_index else opt
            return result
        return result

    def structured_json(self, user_message: str, *, json_schema: Optional[Dict[str, Any]],
                        require_fenced: bool = True, stream: Optional[bool] = None):
        errors = []
        if not json_schema:
            errors.append("`json_schema` is required for structured_json().")
        elif not isinstance(json_schema, dict):
            errors.append("`json_schema` must be a dict (JSON Schema).")
        else:
            try:
                Draft202012Validator.check_schema(json_schema)
            except jsonschema_exceptions.SchemaError as se:
                errors.append(f"Invalid JSON Schema: {se.message}")
        if errors:
            debug_msgs = self._build_debug_messages(
                mode="structured_json", user=user_message,
                provided={"json_schema": json_schema}, errors=errors)
            return self._call_ollama(debug_msgs, stream=stream)

        schema_instr = _as_fenced(json_schema)
        sys_msg = textwrap.dedent(f"""
            You are a structured-output generator.
            Emit a JSON that VALIDATES against the provided JSON Schema.
            {"Wrap in ```json fences." if require_fenced else "Output pure JSON."}
        """).strip()
        msgs = self._assemble_messages(extra_system=sys_msg,
                                       user=f"JSON Schema to satisfy:\n{schema_instr}\n\nNow, respond to:\n{user_message.strip()}")
        out = self._call_ollama(msgs, stream=False if stream is None else stream)
        if isinstance(out, str):
            data = _extract_first_json(out)
            if data is None:
                return f"[STRUCTURED-JSON ERROR] Could not find JSON in model output.\n---\n{out}"
            validator = Draft202012Validator(json_schema)  # type: ignore
            errs = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errs:
                problems = "\n".join(f"- at {'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in errs)
                debug_msgs = self._build_debug_messages(
                    mode="structured_json/validation", user=user_message,
                    provided={"json_schema": json_schema, "model_output": data}, errors=[problems])
                corrected = self._call_ollama(debug_msgs, stream=False)
                corrected_json = _extract_first_json(corrected if isinstance(corrected, str) else "")
                if corrected_json:
                    errs2 = list(Draft202012Validator(json_schema).iter_errors(corrected_json))  # type: ignore
                    if not errs2:
                        return corrected_json
                return f"[STRUCTURED-JSON VALIDATION FAILED]\n{problems}\n---\n{out}"
            return data
        return out

    def produce_system_message(self, instruction_request: str, *, stream: Optional[bool] = None):
        sys_msg = textwrap.dedent("""
            Produce a single, production-ready system message for another assistant.
            Clear, imperative, minimal. No preamble or code fences.
        """).strip()
        msgs = self._assemble_messages(extra_system=sys_msg, user=instruction_request.strip())
        return self._call_ollama(msgs, stream=stream)

    # Internals
    def _assemble_messages(self, *, user: str, extra_system: Optional[str] = None):
        msgs: List[RoleMsg] = []
        if self.config.global_system_prelude:
            msgs.append({"role": "system", "content": self.config.global_system_prelude})
        if self._system_message:
            msgs.append({"role": "system", "content": self._system_message})
        if extra_system:
            msgs.append({"role": "system", "content": extra_system})
        if self._context_snippets:
            ctx = "\n\n".join(self._context_snippets)
            msgs.append({"role": "system", "content": f"Context (verbatim; may include JSON):\n{ctx}"})
        msgs.append({"role": "user", "content": user})
        return msgs

    def _build_debug_messages(self, *, mode: str, user: str,
                              provided: Dict[str, Any], errors: List[str]):
        debug_sys = textwrap.dedent(f"""
            DEBUG ASSISTANT MODE

            Client attempted to call mode="{mode}" with missing/invalid args.
            1) Briefly explain what's wrong.
            2) Suggest exact keys/shapes needed.
            3) Infer minimal defaults if safe; otherwise ask for the minimum.
        """).strip()
        provided_fenced = _as_fenced(provided)
        error_list = "\n".join(f"- {e}" for e in errors)
        return self._assemble_messages(extra_system=debug_sys,
                                       user=f"Original: {user}\n\nProvided:\n{provided_fenced}\n\nErrors:\n{error_list}")

    def _call_ollama(self, messages: List[RoleMsg], *, stream: Optional[bool] = None):
        return_generator = self.config.stream if stream is None else stream
        options = {"temperature": self.config.temperature}
        if self.config.ollama_options:
            options.update(self.config.ollama_options)
        label = f"[LLM:{self.config.model}]"

        def _streaming_iter():
            print(label, end=" ", flush=True)
            buffer: List[str] = []
            try:
                for chunk in ollama.chat(
                    model=self.config.model,
                    messages=messages,
                    stream=True,
                    options=options,
                ):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        buffer.append(delta)
                        combined = "".join(buffer)
                        if _detect_runaway(combined):
                            print("\n[runaway detected — truncated]", flush=True)
                            break
                        yield delta
            finally:
                print("", flush=True)

        if return_generator:
            return _streaming_iter()

        pieces: List[str] = []
        for part in _streaming_iter():
            pieces.append(part)
        result = "".join(pieces)
        if _detect_runaway(result):
            print(f"{label} runaway content suppressed", flush=True)
            return "[runaway output suppressed]"
        return result
