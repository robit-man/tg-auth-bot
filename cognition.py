#!/usr/bin/env python3
"""
Lightweight stand-in for the cognition module expected by tools.py.

Provides:
    - CognitionConfig: dataclass wrapper with load() helper drawing from env vars
    - CognitionChat: minimal chat helper around ollama.chat with JSON convenience
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ollama  # type: ignore


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CognitionConfig:
    model: str
    temperature: float = 0.2
    stream: bool = False
    global_system_prelude: str = ""
    ollama_options: Dict[str, Any] = field(default_factory=dict)
    embed_model: Optional[str] = None

    @classmethod
    def load(cls) -> "CognitionConfig":
        """Load defaults from environment with sane fallbacks."""
        model = os.environ.get("OLLAMA_MODEL") or "llama3"
        embed_model = os.environ.get("OLLAMA_EMBED_MODEL") or None
        system = os.environ.get("COGNITION_SYSTEM_PROMPT") or ""
        temperature = _float_env("COGNITION_TEMPERATURE", 0.2)

        # Allow arbitrary JSON options via env for debugging/customisation.
        options_raw = os.environ.get("COGNITION_OLLAMA_OPTIONS", "{}")
        try:
            options = json.loads(options_raw) if options_raw else {}
            if not isinstance(options, dict):
                options = {}
        except json.JSONDecodeError:
            options = {}

        return cls(
            model=model,
            temperature=temperature,
            stream=False,
            global_system_prelude=system,
            ollama_options=options,
            embed_model=embed_model,
        )


class CognitionChat:
    """
    Minimal chat utility that mirrors the surface needed by tools.py.
    """

    def __init__(self, config: CognitionConfig):
        self.config = config
        self._system_msg = config.global_system_prelude or ""

    def set_system(self, system_prompt: str) -> None:
        self._system_msg = system_prompt or ""

    def _messages(self, user_content: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.config.global_system_prelude:
            msgs.append({"role": "system", "content": self.config.global_system_prelude})
        if self._system_msg and self._system_msg != self.config.global_system_prelude:
            msgs.append({"role": "system", "content": self._system_msg})
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def raw_chat(self, user_message: str, *, stream: Optional[bool] = None) -> str:
        use_stream = self.config.stream if stream is None else stream
        options = {"temperature": self.config.temperature}
        options.update(self.config.ollama_options or {})

        if use_stream:
            pieces: List[str] = []
            for chunk in ollama.chat(
                model=self.config.model,
                messages=self._messages(user_message),
                stream=True,
                options=options,
            ):
                delta = (chunk.get("message") or {}).get("content")
                if delta:
                    pieces.append(delta)
            return "".join(pieces).strip()

        response = ollama.chat(
            model=self.config.model,
            messages=self._messages(user_message),
            stream=False,
            options=options,
        )
        message = response.get("message") if isinstance(response, dict) else {}
        return (message.get("content") or "").strip()

    def structured_json(
        self,
        *,
        user_message: str,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        """
        Ask for JSON output and try to parse it. The schema is ignored but kept for compatibility.
        """
        # Provide a light hint to produce JSON if no explicit system prompt.
        if not self._system_msg:
            self.set_system(
                "Respond strictly with valid JSON that matches the requested structure."
            )
        text = self.raw_chat(user_message, stream=stream)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to extract JSON substring.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        raise ValueError("Model response was not valid JSON.")

