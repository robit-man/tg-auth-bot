#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planning_bridge.py — Lightweight loader for the examples PlanningEngine.

This module bridges the autonomous planning utilities shipped in examples/planning.py
into the main bot runtime without duplicating the implementation. It dynamically
loads the PlanningEngine class (preferring a local planning.py override when
present) and exposes a helper for running plans with a supplied tool registry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional


class PlanningBridge:
    """Lazy loader and executor wrapper for PlanningEngine."""

    _engine_cls: Optional[type] = None
    _module: Optional[ModuleType] = None
    _load_attempted: bool = False

    @classmethod
    def _load_engine_cls(cls) -> Optional[type]:
        if cls._engine_cls or cls._load_attempted:
            return cls._engine_cls

        cls._load_attempted = True
        base_dir = Path(__file__).resolve().parent
        candidates = [
            base_dir / "planning.py",
            base_dir / "planning_engine.py",
            base_dir / "examples" / "planning.py",
        ]

        for idx, path in enumerate(candidates):
            if not path.exists():
                continue

            try:
                module_name = f"_planning_bridge_{idx}"
                spec = importlib.util.spec_from_file_location(module_name, path)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as exc:
                print(f"[planning_bridge] Failed to load {path}: {exc}")
                continue

            engine_cls = getattr(module, "PlanningEngine", None)
            if engine_cls:
                cls._engine_cls = engine_cls  # type: ignore[assignment]
                cls._module = module
                break

        return cls._engine_cls

    @classmethod
    def available(cls) -> bool:
        """Return True if a PlanningEngine implementation is available."""
        return cls._load_engine_cls() is not None

    @classmethod
    def run_plan(
        cls,
        *,
        objective: str,
        tools: Dict[str, Callable[..., Any]],
        constraints: Optional[str] = None,
        deliverables: Optional[str] = None,
        session_id: Optional[str] = None,
        config_path: Optional[str] = None,
        model: Optional[str] = None,
        system_override: Optional[str] = None,
    ) -> dict:
        """
        Execute the PlanningEngine with the provided tool registry.

        Args:
            objective: User objective for the plan.
            tools: Mapping of tool names to callables available to the planner.
            constraints: Optional constraints string.
            deliverables: Optional deliverables description.
            session_id: Optional session identifier used for logging.
            config_path: Optional path to planning configuration JSON.
            model: Optional LLM model override.
            system_override: Optional planner system prompt override.

        Returns:
            dict: Result returned by the loaded PlanningEngine implementation.
        """
        engine_cls = cls._load_engine_cls()
        if not engine_cls:
            raise RuntimeError("Planning engine not available (examples/planning.py missing?)")

        config_path = config_path or str(Path(__file__).resolve().parent / "planning.json")

        try:
            engine = engine_cls(  # type: ignore[call-arg]
                policy=None,
                memory=None,
                knowledge=None,
                tools=tools,
                config_path=config_path,
                model=model,
                system_override=system_override,
            )
        except TypeError:
            # Older constructor signature
            engine = engine_cls(None, None, None, tools)  # type: ignore[call-arg]

        plan_kwargs: Dict[str, Any] = {}
        if session_id:
            plan_kwargs["session_id"] = session_id
        if constraints:
            plan_kwargs["constraints"] = constraints
        if deliverables:
            plan_kwargs["deliverables"] = deliverables

        # Try common execution entry points, mirroring examples/main.py compatibility logic.
        if hasattr(engine, "plan_and_execute"):
            try:
                return engine.plan_and_execute(objective=objective, **plan_kwargs)
            except TypeError:
                try:
                    return engine.plan_and_execute(goal=objective, **plan_kwargs)  # type: ignore[arg-type]
                except TypeError:
                    return engine.plan_and_execute(objective)  # type: ignore[call-arg]
        if hasattr(engine, "run"):
            try:
                return engine.run(goal=objective, **plan_kwargs)  # type: ignore[arg-type]
            except TypeError:
                return engine.run(objective)  # type: ignore[call-arg]
        if hasattr(engine, "execute"):
            try:
                return engine.execute(goal=objective, **plan_kwargs)  # type: ignore[arg-type]
            except TypeError:
                return engine.execute(objective)  # type: ignore[call-arg]

        raise RuntimeError("Planning engine does not expose a supported execution method")


__all__ = ["PlanningBridge"]

