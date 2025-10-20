#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planning.py — Model-led DAG planning & execution with robust chaining
(plan → execute → evaluate → audit/repair)

No deterministic routes. The model plans; engine executes, verifies, and self-corrects.

What’s new in this version
--------------------------
• Handoff Cognition Layer:
  - After every step, produce a compact "handoff card" with:
    primary_text, compact_query, selectors, suggestions, guardrails, schema_hint, eval_checks.
  - Exposed at artifacts: {step_id}.handoff, {step_id}.primary_text, {step_id}.compact_query
  - Greatly reduces failures where an entire page summary becomes the next search query.

• Hardened planner:
  - structured_json → tolerant parse → LLM JSON-fixer → parse again.
  - Logs each failure/repair reason. No deterministic fallback plans.

• Curly-selector discipline:
  - BARE selector strings are auto-wrapped to {…}.
  - Brace expansion against artifacts supports: {stepId.output}, {stepId.field}, {stepId[index].field},
    {stepId.outputs[index].field}, and friendly globals {title},{url},{snippet},{content},{aux_summary},{extracted}.

• Signature-aware arg alignment:
  - filename/filepath/path, content/text/body, topic/query/q, num/limit/count, base_dir, etc.

• Query sanitation:
  - Overlong or multiline query/topic args are replaced with nearest handoff.compact_query.

• Required-arg completion:
  - If a tool needs a "topic" and planner passed only a URL or upstream text, seed a short compact topic.

• File deliverables:
  - Must exist, be non-empty, and not contain unresolved {tokens}.
  - If empty/unresolved, evaluator proposes fix_args using upstream {<dep>.primary_text}.

• Micro-repair on tool crash:
  - Align args to signature, retry once.

• Final audit/repair:
  - If deliverables are missing/empty/unresolved, ask a repair coach to PATCH an EXISTING step’s args
    (e.g., setting content to {dep.primary_text}) and re-run that step once.

Note: All LLM calls are non-streaming to always capture full outputs.
"""

from __future__ import annotations

import json, os, re, time, uuid, inspect, traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_control_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _CTRL_CHARS_RE.sub("", text)

from cognition import CognitionChat, CognitionConfig  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    id: str
    title: str
    instructions: str
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    deps: List[str] = field(default_factory=list)
    success_criteria: Optional[str] = None
    retries: int = 1
    outputs: List[str] = field(default_factory=list)

@dataclass
class PlanSpec:
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    notes: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Handoff Cognition Layer
# ─────────────────────────────────────────────────────────────────────────────

class CognitionHandoff:
    """
    Produce a compact, structured 'handoff card' for any artifact:
      - primary_text   : best text blob to pass downstream
      - compact_query  : safe short topic/query for downstream search-like tools
      - selectors      : example {curly} tokens future steps can use
      - suggestions    : candidate next tool/args (hinting only; engine doesn't force routes)
      - guardrails     : do/don't notes for consumption
      - schema_hint    : minimal shape hint for list/object artifacts
      - eval_checks    : quick validations downstream can run

    Returned as a dict. Never raises.
    All LLM calls are non-streaming to return complete content.
    """

    @staticmethod
    def sanitize_query(txt: str, fallback: str = "news") -> str:
        import re
        if not isinstance(txt, str):
            txt = str(txt or "")
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            return fallback
        if len(txt) > 180 or "\n" in txt:
            # compress into a short keyword query
            words = re.findall(r"[A-Za-z0-9\-]{3,20}", txt)[:10]
            return " ".join(words) if words else fallback
        return txt

    @staticmethod
    def pick_primary_text(artifact) -> str:
        # prefer extracted → aux_summary → content → snippet → title → str(artifact)
        if isinstance(artifact, dict):
            for k in ("extracted", "aux_summary", "content", "snippet", "title"):
                v = artifact.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        if isinstance(artifact, list) and artifact and isinstance(artifact[0], dict):
            # Concatenate top items briefly
            lines = []
            for r in artifact[:3]:
                v = (r.get("extracted") or r.get("aux_summary") or r.get("content") or
                     r.get("snippet") or r.get("title") or "")
                v = str(v).strip()
                if v:
                    lines.append(v[:700])
            if lines:
                return "\n\n".join(lines)
        return str(artifact or "").strip()

    @staticmethod
    def annotate(*, step_id: str, artifact, context: dict, llm: CognitionChat, max_len: int = 6000) -> dict:
        import json
        primary = CognitionHandoff.pick_primary_text(artifact)[:max_len]
        compact_q = CognitionHandoff.sanitize_query(primary, fallback=context.get("objective") or "news")

        shape = "string"
        if isinstance(artifact, list):
            shape = "list"
        if isinstance(artifact, dict):
            shape = "object"

        sys = (
            "You produce a minimal JSON handoff card for downstream tools.\n"
            "No prose. Fields: primary_text, compact_query, selectors, suggestions, guardrails, schema_hint, eval_checks.\n"
            "selectors are example {curly} tokens future steps could use (e.g., '{step_1[0].extracted}')."
        )
        llm.set_system(sys)

        payload = {
            "step_id": step_id,
            "objective": context.get("objective") or "",
            "artifact_shape": shape,
            "artifact_preview": primary[:1200],
            "known_fields": (
                list(artifact.keys())[:12] if isinstance(artifact, dict)
                else (list(artifact[0].keys())[:12] if (isinstance(artifact, list) and artifact and isinstance(artifact[0], dict)) else [])
            ),
        }

        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["primary_text", "compact_query", "selectors", "suggestions"],
            "properties": {
                "primary_text": {"type": "string"},
                "compact_query": {"type": "string"},
                "selectors": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "suggestions": {"type": "array", "items": {"type": "object"}},
                "guardrails": {"type": "array", "items": {"type": "string"}},
                "schema_hint": {"type": "object"},
                "eval_checks": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True
        }

        try:
            res = llm.structured_json(
                user_message=json.dumps(payload, ensure_ascii=False),
                json_schema=schema,
                stream=False
            )
            if not isinstance(res, dict):
                res = json.loads(res)
        except Exception:
            # Minimal safe fallback; do not break chain
            res = {
                "primary_text": primary,
                "compact_query": compact_q,
                "selectors": [f"{{{step_id}.extracted}}", f"{{{step_id}.aux_summary}}", f"{{{step_id}.content}}"],
                "suggestions": [{"tool": "write_file", "args": {"content": f"{{{step_id}.extracted|{step_id}.aux_summary|{step_id}.content}}"}}],
                "guardrails": ["Keep queries ≤ 180 chars", "Avoid multiline topics", "Prefer extracted→summary→content"],
                "schema_hint": {"type": "string"},
                "eval_checks": ["non_empty", "contains_substring_from_upstream"]
            }

        res.setdefault("compact_query", compact_q)
        res.setdefault("primary_text", primary)
        return res


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class PlanningEngine:
    def __init__(self, policy, memory, knowledge, tools: Dict[str, Callable[..., Any]],
                 config_path: str = "planning.json", model: Optional[str] = None,
                 system_override: Optional[str] = None):
        self.policy = policy
        self.memory = memory
        self.knowledge = knowledge
        self.tools = dict(tools) if tools else {}
        self.cfg_path = config_path
        self.cfg = self._load_or_init_config(self.cfg_path)
        self.model = model or CognitionConfig.load().model
        self.system_override = system_override
        self.logs: List[str] = []
        if "llm_bulletize" not in self.tools:
            self.tools["llm_bulletize"] = self._builtin_llm_bulletize

    # ── Config ───────────────────────────────────────────────────────────────

    def _load_or_init_config(self, path: str) -> dict:
        default = {
            "system": {
                "planner": (
                    "You are a meticulous planner. Build a DAG of steps that use ONLY the provided tools "
                    "(tool name, signature, docstring). Each step: id, title, instructions, tool, args, deps, "
                    "success_criteria, retries (0–3), outputs.\n\n"
                    "CHAINING RULES (MANDATORY):\n"
                    "• Use {curly} selectors in args to pull data from dependencies or labels listed in `outputs`.\n"
                    "• Valid forms: {stepId.output}, {stepId.field}, {stepId[index].field}, "
                    "{stepId.outputs[index].field}, {label.field}. Index may be negative.\n"
                    "• Friendly globals MAY be used when available: {title},{url},{snippet},{content},{aux_summary},{extracted}.\n"
                    "If you create a file, ALWAYS pass non-empty `content` via {…} from upstream text.\n"
                    "Output ONLY JSON matching the schema; do NOT include prose or code fences."
                ),
                "arg_planner": (
                    "Convert one step into concrete tool arguments. Obey the function signature & docstring. "
                    "Use {curly} selectors to reference prior artifacts. Return ONLY a JSON object."
                ),
                "evaluator": (
                    "Strictly evaluate if the step meets success_criteria. "
                    "Return JSON: {\"pass\": bool, \"reason\": str, \"improve\": str, \"fix_args\": object|null}."
                ),
                "repair_coach": (
                    "Final audit found missing/empty deliverables. Propose a minimal fix to an EXISTING step. "
                    "Return ONLY: {\"target_step\":\"<id>\",\"new_args\":{…},\"rename_tool\":\"<name>|null\",\"rationale\":\"…\"}.\n"
                    "Use {curly} to inject upstream text (e.g., {step_1[0].aux_summary}\\n{step_1[1].aux_summary})."
                ),
                "json_fixer": (
                    "You repair malformed JSON so it conforms to a given JSON Schema. "
                    "Return ONLY the corrected JSON object; no commentary, no code fences."
                ),
            },
            "limits": {
                "max_steps": 10,
                "max_retries": 2,
                "planning_retries": 3,
                "eval_strictness": "normal",
                "max_repair_rounds": 1
            },
        }
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f: json.dump(default, f, ensure_ascii=False, indent=2)
            return default
        try:
            with open(path, "r", encoding="utf-8") as f: user_cfg = json.load(f)
        except Exception:
            return default
        for sec in ("system", "limits"):
            user_cfg.setdefault(sec, {})
            for k, v in default[sec].items():
                user_cfg[sec].setdefault(k, v)
        return user_cfg

    # ── Public API ───────────────────────────────────────────────────────────

    def plan_and_execute(self, objective: str, session_id: Optional[str] = None,
                         constraints: Optional[str] = None, deliverables: Optional[str] = None) -> dict:
        self._log(f"objective: {objective}")
        if constraints: self._log(f"constraints: {constraints}")
        if deliverables: self._log(f"deliverables: {deliverables}")
        plan = self._make_plan(objective, constraints, deliverables)
        self._sanitize_plan(plan)
        report = self._execute_plan(plan, session_id=session_id, constraints=constraints, deliverables=deliverables)
        audited = self._audit_and_repair(plan, report, session_id=session_id)
        return audited

    run = plan_and_execute
    execute = plan_and_execute

    # ── Planning (model-led; hardened JSON) ──────────────────────────────────

    def _make_plan(self, objective: str, constraints: Optional[str], deliverables: Optional[str]) -> PlanSpec:
        sysmsg = self.system_override or self.cfg["system"]["planner"]

        catalog = []
        for name, fn in self.tools.items():
            try: sig = str(inspect.signature(fn))
            except Exception: sig = "(...)"
            doc = (inspect.getdoc(fn) or "").strip()
            catalog.append({"name": name, "signature": sig, "doc": doc[:1200]})

        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PlanSpec",
            "type": "object",
            "required": ["steps"],
            "properties": {
                "notes": {"type": "string"},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": int(self.cfg["limits"].get("max_steps", 10)),
                    "items": {
                        "type": "object",
                        "required": ["id","title","instructions","tool","args"],
                        "properties": {
                            "id":{"type":"string"},
                            "title":{"type":"string"},
                            "instructions":{"type":"string"},
                            "tool":{"type":"string"},
                            "args":{"type":"object"},
                            "deps":{"type":"array","items":{"type":"string"},"default":[]},
                            "success_criteria":{"type":"string"},
                            "retries":{"type":"integer","minimum":0,"maximum":int(self.cfg["limits"].get("max_retries",2))},
                            "outputs":{"type":"array","items":{"type":"string"},"default":[]}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "additionalProperties": False
        }

        extra = getattr(self, "_session_guidance", "")
        user = (
            f"Objective:\n{objective}\n\n"
            f"Constraints:\n{constraints or '(none)'}\n\n"
            f"Deliverables:\n{deliverables or '(none)'}\n\n"
            f"Available tools:\n{json.dumps(catalog, ensure_ascii=False, indent=2)}"
            + (f"\n\nAdditional guidance:\n{extra}" if extra else "")
        )

        self._log("[plan] generating plan…")
        plan_obj = self._structured_json_with_repair(
            role="planner",
            system=sysmsg,
            user_message=user,
            json_schema=schema,
            attempts=int(self.cfg["limits"].get("planning_retries", 3)) + 1
        )

        steps: List[PlanStep] = []
        for s in plan_obj.get("steps", []):
            steps.append(PlanStep(
                id=s["id"].strip(),
                title=s["title"].strip(),
                instructions=s["instructions"].strip(),
                tool=s.get("tool"),
                args=s.get("args"),
                deps=s.get("deps") or [],
                success_criteria=s.get("success_criteria"),
                retries=int(s.get("retries", 1)),
                outputs=s.get("outputs") or []
            ))
        notes = plan_obj.get("notes") or ""
        self._log(f"plan_steps={len(steps)}")
        return PlanSpec(objective=objective, steps=steps, notes=notes)

    # Hardened structured-json with repair
    def _structured_json_with_repair(self, *, role: str, system: str, user_message: str, json_schema: dict, attempts: int = 2) -> dict:
        """
        Try: direct structured_json → tolerant parse → LLM JSON-fixer → parse again.
        Logs each failure cause. No deterministic plan fallback.
        """
        temp = 0.2 if role == "planner" else 0.0
        cog = CognitionChat(CognitionConfig(model=self.model, temperature=temp, stream=False, global_system_prelude=""))
        cog.set_system(system)

        last_err = None
        for attempt in range(1, max(1, attempts) + 1):
            try:
                raw = cog.structured_json(user_message=user_message, json_schema=json_schema, stream=False)
                if isinstance(raw, dict):
                    return raw
                text = _strip_control_chars(str(raw or ""))
                try:
                    return self._tolerant_json_parse(text)
                except Exception as e1:
                    self._log(f"[plan] planning attempt {attempt} failed: {e1}")
                    # try LLM JSON-fixer
                    try:
                        repaired = _strip_control_chars(self._json_fixer(text, json_schema))
                        return self._tolerant_json_parse(repaired)
                    except Exception as e2:
                        last_err = e2
                        self._log(f"[plan] planning attempt {attempt} repair failed: {e2}")
                        continue
            except Exception as e:
                last_err = e
                self._log(f"[plan] planning attempt {attempt} failed: {e}")
                continue
        # Exhausted
        raise RuntimeError(f"planner failed after {attempts} attempts: {last_err}")

    def _json_fixer(self, broken: str, schema: dict) -> str:
        sysmsg = self.cfg["system"]["json_fixer"]
        cog = CognitionChat(CognitionConfig(model=self.model, temperature=0.0, stream=False, global_system_prelude=""))
        cog.set_system(sysmsg)
        prompt = (
            "JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n\n"
            "BROKEN JSON (or text containing JSON):\n"
            f"{_strip_control_chars(broken)[:18000]}"
        )
        fixed = cog.raw_chat(prompt, stream=False) or ""
        return _strip_control_chars(fixed.strip())

    # Tolerant JSON extraction/parsing helpers
    def _tolerant_json_parse(self, text: str) -> dict:
        s = _strip_control_chars(str(text or "")).strip()
        # Strip code fences
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
        # Remove BOM & invisible junk
        s = s.replace("\ufeff", "")
        # Extract the largest balanced {...} region
        s_obj = self._extract_balanced_object(s)
        # Remove line comments //… and trailing commas ,]
        s_obj = re.sub(r"//.*?$", "", s_obj, flags=re.M)
        s_obj = re.sub(r",(\s*[}\]])", r"\1", s_obj)
        return json.loads(s_obj)

    def _extract_balanced_object(self, s: str) -> str:
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise ValueError("no JSON object braces found")
        # scan for best-balanced region
        best_start, best_end, depth, best_len = None, None, 0, 0
        start = None
        for i, ch in enumerate(s):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    end = i + 1
                    if end - start > best_len:
                        best_len = end - start
                        best_start, best_end = start, end
        if best_start is None:
            # fallback to naive slice
            return s[first:last+1]
        return s[best_start:best_end]

    # ── Plan sanitation ──────────────────────────────────────────────────────

    def _sanitize_plan(self, plan: PlanSpec):
        ids = {s.id for s in plan.steps}
        for s in plan.steps:
            s.deps = [d for d in s.deps if d in ids and d != s.id]
            if not isinstance(s.args, dict): s.args = {}

    # ── Execute ──────────────────────────────────────────────────────────────

    def plan_graph(self, plan: PlanSpec) -> List[PlanStep]:
        return self._toposort(plan.steps) or plan.steps[:]

    def _execute_plan(self, plan: PlanSpec, *, session_id: Optional[str],
                      constraints: Optional[str], deliverables: Optional[str]) -> dict:
        ordered = self.plan_graph(plan)
        shared = {"objective": plan.objective, "constraints": constraints, "deliverables": deliverables}
        artifacts: Dict[str, Any] = {}
        step_results: Dict[str, dict] = {}
        failures: List[str] = []
        success_count = 0

        inv_id = None
        try:
            if self.memory and hasattr(self.memory, "add_invocation"):
                inv_id = self.memory.add_invocation(session_id=session_id or f"plan-{uuid.uuid4().hex[:8]}",
                                                    mode="planning/engine", user_input=plan.objective, meta={"notes": plan.notes})
        except Exception:
            inv_id = None

        for step in ordered:
            self._log(f"→ Step {step.id}: {step.title}  deps={step.deps}")
            ok, out, tries, reason = self._run_step_with_retries(step, shared, artifacts)
            step_results[step.id] = {"ok": ok, "tries": tries, "output": out, "reason": reason,
                                     "title": step.title, "tool": step.tool, "args": step.args}
            self._register_artifacts(step, out, artifacts)

            # Create and attach a handoff card (non-streaming)
            try:
                cog = CognitionChat(CognitionConfig(model=self.model, temperature=0.0, stream=False, global_system_prelude=""))
                handoff = CognitionHandoff.annotate(
                    step_id=step.id,
                    artifact=out,
                    context={"objective": plan.objective},
                    llm=cog
                )
                self._attach_handoff(step.id, handoff, artifacts)
            except Exception:
                pass

            try:
                if self.memory and hasattr(self.memory, "add_memory"):
                    mem_id = self.memory.add_memory(session_id=session_id or f"plan-{uuid.uuid4().hex[:8]}",
                                                    role="assistant",
                                                    content=json.dumps({"step": step.id, "title": step.title, "ok": ok, "output": out}, ensure_ascii=False)[:8000],
                                                    source="planning_step",
                                                    tags=[f"step:{step.id}", f"tool:{step.tool or 'none'}"])
                    if inv_id and hasattr(self.memory, "link_invocation_memory"):
                        self.memory.link_invocation_memory(invocation_id=inv_id, memory_id=mem_id, relation="produced")
            except Exception:
                pass

            if ok: success_count += 1
            else:  failures.append(step.id)

        overall_ok = (len(failures) == 0)
        try:
            if self.policy and hasattr(self.policy, "feedback"):
                self.policy.feedback.log("planning", 1.0 if overall_ok else 0.0)
        except Exception:
            pass

        return {
            "success": overall_ok,
            "summary": {"total_steps": len(ordered), "successes": success_count, "failures": failures},
            "plan": {"objective": plan.objective, "notes": plan.notes, "steps": [self._step_to_dict(s) for s in plan.steps]},
            "artifacts": artifacts,
            "logs": self.logs[-800:],
        }

    # ── Handoff attach ───────────────────────────────────────────────────────

    def _attach_handoff(self, step_id: str, handoff: dict, artifacts: dict):
        artifacts[f"{step_id}.handoff"] = handoff
        if isinstance(handoff, dict):
            if handoff.get("primary_text"):
                artifacts[f"{step_id}.primary_text"] = handoff["primary_text"]
            if handoff.get("compact_query"):
                artifacts[f"{step_id}.compact_query"] = handoff["compact_query"]
        # Promote latest compact/friendly globals when empty
        if "title" not in artifacts and f"{step_id}.title" in artifacts:
            artifacts["title"] = artifacts[f"{step_id}.title"]
        if "content" not in artifacts and f"{step_id}.primary_text" in artifacts:
            artifacts["content"] = artifacts[f"{step_id}.primary_text"]
        if "aux_summary" not in artifacts and f"{step_id}.handoff" in artifacts:
            ps = artifacts[f"{step_id}.handoff"].get("primary_text")
            if ps: artifacts["aux_summary"] = str(ps)[:800]

    # ── Artifacts & binding ─────────────────────────────────────────────────

    def _register_artifacts(self, step: PlanStep, output: Any, artifacts: dict):
        artifacts[step.id] = output
        artifacts[f"{step.id}.output"] = output

        if isinstance(output, list) and output and isinstance(output[0], dict):
            first = output[0]
            artifacts.setdefault("title", str(first.get("title") or ""))
            artifacts.setdefault("url", str(first.get("url") or ""))
            artifacts.setdefault("snippet", str(first.get("snippet") or ""))
            artifacts.setdefault("content", str(first.get("content") or ""))
            if "aux_summary" in first: artifacts.setdefault("aux_summary", str(first.get("aux_summary") or ""))
            if "extracted" in first:   artifacts.setdefault("extracted", str(first.get("extracted") or ""))
        elif isinstance(output, dict):
            for k, v in output.items():
                artifacts[f"{step.id}.{k}"] = v

        self._bind_declared_outputs(step, output, artifacts)

    def _bind_declared_outputs(self, step: PlanStep, output: Any, artifacts: dict):
        labels = [str(x).strip() for x in (step.outputs or []) if str(x).strip()]
        if not labels: return
        if isinstance(output, list):
            for i, label in enumerate(labels):
                if i >= len(output): break
                val = output[i]
                artifacts[label] = val
                artifacts[f"{label}.output"] = val
                if isinstance(val, dict):
                    for k, v in val.items():
                        artifacts[f"{label}.{k}"] = v
        else:
            label = labels[0]
            artifacts[label] = output
            artifacts[f"{label}.output"] = output
            if isinstance(output, dict):
                for k, v in output.items():
                    artifacts[f"{label}.{k}"] = v

    # ── Step run & retries ───────────────────────────────────────────────────

    def _run_step_with_retries(self, step: PlanStep, shared: dict, artifacts: dict) -> Tuple[bool, Any, int, str]:
        max_retries_cfg = int(self.cfg["limits"].get("max_retries", 2))
        max_retries = max(0, min(step.retries if step.retries is not None else 1, max_retries_cfg))
        tries = 0
        args = step.args or self._llm_plan_args(step, shared, artifacts)
        if not isinstance(args, dict): args = {}

        while True:
            tries += 1
            try:
                out = self._invoke_tool(step, args, shared, artifacts)
            except Exception as e:
                # Micro-repair: align args and retry ONCE within the same attempt window
                self._log(f"[step {step.id}] tool '{step.tool}' crashed: {e}")
                try:
                    fn = self.tools.get(step.tool) if step.tool else None
                    if fn:
                        realigned = self._align_args_to_signature(fn, args)
                        if realigned != args:
                            self._log(f"[step {step.id}] retrying with signature-aligned args")
                            out = fn(**self._filter_args_for_callable(fn, realigned))
                        else:
                            raise
                    else:
                        raise
                except Exception as e2:
                    out = {"error": str(e2), "traceback": traceback.format_exc()[-1200:]}

            ok, reason, improve, fix_args = self._evaluate_step(step, out, shared, artifacts)
            if ok:
                self._log(f"[step {step.id}] ✓ pass — {reason}")
                return True, out, tries, reason

            self._log(f"[step {step.id}] ✗ fail — {reason}")
            if tries > max_retries:
                return False, out, tries, reason

            if isinstance(fix_args, dict) and fix_args:
                rename_to = fix_args.pop("_rename_tool", None)
                if rename_to: step.tool = str(rename_to)
                args = self._merge_args(args, fix_args, step.tool)
            if improve and isinstance(improve, str) and improve.strip():
                step.instructions = improve.strip()
            time.sleep(0.05)

    # ── Invocation (model-led; no deterministic routes) ──────────────────────

    def _invoke_tool(self, step: PlanStep, args: dict, shared: dict, artifacts: dict):
        name = step.tool or ""
        if not name:
            return {"message": "No tool specified; reasoning-only step.", "instructions": step.instructions}

        fn = self.tools.get(name)
        if not fn:
            artifacts[f"{step.id}.tool"] = name
            artifacts[f"{step.id}.used_args"] = dict(args or {})
            return {"error": f"unknown_tool:{name}", "available_tools": list(self.tools.keys())[:64]}

        dep_hint = step.deps[-1] if step.deps else None
        args = self._normalize_args_for_resolve(args or {}, artifacts, dep_hint)

        # Aliases → canonical (pre-align)
        if "filepath" not in args and "filename" in args:
            args["filepath"] = args.pop("filename")
        if "filepath" in args and "base_dir" not in args:
            args["base_dir"] = os.getcwd()

        # Align to signature (handles synonyms)
        args = self._align_args_to_signature(fn, args)

        # If a query-like arg is too long/multiline, prefer the nearest handoff.compact_query
        try:
            sig = inspect.signature(fn)
            query_param_name = None
            for pname in sig.parameters.keys():
                low = pname.lower()
                if any(tok in low for tok in ("topic","query","q","search","term","subject")):
                    query_param_name = pname
                    break
            if query_param_name and query_param_name in args:
                qv = str(args[query_param_name] or "")
                if len(qv) > 180 or "\n" in qv:
                    cq = None
                    if dep_hint and artifacts.get(f"{dep_hint}.compact_query"):
                        cq = artifacts.get(f"{dep_hint}.compact_query")
                    elif artifacts.get("compact_query"):
                        cq = artifacts.get("compact_query")
                    # fall back to sanitized version of the overlong text
                    if not cq:
                        cq = CognitionHandoff.sanitize_query(qv, fallback=shared.get("objective") or "news")
                    args[query_param_name] = cq
        except Exception:
            pass

        # Fill any remaining required params conservatively
        args = self._complete_required_args(fn, args, step, shared, artifacts, dep_hint)

        call_args = self._filter_args_for_callable(fn, args)
        artifacts[f"{step.id}.tool"] = name
        artifacts[f"{step.id}.used_args"] = dict(call_args)
        return fn(**call_args)

    # ── Arg normalization & selector expansion ───────────────────────────────

    _BARE_SELECTOR = re.compile(
        r"^(?:"
        r"[A-Za-z0-9_\-]+(?:\[(?:-?\d+)\])?(?:\.[A-Za-z0-9_\-]+)?"
        r"|[A-Za-z0-9_\-]+\.outputs\[(?:-?\d+)\](?:\.[A-Za-z0-9_\-]+)?"
        r")$"
    )

    def _normalize_args_for_resolve(self, args: dict, artifacts: dict, last_dep: Optional[str]) -> dict:
        normalized = {}
        for k, v in (args or {}).items():
            if isinstance(v, str):
                s = v.strip()
                if self._BARE_SELECTOR.match(s):
                    s = "{" + s + "}"
                normalized[k] = self._expand_braces_in_text(s, artifacts, last_dep)
            else:
                normalized[k] = v
        return normalized

    def _expand_braces_in_text(self, template: str, artifacts: dict, last_dep: Optional[str]) -> str:
        if "{" not in str(template):
            return str(template)

        def repl(m):
            token = m.group(1).strip()
            val = self._resolve_selector(token, artifacts, last_dep=last_dep)
            return "" if val is None else self._stringify(val)

        out = re.sub(r"{([^{}]+)}", repl, str(template))
        if not out.strip():
            self._log("[args] brace expansion produced empty string; upstream may be missing")
        return out

    def _resolve_selector(self, token: str, artifacts: dict, last_dep: Optional[str]) -> Any:
        # Direct
        if token in artifacts:
            return artifacts[token]

        # <id>.outputs[i].field
        m = re.match(r"^(?P<id>[A-Za-z0-9_\-]+)\.outputs\[(?P<idx>-?\d+)\](?:\.(?P<field>[A-Za-z0-9_\-]+))?$", token)
        if m:
            sid, idx = m.group("id"), int(m.group("idx"))
            base = artifacts.get(sid)
            field = m.group("field")
            if isinstance(base, list) and base:
                i = idx if (-len(base) <= idx < len(base)) else max(0, min(idx, len(base)-1))
                item = base[i]
                if field:
                    if isinstance(item, dict): return item.get(field)
                    return None
                return item
            return None

        # <id>[i].field
        m = re.match(r"^(?P<id>[A-Za-z0-9_\-]+)\[(?P<idx>-?\d+)\]\.(?P<field>[A-Za-z0-9_\-]+)$", token)
        if m:
            sid, idx, field = m.group("id"), int(m.group("idx")), m.group("field")
            base = artifacts.get(sid)
            if isinstance(base, list) and base:
                i = idx if (-len(base) <= idx < len(base)) else max(0, min(idx, len(base)-1))
                item = base[i]
                if isinstance(item, dict): return item.get(field)
                return item
            return artifacts.get(f"{sid}.{field}")

        # <id>.field
        m = re.match(r"^(?P<id>[A-Za-z0-9_\-]+)\.(?P<field>[A-Za-z0-9_\-]+)$", token)
        if m:
            sid, field = m.group("id"), m.group("field")
            direct = artifacts.get(f"{sid}.{field}")
            if direct is not None: return direct
            base = artifacts.get(sid)
            if isinstance(base, list) and base and isinstance(base[0], dict): return base[0].get(field)
            if isinstance(base, dict): return base.get(field)
            return None

        # scoped to last_dep
        if last_dep and f"{last_dep}.{token}" in artifacts:
            return artifacts.get(f"{last_dep}.{token}")

        # friendly globals
        return artifacts.get(token)

    # ── Signature alignment & completion ─────────────────────────────────────

    def _align_args_to_signature(self, fn: Callable[..., Any], args: dict) -> dict:
        """Map common synonyms onto the function's actual parameter names."""
        try:
            sig = inspect.signature(fn)
        except Exception:
            return dict(args)

        PATH_KEYS    = {"filepath","filename","file","path","dest","destination","output","output_path","out"}
        CONTENT_KEYS = {"content","text","body","data","payload","value"}
        TOPIC_KEYS   = {"topic","query","q","search","term","prompt","subject"}
        NUM_KEYS     = {"num_results","n","limit","top_n","count","num"}
        DIR_KEYS     = {"base_dir","dir","directory","folder","out_dir"}

        src = dict(args)
        aligned: Dict[str, Any] = dict(args)

        def pull(keys: set[str]):
            for k in list(src.keys()):
                if k in keys:
                    return k, src[k]
            return None, None

        for pname, p in sig.parameters.items():
            if pname in aligned: continue
            low = pname.lower()
            if any(tok in low for tok in ("file","path","dest","output")):
                k,v = pull(PATH_KEYS)
                if k is not None:
                    aligned[pname] = v
                    continue
            if any(tok in low for tok in ("content","text","body","data","payload","value")):
                k,v = pull(CONTENT_KEYS)
                if k is not None:
                    aligned[pname] = v
                    continue
            if any(tok in low for tok in ("topic","query","term","prompt","subject","search")):
                k,v = pull(TOPIC_KEYS)
                if k is not None:
                    aligned[pname] = v
                    continue
            if any(tok in low for tok in ("num","count","limit","top","results")):
                k,v = pull(NUM_KEYS)
                if k is not None:
                    aligned[pname] = v
                    continue
            if any(tok in low for tok in ("dir","folder")):
                k,v = pull(DIR_KEYS)
                if k is not None:
                    aligned[pname] = v
                    continue

        if "filepath" in sig.parameters and "filepath" not in aligned:
            for k in PATH_KEYS:
                if k in src:
                    aligned["filepath"] = src[k]; break
        if "filename" in sig.parameters and "filename" not in aligned:
            for k in PATH_KEYS:
                if k in src:
                    aligned["filename"] = src[k]; break

        if any(n in aligned for n in ("filepath","filename","path","output","dest","destination","output_path")) \
           and ("base_dir" in sig.parameters) and ("base_dir" not in aligned):
            aligned["base_dir"] = os.getcwd()

        return aligned

    def _complete_required_args(self, fn: Callable[..., Any], args: dict, step: PlanStep,
                                shared: dict, artifacts: dict, last_dep: Optional[str]) -> dict:
        try:
            sig = inspect.signature(fn)
        except Exception:
            return args
        completed = dict(args or {})
        required = []
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL): continue
            if p.default is inspect._empty and p.name not in completed:
                required.append(p.name)

        if not required: return completed

        def best_text() -> str:
            if last_dep:
                for key in (f"{last_dep}.primary_text", f"{last_dep}.extracted", f"{last_dep}.aux_summary",
                            f"{last_dep}.content", f"{last_dep}.title", f"{last_dep}.url"):
                    if artifacts.get(key):
                        return self._stringify(artifacts[key])[:256]
            for key in ("primary_text","extracted","aux_summary","content","title","url","snippet"):
                if artifacts.get(key): return self._stringify(artifacts[key])[:256]
            return str(shared.get("objective") or step.title or "").strip()[:256]

        seed = best_text()
        # If the tool needs a topic and a URL was passed by planner, derive compact topic
        has_url_hint = any(k for k in ("url","link","href") if k in completed)
        for name in required:
            lname = name.lower()
            if any(tok in lname for tok in ("topic","query","term","prompt","subject","search","q")):
                if has_url_hint and isinstance(completed.get("url"), str):
                    completed[name] = CognitionHandoff.sanitize_query(completed.get("url"), fallback=shared.get("objective") or "news")
                else:
                    completed[name] = CognitionHandoff.sanitize_query(seed, fallback=shared.get("objective") or "news")
            elif any(tok in lname for tok in ("file","path","dest","output")):
                completed[name] = self._auto_name_from_objective(shared.get("objective"), fallback="output.txt")
            elif any(tok in lname for tok in ("content","text","body","data","payload","value")):
                # Prefer most recent handoff.primary_text
                if last_dep and artifacts.get(f"{last_dep}.primary_text"):
                    completed[name] = artifacts.get(f"{last_dep}.primary_text")
                else:
                    completed[name] = seed
            elif any(tok in lname for tok in ("num","count","limit","top","results")):
                completed[name] = 5
            elif lname == "base_dir":
                completed[name] = os.getcwd()
            else:
                completed[name] = seed
        return completed

    # ── LLM arg planner ─────────────────────────────────────────────────────

    def _llm_plan_args(self, step: PlanStep, shared: dict, artifacts: dict) -> dict:
        sysmsg = self.cfg["system"]["arg_planner"]
        cog = CognitionChat(CognitionConfig(model=self.model, temperature=0.2, stream=False, global_system_prelude=""))
        cog.set_system(sysmsg)

        fn = self.tools.get(step.tool) if step.tool else None
        name = step.tool or "(none)"
        sig = "(...)"; doc = ""
        if fn:
            try: sig = str(inspect.signature(fn))
            except Exception: pass
            doc = (inspect.getdoc(fn) or "").strip()

        prior = self._build_artifact_snapshot(artifacts)
        examples = self._selector_examples(artifacts)

        prompt = (
            f"Step ID: {step.id}\nTitle: {step.title}\nInstructions:\n{step.instructions}\n\n"
            f"Objective: {shared.get('objective')}\nConstraints: {shared.get('constraints')}\n"
            f"Deliverables: {shared.get('deliverables')}\n\n"
            f"Tool: {name}\nSignature: {sig}\nDocstring:\n{doc[:1200]}\n\n"
            f"Recent artifacts (values trimmed):\n{json.dumps(prior, ensure_ascii=False, indent=2)}\n\n"
            f"Selector examples you can use with {{curly}}:\n{examples}\n\n"
            "Return ONLY the JSON object of arguments for this tool. Use {curly} selectors where appropriate."
        )

        try:
            raw = cog.structured_json(
                user_message=prompt,
                json_schema={"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": True},
                stream=False
            )
            if isinstance(raw, dict):
                return raw
            return self._tolerant_json_parse(raw)
        except Exception as e:
            self._log(f"[args/{step.id}] arg planning failed: {e}")
            return {}

    def _build_artifact_snapshot(self, artifacts: dict) -> dict:
        snap = {}
        # Include last ~12 artifacts; preserve primary_text fields more fully
        for k, v in list(artifacts.items())[-12:]:
            if k.endswith(".primary_text"):
                snap[k] = str(v)[:1500]
            elif isinstance(v, (dict, list)):
                try: snap[k] = json.dumps(v, ensure_ascii=False)[:300]
                except Exception: snap[k] = str(v)[:300]
            else:
                snap[k] = str(v)[:300]
        snap["keys_hint"] = list(artifacts.keys())[-60:]
        return snap

    def _selector_examples(self, artifacts: dict) -> str:
        out = []
        for k in list(artifacts.keys()):
            v = artifacts.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                fields = [f for f in ("title","url","snippet","content","aux_summary","extracted") if f in v[0]]
                if fields:
                    ex = f"{{{{{k}[0].{fields[0]}}}}}, {{{{{k}[1].{fields[0]}}}}}"
                    out.append(f"- From {k}: {ex}")
            if len(out) >= 3: break
        out.append("- Friendly globals: {title}, {url}, {snippet}, {content}, {aux_summary}, {extracted}")
        out.append("- Handoff fields: {<dep>.primary_text}, {<dep>.compact_query}")
        return "\n".join(out)

    # ── Evaluator ────────────────────────────────────────────────────────────

    TOKEN_LIKE = re.compile(r"^(?:[A-Za-z0-9_\-]+(?:\.outputs\[\d+\])?(?:\.[A-Za-z0-9_\-]+)?|step_\d+\.outputs\[\d+\])$")

    def _evaluate_step(self, step: PlanStep, output: Any, shared: dict, artifacts: dict) -> Tuple[bool, str, str, Optional[dict]]:
        actual_tool = artifacts.get(f"{step.id}.tool") or (step.tool or "")
        used_args = artifacts.get(f"{step.id}.used_args") or (step.args or {})

        if isinstance(output, dict) and str(output.get("error", "")).startswith("unknown_tool:"):
            bad = output["error"].split(":", 1)[-1]
            return False, f"unknown tool: {bad}", "Pick a valid tool from available_tools.", {"_rename_tool": None}

        # Generic search/web result check
        if isinstance(output, list) and (re.search(r"(search|web|internet)", actual_tool, flags=re.I) or any(isinstance(x, dict) for x in output)):
            ok = False; reason = "no usable results"
            if output and isinstance(output[0], dict):
                f = output[0]
                if any(bool(str(f.get(k, "")).strip()) for k in ("content","extracted","aux_summary","title","url","snippet")):
                    ok = True; reason = "non-empty results with meaningful fields"
            if ok: return True, reason, "", None
            fix = {"summarize": True}
            if not any(k in used_args for k in ("topic","query","q")):
                fix["topic"] = shared.get("objective") or step.title or "news"
            return False, reason, "", fix

        # File ops: must exist, non-empty, no unresolved tokens
        FILE_KEYS = ("filepath","filename","path","output","dest","destination","output_path")
        if re.search(r"(write|create|save).*file", str(actual_tool), flags=re.I) or any(k in used_args for k in FILE_KEYS):
            try:
                base_dir = used_args.get("base_dir", os.getcwd())
                path_val = None
                for k in FILE_KEYS:
                    if k in used_args:
                        path_val = used_args.get(k)
                        break
                if not path_val:
                    return False, "file op missing 'filepath'/'filename'", "", {"filepath": self._auto_name_from_objective(shared.get("objective"), "output.txt")}
                abs_path = os.path.join(base_dir, path_val)
                if not os.path.exists(abs_path):
                    return False, f"file not found: {abs_path}", "", None
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    body = f.read().strip()
                if not body:
                    # Propose {<dep>.primary_text} if available
                    last_dep = step.deps[-1] if step.deps else None
                    patch = {}
                    if last_dep and artifacts.get(f"{last_dep}.primary_text"):
                        patch["content"] = f"{{{last_dep}.primary_text}}"
                    return False, "file is empty", "Inject upstream content via {curly} selectors.", patch or None
                if re.search(r"{[^{}]+}", body) or self.TOKEN_LIKE.match(body):
                    last_dep = step.deps[-1] if step.deps else None
                    suggest = "{content}"
                    if last_dep and artifacts.get(f"{last_dep}.primary_text"):
                        suggest = f"{{{last_dep}.primary_text}}"
                    return False, "file contains unresolved selector text", f"Use a brace selector like {suggest}.", {"content": suggest}
                candidate = (artifacts.get("extracted") or artifacts.get("aux_summary") or artifacts.get("content") or artifacts.get("title") or artifacts.get("primary_text"))
                if isinstance(candidate, str):
                    snip = candidate.strip()[:40]
                    if snip and snip not in body:
                        return True, "file created; upstream overlap not detected (allowed)", "", None
                return True, "file exists and content looks valid", "", None
            except Exception as e:
                return False, f"file evaluation error: {e}", "", None

        # Error passthrough
        if isinstance(output, dict) and "error" in output:
            return False, output.get("error")[:200], "", None

        # Non-empty generic
        ok_generic = bool(str(output).strip())
        if ok_generic: return True, "non-empty output", "", None

        # LLM fallback eval
        sysmsg = self.cfg["system"]["evaluator"]
        cog = CognitionChat(CognitionConfig(model=self.model, temperature=0.0, stream=False, global_system_prelude=""))
        cog.set_system(sysmsg)
        preview = json.dumps(output, ensure_ascii=False)[:4000] if isinstance(output, (dict, list)) else str(output)[:4000]
        payload = {
            "objective": shared.get("objective"),
            "step": {"id": step.id, "title": step.title, "instructions": step.instructions,
                     "success_criteria": step.success_criteria or "(not specified)"},
            "output_preview": preview,
            "strictness": self.cfg["limits"].get("eval_strictness","normal"),
        }
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "EvalResult",
            "type": "object",
            "required": ["pass","reason"],
            "properties": {"pass":{"type":"boolean"},"reason":{"type":"string"},
                           "improve":{"type":"string"},"fix_args":{"type":"object"}},
            "additionalProperties": False
        }
        try:
            raw = cog.structured_json(user_message=json.dumps(payload, ensure_ascii=False), json_schema=schema, stream=False)
            res = raw if isinstance(raw, dict) else self._tolerant_json_parse(raw)
            ok = bool(res.get("pass")); reason = str(res.get("reason",""))
            improve = str(res.get("improve") or "") if res.get("improve") is not None else ""
            fix_args = res.get("fix_args") if isinstance(res.get("fix_args"), dict) else None
            return ok, reason, improve, fix_args
        except Exception:
            ok = bool(str(output).strip())
            return ok, "non-empty output" if ok else "empty output", "", None

    # ── Audit & repair (single targeted patch; no route hard-coding) ─────────

    def _audit_and_repair(self, plan: PlanSpec, report: dict, *, session_id: Optional[str]) -> dict:
        candidates: List[Tuple[Optional[PlanStep], str]] = []
        for s in plan.steps:
            used = report["artifacts"].get(f"{s.id}.used_args") or s.args or {}
            tool = report["artifacts"].get(f"{s.id}.tool") or s.tool or ""
            path = None
            for k in ("filepath","filename","path","output","dest","destination","output_path"):
                if k in (used or {}): path = used[k]; break
            if path or re.search(r"(write|create|save).*file", tool or "", flags=re.I):
                if path:
                    base = used.get("base_dir", os.getcwd())
                    candidates.append((s, os.path.join(base, path)))

        obj = plan.objective or ""
        for m in re.findall(r"(?:file (?:called|named)\s+['\"]([^'\"]+)['\"]|`([^`]+)`)", obj, flags=re.I):
            name = (m[0] or m[1]).strip()
            if name: candidates.append((None, os.path.join(os.getcwd(), name)))

        seen = set(); uniq: List[Tuple[Optional[PlanStep], str]] = []
        for pair in candidates:
            if pair[1] not in seen:
                uniq.append(pair); seen.add(pair[1])

        issues = []
        for step, path in uniq:
            try:
                if not os.path.exists(path):
                    issues.append({"step": step.id if step else None, "path": path, "reason": "missing"}); continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    body = f.read().strip()
                if not body:
                    issues.append({"step": step.id if step else None, "path": path, "reason": "empty"}); continue
                if re.search(r"{[^{}]+}", body) or self.TOKEN_LIKE.match(body):
                    issues.append({"step": step.id if step else None, "path": path, "reason": "unresolved_tokens"}); continue
            except Exception as e:
                self._log(f"[audit] error reading {path}: {e}")

        if not issues:
            return report

        self._log("[audit] deliverable issue detected; invoking repair coach")
        sysmsg = self.cfg["system"]["repair_coach"]
        cog = CognitionChat(CognitionConfig(model=self.model, temperature=0.2, stream=False, global_system_prelude=""))
        cog.set_system(sysmsg)

        context = {
            "objective": plan.objective,
            "plan_steps": [self._step_to_dict(s) for s in plan.steps],
            "issues": issues,
            "artifact_keys": list(report["artifacts"].keys())[-200:],
        }
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "RepairPatch",
            "type": "object",
            "required": ["target_step", "new_args"],
            "properties": {
                "target_step": {"type": "string"},
                "new_args": {"type": "object"},
                "rename_tool": {"type": ["string","null"]},
                "rationale": {"type": "string"}
            },
            "additionalProperties": False
        }

        try:
            patch = cog.structured_json(user_message=json.dumps(context, ensure_ascii=False), json_schema=schema, stream=False)
            patch = patch if isinstance(patch, dict) else self._tolerant_json_parse(patch)
        except Exception as e:
            self._log(f"[audit] repair planning failed: {e}")
            return report

        target_id = patch.get("target_step")
        new_args = patch.get("new_args") if isinstance(patch.get("new_args"), dict) else None
        rename_tool = patch.get("rename_tool")
        if not (target_id and new_args is not None): return report

        target_step = next((s for s in plan.steps if s.id == target_id), None)
        if not target_step: return report

        if rename_tool: target_step.tool = str(rename_tool)
        target_step.args = self._merge_args(target_step.args or {}, new_args, target_step.tool)

        self._log(f"[audit] applying patch to step {target_id} and re-executing once")
        artifacts = report["artifacts"]
        ok, out, tries, reason = self._run_step_with_retries(target_step, {"objective": plan.objective}, artifacts)
        self._register_artifacts(target_step, out, artifacts)

        return {**report, "audit": {"patched_step": target_id, "ok": ok, "reason": reason}}

    # ── Utilities ────────────────────────────────────────────────────────────

    def _auto_name_from_objective(self, obj: Optional[str], fallback: str = "output.txt") -> str:
        if not obj: return fallback
        m = re.search(r"file (?:named|called)\s+['\"]([^'\"]+)['\"]", obj, flags=re.I)
        if m: return m.group(1).strip()
        m = re.search(r"\b([A-Za-z0-9_\-]+\.(?:txt|md|json|csv|html))\b", obj)
        if m: return m.group(1)
        return fallback

    def _stringify(self, val: Any) -> str:
        if isinstance(val, list):
            if val and isinstance(val[0], dict) and ("title" in val[0] or "url" in val[0]):
                bullets = []
                for r in val[:12]:
                    title = r.get("title") or "(untitled)"
                    url = r.get("url") or ""
                    bullets.append(f"- {title} {f'({url})' if url else ''}".strip())
                return "\n".join(bullets)
            return "\n".join(self._stringify(x) for x in val)
        if isinstance(val, dict):
            try: return json.dumps(val, ensure_ascii=False, indent=2)
            except Exception: return str(val)
        return str(val)

    def _filter_args_for_callable(self, fn: Callable[..., Any], args: dict) -> dict:
        try:
            sig = inspect.signature(fn)
            return {name: args[name] for name in sig.parameters.keys() if name in args}
        except Exception:
            return dict(args)

    def _merge_args(self, base: dict, patch: dict, tool_name: Optional[str]) -> dict:
        merged = dict(base or {}); merged.update(patch or {})
        fn = self.tools.get(tool_name) if tool_name else None
        if fn: merged = self._filter_args_for_callable(fn, merged)
        return merged

    def _toposort(self, steps: List[PlanStep]) -> Optional[List[PlanStep]]:
        id_map = {s.id: s for s in steps}
        indeg = {s.id: 0 for s in steps}
        graph: Dict[str, List[str]] = {s.id: [] for s in steps}
        for s in steps:
            for d in s.deps:
                if d in id_map:
                    graph[d].append(s.id); indeg[s.id] += 1
        queue = [sid for sid, d in indeg.items() if d == 0]
        out_ids: List[str] = []
        while queue:
            sid = queue.pop(0); out_ids.append(sid)
            for nxt in graph.get(sid, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0: queue.append(nxt)
        if len(out_ids) != len(steps): return None
        return [id_map[sid] for sid in out_ids]

    def _step_to_dict(self, s: PlanStep) -> dict:
        return {
            "id": s.id, "title": s.title, "instructions": s.instructions,
            "tool": s.tool, "args": s.args, "deps": s.deps,
            "success_criteria": s.success_criteria, "retries": s.retries, "outputs": s.outputs
        }

    def _log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.logs.append(line)
        try: print(line, flush=True)
        except Exception: pass

    # ── Built-in utility tool ────────────────────────────────────────────────

    def _builtin_llm_bulletize(self, source: Any, max_items: int = 8) -> str:
        text = ""
        if isinstance(source, list):
            lines = []
            for r in source[: max_items]:
                if isinstance(r, dict):
                    title = r.get("title") or ""
                    url = r.get("url") or ""
                    if title or url:
                        lines.append(f"{title} {f'({url})' if url else ''}".strip())
                    elif r.get("aux_summary"):
                        lines.append(str(r["aux_summary"]).strip())
                else:
                    lines.append(str(r))
            text = "\n".join(lines)
        else:
            text = str(source)

        sysmsg = "You produce a concise bullet list from the provided content. Each bullet is one line."
        cog = CognitionChat(CognitionConfig(model=self.model, temperature=0.2, stream=False, global_system_prelude=""))
        cog.set_system(sysmsg)
        prompt = f"Make ≤ {max_items} short bullets of the most relevant items:\n\n{text[:8000]}"
        out = cog.raw_chat(prompt, stream=False)
        return (out or "").strip()
