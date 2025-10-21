#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_planner.py — Planning utilities for multi-step task generation.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError, validator

from agent_tasks import StepType, TaskRecord, TaskStepRecord, TaskStatus
from agent_capabilities import suggest_tools
from agent_safety import validate_plan_json


class PlanStepModel(BaseModel):
    index: int = Field(..., ge=0)
    kind: StepType
    description: str = Field(..., min_length=1, max_length=600)
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    expected_output: Optional[str] = None
    max_retries: int = Field(1, ge=0, le=5)

    @validator("tool_name", always=True)
    def tool_required(cls, v, values):
        kind = values.get("kind")
        if kind == StepType.TOOL and not v:
            raise ValueError("tool step requires tool_name")
        return v


class PlanModel(BaseModel):
    objective: str = Field(..., min_length=1, max_length=800)
    rationale: str = Field(..., min_length=1, max_length=1200)
    steps: List[PlanStepModel]

    @validator("steps")
    def non_empty(cls, v):
        if not v:
            raise ValueError("plan must contain at least one step")
        return v


@dataclass
class PlannerConfig:
    fast_model: str
    standard_model: str
    advanced_model: str


class TaskPlanner:
    """High level planner that drafts plans using an LLM and validates results."""

    def __init__(self, ai_generate_async, config: PlannerConfig):
        self._ai_generate = ai_generate_async
        self.config = config

    def _select_model(self, complexity_score: float, step_count_hint: int) -> str:
        if complexity_score >= 0.8 or step_count_hint > 6:
            return self.config.advanced_model
        if complexity_score >= 0.4 or step_count_hint > 3:
            return self.config.standard_model
        return self.config.fast_model

    async def draft_plan(
        self,
        *,
        chat_id: int,
        thread_id: int,
        user_prompt: str,
        context_snapshot: str,
        tool_events: str,
        complexity_score: float,
    ) -> Optional[PlanModel]:
        suggested = suggest_tools(user_prompt, tool_events)
        model = self._select_model(complexity_score, len(suggested))

        system_msg = (
            "You are the planning module of Gatekeeper Bot. Build a concise plan with numbered steps. "
            "Use TOOL steps only when tools are required; otherwise use LLM steps. "
            "Return strict JSON matching the schema."
        )
        prompt = textwrap.dedent(
            f"""
            User request:
            {user_prompt}

            Conversation snapshot:
            {context_snapshot}

            Recent tool events:
            {tool_events or "(none)"}

            Candidate tools: {', '.join(suggested) if suggested else '(none)'}

            Produce a plan JSON like:
            {{
              "objective": "...",
              "rationale": "...",
              "steps": [
                {{"index": 0, "kind": "tool", "description": "...", "tool_name": "...", "tool_args": {{...}}, "expected_output": "...", "max_retries": 1}},
                ...
              ]
            }}
            """
        ).strip()

        response = await self._ai_generate(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
        )
        if not response:
            return None

        json_blob = validate_plan_json(response)
        try:
            model = PlanModel.parse_raw(json_blob)
        except ValidationError as exc:
            raise ValueError(f"Plan validation failed: {exc}") from exc
        return model

    @staticmethod
    def plan_to_records(
        plan: PlanModel,
        chat_id: int,
        thread_id: int,
        user_id: Optional[int],
    ) -> (TaskRecord, List[TaskStepRecord]):
        task = TaskRecord(
            chat_id=chat_id,
            thread_id=thread_id,
            user_id=user_id,
            title=plan.objective,
            status=TaskStatus.PENDING,
            metadata={"rationale": plan.rationale},
            current_step=0,
        )
        steps: List[TaskStepRecord] = []
        for step in plan.steps:
            steps.append(
                TaskStepRecord(
                    task_id=-1,  # placeholder until persisted
                    step_index=step.index,
                    step_type=step.kind,
                    objective=step.description,
                    input_payload={
                        "tool_name": step.tool_name,
                        "tool_args": step.tool_args,
                        "expected_output": step.expected_output,
                    },
                    max_retries=step.max_retries,
                )
            )
        return task, steps
