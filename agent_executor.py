#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_executor.py — Executes planned task steps with retries, tool invocation,
and telemetry hooks.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agent_tasks import TaskManager, TaskStatus, TaskStepRecord, StepType
from agent_safety import validate_tool_output, validate_final_reply
from agent_telemetry import telemetry_event
from agent_capabilities import resolve_tool_callable


@dataclass
class ExecutionContext:
    task_id: int
    chat_id: int
    thread_id: int
    user_id: Optional[int]
    complexity_score: float


class TaskExecutor:
    def __init__(self):
        self.generate_reply_fn = None
        self.log_tool_event_fn = None
        self.context_formatter = None
        self.truncate_fn = None

    def configure_runtime(self, *, generate_reply, log_tool_event, format_tool_context, truncate_text):
        self.generate_reply_fn = generate_reply
        self.log_tool_event_fn = log_tool_event
        self.context_formatter = format_tool_context
        self.truncate_fn = truncate_text

    def _ensure_runtime(self):
        if not all([self.generate_reply_fn, self.log_tool_event_fn, self.context_formatter, self.truncate_fn]):
            raise RuntimeError("TaskExecutor.configure_runtime must be called before use")

    async def run(self, ctx: ExecutionContext):
        self._ensure_runtime()
        steps = await TaskManager.fetch_steps(ctx.task_id)
        for step in steps:
            if await TaskManager.is_cancelled(ctx.task_id):
                await TaskManager.update_task_status(ctx.task_id, status=TaskStatus.CANCELLED)
                telemetry_event("task_cancelled", {"task_id": ctx.task_id})
                return

            if step.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                continue

            await TaskManager.update_task_status(ctx.task_id, status=TaskStatus.IN_PROGRESS, current_step=step.step_index)
            await self._execute_step(ctx, step)

        await TaskManager.update_task_status(ctx.task_id, status=TaskStatus.COMPLETED, current_step=None)

    async def _execute_step(self, ctx: ExecutionContext, step: TaskStepRecord):
        attempts = 0
        max_attempts = step.max_retries + 1
        last_error: Optional[str] = None
        while attempts < max_attempts:
            attempts += 1
            try:
                if step.step_type == StepType.TOOL:
                    await self._run_tool_step(ctx, step)
                elif step.step_type == StepType.LLM:
                    await self._run_llm_step(ctx, step)
                elif step.step_type == StepType.SUMMARY:
                    await self._run_summary_step(ctx, step)
                elif step.step_type == StepType.REVIEW:
                    await self._run_review_step(ctx, step)
                else:
                    raise ValueError(f"Unsupported step type: {step.step_type}")
                await TaskManager.update_step(
                    ctx.task_id,
                    step.step_index,
                    status=TaskStatus.COMPLETED,
                    output_payload=step.output_payload,
                    retries=attempts - 1,
                )
                telemetry_event(
                    "task_step_completed",
                    {
                        "task_id": ctx.task_id,
                        "step_index": step.step_index,
                        "step_type": step.step_type.value,
                        "attempts": attempts,
                    },
                )
                return
            except Exception as exc:
                last_error = str(exc)
                await TaskManager.update_step(
                    ctx.task_id,
                    step.step_index,
                    status=TaskStatus.IN_PROGRESS,
                    error=last_error,
                    retries=attempts,
                )
                telemetry_event(
                    "task_step_retry",
                    {
                        "task_id": ctx.task_id,
                        "step_index": step.step_index,
                        "step_type": step.step_type.value,
                        "attempt": attempts,
                        "error": last_error,
                    },
                )
                await asyncio.sleep(min(5, attempts))

        await TaskManager.update_step(
            ctx.task_id,
            step.step_index,
            status=TaskStatus.FAILED,
            error=last_error,
            retries=attempts,
        )
        await TaskManager.update_task_status(ctx.task_id, status=TaskStatus.FAILED, current_step=step.step_index)
        telemetry_event(
            "task_step_failed",
            {
                "task_id": ctx.task_id,
                "step_index": step.step_index,
                "step_type": step.step_type.value,
                "error": last_error,
            },
        )

    async def _run_tool_step(self, ctx: ExecutionContext, step: TaskStepRecord):
        self._ensure_runtime()
        payload = step.input_payload or {}
        tool_name = payload.get("tool_name")
        tool_args = payload.get("tool_args") or {}
        if not tool_name:
            raise ValueError("tool_name missing from tool step")

        tool_callable = resolve_tool_callable(tool_name)
        if not tool_callable:
            raise ValueError(f"Unknown tool: {tool_name}")

        start_ts = time.time()
        result = await tool_callable(**tool_args) if asyncio.iscoroutinefunction(tool_callable) else tool_callable(**tool_args)
        latency = time.time() - start_ts

        validate_tool_output(tool_name, result)

        structured = result if isinstance(result, (dict, list)) else None
        context_text = self.context_formatter(
            tool_name,
            structured,
            user_request=payload.get("planner_request"),
            default_text=json.dumps(result, ensure_ascii=False) if _is_serializable(result) else str(result),
        )
        context_text = self.truncate_fn(context_text, 1600)
        if context_text:
            self.log_tool_event_fn(
                ctx.chat_id,
                ctx.thread_id,
                tool_name,
                context_text,
                state="completed",
                metadata={"source": "task_executor", "latency": latency},
                raw_output=json.dumps(result, ensure_ascii=False) if _is_serializable(result) else str(result),
            )

        step.output_payload = {"result": result, "latency": latency}

    async def _run_llm_step(self, ctx: ExecutionContext, step: TaskStepRecord):
        self._ensure_runtime()
        payload = step.input_payload or {}
        prompt = payload.get("prompt") or ""
        if not prompt:
            raise ValueError("LLM step missing prompt")

        model_override = payload.get("model")
        result, _ = await self.generate_reply_fn(
            {
                "model": model_override,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            ctx.user_id,
            include_examples=False,
            auto_execute=False,
            max_tools=0,
            user_text=prompt,
        )
        validate_final_reply(result or "")
        step.output_payload = {"reply": result}

    async def _run_summary_step(self, ctx: ExecutionContext, step: TaskStepRecord):
        self._ensure_runtime()
        payload = step.input_payload or {}
        text_to_summarise = payload.get("text") or ""
        if not text_to_summarise:
            step.output_payload = {"summary": ""}
            return

        result, _ = await self.generate_reply_fn(
            {
                "model": payload.get("model"),
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarise the provided text in <= 6 bullet points.",
                    },
                    {"role": "user", "content": text_to_summarise},
                ],
                "stream": False,
            },
            ctx.user_id,
            include_examples=False,
            auto_execute=False,
            max_tools=0,
            user_text=text_to_summarise,
        )
        step.output_payload = {"summary": result}

    async def _run_review_step(self, ctx: ExecutionContext, step: TaskStepRecord):
        self._ensure_runtime()
        payload = step.input_payload or {}
        review_target = payload.get("target") or ""
        if not review_target:
            step.output_payload = {"review": ""}
            return
        result, _ = await self.generate_reply_fn(
            {
                "model": payload.get("model"),
                "messages": [
                    {
                        "role": "system",
                        "content": "Review the provided response for quality gaps. Respond with JSON {\"score\":0-1,\"issues\":[\"...\"]}.",
                    },
                    {"role": "user", "content": review_target},
                ],
                "stream": False,
            },
            ctx.user_id,
            include_examples=False,
            auto_execute=False,
            max_tools=0,
            user_text=review_target,
        )
        step.output_payload = {"review": result}


def _is_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
