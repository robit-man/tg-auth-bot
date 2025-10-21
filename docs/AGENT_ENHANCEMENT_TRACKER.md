# Agent Enhancement Tracker

This document captures the state of the multi-step agent roadmap. Update this
file whenever new capabilities land so future iterations can resume quickly.

## Overview

| Capability | Owner | Status | Notes |
| --- | --- | --- | --- |
| Tool result grounding (structured feeds) | Core agent | 🟡 In progress | Web search summaries wired; remaining tools use generic summary fallback. |
| Planner/executor split | Core agent | 🟡 In progress | `agent_tasks.py` introduces task graph scaffolding and executor loop. |
| Conversation memory hygiene | Memory | 🟡 In progress | Chain log now stores typed events; summarisation + tagging queued. |
| Safety & validation hooks | Safety | 🟡 In progress | `agent_safety.py` validates tool outputs and final replies. |
| Self-reflection loops | Safety | 🟡 In progress | Critique upgraded to require fixes on policy violations. |
| Observation/action history | Core agent | 🟢 Complete | Chain log events normalised (user/assistant/tool). |
| Task state machine | Core agent | 🟡 In progress | Tasks/steps persisted; UI commands expose `/plan show|cancel|step`. |
| Capabilities discovery | Planner | 🟡 In progress | Tool embeddings + similarity search provided via `agent_capabilities.py`. |
| Parallelism & interrupts | Core agent | 🟡 In progress | Tasks cancellable; execution loop checks status before each step. |
| Telemetry & analytics | Telemetry | 🟡 In progress | `agent_telemetry.py` logs events; aggregation pipeline TODO. |
| Model cascading | Planner | 🟡 In progress | Model selector stub chooses fast/deep models via heuristics. |

Legend: 🟢 Complete · 🟡 In progress · 🔴 Not started

## Implementation Map

### 1. Tool Result Grounding

* `bot_server.py`: `log_chain_tool_event` captures structured outputs and summaries.
* `agent_safety.validate_tool_output`: schema and policy validation for tool results.
* TODO: per-tool schema registry and richer formatting.

### 2. Planner / Executor Split

* `agent_tasks.py`: defines `TaskRecord`, `TaskStepRecord`, and `TaskManager`.
* `agent_planner.py`: uses LLM to draft JSON plans, validates with Pydantic.
* `agent_executor.py`: runs steps with retries, integrates with tool bridge.
* `/plan show|cancel|step` commands surface task state.

### 3. Conversation Memory Hygiene

* Chain log now stores typed events (user/assistant/tool).
* `agent_tasks.TaskManager` writes observation/action tuples per step.
* TODO: add periodic summarisation job and tagging (facts, commitments).

### 4. Safety and Validation Hooks

* `agent_safety.py` exports `validate_tool_output`, `validate_final_reply`.
* `bot_server.py` checks validation before tool logging and outgoing replies.
* Policy filters configurable via `.env` (`SAFETY_POLICY_PATH`).

### 5. Self-Reflection Loop

* `_evaluate_reply` upgraded to escalate when issues detected.
* TODO: add auto-revision loop when critique fails.

### 6. Observation / Action History

* `agent_telemetry.log_observation` records observation/action pairs (SQLite).
* Prompt now includes “Conversation snapshot” + “Tool events” sections.

### 7. Task State Machine

* `agent_tasks.ensure_schema` creates `agent_tasks` and `agent_task_steps`.
* Steps support status transitions + retry counts.
* TODO: add dependency edges beyond linear sequences.

### 8. Capabilities Discovery

* `agent_capabilities.build_tool_index` captures tool metadata + embeddings.
* Planner uses similarity search to suggest relevant tools.
* TODO: expose `/tools suggest <query>` UI command.

### 9. Parallelism & Interrupts

* Task executor polls `TaskManager.is_cancelled` each iteration.
* `/plan cancel` command flips cancellation flag.
* TODO: implement prioritisation / resume semantics.

### 10. Telemetry & Analytics

* `agent_telemetry.py` logs to `agent_telemetry` table + JSON lines file.
* TODO: add aggregation job + dashboards (e.g., in `docs/telemetry.md`).

### 11. Model Cascading

* `agent_planner.select_model` chooses between fast/standard/advanced models.
* Heuristics based on task complexity + plan depth.
* TODO: surface config via `.env` and measure performance.

### 12. Testing

* `tests/test_agent_tasks.py` and `tests/test_agent_planner.py` provide unit smoke tests.
* TODO: integration test harness simulating multi-step scenario with tool failures.

## Next Steps

1. Implement per-tool schema registry + validation (`agent_safety`).
2. Add summarisation cron job for long conversations.
3. Build telemetry aggregation script + dashboard.
4. Flesh out `/plan step` to rerun failed steps with edited inputs.

Maintainers: Update status + notes in the table above when features progress.
