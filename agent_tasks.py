#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_tasks.py — Task and step management for multi-step agent execution.

This module keeps a persistent record of tasks, their steps, status transitions,
and provides helpers for planner/executor orchestration.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from contextlib import closing

DB_PATH: Optional[str] = None
DBW = None  # Will be injected at runtime
db_factory = None


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(str, enum.Enum):
    LLM = "llm"
    TOOL = "tool"
    SUMMARY = "summary"
    REVIEW = "review"


@dataclass
class TaskStepRecord:
    task_id: int
    step_index: int
    step_type: StepType
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    input_payload: Optional[Dict[str, Any]] = None
    output_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 1
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)

    def to_row(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "step_type": self.step_type.value,
            "objective": self.objective,
            "status": self.status.value,
            "input_payload": json.dumps(self.input_payload, ensure_ascii=False) if self.input_payload else None,
            "output_payload": json.dumps(self.output_payload, ensure_ascii=False) if self.output_payload else None,
            "error": self.error,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
        }


@dataclass
class TaskRecord:
    chat_id: int
    thread_id: int
    user_id: Optional[int]
    title: str
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[int] = None
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    cancelled: bool = False
    id: Optional[int] = None


class TaskManager:
    """High-level interface for creating, updating, and querying tasks."""

    @staticmethod
    def configure(db_path: str, dbw, db_callable):
        global DB_PATH, DBW, db_factory
        DB_PATH = db_path
        DBW = dbw
        db_factory = db_callable

    @staticmethod
    def _dbw():
        if not DBW or not DB_PATH:
            raise RuntimeError("TaskManager.configure() must be called before use")
        return DBW

    @staticmethod
    def ensure_schema():
        if not db_factory:
            raise RuntimeError("TaskManager.configure() must be called before use")
        with closing(db_factory()) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    thread_id INTEGER,
                    user_id INTEGER,
                    title TEXT,
                    status TEXT,
                    metadata TEXT,
                    current_step INTEGER,
                    cancelled INTEGER DEFAULT 0,
                    created_ts REAL,
                    updated_ts REAL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_task_steps(
                    task_id INTEGER,
                    step_index INTEGER,
                    step_type TEXT,
                    objective TEXT,
                    status TEXT,
                    input_payload TEXT,
                    output_payload TEXT,
                    error TEXT,
                    retries INTEGER,
                    max_retries INTEGER,
                    created_ts REAL,
                    updated_ts REAL,
                    PRIMARY KEY(task_id, step_index)
                )
                """
            )
            con.commit()

    @staticmethod
    async def create_task(task: TaskRecord, steps: Sequence[TaskStepRecord]) -> TaskRecord:
        TaskManager.ensure_schema()

        def _insert(con):
            metadata_json = json.dumps(task.metadata, ensure_ascii=False) if task.metadata else None
            cur = con.execute(
                """
                INSERT INTO agent_tasks(chat_id, thread_id, user_id, title, status, metadata,
                                        current_step, cancelled, created_ts, updated_ts)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    task.chat_id,
                    task.thread_id,
                    task.user_id,
                    task.title,
                    task.status.value,
                    metadata_json,
                    task.current_step,
                    1 if task.cancelled else 0,
                    task.created_ts,
                    task.updated_ts,
                ),
            )
            task_id = cur.lastrowid
            for step in steps:
                row = step.to_row()
                row["task_id"] = task_id
                con.execute(
                    """
                    INSERT INTO agent_task_steps(task_id, step_index, step_type, objective, status,
                                                 input_payload, output_payload, error, retries,
                                                 max_retries, created_ts, updated_ts)
                    VALUES(:task_id, :step_index, :step_type, :objective, :status,
                           :input_payload, :output_payload, :error, :retries,
                           :max_retries, :created_ts, :updated_ts)
                    """,
                    row,
                )
            con.commit()
            return task_id

        dbw = TaskManager._dbw()
        task_id = await dbw.run(DB_PATH, _insert)
        task.id = int(task_id)
        return task

    @staticmethod
    async def load_task(task_id: int) -> Optional[TaskRecord]:
        def _fetch(con):
            row = con.execute(
                """
                SELECT id, chat_id, thread_id, user_id, title, status, metadata,
                       current_step, cancelled, created_ts, updated_ts
                FROM agent_tasks WHERE id=?
                """,
                (task_id,),
            ).fetchone()
            if not row:
                return None
            metadata = json.loads(row[6]) if row[6] else {}
            return TaskRecord(
                chat_id=row[1],
                thread_id=row[2],
                user_id=row[3],
                title=row[4],
                status=TaskStatus(row[5]),
                metadata=metadata,
                current_step=row[7],
                cancelled=bool(row[8]),
                created_ts=row[9],
                updated_ts=row[10],
                id=row[0],
            )

        dbw = TaskManager._dbw()
        return await dbw.run(DB_PATH, _fetch)

    @staticmethod
    async def list_active_tasks(chat_id: int, thread_id: int | None) -> List[TaskRecord]:
        thread_val = int(thread_id or 0)

        def _fetch(con):
            rows = con.execute(
                """
                SELECT id, chat_id, thread_id, user_id, title, status, metadata,
                       current_step, cancelled, created_ts, updated_ts
                FROM agent_tasks
                WHERE chat_id=? AND thread_id=? AND status IN ('pending','in_progress')
                ORDER BY created_ts ASC
                """,
                (chat_id, thread_val),
            ).fetchall()
            results: List[TaskRecord] = []
            for row in rows:
                metadata = json.loads(row[6]) if row[6] else {}
                results.append(
                    TaskRecord(
                        chat_id=row[1],
                        thread_id=row[2],
                        user_id=row[3],
                        title=row[4],
                        status=TaskStatus(row[5]),
                        metadata=metadata,
                        current_step=row[7],
                        cancelled=bool(row[8]),
                        created_ts=row[9],
                        updated_ts=row[10],
                        id=row[0],
                    )
                )
            return results

        dbw = TaskManager._dbw()
        return await dbw.run(DB_PATH, _fetch)

    @staticmethod
    async def update_task_status(task_id: int, *, status: Optional[TaskStatus] = None,
                                 current_step: Optional[int] = None,
                                 cancelled: Optional[bool] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        def _update(con):
            row = con.execute(
                "SELECT status, metadata, current_step, cancelled FROM agent_tasks WHERE id=?",
                (task_id,),
            ).fetchone()
            if not row:
                return
            new_status = status.value if status else row[0]
            merged_meta = json.loads(row[1]) if row[1] else {}
            if metadata:
                merged_meta.update(metadata)
            con.execute(
                """
                UPDATE agent_tasks
                SET status=?, metadata=?, current_step=?, cancelled=?, updated_ts=?
                WHERE id=?
                """,
                (
                    new_status,
                    json.dumps(merged_meta, ensure_ascii=False) if merged_meta else None,
                    current_step if current_step is not None else row[2],
                    1 if (cancelled if cancelled is not None else row[3]) else 0,
                    time.time(),
                    task_id,
                ),
            )
            con.commit()

        dbw = TaskManager._dbw()
        await dbw.run(DB_PATH, _update)

    @staticmethod
    async def update_step(task_id: int, step_index: int, *, status: TaskStatus,
                          output_payload: Optional[Dict[str, Any]] = None,
                          error: Optional[str] = None,
                          retries: Optional[int] = None) -> None:
        def _update(con):
            row = con.execute(
                """
                SELECT status, output_payload, error, retries
                FROM agent_task_steps WHERE task_id=? AND step_index=?
                """,
                (task_id, step_index),
            ).fetchone()
            if not row:
                return
            merged_output = json.loads(row[1]) if row[1] else {}
            if output_payload:
                merged_output.update(output_payload)
            con.execute(
                """
                UPDATE agent_task_steps
                SET status=?, output_payload=?, error=?, retries=?, updated_ts=?
                WHERE task_id=? AND step_index=?
                """,
                (
                    status.value,
                    json.dumps(merged_output, ensure_ascii=False) if merged_output else None,
                    error,
                    retries if retries is not None else row[3],
                    time.time(),
                    task_id,
                    step_index,
                ),
            )
            con.commit()

        dbw = TaskManager._dbw()
        await dbw.run(DB_PATH, _update)

    @staticmethod
    async def cancel_task(task_id: int, reason: Optional[str] = None) -> None:
        await TaskManager.update_task_status(
            task_id,
            status=TaskStatus.CANCELLED,
            cancelled=True,
            metadata={"cancel_reason": reason} if reason else None,
        )

    @staticmethod
    async def fetch_steps(task_id: int) -> List[TaskStepRecord]:
        def _fetch(con):
            rows = con.execute(
                """
                SELECT step_index, step_type, objective, status, input_payload,
                       output_payload, error, retries, max_retries, created_ts, updated_ts
                FROM agent_task_steps WHERE task_id=? ORDER BY step_index ASC
                """,
                (task_id,),
            ).fetchall()
            records: List[TaskStepRecord] = []
            for row in rows:
                records.append(
                    TaskStepRecord(
                        task_id=task_id,
                        step_index=row[0],
                        step_type=StepType(row[1]),
                        objective=row[2],
                        status=TaskStatus(row[3]),
                        input_payload=json.loads(row[4]) if row[4] else None,
                        output_payload=json.loads(row[5]) if row[5] else None,
                        error=row[6],
                        retries=row[7],
                        max_retries=row[8],
                        created_ts=row[9],
                        updated_ts=row[10],
                    )
                )
            return records

        dbw = TaskManager._dbw()
        return await dbw.run(DB_PATH, _fetch)

    @staticmethod
    async def is_cancelled(task_id: int) -> bool:
        task = await TaskManager.load_task(task_id)
        return bool(task.cancelled) if task else True
