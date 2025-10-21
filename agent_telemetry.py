#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_telemetry.py — minimal telemetry logging utilities.
"""

from __future__ import annotations

import json
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional

LOG_PATH = Path("data/agent_telemetry.jsonl")
DB_PATH: Optional[str] = None
DBW = None
db_factory = None


def configure(db_path: str, dbw, db_callable):
    global DB_PATH, DBW, db_factory
    DB_PATH = db_path
    DBW = dbw
    db_factory = db_callable
    ensure_schema()


def ensure_schema():
    if not db_factory:
        return
    with closing(db_factory()) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_telemetry(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                payload TEXT,
                ts REAL
            )
            """
        )
        con.commit()


def telemetry_event(event_type: str, payload: Dict[str, Any]):
    ts = time.time()
    record = {"event_type": event_type, "payload": payload, "ts": ts}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not (DB_PATH and DBW):
        return

    async def _write():
        def _insert(con):
            con.execute(
                "INSERT INTO agent_telemetry(event_type, payload, ts) VALUES(?,?,?)",
                (event_type, json.dumps(payload, ensure_ascii=False), ts),
            )
            con.commit()

        await DBW.run(DB_PATH, _insert)

    try:
        import asyncio

        loop = asyncio.get_running_loop()
        loop.create_task(_write())
    except RuntimeError:
        # Not in event loop; skip DB write
        pass
