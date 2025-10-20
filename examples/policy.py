#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy.py — Online learning for routing, tuning, and feedback logging.

Adds:
• Persistent feedback log per action (action_feedback)
• Helpers to compute recent averages and detect “heavy failure”
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import ollama       # type: ignore

from cognition import CognitionConfig

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = SCRIPT_DIR / "policy.db"

# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

KEY_FLAGS = [
    "search", "web", "pdf", "json", "rag", "doc", "medical", "system"
]

@dataclass
class FeatureBuilder:
    dim_embed: int = 64          # take first 64 embedding dims
    add_bias: bool = True

    def _embed(self, text: str) -> List[float]:
        cfg = CognitionConfig.load()
        model = cfg.embed_model or "nomic-embed-text"
        out = ollama.embeddings(model=model, prompt=text.strip())
        if "embedding" in out:
            vec = [float(x) for x in out["embedding"]]
        elif "embeddings" in out and out["embeddings"]:
            vec = [float(x) for x in out["embeddings"][0]]
        else:
            vec = []
        if len(vec) >= self.dim_embed:
            return vec[: self.dim_embed]
        return vec + [0.0] * (self.dim_embed - len(vec))

    def from_intent(self, intent: str) -> np.ndarray:
        e = self._embed(intent)
        flags = [1.0 if k in intent.lower() else 0.0 for k in KEY_FLAGS]
        features: List[float] = []
        if self.add_bias:
            features.append(1.0)
        features.extend(e)
        features.extend(flags)
        return np.array(features, dtype=np.float64)

    @property
    def dim(self) -> int:
        return (1 if self.add_bias else 0) + self.dim_embed + len(KEY_FLAGS)

# ─────────────────────────────────────────────────────────────────────────────
# LinUCB policy
# ─────────────────────────────────────────────────────────────────────────────

class LinUCBPolicy:
    def __init__(self, conn: sqlite3.Connection, d: int, alpha: float = 1.5):
        self.conn = conn
        self.d = d
        self.alpha = alpha
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn:
            self.conn.execute("""
              CREATE TABLE IF NOT EXISTS linucb_models (
                action TEXT PRIMARY KEY,
                d INTEGER NOT NULL,
                A TEXT NOT NULL,
                b TEXT NOT NULL,
                n INTEGER NOT NULL,
                alpha REAL NOT NULL,
                updated_ts INTEGER NOT NULL
              );
            """)

    def _now(self) -> int:
        return int(time.time())

    def _init_if_needed(self, action: str):
        row = self.conn.execute("SELECT action FROM linucb_models WHERE action=?", (action,)).fetchone()
        if row:
            return
        import numpy as np
        A = np.eye(self.d, dtype=np.float64)
        b = np.zeros(self.d, dtype=np.float64)
        with self.conn:
            self.conn.execute(
                "INSERT INTO linucb_models(action,d,A,b,n,alpha,updated_ts) VALUES (?,?,?,?,?,?,?)",
                (action, self.d, json.dumps(A.tolist()), json.dumps(b.tolist()), 0, float(self.alpha), self._now())
            )

    def _load(self, action: str):
        self._init_if_needed(action)
        r = self.conn.execute("SELECT d,A,b,n,alpha FROM linucb_models WHERE action=?", (action,)).fetchone()
        d = int(r["d"])
        A = np.array(json.loads(r["A"]), dtype=np.float64).reshape((d, d))
        b = np.array(json.loads(r["b"]), dtype=np.float64).reshape((d,))
        n = int(r["n"])
        alpha = float(r["alpha"])
        return A, b, n, alpha

    def _save(self, action: str, A, b, n: int, alpha: float):
        with self.conn:
            self.conn.execute(
                "UPDATE linucb_models SET A=?, b=?, n=?, alpha=?, updated_ts=? WHERE action=?",
                (json.dumps(A.tolist()), json.dumps(b.tolist()), n, float(alpha), int(time.time()), action)
            )

    def score(self, action: str, x) -> float:
        import numpy as np
        A, b, n, alpha = self._load(action)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        theta = A_inv @ b
        mean = float(x @ theta)
        conf = float((x @ A_inv @ x) ** 0.5)
        return mean + alpha * conf

    def select(self, actions: Sequence[str], x) -> Tuple[str, Dict[str, float]]:
        scores = {a: self.score(a, x) for a in actions}
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best, scores

    def update(self, action: str, x, reward: float):
        import numpy as np
        reward = max(0.0, min(1.0, reward))
        A, b, n, alpha = self._load(action)
        x = x.reshape((self.d, 1))
        A = A + (x @ x.T)
        b = b + (reward * x.flatten())
        n += 1
        self._save(action, A, b, n, alpha)

# ─────────────────────────────────────────────────────────────────────────────
# Epsilon-greedy parameter bandit
# ─────────────────────────────────────────────────────────────────────────────

class ParamBandit:
    def __init__(self, conn: sqlite3.Connection, epsilon: float = 0.1):
        self.conn = conn
        self.eps = epsilon
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn:
            self.conn.execute("""
              CREATE TABLE IF NOT EXISTS param_arms (
                bandit TEXT NOT NULL,
                arm TEXT NOT NULL,
                n INTEGER NOT NULL,
                sum_reward REAL NOT NULL,
                PRIMARY KEY (bandit, arm)
              );
            """)

    def _get(self, bandit: str, arm: str) -> Tuple[int, float]:
        r = self.conn.execute("SELECT n,sum_reward FROM param_arms WHERE bandit=? AND arm=?", (bandit, arm)).fetchone()
        if r:
            return int(r["n"]), float(r["sum_reward"])
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO param_arms(bandit,arm,n,sum_reward) VALUES (?,?,?,?)",
                              (bandit, arm, 0, 0.0))
        return 0, 0.0

    def select(self, bandit: str, arms: Sequence[str]) -> str:
        import random
        if random.random() < self.eps:
            return random.choice(list(arms))
        best_arm = None
        best_mean = -1.0
        for a in arms:
            n, s = self._get(bandit, a)
            mean = (s / n) if n > 0 else 0.5
            if mean > best_mean:
                best_mean = mean
                best_arm = a
        return best_arm or arms[0]

    def update(self, bandit: str, arm: str, reward: float):
        reward = max(0.0, min(1.0, reward))
        n, s = self._get(bandit, arm)
        with self.conn:
            self.conn.execute(
                "UPDATE param_arms SET n=?, sum_reward=? WHERE bandit=? AND arm=?",
                (n + 1, s + reward, bandit, arm)
            )

# ─────────────────────────────────────────────────────────────────────────────
# Feedback logging & failure detection
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackLogger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn:
            self.conn.execute("""
              CREATE TABLE IF NOT EXISTS action_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                ts INTEGER NOT NULL,
                reward REAL NOT NULL
              );
            """)

    def log(self, action: str, reward: float):
        with self.conn:
            self.conn.execute(
                "INSERT INTO action_feedback(action, ts, reward) VALUES (?, ?, ?)",
                (action, int(time.time()), float(reward))
            )

    def recent_avg(self, action: str, n: int = 10) -> Optional[float]:
        r = self.conn.execute(
            "SELECT AVG(reward) AS m FROM (SELECT reward FROM action_feedback WHERE action=? ORDER BY id DESC LIMIT ?)",
            (action, n)
        ).fetchone()
        if r and r["m"] is not None:
            return float(r["m"])
        return None

    def low_count(self, action: str, threshold: float = 0.25, n: int = 5) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) AS c FROM (SELECT reward FROM action_feedback WHERE action=? ORDER BY id DESC LIMIT ?) WHERE reward < ?",
            (action, n, threshold)
        ).fetchone()
        return int(r["c"] if r and r["c"] is not None else 0)

    def heavy_failure(self, action: str, *, n: int = 5, threshold: float = 0.25, min_count: int = 3) -> bool:
        """
        Heavy failure if >= min_count out of last n rewards are below threshold.
        """
        return self.low_count(action, threshold=threshold, n=n) >= min_count

# ─────────────────────────────────────────────────────────────────────────────
# PolicyManager — one-stop
# ─────────────────────────────────────────────────────────────────────────────

class PolicyManager:
    def __init__(self, db_path: Path | str = DB_PATH_DEFAULT, feature_dim: Optional[int] = None, alpha: float = 1.5, epsilon: float = 0.1):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute("PRAGMA busy_timeout=6000;")

        self.features = FeatureBuilder()
        d = feature_dim or self.features.dim
        self.policy = LinUCBPolicy(self.conn, d=d, alpha=alpha)
        self.params = ParamBandit(self.conn, epsilon=epsilon)
        self.feedback = FeedbackLogger(self.conn)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
