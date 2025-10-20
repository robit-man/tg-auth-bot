#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous planning + execution entrypoint."""

import argparse
import json
import os
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "venv"
PYTHON_BIN = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PIP_BIN = VENV_DIR / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
CONFIG_PATH = SCRIPT_DIR / "config.json"

DEPS = [
    "ollama>=0.3.0",
    "jsonschema>=4.22.0",
    "selenium>=4.22.0",
    "webdriver-manager>=4.0.2",
    "beautifulsoup4>=4.12.3",
    "lxml>=5.2.2",
    "requests>=2.32.3",
    "pymupdf>=1.24.8",
    "python-docx>=1.1.2",
    "chardet>=5.2.0",
    "pyyaml>=6.0.2",
    "numpy>=1.26.0",
]


def ensure_venv_and_reexec():
    if os.environ.get("INSIDE_VENV") == "1":
        return
    if not VENV_DIR.exists():
        print(f"[BOOT] Creating venv at {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True, clear=False).create(str(VENV_DIR))
    print("[BOOT] Upgrading pip/setuptools/wheel ...")
    subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    print("[BOOT] Ensuring dependencies are installed ...")
    subprocess.check_call([str(PIP_BIN), "install", "--upgrade"] + DEPS)
    env = dict(os.environ)
    env["INSIDE_VENV"] = "1"
    cmd = [str(PYTHON_BIN), __file__] + sys.argv[1:]
    print("[BOOT] Re-executing inside venv ...")
    os.execvpe(str(PYTHON_BIN), cmd, env)


ensure_venv_and_reexec()

from cognition import CognitionConfig  # noqa: E402
from memory import MemoryStore  # noqa: E402
from context import KnowledgeStore  # noqa: E402
from policy import PolicyManager  # noqa: E402
from global_context import GlobalContextWorkspace  # noqa: E402
from tools import Tools  # noqa: E402
from agents import AutoOrchestrator, AutoConfig, WorkspaceAtlas, ToolFoundry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous planner executor")
    parser.add_argument("objective", nargs="?", help="Primary objective for the run")
    parser.add_argument("--constraints", help="Constraints to respect")
    parser.add_argument("--deliverables", help="Expected deliverables or outputs")
    parser.add_argument("--no-ui", action="store_true", help="Disable curses chat UI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = CognitionConfig.load()

    objective = args.objective
    if not objective:
        objective = input("Objective: ").strip()
    if not objective:
        print("No objective provided; exiting.")
        return

    runs_root = SCRIPT_DIR / "runs" / "auto"
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    agents_md = SCRIPT_DIR / "AGENTS.md"

    memory_store = MemoryStore(db_path=SCRIPT_DIR / "memory.db", embed_model=base_cfg.embed_model)
    knowledge_store = KnowledgeStore(db_path=SCRIPT_DIR / "knowledge.db", data_dir=SCRIPT_DIR / "data", embed_model=base_cfg.embed_model)
    policy_manager = PolicyManager(db_path=SCRIPT_DIR / "policy.db")
    global_workspace = GlobalContextWorkspace(config=base_cfg)

    config = AutoConfig(
        workspace_dir=run_dir,
        agents_md=agents_md,
        runs_root=runs_root,
        vision_model=os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b"),
        context_limit=6,
        auto_launch_ui=not args.no_ui,
    )

    workspace_atlas = WorkspaceAtlas(run_dir)
    tool_foundry = ToolFoundry(base_cfg, run_dir)

    orchestrator = AutoOrchestrator(
        config=config,
        memory_store=memory_store,
        knowledge_store=knowledge_store,
        policy_manager=policy_manager,
        global_workspace=global_workspace,
        tool_foundry=tool_foundry,
        workspace_atlas=workspace_atlas,
    )

    summary = orchestrator.run(
        objective=objective,
        constraints=args.constraints,
        deliverables=args.deliverables,
    )

    Tools.close_browser()

    print("\n=== Auto Run Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts stored in {run_dir}")


if __name__ == "__main__":
    main()
