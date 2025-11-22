# app/agents/runner.py
from __future__ import annotations
from typing import Dict, Any
from app.agents.graph import build_automl_graph

_app = build_automl_graph()

def _snapshot_key(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a safe, comparable snapshot that does NOT include DataFrames
    (or any pandas objects). Only scalars/booleans/lengths.
    """
    plan_steps = s.get("plan_steps") or []
    return {
        "has_clean_df": s.get("clean_df") is not None,
        "has_pre_df": s.get("pre_df") is not None,
        "has_train_result": bool(s.get("train_result")),
        "has_tuned_result": bool(s.get("tuned_result")),
        "want_preprocess": bool(s.get("want_preprocess")),
        "want_train": bool(s.get("want_train")),
        "want_tune": bool(s.get("want_tune")),
        "require_approval": bool(s.get("require_approval")),
        "approved": bool(s.get("approved")),
        "plan_len": len(plan_steps),
        "plan_index": int(s.get("plan_index") or 0),
        "plan_done": bool(s.get("plan_done")),
        "errors_len": len(s.get("errors") or []),
        "history_len": len(s.get("history") or []),
        "target_col": s.get("target_col"),
        "task_type": s.get("task_type"),
    }

def run_automl_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the AutoML graph/safe runner with given state.
    Loops to complete deterministic workflows (plan_steps or explicit flags).
    """
    base = dict(state)
    cfg = {
        "configurable": {
            "thread_id": base.get("thread_id", "chat-thread"),
            "checkpoint_ns": "automl",
        }
    }

    current = dict(base)
    last_snap = None

    # up to 8 sequential invocations to finish plans/flags
    for _ in range(8):
        try:
            out = _app.invoke(current, config=cfg)
        except TypeError:
            out = _app.invoke(current)

        if not isinstance(out, dict):
            break

        merged = dict(current)
        merged.update(out)

        snap = _snapshot_key(merged)
        if last_snap is not None and snap == last_snap:
            current = merged
            break

        last_snap = snap
        current = merged

        # stop if nothing left to do
        if not (
            current.get("plan_steps")
            and not current.get("plan_done")
        ) and not any(current.get(k) for k in ["want_preprocess", "want_train", "want_tune"]):
            break

    # Safety: never drop messages
    if "messages" not in current and "messages" in base:
        current["messages"] = base["messages"]

    # Debug prints
    print("\n=== Supervisor Reason ===")
    print(current.get("supervisor_reason"))
    print("=== Errors ===")
    print(current.get("errors"))
    print("=== History ===")
    for h in current.get("history", []):
        print(h)
    print("=== END ===\n")

    return current
