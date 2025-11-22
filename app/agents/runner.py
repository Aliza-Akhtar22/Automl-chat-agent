# app/agents/runner.py
from __future__ import annotations

from typing import Dict, Any, Optional
from app.agents.graph import build_automl_graph

_app = build_automl_graph()


def _fingerprint(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a lightweight, JSON-safe snapshot of state, excluding DataFrames
    and other un-comparable objects. This is used to detect convergence
    in the runner loop.
    """
    def has_df(x: Any) -> bool:
        # Avoid importing pandas here; just detect by common attributes
        return hasattr(x, "shape") and hasattr(x, "columns")

    fp: Dict[str, Any] = {
        # presence flags (not the objects)
        "has_clean_df": state.get("clean_df") is not None and has_df(state.get("clean_df")),
        "has_pre_df": state.get("pre_df") is not None and has_df(state.get("pre_df")),
        "has_train_result": bool(state.get("train_result")),
        "has_tuned_result": bool(state.get("tuned_result")),

        # routing flags
        "want_preprocess": bool(state.get("want_preprocess")),
        "want_train": bool(state.get("want_train")),
        "want_tune": bool(state.get("want_tune")),

        # key scalar fields
        "target_col": state.get("target_col"),
        "task_type": state.get("task_type"),
        "chosen_tune_method": state.get("chosen_tune_method"),
        "tune_metric": state.get("tune_metric"),

        # approval gates
        "require_approval": bool(state.get("require_approval")),
        "approved": bool(state.get("approved")),
    }

    # Optional: include shapes if you want extra safety
    try:
        if fp["has_clean_df"]:
            fp["clean_shape"] = tuple(state["clean_df"].shape)  # type: ignore
        if fp["has_pre_df"]:
            fp["pre_shape"] = tuple(state["pre_df"].shape)      # type: ignore
    except Exception:
        pass

    return fp


def run_automl_graph(state: Dict[str, Any], max_loops: int = 6) -> Dict[str, Any]:
    """
    Runs the AutoML graph/safe runner with given state.
    Re-invokes the graph until the high-level fingerprint stops changing
    or we hit max_loops. This allows sequential execution (preprocess -> train -> tune)
    within one user action without comparing DataFrames directly.
    """
    base = dict(state)
    cfg = {
        "configurable": {
            "thread_id": base.get("thread_id", "chat-thread"),
            "checkpoint_ns": "automl",
        }
    }

    last_fp: Optional[Dict[str, Any]] = None
    current = base

    for _ in range(max_loops):
        try:
            out = _app.invoke(current, config=cfg)
        except TypeError:
            out = _app.invoke(current)

        if not isinstance(out, dict):
            break

        merged = dict(current)
        merged.update(out)

        fp = _fingerprint(merged)
        if last_fp is not None and fp == last_fp:
            current = merged
            break

        last_fp = fp
        current = merged

        # Early stop if nothing left to do
        if not (current.get("want_preprocess") or current.get("want_train") or current.get("want_tune")):
            break

    # Safety: never drop messages
    if "messages" not in current and "messages" in base:
        current["messages"] = base["messages"]

    # Debug print (optional)
    print("\n=== Supervisor Reason ===")
    print(current.get("supervisor_reason"))
    print("=== Errors ===")
    print(current.get("errors"))
    print("=== History ===")
    for h in current.get("history", []):
        print(h)
    print("=== END ===\n")

    return current
