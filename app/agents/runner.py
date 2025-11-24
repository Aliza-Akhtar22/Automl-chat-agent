# app/agents/runner.py
from typing import Dict, Any
from app.agents.graph import build_automl_graph

_app = build_automl_graph()

def run_automl_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the AutoML graph/safe runner with given state.
    Always merges the graph output back into the original state
    so chat-specific keys (messages, stage, etc.) are preserved.
    """
    base = dict(state)
    cfg = {
        "configurable": {
            "thread_id": base.get("thread_id", "chat-thread"),
            "checkpoint_ns": "automl",
        }
    }
    try:
        out = _app.invoke(base, config=cfg)
    except TypeError:
        out = _app.invoke(base)

    print("\n=== Supervisor Reason ===")
    print(out.get("supervisor_reason"))
    print("=== Errors ===")
    print(out.get("errors"))
    print("=== History ===")
    for h in out.get("history", []):
        print(h)
    print("=== END ===\n")

    if not isinstance(out, dict):
        return base

    merged = dict(base)
    merged.update(out)

    # Safety: never drop messages
    if "messages" not in merged and "messages" in base:
        merged["messages"] = base["messages"]

    return merged