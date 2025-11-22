# app/agents/runner.py
from typing import Dict, Any
from copy import deepcopy

from app.agents.graph import build_automl_graph

_app = build_automl_graph()


def _invoke_app(base: Dict[str, Any]) -> Dict[str, Any]:
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

    if not isinstance(out, dict):
        return base
    return out


def run_automl_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the AutoML graph/safe runner with given state.

    NEW behavior:
    - If a planner workflow is active (plan_active=True), we keep invoking
      the graph to execute steps sequentially in the SAME call, until:
        * plan_done=True, OR
        * require_target_col=True (need user input), OR
        * require_approval=True (HITL gate in non-plan flow), OR
        * no further progress is made.

    Always merges the graph output back into the original state
    so chat-specific keys (messages, stage, etc.) are preserved.
    """
    base = dict(state)

    # Ensure planner defaults exist (graph also sets these, but runner is defensive)
    base.setdefault("plan_steps", [])
    base.setdefault("plan_index", 0)
    base.setdefault("plan_active", False)
    base.setdefault("plan_done", False)
    base.setdefault("require_target_col", False)

    # We will iteratively invoke to "drain" a plan or chained flags.
    max_iters = 12  # safety cap against infinite loops
    iters = 0

    last_state_snapshot = None
    out = base

    while iters < max_iters:
        iters += 1

        before = deepcopy(out)
        out_step = _invoke_app(out)

        # Merge step output into out
        merged_step = dict(out)
        merged_step.update(out_step)

        # Safety: never drop messages
        if "messages" not in merged_step and "messages" in out:
            merged_step["messages"] = out["messages"]

        out = merged_step

        # ----- Stop conditions -----
        # 1) Planner needs target
        if out.get("require_target_col"):
            break

        # 2) Non-plan HITL gate (rare in plan flow, but safe)
        if out.get("require_approval"):
            break

        # 3) Plan finished
        if out.get("plan_done") or (not out.get("plan_active") and out.get("plan_steps")):
            break

        # 4) No explicit wants + no plan active -> nothing else to do
        if not out.get("plan_active") and not any(
            out.get(k) for k in ("want_preprocess", "want_train", "want_tune", "want_plan")
        ):
            break

        # 5) No progress detected between iterations
        #    (compare a small snapshot of progress-relevant keys)
        snapshot_keys = [
            "pre_df", "train_result", "tuned_result",
            "plan_active", "plan_index", "plan_done",
            "want_preprocess", "want_train", "want_tune", "want_plan",
            "require_target_col", "require_approval",
            "errors", "history", "supervisor_reason",
        ]
        current_snapshot = {k: out.get(k) for k in snapshot_keys}

        if last_state_snapshot is not None and current_snapshot == last_state_snapshot:
            break
        last_state_snapshot = current_snapshot

        # 6) If tuned_result already produced, we can stop early
        if out.get("tuned_result"):
            break

    # --- Debug prints (kept from your original runner) ---
    print("\n=== Supervisor Reason ===")
    print(out.get("supervisor_reason"))
    print("=== Errors ===")
    print(out.get("errors"))
    print("=== History ===")
    for h in out.get("history", []):
        print(h)
    print("=== END ===\n")

    # Final merge into original base so chat keys persist
    merged = dict(base)
    merged.update(out)
    
    if "messages" not in merged and "messages" in base:
        merged["messages"] = base["messages"]

    return merged
