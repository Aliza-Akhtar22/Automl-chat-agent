# app/agents/graph.py
from typing import TypedDict, Optional, Literal, Dict, Any, List

from app.agents import nodes
from app.agents.tools import (
    preprocess_data_tool,
    choose_task_type_tool,
    train_baselines_tool,
    tune_best_model_optuna_tool,
    tune_best_model_random_search_tool,
)
from app.agents.llm_utils import chat_json


class AutoMLState(TypedDict, total=False):
    # Data artifacts
    raw_df: Any
    clean_df: Any
    pre_df: Any
    target_col: Optional[str]
    task_type: Optional[Literal["classification", "regression"]]

    # Preprocess config
    drop_cols: List[str]
    duplicate_strategy: str
    missing_strategy: Optional[Dict[str, str]]
    column_mapping: Optional[Dict[str, str]]
    type_overrides: Optional[Dict[str, str]]
    preserve_column_names: bool

    # Preprocess results
    pre_preview: Any
    pre_col_types: Dict[str, str]
    pre_type_params: Dict[str, Any]
    pre_stats: Dict[str, Any]

    # Train/Tune results
    train_result: Dict[str, Any]
    best_model_name: Optional[str]
    best_model_row: Dict[str, Any]
    tuned_result: Dict[str, Any]

    # Routing flags
    want_preprocess: bool
    want_train: bool
    want_tune: bool

    chosen_tune_method: Optional[Literal["bayesian", "random_search"]]
    tune_metric: Optional[str]

    # NEW: plan execution
    plan_steps: List[str]          # ordered tool names
    plan_index: int               # next step index
    plan_done: bool

    # Supervisor / HITL
    require_approval: bool
    approved: bool
    supervisor_reason: str
    history: List[Dict[str, Any]]
    errors: List[str]


def _build_langgraph_app():
    from langgraph.graph import StateGraph, END
    import json

    TOOL_REGISTRY = {
        "preprocess_data": preprocess_data_tool,
        "choose_task_type": choose_task_type_tool,
        "train_baselines": train_baselines_tool,
        "tune_best_model_optuna": tune_best_model_optuna_tool,
        "tune_best_model_random_search": tune_best_model_random_search_tool,
    }

    def _ensure_defaults(s: Dict[str, Any]) -> None:
        s.setdefault("history", [])
        s.setdefault("errors", [])
        s.setdefault("require_approval", False)
        s.setdefault("approved", False)
        s.setdefault("plan_steps", [])
        s.setdefault("plan_index", 0)
        s.setdefault("plan_done", False)

    def _invoke(tool_name: str, s: Dict[str, Any]) -> Dict[str, Any]:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            s["errors"].append(f"{tool_name} tool not found in registry.")
            s["supervisor_reason"] = f"{tool_name} tool missing."
            s["history"].append({"step": "supervisor", "action": "missing_tool", "tool": tool_name})
            return s
        new_state = tool.invoke({"state": s})
        if not isinstance(new_state, dict):
            raise ValueError(f"{tool_name} did not return a dict-like state.")
        _ensure_defaults(new_state)
        new_state["supervisor_reason"] = f"Ran tool: {tool_name}"
        new_state["history"].append({"step": "supervisor", "action": "ran_tool", "tool": tool_name})
        return new_state

    def node_supervisor(state: AutoMLState) -> AutoMLState:
        s: Dict[str, Any] = dict(state)
        _ensure_defaults(s)

        # -----------------------------
        # A) Deterministic PLAN mode
        # -----------------------------
        steps: List[str] = s.get("plan_steps") or []
        idx: int = int(s.get("plan_index") or 0)
        if steps and idx < len(steps) and not s.get("plan_done"):
            next_tool = steps[idx]

            # Train/tune approval gates still apply, but planner/orchestrator
            # can set approved=True automatically.
            if next_tool == "train_baselines" and not s.get("train_result"):
                if not s.get("approved", False):
                    s["require_approval"] = True
                    s["supervisor_reason"] = "About to train baseline models. Please approve."
                    s["history"].append({"step": "supervisor", "action": "await_train_approval"})
                    return s

            if next_tool.startswith("tune_best_model") and s.get("train_result") and not s.get("tuned_result"):
                if not s.get("approved", False):
                    s["require_approval"] = True
                    s["supervisor_reason"] = "About to run hyperparameter tuning. Please approve."
                    s["history"].append({"step": "supervisor", "action": "await_tune_approval"})
                    return s

            try:
                new_state = _invoke(next_tool, s)
                new_state["plan_index"] = idx + 1
                new_state["approved"] = False
                new_state["require_approval"] = False
                if new_state["plan_index"] >= len(steps):
                    new_state["plan_done"] = True
                return new_state  # type: ignore
            except Exception as e:
                s["errors"].append(f"plan_step_error({next_tool}): {e}")
                s["supervisor_reason"] = f"Error while running planned step '{next_tool}'."
                return s

        # -----------------------------
        # B) Explicit preprocess trigger
        # -----------------------------
        if s.get("want_preprocess") and s.get("clean_df") is not None and s.get("pre_df") is None:
            try:
                new_state = _invoke("preprocess_data", s)
                new_state["want_preprocess"] = False
                return new_state  # type: ignore
            except Exception as e:
                s["errors"].append(f"preprocess_direct_error: {e}")
                s["supervisor_reason"] = "Error while running preprocess_data."
                return s

        # -----------------------------
        # C) HITL for train / tune (non-plan mode)
        # -----------------------------
        if s.get("want_train") and not s.get("train_result"):
            if not s.get("approved", False):
                s["require_approval"] = True
                s["supervisor_reason"] = "About to train baseline models. Please approve."
                s["history"].append({"step": "supervisor", "action": "await_train_approval"})
                return s

        if s.get("want_train") and s.get("approved", False) and not s.get("train_result"):
            try:
                new_state = _invoke("train_baselines", s)
                new_state["want_train"] = False
                new_state["require_approval"] = False
                new_state["approved"] = False
                return new_state
            except Exception as e:
                s["errors"].append(f"train_direct_error: {e}")
                s["supervisor_reason"] = "Error while running train_baselines directly."
                return s

        if s.get("want_tune") and s.get("train_result") and not s.get("tuned_result"):
            if not s.get("approved", False):
                s["require_approval"] = True
                s["supervisor_reason"] = "About to run hyperparameter tuning. Please approve."
                s["history"].append({"step": "supervisor", "action": "await_tune_approval"})
                return s

        if s.get("want_tune") and s.get("approved", False) and s.get("train_result") and not s.get("tuned_result"):
            method = (s.get("chosen_tune_method") or "bayesian").lower()
            tool_name = "tune_best_model_optuna" if method == "bayesian" else "tune_best_model_random_search"
            try:
                new_state = _invoke(tool_name, s)
                new_state["want_tune"] = False
                new_state["require_approval"] = False
                new_state["approved"] = False
                return new_state
            except Exception as e:
                s["errors"].append(f"tune_direct_error: {e}")
                s["supervisor_reason"] = f"Error while running {tool_name} directly."
                return s

        s["require_approval"] = False
        s["approved"] = False

        # -----------------------------
        # D) If no data, idle
        # -----------------------------
        if s.get("clean_df") is None and s.get("pre_df") is None:
            s["supervisor_reason"] = "Waiting for dataset upload / basic cleaning."
            s["history"].append({"step": "supervisor", "action": "idle_no_data"})
            return s

        # -----------------------------
        # E) LLM auto-pilot chooser
        # -----------------------------
        summary = {
            "has_clean_df": s.get("clean_df") is not None,
            "has_pre_df": s.get("pre_df") is not None,
            "has_train_result": s.get("train_result") is not None,
            "has_tuned_result": s.get("tuned_result") is not None,
            "target_col": s.get("target_col"),
            "task_type": s.get("task_type"),
            "want_preprocess": s.get("want_preprocess", False),
            "want_train": s.get("want_train", False),
            "want_tune": s.get("want_tune", False),
            "chosen_tune_method": s.get("chosen_tune_method"),
            "tune_metric": s.get("tune_metric"),
        }

        system = (
            "You are the AutoML Supervisor.\n"
            "Decide the next step by picking ONE tool.\n"
            "Return JSON {\"tool\": \"name\"} or {\"tool\": null}.\n"
            "Valid tools: preprocess_data, choose_task_type, train_baselines, "
            "tune_best_model_optuna, tune_best_model_random_search."
        )
        user = f"Current state summary:\n{json.dumps(summary)}"

        try:
            decision = chat_json(system=system, user=user, model="gpt-4o-mini", temperature=0.0)
            tool_name = decision.get("tool")
        except Exception as e:
            s["errors"].append(f"supervisor_llm_error: {e}")
            s["supervisor_reason"] = "LLM decision failed; no automatic action taken."
            return s

        if not tool_name:
            s["supervisor_reason"] = "No further automatic step selected."
            s["history"].append({"step": "supervisor", "action": "no_tool"})
            return s

        try:
            new_state = _invoke(tool_name, s)
            return new_state  # type: ignore
        except Exception as e:
            s["errors"].append(f"supervisor_tool_error({tool_name}): {e}")
            s["supervisor_reason"] = f"Error while running tool '{tool_name}'."
            return s

    g = StateGraph(AutoMLState)
    g.add_node("supervisor", node_supervisor)
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


class _SimpleApp:
    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        s = dict(state)
        s.setdefault("history", [])
        s.setdefault("errors", [])
        s.setdefault("plan_steps", [])
        s.setdefault("plan_index", 0)
        s.setdefault("plan_done", False)

        # PLAN mode (fallback): run 1 step per invoke
        steps = s.get("plan_steps") or []
        idx = int(s.get("plan_index") or 0)
        if steps and idx < len(steps) and not s.get("plan_done"):
            step = steps[idx]
            # map planned step to existing flags
            if step == "preprocess_data":
                s["want_preprocess"] = True
            elif step == "train_baselines":
                s["want_train"] = True
                s["approved"] = True
            elif step.startswith("tune_best_model"):
                s["want_tune"] = True
                s["approved"] = True

        # Preprocess
        if s.get("want_preprocess") and "clean_df" in s and s.get("pre_df") is None:
            try:
                out = nodes.preprocess_tool(
                    df=s["clean_df"],
                    drop_cols=s.get("drop_cols", []),
                    duplicate_strategy=s.get("duplicate_strategy", "drop"),
                    missing_strategy=s.get("missing_strategy"),
                    column_mapping=s.get("column_mapping"),
                    type_overrides=s.get("type_overrides"),
                    preserve_column_names=s.get("preserve_column_names", False),
                )
                s["pre_df"] = out["df"]
                s["pre_preview"] = out["preview"]
                s["pre_col_types"] = out["col_types"]
                s["pre_type_params"] = out["type_params"]
                s["pre_stats"] = out["stats"]
                s["history"].append({"step": "preprocess", "status": "finished"})
                s["want_preprocess"] = False
            except Exception as e:
                s["errors"].append(f"preprocess_error: {e}")

        # Task type
        df_for_task = s.get("pre_df") if s.get("pre_df") is not None else s.get("clean_df")
        if df_for_task is not None and s.get("target_col") and not s.get("task_type"):
            try:
                y = df_for_task[s["target_col"]]
                s["task_type"] = nodes.choose_task_type(y)
            except Exception as e:
                s["errors"].append(f"task_type_error: {e}")

        # Train
        if s.get("want_train") and df_for_task is not None and s.get("target_col") and not s.get("train_result"):
            try:
                y = df_for_task[s["target_col"]]
                X = df_for_task.drop(columns=[s["target_col"]])
                task = s.get("task_type") or nodes.choose_task_type(y)
                s["task_type"] = task
                res = nodes.baseline_training_tool(X, y, task)
                s["train_result"] = res
                s["history"].append({"step": "train", "status": "finished"})
                s["want_train"] = False
            except Exception as e:
                s["errors"].append(f"train_error: {e}")

        # advance plan index if a planned flag just finished
        if steps and not s.get("want_preprocess") and not s.get("want_train") and not s.get("want_tune"):
            if idx < len(steps):
                s["plan_index"] = idx + 1
                if s["plan_index"] >= len(steps):
                    s["plan_done"] = True

        return s


def build_automl_graph():
    try:
        return _build_langgraph_app()
    except Exception:
        return _SimpleApp()
