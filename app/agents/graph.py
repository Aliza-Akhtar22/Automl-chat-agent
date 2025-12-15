# app/agents/graph.py
"""
LangGraph orchestration for the AutoML Agent with deterministic tool routing.

Production contract:
- Chat/UI sets flags (want_preprocess, want_train, want_tune) + inputs (target_col, tune_metric, chosen_tune_method).
- Planner + ChatOrchestrator handle: PLAN → APPROVAL.
- Graph executes: one safe tool per invocation, updates state, clears flags, appends audit trail.

Key rules:
- No silent execution of expensive steps: training + tuning require approval unless configured otherwise.
- Preprocessing is allowed when explicitly requested (often safe), but can also be gated if desired.
- No LLM supervisor auto-pilot by default. Deterministic routing is easier to test, audit, and ship.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Literal, Dict, Any, List

from app.agents import nodes  # used only by fallback runner
from app.agents.tools import (
    preprocess_data_tool,
    choose_task_type_tool,
    train_baselines_tool,
    tune_best_model_optuna_tool,
    tune_best_model_random_search_tool,
)


# ==============================
# Shared State Schema
# ==============================
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

    # Routing flags (set by chat/UI)
    want_preprocess: bool
    want_train: bool
    want_tune: bool

    # Tuning options
    chosen_tune_method: Optional[Literal["bayesian", "random_search"]]
    tune_metric: Optional[str]  # "f1", "accuracy", "r2", "rmse", ...

    # Supervisor / HITL
    require_approval: bool
    approved: bool
    supervisor_reason: str
    history: List[Dict[str, Any]]
    errors: List[str]


# ==============================
# Production Policies
# ==============================
# In many products, preprocessing is considered “safe” and may be allowed without explicit approval.
# If your manager wants strict HITL for preprocessing too, set this to True.
REQUIRE_APPROVAL_FOR_PREPROCESS = False

# Training and tuning are usually “expensive” and should be HITL-gated.
REQUIRE_APPROVAL_FOR_TRAIN = True
REQUIRE_APPROVAL_FOR_TUNE = True


# ==============================
# LangGraph App (preferred)
# ==============================
def _build_langgraph_app():
    from langgraph.graph import StateGraph, END

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
        s.setdefault("supervisor_reason", "")

    def _audit(s: Dict[str, Any], action: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = {"step": "graph", "action": action}
        if extra:
            rec.update(extra)
        s.setdefault("history", []).append(rec)

    def _invoke(tool_name: str, s: Dict[str, Any]) -> Dict[str, Any]:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            _ensure_defaults(s)
            s["errors"].append(f"tool_not_found: {tool_name}")
            s["supervisor_reason"] = f"Tool missing: {tool_name}"
            _audit(s, "missing_tool", {"tool": tool_name})
            return s

        try:
            new_state = tool.invoke({"state": s})
        except Exception as e:
            _ensure_defaults(s)
            s["errors"].append(f"tool_invoke_error({tool_name}): {e}")
            s["supervisor_reason"] = f"Error while running {tool_name}."
            _audit(s, "tool_error", {"tool": tool_name})
            return s

        if not isinstance(new_state, dict):
            raise ValueError(f"{tool_name} did not return a dict-like state.")

        _ensure_defaults(new_state)
        new_state["supervisor_reason"] = f"Ran tool: {tool_name}"
        _audit(new_state, "ran_tool", {"tool": tool_name})
        return new_state

    def _df_for_ops(s: Dict[str, Any]) -> Any:
        # Prefer pre_df once available, else clean_df
        return s.get("pre_df") if s.get("pre_df") is not None else s.get("clean_df")

    def _needs_preprocess(s: Dict[str, Any]) -> bool:
        return (s.get("clean_df") is not None) and (s.get("pre_df") is None)

    def _approval_gate(s: Dict[str, Any], reason: str) -> Dict[str, Any]:
        _ensure_defaults(s)
        s["require_approval"] = True
        s["approved"] = False
        s["supervisor_reason"] = reason
        _audit(s, "await_approval")
        return s

    def node_supervisor(state: AutoMLState) -> AutoMLState:
        """
        Deterministic supervisor:
        - Executes at most ONE tool per call
        - Enforces prerequisites & HITL approval
        """
        s: Dict[str, Any] = dict(state)
        _ensure_defaults(s)

        # ----------------------------------------
        # 0) No data -> idle
        # ----------------------------------------
        if s.get("clean_df") is None and s.get("pre_df") is None:
            s["supervisor_reason"] = "Waiting for dataset upload."
            _audit(s, "idle_no_data")
            return s  # type: ignore[return-value]

        # ----------------------------------------
        # 1) PREPROCESS (optional approval)
        # ----------------------------------------
        if s.get("want_preprocess"):
            if _needs_preprocess(s):
                if REQUIRE_APPROVAL_FOR_PREPROCESS and not s.get("approved", False):
                    return _approval_gate(s, "About to preprocess the data. Please approve.")  # type: ignore[return-value]

                out = _invoke("preprocess_data", s)
                out["want_preprocess"] = False
                out["require_approval"] = False
                out["approved"] = False
                return out  # type: ignore[return-value]

            # If already preprocessed, just clear the flag
            s["want_preprocess"] = False
            s["supervisor_reason"] = "Preprocessing already completed (pre_df exists)."
            _audit(s, "preprocess_skipped_already_done")
            return s  # type: ignore[return-value]

        # ----------------------------------------
        # 2) TRAIN (requires: target_col, approval)
        # ----------------------------------------
        if s.get("want_train"):
            df_for_ops = _df_for_ops(s)
            target_col = s.get("target_col")

            if df_for_ops is None:
                s["supervisor_reason"] = "No dataset available to train on."
                s["errors"].append("train_blocked: no dataframe")
                _audit(s, "train_blocked_no_df")
                return s  # type: ignore[return-value]

            if not target_col:
                s["supervisor_reason"] = "Training requires a target column. Please provide target_col first."
                s["errors"].append("train_blocked: missing target_col")
                _audit(s, "train_blocked_no_target")
                return s  # type: ignore[return-value]

            # If preprocessing is not done, do not silently do it here.
            # Planner should have included preprocess; Orchestrator should set want_preprocess first.
            if _needs_preprocess(s):
                s["supervisor_reason"] = "Training requires preprocessing first. Set want_preprocess and run again."
                s["errors"].append("train_blocked: preprocessing not done")
                _audit(s, "train_blocked_need_preprocess")
                return s  # type: ignore[return-value]

            if REQUIRE_APPROVAL_FOR_TRAIN and not s.get("approved", False):
                return _approval_gate(
                    s,
                    "About to train baseline models. This can take some time and compute. Please approve.",
                )  # type: ignore[return-value]

            # Ensure task type exists (optional)
            if s.get("task_type") is None:
                # choose_task_type tool expects state with target_col and df available
                out = _invoke("choose_task_type", s)
                # If task type still missing, continue anyway; train tool may infer too.
                s = out

            out = _invoke("train_baselines", s)
            out["want_train"] = False
            out["require_approval"] = False
            out["approved"] = False
            return out  # type: ignore[return-value]

        # ----------------------------------------
        # 3) TUNE (requires: train_result, approval, method)
        # ----------------------------------------
        if s.get("want_tune"):
            if not s.get("train_result"):
                s["supervisor_reason"] = "Tuning requires training results. Train baselines first."
                s["errors"].append("tune_blocked: no train_result")
                _audit(s, "tune_blocked_no_train")
                return s  # type: ignore[return-value]

            # tune_metric is typically selected in chat; allow a safe default if absent
            if not s.get("tune_metric"):
                task = s.get("task_type") or "classification"
                s["tune_metric"] = "f1" if task == "classification" else "r2"
                _audit(s, "tune_metric_defaulted", {"metric": s["tune_metric"]})

            if REQUIRE_APPROVAL_FOR_TUNE and not s.get("approved", False):
                return _approval_gate(
                    s,
                    "About to run hyperparameter tuning. This can be compute-intensive. Please approve.",
                )  # type: ignore[return-value]

            method = (s.get("chosen_tune_method") or "bayesian").lower()
            tool_name = "tune_best_model_optuna" if method == "bayesian" else "tune_best_model_random_search"

            out = _invoke(tool_name, s)
            out["want_tune"] = False
            out["require_approval"] = False
            out["approved"] = False
            return out  # type: ignore[return-value]

        # ----------------------------------------
        # 4) Nothing to do (idle)
        # ----------------------------------------
        s["require_approval"] = False
        s["approved"] = False
        s["supervisor_reason"] = "No pending actions. Waiting for user request."
        _audit(s, "idle_no_flags")
        return s  # type: ignore[return-value]

    g = StateGraph(AutoMLState)
    g.add_node("supervisor", node_supervisor)
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


# ==============================
# Fallback runner (no LangGraph)
# ==============================
class _SimpleApp:
    """
    Deterministic fallback with the same semantics:
    - one step per invoke
    - approval gating
    - prerequisites enforced
    """

    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        s = dict(state)
        s.setdefault("history", [])
        s.setdefault("errors", [])
        s.setdefault("require_approval", False)
        s.setdefault("approved", False)
        s.setdefault("supervisor_reason", "")

        def audit(action: str, extra: Optional[Dict[str, Any]] = None) -> None:
            rec = {"step": "fallback", "action": action}
            if extra:
                rec.update(extra)
            s["history"].append(rec)

        def df_for_ops() -> Any:
            return s.get("pre_df") if s.get("pre_df") is not None else s.get("clean_df")

        def needs_preprocess() -> bool:
            return (s.get("clean_df") is not None) and (s.get("pre_df") is None)

        # 0) No data
        if s.get("clean_df") is None and s.get("pre_df") is None:
            s["supervisor_reason"] = "Waiting for dataset upload."
            audit("idle_no_data")
            return s

        # 1) Preprocess
        if s.get("want_preprocess"):
            if needs_preprocess():
                if REQUIRE_APPROVAL_FOR_PREPROCESS and not s.get("approved", False):
                    s["require_approval"] = True
                    s["approved"] = False
                    s["supervisor_reason"] = "About to preprocess the data. Please approve."
                    audit("await_approval", {"for": "preprocess"})
                    return s

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
                    s["want_preprocess"] = False
                    s["require_approval"] = False
                    s["approved"] = False
                    s["supervisor_reason"] = "Preprocessing complete."
                    audit("ran_preprocess")
                    return s
                except Exception as e:
                    s["errors"].append(f"preprocess_error: {e}")
                    s["supervisor_reason"] = "Error while preprocessing."
                    audit("preprocess_error")
                    return s

            s["want_preprocess"] = False
            s["supervisor_reason"] = "Preprocessing already completed (pre_df exists)."
            audit("preprocess_skipped_already_done")
            return s

        # 2) Train
        if s.get("want_train"):
            df = df_for_ops()
            if df is None:
                s["errors"].append("train_blocked: no dataframe")
                s["supervisor_reason"] = "No dataset available to train on."
                audit("train_blocked_no_df")
                return s

            if not s.get("target_col"):
                s["errors"].append("train_blocked: missing target_col")
                s["supervisor_reason"] = "Training requires a target column."
                audit("train_blocked_no_target")
                return s

            if needs_preprocess():
                s["errors"].append("train_blocked: preprocessing not done")
                s["supervisor_reason"] = "Training requires preprocessing first."
                audit("train_blocked_need_preprocess")
                return s

            if REQUIRE_APPROVAL_FOR_TRAIN and not s.get("approved", False):
                s["require_approval"] = True
                s["approved"] = False
                s["supervisor_reason"] = "About to train baseline models. Please approve."
                audit("await_approval", {"for": "train"})
                return s

            try:
                y = df[s["target_col"]]
                X = df.drop(columns=[s["target_col"]])
                task = s.get("task_type") or nodes.choose_task_type(y)
                s["task_type"] = task
                res = nodes.baseline_training_tool(X, y, task)
                s["train_result"] = res
                s["want_train"] = False
                s["require_approval"] = False
                s["approved"] = False
                s["supervisor_reason"] = "Training complete."
                audit("ran_train")
                return s
            except Exception as e:
                s["errors"].append(f"train_error: {e}")
                s["supervisor_reason"] = "Error while training."
                audit("train_error")
                return s

        # 3) Tune (fallback does not implement full tuning tools here; keep LangGraph path preferred)
        if s.get("want_tune"):
            if not s.get("train_result"):
                s["errors"].append("tune_blocked: no train_result")
                s["supervisor_reason"] = "Tuning requires training first."
                audit("tune_blocked_no_train")
                return s

            if REQUIRE_APPROVAL_FOR_TUNE and not s.get("approved", False):
                s["require_approval"] = True
                s["approved"] = False
                s["supervisor_reason"] = "About to run tuning. Please approve."
                audit("await_approval", {"for": "tune"})
                return s

            s["errors"].append("tune_not_supported_in_fallback: use LangGraph tools path")
            s["supervisor_reason"] = "Tuning requires LangGraph tools path."
            audit("tune_blocked_fallback")
            return s

        # 4) Idle
        s["require_approval"] = False
        s["approved"] = False
        s["supervisor_reason"] = "No pending actions. Waiting for user request."
        audit("idle_no_flags")
        return s


# ==============================
# Public factory
# ==============================
def build_automl_graph():
    try:
        return _build_langgraph_app()
    except Exception:
        return _SimpleApp()
