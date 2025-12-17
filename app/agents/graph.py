#graph.py
"""
LangGraph orchestration for the AutoML Agent with deterministic tool routing.

Supports:
- Preprocessing
- Training (classification / regression)
- Tuning
- Forecasting (Prophet ONLY for now)
"""

from __future__ import annotations

from typing import TypedDict, Optional, Literal, Dict, Any, List

from app.agents import nodes  # fallback runner
from app.agents.tools import (
    preprocess_data_tool,
    choose_task_type_tool,
    train_baselines_tool,
    tune_best_model_optuna_tool,
    tune_best_model_random_search_tool,
    forecast_with_prophet_tool,  # ✅ NEW
)


# ==============================
# Shared State Schema
# ==============================
class AutoMLState(TypedDict, total=False):
    # Data
    raw_df: Any
    clean_df: Any
    pre_df: Any

    # Training
    target_col: Optional[str]
    task_type: Optional[Literal["classification", "regression"]]

    # Forecasting (Prophet)
    ds_col: Optional[str]
    y_col: Optional[str]
    forecast_periods: int
    forecast_freq: Optional[str]
    forecast_result: Dict[str, Any]
    forecast_preview: Any

    # Preprocess config
    drop_cols: List[str]
    duplicate_strategy: str
    missing_strategy: Optional[Dict[str, str]]
    column_mapping: Optional[Dict[str, str]]
    type_overrides: Optional[Dict[str, str]]
    preserve_column_names: bool

    # Results
    pre_preview: Any
    pre_col_types: Dict[str, str]
    pre_type_params: Dict[str, Any]
    pre_stats: Dict[str, Any]

    train_result: Dict[str, Any]
    best_model_name: Optional[str]
    best_model_row: Dict[str, Any]
    tuned_result: Dict[str, Any]

    # Routing flags
    want_preprocess: bool
    want_train: bool
    want_tune: bool
    want_forecast: bool

    # Tuning
    chosen_tune_method: Optional[Literal["bayesian", "random_search"]]
    tune_metric: Optional[str]

    # HITL / Supervisor
    require_approval: bool
    approved: bool
    supervisor_reason: str
    history: List[Dict[str, Any]]
    errors: List[str]


# ==============================
# Policies
# ==============================
REQUIRE_APPROVAL_FOR_PREPROCESS = False
REQUIRE_APPROVAL_FOR_TRAIN = True
REQUIRE_APPROVAL_FOR_TUNE = True
REQUIRE_APPROVAL_FOR_FORECAST = False  # Prophet is safe


# ==============================
# LangGraph App
# ==============================
def _build_langgraph_app():
    from langgraph.graph import StateGraph, END

    TOOL_REGISTRY = {
        "preprocess_data": preprocess_data_tool,
        "choose_task_type": choose_task_type_tool,
        "train_baselines": train_baselines_tool,
        "tune_best_model_optuna": tune_best_model_optuna_tool,
        "tune_best_model_random_search": tune_best_model_random_search_tool,
        "forecast_with_prophet": forecast_with_prophet_tool,  # ✅ NEW
    }

    def _ensure_defaults(s: Dict[str, Any]) -> None:
        s.setdefault("history", [])
        s.setdefault("errors", [])
        s.setdefault("require_approval", False)
        s.setdefault("approved", False)
        s.setdefault("supervisor_reason", "")

    def _audit(s: Dict[str, Any], action: str) -> None:
        s["history"].append({"step": "graph", "action": action})

    def _invoke(tool_name: str, s: Dict[str, Any]) -> Dict[str, Any]:
        tool = TOOL_REGISTRY[tool_name]
        out = tool.invoke({"state": s})
        _ensure_defaults(out)
        out["supervisor_reason"] = f"Ran tool: {tool_name}"
        _audit(out, tool_name)
        return out

    def _df_for_ops(s: Dict[str, Any]):
        return s.get("pre_df") if s.get("pre_df") is not None else s.get("clean_df")

    def _needs_preprocess(s: Dict[str, Any]) -> bool:
        return s.get("clean_df") is not None and s.get("pre_df") is None

    def _approval_gate(s: Dict[str, Any], reason: str):
        s["require_approval"] = True
        s["approved"] = False
        s["supervisor_reason"] = reason
        _audit(s, "await_approval")
        return s

    def node_supervisor(state: AutoMLState) -> AutoMLState:
        s = dict(state)
        _ensure_defaults(s)

        # -------------------------
        # No data
        # -------------------------
        if s.get("clean_df") is None and s.get("pre_df") is None:
            s["supervisor_reason"] = "Waiting for dataset upload."
            return s  # type: ignore

        # -------------------------
        # Preprocess
        # -------------------------
        if s.get("want_preprocess"):
            if _needs_preprocess(s):
                if REQUIRE_APPROVAL_FOR_PREPROCESS and not s.get("approved"):
                    return _approval_gate(s, "Approve preprocessing")  # type: ignore
                out = _invoke("preprocess_data", s)
                out["want_preprocess"] = False
                return out  # type: ignore
            s["want_preprocess"] = False
            return s  # type: ignore

        # -------------------------
        # Forecast (Prophet)
        # -------------------------
        if s.get("want_forecast"):

            if not s.get("ds_col") or not s.get("y_col"):
                s["want_forecast"] = False
                return s 

            if REQUIRE_APPROVAL_FOR_FORECAST and not s.get("approved"):
                return _approval_gate(s, "Approve forecasting")  # type: ignore

            out = _invoke("forecast_with_prophet", s)
            out["want_forecast"] = False
            return out  # type: ignore

        # -------------------------
        # Train
        # -------------------------
        if s.get("want_train"):
            if not s.get("target_col"):
                s["errors"].append("train_blocked: target missing")
                return s  # type: ignore

            if _needs_preprocess(s):
                s["errors"].append("train_blocked: preprocessing required")
                return s  # type: ignore

            if REQUIRE_APPROVAL_FOR_TRAIN and not s.get("approved"):
                return _approval_gate(s, "Approve training")  # type: ignore

            if not s.get("task_type"):
                s = _invoke("choose_task_type", s)

            out = _invoke("train_baselines", s)
            out["want_train"] = False
            return out  # type: ignore

        # -------------------------
        # Tune
        # -------------------------
        if s.get("want_tune"):
            if not s.get("train_result"):
                s["errors"].append("tune_blocked: train first")
                return s  # type: ignore

            if REQUIRE_APPROVAL_FOR_TUNE and not s.get("approved"):
                return _approval_gate(s, "Approve tuning")  # type: ignore

            method = s.get("chosen_tune_method") or "bayesian"
            tool = "tune_best_model_optuna" if method == "bayesian" else "tune_best_model_random_search"
            out = _invoke(tool, s)
            out["want_tune"] = False
            return out  # type: ignore

        # -------------------------
        # Idle
        # -------------------------
        s["supervisor_reason"] = "Idle"
        return s  # type: ignore

    g = StateGraph(AutoMLState)
    g.add_node("supervisor", node_supervisor)
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


# ==============================
# Public factory
# ==============================
def build_automl_graph():
    try:
        return _build_langgraph_app()
    except Exception:
        from app.agents.graph import _SimpleApp
        return _SimpleApp()
