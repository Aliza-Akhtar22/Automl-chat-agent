# app/agents/tools.py

"""
LangChain-style tools wrapping your existing AutoML building blocks.

These tools operate on a shared `state: Dict[str, Any]` that matches
the fields you already use in `AutoMLState` (graph.py) and chat_app.

Idea:
- Supervisor (LLM / LangGraph node) inspects `state` + user request.
- Supervisor then chooses which tool(s) to call.
- Each tool reads from `state`, calls the core function, and returns
  an UPDATED copy of `state` (pure, side-effect free at the interface).

All tools:
- Are defensive (no hard crashes if pieces are missing).
- Append entries into `state["history"]`.
- Append error messages into `state["errors"]` when something goes wrong.
"""

from typing import Dict, Any
import pandas as pd
from langchain.tools import tool

from app.agents.nodes import (
    preprocess_tool as _preprocess_tool,
    baseline_training_tool as _baseline_training_tool,
    choose_task_type as _choose_task_type,
)
from app.core.tuning import (
    tune_with_optuna as _tune_with_optuna,
    tune_with_random_search as _tune_with_random_search,
)
from app.core.utils import (
    best_model_by_task,
    # NEW: metric helpers used to keep tuning simple for non-tech users
    default_metric,
    metric_direction,
    metric_to_sklearn_scorer,
    validate_metric_for_task,
)


# --------- helpers ---------


def _ensure_lists(state: Dict[str, Any]) -> None:
    """Guarantee history/errors exist."""
    state.setdefault("history", [])
    state.setdefault("errors", [])


def _copy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy to keep tools pure from orchestrator POV."""
    return dict(state)


def _get_df_for_model(state: Dict[str, Any]) -> pd.DataFrame:
    """
    Prefer pre_df if present; otherwise fall back to clean_df.
    Avoids using `or` on DataFrames (which is ambiguous).
    """
    if state.get("pre_df") is not None:
        return state["pre_df"]
    return state.get("clean_df")


# --------- TOOLS ---------


@tool("preprocess_data", return_direct=False)
def preprocess_data_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run preprocessing on `clean_df` using config in state:
    - drop_cols
    - duplicate_strategy
    - missing_strategy
    - column_mapping
    - type_overrides
    - preserve_column_names

    Populates:
    - pre_df, pre_preview, pre_col_types, pre_type_params, pre_stats
    """
    st = _copy(state)
    _ensure_lists(st)

    df = st.get("clean_df")
    if df is None:
        st["errors"].append("preprocess_data: 'clean_df' is missing.")
        st["history"].append({"tool": "preprocess_data", "status": "skipped"})
        return st

    try:
        out = _preprocess_tool(
            df=df,
            drop_cols=st.get("drop_cols", []),
            duplicate_strategy=st.get("duplicate_strategy", "drop"),
            missing_strategy=st.get("missing_strategy"),
            column_mapping=st.get("column_mapping"),
            type_overrides=st.get("type_overrides"),
            preserve_column_names=st.get("preserve_column_names", False),
        )

        st["pre_df"] = out["df"]
        st["pre_preview"] = out["preview"]
        st["pre_col_types"] = out["col_types"]
        st["pre_type_params"] = out["type_params"]
        st["pre_stats"] = out["stats"]

        st["history"].append({"tool": "preprocess_data", "status": "finished"})
    except Exception as e:
        st["errors"].append(f"preprocess_data: {e}")
        st["history"].append({"tool": "preprocess_data", "status": "failed"})
    return st


@tool("choose_task_type", return_direct=False)
def choose_task_type_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer and set `task_type` ("classification" or "regression") from
    the current target column.

    Uses:
    - pre_df if available, else clean_df
    - target_col

    Populates:
    - task_type
    """
    st = _copy(state)
    _ensure_lists(st)

    # If already set, no-op.
    if st.get("task_type"):
        st["history"].append(
            {"tool": "choose_task_type", "status": "skipped", "reason": "already_set"}
        )
        return st

    df = _get_df_for_model(st)
    target = st.get("target_col")

    if df is None or target is None or target not in df.columns:
        st["errors"].append("choose_task_type: missing data or 'target_col' not found.")
        st["history"].append({"tool": "choose_task_type", "status": "failed"})
        return st

    try:
        y = df[target]
        # Ensure we operate on a Series, not a DataFrame slice
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        task = _choose_task_type(y)
        st["task_type"] = task
        st["history"].append(
            {"tool": "choose_task_type", "status": "finished", "task_type": task}
        )
    except Exception as e:
        st["errors"].append(f"choose_task_type: {e}")
        st["history"].append({"tool": "choose_task_type", "status": "failed"})

    return st


@tool("train_baselines", return_direct=False)
def train_baselines_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train baseline models on the (pre)processed data.

    Uses:
    - pre_df if available, else clean_df
    - target_col
    - task_type (if missing, auto-detected)

    Populates:
    - train_result (from train_baselines)
    - best_model_name
    - best_model_row
    """
    st = _copy(state)
    _ensure_lists(st)

    df = _get_df_for_model(st)
    target = st.get("target_col")

    if df is None or target is None or target not in df.columns:
        st["errors"].append(
            "train_baselines: need data and a valid 'target_col' before training."
        )
        st["history"].append({"tool": "train_baselines", "status": "failed"})
        return st

    try:
        y = df[target]
        # Ensure Series, not DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        task = st.get("task_type") or _choose_task_type(y)
        st["task_type"] = task

        X = df.drop(columns=[target])
        res = _baseline_training_tool(X, y, task)
        st["train_result"] = res

        # pick best baseline if results exist
        results_df = res.get("results")
        if isinstance(results_df, pd.DataFrame) and not results_df.empty:
            best_name, best_row = best_model_by_task(task, results_df)
            st["best_model_name"] = best_name
            st["best_model_row"] = best_row

        st["history"].append(
            {
                "tool": "train_baselines",
                "status": "finished",
                "task_type": task,
                "best_model": st.get("best_model_name"),
            }
        )
    except Exception as e:
        st["errors"].append(f"train_baselines: {e}")
        st["history"].append({"tool": "train_baselines", "status": "failed"})

    return st


@tool("tune_best_model_optuna", return_direct=False)
def tune_best_model_optuna_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tune the CURRENT best model using Optuna-based Bayesian optimization.

    Uses:
    - task_type
    - best_model_name
    - train_result.{X_train, y_train, X_test, y_test}

    Optional state inputs:
    - tune_metric: user-chosen metric name (e.g., "f1", "accuracy", "r2", "rmse", "mae")
      If absent, we default by task: classification→"f1", regression→"r2".

    Populates:
    - tuned_result
    """
    st = _copy(state)
    _ensure_lists(st)

    tr = st.get("train_result")
    best_name = st.get("best_model_name")
    task = st.get("task_type")

    if not tr or not best_name or not task:
        st["errors"].append(
            "tune_best_model_optuna: requires train_result, best_model_name, and task_type."
        )
        st["history"].append({"tool": "tune_best_model_optuna", "status": "failed"})
        return st

    X_train = tr.get("X_train")
    y_train = tr.get("y_train")
    X_test = tr.get("X_test")
    y_test = tr.get("y_test")

    if X_train is None or y_train is None or X_test is None or y_test is None:
        st["errors"].append(
            "tune_best_model_optuna: missing train/test splits in train_result."
        )
        st["history"].append({"tool": "tune_best_model_optuna", "status": "failed"})
        return st

    try:
        # Metric defaults + direction derived automatically
        chosen_metric = validate_metric_for_task(task, st.get("tune_metric", default_metric(task)))
        direction = metric_direction(chosen_metric)

        tune_res = _tune_with_optuna(
            X_train,
            y_train,
            X_test,
            y_test,
            task_type=task,
            model_name=best_name,
            n_trials=40,              # default trials
            timeout=None,
            direction=direction,
            metric=chosen_metric,
        )
        st["tuned_result"] = tune_res
        st["history"].append(
            {
                "tool": "tune_best_model_optuna",
                "status": "finished",
                "model": best_name,
                "metric": chosen_metric,
                "direction": direction,
                "n_trials": 40,
            }
        )
    except Exception as e:
        st["errors"].append(f"tune_best_model_optuna: {e}")
        st["history"].append({"tool": "tune_best_model_optuna", "status": "failed"})

    return st


@tool("tune_best_model_random_search", return_direct=False)
def tune_best_model_random_search_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tune the CURRENT best model using RandomizedSearchCV.

    Uses:
    - task_type
    - best_model_name
    - train_result.{X_train, y_train, X_test, y_test}

    Optional state inputs:
    - tune_metric: user-chosen metric (if absent defaults by task)
    - rs_n_iter, rs_cv, rs_random_state
    - rs_max_depth_lo, rs_max_depth_hi
    - rs_n_estimators_lo, rs_n_estimators_hi

    Populates:
    - tuned_result
    """
    st = _copy(state)
    _ensure_lists(st)

    tr = st.get("train_result")
    best_name = st.get("best_model_name")
    task = st.get("task_type")

    if not tr or not best_name or not task:
        st["errors"].append(
            "tune_best_model_random_search: requires train_result, best_model_name, and task_type."
        )
        st["history"].append(
            {"tool": "tune_best_model_random_search", "status": "failed"}
        )
        return st

    X_train = tr.get("X_train")
    y_train = tr.get("y_train")
    X_test = tr.get("X_test")
    y_test = tr.get("y_test")

    if X_train is None or y_train is None or X_test is None or y_test is None:
        st["errors"].append(
            "tune_best_model_random_search: missing train/test splits in train_result."
        )
        st["history"].append(
            {"tool": "tune_best_model_random_search", "status": "failed"}
        )
        return st

    # Defaults with optional overrides from state (so UI / chat can control)
    n_iter = int(st.get("rs_n_iter", 40))          # default 40
    cv = int(st.get("rs_cv", 3))                   # default 3
    rs_seed = int(st.get("rs_random_state", 42))

    md_lo = int(st.get("rs_max_depth_lo", 3))
    md_hi = int(st.get("rs_max_depth_hi", 12))
    ne_lo = int(st.get("rs_n_estimators_lo", 100))
    ne_hi = int(st.get("rs_n_estimators_hi", 600))

    try:
        # Pick metric and convert to sklearn scorer
        chosen_metric = validate_metric_for_task(task, st.get("tune_metric", default_metric(task)))
        scoring = metric_to_sklearn_scorer(task, chosen_metric)

        tune_res = _tune_with_random_search(
            X_train,
            y_train,
            X_test,
            y_test,
            task_type=task,
            model_name=best_name,
            n_iter=n_iter,
            cv=cv,
            random_state=rs_seed,
            scoring=scoring,
            max_depth_range=(md_lo, md_hi),
            n_estimators_range=(ne_lo, ne_hi),
        )
        st["tuned_result"] = tune_res
        st["history"].append(
            {
                "tool": "tune_best_model_random_search",
                "status": "finished",
                "model": best_name,
                "metric": chosen_metric,
                "scoring": scoring,
                "n_iter": n_iter,
                "cv": cv,
            }
        )
    except Exception as e:
        st["errors"].append(f"tune_best_model_random_search: {e}")
        st["history"].append(
            {"tool": "tune_best_model_random_search", "status": "failed"}
        )

    return st
