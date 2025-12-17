# app/agents/chat_orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pickle
import json

import pandas as pd
import numpy as np
import streamlit as st_ui

from app.agents.llm_utils import chat_once
from app.agents.prompts import (
    SYSTEM_DATA_SUMMARY,
    SYSTEM_QA_AGENT,
    SYSTEM_PREPROCESS_PLANNER,  # NEW: planner prompt
    SYSTEM_EXPLAINER,           # 👈 add this
    explanation_prompt,
)
from app.core.preprocessing import coerce_nulls, missing_report, dtypes_dict
from app.core.utils import best_model_by_task, detect_task_type
from app.agents.runner import run_automl_graph

from app.agents.intent_router import IntentRouter
from app.agents.planner import Planner
from app.agents.intent_normalizer import IntentNormalizer




class ChatOrchestrator:
    def __init__(self) -> None:
        self.intent_router = IntentRouter()
        self.planner = Planner()
        self.intent_normalizer = IntentNormalizer()

    # -------------------- Public: entry after upload --------------------
    def start_after_upload(self, df: pd.DataFrame, state: Dict[str, Any]) -> Dict[str, Any]:
        st = state.copy()

        # Normalize + keep a clean copy for analysis (full preprocess via graph later)
        st["raw_df"] = df
        st["clean_df"] = coerce_nulls(df.copy())
        st["pre_df"] = None
        st["messages"] = st.get("messages", [])

        # Quick stats
        n_rows, n_cols = st["clean_df"].shape
        miss = missing_report(st["clean_df"])
        all_nan = miss["all_nan_columns"]

        # Brief, warm paragraph via LLM
        para = chat_once(
            system=SYSTEM_DATA_SUMMARY,
            user=(
                f"Shape: {n_rows} rows × {n_cols} columns.\n\n"
                f"Columns: {list(st['clean_df'].columns)}\n\n"
                f"Missing: {miss['missing_by_column']}\n\n"
                f"All-NaN: {all_nan}\n\n"
            ),
            model="gpt-4o-mini",
            temperature=0.2,
        )

        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    f"{para}\n\n"
                    "This is a **quick preview of the first 5 rows** of your dataset so you can verify that it loaded correctly.\n\n"
                    "What would you like to do next?\n"
                    "• Explore or preview the data\n"
                    "• Clean / preprocess the data\n"
                    "• Build a machine learning model\n\n"
                    "Just tell me what you’d like to do."
                ),
            }
        )
        st["stage"] = "post_upload_orientation"

        # User-config buckets (the wizard fills these incrementally)
        st.setdefault("pp_missing_strategy", {})      # col -> mean/median/mode/drop/fill
        st.setdefault("pp_duplicate_strategy", None)  # drop/keep_first/keep_last/mark
        st.setdefault("pp_type_overrides", {})        # col -> int/float/boolean/timestamp/string
        st.setdefault("pp_drop_all_nan_cols", [])     # list of col names
        st.setdefault("pp_column_mapping", {})        # old -> new
        st.setdefault("pp_preserve_column_names", False)

        # Bookkeeping for which wizard steps are done
        st.setdefault("done_missing", False)
        st.setdefault("done_duplicates", False)
        st.setdefault("done_dtypes", False)
        st.setdefault("done_drop_all_nan", False)
        st.setdefault("done_rename", False)

        # Preview / training / tuning helpers
        st.setdefault("show_only_preview", False)
        st.setdefault("tuning_stage", None)           # None / "ask_consent" / "choose_method" / "choose_metric"
        st.setdefault("tuning_offered", False)
        st.setdefault("chosen_tune_method", None)     # default decided later (bayesian)
        st.setdefault("tune_metric", None)            # ask user; defaults by task if omitted

        # Supervisor / graph-related defaults
        st.setdefault("history", [])
        st.setdefault("errors", [])
        st["want_preprocess"] = False
        st["want_train"] = False
        st["want_tune"] = False
        st["require_approval"] = False
        st["approved"] = False
        st["supervisor_reason"] = ""

        st["last_bot"] = None
        st.setdefault("target_task_types", {})
        return st

    # -------------------- Hyperparameter tuning helpers --------------------
    def ask_tuning_opt_in(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called once after baseline training completes.
        New flow (for non-tech users):
          1) Ask consent to tune.
          2) If yes → ask ONLY for metric to optimize (suggest default).
          3) We default the method to Bayesian and default all counts/ranges.
          4) Immediately trigger tuning via graph (no extra buttons).
        """
        st = state.copy()
        if st.get("train_result") is None:
            return st
        if st.get("tuning_offered"):
            return st

        # Suggest a metric based on task
        task = st.get("task_type", "classification")
        suggested = "f1" if task == "classification" else "r2"

        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "Your baseline models have been trained ✅\n\n"
                    "Would you like me to **tune the best model’s hyperparameters** to try and get even better performance? (yes / no)\n\n"
                    f"_If yes, I’ll use **Bayesian optimization** with safe defaults and only need a **metric** (e.g., **{suggested}**)._"
                ),
            }
        )
        st["tuning_stage"] = "ask_consent"
        st["tuning_offered"] = True
        return st

    def _tuning_methods_brief(self) -> str:
        system = (
            "You explain ML concepts to non-technical users in very simple language. "
            "Be brief and friendly."
        )
        user = (
            "In 4–6 short bullet points total, explain what Bayesian optimization "
            "and Random Search are for hyperparameter tuning, and how they differ. "
            "No formulas, no heavy jargon."
        )
        try:
            return chat_once(system=system, user=user, model="gpt-4o-mini", temperature=0.3)
        except Exception:
            return (
                "- **Bayesian optimization**: a smart trial-and-error method that learns which settings look promising.\n"
                "- **Random search**: tries many random setting combinations. Simple and often strong, but it doesn’t learn across tries.\n"
            )

    def _recommend_tuning(self, st: Dict[str, Any]) -> Tuple[str, str]:
        try:
            tr = st.get("train_result", {})
            Xtr = tr.get("X_train")
            Xte = tr.get("X_test")
            n_rows = (len(Xtr) if Xtr is not None else 0) + (len(Xte) if Xte is not None else 0)

            df = tr.get("results")
            task = st.get("task_type", "classification")
            row = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                if task == "classification" and "f1" in df.columns:
                    row = df.loc[df["f1"].idxmax()].to_dict()
                elif task != "classification" and "r2" in df.columns:
                    row = df.loc[df["r2"].idxmax()].to_dict()
                else:
                    row = df.iloc[0].to_dict()
            cv_std = float(row.get("cv_std")) if row and "cv_std" in row and pd.notnull(row["cv_std"]) else None

            if n_rows <= 10_000 or (cv_std is not None and cv_std > 0.05):
                reason = (
                    f"Your dataset is about {n_rows} rows"
                    + (f" and CV variability (std≈{cv_std:.3f}) looks a bit high." if cv_std is not None else ".")
                    + " A smarter search usually pays off on smaller or less-stable setups."
                )
                return "bayesian", reason
            else:
                reason = (
                    f"Your dataset is about {n_rows} rows"
                    + (f" and CV variability (std≈{cv_std:.3f}) looks stable." if cv_std is not None else ".")
                    + " A broader random sweep is a good, fast first step."
                )
                return "random_search", reason
        except Exception:
            return "bayesian", "Bayesian optimization is a safe default when in doubt."

    def _parse_metric(self, text: str, task_type: str) -> Optional[str]:
        t = text.lower()
        # Common aliases
        if "f1" in t or "f-1" in t:
            return "f1"
        if "accuracy" in t or "acc" in t:
            return "accuracy"
        if "precision" in t:
            return "precision"
        if "recall" in t:
            return "recall"
        if "r2" in t or "r^2" in t or "r-squared" in t or "r squared" in t:
            return "r2"
        if "rmse" in t:
            return "rmse"
        if "mae" in t:
            return "mae"
        # Shortcuts like "maximize f1" or "optimize r2" are handled above already.
        # Provide safe defaults by task if user only says "yes".
        return "f1" if task_type == "classification" else "r2"
    
    def _parse_target_from_text(self, text: str, st: Dict[str, Any]) -> Optional[str]:
        """
        Detect if the user is specifying a target column in natural language.

        Supports:
          - "use price as target"
          - "target column is loan_status"
          - "predict sale_price"
          - "my target is churn"
          - "use age as my label"
          - "set target to income"
          - "select income as my target column"
          - "as my target"
          - and also just the bare column name, e.g. "loan_status"
        """
        
        if not text:
            return None

        t_raw = text.strip()
        if not t_raw:
            return None

        t = t_raw.lower()

        # ---- Only attempt this if data is loaded ----
        df: Optional[pd.DataFrame] = None

        pre_df = st.get("pre_df")
        if isinstance(pre_df, pd.DataFrame):
            df = pre_df
        else:
            clean_df = st.get("clean_df")
            if isinstance(clean_df, pd.DataFrame):
                df = clean_df

        if df is None:
            return None

        cols = list(df.columns)
        cols_lower = [c.lower() for c in cols]

        # ----------------------------------------------------
        # 1) If user simply typed a column name → accept it
        # ----------------------------------------------------
        if t in cols_lower:
            idx = cols_lower.index(t)
            return cols[idx]

        # ----------------------------------------------------
        # 2) Check trigger phrases (natural language instructions)
        # ----------------------------------------------------
        trigger_phrases = [
            "target is",
            "target column is",
            "use",
            "predict",
            "label is",
            "my label is",
            "set target to",
            "use column",
            "i want to predict",
            "i want to forecast",
            "is my target column",
        ]
        if not any(p in t for p in trigger_phrases):
            return None

        # ----------------------------------------------------
        # 3) Try exact match inside full text
        # ----------------------------------------------------
        for c in cols:
            if c.lower() in t:
                return c

        # ----------------------------------------------------
        # 4) Fuzzy token match
        # ----------------------------------------------------
        tokens = t.replace(",", " ").replace(".", " ").split()
        for tok in tokens:
            if tok in cols_lower:
                idx = cols_lower.index(tok)
                return cols[idx]

        return None
    
    def _parse_forecast_horizon(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Examples:
          - 'monthly for 5 months'
          - 'next 30 days'
          - 'yearly forecast for 3 years'
        """
        t = text.lower()

        freq_map = {
            "daily": "D",
            "day": "D",
            "days": "D",
            "weekly": "W",
            "week": "W",
            "weeks": "W",
            "monthly": "MS",
            "month": "MS",
            "months": "MS",
            "yearly": "YS",
            "year": "YS",
            "years": "YS",
        }

        freq = None
        for k, v in freq_map.items():
            if k in t:
                freq = v
                break

        import re
        m = re.search(r"(\d+)", t)
        periods = int(m.group(1)) if m else None

        if not freq or not periods:
            return None

        return {
            "forecast_freq": freq,
            "forecast_periods": periods,
        }

    
    def _run_baseline_training(self, st: Dict[str, Any], target_col: str) -> Dict[str, Any]:
        """
        Run baseline training via the LangGraph, assuming:
          - a dataframe (pre_df or clean_df) is available
          - target_col is a valid column name

        It will:
          - set st['target_col']
          - infer the task_type from the *current* target (classification vs regression),
            but remember that decision per-column so it never flips later.
          - reset any previous training/tuning state
          - set the want_train/approved flags
          - call run_automl_graph (with a second pass if preprocessing just got created)
          - append a friendly assistant message about what happened
        """
        st = st.copy()

        # Pick the data frame to train on
        pre_df = st.get("pre_df")
        clean_df = st.get("clean_df")
        df = pre_df if pre_df is not None else clean_df

        # Safety checks
        if df is None:
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I don't see any data loaded yet. "
                        "Please upload a CSV first, then tell me which column is the target."
                    ),
                }
            )
            return st

        if target_col not in df.columns:
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I tried to start training, but I couldn't find that column in your data.\n\n"
                        "Please mention the target again using an existing column name. "
                        "For example: `Use loan_status as my target`."
                    ),
                }
            )
            return st

        # Persist the chosen target in state
        st["target_col"] = target_col

        # Reset ANY previous training / tuning so we can train again cleanly
        st["train_result"] = None
        st["best_model_name"] = None
        st["best_model_row"] = None
        st["tuned_result"] = None
        st["want_tune"] = False
        st["tuning_stage"] = None
        st["tuning_offered"] = False
        st["chosen_tune_method"] = None
        st["tune_metric"] = None
        st["show_training_panel"] = True
        st["show_only_preview"] = False
        st["stage"] = "preview_download"

        # We keep a per-column memory of the task type
        target_task_types = st.get("target_task_types") or {}
        remembered_task = target_task_types.get(target_col)

        # --- Infer task type ONLY if we don't have a remembered one yet ---
        if remembered_task in {"classification", "regression"}:
            task = remembered_task
        else:
            try:
                from app.agents.nodes import choose_task_type  # local import to avoid cycles
                y = df[target_col]
                task = choose_task_type(y)
            except Exception:
                y = df[target_col]
                # Slightly smarter fallback:
                # - many distinct numeric values -> regression
                # - few distinct values (like 0/1) -> classification
                nunique = y.nunique(dropna=True)
                if pd.api.types.is_numeric_dtype(y) and nunique > 10:
                    task = "regression"
                else:
                    task = "classification"

            # Remember this decision for this column so it never flips later
            target_task_types[target_col] = task
            st["target_task_types"] = target_task_types

        st["task_type"] = task

        # Let the user know what we're doing
        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    f"Great, I’ll use **{target_col}** as the target column.\n\n"
                    f"I've detected this as a **{task.capitalize()}** problem.\n\n"
                    "Now I’ll train a set of **baseline models** for you. "
                    "This may take a moment ⏳."
                ),
            }
        )

        # --- Trigger training via the graph ---
        st["want_train"] = True
        st["approved"] = True  # pass HITL gate for training

        out = run_automl_graph(st)

        # If preprocessing ran first and training didn't yet, do one more pass
        if (
            out.get("train_result") is None
            and out.get("pre_df") is not None
            and out.get("want_train")
        ):
            out["approved"] = True
            out = run_automl_graph(out)

        # Final messaging based on outcome
        tr = out.get("train_result")
        if tr is not None and isinstance(tr.get("results"), pd.DataFrame):
            df_res = tr["results"]
            n_models = len(df_res)

            # ------------------------------
            # Save best model for download
            # ------------------------------
            best_model_name = out.get("best_model_name")

            # Fallback (rare): if tool didn't set name, try to pick from results safely
            if not best_model_name and not df_res.empty:
                try:
                    best_model_name = df_res.iloc[0]["model"]
                    out["best_model_name"] = best_model_name
                except Exception:
                    best_model_name = None

            best_model = tr.get("fitted", {}).get(best_model_name) if best_model_name else None
            out["best_model_bytes"] = pickle.dumps(best_model) if best_model is not None else None

            out["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        f"✅ Done! I trained **{n_models} baseline model(s)** using **{target_col}** as the target.\n\n"
                        "You can see the **leaderboard** and download the **best model** in the panel below.\n\n"
                        "Feel free to ask me about the metrics (accuracy, F1, R², etc.) "
                        "or say **tune the model** if you’d like to try hyperparameter tuning.\n\n"
                        "__show_training_results__"
                    ),
                }
            )
        else:
            out["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I tried to run baseline training, but it didn’t complete successfully.\n\n"
                        "Please check the **errors / history** section above, or try again "
                        "after adjusting the data or target column."
                    ),
                }
            )

        if out.get("train_result"):
            has_marker = any(
                isinstance(m.get("content"), str) and "__show_training_results__" in m["content"]
                for m in out.get("messages", [])
                if m.get("role") == "assistant"
                )
        if not has_marker:
            out.setdefault("messages", []).append(
                {
                    "role": "assistant",
                    "content": "__show_training_results__",
                }
            )

        return out



    # -------------------- QA helpers --------------------
    def _qa_snapshot(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """Build a compact snapshot of the current workflow state for the QA LLM."""
        # Dataset size
        df_for_train = st.get("pre_df") if st.get("pre_df") is not None else st.get("clean_df")
        if isinstance(df_for_train, pd.DataFrame):
            n_rows, n_cols = df_for_train.shape
        else:
            n_rows, n_cols = 0, 0

        train_result = st.get("train_result") or {}
        results_df = train_result.get("results")

        best_row = st.get("best_model_row")
        if best_row is None and isinstance(results_df, pd.DataFrame) and not results_df.empty:
            task = st.get("task_type", "classification")
            name, row = best_model_by_task(task, results_df)
            st["best_model_name"] = name
            st["best_model_row"] = row
            best_row = row

        metric_values: Dict[str, float] = {}
        if isinstance(best_row, dict):
            for key in ["f1", "accuracy", "precision", "recall", "r2", "mae", "rmse"]:
                if key in best_row and pd.notnull(best_row[key]):
                    metric_values[key] = float(best_row[key])

        cv_score = None
        cv_std = None
        if isinstance(best_row, dict):
            if "cv_score" in best_row and pd.notnull(best_row["cv_score"]):
                cv_score = float(best_row["cv_score"])
            if "cv_std" in best_row and pd.notnull(best_row["cv_std"]):
                cv_std = float(best_row["cv_std"])

        tuned = st.get("tuned_result") or {}

        snapshot: Dict[str, Any] = {
            "task_type": st.get("task_type"),
            "best_model_name": st.get("best_model_name"),
            "metric_values": metric_values,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "dataset_size": n_rows,
            "leaderboard_top": (
                results_df.head(5).to_dict(orient="records")
                if isinstance(results_df, pd.DataFrame)
                else None
            ),
            "tuning_available": bool(train_result),
            "train_done": bool(train_result),

            # --- tuning status ---
            "tuning_done": bool(tuned),
            "tuned_best_params": tuned.get("best_params"),
            "tuned_metrics": tuned.get("test_metrics"),

            # --- preprocessing status ---
            "preprocessing_done": bool(st.get("pre_df") is not None),
            "done_missing": st.get("done_missing"),
            "done_duplicates": st.get("done_duplicates"),
            "done_dtypes": st.get("done_dtypes"),
            "done_drop_all_nan": st.get("done_drop_all_nan"),
            "done_rename": st.get("done_rename"),
        }
        return snapshot
    
    
    def build_training_explanation(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Use the LLM explainer prompt to generate a short, plain-English
        explanation of the training results and why the best model is recommended.
        """
        st = state.copy()
        tr = st.get("train_result") or {}
        df = tr.get("results")

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        # Pick best model using the same helper as the QA snapshot
        task = st.get("task_type", "classification")
        best_name, best_row = best_model_by_task(task, df)
        st["best_model_name"] = best_name
        st["best_model_row"] = best_row

        # Dataset size (rows/cols)
        pre_df = st.get("pre_df")
        clean_df = st.get("clean_df")
        df_data = pre_df if isinstance(pre_df, pd.DataFrame) else clean_df

        if isinstance(df_data, pd.DataFrame):
            n_rows, n_cols = df_data.shape
        else:
            Xtr = tr.get("X_train")
            Xte = tr.get("X_test")
            n_rows = (len(Xtr) if Xtr is not None else 0) + (
                len(Xte) if Xte is not None else 0
            )
            n_cols = len(df.columns)

        # Collect the key metrics for the best model
        metric_parts = []
        for key in ["accuracy", "f1", "precision", "recall", "r2", "rmse", "mae"]:
            if key in best_row and pd.notnull(best_row[key]):
                metric_parts.append(f"{key}≈{float(best_row[key]):.3f}")
        metrics_text = ", ".join(metric_parts)

        summary = (
            f"Task type: {task}. Dataset has about {n_rows} rows and {n_cols} columns.\n"
            f"Best model on the leaderboard is {best_name} with {metrics_text}."
        )

        recommendation = (
            f"I recommend {best_name} as the default model for this dataset."
        )

        try:
            return chat_once(
                system=SYSTEM_EXPLAINER,
                user=explanation_prompt(summary, recommendation),
                model="gpt-4o-mini",
                temperature=0.25,
            )
        except Exception:
            # Safe fallback if LLM call fails
            return (
                f"The best model is {best_name} with metrics: {metrics_text}. "
                "It offers a good trade-off between performance and robustness on this dataset."
            )



    def _qa_answer(self, user_text: str, st: Dict[str, Any]) -> str:
        """Call the QA LLM to answer questions about accuracy, tuning, preprocessing, etc."""
        snap = self._qa_snapshot(st)
        try:
            payload = {"question": user_text, "snapshot": snap}
            return chat_once(
                system=SYSTEM_QA_AGENT,
                user=json.dumps(payload, default=str),
                model="gpt-4o-mini",
                temperature=0.2,
            )
        except Exception:
            # Fallback: simple, rule-based messages
            if not snap.get("train_done"):
                return (
                    "We haven’t trained any models yet, so metrics like accuracy or F1 "
                    "are not available.\n\n"
                    "Please **tell me in the chat which column should be used as the target**.\n"
                    "For example: `Use loan_status as my target`.\n\n"
                    "I’ll detect the problem type and run **Train baselines** for you."
                )
            if not snap.get("tuning_done"):
                return (
                    "We do have baseline models and metrics, but no hyperparameter tuning results yet.\n\n"
                    "You can say **tune the model** if you’d like me to search for better settings."
                )
            return (
                "I couldn’t run the explanation helper right now, but you can inspect the "
                "**leaderboard** table above for the latest metrics and tuned parameters."
            )


    def _looks_like_qa(self, text: str) -> bool:
        """Heuristic: does this message look like an analytical question?"""
        t = text.lower()
        if "?" in t:
            return True
        keywords = [
            "accuracy", "f1", "precision", "recall", "r2", "rmse", "mae",
            "metric", "score", "leaderboard", "best model", "which model",
            "tuning", "hyperparameter", "best params", "parameters",
            "preprocess", "pre-processing", "cleaning done",
            "have i done", "did we do", "completed", "finished",
            "result", "results", "explain", "explanation",
        ]
        return any(k in t for k in keywords)

    # -------------------- Automatic preprocessing planner --------------------
    def _auto_plan_preprocessing(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use the LLM planner to decide preprocessing strategies internally
        (no user wizard). We will:
          - always DROP duplicate rows,
          - always DROP all-NaN columns (if any),
          - choose per-column missing strategies via LLM,
        then store everything in the pp_* config fields so the graph
        can actually run preprocessing.
        """
        st = st.copy()
        df = st.get("clean_df")
        if df is None:
            return st

        miss = missing_report(df)
        all_nan_cols = miss["all_nan_columns"]
        missing_by_column = miss["missing_by_column"]
        dtypes = dtypes_dict(df)

        user_payload = {
            "columns_and_dtypes": dtypes,
            "missing_by_column": missing_by_column,
            "all_nan_columns": all_nan_cols,
            "notes": (
                "Please propose a good preprocessing plan for non-technical users. "
                "Always drop duplicate rows. Always include all-NaN columns in drop_cols. "
                "For each column that has missing values > 0, pick a reasonable strategy."
            ),
        }

        try:
            raw = chat_once(
                system=SYSTEM_PREPROCESS_PLANNER,
                user=json.dumps(user_payload),
                model="gpt-4o-mini",
                temperature=0.0,
            )
            plan = json.loads(raw)
        except Exception:
            # Fallback: simple defaults
            plan = {
                "drop_cols": all_nan_cols,
                "duplicate_strategy": "drop",
                "missing_strategy": {
                    c: "mean" for c, cnt in missing_by_column.items() if cnt > 0
                },
                "column_mapping": {},
                "type_overrides": {},
                "preserve_column_names": False,
            }

        # Enforce our business rules on top
        drop_cols = plan.get("drop_cols") or all_nan_cols
        duplicate_strategy = "drop"  # always drop duplicates
        missing_strategy = plan.get("missing_strategy") or {
            c: "mean" for c, cnt in missing_by_column.items() if cnt > 0
        }
        column_mapping = plan.get("column_mapping") or {}
        type_overrides = plan.get("type_overrides") or {}
        preserve_column_names = bool(column_mapping) or bool(
            plan.get("preserve_column_names", False)
        )

        # Store into config buckets
        st["pp_drop_all_nan_cols"] = drop_cols
        st["pp_duplicate_strategy"] = duplicate_strategy
        st["pp_missing_strategy"] = missing_strategy
        st["pp_column_mapping"] = column_mapping
        st["pp_type_overrides"] = type_overrides
        st["pp_preserve_column_names"] = preserve_column_names

        # Mark steps as done (so wizard doesn’t nag)
        st["done_missing"] = bool(missing_strategy)
        st["done_duplicates"] = True
        st["done_drop_all_nan"] = bool(drop_cols)
        st["done_dtypes"] = bool(type_overrides)
        st["done_rename"] = bool(column_mapping)

        # We want to go straight to preview after preprocessing
        st["stage"] = "preview_download"
        st["show_only_preview"] = True

        # Ask the graph to run preprocessing
        st["want_preprocess"] = True

        return st
    
    def advance_plan(self, st: Dict[str, Any]) -> Dict[str, Any]:
        st.setdefault("messages", [])
        steps = st.get("plan_steps", [])
        cursor = int(st.get("plan_cursor", 0))

        if cursor >= len(steps):
            st["stage"] = "idle"
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": "Plan completed.",
                }
            )
            return st

        step = steps[cursor]

        # ---------------- PREPROCESS ----------------
        if step == "preprocess":
            st["want_preprocess"] = True
            st["plan_cursor"] = cursor + 1
            return st
        
        # ---------------- CONFIRM FORECAST HORIZON ----------------
        if step == "confirm_forecast_horizon":
            if st.get("forecast_freq") and st.get("forecast_periods"):
                st["plan_cursor"] = cursor + 1
                return st

            st["stage"] = "await_forecast_horizon"
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "How far do you want to forecast?\n\n"
                        "Examples:\n"
                        "• monthly for next 5 months\n"
                        "• daily for 30 days\n"
                        "• yearly for 3 years"
                    ),
                }
            )
            return st

        # ---------------- CONFIRM DS ----------------
        if step == "confirm_ds":
            if st.get("ds_col"):
                st["plan_cursor"] = cursor + 1
                return st

            st["stage"] = "await_ds"
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": "Which column is the date/time column? (this will be ds)",
                }
            )
            return st

        # ---------------- CONFIRM Y ----------------
        if step == "confirm_y":
            if st.get("y_col"):
                st["plan_cursor"] = cursor + 1
                st["stage"] = "executing_plan"
                return st

            st["stage"] = "await_y"
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": "Which numeric column do you want to forecast? (this will be y)",
                }
            )
            return st

        # ---------------- FORECAST ----------------
        if step == "forecast":
            if not st.get("ds_col") or not st.get("y_col"):
                st["errors"].append("Forecast called without ds/y.")
                st["stage"] = "idle"
                return st
            
            st["want_forecast"] = True
            st["plan_cursor"] = cursor + 1
            return st

        # ---------------- CONFIRM TARGET ----------------
        if step == "confirm_target":
            if st.get("target_col"):
                st["plan_cursor"] = cursor + 1
                return st

            st["stage"] = "await_target"
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": "Which column should be used as the target?",
                }
            )
            return st

        # ---------------- TRAIN ----------------
        if step == "train":
            st["want_train"] = True
            st["plan_cursor"] = cursor + 1
            return st

        # ---------------- TUNE ----------------
        if step == "tune":
            st["want_tune"] = True
            st["plan_cursor"] = cursor + 1
            return st

        # ---------------- PREVIEW ----------------
        if step == "preview":
            st["show_only_preview"] = True
            st["plan_cursor"] = cursor + 1
            return st

        return st

    def _parse_column_from_text(self, text: str, st: Dict[str, Any]) -> Optional[str]:
        if not text:
            return None
        t_raw = text.strip()
        if not t_raw:
            return None
        t = t_raw.lower()

        df: Optional[pd.DataFrame] = None
        if isinstance(st.get("pre_df"), pd.DataFrame):
            df = st["pre_df"]
        elif isinstance(st.get("clean_df"), pd.DataFrame):
            df = st["clean_df"]

        if df is None:
            return None

        cols = list(df.columns)
        cols_lower = [c.lower() for c in cols]

        # exact column name
        if t in cols_lower:
            return cols[cols_lower.index(t)]

        # contains column name
        for c in cols:
            if c.lower() in t:
                return c

        # token match
        tokens = t.replace(",", " ").replace(".", " ").split()
        for tok in tokens:
            if tok in cols_lower:
                return cols[cols_lower.index(tok)]

        return None

    def _execute_plan_until_pause(self, st: Dict[str, Any]) -> Dict[str, Any]:
        while True:
            stage = st.get("stage")

            # --------------------------------------------------
            # 1) Pause immediately if user input is required
            # --------------------------------------------------
            if stage in {
                "await_forecast_horizon",
                "await_ds", 
                "await_y", 
                "await_target"
            }:
                return st

            prev_cursor = int(st.get("plan_cursor", 0))

            # --------------------------------------------------
            # 2) Advance ONE plan step
            # --------------------------------------------------
            st = self.advance_plan(st)

            # --------------------------------------------------
            # 3) If a runnable flag is set → RUN GRAPH ONCE
            # --------------------------------------------------
            ran_something = False

            if (
                st.get("want_preprocess")
                or st.get("want_train")
                or st.get("want_tune")
                or st.get("want_forecast")
            ):
                st["approved"] = True
                st = run_automl_graph(st)
                ran_something = True

                # clear flags AFTER execution
                st["want_preprocess"] = False
                st["want_train"] = False
                st["want_tune"] = False
                st["want_forecast"] = False

            # --------------------------------------------------
            # 4) HARD STOP after forecast
            # --------------------------------------------------
            if ran_something and st.get("forecast_result") is not None:
                st["stage"] = "idle"
                return st

            # --------------------------------------------------
            # 5) Safety: stop if cursor did not move
            # --------------------------------------------------
            if (
                int(st.get("plan_cursor", 0)) == prev_cursor
                and not ran_something
                and st.get("stage") not in {"await_ds", "await_y", "await_target"}
            ):
                st["stage"] = "idle"
                return st

            # --------------------------------------------------
            # 6) Plan fully consumed → stop forever
            # --------------------------------------------------
            if int(st.get("plan_cursor", 0)) >= len(st.get("plan_steps", [])):
                st["stage"] = "idle"
                return st


    def handle(self, user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        st = state.copy()
        st.setdefault("messages", [])
        st.setdefault("stage", "await_upload")
        st.setdefault("require_approval", False)
        st.setdefault("approved", False)
        st.setdefault("history", [])
        st.setdefault("errors", [])

        text = (user_text or "").strip()
        if not text:
            return st

        # ======================================================
        # 1) PLAN APPROVAL → EXECUTE
        # ======================================================
        if st.get("stage") == "plan_proposed":
            st = self.planner.handle_confirmation(text, st)
            if st.get("stage") == "executing_plan":
                return self._execute_plan_until_pause(st)
            return st
        
        if st.get("stage") in {"idle", "preview_download"}:
            horizon = self._parse_forecast_horizon(text)
            if horizon:
                st.update(horizon)
                return self.planner.handle_multi_step(
                    user_text=text,
                    state=st,
                    intent={
                        "kind": "plan",
                        "actions": ["forecast"],
                        "reason": "Forecast horizon detected",
                    },
                )        

        # ======================================================
        # 2) USER INPUT FOR CONFIRM STEPS (FIXED)
        # ======================================================
        if st.get("stage") in {
            "await_forecast_horizon",
            "await_ds",
            "await_y",
            "await_target",
        }:
            
            # ---------- CONFIRM FORECAST HORIZON ----------
            if st["stage"] == "await_forecast_horizon":
                parsed = self._parse_forecast_horizon(text)
                if not parsed:
                    st["messages"].append(
                        {
                            "role": "assistant",
                            "content": (
                                "I couldn’t understand that.\n\n"
                                "Please say something like:\n"
                                "• monthly for 5 months\n"
                                "• daily for 30 days\n"
                                "• yearly for 3 years"
                            ),
                        }
                    )
                    return st
                
                st["forecast_freq"] = parsed["forecast_freq"]
                st["forecast_periods"] = parsed["forecast_periods"]

                st["plan_cursor"] = int(st.get("plan_cursor", 0)) + 1
                st["stage"] = "executing_plan"

                return self._execute_plan_until_pause(st)
            
            col = self._parse_column_from_text(text, st)

            if not col:
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": "I couldn’t match that to a column name. Please try again.",
                    }
                )
                return st

            # 🔴 CRITICAL FIX: consume confirm step
            st["plan_cursor"] = int(st.get("plan_cursor", 0)) + 1

            if st["stage"] == "await_ds":
                st["ds_col"] = col
                st["stage"] = "executing_plan"

            elif st["stage"] == "await_y":
                st["y_col"] = col
                st["stage"] = "executing_plan"

            elif st["stage"] == "await_target":
                st["target_col"] = col
                st["stage"] = "executing_plan"

            return self._execute_plan_until_pause(st)

        # ======================================================
        # 3) QA
        # ======================================================
        if st.get("stage") in {None, "idle", "preview_download"} and self._looks_like_qa(text):
            answer = self._qa_answer(text, st)
            st["messages"].append(
                {"role": "assistant", "content": answer}
            )
            return st

        # ======================================================
        # 4) INTENT → PLAN
        # ======================================================
        raw_intent = self.intent_router.classify(text, st)
        safe_intent = self.intent_normalizer.normalize(
            raw_intent=raw_intent,
            state=st,
            user_text=text,
        )

        if safe_intent.kind == "confirm":
            return self.planner.handle_confirmation(text, st)

        if safe_intent.kind == "plan":
            return self.planner.handle_multi_step(
                user_text=text,
                state=st,
                intent={
                    "kind": safe_intent.kind,
                    "actions": safe_intent.actions,
                    "reason": safe_intent.reason,
                },
            )

        return st


    # -------------------- Helper messages --------------------
    def _menu_message(self, st: Dict[str, Any]) -> str:
        df = st.get("clean_df")
        miss = missing_report(df)
        dup_count = int(len(df) - len(df.drop_duplicates()))
        dtypes = dtypes_dict(df)
        all_nan = miss["all_nan_columns"]

        missing_counts = miss["missing_by_column"]
        has_missing = any(v > 0 for v in missing_counts.values())
        has_duplicates = dup_count > 0
        has_all_nan = len(all_nan) > 0

        left = []
        if has_missing and not st.get("done_missing"):
            left.append("• Missing values")
        if has_duplicates and not st.get("done_duplicates"):
            left.append("• Duplicate rows")
        if has_all_nan and not st.get("done_drop_all_nan"):
            left.append("• Drop all-NaN columns")
        if not st.get("done_dtypes"):
            left.append("• Enforce data types")
        if not st.get("done_rename"):
            left.append("• Rename columns")

        todo = "\n".join(left) or "Everything looks clean already! You can go straight to preview or training."
        return (
            "Here’s what I found:\n"
            f"- **Duplicates:** ~{dup_count} duplicate rows detected\n"
            f"- **Missing values:** see the table below\n"
            f"- **Data types:** {len(dtypes)} columns (details shown below)\n"
            f"- **All-NaN columns:** {all_nan if all_nan else 'none'}\n\n"
            "You can pick a step by typing it (e.g., `missing`, `duplicates`, `types`, `drop all nan`, `rename`).\n\n"
            f"**Remaining suggestions:**\n{todo}"
        )

    def _missing_intro(self, st: Dict[str, Any]) -> str:
        df = st["clean_df"]
        miss = df.isna().sum().sort_values(ascending=False)
        top = miss[miss > 0].head(10)
        if top.empty:
            st["done_missing"] = True
            st["stage"] = "prep_menu"
            return "I don’t see missing values. You can skip this step or choose another."
        sugg: List[Tuple[str, str]] = []
        for col in top.index.tolist():
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                skew = float(np.abs(pd.Series(s).dropna().skew())) if s.dropna().size > 0 else 0.0
                method = "median" if skew > 1.0 else "mean"
            elif pd.api.types.is_datetime64_any_dtype(s):
                method = "drop"
            else:
                nunq = s.nunique(dropna=True)
                method = "mode" if nunq <= max(20, int(0.1 * len(s))) else "fill"
            sugg.append((col, method))
        bullets = "\n".join([f"- **{c}** → `{m}`" for c, m in sugg])
        return (
            "Let’s handle **missing values**.\n\n"
            "Suggested strategies (you can edit/add below):\n"
            f"{bullets}\n\n"
            "Use the controls to select column → strategy. Click **Add more** to add rows, and **Done** when finished."
        )

    def _dups_intro(self, st: Dict[str, Any]) -> str:
        df = st["clean_df"]
        dup_count = int(len(df) - len(df.drop_duplicates()))
        if dup_count == 0:
            st["done_duplicates"] = True
            st["stage"] = "prep_menu"
            return "I don’t see duplicate rows. You can skip this step or choose another."
        return (
            f"I detected ~**{dup_count} duplicate rows**.\n\n"
            "Duplicate handling is **row-wise** (not per-column). Suggestions:\n"
            "- If rows are truly identical → `drop`\n"
            "- If the latest record should win → `keep_last`\n"
            "- If you want to keep originals but mark them → `mark`\n\n"
            "Pick a **single strategy** below and click **Done**."
        )

    def _dtypes_intro(self, st: Dict[str, Any]) -> str:
        return (
            "Let’s **enforce data types** for selected columns.\n"
            "Typical picks: numeric fields → `int`/`float`, flags → `boolean`, dates → `timestamp`, everything else → `string`.\n"
            "Add one or more column → type pairs below, then click **Done**."
        )

    def _drop_all_nan_intro(self, st: Dict[str, Any]) -> str:
        all_nan = missing_report(st["clean_df"])["all_nan_columns"]
        if not all_nan:
            st["done_drop_all_nan"] = True
            st["stage"] = "prep_menu"
            return "No columns are completely NaN — nothing to drop here. You can skip this step."
        return (
            f"These columns are completely NaN: **{all_nan}**.\n\n"
            "Select which ones to drop and click **Apply**."
        )

    def _rename_intro(self, st: Dict[str, Any]) -> str:
        return (
            "You can **rename columns** here. Add old → new pairs, then click **Done**.\n"
            "Tip: Keep names simple (letters, numbers, underscores)."
        )

    # -------------------- Apply / Commit helpers --------------------
    def _maybe_all_done_message(self, st: Dict[str, Any]) -> str:
        all_done = all(
            [
                st.get("done_missing"),
                st.get("done_duplicates"),
                st.get("done_dtypes"),
                st.get("done_drop_all_nan"),
                st.get("done_rename"),
            ]
        )
        if all_done:
            st["stage"] = "preview_download"
            st["show_only_preview"] = True
            return (
                "✅ All preprocessing steps are complete. "
                "Would you like to see the **preview** and **download** the preprocessed data?"
            )
        else:
            return "Would you like to **continue preprocessing**, **see a preview**, or **go to training**?"

    def apply_missing(self, st: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        st = st.copy()
        st["pp_missing_strategy"].update({k: v for k, v in mapping.items() if k})
        st["done_missing"] = True
        msg = self._maybe_all_done_message(st)
        st["messages"].append({"role": "assistant", "content": f"✅ Missing value strategies have been recorded. {msg}"})
        st["stage"] = "prep_menu"
        st["show_only_preview"] = False
        return st

    def apply_duplicates(self, st: Dict[str, Any], strategy: Optional[str]) -> Dict[str, Any]:
        st = st.copy()
        st["pp_duplicate_strategy"] = strategy or "drop"
        st["done_duplicates"] = True
        msg = self._maybe_all_done_message(st)
        st["messages"].append({"role": "assistant", "content": f"✅ Duplicate strategy **{st['pp_duplicate_strategy']}** recorded. {msg}"})
        st["stage"] = "prep_menu"
        st["show_only_preview"] = False
        return st

    def apply_dtypes(self, st: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        st = st.copy()
        st["pp_type_overrides"].update({k: v for k, v in mapping.items() if k})
        st["done_dtypes"] = True
        msg = self._maybe_all_done_message(st)
        st["messages"].append({"role": "assistant", "content": f"✅ Data type overrides recorded. {msg}"})
        st["stage"] = "prep_menu"
        st["show_only_preview"] = False
        return st

    def apply_drop_all_nan(self, st: Dict[str, Any], cols: List[str]) -> Dict[str, Any]:
        st = st.copy()
        st["pp_drop_all_nan_cols"] = list(cols or [])
        st["done_drop_all_nan"] = True
        msg = self._maybe_all_done_message(st)
        st["messages"].append({"role": "assistant", "content": f"✅ Dropped the selected all-NaN columns (or recorded your choice). {msg}"})
        st["stage"] = "prep_menu"
        st["show_only_preview"] = False
        return st

    def apply_rename(self, st: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        st = st.copy()
        clean_map = {k: v for k, v in mapping.items() if k and v and str(k) != str(v)}
        st["pp_column_mapping"].update(clean_map)
        st["done_rename"] = True
        msg = self._maybe_all_done_message(st)
        st["messages"].append({"role": "assistant", "content": f"✅ Renaming choices recorded. {msg}"})
        st["stage"] = "prep_menu"
        st["show_only_preview"] = False
        return st

    # -------------------- Where we actually call the graph/tools --------------------
    def run_preprocess_now(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called by chat_app when user is ready to see preview / download.
        Executes preprocessing once (idempotent) via the LangGraph.
        """
        st = st.copy()
        if st.get("clean_df") is None:
            return st

        # Map wizard config → AutoMLState fields
        st["drop_cols"] = st.get("pp_drop_all_nan_cols", [])
        st["duplicate_strategy"] = st.get("pp_duplicate_strategy") or "drop"
        st["missing_strategy"] = st.get("pp_missing_strategy") or None
        st["column_mapping"] = st.get("pp_column_mapping") or None
        st["type_overrides"] = st.get("pp_type_overrides") or None
        st["preserve_column_names"] = bool(st.get("pp_column_mapping"))

        # Ask the graph to run preprocessing once
        st["want_preprocess"] = True

        out = run_automl_graph(st)
        return out
    
    