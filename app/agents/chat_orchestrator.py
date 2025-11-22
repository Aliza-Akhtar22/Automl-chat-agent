# app/agents/chat_orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json

import pandas as pd
import numpy as np

from app.agents.llm_utils import chat_once
from app.agents.prompts import SYSTEM_DATA_SUMMARY, SYSTEM_QA_AGENT
from app.core.preprocessing import coerce_nulls, missing_report, dtypes_dict
from app.core.utils import best_model_by_task  # ✅ added detect_task_type
from app.agents.nodes import choose_task_type
from app.agents.runner import run_automl_graph
from app.agents.planner_agent import make_plan  # ✅ NEW: planner


# --------------- Conversation Stages ---------------
# await_upload        -> ask user to upload
# ask_preprocess      -> show data summary + ask yes/no to proceed
# prep_menu           -> suggest next steps & render per-method widgets
# prep_missing        -> missing values per-column (add more / done)
# prep_duplicates     -> duplicate rows strategy (single choice) (done)
# prep_dtypes         -> enforce dtypes (add more / done)
# prep_drop_all_nan   -> multiselect all-NaN columns (apply)
# prep_rename         -> rename columns (add more / done)
# preview_download    -> show preprocessed/clean preview + download
# await_plan_target   -> planner running; waiting for user target col
# tuning:
#   tuning_stage:
#       None / "ask_consent" / "choose_method" / "choose_metric"
#   chosen_tune_method:
#       None / "bayesian" / "random_search"
#   tune_metric:
#       None / "f1" / "accuracy" / "precision" / "recall" / "r2" / "rmse" / "mae"


class ChatOrchestrator:
    def __init__(self) -> None:
        pass

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
                    "Would you like to proceed with **preprocessing** now? (yes / no)\n"
                    "_You can also say “do everything end-to-end” if you want the full pipeline._"
                ),
            }
        )
        st["stage"] = "ask_preprocess"

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

        # ✅ Planner-related runtime state for graph-driven plan execution
        st.setdefault("plan_steps", [])     # list of tool names in order
        st.setdefault("plan_index", 0)      # current step pointer
        st.setdefault("plan_active", False)
        st.setdefault("plan_done", False)
        st.setdefault("require_target_col", False)
        st.setdefault("last_plan", None)

        st["last_bot"] = None
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
        # Provide safe defaults by task if user only says "yes".
        return "f1" if task_type == "classification" else "r2"

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
            "tuned_done": bool(tuned),
            "tuned_best_params": tuned.get("best_params"),
            "tuned_test_metrics": tuned.get("test_metrics"),
            "preprocessing_done": bool(st.get("pre_df") is not None),
            "pre_steps_done": {
                "missing": st.get("done_missing"),
                "duplicates": st.get("done_duplicates"),
                "dtypes": st.get("done_dtypes"),
                "drop_all_nan": st.get("done_drop_all_nan"),
                "rename": st.get("done_rename"),
            },
            "plan_active": st.get("plan_active"),
            "plan_done": st.get("plan_done"),
        }
        return snapshot

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
                    "are not available. Please choose a **target column** and click "
                    "**Train baselines** to start training."
                )
            if not snap.get("tuned_done"):
                return (
                    "We do have baseline models and metrics, but no hyperparameter tuning results yet. "
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
        ]
        return any(k in t for k in keywords)

    # -------------------- Planner helpers --------------------
    def _looks_like_full_pipeline(self, text: str) -> bool:
        """
        Detect high-level "do everything end-to-end" requests where the Planner should be used.
        Examples:
          - "Just do everything end to end"
          - "Clean the data, train the model and tune the best one"
          - "Please run the full automl pipeline for me"
          - "From upload to the best tuned model, do the whole thing"
        """
        t = text.lower()

        trigger_phrases = [
            "end to end",
            "end-to-end",
            "full pipeline",
            "full automl",
            "full auto ml",
            "do everything",
            "do the whole thing",
            "do the whole pipeline",
            "run everything",
            "run the whole",
            "run the entire",
            "whole thing",
            "entire thing",
            "from upload to the best tuned model",
        ]
        if any(p in t for p in trigger_phrases):
            return True

        # soft pattern: mentions clean/preprocess + train + tune/optimize
        has_clean = any(w in t for w in ["clean", "preprocess", "pre-processing"])
        has_train = "train" in t or "training" in t
        has_tune = any(w in t for w in ["tune", "tuning", "optimize", "hyperparameter"])
        if has_clean and has_train and has_tune:
            return True

        # generic "run automl for me" style
        if "automl" in t and any(w in t for w in ["run", "do", "everything", "pipeline"]):
            return True

        return False

    def _resolve_target_from_text(self, st: Dict[str, Any], user_text: str) -> Optional[str]:
        """
        Simple resolver for planner target gating:
        - exact match
        - case-insensitive
        - underscore/space-insensitive
        """
        df_for_cols = st.get("pre_df") or st.get("clean_df")
        if not isinstance(df_for_cols, pd.DataFrame):
            return None

        cols = df_for_cols.columns.tolist()
        raw = (user_text or "").strip()

        for c in cols:
            if raw == c or raw.lower() == str(c).lower():
                return c

        def norm(x: str) -> str:
            return "".join(ch for ch in x.lower() if ch.isalnum())

        nraw = norm(raw)
        for c in cols:
            if norm(str(c)) == nraw:
                return c

        return None

    def _ask_target_for_plan(self, st: Dict[str, Any]) -> Dict[str, Any]:
        df_for_cols = st.get("pre_df") or st.get("clean_df")
        cols = df_for_cols.columns.tolist() if isinstance(df_for_cols, pd.DataFrame) else []

        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "To continue the **end-to-end run**, I need one thing:\n\n"
                    "**Which column should we predict?** (your target column)\n"
                    + (f"Columns: {cols}\n\n" if cols else "\n")
                    + "Please type the exact column name."
                ),
            }
        )
        st["stage"] = "await_plan_target"
        return st

    def _start_plan(self, user_text: str, st: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner entry:
        - Call planner to get tool-order steps
        - Store plan in state
        - Trigger graph to run step-by-step
        - If training is in plan and target missing, graph will set require_target_col=True
        """
        if st.get("clean_df") is None:
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I can run the full **end-to-end AutoML pipeline**, "
                        "but first I need a dataset. Please upload a CSV."
                    ),
                }
            )
            return st

        try:
            plan = make_plan(user_text)  # {"steps":[...], "reason":"..."}
        except Exception:
            plan = {"steps": ["preprocess_data", "train_baselines", "tune_best_model_optuna"], "reason": ""}

        steps = plan.get("steps") or []
        st["last_plan"] = plan

        pretty = []
        for i, sname in enumerate(steps, 1):
            if sname == "preprocess_data":
                pretty.append(f"{i}. Preprocess / clean the data")
            elif sname == "train_baselines":
                pretty.append(f"{i}. Train baseline models")
            elif "tune_best_model" in sname:
                pretty.append(f"{i}. Tune the best model")
            else:
                pretty.append(f"{i}. {sname}")

        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "Got it — I’ll run this as a **full AutoML workflow**.\n\n"
                    f"**Plan:**\n" + ("\n".join(pretty) if pretty else "1. (No valid steps detected)") +
                    "\n\nI’ll start now and pause only if I need your target column."
                ),
            }
        )

        # Save plan for the graph to execute
        st["plan_steps"] = steps
        st["plan_index"] = 0
        st["plan_active"] = True
        st["plan_done"] = False
        st["require_target_col"] = False

        # Clear manual tuning chat so planner owns the flow
        st["tuning_stage"] = None
        st["tuning_offered"] = False

        out = run_automl_graph(st)

        if out.get("require_target_col"):
            out = self._ask_target_for_plan(out)

        if out.get("plan_done"):
            try:
                summary_q = (
                    "Please summarize, in friendly plain English, what preprocessing, training, "
                    "and tuning (if any) were done, and highlight the best model and key metrics."
                )
                summary = self._qa_answer(summary_q, out)
            except Exception:
                summary = "End-to-end run finished. Please review the results panel."
            out["messages"].append(
                {"role": "assistant", "content": f"🧾 **End-to-end run summary**\n\n{summary}"}
            )

        return out

    def _continue_plan_after_target(self, user_text: str, st: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called when planner paused for target column.
        """
        target = self._resolve_target_from_text(st, user_text)
        if not target:
            df_for_cols = st.get("pre_df") or st.get("clean_df")
            cols = df_for_cols.columns.tolist() if isinstance(df_for_cols, pd.DataFrame) else []
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "I couldn’t match that to a column. "
                        "Please type the exact target column name.\n\n"
                        + (f"Columns: {cols}" if cols else "")
                    ),
                }
            )
            st["stage"] = "await_plan_target"
            return st

        st["target_col"] = target

        # Detect task type early so training runs smoothly
        df_for_train = st.get("pre_df") or st.get("clean_df")
        if isinstance(df_for_train, pd.DataFrame):
            try:
                st["task_type"] = st.get("task_type") or choose_task_type(df_for_train[target])
            except Exception:
                st.setdefault("task_type", "classification")

        st["require_target_col"] = False
        st["stage"] = "preview_download"
        st["messages"].append(
            {"role": "assistant", "content": f"Perfect — using **{target}** as the target. Continuing the plan…"}
        )

        out = run_automl_graph(st)

        if out.get("require_target_col"):
            out = self._ask_target_for_plan(out)

        if out.get("plan_done"):
            try:
                summary_q = (
                    "Please summarize, in friendly plain English, what preprocessing, training, "
                    "and tuning (if any) were done, and highlight the best model and key metrics."
                )
                summary = self._qa_answer(summary_q, out)
            except Exception:
                summary = "End-to-end run finished. Please review the results panel."
            out["messages"].append(
                {"role": "assistant", "content": f"🧾 **End-to-end run summary**\n\n{summary}"}
            )

        return out

    # -------------------- Router for user free text --------------------
    def handle(self, user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conversation brain:
        - Decodes user intent.
        - Existing wizard + tuning flow remains.
        - NEW: planner for end-to-end requests, graph executes plan step-by-step.
        """
        st = state.copy()
        st["last_bot"] = None

        stage = st.get("stage", "await_upload")
        tuning_stage = st.get("tuning_stage")
        text = (user_text or "").strip().lower()

        # ✅ If planner is waiting for target, treat this as target input
        if st.get("require_target_col") or stage == "await_plan_target":
            return self._continue_plan_after_target(user_text, st)

        # ✅ Planner trigger for end-to-end requests
        if self._looks_like_full_pipeline(text):
            return self._start_plan(user_text, st)

        def wants_preview(t: str) -> bool:
            preview_words = ["preview", "show preview", "see preview", "download", "show data", "see data", "show table"]
            return any(w in t for w in preview_words)

        def wants_continue(t: str) -> bool:
            continue_words = ["continue", "next", "keep going", "proceed", "go on", "carry on", "continue preprocessing"]
            return any(w in t for w in continue_words)

        def wants_train(t: str) -> bool:
            strong_phrases = [
                "go to training", "go to the training", "go to the training part", "training part",
                "model training", "start training", "start the training", "run training", "run the training",
                "train baselines", "train the model", "move to training", "move forward with training",
                "move forward with the model training", "proceed to training", "proceed to the training",
                "proceed to train", "go for training",
            ]
            if any(p in t for p in strong_phrases):
                return True
            if ("train" in t or "training" in t) and any(v in t for v in ["go", "start", "run", "move", "proceed", "forward", "jump"]):
                return True
            return t in {"train", "training"}

        def wants_tune(t: str) -> bool:
            return any(w in t for w in ["tune", "tuning", "optimize", "improve model", "hyperparameter"])

        # -------- GLOBAL handling of direct "tune" requests --------
        if wants_tune(text) and tuning_stage is None:
            if not st.get("train_result"):
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Hyperparameter tuning comes **after training**.\n\n"
                            "Please first choose a **target column** and click **Train baselines** "
                            "in the training panel below. Once training finishes, "
                            "you can say **tune the model** again and I’ll start the tuning process."
                        ),
                    }
                )
                return st

            task = st.get("task_type", "classification")
            suggested = "f1" if task == "classification" else "r2"

            method, reason = self._recommend_tuning(st)
            methods_brief = self._tuning_methods_brief()
            st["chosen_tune_method"] = method
            st["tuning_stage"] = "choose_metric"
            human_label = "Bayesian optimization" if method == "bayesian" else "Random search"

            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Great, we can proceed with **hyperparameter tuning**.\n\n"
                        f"Here’s a quick comparison of tuning methods:\n\n{methods_brief}\n\n"
                        f"For your situation, I’d recommend **{human_label}** — {reason}\n\n"
                        f"I’ll use safe defaults and **{human_label}**.\n\n"
                        f"Which **metric** should I optimize? (e.g., **{suggested}**)\n"
                        "- Classification: `f1`, `accuracy`, `precision`, `recall`\n"
                        "- Regression: `r2`, `rmse`, `mae`\n\n"
                        "_You can also say **random search** if you prefer that method._"
                    ),
                }
            )
            return st

        # -------------------- TUNING: consent → metric → auto-run --------------------
        if tuning_stage == "ask_consent":
            if "preprocess" in text or "pre-processing" in text:
                st["tuning_stage"] = None
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Sure 👍 we can go back to **preprocessing**.\n"
                            "Use the preprocessing controls above to adjust your data, then "
                            "you can retrain the baselines when you’re ready."
                        ),
                    }
                )
                st["stage"] = "prep_menu"
                st["show_only_preview"] = False
                return st

            if self._looks_like_qa(text):
                answer = self._qa_answer(user_text, st)
                st["messages"].append({"role": "assistant", "content": answer})
                st["suppress_preview_once"] = True
                return st

            yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure"}
            no_words = {"no", "n", "nope", "not now", "later"}

            if text in no_words:
                st["tuning_stage"] = None
                st["messages"].append(
                    {"role": "assistant", "content": "No worries 👍 We’ll keep the current baseline model. Say **tune the model** anytime."}
                )
                return st

            if text in yes_words or wants_tune(text):
                task = st.get("task_type", "classification")
                suggested = "f1" if task == "classification" else "r2"
                st["tuning_stage"] = "choose_metric"

                method, reason = self._recommend_tuning(st)
                methods_brief = self._tuning_methods_brief()
                st["chosen_tune_method"] = method
                human_label = "Bayesian optimization" if method == "bayesian" else "Random search"

                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Here’s a quick comparison of tuning methods:\n\n{methods_brief}\n\n"
                            f"For your situation, I’d recommend **{human_label}** — {reason}\n\n"
                            f"Great! I’ll use safe defaults and **{human_label}**.\n\n"
                            f"Which **metric** should I optimize? (e.g., **{suggested}**)\n"
                            "- Classification: `f1`, `accuracy`, `precision`, `recall`\n"
                            "- Regression: `r2`, `rmse`, `mae`\n\n"
                            "_You can also say **random search** if you prefer that method._"
                        ),
                    }
                )
                return st

            st["messages"].append(
                {"role": "assistant", "content": "Please reply with **yes** to tune the model, or **no** to keep the baseline."}
            )
            return st

        if tuning_stage == "choose_metric":
            if "random" in text:
                st["chosen_tune_method"] = "random_search"
            elif "bayes" in text or "bayesian" in text:
                st["chosen_tune_method"] = "bayesian"

            task = st.get("task_type", "classification")
            metric = self._parse_metric(text, task)
            if metric:
                st["tune_metric"] = metric
                st["chosen_tune_method"] = st.get("chosen_tune_method") or "bayesian"

                st["want_tune"] = True
                st["approved"] = True
                out = run_automl_graph(st)

                if out.get("tuned_result"):
                    tr = out["tuned_result"]
                    best_params = tr.get("best_params", {})
                    test_metrics = tr.get("test_metrics", {})
                    method_label = "Bayesian optimization" if out.get("chosen_tune_method") == "bayesian" else "Random search"
                    out["messages"].append(
                        {
                            "role": "assistant",
                            "content": (
                                f"✅ **Tuning complete** with **{method_label}** optimizing **{metric}**.\n\n"
                                f"**Best params:** `{best_params}`\n\n"
                                f"**Test metrics:** `{test_metrics}`\n\n"
                                "You can now **download the best model** from the panel or ask me to explain the results."
                            ),
                        }
                    )
                else:
                    out["messages"].append(
                        {
                            "role": "assistant",
                            "content": (
                                "I tried to run tuning, but it didn’t complete. "
                                "Please check the **errors/history** panel above, or try again."
                            ),
                        }
                    )

                out["tuning_stage"] = None
                return out

            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Please tell me **one metric** to optimize:\n"
                        "- Classification: `f1`, `accuracy`, `precision`, or `recall`\n"
                        "- Regression: `r2`, `rmse`, or `mae`\n"
                        "You can also type **random search** to change the tuning method."
                    ),
                }
            )
            return st

        # -------------------- Tuning method chooser (legacy support) --------------------
        if tuning_stage == "choose_method":
            chosen = None
            if "bayes" in text or "bayesian" in text:
                chosen = "bayesian"
            elif "random" in text:
                chosen = "random_search"

            if chosen is not None:
                st["chosen_tune_method"] = chosen
                st["tuning_stage"] = "choose_metric"
                task = st.get("task_type", "classification")
                suggested = "f1" if task == "classification" else "r2"
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Great — we’ll use **{'Bayesian optimization' if chosen=='bayesian' else 'Random search'}**.\n\n"
                            f"Which **metric** should I optimize? (e.g., **{suggested}**)"
                        ),
                    }
                )
                return st

            if wants_tune(text):
                brief = self._tuning_methods_brief()
                st["messages"].append(
                    {"role": "assistant", "content": f"Please pick one option.\n\n{brief}\nType **Bayesian optimization** or **Random search**."}
                )
                return st

        # -------------------- Preprocessing / navigation logic --------------------
        if stage == "ask_preprocess":
            if wants_train(text):
                st["stage"] = "preview_download"
                st["show_only_preview"] = False
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Got it 👍 we’ll move towards training.\n\n"
                            "Below is a quick preview of your data and the **Train baselines** controls. "
                            "You can still adjust preprocessing later if needed."
                        ),
                    }
                )
                return st

            yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure"}
            if text in yes_words:
                st["stage"] = "prep_menu"
                st["show_only_preview"] = False
                st["messages"].append({"role": "assistant", "content": self._menu_message(st)})
            else:
                st["stage"] = "preview_download"
                st["show_only_preview"] = True
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Okay, we’ll skip preprocessing for now. Say **preprocess** anytime to begin.\n\n"
                            "Here’s the **data preview** and a **download option** below.\n"
                            "Would you like to **continue preprocessing** or **proceed to training**?"
                        ),
                    }
                )
            return st

        if stage in {"prep_menu", "preview_download"}:
            if wants_train(text):
                st["stage"] = "preview_download"
                st["show_only_preview"] = False
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Sure ✅ jumping ahead.\n"
                            "Below you’ll see a data preview and the **Train baselines** section "
                            "to start training your models."
                        ),
                    }
                )
                return st

            if "preprocess" in text and stage != "prep_menu":
                st["stage"] = "prep_menu"
                st["show_only_preview"] = False
                st["messages"].append({"role": "assistant", "content": self._menu_message(st)})
                return st

            if wants_preview(text):
                st["stage"] = "preview_download"
                st["show_only_preview"] = True
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "Here’s the **data preview** and a **download option** below.\n\n"
                            "Would you like to **continue preprocessing** or **proceed to training**? "
                            "_(You can run tuning after training.)_"
                        ),
                    }
                )
                return st

            if wants_continue(text) or text in {"yes", "y"}:
                st["stage"] = "prep_menu"
                st["show_only_preview"] = False
                st["messages"].append({"role": "assistant", "content": self._menu_message(st)})
                return st

            if "missing" in text:
                st["stage"] = "prep_missing"
                st["messages"].append({"role": "assistant", "content": self._missing_intro(st)})
                return st
            if "duplicate" in text or "duplicates" in text:
                st["stage"] = "prep_duplicates"
                st["messages"].append({"role": "assistant", "content": self._dups_intro(st)})
                return st
            if "type" in text or "dtype" in text:
                st["stage"] = "prep_dtypes"
                st["messages"].append({"role": "assistant", "content": self._dtypes_intro(st)})
                return st
            if "drop" in text and "nan" in text:
                st["stage"] = "prep_drop_all_nan"
                st["messages"].append({"role": "assistant", "content": self._drop_all_nan_intro(st)})
                return st
            if "rename" in text:
                st["stage"] = "prep_rename"
                st["messages"].append({"role": "assistant", "content": self._rename_intro(st)})
                return st

            if stage == "prep_menu":
                st["messages"].append({"role": "assistant", "content": self._menu_message(st)})
            else:
                st["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Say **preprocess** to start, **preview** to see the data, or **train** to move towards training.",
                    }
                )
            return st

        # In method-specific stages, buttons drive most changes, but allow jumps
        if wants_train(text):
            st["stage"] = "preview_download"
            st["show_only_preview"] = False
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "No problem, we can move on.\n"
                        "Here’s a preview plus the **Train baselines** controls so you can start training."
                    ),
                }
            )
            return st

        if wants_preview(text):
            st["stage"] = "preview_download"
            st["show_only_preview"] = True
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Here’s the **data preview** and a **download option** below.\n\n"
                        "Would you like to **continue preprocessing** or **proceed to training**? "
                        "_(You can run tuning after training.)_"
                    ),
                }
            )
            return st

        if wants_continue(text) or text in {"menu", "back", "go back"}:
            st["stage"] = "prep_menu"
            st["show_only_preview"] = False
            st["messages"].append({"role": "assistant", "content": self._menu_message(st)})
            return st

        # -------------------- QA fallback for workflow questions --------------------
        if self._looks_like_qa(text):
            answer = self._qa_answer(user_text, st)
            st["messages"].append({"role": "assistant", "content": answer})
            st["suppress_preview_once"] = True
            return st

        # Generic fallback hint (non-question, non-action)
        st["messages"].append(
            {
                "role": "assistant",
                "content": "Use the controls above, then click **Done** (or type **preview**, **train**, or **continue**).",
            }
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
