# app/chat_app.py
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import json
import streamlit as st
import pandas as pd
import uuid
import pickle

from app.agents.chat_orchestrator import ChatOrchestrator
from app.agents.nodes import choose_task_type  # only for quick local detection
from app.agents.runner import run_automl_graph
from app.core.utils import best_model_by_task
from app.agents.llm_utils import chat_once
from app.agents.prompts import SYSTEM_QA_AGENT  # Q&A prompt
from app.agents.intent_router import IntentRouter  # NEW: lightweight intent router
from app.agents.planner import Planner


# ---------- Page ----------
st.set_page_config(page_title="Chat AutoML", layout="wide")
st.title("💬 Chat AutoML")

# ---------- Session bootstrap ----------
if "chat_state" not in st.session_state:
    st.session_state["chat_state"] = {
        "stage": "await_upload",
        "raw_df": None,
        "clean_df": None,
        "pre_df": None,
        "pre_preview": None,
        "pre_stats": None,
        "pre_col_types": None,
        "pre_type_params": None,
        "show_only_preview": False,  # show preview only when user explicitly asks
        "messages": [
            {"role": "assistant", "content": "Hi 👋 Upload a CSV to get started."}
        ],
        # preprocessing scratch
        "pp_missing_strategy": {},
        "pp_duplicate_strategy": None,
        "pp_type_overrides": {},
        "pp_drop_all_nan_cols": [],
        "pp_column_mapping": {},
        "pp_preserve_column_names": False,
        "done_missing": False,
        "done_duplicates": False,
        "done_dtypes": False,
        "done_drop_all_nan": False,
        "done_rename": False,
        # ---- Training state ----
        "target_col": None,
        "task_type": None,
        "train_result": None,
        "best_model_name": None,
        "best_model_row": None,
        # ---- Tuning state ----
        "tuned_result": None,
        "chosen_tune_method": None,
        "tune_metric": None,
        # ---- Supervisor / tools state ----
        "want_preprocess": False,
        "want_train": False,
        "want_tune": False,
        "require_approval": False,
        "approved": False,
        "supervisor_reason": "",
        "history": [],
        "errors": [],
        # One-shot flag to suppress preview on simple Q&A
        "suppress_preview_once": False,
        # LangGraph / threading hint
        "thread_id": "chat-thread",
        # Tuning conversational state
        "tuning_stage": None,
        "tuning_offered": False,
    }

orch = ChatOrchestrator()
router = IntentRouter()
planner = Planner()
S = st.session_state["chat_state"]

# ---------- Capture the ask-preprocess question to show AFTER preview ----------
PENDING_ASK_PREPROC = None
MESSAGES_TO_RENDER = S.get("messages", [])

if S.get("stage") == "ask_preprocess" and MESSAGES_TO_RENDER:
    last = MESSAGES_TO_RENDER[-1]
    if (
        last.get("role") == "assistant"
        and "proceed with **preprocessing** now? (yes / no)"
        in last.get("content", "").lower()
    ):
        PENDING_ASK_PREPROC = last
        MESSAGES_TO_RENDER = MESSAGES_TO_RENDER[:-1]

# ---------- Render history (minus the held question) ----------
for msg in MESSAGES_TO_RENDER:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- Helpers: per-step inline UI ----------
def ui_missing():
    df = S["clean_df"]
    miss = df.isna().sum()
    miss_df = (
        pd.DataFrame(
            {
                "column": miss.index,
                "missing_count": miss.values,
                "missing_%": (miss.values / max(len(df), 1) * 100).round(2),
            }
        )
        .sort_values("missing_count", ascending=False)
    )

    st.caption("Add one or more column → strategy pairs.")
    st.dataframe(miss_df, width="stretch")

    allowed = ["mean", "median", "mode", "drop", "fill"]
    st.session_state.setdefault("ui_mv_rows", [])
    if not st.session_state["ui_mv_rows"]:
        top_cols = miss_df.query("missing_count > 0")["column"].tolist()[:1]
        initial = [{"col": c, "strategy": "median"} for c in top_cols] or [
            {"col": None, "strategy": "mean"}
        ]
        st.session_state["ui_mv_rows"] = [
            {"id": str(uuid.uuid4())[:8], **r} for r in initial
        ]

    for i, row in enumerate(list(st.session_state["ui_mv_rows"])):  # copy for safe delete
        cols = st.columns([2, 2, 1])
        with cols[0]:
            choices = df.columns.tolist()
            idx = choices.index(row["col"]) if row["col"] in choices else 0
            st.session_state["ui_mv_rows"][i]["col"] = st.selectbox(
                f"Column (row {i+1})", choices, index=idx, key=f"mv_col_{row['id']}"
            )
        with cols[1]:
            sidx = allowed.index(row["strategy"]) if row["strategy"] in allowed else 0
            st.session_state["ui_mv_rows"][i]["strategy"] = st.selectbox(
                f"Strategy (row {i+1})",
                allowed,
                index=sidx,
                key=f"mv_strat_{row['id']}",
            )
        with cols[2]:
            if st.button("✕", key=f"mv_del_{row['id']}"):
                st.session_state["ui_mv_rows"].pop(i)
                st.rerun()

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Add more", key="mv_add"):
            st.session_state["ui_mv_rows"].append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "col": df.columns.tolist()[0],
                    "strategy": "mean",
                }
            )
            st.rerun()
    with colB:
        if st.button("Done", key="mv_done", type="primary"):
            mapping = {
                r["col"]: r["strategy"]
                for r in st.session_state["ui_mv_rows"]
                if r.get("col") and r.get("strategy")
            }
            st.session_state["chat_state"] = orch.apply_missing(
                st.session_state["chat_state"], mapping
            )
            st.rerun()


def ui_duplicates():
    df = S["clean_df"]
    dup_count = int(len(df) - len(df.drop_duplicates()))
    st.caption("Number of duplicate rows detected in the current dataset.")
    st.dataframe(
        pd.DataFrame([{"duplicate_rows": dup_count}]), width="stretch"
    )
    strategy = st.selectbox(
        "Duplicate strategy",
        ["drop", "keep_first", "keep_last", "mark"],
        index=0,
        help="Choose how to handle duplicate rows.",
    )
    if st.button("Done", key="dups_done", type="primary"):
        st.session_state["chat_state"] = orch.apply_duplicates(
            st.session_state["chat_state"], strategy
        )
        st.rerun()


def ui_dtypes():
    df = S["clean_df"]
    st.caption("Select a column and the type to enforce. Add multiple if needed.")
    st.dataframe(
        pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}),
        width="stretch",
    )

    allowed = ["int", "float", "boolean", "timestamp", "string"]
    st.session_state.setdefault("ui_type_rows", [])
    if not st.session_state["ui_type_rows"]:
        st.session_state["ui_type_rows"] = [
            {"id": str(uuid.uuid4())[:8], "col": df.columns.tolist()[0], "type": "int"}
        ]

    for i, row in enumerate(list(st.session_state["ui_type_rows"])):
        cols = st.columns([2, 2, 1])
        with cols[0]:
            choices = df.columns.tolist()
            idx = choices.index(row["col"]) if row["col"] in choices else 0
            st.session_state["ui_type_rows"][i]["col"] = st.selectbox(
                f"Column (row {i+1})", choices, index=idx, key=f"type_col_{row['id']}"
            )
        with cols[1]:
            sidx = allowed.index(row["type"]) if row["type"] in allowed else 0
            st.session_state["ui_type_rows"][i]["type"] = st.selectbox(
                f"Type (row {i+1})", allowed, index=sidx, key=f"type_val_{row['id']}"
            )
        with cols[2]:
            if st.button("✕", key=f"type_del_{row['id']}"):
                st.session_state["ui_type_rows"].pop(i)
                st.rerun()

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Add more", key="type_add"):
            st.session_state["ui_type_rows"].append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "col": df.columns.tolist()[0],
                    "type": "int",
                }
            )
            st.rerun()
    with colB:
        if st.button("Done", key="type_done", type="primary"):
            mapping = {
                r["col"]: r["type"]
                for r in st.session_state["ui_type_rows"]
                if r.get("col") and r.get("type")
            }
            st.session_state["chat_state"] = orch.apply_dtypes(
                st.session_state["chat_state"], mapping
            )
            st.rerun()


def ui_drop_all_nan():
    df = S["clean_df"]
    all_nan = df.columns[df.isna().sum() == len(df)].tolist()
    st.caption("Select completely-NaN columns to drop.")

    if not all_nan:
        st.info("No columns with all NaN.")
        if st.button("Skip", key="dropnan_skip"):
            st.session_state["chat_state"] = orch.apply_drop_all_nan(
                st.session_state["chat_state"], []
            )
            st.rerun()
        return

    chosen = st.multiselect(
        "Select columns to drop (completely NaN)", options=all_nan, default=all_nan
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("Apply", key="dropnan_apply", type="primary"):
            st.session_state["chat_state"] = orch.apply_drop_all_nan(
                st.session_state["chat_state"], chosen
            )
            st.rerun()
    with colB:
        if st.button("Skip", key="dropnan_skip2"):
            st.session_state["chat_state"] = orch.apply_drop_all_nan(
                st.session_state["chat_state"], []
            )
            st.rerun()


def ui_rename():
    df = S["clean_df"]
    st.caption(
        "Select a column to rename and provide its new name. Add multiple if needed."
    )
    st.session_state.setdefault("ui_map_rows", [])
    if not st.session_state["ui_map_rows"]:
        st.session_state["ui_map_rows"] = [
            {"id": str(uuid.uuid4())[:8], "old": df.columns.tolist()[0], "new": ""}
        ]

    for i, row in enumerate(list(st.session_state["ui_map_rows"])):
        cols = st.columns([2, 3, 1])
        with cols[0]:
            choices = df.columns.tolist()
            idx = choices.index(row["old"]) if row["old"] in choices else 0
            st.session_state["ui_map_rows"][i]["old"] = st.selectbox(
                f"Column (row {i+1})", choices, index=idx, key=f"map_old_{row['id']}"
            )
        with cols[1]:
            st.session_state["ui_map_rows"][i]["new"] = st.text_input(
                f"New name (row {i+1})", value=row["new"], key=f"map_new_{row['id']}"
            )
        with cols[2]:
            if st.button("✕", key=f"map_del_{row['id']}"):
                st.session_state["ui_map_rows"].pop(i)
                st.rerun()

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Add more", key="map_add"):
            st.session_state["ui_map_rows"].append(
                {"id": str(uuid.uuid4())[:8], "old": df.columns.tolist()[0], "new": ""}
            )
            st.rerun()
    with colB:
        if st.button("Done", key="map_done", type="primary"):
            mapping = {
                r["old"]: (r["new"] or "").strip()
                for r in st.session_state["ui_map_rows"]
                if r.get("old")
            }
            st.session_state["chat_state"] = orch.apply_rename(
                st.session_state["chat_state"], mapping
            )
            st.rerun()


def _df_for_training():
    """Use preprocessed data if available, otherwise the cleaned data."""
    S2 = st.session_state["chat_state"]
    df = S2.get("pre_df") if S2.get("pre_df") is not None else S2.get("clean_df")
    return df, S2


def maybe_answer_qa(user_text: str, state: dict) -> str | None:
    """
    If the user is asking a question about metrics / status / preprocessing / tuning,
    answer it via LLM using SYSTEM_QA_AGENT. Otherwise return None.
    """
    txt = (user_text or "").strip().lower()

    # 🧭 If the text clearly looks like a *navigation / action* command,
    # let the orchestrator handle it (preview, train, preprocess, tuning, etc.)
    nav_keywords = [
        "show me the data preview",
        "show me preview",
        "show the preview",
        "show preview",
        "see the preview",
        "see preview",
        "data preview",
        "preview the data",
        "show the data",
        "see the data",
        "go to training",
        "go to the training",
        "go to training part",
        "start training",
        "start the training",
        "run training",
        "run the training",
        "go to preprocess",
        "go to the preprocess",
        "go to preprocess part",
    ]

    status_prefixes = (
        "have we",
        "have i",
        "did we",
        "did i",
        "has the",
        "is ",
        "are ",
        "was ",
        "were ",
    )

    if (
        any(k in txt for k in nav_keywords)
        and not txt.startswith(status_prefixes)
        and txt.strip() not in {"preview"}
    ):
        return None

    tune_nav_keywords = [
        "tune the model",
        "start tuning",
        "go to tuning",
        "proceed with tuning",
        "begin tuning",
        "start hyperparameter tuning",
        "go to hyperparameter tuning",
    ]
    status_words = ["done", "finished", "complete", "completed", "already"]

    if any(k in txt for k in tune_nav_keywords) and not any(w in txt for w in status_words):
        return None

    q_starts = (
        "what",
        "which",
        "how",
        "why",
        "did",
        "have",
        "has",
        "is",
        "are",
        "was",
        "were",
        "can",
        "could",
        "show",
        "give",
        "tell",
        "explain",
    )
    looks_like_q = "?" in txt or txt.startswith(q_starts)

    keywords = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "r2",
        "rmse",
        "mae",
        "score",
        "metric",
        "leaderboard",
        "best model",
        "model name",
        "tune",
        "tuning",
        "hyperparameter",
        "parameters",
        "trained",
        "training",
        "preprocess",
        "preprocessing",
        "cleaned",
        "cleaning",
        "done",
        "finished",
        "steps",
        "so far",
        "result",
        "results",
        "explain",
        "explanation",
    ]
    mentions_project = any(k in txt for k in keywords)

    if not (looks_like_q or mentions_project):
        return None

    # -------- Build snapshot --------
    df_train, _ = _df_for_training()
    n_rows = int(df_train.shape[0]) if df_train is not None else 0

    best_row = state.get("best_model_row") or {}
    metric_values = {
        key: float(best_row.get(key))
        for key in ["f1", "accuracy", "precision", "recall", "r2", "rmse", "mae"]
        if best_row.get(key) is not None and pd.notnull(best_row.get(key))
    }

    cv_score = best_row.get("cv_score")
    cv_std = best_row.get("cv_std")

    leaderboard_top = None
    tr = state.get("train_result")
    if tr is not None:
        df_res = tr.get("results")
        if isinstance(df_res, pd.DataFrame) and not df_res.empty:
            leaderboard_top = df_res.head(5).to_dict(orient="records")

    tuned = state.get("tuned_result")
    tuned_metrics = tuned.get("test_metrics") if isinstance(tuned, dict) else None
    tuned_best_params = tuned.get("best_params") if isinstance(tuned, dict) else None

    snapshot = {
        "task_type": state.get("task_type"),
        "best_model_name": state.get("best_model_name"),
        "dataset_size": n_rows,
        "leaderboard_top": leaderboard_top,
        "metric_values": metric_values,
        "cv_score": float(cv_score) if cv_score is not None and pd.notnull(cv_score) else None,
        "cv_std": float(cv_std) if cv_std is not None and pd.notnull(cv_std) else None,
        "tuning_available": bool(tr is not None),
        "tuning_done": bool(tuned is not None),
        "tuned_metrics": tuned_metrics,
        "tuned_best_params": tuned_best_params,
        "preprocessing_done": bool(state.get("pre_df") is not None),
        "done_missing": state.get("done_missing"),
        "done_duplicates": state.get("done_duplicates"),
        "done_dtypes": state.get("done_dtypes"),
        "done_drop_all_nan": state.get("done_drop_all_nan"),
        "done_rename": state.get("done_rename"),
        "train_done": bool(tr is not None),
    }

    payload = {"snapshot": snapshot, "question": user_text}

    # -------- LLM Answer --------
    try:
        answer = chat_once(
            system=SYSTEM_QA_AGENT,
            user=json.dumps(payload),
            model="gpt-4o-mini",
            temperature=0.2,
        )
        return answer

    except Exception:
        # -------- Fallback QA (updated for chat-based training) --------
        if not snapshot["train_done"]:
            return (
                "You haven’t trained any models yet, so I don’t have accuracy or other metrics.\n\n"
                "Please **tell me which column should be used as the target**.\n"
                "For example:\n"
                "`Use price as target` or `Target column is loan_status`.\n\n"
                "I’ll auto-detect the task type and run **Train baselines** for you."
            )

        if not metric_values:
            return (
                "Training is completed, but I couldn't interpret the metrics.\n"
                "Please check the leaderboard displayed above."
            )

        bits = []
        if metric_values.get("accuracy") is not None:
            bits.append(f"accuracy ≈ {metric_values['accuracy']:.3f}")
        if metric_values.get("f1") is not None:
            bits.append(f"f1 ≈ {metric_values['f1']:.3f}")

        joined = ", ".join(bits) if bits else "metrics are shown in the leaderboard."

        return (
            f"The current best model is **{state.get('best_model_name','(unknown)')}**, "
            f"and {joined}."
        )




def ui_train_inline():
    """
    Training UI under preview or standalone if user skips explicit preprocessing.
    Now fully chat-driven:
      - This function ONLY shows the training section + results.
      - Target selection and the actual "train baselines" action
        are handled via the chat/orchestrator.
    """
    # Hide the entire training block while tuning chat is active
    S2 = st.session_state["chat_state"]
    if S2.get("tuning_stage") in {"ask_consent", "choose_metric", "choose_method"}:
        return

    df_for_train, S2 = _df_for_training()
    if df_for_train is None:
        st.warning("Please upload data first.")
        return

    st.markdown("### Train baselines")

    # -------------------------------------------------
    # 1) Pre-training: just show columns + chat hint
    # -------------------------------------------------
    if S2.get("train_result") is None:
        cols_df = pd.DataFrame({"column": df_for_train.columns.tolist()})
        st.caption("Available columns in your dataset")
        st.dataframe(cols_df, width="stretch")

        st.info(
            "To start training, tell me **in the chat** which column is your target.\n\n"
            "For example: `Use loan_status as my target`.\n\n"
            "I’ll automatically detect whether it’s a **classification** or "
            "**regression** problem and run the baseline training for you."
        )
        return

    # -------------------------------------------------
    # 2) Post-training: show leaderboard + downloads
    # -------------------------------------------------
    S2 = st.session_state["chat_state"]
    if S2.get("tuning_stage") in {"ask_consent", "choose_metric", "choose_method"}:
        return

    if S2.get("train_result") is not None:
        df = S2["train_result"]["results"]
        st.subheader("Leaderboard — Cross-Validation & Test Metrics")
        st.dataframe(df, width="stretch")

        name, row = best_model_by_task(S2.get("task_type", "classification"), df)
        S2["best_model_name"] = name
        S2["best_model_row"] = row
        st.session_state["chat_state"] = S2

        st.success(f"Recommended best model: **{name}**")
        st.json(row)

        # Short model explanation with tuning recommendation hint
        try:
            task_type = S2.get("task_type", "classification")
            n_rows, n_cols = df_for_train.shape
            f1 = float(row.get("f1", 0) or 0)
            # Simple quality buckets
            if f1 >= 0.9:
                quality = "excellent"
            elif f1 >= 0.8:
                quality = "very good"
            elif f1 >= 0.7:
                quality = "good"
            elif f1 >= 0.6:
                quality = "okay"
            else:
                quality = "quite low"

            user_payload = (
                f"Task type: {task_type}\n"
                f"Dataset size: {n_rows} rows × {n_cols} columns\n"
                f"Model: {name}\n"
                f"Best-row metrics JSON: {row}\n"
                f"Quality label based on F1≈{f1:.3f}: {quality}\n"
                "Explain in 4–5 short sentences:\n"
                "- how strong this performance is (e.g. 0.77 is 'good but not amazing')\n"
                "- whether the user should consider hyperparameter tuning or not,\n"
                "- and any simple next suggestions for a non-technical user."
            )

            explanation = chat_once(
                system=(
                    "You are a friendly AutoML assistant for non-technical users. "
                    "Explain model quality using intuitive language like 'okay', 'good', 'great'. "
                    "Explicitly say whether hyperparameter tuning is recommended, optional, or not needed."
                ),
                user=user_payload,
                model="gpt-4o-mini",
                temperature=0.3,
            )
        except Exception:
            explanation = (
                f"The {name} looks **{quality}** for this data (F1≈{row.get('f1', 0):.3f}, "
                f"accuracy≈{row.get('accuracy', 0):.3f}). "
                "Scores around 0.7–0.8 are generally usable but may still benefit from tuning, "
                "while 0.8+ is quite strong. Because the F1 score is below 0.80, "
                "it’s reasonable to run hyperparameter tuning if you want to squeeze out more performance."
            )
        st.markdown(f"**Why this model?** {explanation}")

        preds = S2["train_result"]["predictions"].get(name, {})
        with st.expander("Actual vs Predicted (first 20)"):
            st.dataframe(pd.DataFrame(preds), width="stretch")

        st.download_button(
            "Download best_model.pkl",
            data=pickle.dumps(S2["train_result"]["fitted"][name]),
            file_name="best_model.pkl",
        )

        # Conversational tuning prompt via orchestrator
        S2 = orch.ask_tuning_opt_in(S2)
        st.session_state["chat_state"] = S2
        if S2.get("tuning_stage") == "ask_consent":
            with st.chat_message("assistant"):
                st.markdown(
                    "Would you like me to **tune the best model’s hyperparameters** "
                    "to try and get even better performance? (yes / no)"
                )



def ui_preview_and_download():
    """
    Preview area.
    - Runs preprocessing once (via LangGraph) the first time we need a preview,
      showing a spinner while it runs.
    - Shows preview ONLY when user explicitly asked (show_only_preview == True)
      AND only before training. Never after training/tuning.
    """
    S2 = st.session_state["chat_state"]

    # If we have data but no preprocessed frame yet, run preprocessing with spinner
    if S2.get("clean_df") is not None and S2.get("pre_df") is None:
        with st.spinner("Preprocessing your data… this may take a moment ⏳"):
            st.session_state["chat_state"] = orch.run_preprocess_now(S2)
            S2 = st.session_state["chat_state"]
    else:
        st.session_state["chat_state"] = S2

    S2 = st.session_state["chat_state"]

    # Hide preview/training while tuning chat is active
    if S2.get("tuning_stage") in {"ask_consent", "choose_metric", "choose_method"}:
        return

    # Never show preview after training or tuning has been produced
    if S2.get("train_result") is not None or S2.get("tuned_result") is not None:
        ui_train_inline()
        return

    if S2.get("pre_df") is None and S2.get("clean_df") is None:
        st.info("Nothing to preview yet.")
        return

    # If a simple Q&A was just asked, skip preview once.
    if S2.get("suppress_preview_once"):
        S2["suppress_preview_once"] = False
        st.session_state["chat_state"] = S2
        return

    # 👉 Show preview ONLY if the user asked for it
    if not S2.get("show_only_preview", False):
        # User didn't ask for preview: just show training UI (before training)
        ui_train_inline()
        return

    # --- Render on-demand preview ---
    df_preview = S2["pre_df"] if S2.get("pre_df") is not None else S2.get("clean_df")
    st.caption("Preview (first 15)")
    st.dataframe(df_preview.head(15), width="stretch")

    if S2.get("pre_df") is not None:
        st.download_button(
            "Download preprocessed.csv",
            data=S2["pre_df"].to_csv(index=False),
            file_name="preprocessed.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            "Download cleaned.csv",
            data=S2["clean_df"].to_csv(index=False),
            file_name="cleaned.csv",
            mime="text/csv",
        )

    # 🔹 Planner-specific hint under the preview (now chat-based training)
    if S2.get("planner_active") and S2.get("train_result") is None:
        st.info(
            "Now kindly tell me **in the chat** which column should be used as the **target** "
            "(for example: `Use loan_status as my target`).\n\n"
            "I’ll automatically detect the task type and run **Train baselines** for you."
        )

    st.markdown("---")

    if PENDING_ASK_PREPROC is not None:
        with st.chat_message("assistant"):
            st.markdown(PENDING_ASK_PREPROC["content"])

    # When preview is requested explicitly, we can still show training UI below it (pre-training)
    if S2.get("train_result") is None and not S2.get("want_train", False):
        ui_train_inline()
    else:
        if S2.get("show_only_preview", False):
            st.write(
                "You can now **ask me in the chat to train** (by telling me your target column) "
                "or ask questions about this preview."
            )
        else:
            ui_train_inline()




# ---------- Upload gate ----------
if S.get("raw_df") is None:
    with st.chat_message("assistant"):
        uploaded = st.file_uploader("Upload a CSV to begin", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state["chat_state"] = orch.start_after_upload(df, S)
            st.rerun()
else:
    if S.get("stage") == "ask_preprocess":
        with st.chat_message("assistant"):
            st.caption(
                f"📄 Dataset loaded: {S['raw_df'].shape[0]} rows × {S['raw_df'].shape[1]} columns."
            )

    if S.get("stage") == "ask_preprocess" and S.get("clean_df") is not None:
        with st.chat_message("assistant"):
            with st.expander("Peek raw data (first 5)", expanded=True):
                st.dataframe(S["clean_df"].head(5), width="stretch")
        if PENDING_ASK_PREPROC is not None:
            with st.chat_message("assistant"):
                st.markdown(PENDING_ASK_PREPROC["content"])

    stage = S.get("stage", "ask_preprocess")

    if S.get("train_result") is None and stage == "prep_menu":
        df = S.get("clean_df")
        if df is not None:
            dup_count = int(len(df) - len(df.drop_duplicates()))
            miss = df.isna().sum()
            all_nan_cols = df.columns[df.isna().sum() == len(df)].tolist()
            has_missing = bool((miss > 0).any())
            has_all_nan = len(all_nan_cols) > 0
            has_issues = dup_count > 0 or has_missing or has_all_nan
        else:
            has_issues = False

        if has_issues:
            with st.chat_message("assistant"):
                tabs = st.tabs(
                    ["Duplicate rows", "Missing by column", "Column dtypes"]
                )
                with tabs[0]:
                    st.caption(
                        "Number of duplicate rows detected in the current dataset."
                    )
                    st.dataframe(
                        pd.DataFrame([{"duplicate_rows": dup_count}]),
                        width="stretch",
                    )
                with tabs[1]:
                    miss_df = pd.DataFrame(
                        {
                            "column": miss.index,
                            "missing_count": miss.values,
                            "missing_%": (
                                miss.values / max(len(df), 1) * 100
                            ).round(2),
                        }
                    ).sort_values("missing_count", ascending=False)
                    st.dataframe(miss_df, width="stretch")
                with tabs[2]:
                    st.dataframe(
                        pd.DataFrame(
                            {"column": df.columns, "dtype": df.dtypes.astype(str)}
                        ),
                        width="stretch",
                    )

    if stage == "prep_missing":
        with st.chat_message("assistant"):
            ui_missing()
    elif stage == "prep_duplicates":
        with st.chat_message("assistant"):
            ui_duplicates()
    elif stage == "prep_dtypes":
        with st.chat_message("assistant"):
            ui_dtypes()
    elif stage == "prep_drop_all_nan":
        with st.chat_message("assistant"):
            ui_drop_all_nan()
    elif stage == "prep_rename":
        with st.chat_message("assistant"):
            ui_rename()
    elif stage == "preview_download":
        with st.chat_message("assistant"):
            ui_preview_and_download()

# ---------- Chat input ----------
user_text = st.chat_input("Tell me what to do…")
if user_text:
    # Log user message into conversation history
    S["messages"].append({"role": "user", "content": user_text})

    # 0) If Planner is waiting for YES/NO, handle that FIRST.
    if S.get("planner_stage") == "await_confirm":
        st.session_state["chat_state"] = planner.handle_confirmation(user_text, S)
        st.rerun()

    # 1) Route the intent first (qa / simple_action / multi_step)
    intent = router.classify(user_text, S)
    S["router_intent"] = intent  # store for debugging / downstream
    kind = intent.get("kind", "simple_action")

    # 2) QA: delegate to existing maybe_answer_qa helper
    if kind == "qa":
        qa_answer = maybe_answer_qa(user_text, S)
        if qa_answer is not None:
            S["messages"].append({"role": "assistant", "content": qa_answer})

            # If this was a pure Q&A (no strong action words), suppress preview once
            q = user_text.lower()
            action_words = [
                "train",
                "training",
                "preprocess",
                "preview",
                "upload",
                "clean",
                "duplicate",
                "missing",
                "dtype",
                "type",
                "rename",
                "drop",
                "nan",
                "tune",
                "optimize",
            ]
            if not any(w in q for w in action_words):
                S["suppress_preview_once"] = True

            st.session_state["chat_state"] = S
            st.rerun()
        # if qa_answer is None, just fall through to normal handling

    # 3) Multi-step: go through Planner (plan preview + yes/no)
    if kind == "multi_step":
        st.session_state["chat_state"] = planner.handle_multi_step(user_text, S, intent)
        st.rerun()

    # 4) Simple actions: keep existing orchestrator behaviour
    # If we are in the tuning metric stage, wrap the orchestrator call in a spinner
    if S.get("tuning_stage") == "choose_metric":
        with st.spinner("Tuning the best model... this may take a moment ⏳"):
            st.session_state["chat_state"] = orch.handle(user_text, S)
    else:
        will_train = (
            S.get("tuning_stage") is None
            and orch._parse_target_from_text(user_text, S) is not None
        )
        if will_train:
            with st.spinner("Training the best model… this may take a moment ⏳"):
                st.session_state["chat_state"] = orch.handle(user_text, S)
        else:
            st.session_state["chat_state"] = orch.handle(user_text, S)

    last_bot = st.session_state["chat_state"].pop("last_bot", None)


    if last_bot:
        st.session_state["chat_state"]["messages"].append(
            {"role": "assistant", "content": last_bot}
        )

        # If the user asked a simple Q&A (no action keywords), suppress preview once
        q = user_text.lower()
        action_words = [
            "train",
            "training",
            "preprocess",
            "preview",
            "upload",
            "clean",
            "duplicate",
            "missing",
            "dtype",
            "type",
            "rename",
            "drop",
            "nan",
            "tune",
            "optimize",
        ]
        if not any(w in q for w in action_words):
            st.session_state["chat_state"]["suppress_preview_once"] = True

    st.rerun()

