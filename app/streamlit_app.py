# app/streamlit_app.py
from dotenv import load_dotenv
load_dotenv()

import io
import uuid
import pickle
import json
import streamlit as st
import pandas as pd

from app.agents.nodes import preprocess_tool, baseline_training_tool, choose_task_type  # kept for compatibility (not directly used now)
from app.core.preprocessing import coerce_nulls, missing_report, drop_all_nan_cols, dtypes_dict
from app.core.tuning import tune_with_optuna, tune_with_random_search
from app.core.utils import best_model_by_task

import numpy as np
import shap
import matplotlib.pyplot as plt
from collections import Counter

# LangGraph orchestration (enhanced with Supervisor + HITL)
from app.agents.graph import build_automl_graph

# --- Conversational, step-by-step navigation helpers ---
STEPS = ["1) Upload", "2) Inspect", "3) Clean", "4) Preprocess", "5) Train", "6) Tune", "7) Results"]

def goto(step_label: str):
    # Defer nav change to next run (before widgets are built)
    st.session_state["pending_nav"] = step_label
    st.rerun()

# Stable thread id (harmless even without a checkpointer)
st.session_state.setdefault("thread_id", str(uuid.uuid4()))

# Build the AutoML graph app (uses LangGraph if available, else a safe fallback)
app = build_automl_graph()

st.set_page_config(page_title="AutoML Agent", layout="wide", page_icon="🧠")

# -----------------------------
# Navigation state hygiene (must run BEFORE any widgets that use nav)
# -----------------------------
st.session_state.setdefault("nav", STEPS[0])
# Apply any deferred navigation set in a previous run
if st.session_state.get("pending_nav"):
    st.session_state["nav"] = st.session_state.pop("pending_nav")
# Radio uses a different key to avoid mutation errors
st.session_state.setdefault("nav_radio", st.session_state["nav"])

# -----------------------------
# Sidebar (global navigation)
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 AutoML Agent")
    st.caption("Upload → Inspect → Clean → Preprocess → Train → Tune → Results")

    # NOTE: use nav_radio, then mirror back to canonical nav
    step = st.radio("Navigation", STEPS, index=STEPS.index(st.session_state["nav"]), key="nav_radio")
    st.session_state["nav"] = st.session_state["nav_radio"]

    st.divider()
    with st.expander("Session Summary", expanded=False):
        st.write("**Data loaded:**", "✅" if st.session_state.get("raw_df") is not None else "❌")
        st.write("**Preprocessed:**", "✅" if st.session_state.get("pre_df") is not None else "❌")
        st.write("**Task type:**", st.session_state.get("task_type", "—"))
        if st.session_state.get("train_result") is not None:
            st.write("**Models trained:**", len(st.session_state["train_result"]["results"]))
            st.write("**Best baseline:**", st.session_state.get("best_model_name", "—"))
    st.caption("Tip: you can jump between steps anytime.")

# -----------------------------
# Initialize state
# -----------------------------
for k, v in {
    "raw_df": None,
    "clean_df": None,
    "pre_df": None,
    "task_type": None,
    "train_result": None,
    "best_model_name": None,
    "best_model_row": None,
    "target_col": None,
    "want_train": False,
    "want_tune": False,
}.items():
    st.session_state.setdefault(k, v)

# Graph state (persist Supervisor/HITL info between invokes)
st.session_state.setdefault("graph_state", {})

st.title("AutoML Agent")

# helper pills
def status_pills():
    c = st.columns(4)
    c[0].metric("Data", "Ready" if st.session_state.raw_df is not None else "Missing")
    c[1].metric("Preprocessed", "Yes" if st.session_state.pre_df is not None else "No")
    tr = st.session_state.train_result is not None
    c[2].metric("Baseline", "Done" if tr else "Pending")
    c[3].metric("Tuning", "Available" if tr else "—")

status_pills()
st.divider()

# -----------------------------
# Helpers for training lifecycle
# -----------------------------
def _reset_training_state_for_new_target():
    """
    Clear all training- and approval-related state so selecting a new target
    always runs a clean training session.
    """
    st.session_state["train_result"] = None
    st.session_state["best_model_name"] = None
    st.session_state["best_model_row"] = None
    st.session_state["want_train"] = False
    st.session_state["want_tune"] = False
    st.session_state["tuned_result"] = None
    # Clear supervisor gates
    gs = st.session_state.get("graph_state", {}).copy()
    gs["approved"] = False
    gs["require_approval"] = False
    gs["history"] = []
    gs["errors"] = []
    st.session_state["graph_state"] = gs

def _mark_just_finished_training():
    st.session_state["just_finished_training"] = True

# -----------------------------
# Graph runner with state sync (passes config; falls back if runner ignores it)
# -----------------------------
def run_graph(extra: dict):
    """
    Invoke the graph with current session artifacts + any extra overrides.
    """
    base = {
        "clean_df": st.session_state.get("clean_df"),
        "pre_df": st.session_state.get("pre_df"),
        "target_col": st.session_state.get("target_col"),
        "task_type": st.session_state.get("task_type"),
        "want_preprocess": st.session_state.get("pre_df") is None,  # only preprocess if needed
        "want_train": st.session_state.get("want_train", False),
        "want_tune": st.session_state.get("want_tune", False),
        "train_result": st.session_state.get("train_result"),
        "tuned_result": st.session_state.get("tuned_result"),
    }

    state_in = {**base, **st.session_state.get("graph_state", {}), **(extra or {})}

    cfg = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "checkpoint_ns": "automl",
        }
    }

    try:
        out = app.invoke(state_in, config=cfg)
    except TypeError:
        out = app.invoke(state_in)

    # Sync back
    st.session_state["graph_state"] = out
    st.session_state.pre_df = out.get("pre_df", st.session_state.pre_df)
    st.session_state.task_type = out.get("task_type", st.session_state.get("task_type"))
    st.session_state.train_result = out.get("train_result", st.session_state.get("train_result"))
    st.session_state.tuned_result = out.get("tuned_result", st.session_state.get("tuned_result"))

    # If training completed, clear the approval gate
    if out.get("train_result") is not None:
        st.session_state["want_train"] = False
        gs = st.session_state.get("graph_state", {}).copy()
        gs["approved"] = False
        gs["require_approval"] = False
        st.session_state["graph_state"] = gs

    return out

with st.sidebar:
    st.divider()
    st.markdown("### Supervisor")
    gs = st.session_state.get("graph_state", {})
    st.caption(gs.get("supervisor_reason", "—"))

    if st.session_state.get("rejection_notice"):
        st.warning(st.session_state["rejection_notice"])

    if gs.get("require_approval", False) and not gs.get("approved", False):
        colA, colB = st.columns(2)
        with colA:
            if st.button("Approve", key="approve_supervisor"):
                st.session_state["want_train"] = True
                with st.spinner("Running... ⏳"):
                    out = run_graph({"approved": True, "want_train": True})
                if out.get("train_result") is not None:
                    _mark_just_finished_training()
                    st.session_state["pending_nav"] = "5) Train"
                st.rerun()
        with colB:
            if st.button("Reject", key="reject_supervisor"):
                st.session_state["want_train"] = False
                st.session_state["rejection_notice"] = (
                    "In the case of tuning, baseline training is required."
                )
                run_graph({"approved": False, "want_train": False})
                st.rerun()

    with st.expander("Trace", expanded=False):
        st.write(gs.get("history", []))
    if gs.get("errors"):
        st.error(gs["errors"])

# -----------------------------
# Step 1: Upload
# -----------------------------
if st.session_state["nav"].startswith("1"):
    st.header("Hello! Let’s start by uploading your CSV")
    st.write("Once it loads, click **Next** to continue.")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], help="Drag & drop or browse")

    if uploaded is not None:
        # size warning
        file_size_mb = uploaded.size / (1024 * 1024)
        if file_size_mb > 200:
            st.warning(f"File is large ({file_size_mb:.1f} MB). Consider sampling or chunked loading.")

        st.session_state.raw_df = pd.read_csv(uploaded)
        st.success(f"Loaded dataset with shape {st.session_state.raw_df.shape}")

        with st.expander("Peek raw data (first 5)", expanded=True):
            st.dataframe(st.session_state.raw_df.head(5), use_container_width=True)

        st.button("Next: Inspect data", on_click=goto, args=("2) Inspect",))

# -----------------------------
# Step 2: Inspect
# -----------------------------
elif st.session_state["nav"].startswith("2"):
    st.header("Quick look at your data")
    if st.session_state.raw_df is None:
        st.warning("Please upload a CSV in step 1.")
    else:
        st.write("Great! Here’s a quick peek. If everything looks okay, click **Next** to do a basic clean.")

        t1, t2 = st.tabs(["Preview (first 5)", "Dtypes"])
        with t1:
            st.dataframe(st.session_state.raw_df.head(5), use_container_width=True)
        with t2:
            st.json(dtypes_dict(st.session_state.raw_df))

        st.markdown("---")
        st.button("Next: Basic clean", on_click=goto, args=("3) Clean",))

# -----------------------------
# Step 3: Clean
# -----------------------------
elif st.session_state["nav"].startswith("3"):
    st.header("Basic Clean")
    if st.session_state.raw_df is None:
        st.warning("Please upload a CSV in step 1.")
    else:
        df_clean = coerce_nulls(st.session_state.raw_df.copy())
        rep = missing_report(df_clean)

        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            st.subheader("Missing values by column")
            st.json(rep["missing_by_column"])
        with c2:
            st.subheader("All-NaN columns")
            if rep["all_nan_columns"]:
                st.write(rep["all_nan_columns"])
            else:
                st.info("No all-NaN columns found.")

        st.subheader("Drop all-NaN columns")
        cols_to_drop = st.multiselect(
            "Select columns to drop (detected as completely NaN)",
            options=rep["all_nan_columns"], default=rep["all_nan_columns"],
        )
        if st.button("Apply drop"):
            df_clean = drop_all_nan_cols(df_clean, cols_to_drop)
            st.success("Applied column drop.")
        st.session_state.clean_df = df_clean

        with st.expander("Preview after cleaning", expanded=False):
            st.dataframe(st.session_state.clean_df.head(8), use_container_width=True)

        st.markdown("---")
        st.write("**Would you like to preprocess the data now?**")
        colA, colB = st.columns(2)
        with colA:
            st.button("Yes, go to Preprocess", on_click=goto, args=("4) Preprocess",))
        with colB:
            st.button("Skip for now (go to Train)", on_click=goto, args=("5) Train",))

# -----------------------------
# Step 4: Preprocess (ENHANCED; now via Supervisor/HITL graph)
# -----------------------------
elif st.session_state["nav"].startswith("4"):
    st.header("Preprocess (enhanced)")
    if st.session_state.clean_df is None:
        st.warning("Run step 3 (Clean) first.")
    else:
        st.write(
            "Awesome! Let’s get your data ready. Choose strategies below and click "
            "**Run Enhanced Preprocess**. When it’s done, you can jump straight to **Train**."
        )

        df_clean = st.session_state.clean_df

        # ---- Quick diagnostics (tables)
        cdup, cmiss, cdtype = st.tabs(["Duplicate rows", "Missing by column", "Column dtypes"])
        with cdup:
            dup_count = int(len(df_clean) - len(df_clean.drop_duplicates()))
            st.caption("Number of duplicate rows detected in the current dataset.")
            st.dataframe(pd.DataFrame([{"duplicate_rows": dup_count}]), use_container_width=True)

        with cmiss:
            miss_by_col = df_clean.isna().sum()
            miss_df = pd.DataFrame(
                {
                    "column": miss_by_col.index,
                    "missing_count": miss_by_col.values,
                    "missing_%": (miss_by_col.values / max(len(df_clean), 1) * 100).round(2),
                }
            )
            st.caption("Count and percentage of missing values per column.")
            st.dataframe(miss_df.sort_values("missing_count", ascending=False), use_container_width=True)

        with cdtype:
            dtypes_df = pd.DataFrame({"column": df_clean.columns, "dtype": df_clean.dtypes.astype(str).values})
            st.caption("Current inferred pandas dtypes (pre-preprocessing).")
            st.dataframe(dtypes_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Supervisor setup — guided choices")

        # ---------- STATE: stable, row-IDs so rows don't disappear ----------
        import uuid as _uuid
        def _new_id(): return str(_uuid.uuid4())[:8]

        st.session_state.setdefault("pre_cols_to_drop", [])
        st.session_state.setdefault("ui_mv_rows", [])     # [{id, col, strategy}]
        st.session_state.setdefault("ui_map_rows", [])    # [{id, col, rename}]
        st.session_state.setdefault("ui_type_rows", [])   # [{id, col, type}]

        # Helpers
        miss_cols = miss_df.loc[miss_df["missing_count"] > 0, "column"].astype(str).tolist() if len(df_clean) else []
        all_cols = df_clean.columns.astype(str).tolist()
        allowed_mv = ["mean", "median", "mode", "drop", "fill"]
        allowed_types = ["int", "float", "boolean", "timestamp", "string"]

        # --- Q1: Duplicate strategy
        st.markdown("**Q1. Would you like to apply a duplicate-rows strategy?**")
        apply_dup = st.radio("Apply duplicate strategy?", ["No", "Yes"], index=0, horizontal=True, key="ui_apply_dup")
        dup_strategy = None
        if apply_dup == "Yes":
            dup_strategy = st.selectbox("Duplicate strategy", ["drop", "keep_first", "keep_last", "mark"], index=0,
                                        help="Choose how to handle duplicate rows.", key="ui_dup_strategy")
        else:
            st.caption("Skipping duplicate strategy.")
        st.markdown("---")

        # --- Q2: Missing value strategy (per-column)
        st.markdown("**Q2. Would you like to apply a missing value strategy (per-column)?**")
        if len(miss_cols) == 0:
            st.info("No columns with missing values — skipping this question.")
            apply_mv = "No"
        else:
            apply_mv = st.radio("Apply missing value strategies?", ["No", "Yes"], index=0, horizontal=True, key="ui_apply_mv")

        if apply_mv == "Yes":
            st.caption("Add one or more column → strategy pairs.")
            if not st.session_state["ui_mv_rows"]:
                st.session_state["ui_mv_rows"] = [{"id": _new_id(), "col": (miss_cols[0] if miss_cols else None), "strategy": allowed_mv[0]}]

            # Render each row (DO NOT drop partial rows on rerun)
            for i, row in enumerate(st.session_state["ui_mv_rows"]):
                rid = row["id"]
                cols = st.columns([2, 2, 1])
                with cols[0]:
                    st.session_state["ui_mv_rows"][i]["col"] = st.selectbox(
                        f"Column (row {i+1})", options=miss_cols,
                        index=(miss_cols.index(row["col"]) if row["col"] in miss_cols else 0),
                        key=f"ui_mv_col_{rid}",
                    )
                with cols[1]:
                    st.session_state["ui_mv_rows"][i]["strategy"] = st.selectbox(
                        f"Strategy (row {i+1})", options=allowed_mv,
                        index=(allowed_mv.index(row["strategy"]) if row["strategy"] in allowed_mv else 0),
                        key=f"ui_mv_strat_{rid}",
                    )
                with cols[2]:
                    if st.button("✕", key=f"ui_mv_del_{rid}"):
                        st.session_state["ui_mv_rows"].pop(i)
                        st.rerun()

            if st.button("➕ Add more", key="ui_mv_add"):
                st.session_state["ui_mv_rows"].append(
                    {"id": _new_id(), "col": (miss_cols[0] if miss_cols else None), "strategy": allowed_mv[0]}
                )
                st.rerun()
        else:
            st.caption("Skipping missing value strategies.")
        st.markdown("---")

        # --- Q3: Column mapping (rename)
        st.markdown("**Q3. Would you like to rename any columns (column mapping)?**")
        apply_map = st.radio("Apply column mapping?", ["No", "Yes"], index=0, horizontal=True, key="ui_apply_map")

        if apply_map == "Yes":
            st.caption("Select a column to rename and provide its new name. Add multiple if needed.")
            if not st.session_state["ui_map_rows"]:
                st.session_state["ui_map_rows"] = [{"id": _new_id(), "col": (all_cols[0] if all_cols else None), "rename": ""}]

            for i, row in enumerate(st.session_state["ui_map_rows"]):
                rid = row["id"]
                cols = st.columns([2, 3, 1])
                with cols[0]:
                    st.session_state["ui_map_rows"][i]["col"] = st.selectbox(
                        f"Column (row {i+1})", options=all_cols,
                        index=(all_cols.index(row["col"]) if row["col"] in all_cols else 0),
                        key=f"ui_map_col_{rid}",
                    )
                with cols[1]:
                    st.session_state["ui_map_rows"][i]["rename"] = st.text_input(
                        f"New name (row {i+1})", value=row["rename"] or "", key=f"ui_map_name_{rid}",
                    )
                with cols[2]:
                    if st.button("✕", key=f"ui_map_del_{rid}"):
                        st.session_state["ui_map_rows"].pop(i)
                        st.rerun()

            if st.button("➕ Add more", key="ui_map_add"):
                st.session_state["ui_map_rows"].append({"id": _new_id(), "col": (all_cols[0] if all_cols else None), "rename": ""})
                st.rerun()
        else:
            st.caption("Skipping column mapping.")
        st.markdown("---")

        # --- Q4: Type overrides
        st.markdown("**Q4. Would you like to enforce specific data types (type overrides)?**")
        apply_type = st.radio("Apply type overrides?", ["No", "Yes"], index=0, horizontal=True, key="ui_apply_type")

        if apply_type == "Yes":
            st.caption("Select a column and the type to enforce. Add multiple if needed.")
            if not st.session_state["ui_type_rows"]:
                st.session_state["ui_type_rows"] = [{"id": _new_id(), "col": (all_cols[0] if all_cols else None), "type": allowed_types[0]}]

            for i, row in enumerate(st.session_state["ui_type_rows"]):
                rid = row["id"]
                cols = st.columns([2, 2, 1])
                with cols[0]:
                    st.session_state["ui_type_rows"][i]["col"] = st.selectbox(
                        f"Column (row {i+1})", options=all_cols,
                        index=(all_cols.index(row["col"]) if row["col"] in all_cols else 0),
                        key=f"ui_type_col_{rid}",
                    )
                with cols[1]:
                    st.session_state["ui_type_rows"][i]["type"] = st.selectbox(
                        f"Type (row {i+1})", options=allowed_types,
                        index=(allowed_types.index(row["type"]) if row["type"] in allowed_types else 0),
                        key=f"ui_type_type_{rid}",
                    )
                with cols[2]:
                    if st.button("✕", key=f"ui_type_del_{rid}"):
                        st.session_state["ui_type_rows"].pop(i)
                        st.rerun()

            if st.button("➕ Add more", key="ui_type_add"):
                st.session_state["ui_type_rows"].append({"id": _new_id(), "col": (all_cols[0] if all_cols else None), "type": allowed_types[0]})
                st.rerun()
        else:
            st.caption("Skipping type overrides.")
        st.markdown("---")

        # Build payloads from UI rows (DO NOT drop partial rows; only include filled pairs in dicts)
        mv_dict = None
        if apply_mv == "Yes" and st.session_state["ui_mv_rows"]:
            mv_dict = {r["col"]: r["strategy"] for r in st.session_state["ui_mv_rows"] if r.get("col") and r.get("strategy")}

        cm_dict = None
        if apply_map == "Yes" and st.session_state["ui_map_rows"]:
            cm_dict = {r["col"]: r["rename"].strip() for r in st.session_state["ui_map_rows"] if r.get("col") and (r.get("rename") or "").strip()}

        to_dict = None
        if apply_type == "Yes" and st.session_state["ui_type_rows"]:
            to_dict = {r["col"]: r["type"] for r in st.session_state["ui_type_rows"] if r.get("col") and r.get("type")}

        # Drop all-NaN recompute
        all_nan_cols_list = df_clean.columns[df_clean.isna().sum() == len(df_clean)].tolist()
        st.subheader("Drop all-NaN columns (optional)")
        cols_to_drop = st.multiselect("Select columns to drop (completely NaN)", options=all_nan_cols_list,
                                      default=all_nan_cols_list, key="pre_cols_to_drop")

        # ---------- IMPORTANT: Make mapping visible in preview ----------
        # If the user is mapping, preserve original column names so mapping keys match.
        preserve_names = True if (apply_map == "Yes" and cm_dict) else False

        # Run button
        run_pre = st.button("Run Enhanced Preprocess", type="primary")
        if run_pre:
            ds = dup_strategy if dup_strategy else "drop"

            out = run_graph({
                "drop_cols": cols_to_drop,
                "duplicate_strategy": ds,
                "missing_strategy": mv_dict,
                "column_mapping": cm_dict,
                "type_overrides": to_dict,
                "preserve_column_names": preserve_names,  # <-- key change for rename visibility
                "want_preprocess": True,
                "approved": True,
            })

            if st.session_state.pre_df is not None:
                st.success("Preprocessing complete.")
                st.caption("Preview (first 15 rows)")
                st.dataframe(out.get("pre_preview", st.session_state.pre_df.head(15)), use_container_width=True)

                with st.expander("Inferred column types"):
                    st.json(out.get("pre_col_types", {}))
                with st.expander("Type parameters (e.g., categories)"):
                    st.json(out.get("pre_type_params", {}))
                with st.expander("Processing stats"):
                    st.json(out.get("pre_stats", {}))

        # Downloads + navigation (unchanged)
        if st.session_state.pre_df is not None:
            st.download_button("Download preprocessed.csv", data=st.session_state.pre_df.to_csv(index=False),
                               file_name="preprocessed.csv", mime="text/csv")
            st.markdown("---")
            st.write("**All set! Would you like to train a baseline model now?**")
            c1, c2 = st.columns(2)
            with c1:
                st.button("Yes, go to Train", on_click=goto, args=("5) Train",))
            with c2:
                st.button("Go back to Clean", on_click=goto, args=("3) Clean",))
        else:
            st.info("Review your choices above and click **Run Enhanced Preprocess** to continue.")

# -----------------------------
# Step 5: Train (baseline) — AUTO-APPROVED
# -----------------------------
elif st.session_state["nav"].startswith("5"):
    st.header("5) Baseline Training & Test")
    if st.session_state.pre_df is None:
        st.warning("Run step 4 (Preprocess) first.")
    else:
        # (A) TRAIN (always on top): pick target, detect task, RUN directly
        st.subheader("Train (or retrain) baselines")
        target_col = st.selectbox(
            "Select target column",
            options=st.session_state.pre_df.columns.tolist(),
            key="train_target_select"
        )
        if target_col:
            prev_target = st.session_state.get("last_target_col")
            target_changed = prev_target is not None and prev_target != target_col
            st.session_state["target_col"] = target_col
            st.session_state["last_target_col"] = target_col

            y = st.session_state.pre_df[target_col]
            st.session_state.task_type = choose_task_type(y)
            task = st.session_state.task_type

            if target_changed:
                _reset_training_state_for_new_target()

            st.info(f"Auto-detected task: **{task.capitalize()}**")

            if task == "classification":
                from collections import Counter
                counts = Counter(y)
                if len(counts) > 1:
                    maj = max(counts.values())
                    min_ = min(counts.values())
                    imbalance_ratio = maj / max(min_, 1)
                    if imbalance_ratio > 10:
                        st.warning(
                            "Dataset appears **highly imbalanced** "
                            f"(max/min class count ratio ≈ {imbalance_ratio:.1f}). "
                            "SMOTE or class weighting will be applied automatically."
                        )

            if st.button("Train baselines"):
                st.session_state["want_train"] = True
                st.session_state.pop("rejection_notice", None)

                # AUTO-APPROVE: run training immediately with a spinner
                with st.spinner("Training baselines... this may take a moment ⏳"):
                    out = run_graph({
                        "want_train": True,
                        "approved": True,            # auto-approve
                        "task_type": task,
                        "target_col": target_col,
                    })
                # Mark and refresh to render results immediately (below)
                if out.get("train_result") is not None:
                    _mark_just_finished_training()
                st.rerun()

        if st.session_state.train_result is None:
            st.caption("Tip: pick a target and click **Train baselines**. Results will appear below automatically.")

        st.markdown("---")

        # (B) RESULTS (always below the training controls)
        if st.session_state.get("just_finished_training"):
            st.session_state["just_finished_training"] = False

        if st.session_state.train_result is not None:
            df = st.session_state.train_result["results"]
            st.subheader("Leaderboard — Cross-Validation & Test Metrics")
            st.caption("Includes `cv_score` and `cv_std` (from 5-fold CV) plus held-out test metrics.")
            st.dataframe(df, use_container_width=True)

            name, row = best_model_by_task(st.session_state.task_type, df)
            st.session_state.best_model_name = name
            st.session_state.best_model_row = row

            st.success(f"Recommended best model: **{name}**")
            st.json(row)

            preds = st.session_state.train_result["predictions"][name]
            with st.expander("Actual vs Predicted (first 20)"):
                st.dataframe(pd.DataFrame(preds), use_container_width=True)

            st.download_button(
                "Download best_model.pkl",
                data=pickle.dumps(st.session_state.train_result["fitted"][name]),
                file_name="best_model.pkl",
            )

            st.markdown("---")
            st.write("Would you like to try **hyperparameter tuning** for even better performance?")
            c1, c2 = st.columns(2)
            with c1:
                st.button("Yes, take me to Tuning", on_click=goto, args=("6) Tune",))
            with c2:
                st.button("No thanks, show Results", on_click=goto, args=("7) Results",))

# -----------------------------
# Step 6: Tune (user-controlled) — (keeps HITL)
# -----------------------------
elif st.session_state["nav"].startswith("6"):
    st.header("6) Hyperparameter Tuning")
    if st.session_state.train_result is None:
        st.warning("Train baselines in step 5 first.")
    else:
        st.write("Great job getting this far! Choose a tuning method below. "
                 "When tuning finishes, you can jump straight to the **Results**.")

        task = st.session_state.task_type
        X_train = st.session_state.train_result["X_train"]
        y_train = st.session_state.train_result["y_train"]
        X_test  = st.session_state.train_result["X_test"]
        y_test  = st.session_state.train_result["y_test"]

        method = st.selectbox("Tuning method", ["Bayesian (Optuna)", "Random Search"])

        # ---------- BAYESIAN (OPTUNA) ----------
        if method.startswith("Bayesian"):
            st.subheader("Bayesian (Optuna) Settings")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                trials = st.number_input("n_trials", min_value=5, max_value=1000, value=30, step=5)
            with col2:
                timeout = st.number_input("timeout (seconds, 0 = none)", min_value=0, max_value=36000, value=0, step=60)
                timeout = int(timeout) if timeout > 0 else None
            with col3:
                direction = st.selectbox("direction", ["maximize", "minimize"], index=0,
                                         help="accuracy/f1/etc → maximize; rmse/mae → minimize")
            with col4:
                seed = st.number_input("seed (TPESampler)", min_value=0, max_value=10_000, value=42, step=1)
            with col5:
                metric = st.selectbox(
                    "Objective metric",
                    ["accuracy","f1","precision","recall"] if task=="classification" else ["r2","rmse","mae"],
                    index=0
                )

            hint = "maximize" if (task=="classification" or metric=="r2") else "minimize"
            st.caption(f"For `{metric}`, the usual direction is **{hint}**.")

            base_df = st.session_state.train_result["results"].copy()
            leaderboard_models = base_df["model"].tolist()
            models_to_tune = st.multiselect("Select models to tune", options=leaderboard_models, default=leaderboard_models[:1])

            # HITL gate request for tuning
            if st.button("Run Bayesian Tuning"):
                st.session_state["want_tune"] = True
                out = run_graph({"approved": False})  # request checkpoint
                if out.get("require_approval", False) and not out.get("approved", False):
                    st.info("Supervisor requests approval to start tuning (see sidebar).")

            # Only run tuning when approved
            gs = st.session_state.get("graph_state", {})
            if gs.get("require_approval", False) and not gs.get("approved", False):
                pass  # waiting for approval
            elif st.session_state.get("want_tune", False):
                # Proceed with actual tuning
                tuned_rows, tuned_details = [], {}
                for model_name in models_to_tune:
                    try:
                        tune_res = tune_with_optuna(
                            X_train, y_train, X_test, y_test, task, model_name,
                            n_trials=int(trials), timeout=timeout, direction=direction, seed=int(seed), metric=metric
                        )
                    except Exception as e:
                        st.error(f"Tuning failed for {model_name}: {e}")
                        continue

                    st.success(f"Tuning complete for {model_name}")
                    cols = st.columns(2)
                    with cols[0]:
                        st.caption("Best parameters"); st.json(tune_res["best_params"])
                        st.caption("Validation metrics"); st.json(tune_res.get("val_metrics", {}))
                        st.caption("Objective config"); st.json(tune_res.get("objective", {}))
                    with cols[1]:
                        st.caption("Final Test metrics (held-out)"); st.json(tune_res["test_metrics"])

                    tuned_rows.append({"model": model_name, **tune_res["test_metrics"]})
                    tuned_details[model_name] = tune_res

                if tuned_rows:
                    tuned_df = pd.DataFrame(tuned_rows)
                    st.subheader("Tuned results (Test Set)")
                    st.dataframe(tuned_df, use_container_width=True)
                    best_name, best_row = best_model_by_task(task, tuned_df)
                    st.success(f"Recommended tuned model: **{best_name}**")
                    st.json(best_row)
                    st.download_button(
                        "Download tuned_best_model.pkl",
                        data=pickle.dumps(tuned_details[best_name]["fitted"]),
                        file_name="tuned_best_model.pkl",
                    )
                st.info("Optuna uses only TRAIN data internally; test metrics shown are after refitting best params on full train.")

                # Conversational navigation
                st.markdown("---")
                st.write("All done here. Where to next?")
                c1, c2 = st.columns(2)
                with c1:
                    st.button("View Results Summary", on_click=goto, args=("7) Results",))
                with c2:
                    st.button("Back to Train", on_click=goto, args=("5) Train",))

        # ---------- RANDOM SEARCH ----------
        else:
            st.subheader("Random Search Settings")
            supported = [m for m in st.session_state.train_result["results"]["model"].tolist()]
            estimator = st.selectbox("Estimator", options=supported)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_iter = st.number_input("n_iter", min_value=5, max_value=1000, value=30, step=5)
            with col2:
                cv = st.number_input("cv", min_value=2, max_value=20, value=5, step=1)
            with col3:
                rs = st.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)
            with col4:
                scoring = st.text_input("scoring", value=("f1_weighted" if task=="classification" else "r2"))

            st.caption("Parameter ranges (applies to RF/XGBoost):")
            c1, c2 = st.columns(2)
            with c1:
                md_lo = st.number_input("max_depth (min)", min_value=1, max_value=200, value=3, step=1)
                md_hi = st.number_input("max_depth (max)", min_value=1, max_value=200, value=12, step=1)
            with c2:
                ne_lo = st.number_input("n_estimators (min)", min_value=10, max_value=5000, value=100, step=10)
                ne_hi = st.number_input("n_estimators (max)", min_value=10, max_value=5000, value=600, step=10)

            # HITL gate request for tuning
            if st.button("🚀 Run Random Search"):
                st.session_state["want_tune"] = True
                out = run_graph({"approved": False})  # request checkpoint
                if out.get("require_approval", False) and not out.get("approved", False):
                    st.info("Supervisor requests approval to start tuning (see sidebar).")

            # Only run RS when approved
            gs = st.session_state.get("graph_state", {})
            if gs.get("require_approval", False) and not gs.get("approved", False):
                pass  # waiting for approval
            elif st.session_state.get("want_tune", False):
                try:
                    tune_res = tune_with_random_search(
                        X_train, y_train, X_test, y_test, task, estimator,
                        n_iter=int(n_iter), cv=int(cv), random_state=int(rs), scoring=scoring,
                        max_depth_range=(int(md_lo), int(md_hi)),
                        n_estimators_range=(int(ne_lo), int(ne_hi)),
                    )
                except Exception as e:
                    st.error(f"Random search failed for {estimator}: {e}")
                else:
                    st.success(f"Random search complete for {estimator}")
                    cols = st.columns(2)
                    with cols[0]:
                        st.caption("Best parameters"); st.json(tune_res["best_params"])
                        st.caption("Validation (CV)"); st.json(tune_res.get("val_metrics", {}))
                    with cols[1]:
                        st.caption("Final Test metrics (held-out)"); st.json(tune_res["test_metrics"])

                    tuned_df = pd.DataFrame([{"model": estimator, **tune_res["test_metrics"]}])
                    st.subheader("Tuned result (Test Set)")
                    st.dataframe(tuned_df, use_container_width=True)
                    best_name, best_row = best_model_by_task(task, tuned_df)
                    st.success(f"Recommended tuned model: **{best_name}**")
                    st.json(best_row)
                    st.download_button(
                        "Download tuned_best_model.pkl",
                        data=pickle.dumps(tune_res["fitted"]),
                        file_name="tuned_best_model.pkl",
                    )
            st.info("RandomizedSearchCV runs with your ranges; if a param doesn’t apply to the chosen estimator, it’s ignored.")

            # Conversational navigation
            st.markdown("---")
            st.write("All done here. Where to next?")
            c1, c2 = st.columns(2)
            with c1:
                st.button("View Results Summary", on_click=goto, args=("7) Results",))
            with c2:
                st.button("Back to Train", on_click=goto, args=("5) Train",))

# -----------------------------
# Step 7: Results Recap
# -----------------------------
elif st.session_state["nav"].startswith("7"):
    st.header("7) Results & Downloads")
    if st.session_state.train_result is None:
        st.warning("Train baselines in step 5 first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Baseline leaderboard (Test)")
            st.dataframe(st.session_state.train_result["results"], use_container_width=True)
            if st.session_state.best_model_row:
                st.caption("Best baseline")
                st.json(st.session_state.best_model_row)
        with c2:
            st.subheader("Tuned leaderboard")
            st.info("Run tuning in step 6 to populate this section.")

        # Downloads
        if st.session_state.best_model_name:
            st.download_button(
                "Download best_model.pkl",
                data=pickle.dumps(st.session_state.train_result["fitted"][st.session_state.best_model_name]),
                file_name="best_model.pkl",
            )

        st.markdown("---")
        # -----------------------------
        # Model Interpretability
        # -----------------------------
        st.subheader("🧠 Model Interpretability")
        if st.session_state.best_model_name:
            model = st.session_state.train_result["fitted"][st.session_state.best_model_name]
            X_test = st.session_state.train_result["X_test"]

            with st.expander("Feature Importance / SHAP Analysis", expanded=False):
                st.caption("Explains how each feature influenced predictions.")
                try:
                    model_step = model.named_steps["model"]
                    preproc = model.named_steps["pre"]

                    try:
                        feature_names = preproc.get_feature_names_out()
                    except Exception:
                        feature_names = X_test.columns.tolist()

                    # Tree-based models
                    if hasattr(model_step, "feature_importances_"):
                        importances = model_step.feature_importances_
                        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
                        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
                        st.bar_chart(imp_df.set_index("feature"))
                        st.caption("Top 20 features by importance.")
                    else:
                        # SHAP fallback (robust)
                        try:
                            sample = X_test.sample(min(1000, len(X_test)), random_state=42)
                            f = lambda X: model.predict(pd.DataFrame(X, columns=sample.columns))
                            background = sample.sample(min(100, len(sample)), random_state=42)
                            explainer = shap.KernelExplainer(f, background)
                            shap_values = explainer(sample, silent=True)
                            st.write("Feature impact visualization (SHAP summary):")
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"SHAP fallback failed: {e}")

                except Exception as e:
                    st.error(f"Interpretability failed: {e}")

        # -----------------------------
        # Conversational Navigation
        # -----------------------------
        st.markdown("---")
        st.write("Great work! Where would you like to go next?")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("Back to Tuning", on_click=goto, args=("6) Tune",))
        with c2:
            st.button("Upload another dataset", on_click=goto, args=("1) Upload",))
        with c3:
            st.button("Back to Train", on_click=goto, args=("5) Train",))
