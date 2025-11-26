# app/agents/pipeline_planner.py
from __future__ import annotations

from typing import Dict, Any, List, Literal, Optional, TypedDict
import json

import pandas as pd
import numpy as np

from app.agents.llm_utils import chat_once
from app.core.preprocessing import missing_report, dtypes_dict


# -----------------------------
# Types
# -----------------------------
StepKind = Literal["preprocess", "train", "tune"]


class PlanStep(TypedDict, total=False):
    id: str
    kind: StepKind
    description: str
    auto: bool
    requires_user_input: bool


class PreprocessConfig(TypedDict, total=False):
    duplicate_strategy: str
    drop_all_nan_cols: List[str]
    missing_strategy: Dict[str, str]
    type_overrides: Dict[str, str]
    column_mapping: Dict[str, str]


class PipelinePlan(TypedDict, total=False):
    goal: str
    steps: List[PlanStep]
    preprocess_config: PreprocessConfig
    summary_markdown: str
    needs_user_approval: bool


# ============================================================
#  LLM system prompt for missing-value strategy planner
# ============================================================
_SYSTEM_PREPROCESS_PLANNER = (
    "You are an AutoML preprocessing planner.\n"
    "Your job is to propose simple default strategies for handling missing values.\n\n"
    "You will receive a JSON payload with:\n"
    "  - columns: list of column names\n"
    "  - dtypes: mapping column -> dtype string\n"
    "  - missing_by_column: mapping column -> missing count\n"
    "  - n_rows: number of rows in the dataset\n\n"
    "For each column that has missing values, choose exactly one strategy from:\n"
    "  - mean\n"
    "  - median\n"
    "  - mode\n"
    "  - fill\n"
    "  - drop\n\n"
    "Rules of thumb:\n"
    "  - Numeric columns: mean/median; median for skewed values.\n"
    "  - Datetime columns: often drop rows with missing timestamps.\n"
    "  - Categorical/text: mode for low cardinality, else fill.\n"
    "  - Avoid dropping columns unless the column is entirely missing.\n\n"
    "Return ONLY a JSON object such as:\n"
    "{ \"column_a\": \"mean\", \"column_b\": \"drop\" }\n"
    "No explanations. Only JSON."
)


# ============================================================
#  Missing-strategy planner (LLM + fallback)
# ============================================================
def _llm_missing_strategy(df: pd.DataFrame) -> Dict[str, str]:
    miss = missing_report(df)
    missing_by_col = miss["missing_by_column"]

    cols_with_missing = [c for c, v in missing_by_col.items() if v and v > 0]
    if not cols_with_missing:
        return {}

    payload = {
        "columns": cols_with_missing,
        "dtypes": dtypes_dict(df),
        "missing_by_column": {c: missing_by_col[c] for c in cols_with_missing},
        "n_rows": int(df.shape[0]),
    }

    try:
        raw = chat_once(
            system=_SYSTEM_PREPROCESS_PLANNER,
            user=json.dumps(payload),
            model="gpt-4o-mini",
            temperature=0.1,
        )
        data = json.loads(raw)

        allowed = {"mean", "median", "mode", "fill", "drop"}
        out: Dict[str, str] = {}

        for col in cols_with_missing:
            s = str(data.get(col, "")).strip().lower()
            if s in allowed:
                out[col] = s

        # If LLM returned partial output, fill rest with heuristics
        if out:
            missing = [c for c in cols_with_missing if c not in out]
            if missing:
                out.update(_heuristic_missing_strategy(df[missing]))
            return out

    except Exception:
        pass

    # fallback
    return _heuristic_missing_strategy(df[cols_with_missing])


def _heuristic_missing_strategy(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for col in df.columns:
        s = df[col]
        if s.isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(s):
            skew = float(abs(s.dropna().skew())) if s.dropna().size else 0.0
            out[col] = "median" if skew > 1.0 else "mean"

        elif pd.api.types.is_datetime64_any_dtype(s):
            out[col] = "drop"

        else:
            nunq = s.nunique(dropna=True)
            out[col] = "mode" if nunq <= max(20, int(0.1 * len(s))) else "fill"

    return out


# ============================================================
#  Build full pipeline plan
# ============================================================
def build_preprocess_plan(state: Dict[str, Any], pipeline_goal: str) -> PipelinePlan:
    df = state.get("clean_df") or state.get("raw_df")

    if not isinstance(df, pd.DataFrame):
        return PipelinePlan(
            goal=pipeline_goal,
            steps=[],
            preprocess_config={},
            summary_markdown="No dataset found. Upload data first.",
            needs_user_approval=False,
        )

    df = df.copy()
    miss = missing_report(df)

    # defaults
    duplicate_strategy = "drop"
    dup_count = int(len(df) - len(df.drop_duplicates()))
    all_nan_cols = miss["all_nan_columns"] or []
    missing_strat = _llm_missing_strategy(df)

    preprocess_cfg: PreprocessConfig = {
        "duplicate_strategy": duplicate_strategy,
        "drop_all_nan_cols": all_nan_cols,
        "missing_strategy": missing_strat,
        "type_overrides": {},
        "column_mapping": {},
    }

    steps: List[PlanStep] = [
        {
            "id": "preprocess_auto_defaults",
            "kind": "preprocess",
            "description": (
                f"Apply automatic preprocessing:\n"
                f"- Drop ~{dup_count} duplicate rows\n"
                f"- Drop all-NaN columns: {all_nan_cols or 'none'}\n"
                "- Handle remaining missing values with simple strategies."
            ),
            "auto": True,
            "requires_user_input": True,
        }
    ]

    if pipeline_goal in {"preprocess_train", "preprocess_train_tune"}:
        steps.append(
            {
                "id": "train_after_preprocess",
                "kind": "train",
                "description": (
                    "Train baseline models after preprocessing "
                    "(requires selecting a target column)."
                ),
                "auto": False,
                "requires_user_input": True,
            }
        )

    if pipeline_goal == "preprocess_train_tune":
        steps.append(
            {
                "id": "tune_after_training",
                "kind": "tune",
                "description": (
                    "Tune the best baseline model using a metric you choose."
                ),
                "auto": False,
                "requires_user_input": True,
            }
        )

    # summary markdown
    missing_lines = "\n".join(
        f"- **{col}** → `{method}`" for col, method in missing_strat.items()
    ) or "No columns with missing values."

    n_rows, n_cols = df.shape

    summary = (
        f"I will preprocess your dataset (**{n_rows}×{n_cols}**) using safe defaults:\n\n"
        f"**Duplicates:** drop (~{dup_count})\n"
        f"**All-NaN columns:** {all_nan_cols or 'none'}\n"
        f"**Missing-value strategies:**\n{missing_lines}\n\n"
        "After preprocessing, I will show a preview of the cleaned dataset.\n"
    )

    if pipeline_goal in {"preprocess_train", "preprocess_train_tune"}:
        summary += (
            "\nNext, with your permission, I will proceed to baseline model training.\n"
        )
    if pipeline_goal == "preprocess_train_tune":
        summary += (
            "After training, I can tune the best model based on a metric you choose.\n"
        )

    summary += (
        "\nPlease confirm if you would like me to proceed with these preprocessing steps."
    )

    return PipelinePlan(
        goal=pipeline_goal,
        steps=steps,
        preprocess_config=preprocess_cfg,
        summary_markdown=summary,
        needs_user_approval=True,
    )


# ============================================================
#  Entry point
# ============================================================
def plan_pipeline(state: Dict[str, Any], pipeline_goal: str) -> PipelinePlan:
    return build_preprocess_plan(state, pipeline_goal)
