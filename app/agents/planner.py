# app/agents/planner.py
from __future__ import annotations
from typing import Dict, Any
import json

import pandas as pd

from app.core.preprocessing import missing_report, dtypes_dict
from app.agents.llm_utils import chat_json, chat_once
from app.agents.prompts import (
    SYSTEM_AUTOML_PREPROCESS_PLANNER,
    SYSTEM_AUTOML_PLAN_SUMMARY,
)


class PlannerAgent:
    """
    Planner layer for AutoML.

    For Step 1 we only support a single step:
      - 'preprocess' with default configs.

    Later we can extend this with train/tune steps.
    """

    def make_preprocess_plan(
        self,
        df: pd.DataFrame,
        user_text: str,
    ) -> Dict[str, Any]:
        """
        Inspect the dataset + user request and return:

        {
          "steps": [
            {
              "name": "preprocess",
              "description": "...",
              "configs": {
                 "drop_cols": [...],
                 "duplicate_strategy": "drop",
                 "missing_strategy": {...},
                 "type_overrides": {...},
                 "column_mapping": {...},
                 "preserve_column_names": bool,
              }
            }
          ],
          "needs_user_input": ["target_col", "tune_metric"],
          "natural_language_summary": "<bullets for the chat UI>"
        }
        """
        if df is None or df.empty:
            # degenerate plan: nothing to do
            return {
                "steps": [],
                "needs_user_input": ["target_col", "tune_metric"],
                "natural_language_summary": (
                    "I don't see any rows in the dataset, so there is no preprocessing to run."
                ),
            }

        miss = missing_report(df)
        dup_count = int(len(df) - len(df.drop_duplicates()))
        dtypes = dtypes_dict(df)

        payload = {
            "user_request": user_text,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(df.columns),
            "missing_by_column": miss["missing_by_column"],
            "all_nan_columns": miss["all_nan_columns"],
            "duplicate_row_count": dup_count,
            "dtypes": dtypes,
        }

        # 1) Ask LLM for raw configs (JSON only)
        raw = chat_json(
            system=SYSTEM_AUTOML_PREPROCESS_PLANNER,
            user=json.dumps(payload),
            model="gpt-4o-mini",
            temperature=0.2,
        ) or {}

        cfg = (raw.get("configs") or {}).copy()

        # 2) Enforce our hard rules / defaults
        drop_cols = cfg.get("drop_cols") or miss["all_nan_columns"]
        missing_strategy = cfg.get("missing_strategy") or {}
        type_overrides = cfg.get("type_overrides") or {}
        column_mapping = cfg.get("column_mapping") or {}
        preserve_column_names = bool(column_mapping) or bool(
            cfg.get("preserve_column_names", False)
        )

        # IMPORTANT: system rule — always drop duplicates in planner mode
        duplicate_strategy = "drop"

        resolved_plan = {
            "steps": [
                {
                    "name": "preprocess",
                    "description": "Automatic preprocessing before training.",
                    "configs": {
                        "drop_cols": drop_cols,
                        "duplicate_strategy": duplicate_strategy,
                        "missing_strategy": missing_strategy,
                        "type_overrides": type_overrides,
                        "column_mapping": column_mapping,
                        "preserve_column_names": preserve_column_names,
                    },
                }
            ],
            # Training & tuning still need explicit user input later
            "needs_user_input": ["target_col", "tune_metric"],
        }

        # 3) Human-friendly summary for the chat UI
        summary_payload = {
            "plan": resolved_plan,
            "data_profile": {
                "shape": payload["shape"],
                "duplicate_row_count": dup_count,
                "all_nan_columns": miss["all_nan_columns"],
                "missing_by_column": miss["missing_by_column"],
            },
        }

        try:
            summary = chat_once(
                system=SYSTEM_AUTOML_PLAN_SUMMARY,
                user=json.dumps(summary_payload, default=str),
                model="gpt-4o-mini",
                temperature=0.3,
            )
        except Exception:
            summary = (
                "I will drop duplicate rows, drop any columns that are entirely empty, "
                "and fill in missing values with simple strategies like mean/median for "
                "numbers or the most frequent value for categories."
            )

        resolved_plan["natural_language_summary"] = summary
        return resolved_plan
