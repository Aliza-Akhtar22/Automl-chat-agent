# app/agents/planner_agent.py
from __future__ import annotations
from typing import Dict, Any
import json

from app.agents.llm_utils import chat_json
from app.agents.prompts import SYSTEM_PLANNER_AGENT


def make_plan(user_text: str) -> Dict[str, Any]:
    """
    Turn a high-level user goal into an ordered list of tool steps.
    Output shape:
      {"steps": [...], "reason": "..."}
    """
    system = SYSTEM_PLANNER_AGENT
    user = user_text.strip()

    try:
        plan = chat_json(
            system=system,
            user=user,
            model="gpt-4o-mini",
            temperature=0.0,
        )
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return dict")

        steps = plan.get("steps") or []
        # hard sanitize
        allowed = {
            "preprocess_data",
            "choose_task_type",
            "train_baselines",
            "tune_best_model_optuna",
            "tune_best_model_random_search",
        }
        steps = [s for s in steps if s in allowed]

        return {
            "steps": steps,
            "reason": str(plan.get("reason", "")).strip(),
        }

    except Exception:
        # Fallback heuristic
        txt = user.lower()
        steps = []
        if any(w in txt for w in ["preprocess", "clean", "missing", "duplicates"]):
            steps.append("preprocess_data")
        if any(w in txt for w in ["train", "training", "model"]):
            steps.append("train_baselines")
        if any(w in txt for w in ["tune", "tuning", "hyperparameter", "optimize"]):
            steps.append("tune_best_model_optuna")

        return {
            "steps": steps,
            "reason": "Fallback plan based on detected keywords.",
        }
