# app/agents/planner_agent.py
"""
Planner agent: reads high-level user goals and outputs a small execution plan
for the Supervisor to follow.

Plan is stored in state:
  - plan_steps: List[str]
  - plan_idx: int
  - plan_active: bool
  - plan_reason: str
"""

from typing import Dict, Any, List
import json

from app.agents.llm_utils import chat_json
from app.agents.prompts import SYSTEM_PLANNER_AGENT


VALID_STEPS = [
    "preprocess_data",
    "choose_task_type",
    "train_baselines",
    "tune_best_model_optuna",
    "tune_best_model_random_search",
]

def make_plan(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "steps": [...],                  # subset of VALID_STEPS in correct order
        "needs_target": bool,            # true if train is present and target_col missing
        "reason": str
      }
    """
    summary = {
        "has_clean_df": state.get("clean_df") is not None,
        "has_pre_df": state.get("pre_df") is not None,
        "has_train_result": state.get("train_result") is not None,
        "has_tuned_result": state.get("tuned_result") is not None,
        "target_col": state.get("target_col"),
        "task_type": state.get("task_type"),
    }

    user_payload = json.dumps(
        {"user_request": user_text, "state_summary": summary},
        default=str
    )

    plan = chat_json(
        system=SYSTEM_PLANNER_AGENT,
        user=user_payload,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    # Defensive normalization
    steps: List[str] = []
    raw_steps = plan.get("steps") or []
    for s in raw_steps:
        s = str(s).strip()
        if s in VALID_STEPS and s not in steps:
            steps.append(s)

    # minimal fallback if LLM gives junk
    if not steps:
        steps = ["preprocess_data"]
        if "train" in user_text.lower():
            steps.append("train_baselines")
        if "tune" in user_text.lower():
            steps.append("tune_best_model_optuna")

    needs_target = ("train_baselines" in steps) and not state.get("target_col")

    return {
        "steps": steps,
        "needs_target": needs_target,
        "reason": plan.get("reason", "Planned workflow from your request."),
    }
