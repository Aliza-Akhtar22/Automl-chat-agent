# app/agents/planner_agent.py
from __future__ import annotations
from typing import Dict, Any

from app.agents.llm_utils import chat_json
from app.agents.prompts import SYSTEM_PLANNER


def make_plan(user_text: str) -> Dict[str, Any]:
    """
    Planner: read the user's high-level goal and return a structured plan.

    Expected JSON format from the LLM:
    {
      "steps": [
        {"action": "preprocess", "args": {...}},
        {"action": "train", "args": {"target": "col_name"}},
        {"action": "tune", "args": {"metric": "f1", "method": "bayesian"}}
      ]
    }
    """
    try:
        plan = chat_json(
            system=SYSTEM_PLANNER,
            user=user_text,
            model="gpt-4o-mini",
            temperature=0.0,
        )
        # basic safety
        if not isinstance(plan, dict) or "steps" not in plan:
            raise ValueError("Planner returned invalid format")
        if not isinstance(plan.get("steps"), list):
            plan["steps"] = []
        return plan
    except Exception:
        # safe fallback
        return {
            "steps": [
                {"action": "preprocess", "args": {}},
                {"action": "train", "args": {}},
                {"action": "tune", "args": {"metric": "auto"}},
            ]
        }
