# app/agents/planner.py
from __future__ import annotations
from typing import Dict, Any, List


class Planner:
    def build_plan(self, intent_actions: List[str], state: Dict[str, Any]) -> List[str]:
        plan: List[str] = []

        if "preprocess" in intent_actions:
            plan.append("preprocess")

        # Prophet forecasting
        if "forecast" in intent_actions:
            plan.append("confirm_forecast_horizon")
            plan.append("confirm_ds")
            plan.append("confirm_y")
            plan.append("forecast")

        # Training
        if "train" in intent_actions:
            plan.append("confirm_target")
            plan.append("train")

        # Tuning
        if "tune" in intent_actions:
            if "train" not in plan:
                plan.append("confirm_target")
                plan.append("train")
            plan.append("tune")

        if "preview" in intent_actions and "preview" not in plan:
            plan.append("preview")

        if not plan:
            plan.append("preview")

        return plan

    def handle_multi_step(self, user_text: str, state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        st = state.copy()
        actions = intent.get("actions", [])
        plan_steps = self.build_plan(actions, st)

        st["plan_steps"] = plan_steps
        st["plan_cursor"] = 0

        st["require_approval"] = True
        st["approved"] = False
        st["stage"] = "plan_proposed"

        plan_text = "\n".join(f"{i + 1}. {step.replace('_', ' ').title()}" for i, step in enumerate(plan_steps))

        st.setdefault("messages", [])
        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "Here is the plan:\n\n"
                    f"{plan_text}\n\n"
                    "Reply with yes or no."
                ),
            }
        )
        return st

    def handle_confirmation(self, user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        st = state.copy()
        t = (user_text or "").strip().lower()

        yes_words = {"yes", "y", "ok", "okay", "sure", "do it"}
        no_words = {"no", "n", "cancel", "stop"}

        if t in no_words:
            st["plan_steps"] = []
            st["plan_cursor"] = 0
            st["require_approval"] = False
            st["approved"] = False
            st["stage"] = "preview_download"
            st.setdefault("messages", [])
            st["messages"].append({"role": "assistant", "content": "Plan cancelled."})
            return st

        if t in yes_words:
            st["approved"] = True
            st["require_approval"] = False
            st["stage"] = "executing_plan"
            # IMPORTANT: do NOT add any “I will start…” message.
            return st

        st.setdefault("messages", [])
        st["messages"].append({"role": "assistant", "content": "Please reply with yes or no."})
        return st
