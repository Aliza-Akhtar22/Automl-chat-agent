from __future__ import annotations
from typing import Dict, Any, List


class Planner:
    """
    Production planner.

    Responsibilities:
    - Convert normalized intent into an ordered execution plan
    - Store plan steps + cursor
    - Request user approval (HITL)
    - NEVER execute tools or touch LangGraph directly
    """

    # ---------------------------------------------------------
    # PLAN CONSTRUCTION
    # ---------------------------------------------------------
    def build_plan(self, intent_actions: List[str], state: Dict[str, Any]) -> List[str]:
        """
        Convert intent actions into an ordered, executable plan.

        Rules:
        - Training always requires target confirmation
        - Tuning always requires training first
        - Preprocessing happens before anything else
        """
        plan: List[str] = []

        # 1) Preprocessing
        if "preprocess" in intent_actions:
            plan.append("preprocess")

        # 2) Training (requires target confirmation)
        if "train" in intent_actions:
            plan.append("confirm_target")
            plan.append("train")

        # 3) Tuning (requires training)
        if "tune" in intent_actions:
            if "train" not in plan:
                plan.append("confirm_target")
                plan.append("train")
            plan.append("tune")

        # 4) Preview (non-mutating but still planned)
        if "preview" in intent_actions and "preview" not in plan:
            plan.append("preview")

        # Safety fallback
        if not plan:
            plan.append("preview")

        return plan

    # ---------------------------------------------------------
    # MULTI-STEP ENTRY (PLAN PROPOSAL)
    # ---------------------------------------------------------
    def handle_multi_step(
        self,
        user_text: str,
        state: Dict[str, Any],
        intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Propose a plan and request user approval.
        """
        st = state.copy()

        actions = intent.get("actions", [])
        plan_steps = self.build_plan(actions, st)

        # Store executable plan
        st["plan_steps"] = plan_steps
        st["plan_cursor"] = 0

        # Activate approval gate
        st["require_approval"] = True
        st["approved"] = False
        st["stage"] = "plan_proposed"

        plan_text = "\n".join(
            f"{i + 1}. {step.replace('_', ' ').title()}"
            for i, step in enumerate(plan_steps)
        )

        st.setdefault("messages", [])
        st["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "Here’s the plan I suggest:\n\n"
                    f"{plan_text}\n\n"
                    "Do you want me to execute this plan? (yes / no)"
                ),
            }
        )

        return st

    # ---------------------------------------------------------
    # APPROVAL HANDLER (HITL)
    # ---------------------------------------------------------
    def handle_confirmation(self, user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle yes/no approval for a proposed plan.

        IMPORTANT:
        - Consumes approval exactly once
        - Does NOT execute anything
        - Moves system into executing_plan state
        """
        st = state.copy()
        t = (user_text or "").strip().lower()

        yes_words = {"yes", "y", "ok", "okay", "sure", "do it"}
        no_words = {"no", "n", "cancel", "stop"}

        # ---------------- CANCEL PLAN ----------------
        if t in no_words:
            st["plan_steps"] = []
            st["plan_cursor"] = 0
            st["require_approval"] = False
            st["approved"] = False
            st["stage"] = "preview_download"

            st.setdefault("messages", [])
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": "Okay — I’ve cancelled the plan.",
                }
            )
            return st

        # ---------------- APPROVE PLAN ----------------
        if t in yes_words:
            st["approved"] = True
            st["require_approval"] = False

            # IMPORTANT: move into controlled execution mode
            st["stage"] = "executing_plan"

            st.setdefault("messages", [])
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Great 👍 I’ll start executing the plan step by step.\n\n"
                        "I’ll pause whenever I need your input."
                    ),
                }
            )
            return st

        # ---------------- INVALID RESPONSE ----------------
        st.setdefault("messages", [])
        st["messages"].append(
            {
                "role": "assistant",
                "content": "Please reply with **yes** or **no**.",
            }
        )
        return st
