from __future__ import annotations
from typing import Dict, Any


class Planner:
    """
    Planner for multi-step / end-to-end queries.
    """

    def __init__(self) -> None:
        # Do NOT import ChatOrchestrator here. Avoid circular import.
        pass

    # ----------------- ENTRY FOR FIRST MULTI-STEP MESSAGE -----------------
    def handle_multi_step(
        self,
        user_text: str,
        state: Dict[str, Any],
        intent: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        st = state.copy()
        txt = (user_text or "").lower()

        wants_tune = any(w in txt for w in ["tune", "tuning", "optimize", "hyperparameter"])
        wants_preprocess = any(w in txt for w in ["preprocess", "clean", "cleanup", "clean up"])
        wants_train = any(w in txt for w in ["train", "training", "model", "build a model"])

        steps = []

        # Always propose preprocessing safely
        steps.extend([
            "Drop columns that are completely empty (all NaN).",
            "Drop duplicate rows.",
            "Handle missing values column-by-column with safe defaults.",
        ])

        steps.append("Show you a preprocessed preview of the cleaned data.")
        steps.append("Ask you to pick the **target column**.")
        steps.append("Train baseline models and show you the leaderboard.")

        if wants_tune:
            steps.append("Optionally run hyperparameter tuning if needed.")

        st["planner_plan"] = {
            "origin_text": user_text,
            "steps": steps,
            "wants_tune": wants_tune,
        }
        st["planner_stage"] = "await_confirm"

        bullets = "\n".join([f"- {s}" for s in steps])
        msg = (
            "Here’s what I’ll do for you:\n\n"
            f"{bullets}\n\n"
            "Do you want me to run this plan now? (**yes / no**)"
        )

        st.setdefault("messages", [])
        st["messages"].append({"role": "assistant", "content": msg})

        return st

    # ----------------- YES / NO HANDLER -----------------
    def handle_confirmation(
        self,
        user_text: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:

        st = state.copy()
        txt = (user_text or "").strip().lower()

        yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "please do", "do it"}
        no_words  = {"no", "n", "nope", "not now", "later", "stop"}

        # NO → Cancel plan
        if txt in no_words:
            st["planner_stage"] = None
            st["planner_plan"] = None
            st["planner_active"] = False

            st.setdefault("messages", [])
            st["messages"].append({
                "role": "assistant",
                "content": "Okay — I’ve cancelled the automatic multi-step plan. You can still preprocess, train, or tune anytime."
            })
            return st

        # YES → Run automatic preprocessing
        if txt in yes_words:

            # *** LAZY IMPORT — avoids circular import ***
            from app.agents.chat_orchestrator import ChatOrchestrator
            orch = ChatOrchestrator()

            st = orch._auto_plan_preprocessing(st)

            st["planner_stage"] = "executing"
            st["planner_active"] = True

            st["messages"].append({
                "role": "assistant",
                "content": (
                    "Great — I’ll run the preprocessing now.\n"
                    "You’ll see a **preprocessed preview** below, and after that "
                    "you can select the **target column** for training."
                )
            })

            return st

        # Anything else → Ask again
        st.setdefault("messages", [])
        st["messages"].append({
            "role": "assistant",
            "content": "Please reply with **yes** to run the plan, or **no** to cancel it."
        })
        return st
