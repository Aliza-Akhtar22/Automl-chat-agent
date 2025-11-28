# app/agents/planner.py
from __future__ import annotations
from typing import Dict, Any

from app.agents.chat_orchestrator import ChatOrchestrator


class Planner:
    """
    Planner for multi-step / end-to-end queries.

    Responsibilities:
      - Detect *what* the user wants in high level (preprocess + train [+ tune]).
      - Produce a human-readable plan.
      - Ask for YES/NO approval.
      - On YES: prepare state so that:
          * automatic preprocessing is configured and triggered, and
          * after preview, the user is nudged to select target & train.
    """

    def __init__(self) -> None:
        # separate orchestrator instance just to reuse _auto_plan_preprocessing
        self._orch = ChatOrchestrator()

    # ----------------- ENTRY FOR FIRST MULTI-STEP MESSAGE -----------------
    def handle_multi_step(
        self,
        user_text: str,
        state: Dict[str, Any],
        intent: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Called when the router classifies the query as `multi_step`.

        Example user: "I want to preprocess and then train the model for me and give me results"
        Behaviour:
          1) Build a simple plan (preprocess → preview → train [+ tune?]).
          2) Store it in state.
          3) Ask user for YES/NO approval.
        """
        st = state.copy()
        txt = (user_text or "").lower()

        # Very lightweight understanding of the multi-step request
        wants_tune = any(w in txt for w in ["tune", "tuning", "optimize", "hyperparameter"])
        wants_preprocess = any(w in txt for w in ["preprocess", "clean", "cleanup", "clean up"])
        wants_train = any(w in txt for w in ["train", "training", "model", "build a model"])

        # Default: preprocess + train. Tuning optional.
        steps = []

        if wants_preprocess or True:
            steps.extend([
                "Drop columns that are completely empty (all NaN).",
                "Drop duplicate rows.",
                "Handle missing values column-by-column with safe defaults.",
            ])

        steps.append(
            "Show you a preprocessed preview of the cleaned data so you can quickly check it."
        )

        # IMPORTANT: explicitly call out *user* choice of target
        steps.append(
            "Ask you to pick the **target column** (what you want to predict)."
        )
        steps.append(
            "Train a set of baseline models and show you the leaderboard with the recommended best model."
        )

        if wants_tune:
            steps.append(
                "Optionally run hyperparameter tuning on the best model if the baseline scores look only okay."
            )

        # Store plan in state
        st["planner_plan"] = {
            "origin_text": user_text,
            "steps": steps,
            "wants_tune": wants_tune,
        }
        st["planner_stage"] = "await_confirm"

        # Human-readable markdown plan
        bullets = "\n".join([f"- {s}" for s in steps])
        plan_md = (
            "Here’s what I’ll do for you, step by step:\n\n"
            f"{bullets}\n\n"
            "After the preprocessing step, I’ll show you a **preprocessed preview** and then "
            "ask you to **select the target column and train**.\n\n"
            "Do you want me to run this plan now? (**yes / no**)"
        )

        st.setdefault("messages", [])
        st["messages"].append({"role": "assistant", "content": plan_md})

        return st

    # ----------------- YES / NO HANDLER -----------------
    def handle_confirmation(
        self,
        user_text: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Called when we are in `planner_stage == 'await_confirm'` and the
        user replies with something like yes/no.

        YES:
          - Run automatic preprocessing planner.
          - Set stage to preview_download and show_only_preview=True.
          - Mark planner_active so preview can instruct: "Now kindly select target and train".
        NO:
          - Cancel the plan, go back to normal chat.
        """
        st = state.copy()
        txt = (user_text or "").strip().lower()

        yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "please do", "do it"}
        no_words = {"no", "n", "nope", "not now", "later", "stop"}

        # 1) User says NO → cancel plan
        if txt in no_words:
            st["planner_stage"] = None
            st["planner_plan"] = None
            st["planner_active"] = False

            st.setdefault("messages", [])
            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "No worries 👍 I won’t run the full pipeline automatically.\n"
                        "You can still **preview**, **train**, or **tune** step by step whenever you like."
                    ),
                }
            )
            return st

        # 2) User says YES → activate plan
        if txt in yes_words:
            st.setdefault("messages", [])

            # Use the auto-preprocessing planner you already implemented
            # This fills in pp_* configs and sets stage → preview_download, show_only_preview=True,
            # and want_preprocess=True so ui_preview_and_download + graph will actually run.
            st = self._orch._auto_plan_preprocessing(st)

            # Mark that this was a planner-driven flow
            st["planner_stage"] = "executing"
            st["planner_active"] = True

            st["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Great 👍 I’ll run this plan now.\n\n"
                        "- I’ll automatically **clean the data** and then show you a **preprocessed preview**.\n"
                        "- **Below the preview**, you’ll see a message guiding you to **select the target column** "
                        "and **train the models**.\n"
                    ),
                }
            )
            return st

        # 3) Anything else → ask again
        st.setdefault("messages", [])
        st["messages"].append(
            {
                "role": "assistant",
                "content": "Please reply with **yes** to run the plan, or **no** to cancel it.",
            }
        )
        return st
